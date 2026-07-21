"""CLI-owned orchestration for supervised transcript extraction."""

from __future__ import annotations

import io
import os
import stat
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping
from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from keyframe import transcript
from keyframe.output_session import (
    RUN_DIRECTORY_PREFIX,
    OutputSessionError,
    remove_keyframe_owned_directory,
)
from keyframe.stage_scheduler import (
    CONCURRENCY_POLICIES,
    ActiveStage,
    ScheduleDecision,
    StageScheduler,
    complete_transcription_with_auto_fallback,
    diarization_demand,
    transcription_demand,
)
from keyframe.stage_supervisor import StageProgress, StageSupervisor


@dataclass(frozen=True)
class TranscriptRunConfig:
    model_name: str = "medium"
    fmt: str = "txt"
    transcription_backend: str = "auto"
    diarization_device: str = "auto"
    stage_concurrency: str = "auto"
    speaker_detection: bool = True

    def __post_init__(self) -> None:
        if self.model_name not in transcript.MLX_MODEL_SPECS:
            raise ValueError(f"unknown Whisper model: {self.model_name!r}")
        if self.fmt not in transcript.WRITERS:
            raise ValueError(f"unknown transcript format: {self.fmt!r}")
        if self.transcription_backend not in transcript.TRANSCRIPTION_BACKENDS:
            raise ValueError(
                f"unknown transcription backend: {self.transcription_backend!r}"
            )
        if self.diarization_device not in transcript.DIARIZATION_DEVICES:
            raise ValueError(f"unknown diarization device: {self.diarization_device!r}")
        if self.stage_concurrency not in CONCURRENCY_POLICIES:
            raise ValueError(f"unknown stage concurrency: {self.stage_concurrency!r}")


@dataclass(frozen=True)
class TranscriptPreflight:
    config: TranscriptRunConfig
    runtime_platform: transcript.RuntimePlatform
    effective_backend: str
    transcription_device: str
    hf_token: str | None
    effective_diarization_device: str | None
    missing_hf_token: bool

    @property
    def diarization_enabled(self) -> bool:
        return self.effective_diarization_device is not None


@dataclass(frozen=True)
class TranscriptRunResult:
    segments: tuple[transcript.TranscriptSegment, ...]
    language: str
    effective_backend: str
    fallback_used: bool
    initial_schedule: ScheduleDecision
    timings: Mapping[str, float]
    metadata: Mapping[str, Any]


class TranscriptOutputError(OutputSessionError):
    """A complete final transcript generation could not be published."""


@dataclass(frozen=True)
class _FinalOutputArtifact:
    public_path: Path
    staged_path: Path
    backup_path: Path
    had_public: bool


def preflight_transcript_run(
    config: TranscriptRunConfig,
    *,
    environment: Mapping[str, str] | None = None,
    runtime_platform: transcript.RuntimePlatform | None = None,
    cuda_probe: Callable[[], bool] | None = None,
) -> TranscriptPreflight:
    """Resolve backend and devices before creating outputs or loading models."""

    environment = os.environ if environment is None else environment
    runtime_platform = runtime_platform or transcript.current_runtime_platform()
    effective_backend = transcript.resolve_transcription_backend(
        config.transcription_backend,
        runtime_platform,
    )
    cuda_probe = cuda_probe or transcript.cuda_is_available
    cuda_available: bool | None = None

    def has_cuda() -> bool:
        nonlocal cuda_available
        if cuda_available is None:
            # PyTorch does not provide CUDA on macOS. Avoid importing the model
            # stack in the CLI parent for the common MLX + CPU diarization path.
            cuda_available = (
                False if runtime_platform.system == "Darwin" else bool(cuda_probe())
            )
        return cuda_available

    transcription_device = (
        "mlx"
        if effective_backend == "mlx"
        else ("cuda" if has_cuda() else "cpu")
    )
    raw_token = environment.get("HF_TOKEN") or ""
    hf_token = raw_token.strip() or None
    missing_hf_token = bool(config.speaker_detection and hf_token is None)
    effective_diarization_device: str | None = None
    if config.speaker_detection and hf_token is not None:
        requested_device = config.diarization_device
        if requested_device == "cpu":
            effective_diarization_device = "cpu"
        else:
            effective_diarization_device = transcript.resolve_diarization_device(
                requested_device,
                cuda_available=has_cuda(),
            )

    return TranscriptPreflight(
        config=config,
        runtime_platform=runtime_platform,
        effective_backend=effective_backend,
        transcription_device=transcription_device,
        hf_token=hf_token,
        effective_diarization_device=effective_diarization_device,
        missing_hf_token=missing_hf_token,
    )


def print_stage_progress(event: StageProgress) -> None:
    detail = f": {event.message}" if event.message else ""
    print(f"[{event.stage}] {event.event}{detail}", flush=True)


def _final_output_paths(output_dir: Path, fmt: str) -> tuple[Path, ...]:
    primary = output_dir / f"transcript.{fmt}"
    if fmt == "json":
        return (primary,)
    return (primary, output_dir / "transcript.json")


def _print_preflight(preflight: TranscriptPreflight) -> None:
    config = preflight.config
    print(f"Video backend: requested={config.transcription_backend}, "
          f"effective={preflight.effective_backend}")
    print(f"Transcription device: {preflight.transcription_device}")
    print(f"Whisper model: {config.model_name}")
    if preflight.effective_backend == "mlx":
        model_spec = transcript.MLX_MODEL_SPECS[config.model_name]
        print(f"MLX model: {model_spec.repository}@{model_spec.revision}")
    if preflight.diarization_enabled:
        print(
            "Diarization device: "
            f"requested={config.diarization_device}, "
            f"effective={preflight.effective_diarization_device}"
        )
    else:
        print("Diarization device: disabled")


def _print_schedule(decision: ScheduleDecision) -> None:
    print(
        f"Stage schedule: policy={decision.policy}, mode={decision.mode}, "
        f"reason={decision.reason}"
    )
    budgets = ", ".join(
        f"{budget.stage}={budget.cpu_threads}" for budget in decision.budgets
    )
    memory = decision.resources.available_memory_bytes
    available = "unknown" if memory is None else str(memory)
    print(f"Worker thread budgets: {budgets}")
    print(
        "Memory admission: "
        f"required={decision.required_memory_bytes}, available={available}"
    )


def _write_final_outputs(
    segments: tuple[transcript.TranscriptSegment, ...],
    output_paths: tuple[Path, ...],
    fmt: str,
    *,
    staging_root: str | Path | None = None,
) -> None:
    if not output_paths:
        raise ValueError("final transcript publication requires at least one output")
    writers = [transcript.WRITERS[fmt]]
    if fmt != "json":
        writers.append(transcript.write_json)
    if len(writers) != len(output_paths):
        raise ValueError("final transcript writer and output counts do not match")

    output_parent = output_paths[0].parent
    staging_parent = Path(staging_root) if staging_root is not None else output_parent
    prefix = (
        "final-transcript-"
        if staging_root is not None
        else f"{RUN_DIRECTORY_PREFIX}final-transcript-"
    )
    try:
        publication_root = Path(
            tempfile.mkdtemp(prefix=prefix, dir=staging_parent)
        )
    except OSError as exc:
        raise TranscriptOutputError(
            f"failed to stage final transcript outputs: {exc}"
        ) from exc

    artifacts: list[_FinalOutputArtifact] = []
    try:
        for index, (writer, public_path) in enumerate(
            zip(writers, output_paths, strict=True)
        ):
            staged_path = publication_root / f"staged-{index}-{public_path.name}"
            backup_path = publication_root / f"backup-{index}-{public_path.name}"
            with redirect_stdout(io.StringIO()):
                writer(segments, staged_path)
            if staged_path.is_symlink() or not staged_path.is_file():
                raise TranscriptOutputError(
                    f"final transcript writer did not stage a regular file: {staged_path}"
                )
            had_public = os.path.lexists(public_path)
            if had_public and public_path.is_dir() and not public_path.is_symlink():
                raise TranscriptOutputError(
                    f"final transcript output path is a directory: {public_path}"
                )
            if had_public:
                try:
                    mode = stat.S_IMODE(public_path.stat().st_mode)
                except FileNotFoundError:
                    pass
                else:
                    staged_path.chmod(mode)
            artifacts.append(
                _FinalOutputArtifact(
                    public_path=public_path,
                    staged_path=staged_path,
                    backup_path=backup_path,
                    had_public=had_public,
                )
            )

        try:
            for artifact in artifacts:
                if artifact.had_public:
                    _replace_final_output(
                        artifact.public_path,
                        artifact.backup_path,
                    )
            for artifact in artifacts:
                _replace_final_output(
                    artifact.staged_path,
                    artifact.public_path,
                )
        except BaseException as exc:
            rollback_errors = _rollback_final_outputs(artifacts)
            for rollback_error in rollback_errors:
                exc.add_note(
                    "failed to roll back final transcript output: "
                    f"{type(rollback_error).__name__}: {rollback_error}"
                )
            if isinstance(exc, OSError):
                raise TranscriptOutputError(
                    f"failed to publish final transcript outputs: {exc}"
                ) from exc
            raise

        for output_path in output_paths:
            print(f"Saved: {output_path}")
    except OSError as exc:
        raise TranscriptOutputError(
            f"failed to stage final transcript outputs: {exc}"
        ) from exc
    finally:
        remove_keyframe_owned_directory(publication_root)


def _replace_final_output(source: Path, target: Path) -> None:
    os.replace(source, target)


def _rollback_final_outputs(
    artifacts: Iterable[_FinalOutputArtifact],
) -> tuple[BaseException, ...]:
    errors = []
    for artifact in reversed(tuple(artifacts)):
        try:
            if os.path.lexists(artifact.backup_path):
                _replace_final_output(artifact.backup_path, artifact.public_path)
            elif (
                not artifact.had_public
                and not os.path.lexists(artifact.staged_path)
                and os.path.lexists(artifact.public_path)
            ):
                artifact.public_path.unlink()
        except BaseException as exc:
            errors.append(exc)
    return tuple(errors)


def _final_output_staging_root(supervisor: Any) -> Path | None:
    staging = getattr(supervisor, "staging", None)
    root = getattr(staging, "root", None)
    return Path(root) if root is not None else None


def _print_transcript_result(
    segments: tuple[transcript.TranscriptSegment, ...],
    language: str,
    timings: Mapping[str, float],
) -> None:
    print(f"Detected language: {language}")
    print(f"Segments: {len(segments)}")
    if segments:
        print(f"Duration covered: {transcript.format_time(segments[-1].end)}")
    rendered_timings = ", ".join(
        f"{stage}={seconds:.2f}s" for stage, seconds in timings.items()
    )
    print(f"Stage timings: {rendered_timings}")
    print("\n--- Preview (first 10 segments) ---")
    for segment in segments[:10]:
        speaker_prefix = f"{segment.speaker} " if segment.speaker else ""
        print(
            f"  [{transcript.format_time(segment.start)}] "
            f"{speaker_prefix}{segment.text}"
        )
    if len(segments) > 10:
        print(f"  ... ({len(segments) - 10} more segments)")


def run_supervised_transcript(
    video_path: str | Path,
    output_dir: str | Path,
    preflight: TranscriptPreflight,
    *,
    scheduler: StageScheduler | None = None,
    supervisor: Any | None = None,
    supervisor_factory: Callable[..., Any] | None = None,
    fallback_runner: Callable[..., Any] | None = None,
    clock: Callable[[], float] | None = None,
) -> TranscriptRunResult:
    """Run current-run-only transcript stages and atomically write final formats."""

    video = Path(video_path)
    output_dir = Path(output_dir)
    config = preflight.config
    scheduler = scheduler or StageScheduler(config.stage_concurrency)
    if supervisor is not None and supervisor_factory is not None:
        raise ValueError("provide either supervisor or supervisor_factory, not both")
    supervisor_factory = supervisor_factory or StageSupervisor
    fallback_runner = fallback_runner or complete_transcription_with_auto_fallback
    clock = clock or time.monotonic
    output_paths = _final_output_paths(output_dir, config.fmt)
    final_paths: Iterable[Path] = output_paths

    transcription_stage = transcription_demand(
        config.model_name,
        backend=preflight.effective_backend,
        device=(
            None
            if preflight.effective_backend == "mlx"
            else preflight.transcription_device
        ),
    )
    stages = [transcription_stage]
    diarization_stage = None
    if preflight.effective_diarization_device is not None:
        diarization_stage = diarization_demand(
            preflight.effective_diarization_device
        )
        stages.append(diarization_stage)
    decision = scheduler.decide(stages)
    _print_preflight(preflight)
    _print_schedule(decision)

    timings: dict[str, float] = {}
    diarization_handle = None
    diarization_started: float | None = None
    supervisor_context = (
        nullcontext(supervisor)
        if supervisor is not None
        else supervisor_factory(
            output_dir,
            progress_callback=print_stage_progress,
        )
    )
    with supervisor_context as active_supervisor:
        if active_supervisor is None or active_supervisor.public is None:
            raise RuntimeError("stage supervisor did not initialize public paths")
        if Path(active_supervisor.output_dir).resolve() != output_dir.resolve():
            raise ValueError(
                "borrowed stage supervisor owns a different output directory"
            )
        # A diarization checkpoint is meaningful only for the current recording
        # and run. Raw transcripts are never read by this orchestrator, so a
        # previous validated raw checkpoint can safely survive a failed rerun.
        active_supervisor.public.diarization.unlink(missing_ok=True)

        transcription_started = clock()
        transcription_handle = active_supervisor.start_transcription(
            video,
            model_name=config.model_name,
            requested_backend=config.transcription_backend,
            final_output_paths=final_paths,
            thread_budget=decision.cpu_threads_for("transcription"),
        )
        if decision.parallel and diarization_stage is not None:
            diarization_started = clock()
            diarization_handle = active_supervisor.start_diarization(
                video,
                hf_token=preflight.hf_token or "",
                final_output_paths=final_paths,
                thread_budget=decision.cpu_threads_for("diarization"),
                device=preflight.effective_diarization_device,
            )

        active_stages = ()
        if diarization_handle is not None and diarization_stage is not None:
            active_stages = (ActiveStage(diarization_stage, diarization_handle),)
        execution = fallback_runner(
            active_supervisor,
            transcription_handle,
            scheduler=scheduler,
            video_path=video,
            model_name=config.model_name,
            requested_backend=config.transcription_backend,
            effective_backend=preflight.effective_backend,
            active_stages=active_stages,
            final_output_paths=final_paths,
        )
        timings["transcription"] = clock() - transcription_started
        if execution.fallback_schedule is not None:
            print("MLX fallback selected a fresh CPU Whisper worker")
            _print_schedule(execution.fallback_schedule)

        segments = tuple(execution.completion.records)
        language = str(execution.completion.metadata.get("language") or "unknown")
        effective_backend = str(
            execution.completion.metadata.get("effective_backend")
            or preflight.effective_backend
        )

        if not segments:
            if diarization_handle is not None:
                active_supervisor.cancel(diarization_handle)
                if diarization_started is not None:
                    timings["diarization"] = clock() - diarization_started
            active_supervisor.public.diarization.unlink(missing_ok=True)
            _write_final_outputs(
                segments,
                output_paths,
                config.fmt,
                staging_root=_final_output_staging_root(active_supervisor),
            )
            _print_transcript_result(segments, language, timings)
            return TranscriptRunResult(
                segments=segments,
                language=language,
                effective_backend=effective_backend,
                fallback_used=execution.fallback_used,
                initial_schedule=decision,
                timings=dict(timings),
                metadata=dict(execution.completion.metadata),
            )

        if preflight.missing_hf_token:
            transcript._print_missing_hf_token_warning()

        if diarization_stage is not None:
            if diarization_handle is None:
                diarization_started = clock()
                diarization_handle = active_supervisor.start_diarization(
                    video,
                    hf_token=preflight.hf_token or "",
                    final_output_paths=final_paths,
                    thread_budget=decision.cpu_threads_for("diarization"),
                    device=preflight.effective_diarization_device,
                )
            try:
                diarization_completion = active_supervisor.complete(diarization_handle)
                if diarization_started is not None:
                    timings["diarization"] = clock() - diarization_started
                labeled = transcript._assign_speakers(
                    segments,
                    diarization_completion.records,
                )
                if not any(segment.speaker for segment in labeled):
                    raise RuntimeError(
                        "speaker diarization produced no usable speaker overlaps"
                    )
                segments = labeled
            except Exception as exc:
                if diarization_started is not None:
                    timings["diarization"] = clock() - diarization_started
                active_supervisor.public.diarization.unlink(missing_ok=True)
                transcript._print_speaker_detection_failure(exc)

        _write_final_outputs(
            segments,
            output_paths,
            config.fmt,
            staging_root=_final_output_staging_root(active_supervisor),
        )
        _print_transcript_result(segments, language, timings)
        return TranscriptRunResult(
            segments=segments,
            language=language,
            effective_backend=effective_backend,
            fallback_used=execution.fallback_used,
            initial_schedule=decision,
            timings=dict(timings),
            metadata=dict(execution.completion.metadata),
        )
