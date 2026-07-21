"""Resource-aware orchestration for the CLI full-extraction dependency graph."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from keyframe import transcript
from keyframe.output_session import OutputSessionError
from keyframe.stage_scheduler import (
    ActiveStage,
    ScheduleDecision,
    StageScheduler,
    complete_transcription_with_auto_fallback,
    diarization_demand,
    frame_demand,
    transcription_demand,
)
from keyframe.transcript_cli import (
    TranscriptPreflight,
    TranscriptRunResult,
    _final_output_paths,
    _print_preflight,
    _print_schedule,
    _print_transcript_result,
    _write_final_outputs,
)


class FullPipelineFrameError(OutputSessionError):
    """The transcript completed but the staged frame generation did not."""


@dataclass(frozen=True)
class FullPipelineRunResult:
    transcript: TranscriptRunResult
    frames: Any
    frame_device: str
    initial_schedule: ScheduleDecision
    frame_schedule: ScheduleDecision
    critical_path: str
    timings: Mapping[str, float]


def resolve_frame_device(preflight: TranscriptPreflight) -> str:
    """Resolve the frame accelerator without importing Torch in the parent."""

    if preflight.runtime_platform.supports_mlx_whisper:
        return "mps"
    devices = (
        preflight.transcription_device,
        preflight.effective_diarization_device or "",
    )
    if any(device == "cuda" or device.startswith("cuda:") for device in devices):
        return "cuda"
    return "cpu"


def _handle_is_running(handle: Any) -> bool:
    process = handle.process
    return process.pid is not None and process.is_alive()


def _partial_frame_error(
    error: BaseException,
    output_dir: Path,
) -> FullPipelineFrameError:
    return FullPipelineFrameError(
        "partial output: transcript outputs were saved, but frame extraction "
        "did not complete; no incomplete staged generation was published and any "
        "prior frame generation remains public or in its recovery backup at "
        f"{output_dir}: "
        f"{type(error).__name__}: {error}"
    )


def run_supervised_full_pipeline(
    video_path: str | Path,
    output_dir: str | Path,
    preflight: TranscriptPreflight,
    *,
    supervisor: Any,
    frame_runner: Callable[[], Any],
    frame_device: str | None = None,
    scheduler: StageScheduler | None = None,
    fallback_runner: Callable[..., Any] | None = None,
    clock: Callable[[], float] | None = None,
) -> FullPipelineRunResult:
    """Run transcription, diarization, frames, merge, and publication in order."""

    video = Path(video_path)
    output_dir = Path(output_dir)
    config = preflight.config
    scheduler = scheduler or StageScheduler(config.stage_concurrency)
    fallback_runner = fallback_runner or complete_transcription_with_auto_fallback
    clock = clock or time.monotonic
    frame_device = frame_device or resolve_frame_device(preflight)
    frame_stage = frame_demand(frame_device)
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
    diarization_stage = (
        diarization_demand(preflight.effective_diarization_device)
        if preflight.effective_diarization_device is not None
        else None
    )
    initial_stages = [transcription_stage]
    if diarization_stage is not None:
        initial_stages.append(diarization_stage)
    initial_schedule = scheduler.decide(initial_stages)
    _print_preflight(preflight)
    _print_schedule(initial_schedule)

    if supervisor is None or supervisor.public is None:
        raise RuntimeError("stage supervisor did not initialize public paths")
    if Path(supervisor.output_dir).resolve() != output_dir.resolve():
        raise ValueError("borrowed stage supervisor owns a different output directory")

    supervisor.public.diarization.unlink(missing_ok=True)
    timings: dict[str, float] = {}
    diarization_handle = None
    diarization_started: float | None = None
    diarization_completion = None
    diarization_settled = False

    def start_diarization() -> Any:
        nonlocal diarization_handle, diarization_started
        if diarization_stage is None:
            raise RuntimeError("diarization is disabled")
        if diarization_handle is not None:
            return diarization_handle
        diarization_started = clock()
        diarization_handle = supervisor.start_diarization(
            video,
            hf_token=preflight.hf_token or "",
            final_output_paths=final_paths,
            thread_budget=initial_schedule.cpu_threads_for("diarization"),
            device=preflight.effective_diarization_device,
        )
        return diarization_handle

    def settle_diarization() -> Any | None:
        nonlocal diarization_completion, diarization_settled
        if diarization_handle is None or diarization_settled:
            return diarization_completion
        diarization_settled = True
        try:
            diarization_completion = supervisor.complete(diarization_handle)
        except Exception as exc:
            supervisor.public.diarization.unlink(missing_ok=True)
            transcript._print_speaker_detection_failure(exc)
            diarization_completion = None
        finally:
            if diarization_started is not None:
                timings["diarization"] = clock() - diarization_started
        return diarization_completion

    transcription_started = clock()
    transcription_handle = supervisor.start_transcription(
        video,
        model_name=config.model_name,
        requested_backend=config.transcription_backend,
        final_output_paths=final_paths,
        thread_budget=initial_schedule.cpu_threads_for("transcription"),
    )
    if initial_schedule.parallel and diarization_stage is not None:
        start_diarization()

    active_stages = ()
    if diarization_handle is not None and diarization_stage is not None:
        active_stages = (ActiveStage(diarization_stage, diarization_handle),)
    try:
        execution = fallback_runner(
            supervisor,
            transcription_handle,
            scheduler=scheduler,
            video_path=video,
            model_name=config.model_name,
            requested_backend=config.transcription_backend,
            effective_backend=preflight.effective_backend,
            active_stages=active_stages,
            final_output_paths=final_paths,
        )
    except BaseException as exc:
        if diarization_handle is not None:
            try:
                supervisor.cancel(diarization_handle)
            except BaseException as cancel_error:
                exc.add_note(
                    "failed to cancel diarization after transcription failure: "
                    f"{type(cancel_error).__name__}: {cancel_error}"
                )
        supervisor.public.diarization.unlink(missing_ok=True)
        raise
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
            supervisor.cancel(diarization_handle)
            diarization_settled = True
            if diarization_started is not None:
                timings["diarization"] = clock() - diarization_started
            supervisor.public.diarization.unlink(missing_ok=True)
        diarization_stage = None
    elif preflight.missing_hf_token:
        transcript._print_missing_hf_token_warning()

    frame_schedule: ScheduleDecision
    frame_overlapped_diarization = False
    if diarization_stage is None:
        frame_schedule = scheduler.decide((frame_stage,))
    elif diarization_handle is None:
        # An initially serialized run remains serialized so its logged critical
        # path and worker budgets match the actual execution.
        start_diarization()
        settle_diarization()
        frame_schedule = scheduler.decide((frame_stage,))
    elif _handle_is_running(diarization_handle):
        # Re-probe after transcription releases its model/accelerator. This is
        # the only point at which frames may overlap a running diarization worker.
        frame_schedule = scheduler.decide((frame_stage, diarization_stage))
        frame_overlapped_diarization = frame_schedule.parallel
    else:
        settle_diarization()
        frame_schedule = scheduler.decide((frame_stage,))
    print("Frame-stage admission after transcription:")
    _print_schedule(frame_schedule)

    if diarization_stage is None:
        critical_path = "T + F + M + E"
    elif (
        not initial_schedule.parallel
        or (
            execution.fallback_schedule is not None
            and not execution.fallback_schedule.parallel
        )
    ):
        critical_path = "T + D + F + M + E"
    elif frame_overlapped_diarization:
        critical_path = "max(T + F, D) + M + E"
    else:
        critical_path = "max(T, D) + F + M + E"
    print(f"Full-run critical path: {critical_path}")

    if (
        diarization_handle is not None
        and not diarization_settled
        and not frame_overlapped_diarization
    ):
        settle_diarization()

    frame_generation = None
    frame_error: Exception | None = None
    frame_started = clock()
    try:
        frame_generation = frame_runner()
    except Exception as exc:
        frame_error = exc
    finally:
        timings["frames"] = clock() - frame_started

    if diarization_handle is not None and not diarization_settled:
        settle_diarization()

    merge_started = clock()
    if diarization_completion is not None and segments:
        try:
            labeled = transcript._assign_speakers(
                segments,
                diarization_completion.records,
            )
            if not any(segment.speaker for segment in labeled):
                raise RuntimeError(
                    "speaker diarization produced no usable speaker overlaps"
                )
            segments = tuple(labeled)
        except Exception as exc:
            supervisor.public.diarization.unlink(missing_ok=True)
            transcript._print_speaker_detection_failure(exc)

    try:
        _write_final_outputs(segments, output_paths, config.fmt)
    except BaseException as exc:
        if frame_error is not None:
            exc.add_note(
                "frame extraction also failed before transcript output writing: "
                f"{type(frame_error).__name__}: {frame_error}"
            )
        raise
    timings["merge"] = clock() - merge_started
    if frame_error is not None:
        _print_transcript_result(segments, language, timings)
        raise _partial_frame_error(frame_error, output_dir) from frame_error
    if frame_generation is None:
        raise RuntimeError("frame runner returned no staged generation")

    enrichment_started = clock()
    try:
        frame_generation.enrich_manifest(segments)
        public_frames = frame_generation.promote()
    except Exception as exc:
        timings["manifest"] = clock() - enrichment_started
        _print_transcript_result(segments, language, timings)
        raise _partial_frame_error(exc, output_dir) from exc
    timings["manifest"] = clock() - enrichment_started
    _print_transcript_result(segments, language, timings)

    transcript_result = TranscriptRunResult(
        segments=segments,
        language=language,
        effective_backend=effective_backend,
        fallback_used=execution.fallback_used,
        initial_schedule=initial_schedule,
        timings=dict(timings),
    )
    return FullPipelineRunResult(
        transcript=transcript_result,
        frames=public_frames,
        frame_device=frame_device,
        initial_schedule=initial_schedule,
        frame_schedule=frame_schedule,
        critical_path=critical_path,
        timings=dict(timings),
    )
