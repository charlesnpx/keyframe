"""Resource-aware orchestration for the CLI full-extraction dependency graph."""

from __future__ import annotations

import math
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
    configure_worker_thread_budget,
    diarization_demand,
    frame_demand,
    transcription_demand,
)
from keyframe.transcript_cli import (
    TranscriptPreflight,
    TranscriptRunResult,
    _final_output_paths,
    _final_output_staging_root,
    _print_preflight,
    _print_schedule,
    _print_transcript_result,
    _write_final_outputs,
)


class FullPipelineFrameError(OutputSessionError):
    """The transcript completed but the staged frame generation did not."""

    pipeline_evidence: PipelineEvidence | None = None


@dataclass(frozen=True)
class StageInterval:
    """One half-open, parent-monotonic stage interval."""

    stage: str
    launch_wave: str
    started_at: float
    ended_at: float
    outcome: str

    def __post_init__(self) -> None:
        if self.stage not in {"transcription", "diarization", "frames"}:
            raise ValueError(f"unknown pipeline stage: {self.stage!r}")
        if self.launch_wave not in {"initial", "post-transcription"}:
            raise ValueError(f"unknown launch wave: {self.launch_wave!r}")
        if self.outcome not in {"completed", "failed", "cancelled"}:
            raise ValueError(f"unknown stage outcome: {self.outcome!r}")
        values = (float(self.started_at), float(self.ended_at))
        if any(not math.isfinite(value) for value in values):
            raise ValueError("stage interval bounds must be finite")
        if values[1] < values[0]:
            raise ValueError("stage interval end precedes its start")
        object.__setattr__(self, "started_at", values[0])
        object.__setattr__(self, "ended_at", values[1])

    @property
    def duration_seconds(self) -> float:
        return self.ended_at - self.started_at

    def overlaps(self, other: StageInterval) -> bool:
        return max(self.started_at, other.started_at) < min(
            self.ended_at,
            other.ended_at,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "launch_wave": self.launch_wave,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": self.duration_seconds,
            "outcome": self.outcome,
        }


@dataclass(frozen=True)
class PipelineEvidence:
    """Reliable launch topology independent of lossy progress output."""

    intervals: tuple[StageInterval, ...]

    def interval(self, stage: str) -> StageInterval | None:
        matches = tuple(item for item in self.intervals if item.stage == stage)
        if len(matches) > 1:
            raise ValueError(f"pipeline evidence has duplicate {stage!r} intervals")
        return matches[0] if matches else None

    def to_dict(self) -> dict[str, Any]:
        return {item.stage: item.to_dict() for item in self.intervals}


class _PipelineEvidenceBuilder:
    _ORDER = {"transcription": 0, "diarization": 1, "frames": 2}

    def __init__(self) -> None:
        self._open: dict[str, tuple[str, float]] = {}
        self._closed: dict[str, StageInterval] = {}

    def start(self, stage: str, launch_wave: str, started_at: float) -> None:
        if stage in self._open or stage in self._closed:
            raise RuntimeError(f"pipeline stage {stage!r} was launched more than once")
        self._open[stage] = (launch_wave, float(started_at))

    def finish(self, stage: str, ended_at: float, outcome: str) -> StageInterval:
        if stage in self._closed:
            return self._closed[stage]
        try:
            launch_wave, started_at = self._open.pop(stage)
        except KeyError as exc:
            raise RuntimeError(f"pipeline stage {stage!r} was never launched") from exc
        interval = StageInterval(
            stage=stage,
            launch_wave=launch_wave,
            started_at=started_at,
            ended_at=ended_at,
            outcome=outcome,
        )
        self._closed[stage] = interval
        return interval

    def snapshot(self) -> PipelineEvidence:
        intervals = tuple(
            sorted(
                self._closed.values(),
                key=lambda item: self._ORDER[item.stage],
            )
        )
        return PipelineEvidence(intervals)


@dataclass(frozen=True)
class FullPipelineRunResult:
    transcript: TranscriptRunResult
    frames: Any
    frame_device: str
    initial_schedule: ScheduleDecision
    frame_schedule: ScheduleDecision
    critical_path: str
    timings: Mapping[str, float]
    pipeline_evidence: PipelineEvidence
    fallback_waited_for_diarization: bool

    @property
    def transcription_metadata(self) -> Mapping[str, Any]:
        return self.transcript.metadata


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
    evidence: PipelineEvidence,
) -> FullPipelineFrameError:
    wrapped = FullPipelineFrameError(
        "partial output: transcript outputs were saved, but frame extraction "
        "did not complete; no incomplete staged generation was published and any "
        "prior frame generation remains public or in its recovery backup at "
        f"{output_dir}: "
        f"{type(error).__name__}: {error}"
    )
    wrapped.pipeline_evidence = evidence
    return wrapped


def _attach_pipeline_evidence(
    error: BaseException,
    evidence: PipelineEvidence,
) -> None:
    try:
        error.pipeline_evidence = evidence
    except (AttributeError, TypeError):
        error.add_note(f"pipeline evidence: {evidence.to_dict()!r}")


def critical_path_from_pipeline_evidence(
    evidence: PipelineEvidence,
    *,
    fallback_waited_for_diarization: bool,
) -> str:
    transcription_interval = evidence.interval("transcription")
    frame_interval = evidence.interval("frames")
    diarization_interval = evidence.interval("diarization")
    if transcription_interval is None or frame_interval is None:
        raise RuntimeError("completed pipeline evidence is missing required stages")
    if transcription_interval.launch_wave != "initial":
        raise RuntimeError("transcription must belong to the initial launch wave")
    if frame_interval.launch_wave != "post-transcription":
        raise RuntimeError("frames must belong to the post-transcription launch wave")
    if transcription_interval.ended_at > frame_interval.started_at:
        raise RuntimeError("frames started before transcription completed")
    if diarization_interval is not None:
        if (
            diarization_interval.launch_wave == "initial"
            and diarization_interval.started_at > transcription_interval.ended_at
        ):
            raise RuntimeError(
                "initial-wave diarization started after transcription completed"
            )
        if (
            diarization_interval.launch_wave == "post-transcription"
            and diarization_interval.started_at < transcription_interval.ended_at
        ):
            raise RuntimeError(
                "post-transcription diarization started before transcription completed"
            )
    if fallback_waited_for_diarization:
        if (
            diarization_interval is None
            or diarization_interval.launch_wave != "initial"
            or diarization_interval.outcome == "cancelled"
            or diarization_interval.ended_at > transcription_interval.ended_at
        ):
            raise RuntimeError(
                "fallback wait evidence requires settled initial-wave diarization "
                "within transcription"
            )
    if (
        diarization_interval is None
        or diarization_interval.outcome == "cancelled"
        or fallback_waited_for_diarization
    ):
        return "T + F + M + E"
    if diarization_interval.launch_wave == "initial":
        if diarization_interval.overlaps(frame_interval):
            return "max(T + F, D) + M + E"
        return "max(T, D) + F + M + E"
    if diarization_interval.overlaps(frame_interval):
        return "T + max(D, F) + M + E"
    return "T + D + F + M + E"


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
    evidence_builder = _PipelineEvidenceBuilder()
    timings: dict[str, float] = {}
    diarization_handle = None
    diarization_started: float | None = None
    diarization_completion = None
    diarization_settled = False
    diarization_observed_end: float | None = None

    def start_diarization(
        decision: ScheduleDecision,
        launch_wave: str,
    ) -> Any:
        nonlocal diarization_handle, diarization_settled, diarization_started
        if diarization_stage is None:
            raise RuntimeError("diarization is disabled")
        if diarization_handle is not None:
            return diarization_handle
        diarization_started = clock()
        evidence_builder.start("diarization", launch_wave, diarization_started)
        try:
            diarization_handle = supervisor.start_diarization(
                video,
                hf_token=preflight.hf_token or "",
                final_output_paths=final_paths,
                thread_budget=decision.cpu_threads_for("diarization"),
                device=preflight.effective_diarization_device,
            )
        except BaseException as exc:
            diarization_settled = True
            finish_diarization("failed")
            _attach_pipeline_evidence(exc, evidence_builder.snapshot())
            raise
        return diarization_handle

    def finish_diarization(
        outcome: str,
        *,
        ended_at: float | None = None,
    ) -> None:
        if diarization_started is None:
            return
        effective_end = (
            ended_at
            if ended_at is not None
            else diarization_observed_end
            if diarization_observed_end is not None
            else clock()
        )
        interval = evidence_builder.finish("diarization", effective_end, outcome)
        timings["diarization"] = interval.duration_seconds

    def settle_diarization() -> Any | None:
        nonlocal diarization_completion, diarization_settled
        if diarization_handle is None or diarization_settled:
            return diarization_completion
        diarization_settled = True
        outcome = "completed"
        try:
            diarization_completion = supervisor.complete(diarization_handle)
        except Exception as exc:
            outcome = "failed"
            supervisor.public.diarization.unlink(missing_ok=True)
            transcript._print_speaker_detection_failure(exc)
            diarization_completion = None
        finally:
            finish_diarization(outcome)
        return diarization_completion

    def cancel_diarization_after_error(
        error: BaseException,
        *,
        context: str,
    ) -> None:
        nonlocal diarization_settled
        if diarization_started is None or diarization_settled:
            return
        if diarization_handle is None:
            diarization_settled = True
            finish_diarization("failed")
            return
        try:
            supervisor.cancel(diarization_handle)
        except BaseException as cancel_error:
            error.add_note(
                f"failed to cancel diarization {context}: "
                f"{type(cancel_error).__name__}: {cancel_error}"
            )
        finally:
            diarization_settled = True
            finish_diarization("cancelled")

    transcription_started = clock()
    evidence_builder.start("transcription", "initial", transcription_started)
    transcription_handle = None
    try:
        transcription_handle = supervisor.start_transcription(
            video,
            model_name=config.model_name,
            requested_backend=config.transcription_backend,
            final_output_paths=final_paths,
            thread_budget=initial_schedule.cpu_threads_for("transcription"),
        )
        if initial_schedule.parallel and diarization_stage is not None:
            start_diarization(initial_schedule, "initial")
    except BaseException as exc:
        transcription_outcome = "failed"
        if transcription_handle is not None:
            transcription_outcome = "cancelled"
            try:
                supervisor.cancel(transcription_handle)
            except BaseException as cancel_error:
                exc.add_note(
                    "failed to cancel transcription after worker launch failure: "
                    f"{type(cancel_error).__name__}: {cancel_error}"
                )
        transcription_interval = evidence_builder.finish(
            "transcription",
            clock(),
            transcription_outcome,
        )
        timings["transcription"] = transcription_interval.duration_seconds
        cancel_diarization_after_error(
            exc,
            context="after worker launch failure",
        )
        supervisor.public.diarization.unlink(missing_ok=True)
        _attach_pipeline_evidence(exc, evidence_builder.snapshot())
        raise

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
            clock=clock,
        )
    except BaseException as exc:
        transcription_ended = clock()
        transcription_interval = evidence_builder.finish(
            "transcription",
            transcription_ended,
            "failed",
        )
        timings["transcription"] = transcription_interval.duration_seconds
        settled_active_stages = {
            stage: (ended_at, outcome)
            for stage, ended_at, outcome in getattr(
                exc,
                "settled_active_stages",
                (),
            )
        }
        if diarization_handle is not None and "diarization" in settled_active_stages:
            ended_at, outcome = settled_active_stages["diarization"]
            diarization_settled = True
            finish_diarization(outcome, ended_at=ended_at)
        else:
            cancel_diarization_after_error(
                exc,
                context="after transcription failure",
            )
        supervisor.public.diarization.unlink(missing_ok=True)
        _attach_pipeline_evidence(exc, evidence_builder.snapshot())
        raise
    transcription_ended = clock()
    transcription_interval = evidence_builder.finish(
        "transcription",
        transcription_ended,
        "completed",
    )
    timings["transcription"] = transcription_interval.duration_seconds
    settled_active_stages = {
        stage: (ended_at, outcome)
        for stage, ended_at, outcome in execution.settled_active_stages
    }
    if "diarization" in settled_active_stages:
        diarization_observed_end = settled_active_stages["diarization"][0]
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
            if _handle_is_running(diarization_handle):
                supervisor.cancel(diarization_handle)
                diarization_settled = True
                finish_diarization("cancelled")
            else:
                settle_diarization()
            supervisor.public.diarization.unlink(missing_ok=True)
        diarization_stage = None
    elif preflight.missing_hf_token:
        transcript._print_missing_hf_token_warning()

    frame_schedule: ScheduleDecision
    diarization_runs_with_frames = False
    if diarization_stage is None:
        frame_schedule = scheduler.decide((frame_stage,))
    elif diarization_handle is None:
        frame_schedule = scheduler.decide((diarization_stage, frame_stage))
        diarization_runs_with_frames = frame_schedule.parallel
    elif _handle_is_running(diarization_handle):
        frame_schedule = scheduler.decide((diarization_stage, frame_stage))
        diarization_runs_with_frames = frame_schedule.parallel
    else:
        settle_diarization()
        frame_schedule = scheduler.decide((frame_stage,))
    print("Frame-stage admission after transcription:")
    _print_schedule(frame_schedule)

    if diarization_stage is not None and diarization_handle is None:
        start_diarization(frame_schedule, "post-transcription")
    if (
        diarization_handle is not None
        and not diarization_settled
        and not diarization_runs_with_frames
    ):
        settle_diarization()

    frame_generation = None
    frame_error: BaseException | None = None
    frame_started = clock()
    evidence_builder.start("frames", "post-transcription", frame_started)
    frame_outcome = "completed"
    try:
        try:
            configure_worker_thread_budget(
                frame_schedule.cpu_threads_for("frames"),
                torch_threads=True,
            )
            frame_generation = frame_runner()
        except SystemExit as exc:
            if exc.code in (None, 0):
                frame_outcome = "cancelled"
                raise
            frame_outcome = "failed"
            frame_error = exc
        except Exception as exc:
            frame_outcome = "failed"
            frame_error = exc
        except BaseException:
            frame_outcome = "cancelled"
            raise
        finally:
            frame_ended = clock()
            frame_interval = evidence_builder.finish(
                "frames",
                frame_ended,
                frame_outcome,
            )
            timings["frames"] = frame_interval.duration_seconds
    except BaseException as exc:
        cancel_diarization_after_error(
            exc,
            context="after parent interruption",
        )
        supervisor.public.diarization.unlink(missing_ok=True)
        _attach_pipeline_evidence(exc, evidence_builder.snapshot())
        raise

    if diarization_handle is not None and not diarization_settled:
        settle_diarization()

    pipeline_evidence = evidence_builder.snapshot()
    critical_path = critical_path_from_pipeline_evidence(
        pipeline_evidence,
        fallback_waited_for_diarization=execution.waited_for_active_stages,
    )
    if execution.waited_for_active_stages:
        print("Fallback timing note: any serialized D wait is included in T.")
    print(f"Full-run critical path: {critical_path}")

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
        _write_final_outputs(
            segments,
            output_paths,
            config.fmt,
            staging_root=_final_output_staging_root(supervisor),
        )
    except BaseException as exc:
        if frame_error is not None:
            exc.add_note(
                "frame extraction also failed before transcript output writing: "
                f"{type(frame_error).__name__}: {frame_error}"
            )
        _attach_pipeline_evidence(exc, pipeline_evidence)
        raise
    timings["merge"] = clock() - merge_started
    if frame_error is not None:
        _print_transcript_result(segments, language, timings)
        raise _partial_frame_error(
            frame_error,
            output_dir,
            pipeline_evidence,
        ) from frame_error
    if frame_generation is None:
        error = RuntimeError("frame runner returned no staged generation")
        _attach_pipeline_evidence(error, pipeline_evidence)
        raise error

    enrichment_started = clock()
    try:
        frame_generation.enrich_manifest(segments)
        public_frames = frame_generation.promote()
    except Exception as exc:
        timings["manifest"] = clock() - enrichment_started
        _print_transcript_result(segments, language, timings)
        raise _partial_frame_error(
            exc,
            output_dir,
            pipeline_evidence,
        ) from exc
    timings["manifest"] = clock() - enrichment_started
    _print_transcript_result(segments, language, timings)

    transcript_result = TranscriptRunResult(
        segments=segments,
        language=language,
        effective_backend=effective_backend,
        fallback_used=execution.fallback_used,
        initial_schedule=initial_schedule,
        timings=dict(timings),
        metadata=dict(execution.completion.metadata),
    )
    return FullPipelineRunResult(
        transcript=transcript_result,
        frames=public_frames,
        frame_device=frame_device,
        initial_schedule=initial_schedule,
        frame_schedule=frame_schedule,
        critical_path=critical_path,
        timings=dict(timings),
        pipeline_evidence=pipeline_evidence,
        fallback_waited_for_diarization=execution.waited_for_active_stages,
    )
