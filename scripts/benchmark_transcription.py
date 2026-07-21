#!/usr/bin/env python3
"""Run and replay the Keyframe 0.6.0 transcription release benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from keyframe import cli, transcript
from keyframe.artifacts import (
    ArtifactPathCollisionError,
    atomic_write_json,
    reject_path_aliases,
)
from keyframe.full_pipeline import (
    PipelineEvidence,
    StageInterval,
    critical_path_from_pipeline_evidence,
    resolve_frame_device,
    run_supervised_full_pipeline,
)
from keyframe.process_memory import (
    conservative_process_tree_high_water,
    resource_peak_rss_bytes,
)
from keyframe.stage_supervisor import StageSupervisor
from keyframe.transcript_cli import (
    TranscriptRunConfig,
    preflight_transcript_run,
    print_stage_progress,
    run_supervised_transcript,
)
from keyframe.validation import (
    CRITICAL_PATH_EXPRESSIONS,
    compare_diarization_partitions,
    compare_transcript_quality,
    expected_critical_path_seconds,
)


GIB = 1024**3
REPORT_SCHEMA_VERSION = 4
DEFAULT_TIMESTAMP_TOLERANCE_SECONDS = 0.05
DEFAULT_CRITICAL_PATH_TOLERANCE_SECONDS = 5.0
PROCESS_TREE_PEAK_METHOD = "conservative-kernel-high-water-bound"
HISTORICAL_CANDIDATE_WALL_SECONDS = 613.67
MAX_HISTORICAL_WALL_MULTIPLIER = 1.15
MAX_SERIAL_REFERENCE_WALL_MULTIPLIER = 0.85
MAX_PROCESS_TREE_RSS_GIB = 6.60
MAX_MLX_ALLOCATOR_PEAK_GIB = 5.96
MAX_LOCAL_MODEL_RESOLUTION_SECONDS = 1.0
EXPECTED_RUNTIME_PACKAGES = {
    "keyframe": "0.6.0",
    "mlx": "0.32.0",
    "mlx_whisper": "0.4.3",
    "whisperx": "3.8.6",
}
REFERENCE_CONTRACT = {
    "requested_backend": "whisper",
    "backend": "whisper",
    "device": "cpu",
    "diarization_device": "cpu",
    "schedule_policy": "serial",
    "schedule_mode": "serial",
}
CANDIDATE_CONTRACT = {
    "requested_backend": "auto",
    "backend": "mlx",
    "device": "mlx",
    "diarization_device": "cpu",
    "frame_device": "mps",
    "schedule_policy": "auto",
    "schedule_mode": "parallel",
    "schedule_source": "macos-memory-pressure",
    "frame_schedule_policy": "auto",
    "frame_schedule_mode": "parallel",
    "frame_schedule_source": "macos-memory-pressure",
    "fallback_used": False,
    "fallback_waited_for_diarization": False,
    "model_resolution_source": "local-hit",
}


class BenchmarkError(RuntimeError):
    """The benchmark could not run or did not satisfy its release contract."""


@dataclass(frozen=True)
class _CaseRequest:
    name: str
    input_path: str
    output_dir: str


def _artifact_summary(output_dir: Path) -> dict[str, bool]:
    final_json = output_dir / "transcript.json"
    labeled = False
    if final_json.is_file():
        try:
            rows = json.loads(final_json.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            rows = []
        labeled = isinstance(rows, list) and any(
            isinstance(row, dict) and bool(row.get("speaker"))
            for row in rows
        )
    return {
        "raw_checkpoint": (output_dir / "transcript.raw.json").is_file(),
        "diarization_checkpoint": (output_dir / "diarization.json").is_file(),
        "final_txt": (output_dir / "transcript.txt").is_file(),
        "final_json": final_json.is_file(),
        "speaker_labeled_final": labeled,
        "frames_manifest": (output_dir / "frames" / "manifest.json").is_file(),
    }


def _run_reference_case(request: _CaseRequest) -> dict[str, Any]:
    video = Path(request.input_path)
    output_dir = Path(request.output_dir)
    config = TranscriptRunConfig(
        model_name="medium",
        fmt="txt",
        transcription_backend="whisper",
        diarization_device="cpu",
        stage_concurrency="serial",
        speaker_detection=True,
    )
    preflight = preflight_transcript_run(config)
    if preflight.hf_token is None:
        raise BenchmarkError("the reference diarization run requires HF_TOKEN")
    started = time.monotonic()
    result = run_supervised_transcript(video, output_dir, preflight)
    wall_time = time.monotonic() - started
    return {
        "name": request.name,
        "requested_backend": config.transcription_backend,
        "backend": result.effective_backend,
        "device": preflight.transcription_device,
        "diarization_device": preflight.effective_diarization_device,
        "schedule_policy": result.initial_schedule.policy,
        "schedule_mode": result.initial_schedule.mode,
        "schedule_reason": result.initial_schedule.reason,
        "wall_time_seconds": wall_time,
        "timings": dict(result.timings),
        "artifacts": _artifact_summary(output_dir),
    }


def _run_candidate_case(request: _CaseRequest) -> dict[str, Any]:
    video = Path(request.input_path)
    output_dir = Path(request.output_dir)
    args = cli._parse_extract_args(
        [
            str(video),
            "--output",
            str(output_dir),
            "--whisper-model",
            "medium",
            "--transcript-format",
            "txt",
            "--transcription-backend",
            "auto",
            "--diarization-device",
            "cpu",
            "--stage-concurrency",
            "auto",
        ]
    )
    preflight = preflight_transcript_run(cli._transcript_config(args))
    if preflight.effective_backend != "mlx":
        raise BenchmarkError(
            "the release candidate must select MLX on supported Apple Silicon; "
            f"selected {preflight.effective_backend!r}"
        )
    if preflight.hf_token is None:
        raise BenchmarkError("the concurrent diarization run requires HF_TOKEN")
    frame_device = resolve_frame_device(preflight)
    case_process_phase_peaks: dict[str, int] = {}

    def run_frames(supervisor: StageSupervisor) -> Any:
        case_process_phase_peaks["initial-wave"] = resource_peak_rss_bytes(
            "self"
        )
        try:
            return cli._run_frame_generation(
                video,
                output_dir,
                args,
                supervisor,
                frame_device=frame_device,
            )
        finally:
            case_process_phase_peaks["second-wave"] = resource_peak_rss_bytes(
                "self"
            )

    started = time.monotonic()
    with StageSupervisor(
        output_dir,
        progress_callback=print_stage_progress,
    ) as supervisor:
        result = run_supervised_full_pipeline(
            video,
            output_dir,
            preflight,
            supervisor=supervisor,
            frame_device=frame_device,
            frame_runner=lambda: run_frames(supervisor),
        )
    wall_time = time.monotonic() - started
    if not result.initial_schedule.parallel:
        raise BenchmarkError(
            "the controlled release candidate did not overlap MLX transcription "
            "with CPU diarization"
        )
    stage_peak_rss_bytes = supervisor.completed_stage_peak_rss_bytes()
    current_case_peak = resource_peak_rss_bytes("self")
    case_process_phase_peaks.setdefault("initial-wave", current_case_peak)
    case_process_phase_peaks.setdefault("second-wave", current_case_peak)
    for label, schedule in (
        ("initial", result.initial_schedule),
        ("second-wave", result.frame_schedule),
    ):
        if schedule.policy != "auto":
            raise BenchmarkError(
                f"the {label} candidate schedule did not use automatic policy"
            )
        if schedule.mode != "parallel":
            raise BenchmarkError(
                f"the {label} automatic schedule did not admit parallel work"
            )
        if schedule.resources.source != "macos-memory-pressure":
            raise BenchmarkError(
                f"the {label} automatic schedule did not use macOS pressure evidence"
            )
    if result.transcript.fallback_used:
        raise BenchmarkError("the release candidate fell back from pinned MLX")
    missing_peak_stages = {"transcription", "diarization"} - set(
        stage_peak_rss_bytes
    )
    if missing_peak_stages:
        raise BenchmarkError(
            "the release candidate is missing worker high-water evidence for: "
            + ", ".join(sorted(missing_peak_stages))
        )
    if result.critical_path != "max(T + F, D) + M + E":
        raise BenchmarkError(
            "the controlled Apple release candidate reported an unexpected "
            f"critical path: {result.critical_path}"
        )
    metadata = result.transcript.metadata
    return {
        "name": request.name,
        "requested_backend": preflight.config.transcription_backend,
        "backend": result.transcript.effective_backend,
        "device": preflight.transcription_device,
        "diarization_device": preflight.effective_diarization_device,
        "frame_device": result.frame_device,
        "model_repository": metadata.get("model_repository"),
        "model_revision": metadata.get("model_revision"),
        "model_resolution_source": metadata.get("model_resolution_source"),
        "model_resolution_seconds": metadata.get("model_resolution_seconds"),
        "mlx_peak_memory_bytes": metadata.get("mlx_peak_memory_bytes"),
        "stage_process_tree_peak_rss_bytes": stage_peak_rss_bytes,
        "case_process_phase_peak_rss_bytes": case_process_phase_peaks,
        "fallback_used": result.transcript.fallback_used,
        "schedule_policy": result.initial_schedule.policy,
        "schedule_mode": result.initial_schedule.mode,
        "schedule_source": result.initial_schedule.resources.source,
        "schedule_reason": result.initial_schedule.reason,
        "frame_schedule_policy": result.frame_schedule.policy,
        "frame_schedule_mode": result.frame_schedule.mode,
        "frame_schedule_source": result.frame_schedule.resources.source,
        "frame_schedule_reason": result.frame_schedule.reason,
        "critical_path": result.critical_path,
        "pipeline_evidence": result.pipeline_evidence.to_dict(),
        "fallback_waited_for_diarization": (
            result.fallback_waited_for_diarization
        ),
        "wall_time_seconds": wall_time,
        "timings": dict(result.timings),
        "artifacts": _artifact_summary(output_dir),
    }


def _case_worker(request: _CaseRequest, terminal_send: Any) -> None:
    try:
        runner = (
            _run_reference_case
            if request.name == "whisper_cpu_serial"
            else _run_candidate_case
        )
        result = runner(request)
        case_process_peak = resource_peak_rss_bytes("self")
        stage_peaks = result.get("stage_process_tree_peak_rss_bytes", {})
        case_phase_peaks = dict(
            result.get("case_process_phase_peak_rss_bytes", {})
        )
        if case_phase_peaks:
            case_phase_peaks["finalization"] = case_process_peak
            transcription_peak = stage_peaks.get("transcription", 0)
            diarization_peak = stage_peaks.get("diarization", 0)
            phase_stage_peaks = {
                "initial-wave": (transcription_peak, diarization_peak),
                "second-wave": (diarization_peak,),
                "finalization": (),
            }
        else:
            case_phase_peaks = {"complete": case_process_peak}
            phase_stage_peaks = {"complete": tuple(stage_peaks.values())}
        result["case_process_phase_peak_rss_bytes"] = case_phase_peaks
        memory_evidence = conservative_process_tree_high_water(
            case_process_bytes=max(case_phase_peaks.values()),
            max_reaped_child_bytes=resource_peak_rss_bytes("children"),
            concurrent_stage_peaks=stage_peaks,
            case_phase_peaks=case_phase_peaks,
            phase_stage_peaks=phase_stage_peaks,
        )
        result["peak_memory_method"] = PROCESS_TREE_PEAK_METHOD
        result["peak_memory_components_bytes"] = memory_evidence.to_dict()
        result["peak_memory_gib"] = memory_evidence.tree_upper_bound_bytes / GIB
        atomic_write_json(
            Path(request.output_dir) / "benchmark-case.json",
            result,
            allow_nan=False,
        )
        terminal_send.send({"status": "success", "result": result})
    except BaseException as exc:
        try:
            terminal_send.send(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        terminal_send.close()


def _run_isolated_case(request: _CaseRequest) -> dict[str, Any]:
    context = mp.get_context("spawn")
    terminal_receive, terminal_send = context.Pipe(duplex=False)
    process = context.Process(
        target=_case_worker,
        args=(request, terminal_send),
        name=f"keyframe-benchmark-{request.name}",
        daemon=False,
    )
    try:
        process.start()
    except BaseException:
        terminal_receive.close()
        terminal_send.close()
        raise
    terminal_send.close()
    message: dict[str, Any] | None = None
    try:
        while True:
            if terminal_receive.poll(0.2):
                try:
                    message = terminal_receive.recv()
                except EOFError:
                    pass
                break
            if not process.is_alive():
                break
    except BaseException:
        process.join(timeout=10.0)
        if process.is_alive():
            process.terminate()
        process.join(timeout=10.0)
        if process.is_alive():
            process.kill()
            process.join()
        raise
    finally:
        terminal_receive.close()
    process.join()
    exitcode = process.exitcode
    process.close()
    if message is None:
        raise BenchmarkError(
            f"benchmark case {request.name} exited without a result "
            f"(status {exitcode})"
        )
    if exitcode != 0 or message.get("status") != "success":
        detail = message.get("traceback") or message.get("error_message") or "unknown error"
        raise BenchmarkError(f"benchmark case {request.name} failed:\n{detail}")
    result = dict(message["result"])
    atomic_write_json(
        Path(request.output_dir) / "benchmark-case.json",
        result,
        allow_nan=False,
    )
    return result


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"could not read {label} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BenchmarkError(f"{label} must contain a JSON object: {path}")
    return value


def _probe_duration_seconds(input_path: Path) -> float:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(input_path),
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        return float(result.stdout.strip())
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError) as exc:
        raise BenchmarkError(f"could not determine input duration with ffprobe: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _pipeline_evidence_from_report(candidate: dict[str, Any]) -> PipelineEvidence:
    raw_evidence = candidate.get("pipeline_evidence")
    if not isinstance(raw_evidence, dict):
        raise ValueError("candidate.pipeline_evidence must be an object")
    unexpected = set(raw_evidence) - {"transcription", "diarization", "frames"}
    if unexpected:
        raise ValueError(
            "candidate.pipeline_evidence has unknown stages: "
            + ", ".join(sorted(str(stage) for stage in unexpected))
        )

    intervals = []
    for stage, raw_interval in raw_evidence.items():
        if not isinstance(raw_interval, dict):
            raise ValueError(
                f"candidate.pipeline_evidence.{stage} must be an object"
            )
        if raw_interval.get("stage") != stage:
            raise ValueError(
                f"candidate.pipeline_evidence.{stage}.stage must be {stage!r}"
            )
        try:
            interval = StageInterval(
                stage=stage,
                launch_wave=raw_interval["launch_wave"],
                started_at=raw_interval["started_at"],
                ended_at=raw_interval["ended_at"],
                outcome=raw_interval["outcome"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"candidate.pipeline_evidence.{stage} is invalid: {exc}"
            ) from exc
        try:
            reported_duration = _finite_number(
                raw_interval["duration_seconds"],
                f"candidate.pipeline_evidence.{stage}.duration_seconds",
            )
        except KeyError as exc:
            raise ValueError(
                f"candidate.pipeline_evidence.{stage} is missing duration_seconds"
            ) from exc
        if not math.isclose(
            reported_duration,
            interval.duration_seconds,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                f"candidate.pipeline_evidence.{stage}.duration_seconds "
                "does not match its interval"
            )
        intervals.append(interval)

    evidence = PipelineEvidence(tuple(intervals))
    transcription_interval = evidence.interval("transcription")
    frame_interval = evidence.interval("frames")
    if transcription_interval is None or frame_interval is None:
        raise ValueError(
            "candidate.pipeline_evidence must include transcription and frames"
        )
    if transcription_interval.outcome != "completed":
        raise ValueError("candidate transcription interval must be completed")
    if frame_interval.outcome != "completed":
        raise ValueError("candidate frame interval must be completed")
    timings = candidate.get("timings")
    if not isinstance(timings, dict):
        raise ValueError("candidate.timings must be an object")
    for interval in intervals:
        try:
            reported_timing = _finite_number(
                timings[interval.stage],
                f"candidate.timings.{interval.stage}",
            )
        except KeyError as exc:
            raise ValueError(
                f"candidate.timings is missing {interval.stage}"
            ) from exc
        if not math.isclose(
            reported_timing,
            interval.duration_seconds,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                f"candidate.timings.{interval.stage} does not match its interval"
            )

    return evidence


def _critical_path_from_reported_evidence(candidate: dict[str, Any]) -> str:
    evidence = _pipeline_evidence_from_report(candidate)
    waited = candidate.get("fallback_waited_for_diarization")
    if not isinstance(waited, bool):
        raise ValueError(
            "candidate.fallback_waited_for_diarization must be a boolean"
        )
    try:
        return critical_path_from_pipeline_evidence(
            evidence,
            fallback_waited_for_diarization=waited,
        )
    except RuntimeError as exc:
        raise ValueError(f"candidate pipeline topology is invalid: {exc}") from exc


def _release_contract_failures(report: dict[str, Any]) -> list[str]:
    failures = []
    reference = _mapping(report.get("reference"))
    candidate = _mapping(report.get("candidate"))
    runtime = _mapping(report.get("runtime"))

    for case_name, case, contract in (
        ("reference", reference, REFERENCE_CONTRACT),
        ("candidate", candidate, CANDIDATE_CONTRACT),
    ):
        for field, expected in contract.items():
            if case.get(field) != expected:
                failures.append(
                    f"{case_name} {field} must be {expected!r}, "
                    f"found {case.get(field)!r}"
                )

    for package, expected in EXPECTED_RUNTIME_PACKAGES.items():
        if runtime.get(package) != expected:
            failures.append(
                f"runtime {package} must be {expected!r}, "
                f"found {runtime.get(package)!r}"
            )
    if runtime.get("system") != "Darwin":
        failures.append("runtime system must be 'Darwin'")
    if str(runtime.get("machine", "")).lower() != "arm64":
        failures.append("runtime machine must be 'arm64'")
    try:
        python_parts = tuple(
            int(part) for part in str(runtime["python"]).split(".")[:2]
        )
    except (KeyError, TypeError, ValueError):
        python_parts = ()
    if python_parts not in {(3, 11), (3, 12), (3, 13)}:
        failures.append("runtime Python must be a supported 3.11 through 3.13 release")

    try:
        evidence = _pipeline_evidence_from_report(candidate)
    except ValueError as exc:
        failures.append(f"candidate pipeline evidence is invalid: {exc}")
    else:
        transcription_interval = evidence.interval("transcription")
        diarization_interval = evidence.interval("diarization")
        frame_interval = evidence.interval("frames")
        assert transcription_interval is not None
        assert frame_interval is not None
        if diarization_interval is None:
            failures.append("candidate pipeline evidence is missing diarization")
        else:
            if diarization_interval.outcome != "completed":
                failures.append("candidate diarization interval must be completed")
            if not transcription_interval.overlaps(diarization_interval):
                failures.append(
                    "candidate transcription and diarization intervals must overlap"
                )
            if not frame_interval.overlaps(diarization_interval):
                failures.append("candidate frames and diarization intervals must overlap")
        if transcription_interval.ended_at > frame_interval.started_at:
            failures.append(
                "candidate frames must not start before transcription completes"
            )
        try:
            topology_expression = _critical_path_from_reported_evidence(candidate)
        except ValueError as exc:
            failures.append(f"candidate pipeline evidence is invalid: {exc}")
            return failures
        if candidate.get("critical_path") != topology_expression:
            failures.append(
                "candidate critical path does not match its pipeline evidence: "
                f"expected {topology_expression!r}"
            )
    return failures


def _performance_contract_failures(report: dict[str, Any]) -> list[str]:
    failures = []
    reference = _mapping(report.get("reference"))
    candidate = _mapping(report.get("candidate"))
    try:
        reference_wall = _finite_number(
            reference.get("wall_time_seconds"),
            "reference.wall_time_seconds",
        )
        candidate_wall = _finite_number(
            candidate.get("wall_time_seconds"),
            "candidate.wall_time_seconds",
        )
        process_peak_gib = _finite_number(
            candidate.get("peak_memory_gib"),
            "candidate.peak_memory_gib",
        )
        resolution_seconds = _finite_number(
            candidate.get("model_resolution_seconds"),
            "candidate.model_resolution_seconds",
        )
        mlx_peak_bytes = candidate.get("mlx_peak_memory_bytes")
        if isinstance(mlx_peak_bytes, bool) or not isinstance(mlx_peak_bytes, int):
            raise ValueError("candidate.mlx_peak_memory_bytes must be an integer")
        if mlx_peak_bytes < 0:
            raise ValueError(
                "candidate.mlx_peak_memory_bytes must be non-negative"
            )
        if min(
            reference_wall,
            candidate_wall,
            process_peak_gib,
            resolution_seconds,
        ) < 0:
            raise ValueError("performance measurements must be non-negative")
        if reference_wall == 0:
            raise ValueError("reference.wall_time_seconds must be positive")
        if candidate.get("peak_memory_method") != PROCESS_TREE_PEAK_METHOD:
            raise ValueError(
                "candidate.peak_memory_method must use kernel high-water evidence"
            )
        raw_components = candidate.get("peak_memory_components_bytes")
        if not isinstance(raw_components, dict):
            raise ValueError(
                "candidate.peak_memory_components_bytes must be an object"
            )
        scalar_component_names = {
            "case_process_bytes",
            "max_reaped_child_bytes",
            "concurrent_stage_sum_bytes",
            "descendant_bound_bytes",
            "tree_upper_bound_bytes",
        }
        if set(raw_components) != scalar_component_names | {
            "phase_upper_bound_bytes"
        }:
            raise ValueError(
                "candidate.peak_memory_components_bytes has invalid fields"
            )
        components: dict[str, int] = {}
        for name in scalar_component_names:
            value = raw_components[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"candidate.peak_memory_components_bytes.{name} "
                    "must be a non-negative integer"
                )
            components[name] = value
        raw_phase_bounds = raw_components["phase_upper_bound_bytes"]
        raw_case_phases = candidate.get("case_process_phase_peak_rss_bytes")
        phase_names = {"initial-wave", "second-wave", "finalization"}
        if (
            not isinstance(raw_phase_bounds, dict)
            or set(raw_phase_bounds) != phase_names
            or not isinstance(raw_case_phases, dict)
            or set(raw_case_phases) != phase_names
        ):
            raise ValueError("candidate process high-water phases are invalid")
        phase_bounds: dict[str, int] = {}
        case_phases: dict[str, int] = {}
        for name in phase_names:
            for source, target, label in (
                (raw_phase_bounds, phase_bounds, "phase bound"),
                (raw_case_phases, case_phases, "case phase"),
            ):
                value = source[name]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                ):
                    raise ValueError(
                        f"candidate {label} {name!r} must be a "
                        "non-negative integer"
                    )
                target[name] = value
        raw_stage_peaks = candidate.get("stage_process_tree_peak_rss_bytes")
        if not isinstance(raw_stage_peaks, dict):
            raise ValueError(
                "candidate.stage_process_tree_peak_rss_bytes must be an object"
            )
        if set(raw_stage_peaks) != {"transcription", "diarization"}:
            raise ValueError("candidate worker high-water evidence is incomplete")
        stage_peaks: dict[str, int] = {}
        for stage, value in raw_stage_peaks.items():
            if (
                stage not in {"transcription", "diarization"}
                or isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError("candidate worker high-water evidence is invalid")
            stage_peaks[stage] = value
        stage_peak_sum = sum(stage_peaks.values())
        expected_phase_stage_sums = {
            "initial-wave": stage_peak_sum,
            "second-wave": stage_peaks["diarization"],
            "finalization": 0,
        }
        expected_phase_bounds = {
            name: case_phases[name]
            + max(
                components["max_reaped_child_bytes"],
                expected_phase_stage_sums[name],
            )
            for name in phase_names
        }
        if phase_bounds != expected_phase_bounds:
            raise ValueError("candidate phase high-water bounds are inconsistent")
        expected_descendant_bound = max(
            components["max_reaped_child_bytes"],
            max(expected_phase_stage_sums.values()),
        )
        expected_tree_bound = max(expected_phase_bounds.values())
        if components["case_process_bytes"] != max(case_phases.values()):
            raise ValueError("candidate case-process high-water peak is inconsistent")
        if components["concurrent_stage_sum_bytes"] != stage_peak_sum:
            raise ValueError(
                "candidate worker high-water sum disagrees with memory evidence"
            )
        if components["descendant_bound_bytes"] != expected_descendant_bound:
            raise ValueError("candidate descendant high-water bound is inconsistent")
        if components["tree_upper_bound_bytes"] != expected_tree_bound:
            raise ValueError("candidate process-tree high-water bound is inconsistent")
        if not math.isclose(
            process_peak_gib,
            expected_tree_bound / GIB,
            rel_tol=0.0,
            abs_tol=1 / GIB,
        ):
            raise ValueError(
                "candidate.peak_memory_gib disagrees with high-water evidence"
            )
    except ValueError as exc:
        return [f"performance measurements are invalid: {exc}"]

    historical_limit = (
        HISTORICAL_CANDIDATE_WALL_SECONDS * MAX_HISTORICAL_WALL_MULTIPLIER
    )
    same_run_limit = reference_wall * MAX_SERIAL_REFERENCE_WALL_MULTIPLIER
    if candidate_wall > historical_limit:
        failures.append("candidate wall time exceeds the historical runtime limit")
    if candidate_wall > same_run_limit:
        failures.append(
            "candidate is not at least 15 percent faster than the serial reference"
        )
    if process_peak_gib > MAX_PROCESS_TREE_RSS_GIB:
        failures.append(
            "candidate process-tree RSS high-water bound exceeds 6.60 GiB"
        )
    if mlx_peak_bytes > MAX_MLX_ALLOCATOR_PEAK_GIB * GIB:
        failures.append("candidate MLX allocator peak exceeds 5.96 GiB")
    if resolution_seconds >= MAX_LOCAL_MODEL_RESOLUTION_SECONDS:
        failures.append("candidate cached MLX resolution did not finish under one second")

    report["performance_validation"] = {
        "historical_candidate_wall_seconds": HISTORICAL_CANDIDATE_WALL_SECONDS,
        "historical_multiplier": MAX_HISTORICAL_WALL_MULTIPLIER,
        "same_run_serial_reference_multiplier": (
            MAX_SERIAL_REFERENCE_WALL_MULTIPLIER
        ),
        "process_tree_rss_ceiling_gib": MAX_PROCESS_TREE_RSS_GIB,
        "process_tree_peak_method": PROCESS_TREE_PEAK_METHOD,
        "mlx_allocator_peak_ceiling_gib": MAX_MLX_ALLOCATOR_PEAK_GIB,
        "local_resolution_ceiling_seconds": MAX_LOCAL_MODEL_RESOLUTION_SECONDS,
        "historical_wall_limit_seconds": historical_limit,
        "same_run_wall_limit_seconds": same_run_limit,
    }
    return failures


def _quality_failures(
    report: dict[str, Any],
    baseline: dict[str, Any],
) -> list[str]:
    failures = []
    thresholds = baseline.get("quality_thresholds")
    if not isinstance(thresholds, dict):
        return ["baseline is missing quality_thresholds"]
    quality = report.get("quality")
    if not isinstance(quality, dict):
        return ["report is missing transcript quality metrics"]

    minimums = {
        "normalized_word_agreement": "minimum_normalized_word_agreement_vs_cpu",
        "character_agreement": "minimum_character_agreement_vs_cpu",
        "exact_opening_segments": "minimum_exact_opening_segments",
    }
    maximums = {
        "normalized_word_error_rate": "maximum_normalized_word_error_rate_vs_cpu",
        "segment_count_relative_delta": "maximum_segment_count_relative_delta",
    }
    try:
        for metric, threshold_name in minimums.items():
            measured = _finite_number(quality.get(metric), f"quality.{metric}")
            threshold = _finite_number(
                thresholds.get(threshold_name),
                f"quality_thresholds.{threshold_name}",
            )
            if measured < threshold:
                failures.append(f"{metric} is below {threshold_name}")
        for metric, threshold_name in maximums.items():
            measured = _finite_number(quality.get(metric), f"quality.{metric}")
            threshold = _finite_number(
                thresholds.get(threshold_name),
                f"quality_thresholds.{threshold_name}",
            )
            if measured > threshold:
                failures.append(f"{metric} exceeds {threshold_name}")

        duplicate_increase = int(quality["candidate_duplicate_ngrams"]) - int(
            quality["reference_duplicate_ngrams"]
        )
        maximum_duplicate_increase = int(
            thresholds["maximum_duplicate_five_gram_increase"]
        )
        if duplicate_increase > maximum_duplicate_increase:
            failures.append("candidate duplicate five-grams increased")
        if thresholds.get("require_no_opening_loss") and int(
            quality["exact_opening_segments"]
        ) < int(thresholds["minimum_exact_opening_segments"]):
            failures.append("candidate has opening loss")
        if thresholds.get("require_no_long_form_collapse"):
            reference_end = _finite_number(
                quality.get("reference_end_seconds"),
                "quality.reference_end_seconds",
            )
            candidate_end = _finite_number(
                quality.get("candidate_end_seconds"),
                "quality.candidate_end_seconds",
            )
            if candidate_end + 1.0 < reference_end:
                failures.append("candidate has long-form coverage collapse")
    except (KeyError, TypeError, ValueError) as exc:
        failures.append(f"quality metrics or thresholds are invalid: {exc}")
    return failures


def evaluate_report(
    report: dict[str, Any],
    baseline: dict[str, Any],
    *,
    critical_path_tolerance_seconds: float,
) -> list[str]:
    """Return release-contract failures for a fresh or replayed report."""

    failures = _quality_failures(report, baseline)
    failures.extend(_release_contract_failures(report))
    failures.extend(_performance_contract_failures(report))
    try:
        critical_path_tolerance = _finite_number(
            critical_path_tolerance_seconds,
            "critical_path_tolerance_seconds",
        )
        if critical_path_tolerance < 0:
            raise ValueError("critical_path_tolerance_seconds must be non-negative")
    except ValueError as exc:
        failures.append(f"critical-path tolerance is invalid: {exc}")
        critical_path_tolerance = None
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        failures.append(
            f"report schema_version must be {REPORT_SCHEMA_VERSION}"
        )
    baseline_model = _mapping(baseline.get("model"))
    candidate = _mapping(report.get("candidate"))
    if candidate.get("model_repository") != baseline_model.get("mlx_repository"):
        failures.append("candidate MLX repository does not match the baseline")
    if candidate.get("model_revision") != baseline_model.get("mlx_revision"):
        failures.append("candidate MLX revision does not match the baseline")

    for case_name in ("reference", "candidate"):
        case = _mapping(report.get(case_name))
        artifacts = _mapping(case.get("artifacts"))
        for artifact in (
            "raw_checkpoint",
            "diarization_checkpoint",
            "final_txt",
            "final_json",
            "speaker_labeled_final",
        ):
            if not artifacts.get(artifact):
                failures.append(f"{case_name} is missing {artifact}")
    if not _mapping(candidate.get("artifacts")).get("frames_manifest"):
        failures.append("candidate is missing frames_manifest")

    diarization = _mapping(report.get("diarization"))
    if not diarization.get("equivalent"):
        failures.append(
            "candidate diarization partition changed: "
            f"{diarization.get('reason', 'unknown reason')}"
        )

    expression = candidate.get("critical_path")
    if expression not in CRITICAL_PATH_EXPRESSIONS:
        failures.append(f"candidate reported unsupported critical path {expression!r}")
    else:
        try:
            predicted = expected_critical_path_seconds(
                expression,
                _mapping(candidate.get("timings")),
            )
            measured = _finite_number(
                candidate.get("wall_time_seconds"),
                "candidate.wall_time_seconds",
            )
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(f"candidate critical path could not be evaluated: {exc}")
        else:
            if critical_path_tolerance is not None:
                report["critical_path_validation"] = {
                    "expression": expression,
                    "predicted_seconds": predicted,
                    "measured_wall_seconds": measured,
                    "absolute_delta_seconds": abs(predicted - measured),
                    "tolerance_seconds": critical_path_tolerance,
                }
                if abs(predicted - measured) > critical_path_tolerance:
                    failures.append("candidate wall time exceeds critical-path tolerance")
    return failures


def _new_report(
    input_path: Path,
    duration_seconds: float,
    baseline: dict[str, Any],
    output_dir: Path,
    *,
    timestamp_tolerance_seconds: float,
) -> dict[str, Any]:
    reference_dir = output_dir / "reference-whisper-cpu-serial"
    candidate_dir = output_dir / "candidate-mlx-concurrent-full"
    reference = _run_isolated_case(
        _CaseRequest("whisper_cpu_serial", str(input_path), str(reference_dir))
    )
    candidate = _run_isolated_case(
        _CaseRequest("mlx_concurrent_full", str(input_path), str(candidate_dir))
    )
    reference_segments = transcript.read_raw_transcript_checkpoint(
        reference_dir / "transcript.raw.json"
    )
    candidate_segments = transcript.read_raw_transcript_checkpoint(
        candidate_dir / "transcript.raw.json"
    )
    reference_diarization = transcript.read_diarization_checkpoint(
        reference_dir / "diarization.json"
    )
    candidate_diarization = transcript.read_diarization_checkpoint(
        candidate_dir / "diarization.json"
    )
    quality = compare_transcript_quality(reference_segments, candidate_segments)
    diarization = compare_diarization_partitions(
        reference_diarization,
        candidate_diarization,
        timestamp_tolerance_seconds=timestamp_tolerance_seconds,
    )
    model_spec = transcript.MLX_MODEL_SPECS["medium"]
    if model_spec.repository != baseline.get("model", {}).get("mlx_repository"):
        raise BenchmarkError("checked-in baseline names a different MLX repository")
    if model_spec.revision != baseline.get("model", {}).get("mlx_revision"):
        raise BenchmarkError("checked-in baseline names a different MLX revision")
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "recording": {
            "filename": input_path.name,
            "size_bytes": input_path.stat().st_size,
            "sha256": _sha256(input_path),
            "duration_seconds": duration_seconds,
        },
        "runtime": {
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "keyframe": _package_version("keyframe") or "source-checkout",
            "mlx": _package_version("mlx"),
            "mlx_whisper": _package_version("mlx-whisper"),
            "whisperx": _package_version("whisperx"),
        },
        "reference": reference,
        "candidate": candidate,
        "quality": asdict(quality),
        "diarization": asdict(diarization),
    }


def _prepare_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise BenchmarkError(f"benchmark output directory must be empty: {path}")


def _validate_report_recording(
    report: dict[str, Any],
    input_path: Path,
    duration_seconds: float,
) -> None:
    recording = _mapping(report.get("recording"))
    try:
        reported_duration = _finite_number(
            recording.get("duration_seconds"),
            "report recording duration_seconds",
        )
    except ValueError as exc:
        raise BenchmarkError(str(exc)) from exc
    if abs(reported_duration - duration_seconds) > 0.01:
        raise BenchmarkError(
            "replayed report does not describe the supplied recording duration"
        )
    reported_sha256 = recording.get("sha256")
    if not isinstance(reported_sha256, str) or reported_sha256 != _sha256(input_path):
        raise BenchmarkError(
            "replayed report does not describe the supplied recording contents"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or replay the Keyframe transcription release benchmark.",
    )
    parser.add_argument("--input", required=True, help="Explicit benchmark recording")
    parser.add_argument("--baseline", required=True, help="Checked-in baseline JSON")
    parser.add_argument(
        "--output",
        help="Empty output directory for fresh benchmark artifacts (default: /tmp)",
    )
    parser.add_argument("--report", help="Report JSON path (default: <output>/report.json)")
    parser.add_argument(
        "--replay-report",
        help="Validate an existing report without running models",
    )
    parser.add_argument(
        "--timestamp-tolerance-seconds",
        type=float,
        default=DEFAULT_TIMESTAMP_TOLERANCE_SECONDS,
    )
    parser.add_argument(
        "--critical-path-tolerance-seconds",
        type=float,
        default=DEFAULT_CRITICAL_PATH_TOLERANCE_SECONDS,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = Path(args.input).expanduser()
    baseline_path = Path(args.baseline).expanduser()
    replay_report_path = (
        Path(args.replay_report).expanduser() if args.replay_report else None
    )
    explicit_report_path = Path(args.report).expanduser() if args.report else None
    if not input_path.is_file():
        raise BenchmarkError(f"benchmark input does not exist: {input_path}")
    if not baseline_path.is_file():
        raise BenchmarkError(f"benchmark baseline does not exist: {baseline_path}")
    if replay_report_path is not None and not replay_report_path.is_file():
        raise BenchmarkError(
            f"benchmark replay report does not exist: {replay_report_path}"
        )
    if explicit_report_path is not None:
        protected_paths = [input_path, baseline_path]
        if replay_report_path is not None:
            protected_paths.append(replay_report_path)
        try:
            reject_path_aliases(explicit_report_path, protected_paths)
        except ArtifactPathCollisionError as exc:
            raise BenchmarkError(
                "benchmark report path must not alias the recording, baseline, "
                f"or replay report: {exc}"
            ) from exc
    if not math.isfinite(args.timestamp_tolerance_seconds) or args.timestamp_tolerance_seconds < 0:
        raise BenchmarkError("timestamp tolerance must be finite and non-negative")
    if (
        not math.isfinite(args.critical_path_tolerance_seconds)
        or args.critical_path_tolerance_seconds < 0
    ):
        raise BenchmarkError("critical-path tolerance must be finite and non-negative")
    baseline = _load_json(baseline_path, "baseline")
    duration_seconds = _probe_duration_seconds(input_path)
    expected_duration = float(baseline.get("recording", {}).get("duration_seconds", 0.0))
    if abs(duration_seconds - expected_duration) > 1.0:
        raise BenchmarkError(
            "benchmark input duration does not match the checked-in baseline: "
            f"expected {expected_duration:.3f}s, found {duration_seconds:.3f}s"
        )

    if replay_report_path is not None:
        report = _load_json(replay_report_path, "benchmark report")
        _validate_report_recording(report, input_path, duration_seconds)
        report_path = explicit_report_path
    else:
        output_dir = (
            Path(args.output).expanduser()
            if args.output
            else Path(tempfile.mkdtemp(prefix="keyframe-benchmark-", dir="/tmp"))
        )
        _prepare_output_dir(output_dir)
        report = _new_report(
            input_path,
            duration_seconds,
            baseline,
            output_dir,
            timestamp_tolerance_seconds=args.timestamp_tolerance_seconds,
        )
        report_path = (
            explicit_report_path
            if explicit_report_path is not None
            else output_dir / "report.json"
        )

    failures = evaluate_report(
        report,
        baseline,
        critical_path_tolerance_seconds=args.critical_path_tolerance_seconds,
    )
    report["validation"] = {"passed": not failures, "failures": failures}
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(report_path, report, allow_nan=False)
        print(f"Benchmark report: {report_path.resolve()}")
    print(json.dumps(report["validation"], indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BenchmarkError as exc:
        print(f"Benchmark error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    except KeyboardInterrupt:
        print("Benchmark interrupted", file=sys.stderr)
        raise SystemExit(130) from None
