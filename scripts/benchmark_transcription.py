#!/usr/bin/env python3
"""Run and replay the Keyframe 0.6.0 transcription release benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import platform
import resource
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
from keyframe.artifacts import atomic_write_json
from keyframe.full_pipeline import resolve_frame_device, run_supervised_full_pipeline
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
REPORT_SCHEMA_VERSION = 1
DEFAULT_TIMESTAMP_TOLERANCE_SECONDS = 0.05
DEFAULT_CRITICAL_PATH_TOLERANCE_SECONDS = 5.0


class BenchmarkError(RuntimeError):
    """The benchmark could not run or did not satisfy its release contract."""


@dataclass(frozen=True)
class _CaseRequest:
    name: str
    input_path: str
    output_dir: str


def _maximum_resident_set_gib() -> float:
    scale = 1 if sys.platform == "darwin" else 1024
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * scale
    return (own + children) / GIB


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
        "backend": result.effective_backend,
        "device": preflight.transcription_device,
        "diarization_device": preflight.effective_diarization_device,
        "schedule_mode": result.initial_schedule.mode,
        "schedule_reason": result.initial_schedule.reason,
        "wall_time_seconds": wall_time,
        "timings": dict(result.timings),
        "peak_memory_gib": _maximum_resident_set_gib(),
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
            "parallel",
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
    model_spec = transcript.MLX_MODEL_SPECS[args.whisper_model]
    frame_device = resolve_frame_device(preflight)
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
            frame_runner=lambda: cli._run_frame_generation(
                video,
                output_dir,
                args,
                supervisor,
                frame_device=frame_device,
            ),
        )
    wall_time = time.monotonic() - started
    if not result.initial_schedule.parallel:
        raise BenchmarkError(
            "the controlled release candidate did not overlap MLX transcription "
            "with CPU diarization"
        )
    if result.critical_path != "max(T + F, D) + M + E":
        raise BenchmarkError(
            "the controlled Apple release candidate reported an unexpected "
            f"critical path: {result.critical_path}"
        )
    return {
        "name": request.name,
        "backend": result.transcript.effective_backend,
        "device": preflight.transcription_device,
        "diarization_device": preflight.effective_diarization_device,
        "frame_device": result.frame_device,
        "model_repository": model_spec.repository,
        "model_revision": model_spec.revision,
        "schedule_mode": result.initial_schedule.mode,
        "schedule_reason": result.initial_schedule.reason,
        "frame_schedule_mode": result.frame_schedule.mode,
        "frame_schedule_reason": result.frame_schedule.reason,
        "critical_path": result.critical_path,
        "wall_time_seconds": wall_time,
        "timings": dict(result.timings),
        "peak_memory_gib": _maximum_resident_set_gib(),
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
    return dict(message["result"])


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
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


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
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        failures.append(
            f"report schema_version must be {REPORT_SCHEMA_VERSION}"
        )
    baseline_model = _mapping(baseline.get("model"))
    candidate = _mapping(report.get("candidate"))
    if candidate.get("backend") != "mlx":
        failures.append("candidate did not select the MLX backend")
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
            report["critical_path_validation"] = {
                "expression": expression,
                "predicted_seconds": predicted,
                "measured_wall_seconds": measured,
                "absolute_delta_seconds": abs(predicted - measured),
                "tolerance_seconds": critical_path_tolerance_seconds,
            }
            if abs(predicted - measured) > critical_path_tolerance_seconds:
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
    if not input_path.is_file():
        raise BenchmarkError(f"benchmark input does not exist: {input_path}")
    if not baseline_path.is_file():
        raise BenchmarkError(f"benchmark baseline does not exist: {baseline_path}")
    if args.timestamp_tolerance_seconds < 0:
        raise BenchmarkError("timestamp tolerance must be non-negative")
    if args.critical_path_tolerance_seconds < 0:
        raise BenchmarkError("critical-path tolerance must be non-negative")
    baseline = _load_json(baseline_path, "baseline")
    duration_seconds = _probe_duration_seconds(input_path)
    expected_duration = float(baseline.get("recording", {}).get("duration_seconds", 0.0))
    if abs(duration_seconds - expected_duration) > 1.0:
        raise BenchmarkError(
            "benchmark input duration does not match the checked-in baseline: "
            f"expected {expected_duration:.3f}s, found {duration_seconds:.3f}s"
        )

    if args.replay_report:
        report = _load_json(Path(args.replay_report).expanduser(), "benchmark report")
        _validate_report_recording(report, input_path, duration_seconds)
        report_path = Path(args.report).expanduser() if args.report else None
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
            Path(args.report).expanduser()
            if args.report
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
