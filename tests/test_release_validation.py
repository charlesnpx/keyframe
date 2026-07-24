from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from keyframe.transcript import TranscriptSegment
from keyframe.validation import (
    compare_transcript_quality,
    expected_critical_path_seconds,
    normalize_transcript_words,
)
from scripts import benchmark_transcription as benchmark
from tests.release_evidence_helpers import write_cross_platform_frame_evidence


ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "tests/fixtures/transcription-benchmark-baseline.json"


def _short_lived_child_peak_worker(terminal_send):
    subprocess.run(
        [
            sys.executable,
            "-c",
            "payload = bytearray(64 * 1024 * 1024); len(payload)",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    terminal_send.send(benchmark.resource_peak_rss_bytes("children"))
    terminal_send.close()


def _baseline() -> dict:
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def _artifacts(*, frames: bool) -> dict[str, bool]:
    return {
        "raw_checkpoint": True,
        "diarization_checkpoint": True,
        "final_txt": True,
        "final_json": True,
        "speaker_labeled_final": True,
        "frames_manifest": frames,
    }


def _stage_interval(
    stage: str,
    launch_wave: str,
    started_at: float,
    ended_at: float,
    *,
    outcome: str = "completed",
) -> dict:
    return {
        "stage": stage,
        "launch_wave": launch_wave,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_seconds": ended_at - started_at,
        "outcome": outcome,
    }


def _passing_report(input_path: Path) -> dict:
    baseline = _baseline()
    timings = {
        "transcription": 2.0,
        "diarization": 3.0,
        "frames": 4.0,
        "merge": 0.5,
        "manifest": 0.5,
    }
    return {
        "schema_version": benchmark.REPORT_SCHEMA_VERSION,
        "recording": {
            "duration_seconds": 988.75,
            "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        },
        "runtime": {
            "python": "3.12.13",
            "system": "Darwin",
            "machine": "arm64",
            **benchmark.EXPECTED_RUNTIME_PACKAGES,
        },
        "reference": {
            **benchmark.REFERENCE_CONTRACT,
            "wall_time_seconds": 20.0,
            "artifacts": _artifacts(frames=False),
        },
        "candidate": {
            **benchmark.CANDIDATE_CONTRACT,
            "model_repository": baseline["model"]["mlx_repository"],
            "model_revision": baseline["model"]["mlx_revision"],
            "model_resolution_seconds": 0.125,
            "mlx_peak_memory_bytes": 5 * benchmark.GIB,
            "peak_memory_gib": 4.0,
            "peak_memory_method": benchmark.PROCESS_TREE_PEAK_METHOD,
            "stage_process_tree_peak_rss_bytes": {
                "transcription": 3 * benchmark.GIB,
                "diarization": 1 * benchmark.GIB,
            },
            "case_process_phase_peak_rss_bytes": {
                "initial-wave": 1 * benchmark.GIB,
                "second-wave": 1 * benchmark.GIB,
                "finalization": 1 * benchmark.GIB,
            },
            "peak_memory_components_bytes": {
                "case_process_bytes": 1 * benchmark.GIB,
                "max_reaped_child_bytes": 3 * benchmark.GIB,
                "concurrent_stage_sum_bytes": 3 * benchmark.GIB,
                "descendant_bound_bytes": 3 * benchmark.GIB,
                "tree_upper_bound_bytes": 4 * benchmark.GIB,
                "phase_upper_bound_bytes": {
                    "initial-wave": 4 * benchmark.GIB,
                    "second-wave": 4 * benchmark.GIB,
                    "finalization": 4 * benchmark.GIB,
                },
            },
            "critical_path": "T + D + F + M + E",
            "pipeline_evidence": {
                "transcription": _stage_interval(
                    "transcription", "initial", 0.0, 2.0
                ),
                "diarization": _stage_interval(
                    "diarization", "post-transcription", 2.0, 5.0
                ),
                "frames": _stage_interval(
                    "frames", "post-transcription", 5.0, 9.0
                ),
            },
            "fallback_waited_for_diarization": False,
            "wall_time_seconds": 10.0,
            "timings": timings,
            "artifacts": _artifacts(frames=True),
        },
        "quality": {
            "normalized_word_agreement": 0.995,
            "normalized_word_error_rate": 0.01,
            "character_agreement": 0.995,
            "exact_opening_segments": 8,
            "segment_count_relative_delta": 0.01,
            "reference_duplicate_ngrams": 17,
            "candidate_duplicate_ngrams": 17,
            "reference_end_seconds": 980.0,
            "candidate_end_seconds": 980.0,
        },
        "diarization": {
            "equivalent": True,
            "reason": "partitions are equivalent",
        },
        "frame_evidence": write_cross_platform_frame_evidence(
            input_path.parent
        ),
    }


def test_release_contracts_compare_cpu_reference_to_mps_candidate():
    assert benchmark.REFERENCE_CONTRACT["diarization_device"] == "cpu"
    assert "diarization_attempted_devices" not in benchmark.REFERENCE_CONTRACT
    assert benchmark.CANDIDATE_CONTRACT["diarization_device"] == "mps"
    assert benchmark.CANDIDATE_CONTRACT["diarization_attempted_devices"] == [
        "mps"
    ]
    assert benchmark.CANDIDATE_CONTRACT["diarization_fallback_used"] is False
    assert benchmark.CANDIDATE_CONTRACT[
        "pytorch_mps_fallback_enabled"
    ] is False
    assert benchmark.CANDIDATE_CONTRACT["schedule_reason"] == (
        benchmark.APPLE_ACCELERATOR_SERIAL_REASON
    )
    assert benchmark.CANDIDATE_CONTRACT["frame_schedule_reason"] == (
        benchmark.APPLE_ACCELERATOR_SERIAL_REASON
    )


@pytest.mark.parametrize(
    "mutation, expected",
    [
        (
            lambda frame: frame.pop("linux_x86_64"),
            "frame evidence is missing linux_x86_64",
        ),
        (
            lambda frame: frame["darwin_arm64"]["evidence"]["targets"][0].update(
                {"passed": False}
            ),
            "darwin_arm64 has a failed frame target",
        ),
        (
            lambda frame: frame["linux_x86_64"]["evidence"]["budgets"].update(
                {"passed": False}
            ),
            "linux_x86_64 has a failed frame budget",
        ),
        (
            lambda frame: frame["linux_x86_64"]["evidence"][
                "redundancy"
            ].update({"passed": False}),
            "linux_x86_64 has a failed redundancy budget",
        ),
        (
            lambda frame: frame["darwin_arm64"]["evidence"]["platform"].update(
                {"machine": "x86_64"}
            ),
            "darwin_arm64 evidence machine must be 'arm64'",
        ),
        (
            lambda frame: frame["linux_x86_64"]["evidence"][
                "source_identity"
            ].update({"commit_sha": "b" * 40}),
            "Darwin and Linux frame evidence must use the same source_identity",
        ),
        (
            lambda frame: frame["linux_x86_64"]["evidence"]["packages"].update(
                {"keyframe": "0.6.2"}
            ),
            "linux_x86_64 package version must match source_identity",
        ),
    ],
)
def test_schema_7_aggregate_rejects_frame_gate_failures(
    tmp_path,
    mutation,
    expected,
):
    input_path = tmp_path / "recording.mp4"
    input_path.write_bytes(b"benchmark recording")
    report = _passing_report(input_path)
    mutation(report["frame_evidence"])

    failures = benchmark.evaluate_report(
        report,
        _baseline(),
        critical_path_tolerance_seconds=5.0,
    )

    assert expected in failures


def test_schema_7_aggregate_replays_embedded_artifact_trees(tmp_path):
    input_path = tmp_path / "recording.mp4"
    input_path.write_bytes(b"benchmark recording")
    report = _passing_report(input_path)
    png_record = report["frame_evidence"]["linux_x86_64"]["evidence"][
        "artifacts"
    ]["pngs"][0]
    png_path = (
        tmp_path
        / "frame-evidence"
        / "linux-x86_64"
        / png_record["path"]
    )
    png_path.write_bytes(png_path.read_bytes() + b"tamper")

    failures = benchmark.evaluate_report(
        report,
        _baseline(),
        critical_path_tolerance_seconds=5.0,
        report_root=tmp_path,
    )

    assert any(
        failure.startswith("linux_x86_64 replay:")
        for failure in failures
    )


def test_schema_7_aggregate_hashes_completed_standalone_reports(tmp_path):
    input_path = tmp_path / "recording.mp4"
    input_path.write_bytes(b"benchmark recording")
    report = _passing_report(input_path)
    report["frame_evidence"]["darwin_arm64"]["report_sha256"] = "0" * 64

    failures = benchmark.evaluate_report(
        report,
        _baseline(),
        critical_path_tolerance_seconds=5.0,
        report_root=tmp_path,
    )

    assert "darwin_arm64 standalone report hash does not match" in failures


def _set_process_tree_peak(report: dict, gibibytes: float) -> None:
    candidate = report["candidate"]
    components = candidate["peak_memory_components_bytes"]
    tree_upper_bound = int(gibibytes * benchmark.GIB)
    descendant_bound = max(
        components["max_reaped_child_bytes"],
        components["concurrent_stage_sum_bytes"],
    )
    components["case_process_bytes"] = tree_upper_bound - descendant_bound
    report["candidate"]["case_process_phase_peak_rss_bytes"][
        "initial-wave"
    ] = tree_upper_bound - descendant_bound
    components["descendant_bound_bytes"] = descendant_bound
    components["tree_upper_bound_bytes"] = tree_upper_bound
    components["phase_upper_bound_bytes"]["initial-wave"] = tree_upper_bound
    candidate["peak_memory_gib"] = tree_upper_bound / benchmark.GIB


def test_transcript_quality_normalizes_text_and_records_regression_signals():
    reference = (
        TranscriptSegment(0.0, 1.0, "Hello, WORLD!"),
        TranscriptSegment(1.0, 2.0, "We can’t stop now."),
        TranscriptSegment(2.0, 3.0, "alpha beta gamma alpha beta gamma"),
    )
    candidate = (
        TranscriptSegment(0.0, 1.0, "hello world"),
        TranscriptSegment(1.0, 2.0, "We can't pause now"),
        TranscriptSegment(2.0, 3.0, "alpha beta gamma alpha beta gamma"),
    )

    comparison = compare_transcript_quality(
        reference,
        candidate,
        duplicate_ngram_size=3,
    )

    assert normalize_transcript_words("It’s déjà-vu_2") == (
        "it's",
        "déjà",
        "vu",
        "2",
    )
    assert comparison.reference_word_count == 12
    assert comparison.candidate_word_count == 12
    assert comparison.normalized_word_edit_distance == 1
    assert comparison.normalized_word_error_rate == pytest.approx(1 / 12)
    assert comparison.normalized_word_agreement == pytest.approx(11 / 12)
    assert comparison.character_agreement < 1.0
    assert comparison.reference_duplicate_ngrams == 1
    assert comparison.candidate_duplicate_ngrams == 1
    assert comparison.exact_opening_segments == 1
    assert comparison.segment_count_relative_delta == 0.0
    assert comparison.reference_end_seconds == 3.0
    assert comparison.candidate_end_seconds == 3.0


def test_transcript_quality_rejects_invalid_rows_and_ngram_size():
    with pytest.raises(ValueError, match="invalid end"):
        compare_transcript_quality(
            [{"text": "bad", "end": float("nan")}],
            [],
        )
    with pytest.raises(ValueError, match="ngram size"):
        compare_transcript_quality([], [], duplicate_ngram_size=0)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("T + F + M + E", 16.0),
        ("max(T + F, D) + M + E", 23.0),
        ("max(T, D) + F + M + E", 26.0),
        ("T + max(D, F) + M + E", 33.0),
        ("T + D + F + M + E", 36.0),
        ("T + R + F + M + E", 20.0),
        ("T + R + max(D, F) + M + E", 37.0),
        ("T + R + D + F + M + E", 40.0),
        ("max(T, R) + F + M + E", 16.0),
        ("max(T, R) + max(D, F) + M + E", 33.0),
        ("max(T, R) + D + F + M + E", 36.0),
        ("T + max(R, F) + M + E", 17.0),
        ("T + max(R, F) + D + M + E", 37.0),
        ("max(T + F, R) + M + E", 16.0),
        ("max(T + F, R) + D + M + E", 36.0),
    ],
)
def test_expected_critical_path_supports_each_release_schedule(expression, expected):
    timings = {
        "transcription": 10.0,
        "diarization_retry": 4.0,
        "diarization": 20.0,
        "frames": 3.0,
        "merge": 2.0,
        "manifest": 1.0,
    }

    assert expected_critical_path_seconds(expression, timings) == expected


def test_critical_path_without_diarization_does_not_require_a_d_timing():
    assert expected_critical_path_seconds(
        "T + F + M + E",
        {
            "transcription": 10.0,
            "frames": 3.0,
            "merge": 2.0,
            "manifest": 1.0,
        },
    ) == 16.0


@pytest.mark.parametrize(
    ("expression", "timings"),
    [
        ("unknown", {}),
        ("T + D + F + M + E", {}),
        (
            "T + D + F + M + E",
            {
                "transcription": float("nan"),
                "diarization": 1,
                "frames": 1,
                "merge": 1,
                "manifest": 1,
            },
        ),
    ],
)
def test_expected_critical_path_rejects_unsupported_or_invalid_inputs(
    expression,
    timings,
):
    with pytest.raises(ValueError, match="critical-path|unsupported"):
        expected_critical_path_seconds(expression, timings)


@pytest.mark.parametrize(
    ("argv", "missing"),
    [
        (["--baseline", "baseline.json"], "--input"),
        (["--input", "recording.mp4"], "--baseline"),
    ],
)
def test_benchmark_cli_requires_explicit_input_and_baseline(argv, missing, capsys):
    with pytest.raises(SystemExit) as caught:
        benchmark.build_parser().parse_args(argv)

    assert caught.value.code == 2
    assert missing in capsys.readouterr().err


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--timestamp-tolerance-seconds", "nan", "timestamp tolerance"),
        ("--timestamp-tolerance-seconds", "inf", "timestamp tolerance"),
        ("--timestamp-tolerance-seconds", "-inf", "timestamp tolerance"),
        ("--critical-path-tolerance-seconds", "nan", "critical-path tolerance"),
        ("--critical-path-tolerance-seconds", "inf", "critical-path tolerance"),
        ("--critical-path-tolerance-seconds", "-inf", "critical-path tolerance"),
    ],
)
def test_benchmark_cli_rejects_nonfinite_tolerances_before_model_work(
    monkeypatch,
    tmp_path,
    option,
    value,
    message,
):
    input_path = tmp_path / "recording.mp4"
    input_path.write_bytes(b"benchmark recording")
    monkeypatch.setattr(
        benchmark,
        "_probe_duration_seconds",
        lambda _path: pytest.fail("tolerance validation must precede model work"),
    )

    with pytest.raises(benchmark.BenchmarkError, match=message):
        benchmark.main(
            [
                "--input",
                str(input_path),
                "--baseline",
                str(BASELINE_PATH),
                f"{option}={value}",
            ]
        )


def test_kernel_high_water_survives_a_short_lived_child_between_observations():
    context = mp.get_context("spawn")
    terminal_receive, terminal_send = context.Pipe(duplex=False)
    process = context.Process(
        target=_short_lived_child_peak_worker,
        args=(terminal_send,),
    )
    process.start()
    terminal_send.close()
    try:
        assert terminal_receive.poll(15.0)
        peak_bytes = terminal_receive.recv()
    finally:
        terminal_receive.close()
        process.join(timeout=15.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)

    assert process.exitcode == 0
    assert peak_bytes >= 64 * 1024 * 1024


def test_process_tree_high_water_sums_concurrent_stages_conservatively():
    evidence = benchmark.conservative_process_tree_high_water(
        case_process_bytes=1 * benchmark.GIB,
        max_reaped_child_bytes=3 * benchmark.GIB,
        concurrent_stage_peaks={
            "transcription": 4 * benchmark.GIB,
            "diarization": 2 * benchmark.GIB,
        },
    )

    assert evidence.concurrent_stage_sum_bytes == 6 * benchmark.GIB
    assert evidence.descendant_bound_bytes == 6 * benchmark.GIB
    assert evidence.tree_upper_bound_bytes == 7 * benchmark.GIB


def test_performance_thresholds_are_named_exact_and_finite():
    thresholds = {
        "historical": benchmark.HISTORICAL_CANDIDATE_WALL_SECONDS,
        "historical_multiplier": benchmark.MAX_HISTORICAL_WALL_MULTIPLIER,
        "same_run_multiplier": benchmark.MAX_SERIAL_REFERENCE_WALL_MULTIPLIER,
        "process_rss": benchmark.MAX_PROCESS_TREE_RSS_GIB,
        "mlx_peak": benchmark.MAX_MLX_ALLOCATOR_PEAK_GIB,
        "resolution": benchmark.MAX_LOCAL_MODEL_RESOLUTION_SECONDS,
        "mps_diarization": benchmark.MAX_MPS_DIARIZATION_SECONDS,
    }

    assert thresholds == {
        "historical": 613.67,
        "historical_multiplier": 1.15,
        "same_run_multiplier": 0.85,
        "process_rss": 6.60,
        "mlx_peak": 5.96,
        "resolution": 1.0,
        "mps_diarization": 335.0,
    }
    assert all(
        isinstance(value, float) and value > 0 and value < float("inf")
        for value in thresholds.values()
    )


def test_report_evaluation_passes_and_records_critical_path(tmp_path):
    input_path = tmp_path / "recording.mp4"
    input_path.write_bytes(b"benchmark recording")
    report = _passing_report(input_path)

    failures = benchmark.evaluate_report(
        report,
        _baseline(),
        critical_path_tolerance_seconds=0.01,
    )

    assert failures == []
    assert report["critical_path_validation"] == {
        "expression": "T + D + F + M + E",
        "predicted_seconds": 10.0,
        "measured_wall_seconds": 10.0,
        "absolute_delta_seconds": 0.0,
        "tolerance_seconds": 0.01,
    }
    assert report["performance_validation"] == {
        "historical_candidate_wall_seconds": 613.67,
        "historical_multiplier": 1.15,
        "same_run_serial_reference_multiplier": 0.85,
        "process_tree_rss_ceiling_gib": 6.6,
        "process_tree_peak_method": benchmark.PROCESS_TREE_PEAK_METHOD,
        "mlx_allocator_peak_ceiling_gib": 5.96,
        "local_resolution_ceiling_seconds": 1.0,
        "mps_diarization_ceiling_seconds": 335.0,
        "historical_wall_limit_seconds": pytest.approx(705.7205),
        "same_run_wall_limit_seconds": 17.0,
    }


@pytest.mark.parametrize(
    ("evidence", "fallback_waited", "expression", "wall_time"),
    [
        (
            {
                "transcription": _stage_interval(
                    "transcription", "initial", 0.0, 2.0
                ),
                "diarization": _stage_interval(
                    "diarization", "initial", 0.0, 3.0
                ),
                "frames": _stage_interval(
                    "frames", "post-transcription", 2.0, 6.0
                ),
            },
            False,
            "max(T + F, D) + M + E",
            7.0,
        ),
        (
            {
                "transcription": _stage_interval(
                    "transcription", "initial", 0.0, 2.0
                ),
                "diarization": _stage_interval(
                    "diarization", "initial", 0.0, 2.0
                ),
                "frames": _stage_interval(
                    "frames", "post-transcription", 2.0, 6.0
                ),
            },
            False,
            "max(T, D) + F + M + E",
            7.0,
        ),
        (
            {
                "transcription": _stage_interval(
                    "transcription", "initial", 0.0, 2.0
                ),
                "diarization": _stage_interval(
                    "diarization", "post-transcription", 2.0, 5.0
                ),
                "frames": _stage_interval(
                    "frames", "post-transcription", 2.0, 6.0
                ),
            },
            False,
            "T + max(D, F) + M + E",
            7.0,
        ),
        (
            {
                "transcription": _stage_interval(
                    "transcription", "initial", 0.0, 2.0
                ),
                "diarization": _stage_interval(
                    "diarization", "post-transcription", 2.0, 5.0
                ),
                "frames": _stage_interval(
                    "frames", "post-transcription", 5.0, 9.0
                ),
            },
            False,
            "T + D + F + M + E",
            10.0,
        ),
        (
            {
                "transcription": _stage_interval(
                    "transcription", "initial", 0.0, 2.0
                ),
                "frames": _stage_interval(
                    "frames", "post-transcription", 2.0, 6.0
                ),
            },
            False,
            "T + F + M + E",
            7.0,
        ),
        (
            {
                "transcription": _stage_interval(
                    "transcription", "initial", 0.0, 2.0
                ),
                "diarization_retry": _stage_interval(
                    "diarization_retry",
                    "post-transcription",
                    2.0,
                    3.0,
                    outcome="failed",
                ),
                "diarization": _stage_interval(
                    "diarization", "post-transcription", 3.0, 5.0
                ),
                "frames": _stage_interval(
                    "frames", "post-transcription", 3.0, 6.0
                ),
            },
            False,
            "T + R + max(D, F) + M + E",
            7.0,
        ),
        (
            {
                "transcription": _stage_interval(
                    "transcription", "initial", 0.0, 2.0
                ),
                "diarization_retry": _stage_interval(
                    "diarization_retry",
                    "post-transcription",
                    2.0,
                    3.0,
                    outcome="failed",
                ),
                "diarization": _stage_interval(
                    "diarization", "post-transcription", 3.0, 5.0
                ),
                "frames": _stage_interval(
                    "frames", "post-transcription", 5.0, 8.0
                ),
            },
            False,
            "T + R + D + F + M + E",
            9.0,
        ),
        (
            {
                "transcription": _stage_interval(
                    "transcription", "initial", 0.0, 2.0
                ),
                "diarization_retry": _stage_interval(
                    "diarization_retry",
                    "post-transcription",
                    2.0,
                    3.0,
                    outcome="failed",
                ),
                "frames": _stage_interval(
                    "frames", "post-transcription", 3.0, 6.0
                ),
            },
            False,
            "T + R + F + M + E",
            7.0,
        ),
        (
            {
                "transcription": _stage_interval(
                    "transcription", "initial", 0.0, 2.0
                ),
                "diarization_retry": _stage_interval(
                    "diarization_retry",
                    "initial",
                    0.0,
                    3.0,
                    outcome="failed",
                ),
                "diarization": _stage_interval(
                    "diarization", "post-transcription", 3.0, 5.0
                ),
                "frames": _stage_interval(
                    "frames", "post-transcription", 3.0, 6.0
                ),
            },
            False,
            "max(T, R) + max(D, F) + M + E",
            7.0,
        ),
        (
            {
                "transcription": _stage_interval(
                    "transcription", "initial", 0.0, 2.0
                ),
                "diarization_retry": _stage_interval(
                    "diarization_retry",
                    "initial",
                    0.0,
                    3.0,
                    outcome="failed",
                ),
                "diarization": _stage_interval(
                    "diarization", "post-transcription", 3.0, 5.0
                ),
                "frames": _stage_interval(
                    "frames", "post-transcription", 5.0, 8.0
                ),
            },
            False,
            "max(T, R) + D + F + M + E",
            9.0,
        ),
        (
            {
                "transcription": _stage_interval(
                    "transcription", "initial", 0.0, 2.0
                ),
                "diarization_retry": _stage_interval(
                    "diarization_retry",
                    "initial",
                    0.0,
                    3.0,
                    outcome="failed",
                ),
                "frames": _stage_interval(
                    "frames", "post-transcription", 3.0, 6.0
                ),
            },
            False,
            "max(T, R) + F + M + E",
            7.0,
        ),
        (
            {
                "transcription": _stage_interval(
                    "transcription", "initial", 0.0, 2.0
                ),
                "diarization_retry": _stage_interval(
                    "diarization_retry",
                    "post-transcription",
                    2.0,
                    5.0,
                    outcome="failed",
                ),
                "frames": _stage_interval(
                    "frames", "post-transcription", 2.0, 6.0
                ),
            },
            False,
            "T + max(R, F) + M + E",
            7.0,
        ),
        (
            {
                "transcription": _stage_interval(
                    "transcription", "initial", 0.0, 2.0
                ),
                "diarization_retry": _stage_interval(
                    "diarization_retry",
                    "post-transcription",
                    2.0,
                    5.0,
                    outcome="failed",
                ),
                "frames": _stage_interval(
                    "frames", "post-transcription", 2.0, 6.0
                ),
                "diarization": _stage_interval(
                    "diarization", "post-transcription", 6.0, 8.0
                ),
            },
            False,
            "T + max(R, F) + D + M + E",
            9.0,
        ),
        (
            {
                "transcription": _stage_interval(
                    "transcription", "initial", 0.0, 2.0
                ),
                "diarization_retry": _stage_interval(
                    "diarization_retry",
                    "initial",
                    0.0,
                    5.0,
                    outcome="failed",
                ),
                "frames": _stage_interval(
                    "frames", "post-transcription", 2.0, 6.0
                ),
            },
            False,
            "max(T + F, R) + M + E",
            7.0,
        ),
        (
            {
                "transcription": _stage_interval(
                    "transcription", "initial", 0.0, 2.0
                ),
                "diarization_retry": _stage_interval(
                    "diarization_retry",
                    "initial",
                    0.0,
                    5.0,
                    outcome="failed",
                ),
                "frames": _stage_interval(
                    "frames", "post-transcription", 2.0, 6.0
                ),
                "diarization": _stage_interval(
                    "diarization", "post-transcription", 6.0, 8.0
                ),
            },
            False,
            "max(T + F, R) + D + M + E",
            9.0,
        ),
    ],
)
def test_reported_evidence_derives_each_supported_critical_path(
    tmp_path,
    evidence,
    fallback_waited,
    expression,
    wall_time,
):
    input_path = tmp_path / "recording.mp4"
    input_path.write_bytes(b"benchmark recording")
    report = _passing_report(input_path)
    candidate = report["candidate"]
    candidate["pipeline_evidence"] = evidence
    candidate["fallback_waited_for_diarization"] = fallback_waited
    candidate["critical_path"] = expression
    candidate["timings"] = {
        stage: interval["duration_seconds"]
        for stage, interval in evidence.items()
    } | {"merge": 0.5, "manifest": 0.5}
    candidate["wall_time_seconds"] = wall_time

    assert benchmark._critical_path_from_reported_evidence(candidate) == expression
    assert expected_critical_path_seconds(
        expression,
        candidate["timings"],
    ) == pytest.approx(wall_time)


def test_reported_evidence_uses_fallback_wait_without_double_counting(
    tmp_path,
):
    input_path = tmp_path / "recording.mp4"
    input_path.write_bytes(b"benchmark recording")
    report = _passing_report(input_path)
    candidate = report["candidate"]
    candidate["pipeline_evidence"] = {
        "transcription": _stage_interval(
            "transcription", "initial", 0.0, 4.0
        ),
        "diarization": _stage_interval(
            "diarization", "initial", 0.0, 2.0
        ),
        "frames": _stage_interval(
            "frames", "post-transcription", 4.0, 8.0
        ),
    }
    candidate["fallback_waited_for_diarization"] = True
    candidate["critical_path"] = "T + F + M + E"
    candidate["timings"].update(transcription=4.0, diarization=2.0)
    candidate["wall_time_seconds"] = 9.0

    assert benchmark._critical_path_from_reported_evidence(candidate) == (
        "T + F + M + E"
    )
    assert expected_critical_path_seconds(
        candidate["critical_path"],
        candidate["timings"],
    ) == pytest.approx(9.0)


def test_report_evaluation_rejects_path_that_disagrees_with_pipeline_evidence(
    tmp_path,
):
    input_path = tmp_path / "recording.mp4"
    input_path.write_bytes(b"benchmark recording")
    report = _passing_report(input_path)
    report["candidate"]["critical_path"] = "max(T + F, D) + M + E"

    failures = benchmark.evaluate_report(
        report,
        _baseline(),
        critical_path_tolerance_seconds=5.0,
    )

    assert any(
        "critical path does not match its pipeline evidence" in failure
        for failure in failures
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda candidate: candidate["timings"].update(frames=3.5),
            "timings.frames does not match its interval",
        ),
        (
            lambda candidate: candidate.update(
                fallback_waited_for_diarization=True
            ),
            "fallback wait evidence",
        ),
    ],
)
def test_report_evaluation_rejects_inconsistent_reliable_topology(
    tmp_path,
    mutate,
    message,
):
    input_path = tmp_path / "recording.mp4"
    input_path.write_bytes(b"benchmark recording")
    report = _passing_report(input_path)
    mutate(report["candidate"])

    failures = benchmark.evaluate_report(
        report,
        _baseline(),
        critical_path_tolerance_seconds=5.0,
    )

    assert any(message in failure for failure in failures)


@pytest.mark.parametrize(
    ("mutation", "expected_failure"),
    [
        (
            lambda report: report["quality"].update(
                normalized_word_agreement=0.5
            ),
            "normalized_word_agreement is below",
        ),
        (
            lambda report: report["candidate"].update(model_revision="moving"),
            "candidate MLX revision does not match",
        ),
        (
            lambda report: report["candidate"]["artifacts"].update(
                frames_manifest=False
            ),
            "candidate is missing frames_manifest",
        ),
        (
            lambda report: report["diarization"].update(equivalent=False),
            "candidate diarization partition changed",
        ),
        (
            lambda report: report["candidate"].update(wall_time_seconds=20.0),
            "candidate wall time exceeds critical-path tolerance",
        ),
        (
            lambda report: report["candidate"].update(
                model_resolution_source="downloaded"
            ),
            "candidate model_resolution_source",
        ),
        (
            lambda report: report["candidate"].update(
                model_resolution_seconds=1.0
            ),
            "candidate cached MLX resolution did not finish under one second",
        ),
        (
            lambda report: report["candidate"].update(fallback_used=True),
            "candidate fallback_used",
        ),
        (
            lambda report: report["candidate"].update(
                pytorch_mps_fallback_enabled=True
            ),
            "candidate pytorch_mps_fallback_enabled",
        ),
        (
            lambda report: report["candidate"].update(
                schedule_policy="parallel"
            ),
            "candidate schedule_policy",
        ),
        (
            lambda report: report["candidate"].update(
                schedule_source="macos-vm-stat"
            ),
            "candidate schedule_source",
        ),
        (
            lambda report: report["candidate"].update(
                schedule_reason="memory admission failed"
            ),
            "candidate schedule_reason",
        ),
        (
            lambda report: report["candidate"].update(
                frame_schedule_source="macos-vm-stat"
            ),
            "candidate frame_schedule_source",
        ),
        (
            lambda report: report["candidate"].update(
                frame_schedule_reason="memory admission failed"
            ),
            "candidate frame_schedule_reason",
        ),
        (
            lambda report: report["candidate"].update(wall_time_seconds=800.0),
            "candidate wall time exceeds the historical runtime limit",
        ),
        (
            lambda report: report["reference"].update(wall_time_seconds=8.0),
            "candidate is not at least 15 percent faster",
        ),
        (
            lambda report: _set_process_tree_peak(report, 6.61),
            "candidate process-tree RSS high-water bound exceeds 6.60 GiB",
        ),
        (
            lambda report: report["candidate"].update(
                mlx_peak_memory_bytes=int(5.97 * benchmark.GIB)
            ),
            "candidate MLX allocator peak exceeds 5.96 GiB",
        ),
        (
            lambda report: report["candidate"]["timings"].update(
                diarization=336.0
            ),
            "candidate MPS diarization exceeds the 335-second break-even bound",
        ),
    ],
)
def test_report_evaluation_surfaces_release_threshold_failures(
    tmp_path,
    mutation,
    expected_failure,
):
    input_path = tmp_path / "recording.mp4"
    input_path.write_bytes(b"benchmark recording")
    report = _passing_report(input_path)
    mutation(report)

    failures = benchmark.evaluate_report(
        report,
        _baseline(),
        critical_path_tolerance_seconds=0.01,
    )

    assert any(expected_failure in failure for failure in failures)


def test_report_evaluation_rejects_release_configuration_drift(tmp_path):
    input_path = tmp_path / "recording.mp4"
    input_path.write_bytes(b"benchmark recording")
    report = _passing_report(input_path)
    report["reference"].update(
        backend="mlx",
        device="mps",
        schedule_mode="parallel",
    )
    report["candidate"].update(
        device="cpu",
        diarization_device="cuda",
        frame_device="cpu",
        schedule_policy="parallel",
        schedule_mode="serial",
        schedule_source="macos-vm-stat",
        frame_schedule_policy="parallel",
        frame_schedule_mode="serial",
        frame_schedule_source="macos-vm-stat",
    )
    report["runtime"].update(
        python="3.14.0",
        keyframe="0.5.2",
        mlx="0.1.0",
    )

    failures = benchmark.evaluate_report(
        report,
        _baseline(),
        critical_path_tolerance_seconds=0.01,
    )

    assert any("reference backend" in failure for failure in failures)
    assert any("candidate device" in failure for failure in failures)
    assert any("candidate diarization_device" in failure for failure in failures)
    assert any("candidate frame_device" in failure for failure in failures)
    assert any("candidate schedule_policy" in failure for failure in failures)
    assert any("candidate schedule_source" in failure for failure in failures)
    assert any("candidate frame_schedule_policy" in failure for failure in failures)
    assert any("candidate frame_schedule_source" in failure for failure in failures)
    assert any("runtime keyframe" in failure for failure in failures)
    assert any("runtime mlx" in failure for failure in failures)
    assert any("runtime Python" in failure for failure in failures)


@pytest.mark.parametrize(
    ("mutation", "expected_failure"),
    [
        (
            lambda candidate: candidate["pipeline_evidence"]["diarization"].update(
                started_at=1.0,
            ),
            "diarization must not start before transcription completes",
        ),
        (
            lambda candidate: candidate["pipeline_evidence"]["diarization"].update(
                ended_at=6.0,
            ),
            "frames must not start before diarization completes",
        ),
        (
            lambda candidate: candidate["pipeline_evidence"]["frames"].update(
                started_at=1.0,
                duration_seconds=5.0,
            ),
            "frames must not start before transcription completes",
        ),
    ],
)
def test_report_evaluation_rejects_required_candidate_serialization_failures(
    tmp_path,
    mutation,
    expected_failure,
):
    input_path = tmp_path / "recording.mp4"
    input_path.write_bytes(b"benchmark recording")
    report = _passing_report(input_path)
    candidate = report["candidate"]
    mutation(candidate)
    for stage, interval in candidate["pipeline_evidence"].items():
        interval["duration_seconds"] = (
            interval["ended_at"] - interval["started_at"]
        )
        candidate["timings"][stage] = interval["duration_seconds"]

    failures = benchmark.evaluate_report(
        report,
        _baseline(),
        critical_path_tolerance_seconds=5.0,
    )

    assert any(expected_failure in failure for failure in failures)


@pytest.mark.parametrize("tolerance", [float("nan"), float("inf"), float("-inf"), -1.0])
def test_report_evaluation_rejects_nonfinite_or_negative_tolerance(
    tmp_path,
    tolerance,
):
    input_path = tmp_path / "recording.mp4"
    input_path.write_bytes(b"benchmark recording")

    failures = benchmark.evaluate_report(
        _passing_report(input_path),
        _baseline(),
        critical_path_tolerance_seconds=tolerance,
    )

    assert any("critical-path tolerance is invalid" in failure for failure in failures)


@pytest.mark.parametrize(
    ("case_name", "field", "value"),
    [
        ("reference", "wall_time_seconds", float("nan")),
        ("candidate", "wall_time_seconds", float("inf")),
        ("candidate", "peak_memory_gib", -1.0),
        ("candidate", "peak_memory_gib", False),
        ("candidate", "model_resolution_seconds", "slow"),
        ("candidate", "model_resolution_seconds", False),
        ("candidate", "mlx_peak_memory_bytes", 1.5),
    ],
)
def test_report_evaluation_rejects_invalid_performance_measurements(
    tmp_path,
    case_name,
    field,
    value,
):
    input_path = tmp_path / "recording.mp4"
    input_path.write_bytes(b"benchmark recording")
    report = _passing_report(input_path)
    report[case_name][field] = value

    failures = benchmark.evaluate_report(
        report,
        _baseline(),
        critical_path_tolerance_seconds=5.0,
    )

    assert any("performance measurements are invalid" in item for item in failures)


def test_report_evaluation_returns_failures_for_malformed_schema(tmp_path):
    input_path = tmp_path / "recording.mp4"
    input_path.write_bytes(b"benchmark recording")
    report = _passing_report(input_path)
    report["schema_version"] = 99
    report["quality"]["character_agreement"] = "not-a-number"
    baseline = deepcopy(_baseline())
    del baseline["quality_thresholds"]["maximum_segment_count_relative_delta"]

    failures = benchmark.evaluate_report(
        report,
        baseline,
        critical_path_tolerance_seconds=0.01,
    )

    assert f"report schema_version must be {benchmark.REPORT_SCHEMA_VERSION}" in failures
    assert any("metrics or thresholds are invalid" in failure for failure in failures)


def test_replay_rejects_previous_report_schema_version(monkeypatch, tmp_path):
    input_path = tmp_path / "recording.mp4"
    input_path.write_bytes(b"benchmark recording")
    report = _passing_report(input_path)
    report["schema_version"] = benchmark.REPORT_SCHEMA_VERSION - 1
    replay_path = tmp_path / "replay-v1.json"
    replay_path.write_text(json.dumps(report), encoding="utf-8")
    output_path = tmp_path / "validated-report.json"
    monkeypatch.setattr(benchmark, "_probe_duration_seconds", lambda _path: 988.75)

    result = benchmark.main(
        [
            "--input",
            str(input_path),
            "--baseline",
            str(BASELINE_PATH),
            "--replay-report",
            str(replay_path),
            "--report",
            str(output_path),
        ]
    )

    assert result == 1
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["validation"] == {
        "passed": False,
        "failures": [
            f"report schema_version must be {benchmark.REPORT_SCHEMA_VERSION}"
        ],
    }


def test_replay_validates_explicit_recording_and_writes_result(
    monkeypatch,
    tmp_path,
):
    input_path = tmp_path / "recording.mp4"
    input_path.write_bytes(b"benchmark recording")
    replay_path = tmp_path / "replay.json"
    replay_path.write_text(
        json.dumps(_passing_report(input_path)),
        encoding="utf-8",
    )
    output_path = tmp_path / "validated-report.json"
    monkeypatch.setattr(benchmark, "_probe_duration_seconds", lambda _path: 988.75)

    result = benchmark.main(
        [
            "--input",
            str(input_path),
            "--baseline",
            str(BASELINE_PATH),
            "--replay-report",
            str(replay_path),
            "--report",
            str(output_path),
            "--critical-path-tolerance-seconds",
            "0.01",
        ]
    )

    assert result == 0
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["validation"] == {"passed": True, "failures": []}


def test_replay_rejects_report_for_a_different_recording(monkeypatch, tmp_path):
    input_path = tmp_path / "recording.mp4"
    input_path.write_bytes(b"benchmark recording")
    report = _passing_report(input_path)
    report["recording"]["sha256"] = "0" * 64
    replay_path = tmp_path / "replay.json"
    replay_path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(benchmark, "_probe_duration_seconds", lambda _path: 988.75)

    with pytest.raises(benchmark.BenchmarkError, match="recording contents"):
        benchmark.main(
            [
                "--input",
                str(input_path),
                "--baseline",
                str(BASELINE_PATH),
                "--replay-report",
                str(replay_path),
            ]
        )


@pytest.mark.parametrize("protected_name", ["input", "baseline", "replay"])
@pytest.mark.parametrize("alias_kind", ["lexical", "symlink", "hardlink"])
def test_report_rejects_protected_path_aliases_before_model_work(
    monkeypatch,
    tmp_path,
    protected_name,
    alias_kind,
):
    input_path = tmp_path / "recording.mp4"
    input_path.write_bytes(b"benchmark recording")
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(BASELINE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    replay_path = tmp_path / "replay.json"
    replay_path.write_text("{}", encoding="utf-8")
    protected_paths = {
        "input": input_path,
        "baseline": baseline_path,
        "replay": replay_path,
    }
    protected = protected_paths[protected_name]
    original = protected.read_bytes()

    if alias_kind == "lexical":
        report_path = protected
    else:
        report_path = tmp_path / f"report-{protected_name}-{alias_kind}.json"
        if alias_kind == "symlink":
            report_path.symlink_to(protected)
        else:
            os.link(protected, report_path)

    monkeypatch.setattr(
        benchmark,
        "_probe_duration_seconds",
        lambda _path: pytest.fail("alias validation must run before model work"),
    )

    with pytest.raises(benchmark.BenchmarkError, match="must not alias"):
        benchmark.main(
            [
                "--input",
                str(input_path),
                "--baseline",
                str(baseline_path),
                "--replay-report",
                str(replay_path),
                "--report",
                str(report_path),
            ]
        )

    assert protected.read_bytes() == original


def test_candidate_case_uses_and_verifies_automatic_apple_scheduling(
    monkeypatch,
    tmp_path,
):
    parsed_argv = []
    frame_config = object()
    frame_generation = object()
    frame_calls = []

    def parse_args(argv):
        parsed_argv.extend(argv)
        return SimpleNamespace(whisper_model="medium")

    preflight = SimpleNamespace(
        effective_backend="mlx",
        hf_token="hf_test",
        transcription_device="mlx",
        effective_diarization_device="mps",
        config=SimpleNamespace(transcription_backend="auto"),
    )
    schedule = SimpleNamespace(
        parallel=False,
        policy="auto",
        mode="serial",
        reason="stages share exclusive accelerator apple:0",
        resources=SimpleNamespace(source="macos-memory-pressure"),
    )
    pipeline_result = SimpleNamespace(
        transcript=SimpleNamespace(
            effective_backend="mlx",
            fallback_used=False,
            metadata={
                "model_repository": (
                    "mlx-community/whisper-medium-mlx"
                ),
                "model_revision": (
                    "7fc08c4eac4c316526498f147dfdee6f6303f975"
                ),
                "model_resolution_source": "local-hit",
                "model_resolution_seconds": 0.125,
                "mlx_peak_memory_bytes": 123456789,
            },
        ),
        frame_device="mps",
        diarization_attempted_devices=("mps",),
        diarization_fallback_used=False,
        initial_schedule=schedule,
        frame_schedule=schedule,
        critical_path="T + D + F + M + E",
        pipeline_evidence=SimpleNamespace(
            to_dict=lambda: {
                "transcription": _stage_interval(
                    "transcription", "initial", 0.0, 1.0
                ),
                "diarization": _stage_interval(
                    "diarization", "post-transcription", 1.0, 3.0
                ),
                "frames": _stage_interval(
                    "frames", "post-transcription", 3.0, 4.0
                ),
            }
        ),
        fallback_waited_for_diarization=False,
        timings={
            "transcription": 1.0,
            "diarization": 2.0,
            "frames": 1.0,
            "merge": 0.1,
            "manifest": 0.1,
        },
    )

    class FakeSupervisor:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def completed_stage_peak_rss_bytes(self):
            return {
                "transcription": 3 * benchmark.GIB,
                "diarization": 1 * benchmark.GIB,
            }

        def completed_stage_metadata(self, stage):
            assert stage == "diarization"
            return ({"pytorch_mps_fallback_enabled": False},)

    monkeypatch.setattr(benchmark.cli, "_parse_extract_args", parse_args)
    monkeypatch.setattr(benchmark.cli, "_transcript_config", lambda _args: object())
    monkeypatch.setattr(
        benchmark.cli,
        "_frame_config",
        lambda args, *, device: (
            frame_calls.append(("config", args, device)) or frame_config
        ),
    )
    monkeypatch.setattr(
        benchmark.cli,
        "_run_frame_generation",
        lambda video, output, config, supervisor: (
            frame_calls.append(
                ("run", video, output, config, supervisor)
            )
            or frame_generation
        ),
    )
    monkeypatch.setattr(benchmark, "preflight_transcript_run", lambda _config: preflight)
    monkeypatch.setattr(benchmark, "resolve_frame_device", lambda _preflight: "mps")
    monkeypatch.setattr(benchmark, "StageSupervisor", FakeSupervisor)

    def run_pipeline(*_args, **kwargs):
        assert kwargs["frame_runner"]() is frame_generation
        return pipeline_result

    monkeypatch.setattr(
        benchmark,
        "run_supervised_full_pipeline",
        run_pipeline,
    )
    monkeypatch.setattr(
        benchmark,
        "_artifact_summary",
        lambda _output: _artifacts(frames=True),
    )
    monotonic = iter((10.0, 12.0))
    monkeypatch.setattr(benchmark.time, "monotonic", lambda: next(monotonic))

    result = benchmark._run_candidate_case(
        benchmark._CaseRequest(
            "mlx_mps_serial_full",
            str(tmp_path / "input.mp4"),
            str(tmp_path / "output"),
        )
    )

    policy_index = parsed_argv.index("--stage-concurrency")
    assert parsed_argv[policy_index + 1] == "auto"
    assert result["requested_backend"] == "auto"
    assert result["schedule_policy"] == "auto"
    assert result["schedule_mode"] == "serial"
    assert result["schedule_source"] == "macos-memory-pressure"
    assert result["schedule_reason"] == benchmark.APPLE_ACCELERATOR_SERIAL_REASON
    assert result["frame_schedule_policy"] == "auto"
    assert result["frame_schedule_mode"] == "serial"
    assert result["frame_schedule_source"] == "macos-memory-pressure"
    assert result["frame_schedule_reason"] == (
        benchmark.APPLE_ACCELERATOR_SERIAL_REASON
    )
    assert result["model_resolution_source"] == "local-hit"
    assert result["model_resolution_seconds"] == pytest.approx(0.125)
    assert result["mlx_peak_memory_bytes"] == 123456789
    assert result["stage_process_tree_peak_rss_bytes"] == {
        "transcription": 3 * benchmark.GIB,
        "diarization": 1 * benchmark.GIB,
    }
    assert result["fallback_used"] is False
    assert result["diarization_attempted_devices"] == ["mps"]
    assert result["diarization_fallback_used"] is False
    assert result["pytorch_mps_fallback_enabled"] is False
    assert result["critical_path"] == "T + D + F + M + E"
    assert result["fallback_waited_for_diarization"] is False
    assert result["pipeline_evidence"]["frames"]["launch_wave"] == (
        "post-transcription"
    )
    assert frame_calls[0][0] == "config"
    assert frame_calls[0][2] == "mps"
    run_call = frame_calls[1]
    assert run_call[0] == "run"
    assert run_call[1] == tmp_path / "input.mp4"
    assert run_call[2] == tmp_path / "output"
    assert run_call[3] is frame_config
    assert isinstance(run_call[4], FakeSupervisor)


def test_candidate_case_rejects_an_overlapping_apple_result(monkeypatch, tmp_path):
    preflight = SimpleNamespace(
        effective_backend="mlx",
        hf_token="hf_test",
        transcription_device="mlx",
        effective_diarization_device="mps",
        config=SimpleNamespace(transcription_backend="auto"),
    )
    result = SimpleNamespace(
        initial_schedule=SimpleNamespace(parallel=True),
    )

    class FakeSupervisor:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        benchmark.cli,
        "_parse_extract_args",
        lambda _argv: SimpleNamespace(whisper_model="medium"),
    )
    monkeypatch.setattr(benchmark.cli, "_transcript_config", lambda _args: object())
    monkeypatch.setattr(benchmark, "preflight_transcript_run", lambda _config: preflight)
    monkeypatch.setattr(benchmark, "resolve_frame_device", lambda _preflight: "mps")
    monkeypatch.setattr(benchmark, "StageSupervisor", FakeSupervisor)
    monkeypatch.setattr(
        benchmark,
        "run_supervised_full_pipeline",
        lambda *_args, **_kwargs: result,
    )

    with pytest.raises(benchmark.BenchmarkError, match="overlapped"):
        benchmark._run_candidate_case(
            benchmark._CaseRequest(
                "mlx_mps_serial_full",
                str(tmp_path / "input.mp4"),
                str(tmp_path / "output"),
            )
        )


def test_case_worker_persists_case_metadata_before_reporting_success(
    monkeypatch,
    tmp_path,
):
    expected = {"name": "whisper_cpu_serial", "wall_time_seconds": 1.5}
    sent = []

    class FakeSender:
        def send(self, value):
            sent.append(value)

        def close(self):
            sent.append("closed")

    monkeypatch.setattr(benchmark, "_run_reference_case", lambda _request: expected)
    request = benchmark._CaseRequest(
        "whisper_cpu_serial",
        str(tmp_path / "input.mp4"),
        str(tmp_path),
    )

    benchmark._case_worker(request, FakeSender())

    assert json.loads(
        (tmp_path / "benchmark-case.json").read_text(encoding="utf-8")
    ) == expected
    assert sent == [{"status": "success", "result": expected}, "closed"]
