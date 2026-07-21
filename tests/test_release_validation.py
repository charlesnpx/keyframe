from __future__ import annotations

import hashlib
import json
import os
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


ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "tests/fixtures/transcription-benchmark-baseline.json"


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
        "reference": {"artifacts": _artifacts(frames=False)},
        "candidate": {
            "backend": "mlx",
            "model_repository": baseline["model"]["mlx_repository"],
            "model_revision": baseline["model"]["mlx_revision"],
            "critical_path": "max(T + F, D) + M + E",
            "wall_time_seconds": 7.0,
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
    }


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
        ("max(T + F, D) + M + E", 23.0),
        ("max(T, D) + F + M + E", 26.0),
        ("T + D + F + M + E", 36.0),
    ],
)
def test_expected_critical_path_supports_each_release_schedule(expression, expected):
    timings = {
        "transcription": 10.0,
        "diarization": 20.0,
        "frames": 3.0,
        "merge": 2.0,
        "manifest": 1.0,
    }

    assert expected_critical_path_seconds(expression, timings) == expected


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
        "expression": "max(T + F, D) + M + E",
        "predicted_seconds": 7.0,
        "measured_wall_seconds": 7.0,
        "absolute_delta_seconds": 0.0,
        "tolerance_seconds": 0.01,
    }


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

    assert "report schema_version must be 1" in failures
    assert any("metrics or thresholds are invalid" in failure for failure in failures)


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


def test_candidate_case_forces_and_verifies_the_parallel_apple_schedule(
    monkeypatch,
    tmp_path,
):
    parsed_argv = []

    def parse_args(argv):
        parsed_argv.extend(argv)
        return SimpleNamespace(whisper_model="medium")

    preflight = SimpleNamespace(
        effective_backend="mlx",
        hf_token="hf_test",
        transcription_device="mlx",
        effective_diarization_device="cpu",
    )
    schedule = SimpleNamespace(
        parallel=True,
        mode="parallel",
        reason="explicit parallel override",
    )
    pipeline_result = SimpleNamespace(
        transcript=SimpleNamespace(effective_backend="mlx"),
        frame_device="mps",
        initial_schedule=schedule,
        frame_schedule=schedule,
        critical_path="max(T + F, D) + M + E",
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

    monkeypatch.setattr(benchmark.cli, "_parse_extract_args", parse_args)
    monkeypatch.setattr(benchmark.cli, "_transcript_config", lambda _args: object())
    monkeypatch.setattr(benchmark, "preflight_transcript_run", lambda _config: preflight)
    monkeypatch.setattr(benchmark, "resolve_frame_device", lambda _preflight: "mps")
    monkeypatch.setattr(benchmark, "StageSupervisor", FakeSupervisor)
    monkeypatch.setattr(
        benchmark,
        "run_supervised_full_pipeline",
        lambda *_args, **_kwargs: pipeline_result,
    )
    monkeypatch.setattr(benchmark, "_maximum_resident_set_gib", lambda: 5.0)
    monkeypatch.setattr(
        benchmark,
        "_artifact_summary",
        lambda _output: _artifacts(frames=True),
    )
    monotonic = iter((10.0, 12.0))
    monkeypatch.setattr(benchmark.time, "monotonic", lambda: next(monotonic))

    result = benchmark._run_candidate_case(
        benchmark._CaseRequest(
            "mlx_concurrent_full",
            str(tmp_path / "input.mp4"),
            str(tmp_path / "output"),
        )
    )

    policy_index = parsed_argv.index("--stage-concurrency")
    assert parsed_argv[policy_index + 1] == "parallel"
    assert result["schedule_mode"] == "parallel"
    assert result["frame_schedule_mode"] == "parallel"
    assert result["critical_path"] == "max(T + F, D) + M + E"


def test_candidate_case_rejects_a_nonparallel_result(monkeypatch, tmp_path):
    preflight = SimpleNamespace(
        effective_backend="mlx",
        hf_token="hf_test",
        transcription_device="mlx",
        effective_diarization_device="cpu",
    )
    result = SimpleNamespace(
        initial_schedule=SimpleNamespace(parallel=False),
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

    with pytest.raises(benchmark.BenchmarkError, match="did not overlap"):
        benchmark._run_candidate_case(
            benchmark._CaseRequest(
                "mlx_concurrent_full",
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
