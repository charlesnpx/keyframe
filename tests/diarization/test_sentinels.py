from dataclasses import replace
from pathlib import Path

import pytest

from keyframe.diarization import (
    SENTINEL_BASELINE_IDS,
    ReferenceBundle,
    SentinelBaselineReport,
    ScoringRegion,
    SpeakerRecord,
    SpeakerSpan,
    ValidationError,
    build_sentinel_baseline_outputs,
    evaluate_sentinel_baselines,
    read_recording_json,
    require_passing_sentinel_baselines,
)


FIXTURE_DIR = Path("tests/diarization/fixtures")


def _reference_bundle(name="clean_two_speaker.json"):
    recording = read_recording_json(FIXTURE_DIR / name)
    return ReferenceBundle.from_recording(recording, artifact_id="sentinel-reference")


def _checks_by_id(report):
    return {check.baseline_id: check for check in report.checks}


def test_sentinel_outputs_are_labeled_as_fixture_baselines_not_candidate_engines():
    reference = _reference_bundle()

    outputs = build_sentinel_baseline_outputs(reference)

    assert [output.config.config_id for output in outputs] == list(SENTINEL_BASELINE_IDS)
    for output in outputs:
        assert output.output_id.startswith(f"sentinel-baseline:{reference.recording.recording_id}:")
        assert output.artifact.artifact_kind == "fixture"
        assert output.config.adapter_id == "keyframe-sentinel-baseline"
        assert output.config.parameters["sentinel"] is True
        assert output.config.parameters["baseline_id"] in SENTINEL_BASELINE_IDS


def test_sentinel_baseline_report_passes_on_clean_two_speaker_fixture():
    report = evaluate_sentinel_baselines(_reference_bundle())
    checks = _checks_by_id(report)

    assert report.status == "passed"
    assert report.passed is True
    assert [check.baseline_id for check in report.checks] == list(SENTINEL_BASELINE_IDS)
    assert all(check.passed for check in report.checks)
    assert checks["oracle"].policy_id == "sentinel-diagnostic-diarization-v1"
    require_passing_sentinel_baselines(report)


@pytest.mark.parametrize(
    ("baseline_id", "metric", "expected"),
    [
        ("oracle", "diarization_error_rate", 0.0),
        ("single_speaker_collapse", "diarization_error_rate", "degraded"),
        ("channel_only", "diarization_error_rate", "degraded"),
        ("timestamp_shifted", "diarization_error_rate", "degraded"),
        ("shuffled_speakers", "diarization_error_rate", "degraded"),
        ("perfect_text_wrong_speaker", "word_speaker_label_accuracy", "below_perfect"),
        ("bad_text_perfect_speaker", "sentinel_text_mismatch_rate", "degraded"),
        ("bad_turn_builder", "turn_speaker_label_accuracy", "below_perfect"),
    ],
)
def test_each_sentinel_baseline_has_expected_metric_direction(baseline_id, metric, expected):
    report = evaluate_sentinel_baselines(_reference_bundle())
    check = _checks_by_id(report)[baseline_id]
    value = check.metrics[metric]

    if expected == "degraded":
        assert value > 0.0
    elif expected == "below_perfect":
        assert value < 1.0
    else:
        assert value == expected


def test_bad_text_baseline_separates_asr_failure_from_speaker_attribution():
    report = evaluate_sentinel_baselines(_reference_bundle())
    check = _checks_by_id(report)["bad_text_perfect_speaker"]

    assert check.metrics["sentinel_text_mismatch_rate"] == 1.0
    assert check.metrics["word_speaker_label_accuracy"] == 1.0
    assert check.metrics["turn_speaker_label_accuracy"] == 1.0
    assert check.metrics["speaker_count_error"] == 0


def test_shuffled_speaker_baseline_degrades_word_only_references():
    reference = _reference_bundle()
    recording = replace(
        reference.recording,
        speaker_spans=(),
    )

    report = evaluate_sentinel_baselines(ReferenceBundle.from_recording(recording, artifact_id="word-only-reference"))
    check = _checks_by_id(report)["shuffled_speakers"]

    assert check.status == "passed"
    assert check.metrics["diarization_error_rate"] > 0.0


def test_timestamp_shift_baseline_degrades_end_aligned_intervals():
    reference = _reference_bundle()
    recording = replace(
        reference.recording,
        duration_ms=1_000,
        speakers=(SpeakerRecord("spk-a"),),
        speaker_spans=(
            SpeakerSpan(
                span_id="span-end-aligned",
                speaker_ref="spk-a",
                start_ms=500,
                end_ms=1_000,
                channel_id="ch-1",
            ),
        ),
        words=(),
        scoring_regions=(ScoringRegion("uem-1", 0, 1_000, channel_id="ch-1"),),
    )

    report = evaluate_sentinel_baselines(ReferenceBundle.from_recording(recording, artifact_id="end-aligned-reference"))
    check = _checks_by_id(report)["timestamp_shifted"]

    assert check.status == "passed"
    assert check.metrics["diarization_error_rate"] > 0.0


def test_sentinel_report_skips_baselines_that_single_speaker_references_cannot_exercise():
    reference = _reference_bundle()
    recording = replace(
        reference.recording,
        duration_ms=1_000,
        speakers=(SpeakerRecord("spk-a"),),
        speaker_spans=(
            SpeakerSpan(
                span_id="span-single",
                speaker_ref="spk-a",
                start_ms=500,
                end_ms=1_000,
                channel_id="ch-1",
            ),
        ),
        words=(),
        scoring_regions=(ScoringRegion("uem-1", 0, 1_000, channel_id="ch-1"),),
    )

    report = evaluate_sentinel_baselines(ReferenceBundle.from_recording(recording, artifact_id="single-speaker-reference"))

    assert report.status == "passed"
    assert [check.baseline_id for check in report.checks] == ["oracle", "timestamp_shifted"]
    require_passing_sentinel_baselines(report)


def test_bad_turn_builder_preserves_word_attribution_while_degrading_turns():
    report = evaluate_sentinel_baselines(_reference_bundle())
    check = _checks_by_id(report)["bad_turn_builder"]

    assert check.metrics["word_speaker_label_accuracy"] == 1.0
    assert check.metrics["turn_speaker_label_accuracy"] < 1.0
    assert check.metrics["speaker_count_error"] == 0


def test_sentinel_gate_blocks_engine_benchmark_execution_when_health_fails():
    report = evaluate_sentinel_baselines(_reference_bundle())
    failed_check = replace(
        report.checks[0],
        status="failed",
        failures=("forced sentinel failure",),
    )
    failed_report = SentinelBaselineReport(
        status="failed",
        checks=(failed_check, *report.checks[1:]),
    )

    with pytest.raises(ValidationError, match="refusing engine benchmark execution"):
        require_passing_sentinel_baselines(failed_report)
