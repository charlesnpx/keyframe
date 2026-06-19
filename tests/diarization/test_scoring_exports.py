from pathlib import Path

import pytest

from keyframe.diarization import (
    CanonicalRecording,
    ChannelRecord,
    ScoringRegion,
    SpeakerRecord,
    SpeakerSpan,
    ValidationError,
    read_recording_json,
    recording_to_rttm_text,
    recording_to_uem_text,
    rttm_oracle_self_score,
    score_rttm_pair,
    speaker_spans_to_rttm_text,
    validate_rttm_text,
    validate_uem_text,
    write_rttm,
    write_uem,
)


FIXTURE_DIR = Path("tests/diarization/fixtures")


def _recording(name="clean_two_speaker.json"):
    return read_recording_json(FIXTURE_DIR / name)


def test_canonical_speaker_spans_export_to_valid_rttm_rows():
    recording = _recording()

    rttm = recording_to_rttm_text(recording)

    assert rttm == (
        "SPEAKER fixture-clean-two-speaker ch-1 0.000 0.350 <NA> <NA> spk-a <NA> <NA>\n"
        "SPEAKER fixture-clean-two-speaker ch-1 0.450 0.400 <NA> <NA> spk-b <NA> <NA>\n"
    )
    assert [row.to_dict() for row in validate_rttm_text(rttm)] == [
        {
            "channel_id": "ch-1",
            "duration_ms": 350,
            "end_ms": 350,
            "recording_id": "fixture-clean-two-speaker",
            "speaker_ref": "spk-a",
            "start_ms": 0,
        },
        {
            "channel_id": "ch-1",
            "duration_ms": 400,
            "end_ms": 850,
            "recording_id": "fixture-clean-two-speaker",
            "speaker_ref": "spk-b",
            "start_ms": 450,
        },
    ]


def test_canonical_scoring_regions_export_to_valid_uem_rows():
    uem = recording_to_uem_text(_recording())

    assert uem == "fixture-clean-two-speaker ch-1 0.000 1.000\n"
    assert [row.to_dict() for row in validate_uem_text(uem)] == [
        {
            "channel_id": "ch-1",
            "end_ms": 1000,
            "recording_id": "fixture-clean-two-speaker",
            "start_ms": 0,
        }
    ]


def test_rttm_export_preserves_overlapping_speaker_spans_as_separate_rows():
    rttm = recording_to_rttm_text(_recording("overlap.json"))

    assert rttm == (
        "SPEAKER fixture-overlap ch-1 0.100 0.350 <NA> <NA> spk-a <NA> <NA>\n"
        "SPEAKER fixture-overlap ch-1 0.250 0.350 <NA> <NA> spk-b <NA> <NA>\n"
    )
    rows = validate_rttm_text(rttm)
    assert rows[0].end_ms > rows[1].start_ms


def test_speaker_span_export_supports_hypothesis_views_without_canonical_rewrite():
    recording = _recording()

    rttm = speaker_spans_to_rttm_text("candidate-hypothesis", recording.speaker_spans)

    assert rttm.startswith("SPEAKER candidate-hypothesis ch-1")
    assert recording.words[0].text == "hello"
    assert recording.words[0].text_confidence == 0.99
    assert recording.speakers[0].display_label.label == "person_1"


@pytest.mark.parametrize("fixture_name", ["clean_two_speaker.json", "overlap.json"])
def test_oracle_reference_vs_reference_self_score_passes_strict_threshold(fixture_name):
    result = rttm_oracle_self_score(_recording(fixture_name), threshold=0.999999)

    assert result.passed is True
    assert result.score == 1.0
    assert result.false_alarm_ms == 0
    assert result.missed_speech_ms == 0


def test_score_rttm_pair_fails_below_threshold_for_degraded_hypothesis():
    recording = _recording()
    reference = recording_to_rttm_text(recording)
    degraded = reference.replace("0.450 0.400", "0.500 0.300")

    result = score_rttm_pair(reference, degraded, recording_to_uem_text(recording), threshold=0.999999)

    assert result.passed is False
    assert result.score < 1.0
    assert result.missed_speech_ms > 0


def test_score_rttm_pair_clips_rows_to_uem_instead_of_rejecting_crossing_rows():
    reference = "SPEAKER rec ch-1 0.000 1.000 <NA> <NA> spk-a <NA> <NA>\n"
    uem = "rec ch-1 0.250 0.750\n"

    result = score_rttm_pair(reference, reference, uem, threshold=0.999999)

    assert result.passed is True
    assert result.score == 1.0
    assert result.reference_speech_ms == 500
    assert result.hypothesis_speech_ms == 500
    assert result.matched_speech_ms == 500


def test_writers_create_rttm_and_uem_artifacts(tmp_path):
    recording = _recording()
    rttm_path = tmp_path / "exports" / "rttm" / "fixture.rttm"
    uem_path = tmp_path / "exports" / "uem" / "fixture.uem"

    write_rttm(rttm_path, recording)
    write_uem(uem_path, recording)

    assert validate_rttm_text(rttm_path.read_text(encoding="utf-8"))
    assert validate_uem_text(uem_path.read_text(encoding="utf-8"))


def test_rttm_export_rejects_missing_spans_and_non_token_speaker_refs():
    empty = CanonicalRecording(
        recording_id="empty",
        original_audio_id="empty-original",
        canonical_audio_id="empty-canonical",
        timeline_id="empty-timeline",
        duration_ms=1000,
        channels=(ChannelRecord("ch-1"),),
        scoring_regions=(ScoringRegion("uem-1", 0, 1000, channel_id="ch-1"),),
    )
    invalid_speaker = CanonicalRecording(
        recording_id="invalid-speaker",
        original_audio_id="invalid-original",
        canonical_audio_id="invalid-canonical",
        timeline_id="invalid-timeline",
        duration_ms=1000,
        channels=(ChannelRecord("ch-1"),),
        speakers=(SpeakerRecord("bad speaker"),),
        speaker_spans=(SpeakerSpan("span-1", "bad speaker", 0, 500, channel_id="ch-1"),),
        scoring_regions=(ScoringRegion("uem-1", 0, 1000, channel_id="ch-1"),),
    )

    with pytest.raises(ValidationError, match="RTTM export requires"):
        recording_to_rttm_text(empty)
    with pytest.raises(ValidationError, match="speaker_ref must be a single token"):
        recording_to_rttm_text(invalid_speaker)


def test_validators_reject_invalid_scoring_contract_rows():
    with pytest.raises(ValidationError, match="RTTM line 1 duration must be greater than 0"):
        validate_rttm_text("SPEAKER rec ch-1 0.000 0.000 <NA> <NA> spk-a <NA> <NA>\n")
    with pytest.raises(ValidationError, match="uem.end_ms must be greater"):
        validate_uem_text("rec ch-1 1.000 1.000\n")
