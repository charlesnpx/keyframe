import json
import sys

import pytest

from keyframe import cli
from keyframe.diarization import (
    CanonicalRecording,
    CanonicalWord,
    ChannelRecord,
    DisplayLabel,
    ScoringRegion,
    SpeakerRecord,
    SpeakerSpan,
    ValidationError,
)


def _recording(**overrides):
    base = {
        "recording_id": "rec-1",
        "original_audio_id": "original-audio-local-fixture",
        "canonical_audio_id": "canonical-audio-local-fixture",
        "timeline_id": "timeline-1",
        "duration_ms": 4_000,
        "channels": (
            ChannelRecord("ch-1", name="left"),
            ChannelRecord("ch-2", name="right"),
        ),
        "speakers": (
            SpeakerRecord(
                "spk-a",
                DisplayLabel(
                    label="person_1",
                    source="diarization_cluster",
                    confidence=0.81,
                    source_ref="spk-a",
                ),
            ),
            SpeakerRecord(
                "spk-b",
                DisplayLabel(label="person_2", source="diarization_cluster", source_ref="spk-b"),
            ),
        ),
        "words": (
            CanonicalWord(
                word_id="w-1",
                text="hello",
                start_ms=0,
                end_ms=500,
                speaker_ref="spk-a",
                channel_id="ch-1",
                text_confidence=0.9,
                speaker_confidence=0.8,
                display_label=DisplayLabel(label="person_1", source="diarization_cluster"),
            ),
            CanonicalWord(
                word_id="w-2",
                text="there",
                start_ms=450,
                end_ms=800,
                speaker_ref="spk-b",
                channel_id="ch-2",
                speaker_confidence=None,
                overlap=True,
                display_label=DisplayLabel(label="person_2", source="diarization_cluster"),
            ),
        ),
        "speaker_spans": (
            SpeakerSpan(
                span_id="span-1",
                speaker_ref="spk-a",
                start_ms=0,
                end_ms=500,
                channel_id="ch-1",
                confidence=0.8,
            ),
            SpeakerSpan(
                span_id="span-2",
                speaker_ref="spk-b",
                start_ms=450,
                end_ms=800,
                channel_id="ch-2",
                overlap=True,
            ),
        ),
        "scoring_regions": (
            ScoringRegion(region_id="uem-1", start_ms=0, end_ms=1_000, channel_id="ch-1"),
        ),
    }
    base.update(overrides)
    return CanonicalRecording(**base)


def test_valid_multichannel_recording_preserves_overlap_and_nullable_confidence():
    recording = _recording()

    payload = recording.to_dict()

    assert payload["schema_version"] == 1
    assert payload["words"][1]["overlap"] is True
    assert payload["words"][1]["speaker_confidence"] is None
    assert payload["speakers"][0]["display_label"]["label"] == "person_1"
    assert payload["speakers"][0]["display_label"]["scope"] == "recording"


def test_serialization_is_deterministic_and_contains_no_persistent_identity_fields():
    payload = _recording().to_dict()

    first = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    second = json.dumps(payload, sort_keys=True, separators=(",", ":"))

    assert first == second
    assert "embedding" not in first
    assert "voice_profile" not in first
    assert "cross_call" not in first


@pytest.mark.parametrize(
    "make_overrides, message",
    [
        (lambda: {"original_audio_id": ""}, "original_audio_id is required"),
        (lambda: {"duration_ms": 0}, "duration_ms must be greater than 0"),
        (lambda: {"words": (CanonicalWord("w-bad", "oops", 10, 10),)}, "end_ms must be greater"),
        (
            lambda: {"words": (CanonicalWord("w-bad", "oops", 0, 1, speaker_ref="missing"),)},
            "unknown speaker_ref",
        ),
        (
            lambda: {"words": (CanonicalWord("w-bad", "oops", 0, 1, channel_id="missing"),)},
            "unknown channel_id",
        ),
        (
            lambda: {"words": (CanonicalWord("w-bad", "oops", 0, 5_000),)},
            "ends after recording duration",
        ),
    ],
)
def test_recording_validation_rejects_invalid_or_unresolved_references(make_overrides, message):
    with pytest.raises(ValidationError, match=message):
        _recording(**make_overrides())


def test_display_label_scope_is_recording_only():
    with pytest.raises(ValidationError, match="scoped to one recording"):
        DisplayLabel(label="person_1", source="diarization_cluster", scope="global")


def test_confidence_values_are_bounded():
    with pytest.raises(ValidationError, match="must be between"):
        CanonicalWord("w-1", "hello", 0, 1, text_confidence=1.1)


def test_existing_cli_video_dispatch_still_uses_extract_mode(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["keyframe", "missing-input.mp4"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
    assert "file not found: missing-input.mp4" in capsys.readouterr().err
