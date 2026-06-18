import json
from pathlib import Path

import pytest

from keyframe.diarization import (
    CanonicalRecording,
    ValidationError,
    canonical_json_dumps,
    canonical_json_loads,
    canonical_jsonl_dumps,
    canonical_jsonl_loads,
    read_recording_json,
    read_recordings_jsonl,
    write_recording_json,
    write_recordings_jsonl,
)


FIXTURE_DIR = Path(__file__).with_name("fixtures")
EXPECTED_FIXTURES = {
    "clean_two_speaker.json",
    "degraded_output.json",
    "missing_confidence.json",
    "missing_speaker.json",
    "overlap.json",
}


def _fixture_paths() -> list[Path]:
    return sorted(FIXTURE_DIR.glob("*.json"))


def test_fixture_set_covers_required_canonical_cases():
    assert {path.name for path in _fixture_paths()} == EXPECTED_FIXTURES


@pytest.mark.parametrize("fixture_path", _fixture_paths(), ids=lambda path: path.name)
def test_every_fixture_loads_and_rewrites_byte_stable_json(fixture_path):
    recording = read_recording_json(fixture_path)

    assert isinstance(recording, CanonicalRecording)
    assert canonical_json_dumps(recording) == fixture_path.read_text(encoding="utf-8")


def test_json_file_round_trip_is_stable(tmp_path):
    source = FIXTURE_DIR / "clean_two_speaker.json"
    recording = read_recording_json(source)
    target = tmp_path / "recording.json"

    write_recording_json(target, recording)

    assert target.read_bytes() == source.read_bytes()
    assert b"\r\n" not in target.read_bytes()
    assert read_recording_json(target).to_dict() == recording.to_dict()


def test_jsonl_round_trip_uses_one_stable_recording_per_line(tmp_path):
    recordings = (
        read_recording_json(FIXTURE_DIR / "clean_two_speaker.json"),
        read_recording_json(FIXTURE_DIR / "overlap.json"),
    )
    target = tmp_path / "recordings.jsonl"

    write_recordings_jsonl(target, recordings)

    text = target.read_text(encoding="utf-8")
    assert text == canonical_jsonl_dumps(recordings)
    assert b"\r\n" not in target.read_bytes()
    assert len(text.splitlines()) == 2
    assert [recording.to_dict() for recording in read_recordings_jsonl(target)] == [
        recording.to_dict() for recording in recordings
    ]
    assert [recording.to_dict() for recording in canonical_jsonl_loads(text)] == [
        recording.to_dict() for recording in recordings
    ]


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda payload: payload.pop("schema_version"), "recording.schema_version is required"),
        (lambda payload: payload.update({"schema_version": 2}), "schema_version is not supported"),
        (lambda payload: payload.update({"schema_version": 1.0}), "schema_version must be an integer"),
    ],
)
def test_schema_version_checks_fail_closed(mutate, message):
    payload = json.loads((FIXTURE_DIR / "clean_two_speaker.json").read_text(encoding="utf-8"))
    mutate(payload)

    with pytest.raises(ValidationError, match=message):
        canonical_json_loads(json.dumps(payload))


@pytest.mark.parametrize("field_name", ["channels", "speakers", "words", "speaker_spans", "scoring_regions"])
def test_required_canonical_collection_fields_fail_closed_when_missing(field_name):
    payload = json.loads((FIXTURE_DIR / "clean_two_speaker.json").read_text(encoding="utf-8"))
    payload.pop(field_name)

    with pytest.raises(ValidationError, match=rf"recording\.{field_name} is required"):
        canonical_json_loads(json.dumps(payload))


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda payload: payload.update({"future_field": "not-current-schema"}), "unsupported fields"),
        (lambda payload: payload["words"][0].update({"token_confidence": 0.7}), "word has unsupported fields"),
    ],
)
def test_unknown_artifact_fields_fail_closed(mutate, message):
    payload = json.loads((FIXTURE_DIR / "clean_two_speaker.json").read_text(encoding="utf-8"))
    mutate(payload)

    with pytest.raises(ValidationError, match=message):
        canonical_json_loads(json.dumps(payload))


def test_invalid_json_and_jsonl_fail_with_validation_error():
    with pytest.raises(ValidationError, match="canonical JSON is invalid"):
        canonical_json_loads("{")

    with pytest.raises(ValidationError, match="canonical JSONL line 2 is empty"):
        recording = read_recording_json(FIXTURE_DIR / "clean_two_speaker.json")
        canonical_jsonl_loads(canonical_jsonl_dumps((recording,)) + "\n")
