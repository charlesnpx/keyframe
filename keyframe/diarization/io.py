"""Stable JSON and JSONL IO for canonical diarization artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from keyframe.diarization.models import (
    SCHEMA_VERSION,
    CanonicalRecording,
    CanonicalWord,
    ChannelRecord,
    DisplayLabel,
    ScoringRegion,
    SpeakerRecord,
    SpeakerSpan,
    ValidationError,
)


_RECORDING_FIELDS = frozenset(
    {
        "schema_version",
        "recording_id",
        "original_audio_id",
        "canonical_audio_id",
        "timeline_id",
        "duration_ms",
        "channels",
        "speakers",
        "words",
        "speaker_spans",
        "scoring_regions",
    }
)
_DISPLAY_LABEL_FIELDS = frozenset({"label", "source", "scope", "confidence", "source_ref"})
_CHANNEL_FIELDS = frozenset({"channel_id", "name"})
_SPEAKER_FIELDS = frozenset({"speaker_ref", "display_label"})
_WORD_FIELDS = frozenset(
    {
        "word_id",
        "text",
        "start_ms",
        "end_ms",
        "speaker_ref",
        "channel_id",
        "text_confidence",
        "speaker_confidence",
        "overlap",
        "display_label",
    }
)
_SPAN_FIELDS = frozenset({"span_id", "speaker_ref", "start_ms", "end_ms", "channel_id", "confidence", "overlap"})
_SCORING_REGION_FIELDS = frozenset({"region_id", "start_ms", "end_ms", "channel_id"})


def recording_to_dict(recording: CanonicalRecording) -> dict[str, Any]:
    """Return the canonical dictionary form for one recording."""

    if not isinstance(recording, CanonicalRecording):
        raise ValidationError("recording must be a CanonicalRecording")
    return recording.to_dict()


def recording_from_dict(payload: Mapping[str, Any]) -> CanonicalRecording:
    """Build a canonical recording from a strict schema-versioned mapping."""

    data = _require_mapping(payload, "recording")
    _reject_unknown_fields(data, _RECORDING_FIELDS, "recording")
    validate_schema_version(data, context="recording")
    return CanonicalRecording(
        recording_id=_required(data, "recording_id", "recording"),
        original_audio_id=_required(data, "original_audio_id", "recording"),
        canonical_audio_id=_required(data, "canonical_audio_id", "recording"),
        timeline_id=_required(data, "timeline_id", "recording"),
        duration_ms=_required(data, "duration_ms", "recording"),
        channels=_decode_sequence(_required(data, "channels", "recording"), _channel_from_dict, "recording.channels"),
        speakers=_decode_sequence(_required(data, "speakers", "recording"), _speaker_from_dict, "recording.speakers"),
        words=_decode_sequence(_required(data, "words", "recording"), _word_from_dict, "recording.words"),
        speaker_spans=_decode_sequence(
            _required(data, "speaker_spans", "recording"),
            _speaker_span_from_dict,
            "recording.speaker_spans",
        ),
        scoring_regions=_decode_sequence(
            _required(data, "scoring_regions", "recording"),
            _scoring_region_from_dict,
            "recording.scoring_regions",
        ),
    )


def validate_schema_version(payload: Mapping[str, Any], *, context: str) -> int:
    """Return the supported schema version or fail closed."""

    version = _required(payload, "schema_version", context)
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValidationError(f"{context}.schema_version must be an integer")
    if version != SCHEMA_VERSION:
        raise ValidationError(f"{context}.schema_version is not supported: {version}")
    return version


def canonical_json_dumps(recording: CanonicalRecording) -> str:
    """Serialize a recording with byte-stable formatting."""

    return json.dumps(
        recording_to_dict(recording),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def canonical_json_loads(text: str) -> CanonicalRecording:
    """Deserialize one canonical recording from JSON text."""

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"canonical JSON is invalid: {exc.msg}") from exc
    return recording_from_dict(payload)


def write_recording_json(path: str | Path, recording: CanonicalRecording) -> None:
    """Write one canonical recording JSON artifact."""

    Path(path).write_text(canonical_json_dumps(recording), encoding="utf-8", newline="\n")


def read_recording_json(path: str | Path) -> CanonicalRecording:
    """Read one canonical recording JSON artifact."""

    return canonical_json_loads(Path(path).read_text(encoding="utf-8"))


def canonical_jsonl_dumps(recordings: Iterable[CanonicalRecording]) -> str:
    """Serialize recordings as byte-stable JSONL, one compact JSON object per line."""

    lines = [
        json.dumps(
            recording_to_dict(recording),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        for recording in recordings
    ]
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def canonical_jsonl_loads(text: str) -> tuple[CanonicalRecording, ...]:
    """Deserialize canonical recordings from JSONL text."""

    recordings: list[CanonicalRecording] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ValidationError(f"canonical JSONL line {line_number} is empty")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"canonical JSONL line {line_number} is invalid: {exc.msg}") from exc
        try:
            recordings.append(recording_from_dict(payload))
        except ValidationError as exc:
            raise ValidationError(f"canonical JSONL line {line_number}: {exc}") from exc
    return tuple(recordings)


def write_recordings_jsonl(path: str | Path, recordings: Iterable[CanonicalRecording]) -> None:
    """Write canonical recordings to a JSONL artifact."""

    Path(path).write_text(canonical_jsonl_dumps(recordings), encoding="utf-8", newline="\n")


def read_recordings_jsonl(path: str | Path) -> tuple[CanonicalRecording, ...]:
    """Read canonical recordings from a JSONL artifact."""

    return canonical_jsonl_loads(Path(path).read_text(encoding="utf-8"))


def _display_label_from_dict(payload: object, context: str) -> DisplayLabel:
    data = _require_mapping(payload, context)
    _reject_unknown_fields(data, _DISPLAY_LABEL_FIELDS, context)
    return DisplayLabel(
        label=_required(data, "label", context),
        source=_required(data, "source", context),
        scope=data.get("scope", "recording"),
        confidence=data.get("confidence"),
        source_ref=data.get("source_ref"),
    )


def _optional_display_label_from_dict(payload: object, context: str) -> DisplayLabel | None:
    if payload is None:
        return None
    return _display_label_from_dict(payload, context)


def _channel_from_dict(payload: object) -> ChannelRecord:
    data = _require_mapping(payload, "channel")
    _reject_unknown_fields(data, _CHANNEL_FIELDS, "channel")
    return ChannelRecord(channel_id=_required(data, "channel_id", "channel"), name=data.get("name"))


def _speaker_from_dict(payload: object) -> SpeakerRecord:
    data = _require_mapping(payload, "speaker")
    _reject_unknown_fields(data, _SPEAKER_FIELDS, "speaker")
    return SpeakerRecord(
        speaker_ref=_required(data, "speaker_ref", "speaker"),
        display_label=_optional_display_label_from_dict(data.get("display_label"), "speaker.display_label"),
    )


def _word_from_dict(payload: object) -> CanonicalWord:
    data = _require_mapping(payload, "word")
    _reject_unknown_fields(data, _WORD_FIELDS, "word")
    return CanonicalWord(
        word_id=_required(data, "word_id", "word"),
        text=_required(data, "text", "word"),
        start_ms=_required(data, "start_ms", "word"),
        end_ms=_required(data, "end_ms", "word"),
        speaker_ref=data.get("speaker_ref"),
        channel_id=data.get("channel_id"),
        text_confidence=data.get("text_confidence"),
        speaker_confidence=data.get("speaker_confidence"),
        overlap=data.get("overlap", False),
        display_label=_optional_display_label_from_dict(data.get("display_label"), "word.display_label"),
    )


def _speaker_span_from_dict(payload: object) -> SpeakerSpan:
    data = _require_mapping(payload, "speaker_span")
    _reject_unknown_fields(data, _SPAN_FIELDS, "speaker_span")
    return SpeakerSpan(
        span_id=_required(data, "span_id", "speaker_span"),
        speaker_ref=_required(data, "speaker_ref", "speaker_span"),
        start_ms=_required(data, "start_ms", "speaker_span"),
        end_ms=_required(data, "end_ms", "speaker_span"),
        channel_id=data.get("channel_id"),
        confidence=data.get("confidence"),
        overlap=data.get("overlap", False),
    )


def _scoring_region_from_dict(payload: object) -> ScoringRegion:
    data = _require_mapping(payload, "scoring_region")
    _reject_unknown_fields(data, _SCORING_REGION_FIELDS, "scoring_region")
    return ScoringRegion(
        region_id=_required(data, "region_id", "scoring_region"),
        start_ms=_required(data, "start_ms", "scoring_region"),
        end_ms=_required(data, "end_ms", "scoring_region"),
        channel_id=data.get("channel_id"),
    )


def _decode_sequence(
    value: object,
    decoder: Any,
    context: str,
) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValidationError(f"{context} must be a list")
    return tuple(decoder(item) for item in value)


def _require_mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{context} must be an object")
    for key in value:
        if not isinstance(key, str):
            raise ValidationError(f"{context} field names must be strings")
    return value


def _required(data: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in data:
        raise ValidationError(f"{context}.{key} is required")
    return data[key]


def _reject_unknown_fields(data: Mapping[str, Any], allowed: frozenset[str], context: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValidationError(f"{context} has unsupported fields: {', '.join(unknown)}")
