"""Session-local diarization benchmark domain models."""

from keyframe.diarization.io import (
    canonical_json_dumps,
    canonical_json_loads,
    canonical_jsonl_dumps,
    canonical_jsonl_loads,
    read_recording_json,
    read_recordings_jsonl,
    recording_from_dict,
    recording_to_dict,
    validate_schema_version,
    write_recording_json,
    write_recordings_jsonl,
)
from keyframe.diarization.models import (
    CanonicalRecording,
    CanonicalWord,
    ChannelRecord,
    DisplayLabel,
    ScoringRegion,
    SpeakerRecord,
    SpeakerSpan,
    ValidationError,
)

__all__ = [
    "CanonicalRecording",
    "CanonicalWord",
    "ChannelRecord",
    "DisplayLabel",
    "ScoringRegion",
    "SpeakerRecord",
    "SpeakerSpan",
    "ValidationError",
    "canonical_json_dumps",
    "canonical_json_loads",
    "canonical_jsonl_dumps",
    "canonical_jsonl_loads",
    "read_recording_json",
    "read_recordings_jsonl",
    "recording_from_dict",
    "recording_to_dict",
    "validate_schema_version",
    "write_recording_json",
    "write_recordings_jsonl",
]
