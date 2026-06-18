"""Canonical session-local diarization data model.

These models describe one recording/transcript at a time. Identifiers such as
``speaker_ref`` and ``person_1`` are scoped to that recording only; this module
intentionally has no fields for cross-call identity, voice profiles, embeddings,
or audio fingerprints.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


SCHEMA_VERSION = 1
LabelSource = Literal[
    "diarization_cluster",
    "channel_metadata",
    "reviewer_rename",
    "unknown",
]
LabelScope = Literal["recording"]


class ValidationError(ValueError):
    """Raised when canonical diarization records are internally inconsistent."""


def _require_id(value: str, field_name: str) -> str:
    value = str(value).strip()
    if not value:
        raise ValidationError(f"{field_name} is required")
    return value


def _validate_interval(start_ms: int, end_ms: int, *, context: str) -> tuple[int, int]:
    start_ms = int(start_ms)
    end_ms = int(end_ms)
    if start_ms < 0:
        raise ValidationError(f"{context}.start_ms must be >= 0")
    if end_ms <= start_ms:
        raise ValidationError(f"{context}.end_ms must be greater than start_ms")
    return start_ms, end_ms


def _validate_confidence(value: float | None, *, field_name: str) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValidationError(f"{field_name} must be between 0.0 and 1.0")
    return value


@dataclass(frozen=True)
class DisplayLabel:
    """A user-facing label for one recording, such as ``person_1``."""

    label: str
    source: LabelSource
    scope: LabelScope = "recording"
    confidence: float | None = None
    source_ref: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", _require_id(self.label, "display_label.label"))
        if self.scope != "recording":
            raise ValidationError("display labels must be scoped to one recording")
        object.__setattr__(
            self,
            "confidence",
            _validate_confidence(self.confidence, field_name="display_label.confidence"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChannelRecord:
    """A channel or participant track available within one recording."""

    channel_id: str
    name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "channel_id", _require_id(self.channel_id, "channel_id"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SpeakerRecord:
    """A session-local speaker reference produced by diarization or metadata."""

    speaker_ref: str
    display_label: DisplayLabel | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "speaker_ref", _require_id(self.speaker_ref, "speaker_ref"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CanonicalWord:
    """A timestamped word with optional speaker and channel attribution."""

    word_id: str
    text: str
    start_ms: int
    end_ms: int
    speaker_ref: str | None = None
    channel_id: str | None = None
    text_confidence: float | None = None
    speaker_confidence: float | None = None
    overlap: bool = False
    display_label: DisplayLabel | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "word_id", _require_id(self.word_id, "word_id"))
        object.__setattr__(self, "text", str(self.text))
        start_ms, end_ms = _validate_interval(self.start_ms, self.end_ms, context=f"word {self.word_id}")
        object.__setattr__(self, "start_ms", start_ms)
        object.__setattr__(self, "end_ms", end_ms)
        object.__setattr__(
            self,
            "text_confidence",
            _validate_confidence(self.text_confidence, field_name=f"word {self.word_id}.text_confidence"),
        )
        object.__setattr__(
            self,
            "speaker_confidence",
            _validate_confidence(
                self.speaker_confidence,
                field_name=f"word {self.word_id}.speaker_confidence",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SpeakerSpan:
    """A region of speech attributed to a session-local speaker."""

    span_id: str
    speaker_ref: str
    start_ms: int
    end_ms: int
    channel_id: str | None = None
    confidence: float | None = None
    overlap: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "span_id", _require_id(self.span_id, "span_id"))
        object.__setattr__(self, "speaker_ref", _require_id(self.speaker_ref, "speaker_ref"))
        start_ms, end_ms = _validate_interval(self.start_ms, self.end_ms, context=f"span {self.span_id}")
        object.__setattr__(self, "start_ms", start_ms)
        object.__setattr__(self, "end_ms", end_ms)
        object.__setattr__(
            self,
            "confidence",
            _validate_confidence(self.confidence, field_name=f"span {self.span_id}.confidence"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScoringRegion:
    """A scored interval on the canonical timeline."""

    region_id: str
    start_ms: int
    end_ms: int
    channel_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "region_id", _require_id(self.region_id, "region_id"))
        start_ms, end_ms = _validate_interval(
            self.start_ms,
            self.end_ms,
            context=f"scoring_region {self.region_id}",
        )
        object.__setattr__(self, "start_ms", start_ms)
        object.__setattr__(self, "end_ms", end_ms)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CanonicalRecording:
    """A complete canonical transcript/evidence record for one recording."""

    recording_id: str
    original_audio_id: str
    canonical_audio_id: str
    timeline_id: str
    duration_ms: int
    channels: tuple[ChannelRecord, ...] = ()
    speakers: tuple[SpeakerRecord, ...] = ()
    words: tuple[CanonicalWord, ...] = ()
    speaker_spans: tuple[SpeakerSpan, ...] = ()
    scoring_regions: tuple[ScoringRegion, ...] = ()
    schema_version: int = field(default=SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "recording_id", _require_id(self.recording_id, "recording_id"))
        object.__setattr__(self, "original_audio_id", _require_id(self.original_audio_id, "original_audio_id"))
        object.__setattr__(self, "canonical_audio_id", _require_id(self.canonical_audio_id, "canonical_audio_id"))
        object.__setattr__(self, "timeline_id", _require_id(self.timeline_id, "timeline_id"))
        duration_ms = int(self.duration_ms)
        if duration_ms <= 0:
            raise ValidationError("duration_ms must be greater than 0")
        object.__setattr__(self, "duration_ms", duration_ms)

        channels = tuple(self.channels)
        speakers = tuple(self.speakers)
        words = tuple(self.words)
        speaker_spans = tuple(self.speaker_spans)
        scoring_regions = tuple(self.scoring_regions)
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "speakers", speakers)
        object.__setattr__(self, "words", words)
        object.__setattr__(self, "speaker_spans", speaker_spans)
        object.__setattr__(self, "scoring_regions", scoring_regions)
        self.validate()

    def validate(self) -> None:
        channel_ids = _unique_ids((channel.channel_id for channel in self.channels), "channel_id")
        speaker_refs = _unique_ids((speaker.speaker_ref for speaker in self.speakers), "speaker_ref")

        for word in self.words:
            _validate_within_duration(word.start_ms, word.end_ms, self.duration_ms, f"word {word.word_id}")
            _validate_optional_ref(word.channel_id, channel_ids, "channel_id", f"word {word.word_id}")
            _validate_optional_ref(word.speaker_ref, speaker_refs, "speaker_ref", f"word {word.word_id}")

        for span in self.speaker_spans:
            _validate_within_duration(span.start_ms, span.end_ms, self.duration_ms, f"span {span.span_id}")
            _validate_optional_ref(span.channel_id, channel_ids, "channel_id", f"span {span.span_id}")
            _validate_optional_ref(span.speaker_ref, speaker_refs, "speaker_ref", f"span {span.span_id}")

        for region in self.scoring_regions:
            _validate_within_duration(
                region.start_ms,
                region.end_ms,
                self.duration_ms,
                f"scoring_region {region.region_id}",
            )
            _validate_optional_ref(region.channel_id, channel_ids, "channel_id", f"scoring_region {region.region_id}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _unique_ids(values: object, field_name: str) -> set[str]:
    seen: set[str] = set()
    for value in values:
        value = _require_id(str(value), field_name)
        if value in seen:
            raise ValidationError(f"duplicate {field_name}: {value}")
        seen.add(value)
    return seen


def _validate_optional_ref(value: str | None, allowed: set[str], field_name: str, context: str) -> None:
    if value is None:
        return
    if value not in allowed:
        raise ValidationError(f"{context} references unknown {field_name}: {value}")


def _validate_within_duration(start_ms: int, end_ms: int, duration_ms: int, context: str) -> None:
    if end_ms > duration_ms:
        raise ValidationError(f"{context} ends after recording duration")
