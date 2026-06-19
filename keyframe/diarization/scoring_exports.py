"""Derived RTTM/UEM scoring exports for canonical diarization records."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from keyframe.diarization.models import CanonicalRecording, ScoringRegion, SpeakerSpan, ValidationError


@dataclass(frozen=True)
class RttmRow:
    """One strict SPEAKER row in the derived RTTM contract."""

    recording_id: str
    channel_id: str
    start_ms: int
    duration_ms: int
    speaker_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "recording_id", _require_token(self.recording_id, "rttm.recording_id"))
        object.__setattr__(self, "channel_id", _require_token(self.channel_id, "rttm.channel_id"))
        object.__setattr__(self, "start_ms", _require_non_negative_int(self.start_ms, "rttm.start_ms"))
        object.__setattr__(self, "duration_ms", _require_positive_int(self.duration_ms, "rttm.duration_ms"))
        object.__setattr__(self, "speaker_ref", _require_token(self.speaker_ref, "rttm.speaker_ref"))

    @property
    def end_ms(self) -> int:
        return self.start_ms + self.duration_ms

    def to_line(self) -> str:
        return " ".join(
            (
                "SPEAKER",
                self.recording_id,
                self.channel_id,
                _format_seconds(self.start_ms),
                _format_seconds(self.duration_ms),
                "<NA>",
                "<NA>",
                self.speaker_ref,
                "<NA>",
                "<NA>",
            )
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["end_ms"] = self.end_ms
        return payload


@dataclass(frozen=True)
class UemRow:
    """One strict scoring region row in the derived UEM contract."""

    recording_id: str
    channel_id: str
    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "recording_id", _require_token(self.recording_id, "uem.recording_id"))
        object.__setattr__(self, "channel_id", _require_token(self.channel_id, "uem.channel_id"))
        start_ms = _require_non_negative_int(self.start_ms, "uem.start_ms")
        end_ms = _require_positive_int(self.end_ms, "uem.end_ms")
        if end_ms <= start_ms:
            raise ValidationError("uem.end_ms must be greater than start_ms")
        object.__setattr__(self, "start_ms", start_ms)
        object.__setattr__(self, "end_ms", end_ms)

    def to_line(self) -> str:
        return " ".join(
            (
                self.recording_id,
                self.channel_id,
                _format_seconds(self.start_ms),
                _format_seconds(self.end_ms),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RttmPairScore:
    """Strict deterministic score summary for RTTM-vs-RTTM fixture checks."""

    reference_speech_ms: int
    hypothesis_speech_ms: int
    matched_speech_ms: int
    false_alarm_ms: int
    missed_speech_ms: int
    score: float
    threshold: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def recording_to_rttm_text(recording: CanonicalRecording) -> str:
    """Export canonical speaker spans to deterministic RTTM text."""

    if not isinstance(recording, CanonicalRecording):
        raise ValidationError("recording must be a CanonicalRecording")
    return speaker_spans_to_rttm_text(recording.recording_id, recording.speaker_spans)


def speaker_spans_to_rttm_text(recording_id: str, speaker_spans: Iterable[SpeakerSpan]) -> str:
    """Export reference or hypothesis speaker spans to deterministic RTTM text."""

    recording_id = _require_token(recording_id, "recording_id")
    spans = tuple(speaker_spans)
    if not spans:
        raise ValidationError("RTTM export requires at least one speaker span")
    rows: list[RttmRow] = []
    for index, span in enumerate(spans):
        if not isinstance(span, SpeakerSpan):
            raise ValidationError(f"speaker_spans[{index}] must be a SpeakerSpan")
        rows.append(
            RttmRow(
                recording_id=recording_id,
                channel_id=_scoring_channel_id(span.channel_id),
                start_ms=span.start_ms,
                duration_ms=span.end_ms - span.start_ms,
                speaker_ref=span.speaker_ref,
            )
        )
    rows = sorted(rows, key=lambda row: (row.start_ms, row.end_ms, row.channel_id, row.speaker_ref))
    return _join_lines(row.to_line() for row in rows)


def recording_to_uem_text(recording: CanonicalRecording) -> str:
    """Export canonical scoring regions to deterministic UEM text."""

    if not isinstance(recording, CanonicalRecording):
        raise ValidationError("recording must be a CanonicalRecording")
    return scoring_regions_to_uem_text(recording.recording_id, recording.scoring_regions)


def scoring_regions_to_uem_text(recording_id: str, scoring_regions: Iterable[ScoringRegion]) -> str:
    """Export scoring regions to deterministic UEM text."""

    recording_id = _require_token(recording_id, "recording_id")
    regions = tuple(scoring_regions)
    if not regions:
        raise ValidationError("UEM export requires at least one scoring region")
    rows: list[UemRow] = []
    for index, region in enumerate(regions):
        if not isinstance(region, ScoringRegion):
            raise ValidationError(f"scoring_regions[{index}] must be a ScoringRegion")
        rows.append(
            UemRow(
                recording_id=recording_id,
                channel_id=_scoring_channel_id(region.channel_id),
                start_ms=region.start_ms,
                end_ms=region.end_ms,
            )
        )
    rows = sorted(rows, key=lambda row: (row.channel_id, row.start_ms, row.end_ms))
    return _join_lines(row.to_line() for row in rows)


def write_rttm(path: str | Path, recording: CanonicalRecording) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(recording_to_rttm_text(recording), encoding="utf-8", newline="\n")


def write_uem(path: str | Path, recording: CanonicalRecording) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(recording_to_uem_text(recording), encoding="utf-8", newline="\n")


def validate_rttm_text(text: str) -> tuple[RttmRow, ...]:
    """Parse and validate strict RTTM text emitted by this package."""

    if not isinstance(text, str):
        raise ValidationError("RTTM text must be a string")
    lines = text.splitlines()
    if not lines:
        raise ValidationError("RTTM export is empty")
    rows: list[RttmRow] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValidationError(f"RTTM line {line_number} is empty")
        fields = line.split()
        if len(fields) != 10:
            raise ValidationError(f"RTTM line {line_number} must have 10 fields")
        if fields[0] != "SPEAKER":
            raise ValidationError(f"RTTM line {line_number} must start with SPEAKER")
        if fields[5] != "<NA>" or fields[6] != "<NA>" or fields[8] != "<NA>" or fields[9] != "<NA>":
            raise ValidationError(f"RTTM line {line_number} uses unsupported optional fields")
        rows.append(
            RttmRow(
                recording_id=_require_token(fields[1], f"RTTM line {line_number} recording_id"),
                channel_id=_require_token(fields[2], f"RTTM line {line_number} channel_id"),
                start_ms=_parse_seconds_ms(fields[3], f"RTTM line {line_number} start"),
                duration_ms=_parse_positive_seconds_ms(fields[4], f"RTTM line {line_number} duration"),
                speaker_ref=_require_token(fields[7], f"RTTM line {line_number} speaker"),
            )
        )
    return tuple(rows)


def validate_uem_text(text: str) -> tuple[UemRow, ...]:
    """Parse and validate strict UEM text emitted by this package."""

    if not isinstance(text, str):
        raise ValidationError("UEM text must be a string")
    lines = text.splitlines()
    if not lines:
        raise ValidationError("UEM export is empty")
    rows: list[UemRow] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValidationError(f"UEM line {line_number} is empty")
        fields = line.split()
        if len(fields) != 4:
            raise ValidationError(f"UEM line {line_number} must have 4 fields")
        rows.append(
            UemRow(
                recording_id=_require_token(fields[0], f"UEM line {line_number} recording_id"),
                channel_id=_require_token(fields[1], f"UEM line {line_number} channel_id"),
                start_ms=_parse_seconds_ms(fields[2], f"UEM line {line_number} start"),
                end_ms=_parse_positive_seconds_ms(fields[3], f"UEM line {line_number} end"),
            )
        )
    return tuple(rows)


def score_rttm_pair(
    reference_rttm: str,
    hypothesis_rttm: str,
    uem: str,
    *,
    threshold: float = 0.999999,
) -> RttmPairScore:
    """Score two strict RTTM exports by exact speaker-span agreement inside UEM regions."""

    threshold = _validate_threshold(threshold)
    uem_rows = validate_uem_text(uem)
    reference_rows = _clip_rows_to_uem(validate_rttm_text(reference_rttm), uem_rows)
    hypothesis_rows = _clip_rows_to_uem(validate_rttm_text(hypothesis_rttm), uem_rows)

    reference_counter = Counter(_row_key(row) for row in reference_rows)
    hypothesis_counter = Counter(_row_key(row) for row in hypothesis_rows)
    reference_speech_ms = sum(row.duration_ms for row in reference_rows)
    hypothesis_speech_ms = sum(row.duration_ms for row in hypothesis_rows)
    matched_speech_ms = 0
    for key, reference_count in reference_counter.items():
        matched_count = min(reference_count, hypothesis_counter.get(key, 0))
        matched_speech_ms += matched_count * key[4]
    false_alarm_ms = max(0, hypothesis_speech_ms - matched_speech_ms)
    missed_speech_ms = max(0, reference_speech_ms - matched_speech_ms)
    denominator = max(reference_speech_ms, hypothesis_speech_ms)
    score = 1.0 if denominator == 0 else matched_speech_ms / denominator
    return RttmPairScore(
        reference_speech_ms=reference_speech_ms,
        hypothesis_speech_ms=hypothesis_speech_ms,
        matched_speech_ms=matched_speech_ms,
        false_alarm_ms=false_alarm_ms,
        missed_speech_ms=missed_speech_ms,
        score=score,
        threshold=threshold,
        passed=score >= threshold,
    )


def rttm_oracle_self_score(recording: CanonicalRecording, *, threshold: float = 0.999999) -> RttmPairScore:
    """Validate that a reference exported as both reference and hypothesis self-scores strictly."""

    rttm = recording_to_rttm_text(recording)
    uem = recording_to_uem_text(recording)
    return score_rttm_pair(rttm, rttm, uem, threshold=threshold)


def _join_lines(lines: Iterable[str]) -> str:
    return "\n".join(lines) + "\n"


def _scoring_channel_id(channel_id: str | None) -> str:
    return _require_token("1" if channel_id is None else channel_id, "channel_id")


def _row_key(row: RttmRow) -> tuple[str, str, int, int, int, str]:
    return (row.recording_id, row.channel_id, row.start_ms, row.end_ms, row.duration_ms, row.speaker_ref)


def _clip_rows_to_uem(rows: tuple[RttmRow, ...], uem_rows: tuple[UemRow, ...]) -> tuple[RttmRow, ...]:
    clipped: list[RttmRow] = []
    for row in rows:
        for uem_row in uem_rows:
            if uem_row.recording_id != row.recording_id or uem_row.channel_id != row.channel_id:
                continue
            start_ms = max(row.start_ms, uem_row.start_ms)
            end_ms = min(row.end_ms, uem_row.end_ms)
            if end_ms <= start_ms:
                continue
            clipped.append(
                RttmRow(
                    recording_id=row.recording_id,
                    channel_id=row.channel_id,
                    start_ms=start_ms,
                    duration_ms=end_ms - start_ms,
                    speaker_ref=row.speaker_ref,
                )
            )
    return tuple(
        sorted(
            clipped,
            key=lambda item: (item.recording_id, item.channel_id, item.start_ms, item.end_ms, item.speaker_ref),
        )
    )


def _format_seconds(milliseconds: int) -> str:
    return f"{milliseconds / 1000:.3f}"


def _parse_seconds_ms(value: str, field_name: str) -> int:
    value = _require_token(value, field_name)
    try:
        seconds = float(value)
    except ValueError as exc:
        raise ValidationError(f"{field_name} must be a number of seconds") from exc
    if not math.isfinite(seconds):
        raise ValidationError(f"{field_name} must be finite")
    milliseconds = round(seconds * 1000)
    if milliseconds < 0:
        raise ValidationError(f"{field_name} must be >= 0")
    return milliseconds


def _parse_positive_seconds_ms(value: str, field_name: str) -> int:
    milliseconds = _parse_seconds_ms(value, field_name)
    if milliseconds <= 0:
        raise ValidationError(f"{field_name} must be greater than 0")
    return milliseconds


def _require_token(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{field_name} must be a string")
    value = value.strip()
    if not value:
        raise ValidationError(f"{field_name} is required")
    if any(char.isspace() for char in value):
        raise ValidationError(f"{field_name} must be a single token")
    return value


def _require_non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{field_name} must be an integer")
    if value < 0:
        raise ValidationError(f"{field_name} must be >= 0")
    return value


def _require_positive_int(value: object, field_name: str) -> int:
    value = _require_non_negative_int(value, field_name)
    if value <= 0:
        raise ValidationError(f"{field_name} must be greater than 0")
    return value


def _validate_threshold(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError("threshold must be a number")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValidationError("threshold must be between 0.0 and 1.0")
    return value
