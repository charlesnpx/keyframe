"""Release-validation helpers for comparing nondeterministic model output."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from keyframe.transcript import DiarizationRow, TranscriptSegment


@dataclass(frozen=True)
class DiarizationPartitionComparison:
    """Result of comparing two diarization partitions modulo speaker labels."""

    equivalent: bool
    reason: str
    label_mapping: tuple[tuple[str, str], ...]
    max_timestamp_delta_seconds: float
    reference_row_count: int
    candidate_row_count: int


@dataclass(frozen=True)
class TranscriptQualityComparison:
    """Deterministic transcript agreement metrics used by release benchmarks."""

    reference_word_count: int
    candidate_word_count: int
    normalized_word_agreement: float
    normalized_word_edit_distance: int
    normalized_word_error_rate: float
    character_agreement: float
    reference_duplicate_ngrams: int
    candidate_duplicate_ngrams: int
    exact_opening_segments: int
    segment_count_relative_delta: float
    reference_end_seconds: float
    candidate_end_seconds: float


_WORD_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


def _segment_text(value: TranscriptSegment | Mapping[str, Any]) -> str:
    if isinstance(value, TranscriptSegment):
        return value.text
    if not isinstance(value, Mapping):
        raise TypeError(f"transcript comparison row must be a segment or mapping: {value!r}")
    text = value.get("text")
    if not isinstance(text, str):
        raise ValueError(f"transcript comparison row has invalid text: {value!r}")
    return text


def _segment_end(value: TranscriptSegment | Mapping[str, Any]) -> float:
    raw_end = value.end if isinstance(value, TranscriptSegment) else value.get("end")
    try:
        end = float(raw_end)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"transcript comparison row has invalid end: {value!r}") from exc
    if not math.isfinite(end) or end < 0:
        raise ValueError(f"transcript comparison row has invalid end: {value!r}")
    return end


def normalize_transcript_words(text: str) -> tuple[str, ...]:
    """Normalize case, apostrophes, and punctuation into comparison words."""

    if not isinstance(text, str):
        raise TypeError("transcript text must be a string")
    return tuple(
        match.group(0).replace("’", "'").casefold()
        for match in _WORD_PATTERN.finditer(text)
    )


def _edit_distance(reference: tuple[str, ...], candidate: tuple[str, ...]) -> int:
    if len(reference) < len(candidate):
        reference, candidate = candidate, reference
    previous = list(range(len(candidate) + 1))
    for row_index, reference_word in enumerate(reference, 1):
        current = [row_index]
        for column_index, candidate_word in enumerate(candidate, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column_index] + 1,
                    previous[column_index - 1]
                    + (reference_word != candidate_word),
                )
            )
        previous = current
    return previous[-1]


def _duplicate_ngrams(words: tuple[str, ...], size: int) -> int:
    if size < 1:
        raise ValueError("ngram size must be at least one")
    counts: dict[tuple[str, ...], int] = {}
    for index in range(max(0, len(words) - size + 1)):
        ngram = words[index : index + size]
        counts[ngram] = counts.get(ngram, 0) + 1
    return sum(count - 1 for count in counts.values() if count > 1)


def compare_transcript_quality(
    reference: Iterable[TranscriptSegment | Mapping[str, Any]],
    candidate: Iterable[TranscriptSegment | Mapping[str, Any]],
    *,
    duplicate_ngram_size: int = 5,
) -> TranscriptQualityComparison:
    """Compare normalized text while retaining segment and coverage signals."""

    reference_segments = tuple(reference)
    candidate_segments = tuple(candidate)
    reference_segment_words = tuple(
        normalize_transcript_words(_segment_text(segment))
        for segment in reference_segments
    )
    candidate_segment_words = tuple(
        normalize_transcript_words(_segment_text(segment))
        for segment in candidate_segments
    )
    reference_words = tuple(
        word for segment in reference_segment_words for word in segment
    )
    candidate_words = tuple(
        word for segment in candidate_segment_words for word in segment
    )
    edit_distance = _edit_distance(reference_words, candidate_words)
    word_error_rate = edit_distance / max(1, len(reference_words))
    normalized_reference = " ".join(reference_words)
    normalized_candidate = " ".join(candidate_words)
    opening_matches = 0
    for expected, actual in zip(reference_segment_words, candidate_segment_words):
        if expected != actual:
            break
        opening_matches += 1
    segment_delta = abs(len(candidate_segments) - len(reference_segments)) / max(
        1,
        len(reference_segments),
    )

    return TranscriptQualityComparison(
        reference_word_count=len(reference_words),
        candidate_word_count=len(candidate_words),
        normalized_word_agreement=SequenceMatcher(
            None,
            reference_words,
            candidate_words,
            autojunk=False,
        ).ratio(),
        normalized_word_edit_distance=edit_distance,
        normalized_word_error_rate=word_error_rate,
        character_agreement=SequenceMatcher(
            None,
            normalized_reference,
            normalized_candidate,
            autojunk=False,
        ).ratio(),
        reference_duplicate_ngrams=_duplicate_ngrams(
            reference_words,
            duplicate_ngram_size,
        ),
        candidate_duplicate_ngrams=_duplicate_ngrams(
            candidate_words,
            duplicate_ngram_size,
        ),
        exact_opening_segments=opening_matches,
        segment_count_relative_delta=segment_delta,
        reference_end_seconds=(
            max(_segment_end(segment) for segment in reference_segments)
            if reference_segments
            else 0.0
        ),
        candidate_end_seconds=(
            max(_segment_end(segment) for segment in candidate_segments)
            if candidate_segments
            else 0.0
        ),
    )


CRITICAL_PATH_EXPRESSIONS = frozenset(
    {
        "max(T + F, D) + M + E",
        "max(T, D) + F + M + E",
        "T + D + F + M + E",
    }
)


def expected_critical_path_seconds(
    expression: str,
    timings: Mapping[str, float],
) -> float:
    """Evaluate one documented full-pipeline dependency expression."""

    if expression not in CRITICAL_PATH_EXPRESSIONS:
        raise ValueError(f"unsupported critical-path expression: {expression!r}")
    try:
        transcription = float(timings["transcription"])
        diarization = float(timings["diarization"])
        frames = float(timings["frames"])
        merge = float(timings["merge"])
        enrichment = float(timings["manifest"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("critical-path timings are incomplete or invalid") from exc
    values = (transcription, diarization, frames, merge, enrichment)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("critical-path timings must be finite and non-negative")

    if expression == "max(T + F, D) + M + E":
        return max(transcription + frames, diarization) + merge + enrichment
    if expression == "max(T, D) + F + M + E":
        return max(transcription, diarization) + frames + merge + enrichment
    return transcription + diarization + frames + merge + enrichment


def _comparison_row(value: DiarizationRow | Mapping[str, Any]) -> DiarizationRow:
    if isinstance(value, DiarizationRow):
        row = value
    elif isinstance(value, Mapping):
        try:
            row = DiarizationRow(
                start=float(value["start"]),
                end=float(value["end"]),
                speaker=str(value["speaker"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid diarization comparison row: {value!r}") from exc
    else:
        raise TypeError(f"diarization comparison row must be a row or mapping: {value!r}")

    if (
        not math.isfinite(row.start)
        or not math.isfinite(row.end)
        or row.start < 0
        or row.end <= row.start
        or not row.speaker.strip()
    ):
        raise ValueError(f"invalid diarization comparison row: {value!r}")
    return DiarizationRow(row.start, row.end, row.speaker.strip())


def _comparison_rows(
    values: Iterable[DiarizationRow | Mapping[str, Any]],
) -> tuple[DiarizationRow, ...]:
    return tuple(_comparison_row(value) for value in values)


def _speaker_intervals(
    rows: Iterable[DiarizationRow],
) -> dict[str, tuple[tuple[float, float], ...]]:
    grouped: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        grouped.setdefault(row.speaker, []).append((row.start, row.end))
    return {
        speaker: tuple(sorted(intervals))
        for speaker, intervals in grouped.items()
    }


def _interval_delta(
    reference: tuple[tuple[float, float], ...],
    candidate: tuple[tuple[float, float], ...],
) -> float | None:
    if len(reference) != len(candidate):
        return None
    maximum = 0.0
    for expected, actual in zip(reference, candidate, strict=True):
        maximum = max(
            maximum,
            abs(expected[0] - actual[0]),
            abs(expected[1] - actual[1]),
        )
    return maximum


def compare_diarization_partitions(
    reference: Iterable[DiarizationRow | Mapping[str, Any]],
    candidate: Iterable[DiarizationRow | Mapping[str, Any]],
    *,
    timestamp_tolerance_seconds: float = 0.05,
) -> DiarizationPartitionComparison:
    """Compare row partitions with timing tolerance and a bijective label map.

    Rows are ordered by their time interval, then compared one-for-one. Speaker
    names may differ completely, but one candidate label must map consistently
    to exactly one reference label and vice versa. A changed boundary, split,
    merge, or speaker partition therefore remains visible to release checks.
    """

    try:
        tolerance = float(timestamp_tolerance_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("timestamp tolerance must be a finite non-negative number") from exc
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("timestamp tolerance must be a finite non-negative number")

    reference_rows = _comparison_rows(reference)
    candidate_rows = _comparison_rows(candidate)
    reference_count = len(reference_rows)
    candidate_count = len(candidate_rows)

    def result(
        equivalent: bool,
        reason: str,
        mapping: Mapping[str, str],
        maximum_delta: float,
    ) -> DiarizationPartitionComparison:
        return DiarizationPartitionComparison(
            equivalent=equivalent,
            reason=reason,
            label_mapping=tuple(sorted(mapping.items())),
            max_timestamp_delta_seconds=maximum_delta,
            reference_row_count=reference_count,
            candidate_row_count=candidate_count,
        )

    if reference_count != candidate_count:
        return result(
            False,
            f"row count changed from {reference_count} to {candidate_count}",
            {},
            0.0,
        )

    reference_speakers = _speaker_intervals(reference_rows)
    candidate_speakers = _speaker_intervals(candidate_rows)
    if len(reference_speakers) != len(candidate_speakers):
        return result(
            False,
            (
                "speaker partition changed from "
                f"{len(reference_speakers)} to {len(candidate_speakers)} labels"
            ),
            {},
            0.0,
        )

    edge_deltas: dict[tuple[str, str], float] = {}
    compatible: dict[str, tuple[str, ...]] = {}
    for candidate_speaker, candidate_intervals in sorted(candidate_speakers.items()):
        candidates = []
        closest_delta: float | None = None
        for reference_speaker, reference_intervals in sorted(reference_speakers.items()):
            delta = _interval_delta(reference_intervals, candidate_intervals)
            if delta is None:
                continue
            closest_delta = delta if closest_delta is None else min(closest_delta, delta)
            if delta <= tolerance:
                candidates.append(reference_speaker)
                edge_deltas[(candidate_speaker, reference_speaker)] = delta
        if not candidates:
            if closest_delta is None:
                reason = (
                    f"speaker {candidate_speaker!r} has a different number of turns"
                )
                maximum_delta = 0.0
            else:
                reason = (
                    f"speaker {candidate_speaker!r} boundary changed by at least "
                    f"{closest_delta:.9f}s (tolerance {tolerance:.9f}s)"
                )
                maximum_delta = closest_delta
            return result(False, reason, {}, maximum_delta)
        compatible[candidate_speaker] = tuple(candidates)

    reference_to_candidate: dict[str, str] = {}

    def assign(candidate_speaker: str, seen: set[str]) -> bool:
        for reference_speaker in compatible[candidate_speaker]:
            if reference_speaker in seen:
                continue
            seen.add(reference_speaker)
            previous = reference_to_candidate.get(reference_speaker)
            if previous is None or assign(previous, seen):
                reference_to_candidate[reference_speaker] = candidate_speaker
                return True
        return False

    for candidate_speaker in sorted(
        compatible,
        key=lambda speaker: (len(compatible[speaker]), speaker),
    ):
        if not assign(candidate_speaker, set()):
            return result(
                False,
                "speaker turns do not admit one global bijective label mapping",
                {},
                0.0,
            )

    candidate_to_reference = {
        candidate_speaker: reference_speaker
        for reference_speaker, candidate_speaker in reference_to_candidate.items()
    }
    maximum_delta = max(
        (
            edge_deltas[(candidate_speaker, reference_speaker)]
            for candidate_speaker, reference_speaker in candidate_to_reference.items()
        ),
        default=0.0,
    )

    return result(
        True,
        "partitions are equivalent within timestamp tolerance",
        candidate_to_reference,
        maximum_delta,
    )
