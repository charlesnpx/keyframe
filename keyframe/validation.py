"""Release-validation helpers for comparing nondeterministic model output."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from keyframe.transcript import DiarizationRow


@dataclass(frozen=True)
class DiarizationPartitionComparison:
    """Result of comparing two diarization partitions modulo speaker labels."""

    equivalent: bool
    reason: str
    label_mapping: tuple[tuple[str, str], ...]
    max_timestamp_delta_seconds: float
    reference_row_count: int
    candidate_row_count: int


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
    return tuple(
        sorted(
            (_comparison_row(value) for value in values),
            key=lambda row: (row.start, row.end, row.speaker),
        )
    )


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

    candidate_to_reference: dict[str, str] = {}
    reference_to_candidate: dict[str, str] = {}
    maximum_delta = 0.0
    for index, (expected, actual) in enumerate(
        zip(reference_rows, candidate_rows, strict=True)
    ):
        start_delta = abs(expected.start - actual.start)
        end_delta = abs(expected.end - actual.end)
        row_delta = max(start_delta, end_delta)
        maximum_delta = max(maximum_delta, row_delta)
        if row_delta > tolerance:
            return result(
                False,
                (
                    f"row {index} boundary changed by {row_delta:.9f}s "
                    f"(tolerance {tolerance:.9f}s)"
                ),
                candidate_to_reference,
                maximum_delta,
            )

        mapped_reference = candidate_to_reference.get(actual.speaker)
        mapped_candidate = reference_to_candidate.get(expected.speaker)
        if mapped_reference is not None and mapped_reference != expected.speaker:
            return result(
                False,
                f"candidate speaker {actual.speaker!r} maps to multiple reference speakers",
                candidate_to_reference,
                maximum_delta,
            )
        if mapped_candidate is not None and mapped_candidate != actual.speaker:
            return result(
                False,
                f"reference speaker {expected.speaker!r} maps to multiple candidate speakers",
                candidate_to_reference,
                maximum_delta,
            )
        candidate_to_reference[actual.speaker] = expected.speaker
        reference_to_candidate[expected.speaker] = actual.speaker

    return result(
        True,
        "partitions are equivalent within timestamp tolerance",
        candidate_to_reference,
        maximum_delta,
    )
