from __future__ import annotations

import pytest

from keyframe.transcript import DiarizationRow
from keyframe.validation import compare_diarization_partitions


def _reference_partition():
    return (
        DiarizationRow(0.0, 1.25, "SPEAKER_00"),
        DiarizationRow(1.25, 2.5, "SPEAKER_01"),
        DiarizationRow(2.5, 4.0, "SPEAKER_00"),
    )


def test_partition_comparison_tolerates_precision_order_and_label_permutation():
    candidate = (
        {"start": 2.5004, "end": 4.0003, "speaker": "person-b"},
        {"start": 0.0002, "end": 1.2497, "speaker": "person-b"},
        {"start": 1.2498, "end": 2.5001, "speaker": "person-a"},
    )

    comparison = compare_diarization_partitions(
        _reference_partition(),
        candidate,
        timestamp_tolerance_seconds=0.001,
    )

    assert comparison.equivalent
    assert comparison.label_mapping == (
        ("person-a", "SPEAKER_01"),
        ("person-b", "SPEAKER_00"),
    )
    assert comparison.max_timestamp_delta_seconds == pytest.approx(0.0004)


def test_partition_comparison_resolves_ambiguous_simultaneous_speaker_rows():
    reference = (
        DiarizationRow(0.0, 1.0, "SPEAKER_00"),
        DiarizationRow(0.0, 1.0, "SPEAKER_01"),
        DiarizationRow(1.0, 2.0, "SPEAKER_00"),
    )
    candidate = (
        DiarizationRow(0.0, 1.0, "person-a"),
        DiarizationRow(0.0, 1.0, "person-b"),
        DiarizationRow(1.0, 2.0, "person-b"),
    )

    comparison = compare_diarization_partitions(reference, candidate)

    assert comparison.equivalent
    assert comparison.label_mapping == (
        ("person-a", "SPEAKER_01"),
        ("person-b", "SPEAKER_00"),
    )


@pytest.mark.parametrize(
    ("candidate", "reason"),
    [
        (
            (
                DiarizationRow(0.0, 1.25, "A"),
                DiarizationRow(1.25, 2.5, "A"),
                DiarizationRow(2.5, 4.0, "A"),
            ),
            "speaker partition changed",
        ),
        (
            (
                DiarizationRow(0.0, 1.25, "A"),
                DiarizationRow(1.25, 2.75, "B"),
                DiarizationRow(2.75, 4.0, "A"),
            ),
            "boundary changed",
        ),
        (
            (
                DiarizationRow(0.0, 1.25, "A"),
                DiarizationRow(1.25, 4.0, "B"),
            ),
            "row count changed",
        ),
    ],
)
def test_partition_comparison_detects_speaker_and_boundary_regressions(
    candidate,
    reason,
):
    comparison = compare_diarization_partitions(
        _reference_partition(),
        candidate,
        timestamp_tolerance_seconds=0.01,
    )

    assert not comparison.equivalent
    assert reason in comparison.reason


@pytest.mark.parametrize("tolerance", [-0.1, float("nan"), float("inf"), "bad"])
def test_partition_comparison_rejects_invalid_tolerance(tolerance):
    with pytest.raises(ValueError, match="timestamp tolerance"):
        compare_diarization_partitions(
            _reference_partition(),
            _reference_partition(),
            timestamp_tolerance_seconds=tolerance,
        )
