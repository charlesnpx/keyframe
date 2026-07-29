from __future__ import annotations

import numpy as np
import pytest

from keyframe.pipeline.primary_selection import (
    _coverage_candidate_pool,
    build_durable_state_groups,
    build_visual_state_occurrences,
    select_primary_candidates,
)
from keyframe.visual import build_compact_frame_metric_table


def _sample_series(
    samples_per_state: list[int],
    *,
    interval_seconds: float = 0.5,
    state_dhashes: list[int] | None = None,
):
    state_ids = [
        state_id
        for state_id, sample_count in enumerate(samples_per_state)
        for _ in range(sample_count)
    ]
    sample_count = len(state_ids)
    timestamps = [index * interval_seconds for index in range(sample_count)]
    frame_indices = list(range(sample_count))
    dhash_values = state_dhashes or [0xFF << (8 * state_id) for state_id in range(len(samples_per_state))]
    dhashes = [dhash_values[state_id] for state_id in state_ids]
    embeddings = np.zeros((sample_count, len(samples_per_state)), dtype=np.float32)
    for sample_idx, state_id in enumerate(state_ids):
        embeddings[sample_idx, state_id] = 1.0
    signatures = np.stack(
        [np.full((4, 4), state_id * 24, dtype=np.uint8) for state_id in state_ids],
        axis=0,
    )
    prev = [0.0]
    prev.extend(
        float(np.mean(np.abs(signatures[index].astype(float) - signatures[index - 1].astype(float))))
        for index in range(1, sample_count)
    )
    next_delta = [*prev[1:], 0.0]
    rows = [
        {
            "textline_score": float(state_id + 1),
            "edge_score": float(state_id + 1),
            "entropy": float(state_id + 1),
            "sharpness": float(100 + sample_idx),
            "visual_stddev": 32.0,
            "visual_edge_score": 32.0,
            "visual_entropy": 3.0,
            "visual_unique_buckets": 8.0,
        }
        for sample_idx, state_id in enumerate(state_ids)
    ]
    metrics = build_compact_frame_metric_table(
        rows,
        timestamps=timestamps,
        frame_indices=frame_indices,
        content_prev_delta=prev,
        content_next_delta=next_delta,
        content_signature_stack=signatures,
    )
    return timestamps, frame_indices, dhashes, embeddings, metrics


def _select(
    samples_per_state: list[int],
    *,
    coverage_interval_seconds: float,
    max_primary_candidates: int,
    state_dhashes: list[int] | None = None,
):
    timestamps, frame_indices, dhashes, embeddings, metrics = _sample_series(
        samples_per_state,
        state_dhashes=state_dhashes,
    )
    duration_seconds = len(timestamps) * 0.5
    sample_scenes = {
        sample_idx: int(timestamp // coverage_interval_seconds)
        for sample_idx, timestamp in enumerate(timestamps)
    }
    coverage_pool = _coverage_candidate_pool(
        timestamps=timestamps,
        frame_indices=frame_indices,
        dhashes=dhashes,
        clip_embeddings=embeddings,
        frame_metrics=metrics,
        coverage_interval_seconds=coverage_interval_seconds,
        minimum_settled_dwell_seconds=2.0,
        duration_seconds=duration_seconds,
        sample_scenes=sample_scenes,
    )
    result = select_primary_candidates(
        semantic_candidates=(),
        coverage_pool=coverage_pool,
        timestamps=timestamps,
        frame_indices=frame_indices,
        dhashes=dhashes,
        clip_embeddings=embeddings,
        frame_metrics=metrics,
        sample_scenes=sample_scenes,
        coverage_interval_seconds=coverage_interval_seconds,
        minimum_settled_dwell_seconds=2.0,
        duration_seconds=duration_seconds,
        max_primary_candidates=max_primary_candidates,
    )
    return coverage_pool, result


def test_five_durable_states_in_one_window_fill_four_unused_primary_slots():
    coverage_pool, result = _select(
        [6, 6, 6, 6, 6],
        coverage_interval_seconds=90.0,
        max_primary_candidates=96,
    )

    assert len(coverage_pool) == 1
    assert len(result.candidates) == 5
    assert len(result.durable_state_fill) == 4
    assert {
        (candidate.selection.selection_role, candidate.selection.selection_reason)
        for candidate in result.durable_state_fill
    } == {("durable_state", "unrepresented_durable_state")}
    assert result.metadata["unique_base_primary_count"] == 1
    assert result.metadata["remaining_primary_capacity"] == 95
    assert result.metadata["durable_occurrence_count"] == 5
    assert result.metadata["durable_group_count"] == 5
    assert result.metadata["eligible_durable_group_count"] == 4
    assert result.metadata["durable_fill_count"] == 4
    assert result.metadata["final_primary_count"] == 5
    assert result.metadata["unused_primary_capacity"] == 91
    assert len(result.candidates) <= 96


def test_six_bit_dhash_change_enters_durable_fill_as_a_new_state():
    coverage_pool, result = _select(
        [6, 6],
        coverage_interval_seconds=90.0,
        max_primary_candidates=96,
        state_dhashes=[0, 0b111111],
    )

    assert len(coverage_pool) == 1
    assert result.metadata["durable_occurrence_count"] == 2
    assert result.metadata["durable_group_count"] == 2
    assert result.metadata["eligible_durable_group_count"] == 1
    assert result.metadata["durable_fill_count"] == 1
    assert result.metadata["final_primary_count"] == 2


def test_secondary_fill_balances_windows_before_returning_to_dense_window():
    _coverage_pool, result = _select(
        [5, 5, 5, 5, 10, 10, 10, 10],
        coverage_interval_seconds=10.0,
        max_primary_candidates=6,
    )

    assert len(result.durable_state_fill) == 3
    assert {
        candidate.temporal.temporal_window_id
        for candidate in result.durable_state_fill
    } == {0, 1, 2}


def test_secondary_fill_allocates_extra_slots_to_a_denser_window():
    _coverage_pool, result = _select(
        [6] * 9,
        coverage_interval_seconds=18.0,
        max_primary_candidates=6,
    )

    fill_count_by_window = {
        window_id: sum(
            candidate.temporal.temporal_window_id == window_id
            for candidate in result.durable_state_fill
        )
        for window_id in (0, 1)
    }
    assert fill_count_by_window == {0: 3, 1: 1}


def test_secondary_fill_prefers_a_multi_state_scene_over_an_isolated_change():
    timestamps, frame_indices, dhashes, embeddings, metrics = _sample_series([6] * 6)
    duration_seconds = len(timestamps) * 0.5
    sample_scenes = {
        sample_idx: (0 if 6 <= sample_idx < 12 else 1)
        for sample_idx in range(len(timestamps))
    }
    coverage_pool = _coverage_candidate_pool(
        timestamps=timestamps,
        frame_indices=frame_indices,
        dhashes=dhashes,
        clip_embeddings=embeddings,
        frame_metrics=metrics,
        coverage_interval_seconds=90.0,
        minimum_settled_dwell_seconds=2.0,
        duration_seconds=duration_seconds,
        sample_scenes=sample_scenes,
    )
    result = select_primary_candidates(
        semantic_candidates=(),
        coverage_pool=coverage_pool,
        timestamps=timestamps,
        frame_indices=frame_indices,
        dhashes=dhashes,
        clip_embeddings=embeddings,
        frame_metrics=metrics,
        sample_scenes=sample_scenes,
        coverage_interval_seconds=90.0,
        minimum_settled_dwell_seconds=2.0,
        duration_seconds=duration_seconds,
        max_primary_candidates=5,
    )

    assert len(result.durable_state_fill) == 4
    assert all(
        not 6 <= candidate.sample_idx < 12
        for candidate in result.durable_state_fill
    )


def test_secondary_fill_completes_dense_scene_after_each_window_gets_one_fill():
    timestamps, frame_indices, dhashes, embeddings, metrics = _sample_series([8] * 9)
    duration_seconds = len(timestamps) * 0.5
    sample_scenes = {
        sample_idx: (
            0
            if sample_idx < 40
            else 1 + ((sample_idx - 40) // 8)
        )
        for sample_idx in range(len(timestamps))
    }
    coverage_pool = _coverage_candidate_pool(
        timestamps=timestamps,
        frame_indices=frame_indices,
        dhashes=dhashes,
        clip_embeddings=embeddings,
        frame_metrics=metrics,
        coverage_interval_seconds=20.0,
        minimum_settled_dwell_seconds=2.0,
        duration_seconds=duration_seconds,
        sample_scenes=sample_scenes,
    )
    result = select_primary_candidates(
        semantic_candidates=(),
        coverage_pool=coverage_pool,
        timestamps=timestamps,
        frame_indices=frame_indices,
        dhashes=dhashes,
        clip_embeddings=embeddings,
        frame_metrics=metrics,
        sample_scenes=sample_scenes,
        coverage_interval_seconds=20.0,
        minimum_settled_dwell_seconds=2.0,
        duration_seconds=duration_seconds,
        max_primary_candidates=6,
    )

    fill_count_by_window = {
        window_id: sum(
            candidate.temporal.temporal_window_id == window_id
            for candidate in result.durable_state_fill
        )
        for window_id in (0, 1)
    }
    assert fill_count_by_window == {0: 3, 1: 1}


def test_repeated_occurrences_aggregate_only_with_timestamp_duration_and_local_context():
    timestamps = [0.0, 0.55, 1.1, 1.6, 2.15]
    frame_indices = list(range(len(timestamps)))
    state_ids = [0, 0, 1, 0, 0]
    dhashes = [0xFF << (8 * state_id) for state_id in state_ids]
    embeddings = np.eye(2, dtype=np.float32)[state_ids]
    signatures = np.stack(
        [np.full((4, 4), state_id * 32, dtype=np.uint8) for state_id in state_ids]
    )
    metrics = build_compact_frame_metric_table(
        [{"sharpness": 10.0 + index} for index in range(len(timestamps))],
        timestamps=timestamps,
        frame_indices=frame_indices,
        content_prev_delta=[0.0, 0.0, 32.0, 32.0, 0.0],
        content_next_delta=[0.0, 32.0, 32.0, 0.0, 0.0],
        content_signature_stack=signatures,
    )
    occurrences = build_visual_state_occurrences(
        timestamps=timestamps,
        frame_indices=frame_indices,
        dhashes=dhashes,
        clip_embeddings=embeddings,
        frame_metrics=metrics,
        coverage_interval_seconds=90.0,
        duration_seconds=2.7,
        sample_scenes={index: 0 for index in range(len(timestamps))},
        dwell_ids=np.asarray([0, 0, 1, 2, 2]),
    )
    groups = build_durable_state_groups(
        occurrences,
        timestamps=timestamps,
        frame_indices=frame_indices,
        dhashes=dhashes,
        clip_embeddings=embeddings,
        frame_metrics=metrics,
    )

    assert len(groups) == 1
    assert groups[0].direct_durable is False
    assert groups[0].aggregate_duration_seconds == pytest.approx(2.2)
    assert len(groups[0].occurrences) == 2

    short_timestamps = [0.0, 0.4, 0.8, 1.2, 1.6]
    short_occurrences = build_visual_state_occurrences(
        timestamps=short_timestamps,
        frame_indices=frame_indices,
        dhashes=dhashes,
        clip_embeddings=embeddings,
        frame_metrics=None,
        coverage_interval_seconds=90.0,
        duration_seconds=2.0,
        dwell_ids=[0, 0, 1, 2, 2],
    )
    short_groups = build_durable_state_groups(
        short_occurrences,
        timestamps=short_timestamps,
        frame_indices=frame_indices,
        dhashes=dhashes,
        clip_embeddings=embeddings,
        frame_metrics=metrics,
    )
    assert short_groups == ()


def test_repeated_short_occurrences_do_not_aggregate_across_distant_windows():
    timestamps = [0.0, 0.55, 1.1, 100.0, 100.55]
    frame_indices = list(range(len(timestamps)))
    state_ids = [0, 0, 1, 0, 0]
    dhashes = [0xFF << (8 * state_id) for state_id in state_ids]
    embeddings = np.eye(2, dtype=np.float32)[state_ids]
    signatures = np.stack(
        [np.full((4, 4), state_id * 32, dtype=np.uint8) for state_id in state_ids]
    )
    metrics = build_compact_frame_metric_table(
        [{"sharpness": 10.0} for _ in timestamps],
        timestamps=timestamps,
        frame_indices=frame_indices,
        content_prev_delta=[0.0, 0.0, 32.0, 32.0, 0.0],
        content_next_delta=[0.0, 32.0, 32.0, 0.0, 0.0],
        content_signature_stack=signatures,
    )
    occurrences = build_visual_state_occurrences(
        timestamps=timestamps,
        frame_indices=frame_indices,
        dhashes=dhashes,
        clip_embeddings=embeddings,
        frame_metrics=metrics,
        coverage_interval_seconds=30.0,
        duration_seconds=101.1,
        sample_scenes={0: 0, 1: 0, 2: 1, 3: 2, 4: 2},
        dwell_ids=[0, 0, 1, 2, 2],
    )
    groups = build_durable_state_groups(
        occurrences,
        timestamps=timestamps,
        frame_indices=frame_indices,
        dhashes=dhashes,
        clip_embeddings=embeddings,
        frame_metrics=metrics,
    )

    assert all(len(group.occurrences) == 1 for group in groups)


def test_coverage_candidates_from_one_long_dwell_share_one_primary_slot():
    coverage_pool, result = _select(
        [360],
        coverage_interval_seconds=90.0,
        max_primary_candidates=96,
    )

    assert len(coverage_pool) == 2
    assert len(result.candidates) == 1
    assert result.candidates[0].temporal.coverage_window_ids == (0, 1)
    assert result.metadata["unique_base_primary_count"] == 1
    assert result.metadata["durable_state_fill_count"] == 0
