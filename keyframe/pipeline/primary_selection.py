from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from keyframe.pipeline.contracts import CandidateRecord
from keyframe.visual import (
    FrameMetricTable,
    dhash_hamming_distance,
    material_visual_difference,
    strong_visual_equivalence,
    timestamp_interval_duration_seconds,
)


# Material dHash differences start at six bits; durable dwells must split there.
_DURABLE_DWELL_HAMMING_THRESHOLD = 5


@dataclass(frozen=True)
class VisualStateOccurrence:
    occurrence_id: int
    scene_id: int | None
    coverage_window_ids: tuple[int, ...]
    sample_indices: tuple[int, ...]
    start_seconds: float
    end_seconds: float
    representative_sample_idx: int

    @property
    def duration_seconds(self) -> float:
        return timestamp_interval_duration_seconds(self.start_seconds, self.end_seconds)


@dataclass(frozen=True)
class DurableStateGroup:
    group_id: int
    occurrences: tuple[VisualStateOccurrence, ...]
    coverage_window_ids: tuple[int, ...]
    sample_indices: tuple[int, ...]
    start_seconds: float
    end_seconds: float
    aggregate_duration_seconds: float
    representative_sample_idx: int
    direct_durable: bool

    @property
    def qualifying_repeated_occurrence_count(self) -> int:
        return len(self.occurrences) if len(self.occurrences) > 1 else 0


@dataclass(frozen=True)
class _MaterialComparison:
    is_material: bool
    distance_score: float
    central_delta: float | None
    dhash_distance: int | None
    clip_distance: float | None


@dataclass(frozen=True)
class PrimarySelectionResult:
    candidates: tuple[CandidateRecord, ...]
    durable_state_fill: tuple[CandidateRecord, ...]
    metadata: dict[str, Any]
    dwell_ids: tuple[int, ...]


def _coverage_window_bounds(
    duration_seconds: float,
    interval_seconds: float,
) -> list[tuple[int, float, float]]:
    if duration_seconds <= 0:
        return []
    count = max(1, int(math.ceil(duration_seconds / interval_seconds)))
    windows = []
    for window_id in range(count):
        start = window_id * interval_seconds
        end = duration_seconds if window_id == count - 1 else (window_id + 1) * interval_seconds
        if end > start:
            windows.append((window_id, float(start), float(end)))
    return windows


def _interval_end_seconds(
    timestamps: Sequence[float],
    end_idx: int,
    duration_seconds: float,
) -> float:
    next_idx = int(end_idx) + 1
    if 0 <= next_idx < len(timestamps):
        return min(float(duration_seconds), float(timestamps[next_idx]))
    return float(duration_seconds)


def _window_ids_for_interval(
    bounds: Sequence[tuple[int, float, float]],
    start_seconds: float,
    end_seconds: float,
) -> tuple[int, ...]:
    ids = []
    for window_id, window_start, window_end in bounds:
        overlap_start = max(float(start_seconds), float(window_start))
        overlap_end = min(float(end_seconds), float(window_end))
        if overlap_end > overlap_start:
            ids.append(int(window_id))
    return tuple(ids)


def _window_id_for_timestamp(
    bounds: Sequence[tuple[int, float, float]],
    timestamp: float,
) -> int | None:
    if not bounds:
        return None
    ts = float(timestamp)
    for window_id, start, end in bounds:
        if ts >= start and (ts < end or window_id == bounds[-1][0]):
            return int(window_id)
    return int(bounds[-1][0])


def _embedding_vector(clip_embeddings: Any | None, sample_idx: int) -> np.ndarray | None:
    if clip_embeddings is None:
        return None
    try:
        vector = np.asarray(clip_embeddings[int(sample_idx)], dtype=np.float32)
    except Exception:
        return None
    if vector.ndim != 1 or vector.size == 0:
        return None
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        return None
    return vector / norm


def _clip_distance(clip_embeddings: Any | None, sample_idx_a: int, sample_idx_b: int) -> float | None:
    left = _embedding_vector(clip_embeddings, sample_idx_a)
    right = _embedding_vector(clip_embeddings, sample_idx_b)
    if left is None or right is None:
        return None
    return float(1.0 - np.dot(left, right))


def _representative_sample_idx(
    sample_indices: Sequence[int],
    *,
    clip_embeddings: Any | None,
    timestamps: Sequence[float],
    frame_indices: Sequence[int],
    frame_metrics: FrameMetricTable | None,
    midpoint_seconds: float | None = None,
) -> int:
    indices = tuple(int(idx) for idx in sample_indices)
    if not indices:
        raise ValueError("representative selection requires at least one sample")
    midpoint = (
        float(midpoint_seconds)
        if midpoint_seconds is not None
        else (float(timestamps[indices[0]]) + float(timestamps[indices[-1]])) / 2.0
    )
    vectors = [
        _embedding_vector(clip_embeddings, idx)
        for idx in indices
    ]
    valid_vectors = [vector for vector in vectors if vector is not None]
    centroid: np.ndarray | None = None
    if valid_vectors:
        centroid = np.mean(np.stack(valid_vectors, axis=0), axis=0)
        norm = float(np.linalg.norm(centroid))
        centroid = centroid / norm if norm > 0 else None

    def rank(idx: int) -> tuple[float, float, float, int, int]:
        vector = _embedding_vector(clip_embeddings, idx)
        similarity = float(np.dot(vector, centroid)) if vector is not None and centroid is not None else 0.0
        sharpness = (
            float(frame_metrics.sharpness_for(idx) or 0.0)
            if frame_metrics is not None
            else 0.0
        )
        frame_idx = int(frame_indices[idx]) if 0 <= idx < len(frame_indices) else idx
        return (
            similarity,
            -abs(float(timestamps[idx]) - midpoint),
            sharpness,
            -frame_idx,
            -idx,
        )

    return max(indices, key=rank)


def _dhash_for(dhashes: Sequence[int] | Mapping[int, int], sample_idx: int) -> int | None:
    idx = int(sample_idx)
    try:
        if isinstance(dhashes, Mapping):
            value = dhashes.get(idx)
        else:
            value = dhashes[idx] if 0 <= idx < len(dhashes) else None
    except Exception:
        return None
    return int(value) if value is not None else None


def _central_delta(
    frame_metrics: FrameMetricTable | None,
    sample_idx_a: int,
    sample_idx_b: int,
) -> float | None:
    if frame_metrics is None:
        return None
    return frame_metrics.content_delta_between(sample_idx_a, sample_idx_b)


def _samples_strongly_equivalent(
    sample_idx_a: int,
    sample_idx_b: int,
    *,
    dhashes: Sequence[int] | Mapping[int, int],
    frame_metrics: FrameMetricTable | None,
) -> bool:
    if int(sample_idx_a) == int(sample_idx_b):
        return True
    return strong_visual_equivalence(
        dhash_a=_dhash_for(dhashes, sample_idx_a),
        dhash_b=_dhash_for(dhashes, sample_idx_b),
        central_delta=_central_delta(frame_metrics, sample_idx_a, sample_idx_b),
    )


def _material_comparison(
    sample_idx_a: int,
    sample_idx_b: int,
    *,
    dhashes: Sequence[int] | Mapping[int, int],
    clip_embeddings: Any | None,
    frame_metrics: FrameMetricTable | None,
) -> _MaterialComparison:
    left_hash = _dhash_for(dhashes, sample_idx_a)
    right_hash = _dhash_for(dhashes, sample_idx_b)
    dhash_distance = (
        dhash_hamming_distance(left_hash, right_hash)
        if left_hash is not None and right_hash is not None
        else None
    )
    central_delta = _central_delta(frame_metrics, sample_idx_a, sample_idx_b)
    clip_distance = _clip_distance(clip_embeddings, sample_idx_a, sample_idx_b)
    is_material = material_visual_difference(
        central_delta=central_delta,
        dhash_distance=dhash_distance,
        clip_distance=clip_distance,
    )
    components = []
    if central_delta is not None:
        components.append(float(central_delta) / 2.5)
    if dhash_distance is not None:
        components.append(float(dhash_distance) / 6.0)
    if clip_distance is not None:
        components.append(float(clip_distance) / 0.08)
    return _MaterialComparison(
        is_material=is_material,
        distance_score=max(components, default=0.0),
        central_delta=central_delta,
        dhash_distance=dhash_distance,
        clip_distance=clip_distance,
    )


def build_visual_state_occurrences(
    *,
    timestamps: Sequence[float],
    frame_indices: Sequence[int],
    dhashes: Sequence[int] | Mapping[int, int],
    clip_embeddings: Any | None,
    frame_metrics: FrameMetricTable | None,
    coverage_interval_seconds: float,
    duration_seconds: float,
    sample_scenes: Mapping[int, int] | None = None,
    dwell_ids: Sequence[int] | None = None,
) -> tuple[VisualStateOccurrence, ...]:
    from keyframe.scoring import assign_dwell_ids

    if not timestamps:
        return ()
    source_dwell_ids = (
        assign_dwell_ids(dhashes, hamming_threshold=_DURABLE_DWELL_HAMMING_THRESHOLD)
        if dwell_ids is None
        else dwell_ids
    )
    dwell_ids = tuple(int(value) for value in source_dwell_ids)
    if len(dwell_ids) != len(timestamps):
        raise ValueError("dwell id count must match timestamp count")
    bounds = _coverage_window_bounds(duration_seconds, coverage_interval_seconds)
    occurrences: list[VisualStateOccurrence] = []
    start_idx = 0

    def append_occurrence(end_idx: int) -> None:
        sample_indices = tuple(range(start_idx, int(end_idx) + 1))
        start_seconds = float(timestamps[start_idx])
        end_seconds = _interval_end_seconds(timestamps, int(end_idx), duration_seconds)
        scene_values = {
            int(sample_scenes[idx])
            for idx in sample_indices
            if sample_scenes is not None and idx in sample_scenes and sample_scenes[idx] is not None
        }
        scene_id = next(iter(scene_values)) if len(scene_values) == 1 else None
        midpoint = (start_seconds + end_seconds) / 2.0
        representative = _representative_sample_idx(
            sample_indices,
            clip_embeddings=clip_embeddings,
            timestamps=timestamps,
            frame_indices=frame_indices,
            frame_metrics=frame_metrics,
            midpoint_seconds=midpoint,
        )
        occurrences.append(
            VisualStateOccurrence(
                occurrence_id=int(dwell_ids[start_idx]),
                scene_id=scene_id,
                coverage_window_ids=_window_ids_for_interval(bounds, start_seconds, end_seconds),
                sample_indices=sample_indices,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                representative_sample_idx=representative,
            )
        )

    for idx in range(1, len(dwell_ids)):
        if int(dwell_ids[idx]) == int(dwell_ids[start_idx]):
            continue
        append_occurrence(idx - 1)
        start_idx = idx
    append_occurrence(len(dwell_ids) - 1)
    return tuple(occurrences)


def _occurrences_have_local_context(
    left: VisualStateOccurrence,
    right: VisualStateOccurrence,
    *,
    max_gap_seconds: float = 30.0,
) -> bool:
    if left.scene_id is not None and left.scene_id == right.scene_id:
        return True
    if set(left.coverage_window_ids) & set(right.coverage_window_ids):
        return True
    adjacent_windows = any(
        abs(int(a) - int(b)) == 1
        for a in left.coverage_window_ids
        for b in right.coverage_window_ids
    )
    if not adjacent_windows:
        return False
    gap = max(
        0.0,
        max(float(left.start_seconds), float(right.start_seconds))
        - min(float(left.end_seconds), float(right.end_seconds)),
    )
    return gap <= float(max_gap_seconds)


def _occurrences_strongly_equivalent(
    left: VisualStateOccurrence,
    right: VisualStateOccurrence,
    *,
    dhashes: Sequence[int] | Mapping[int, int],
    frame_metrics: FrameMetricTable | None,
) -> bool:
    return _samples_strongly_equivalent(
        left.representative_sample_idx,
        right.representative_sample_idx,
        dhashes=dhashes,
        frame_metrics=frame_metrics,
    )


def build_durable_state_groups(
    occurrences: Sequence[VisualStateOccurrence],
    *,
    timestamps: Sequence[float],
    frame_indices: Sequence[int],
    dhashes: Sequence[int] | Mapping[int, int],
    clip_embeddings: Any | None,
    frame_metrics: FrameMetricTable | None,
    minimum_settled_dwell_seconds: float = 2.0,
    aggregate_occurrence_min_seconds: float = 1.0,
) -> tuple[DurableStateGroup, ...]:
    groupable = [
        occurrence
        for occurrence in sorted(occurrences, key=lambda item: (item.start_seconds, item.occurrence_id))
        if occurrence.duration_seconds >= aggregate_occurrence_min_seconds
    ]
    used: set[int] = set()
    groups: list[DurableStateGroup] = []

    for occurrence in groupable:
        if occurrence.occurrence_id in used:
            continue
        members = [occurrence]
        used.add(occurrence.occurrence_id)
        for other in groupable:
            if other.occurrence_id in used:
                continue
            if not all(
                _occurrences_have_local_context(other, member)
                and _occurrences_strongly_equivalent(
                    other,
                    member,
                    dhashes=dhashes,
                    frame_metrics=frame_metrics,
                )
                for member in members
            ):
                continue
            members.append(other)
            used.add(other.occurrence_id)

        aggregate_duration = sum(member.duration_seconds for member in members)
        direct = any(member.duration_seconds >= minimum_settled_dwell_seconds for member in members)
        stable_aggregate = (
            len(members) > 1
            and all(member.duration_seconds >= aggregate_occurrence_min_seconds for member in members)
            and aggregate_duration >= minimum_settled_dwell_seconds
        )
        if not (direct or stable_aggregate):
            continue

        sample_indices = tuple(
            idx
            for member in members
            for idx in member.sample_indices
        )
        start_seconds = min(member.start_seconds for member in members)
        end_seconds = max(member.end_seconds for member in members)
        representative = _representative_sample_idx(
            sample_indices,
            clip_embeddings=clip_embeddings,
            timestamps=timestamps,
            frame_indices=frame_indices,
            frame_metrics=frame_metrics,
            midpoint_seconds=(start_seconds + end_seconds) / 2.0,
        )
        groups.append(
            DurableStateGroup(
                group_id=len(groups),
                occurrences=tuple(sorted(members, key=lambda item: item.start_seconds)),
                coverage_window_ids=tuple(
                    sorted({window for member in members for window in member.coverage_window_ids})
                ),
                sample_indices=tuple(sorted(sample_indices)),
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                aggregate_duration_seconds=aggregate_duration,
                representative_sample_idx=representative,
                direct_durable=direct,
            )
        )

    return tuple(sorted(groups, key=lambda item: (item.start_seconds, item.representative_sample_idx)))


def _candidate_with_sample_context(
    candidate: CandidateRecord,
    *,
    dwell_ids: Sequence[int],
    sample_scenes: Mapping[int, int] | None,
    dhashes: Sequence[int] | Mapping[int, int],
) -> CandidateRecord:
    sample_idx = int(candidate.sample_idx)
    updates: dict[str, Any] = {}
    if candidate.temporal.dwell_id is None and 0 <= sample_idx < len(dwell_ids):
        updates["dwell_id"] = int(dwell_ids[sample_idx])
    if candidate.temporal.scene_id is None and sample_scenes is not None and sample_idx in sample_scenes:
        updates["scene_id"] = int(sample_scenes[sample_idx])
    if updates:
        candidate = candidate.with_temporal(**updates)
    dhash = _dhash_for(dhashes, sample_idx)
    if dhash is not None and candidate.visual.dhash is None:
        candidate = candidate.with_visual(dhash=dhash, dhash_hex=f"{dhash:016x}")
    return candidate


def _with_coverage_lineage(
    candidate: CandidateRecord,
    coverage: CandidateRecord,
) -> CandidateRecord:
    window_ids = tuple(
        sorted(
            {
                *(int(value) for value in candidate.temporal.coverage_window_ids),
                *(int(value) for value in coverage.temporal.coverage_window_ids),
            }
        )
    )
    roles = set(candidate.lineage.lineage_roles)
    roles.add("coverage")
    if candidate.selection.selection_role:
        roles.add(str(candidate.selection.selection_role))
    if coverage.selection.selection_role:
        roles.add(str(coverage.selection.selection_role))
    merged_from_sample_idxs = tuple(
        sorted(
            {
                *(int(idx) for idx in candidate.lineage.merged_from_sample_idxs),
                *(int(idx) for idx in coverage.lineage.merged_from_sample_idxs),
            }
        )
    )
    merged_timestamps = tuple(
        sorted(
            {
                *(float(ts) for ts in candidate.lineage.merged_timestamps),
                *(float(ts) for ts in coverage.lineage.merged_timestamps),
            }
        )
    )
    return candidate.with_temporal(
        coverage_window_ids=window_ids,
        temporal_window_id=(
            candidate.temporal.temporal_window_id
            if candidate.temporal.temporal_window_id is not None
            else coverage.temporal.temporal_window_id
        ),
        temporal_window_seconds=(
            candidate.temporal.temporal_window_seconds
            if candidate.temporal.temporal_window_seconds is not None
            else coverage.temporal.temporal_window_seconds
        ),
        dwell_id=(
            candidate.temporal.dwell_id
            if candidate.temporal.dwell_id is not None
            else coverage.temporal.dwell_id
        ),
    ).with_lineage(
        merged_from_sample_idxs=merged_from_sample_idxs,
        merged_timestamps=merged_timestamps,
        lineage_roles=tuple(sorted(roles)),
    )


def _distributed_window_subset(
    candidates: list[CandidateRecord],
    capacity: int,
) -> list[CandidateRecord]:
    if capacity <= 0:
        return []
    if len(candidates) <= capacity:
        return candidates
    if capacity == 1:
        return [candidates[len(candidates) // 2]]
    last = len(candidates) - 1
    selected_positions = {
        round(index * last / (capacity - 1))
        for index in range(capacity)
    }
    return [candidate for index, candidate in enumerate(candidates) if index in selected_positions]


def _coverage_candidate_pool(
    *,
    timestamps: Sequence[float],
    frame_indices: Sequence[int],
    dhashes: Sequence[int] | Mapping[int, int],
    clip_embeddings: Any,
    frame_metrics: FrameMetricTable | None,
    coverage_interval_seconds: float,
    minimum_settled_dwell_seconds: float,
    duration_seconds: float,
    sample_scenes: Mapping[int, int] | None = None,
) -> list[CandidateRecord]:
    from keyframe.scoring import assign_dwell_ids

    if not timestamps:
        return []
    dwell_ids = assign_dwell_ids(
        dhashes,
        hamming_threshold=_DURABLE_DWELL_HAMMING_THRESHOLD,
    )
    occurrences = build_visual_state_occurrences(
        timestamps=timestamps,
        frame_indices=frame_indices,
        dhashes=dhashes,
        clip_embeddings=clip_embeddings,
        frame_metrics=frame_metrics,
        coverage_interval_seconds=coverage_interval_seconds,
        duration_seconds=duration_seconds,
        sample_scenes=sample_scenes,
        dwell_ids=dwell_ids,
    )
    dwell_ranges = {occurrence.occurrence_id: occurrence for occurrence in occurrences}
    bounds = _coverage_window_bounds(duration_seconds, coverage_interval_seconds)

    candidates: list[CandidateRecord] = []
    for window_id, window_start, window_end in bounds:
        ranked_dwells = []
        for dwell_id, occurrence in dwell_ranges.items():
            overlap_start = max(window_start, float(occurrence.start_seconds))
            overlap_end = min(window_end, float(occurrence.end_seconds))
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap < minimum_settled_dwell_seconds:
                continue
            ranked_dwells.append(
                (
                    overlap,
                    float(occurrence.duration_seconds),
                    -float(occurrence.start_seconds),
                    int(dwell_id),
                    overlap_start,
                    overlap_end,
                    occurrence,
                )
            )
        if not ranked_dwells:
            continue
        ranked_dwells.sort(reverse=True)
        _overlap, _duration, _neg_start, dwell_id, overlap_start, overlap_end, occurrence = ranked_dwells[0]
        overlap_indices = [
            idx
            for idx in occurrence.sample_indices
            if float(timestamps[idx]) >= overlap_start
            and (
                float(timestamps[idx]) < overlap_end
                or (window_end == duration_seconds and float(timestamps[idx]) <= overlap_end)
            )
        ]
        if not overlap_indices:
            midpoint = (overlap_start + overlap_end) / 2.0
            overlap_indices = [
                min(
                    occurrence.sample_indices,
                    key=lambda idx: abs(float(timestamps[idx]) - midpoint),
                )
            ]
        sample_idx = _representative_sample_idx(
            overlap_indices,
            clip_embeddings=clip_embeddings,
            timestamps=timestamps,
            frame_indices=frame_indices,
            frame_metrics=frame_metrics,
            midpoint_seconds=(overlap_start + overlap_end) / 2.0,
        )
        sharpness = (
            float(frame_metrics.sharpness_for(sample_idx))
            if frame_metrics is not None
            else None
        )
        dhash = _dhash_for(dhashes, sample_idx)
        candidates.append(
            CandidateRecord(
                sample_idx=sample_idx,
                frame_idx=int(frame_indices[sample_idx]),
                timestamp=float(timestamps[sample_idx]),
            )
            .with_visual(
                cluster_role="coverage",
                sharpness=sharpness,
                dhash=dhash,
                dhash_hex=f"{int(dhash):016x}" if dhash is not None else None,
            )
            .with_temporal(
                scene_id=occurrence.scene_id,
                dwell_id=int(dwell_id),
                temporal_window_id=int(window_id),
                temporal_window_seconds=float(coverage_interval_seconds),
                coverage_window_ids=(int(window_id),),
            )
            .with_selection(
                proposal_lane="coverage",
                selection_role="coverage",
                selection_reason="duration_window",
                candidate_score=sharpness,
            )
            .with_lineage(lineage_roles=("coverage",))
        )
    return candidates


def _consolidate_coverage_pool_by_dwell(
    coverage_pool: Sequence[CandidateRecord],
) -> tuple[CandidateRecord, ...]:
    by_key: dict[tuple[str, int], CandidateRecord] = {}
    ordered_keys: list[tuple[str, int]] = []
    for coverage in coverage_pool:
        if coverage.temporal.dwell_id is None:
            key = ("sample", int(coverage.sample_idx))
        else:
            key = ("dwell", int(coverage.temporal.dwell_id))
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = coverage
            ordered_keys.append(key)
            continue
        by_key[key] = _with_coverage_lineage(existing, coverage)
    return tuple(by_key[key] for key in ordered_keys)


def _combine_primary_lanes(
    semantic_candidates: Sequence[CandidateRecord],
    coverage_pool: Sequence[CandidateRecord],
    *,
    max_primary_candidates: int,
) -> tuple[CandidateRecord, ...]:
    selected_by_sample: dict[int, CandidateRecord] = {}
    selected_sample_by_dwell: dict[int, int] = {}

    def remember(candidate: CandidateRecord) -> None:
        selected_by_sample[int(candidate.sample_idx)] = candidate
        if candidate.temporal.dwell_id is not None:
            selected_sample_by_dwell[int(candidate.temporal.dwell_id)] = int(candidate.sample_idx)

    for candidate in semantic_candidates:
        roles = set(candidate.lineage.lineage_roles)
        roles.add("semantic")
        candidate = candidate.with_selection(
            selection_role=candidate.selection.selection_role or "semantic",
            selection_reason=candidate.selection.selection_reason or "semantic_cluster",
            proposal_lane=candidate.selection.proposal_lane or "semantic",
        ).with_lineage(lineage_roles=tuple(sorted(roles)))
        existing = selected_by_sample.get(int(candidate.sample_idx))
        if existing is not None:
            remember(existing)
            continue
        remember(candidate)

    coverage_capacity = max(0, int(max_primary_candidates) - len(selected_by_sample))
    added_coverage = 0
    for coverage in coverage_pool:
        existing = selected_by_sample.get(int(coverage.sample_idx))
        if existing is not None:
            remember(_with_coverage_lineage(existing, coverage))
            continue
        if coverage.temporal.dwell_id is not None:
            existing_sample = selected_sample_by_dwell.get(int(coverage.temporal.dwell_id))
            if existing_sample is not None and existing_sample in selected_by_sample:
                remember(_with_coverage_lineage(selected_by_sample[existing_sample], coverage))
                continue
        if added_coverage >= coverage_capacity:
            continue
        remember(coverage)
        added_coverage += 1
        if len(selected_by_sample) >= max_primary_candidates:
            break
    return tuple(sorted(selected_by_sample.values(), key=lambda c: (float(c.timestamp), int(c.sample_idx))))


def _group_occurrence_sample_idxs(group: DurableStateGroup) -> tuple[int, ...]:
    return tuple(int(occurrence.representative_sample_idx) for occurrence in group.occurrences)


def _group_occurrence_timestamps(group: DurableStateGroup) -> tuple[float, ...]:
    return tuple(float(occurrence.start_seconds) for occurrence in group.occurrences)


def _annotate_candidate_with_durable_group(
    candidate: CandidateRecord,
    group: DurableStateGroup,
) -> CandidateRecord:
    roles = set(candidate.lineage.lineage_roles)
    roles.add("durable_state")
    if candidate.selection.selection_role:
        roles.add(str(candidate.selection.selection_role))
    merged_sample_idxs = tuple(
        sorted(
            {
                *(int(idx) for idx in candidate.lineage.merged_from_sample_idxs),
                *_group_occurrence_sample_idxs(group),
            }
        )
    )
    merged_timestamps = tuple(
        sorted(
            {
                *(float(ts) for ts in candidate.lineage.merged_timestamps),
                *_group_occurrence_timestamps(group),
            }
        )
    )
    return candidate.with_temporal(
        coverage_window_ids=tuple(
            sorted(
                {
                    *(int(value) for value in candidate.temporal.coverage_window_ids),
                    *(int(value) for value in group.coverage_window_ids),
                }
            )
        ),
        durable_state_group_id=int(group.group_id),
    ).with_lineage(
        merged_from_sample_idxs=merged_sample_idxs,
        merged_timestamps=merged_timestamps,
        lineage_roles=tuple(sorted(roles)),
    )


def _candidate_represents_group(
    candidate: CandidateRecord,
    group: DurableStateGroup,
    *,
    dhashes: Sequence[int] | Mapping[int, int],
    frame_metrics: FrameMetricTable | None,
    group_sample_indices: frozenset[int] | None = None,
) -> bool:
    sample_indices = group_sample_indices or frozenset(group.sample_indices)
    if int(candidate.sample_idx) in sample_indices:
        return True
    return _samples_strongly_equivalent(
        candidate.sample_idx,
        group.representative_sample_idx,
        dhashes=dhashes,
        frame_metrics=frame_metrics,
    )


def _nearest_retained_comparison(
    group: DurableStateGroup,
    retained: Sequence[CandidateRecord],
    *,
    dhashes: Sequence[int] | Mapping[int, int],
    clip_embeddings: Any | None,
    frame_metrics: FrameMetricTable | None,
) -> _MaterialComparison:
    comparisons = [
        _material_comparison(
            group.representative_sample_idx,
            candidate.sample_idx,
            dhashes=dhashes,
            clip_embeddings=clip_embeddings,
            frame_metrics=frame_metrics,
        )
        for candidate in retained
    ]
    if not comparisons:
        return _MaterialComparison(True, float("inf"), None, None, None)
    return min(comparisons, key=lambda item: item.distance_score)


def _group_primary_window_id(
    group: DurableStateGroup,
    *,
    timestamps: Sequence[float],
    coverage_bounds: Sequence[tuple[int, float, float]],
) -> int:
    window_id = _window_id_for_timestamp(
        coverage_bounds,
        float(timestamps[group.representative_sample_idx]),
    )
    if window_id is not None:
        return int(window_id)
    return int(group.coverage_window_ids[0]) if group.coverage_window_ids else 0


def _candidate_from_group(
    group: DurableStateGroup,
    *,
    timestamps: Sequence[float],
    frame_indices: Sequence[int],
    dhashes: Sequence[int] | Mapping[int, int],
    frame_metrics: FrameMetricTable | None,
    coverage_interval_seconds: float,
    coverage_bounds: Sequence[tuple[int, float, float]],
    comparison: _MaterialComparison,
) -> CandidateRecord:
    sample_idx = int(group.representative_sample_idx)
    occurrence = next(
        (
            occurrence
            for occurrence in group.occurrences
            if sample_idx in occurrence.sample_indices
        ),
        group.occurrences[0],
    )
    dhash = _dhash_for(dhashes, sample_idx)
    sharpness = (
        float(frame_metrics.sharpness_for(sample_idx) or 0.0)
        if frame_metrics is not None
        else None
    )
    proxy = (
        float(frame_metrics.proxy_content_score[sample_idx])
        if frame_metrics is not None and frame_metrics.has_sample(sample_idx)
        else 0.0
    )
    window_id = _group_primary_window_id(group, timestamps=timestamps, coverage_bounds=coverage_bounds)
    return (
        CandidateRecord(
            sample_idx=sample_idx,
            frame_idx=int(frame_indices[sample_idx]),
            timestamp=float(timestamps[sample_idx]),
        )
        .with_visual(
            cluster_role="durable_state",
            sharpness=sharpness,
            dhash=dhash,
            dhash_hex=f"{int(dhash):016x}" if dhash is not None else None,
        )
        .with_temporal(
            scene_id=occurrence.scene_id,
            dwell_id=int(occurrence.occurrence_id),
            temporal_window_id=window_id,
            temporal_window_seconds=float(coverage_interval_seconds),
            coverage_window_ids=group.coverage_window_ids,
            durable_state_group_id=int(group.group_id),
        )
        .with_selection(
            proposal_lane="durable_state",
            selection_role="durable_state",
            selection_reason="unrepresented_durable_state",
            candidate_score=float(comparison.distance_score),
            proxy_content_score=proxy,
            content_area_delta_score=(
                float(comparison.central_delta) / 255.0
                if comparison.central_delta is not None
                else None
            ),
        )
        .with_lineage(
            merged_from_sample_idxs=_group_occurrence_sample_idxs(group),
            merged_timestamps=_group_occurrence_timestamps(group),
            lineage_roles=("durable_state",),
        )
    )


def _fill_durable_state_capacity(
    base_candidates: Sequence[CandidateRecord],
    groups: Sequence[DurableStateGroup],
    *,
    timestamps: Sequence[float],
    frame_indices: Sequence[int],
    dhashes: Sequence[int] | Mapping[int, int],
    clip_embeddings: Any | None,
    frame_metrics: FrameMetricTable | None,
    coverage_interval_seconds: float,
    coverage_bounds: Sequence[tuple[int, float, float]],
    max_primary_candidates: int,
) -> tuple[tuple[CandidateRecord, ...], tuple[CandidateRecord, ...], int]:
    retained = list(base_candidates)
    selected_fill: list[CandidateRecord] = []
    selected_group_ids: set[int] = set()
    sample_indices_by_group = {
        int(group.group_id): frozenset(int(idx) for idx in group.sample_indices)
        for group in groups
    }

    for group in groups:
        represented_positions = [
            index
            for index, candidate in enumerate(retained)
            if _candidate_represents_group(
                candidate,
                group,
                dhashes=dhashes,
                frame_metrics=frame_metrics,
                group_sample_indices=sample_indices_by_group[int(group.group_id)],
            )
        ]
        for index in represented_positions:
            retained[index] = _annotate_candidate_with_durable_group(retained[index], group)
        if represented_positions:
            selected_group_ids.add(int(group.group_id))

    secondary_count_by_window: dict[int, int] = defaultdict(int)
    initial_eligible_group_count: int | None = None
    initial_eligible_count_by_window: dict[int, int] = defaultdict(int)
    initial_eligible_count_by_scene: dict[tuple[int, int], int] = defaultdict(int)

    def scene_key(group: DurableStateGroup, window_id: int) -> tuple[int, int]:
        scene_ids = {
            int(occurrence.scene_id)
            for occurrence in group.occurrences
            if occurrence.scene_id is not None
        }
        scene_id = next(iter(scene_ids)) if len(scene_ids) == 1 else -(int(group.group_id) + 1)
        return int(window_id), int(scene_id)

    while len(retained) < int(max_primary_candidates):
        eligible: list[tuple[DurableStateGroup, _MaterialComparison, int]] = []
        for group in groups:
            if int(group.group_id) in selected_group_ids:
                continue
            group_sample_indices = sample_indices_by_group[int(group.group_id)]
            if any(int(candidate.sample_idx) in group_sample_indices for candidate in retained):
                selected_group_ids.add(int(group.group_id))
                continue
            if any(
                _candidate_represents_group(
                    candidate,
                    group,
                    dhashes=dhashes,
                    frame_metrics=frame_metrics,
                    group_sample_indices=group_sample_indices,
                )
                for candidate in retained
            ):
                selected_group_ids.add(int(group.group_id))
                continue
            comparison = _nearest_retained_comparison(
                group,
                retained,
                dhashes=dhashes,
                clip_embeddings=clip_embeddings,
                frame_metrics=frame_metrics,
            )
            if not comparison.is_material:
                continue
            window_id = _group_primary_window_id(
                group,
                timestamps=timestamps,
                coverage_bounds=coverage_bounds,
            )
            eligible.append((group, comparison, window_id))
        if initial_eligible_group_count is None:
            initial_eligible_group_count = len(eligible)
            for group, _comparison, window_id in eligible:
                initial_eligible_count_by_window[int(window_id)] += 1
                initial_eligible_count_by_scene[scene_key(group, window_id)] += 1
        if not eligible:
            break

        def priority(item: tuple[DurableStateGroup, _MaterialComparison, int]) -> tuple[float, ...]:
            group, comparison, window_id = item
            sample_idx = int(group.representative_sample_idx)
            sharpness = (
                float(frame_metrics.sharpness_for(sample_idx) or 0.0)
                if frame_metrics is not None
                else 0.0
            )
            proxy = (
                float(frame_metrics.proxy_content_score[sample_idx])
                if frame_metrics is not None and frame_metrics.has_sample(sample_idx)
                else 0.0
            )
            frame_idx = int(frame_indices[sample_idx]) if 0 <= sample_idx < len(frame_indices) else sample_idx
            selected_in_window = secondary_count_by_window[int(window_id)]
            eligible_in_window = max(initial_eligible_count_by_window[int(window_id)], 1)
            eligible_in_scene = initial_eligible_count_by_scene[scene_key(group, window_id)]
            return (
                float(selected_in_window > 0),
                -float(eligible_in_scene),
                float(selected_in_window) / float(eligible_in_window),
                -float(group.aggregate_duration_seconds),
                -float(comparison.distance_score),
                -float(group.qualifying_repeated_occurrence_count),
                -proxy,
                -sharpness,
                float(timestamps[sample_idx]),
                float(frame_idx),
                float(sample_idx),
            )

        group, comparison, window_id = min(eligible, key=priority)
        candidate = _candidate_from_group(
            group,
            timestamps=timestamps,
            frame_indices=frame_indices,
            dhashes=dhashes,
            frame_metrics=frame_metrics,
            coverage_interval_seconds=coverage_interval_seconds,
            coverage_bounds=coverage_bounds,
            comparison=comparison,
        )
        retained.append(candidate)
        selected_fill.append(candidate)
        selected_group_ids.add(int(group.group_id))
        secondary_count_by_window[int(window_id)] += 1

    return (
        tuple(sorted(retained, key=lambda c: (float(c.timestamp), int(c.sample_idx)))),
        tuple(sorted(selected_fill, key=lambda c: (float(c.timestamp), int(c.sample_idx)))),
        int(initial_eligible_group_count or 0),
    )


def select_primary_candidates(
    *,
    semantic_candidates: Sequence[CandidateRecord],
    coverage_pool: Sequence[CandidateRecord],
    timestamps: Sequence[float],
    frame_indices: Sequence[int],
    dhashes: Sequence[int] | Mapping[int, int],
    clip_embeddings: Any | None,
    frame_metrics: FrameMetricTable | None,
    sample_scenes: Mapping[int, int] | None,
    coverage_interval_seconds: float,
    minimum_settled_dwell_seconds: float,
    duration_seconds: float,
    max_primary_candidates: int,
) -> PrimarySelectionResult:
    from keyframe.scoring import assign_dwell_ids

    dwell_ids = tuple(
        assign_dwell_ids(
            dhashes,
            hamming_threshold=_DURABLE_DWELL_HAMMING_THRESHOLD,
        )
    )
    semantic = tuple(
        _candidate_with_sample_context(
            candidate,
            dwell_ids=dwell_ids,
            sample_scenes=sample_scenes,
            dhashes=dhashes,
        )
        for candidate in semantic_candidates
    )
    coverage = tuple(
        _candidate_with_sample_context(
            candidate,
            dwell_ids=dwell_ids,
            sample_scenes=sample_scenes,
            dhashes=dhashes,
        )
        for candidate in coverage_pool
    )
    consolidated_coverage = _consolidate_coverage_pool_by_dwell(coverage)
    semantic_unique_count = len({int(candidate.sample_idx) for candidate in semantic})
    coverage_subset = _distributed_window_subset(
        list(consolidated_coverage),
        max(0, int(max_primary_candidates) - semantic_unique_count),
    )
    base = _combine_primary_lanes(
        semantic,
        coverage_subset,
        max_primary_candidates=max_primary_candidates,
    )
    base_count = len(base)
    coverage_bounds = _coverage_window_bounds(duration_seconds, coverage_interval_seconds)
    occurrences = build_visual_state_occurrences(
        timestamps=timestamps,
        frame_indices=frame_indices,
        dhashes=dhashes,
        clip_embeddings=clip_embeddings,
        frame_metrics=frame_metrics,
        coverage_interval_seconds=coverage_interval_seconds,
        duration_seconds=duration_seconds,
        sample_scenes=sample_scenes,
        dwell_ids=dwell_ids,
    )
    groups = build_durable_state_groups(
        occurrences,
        timestamps=timestamps,
        frame_indices=frame_indices,
        dhashes=dhashes,
        clip_embeddings=clip_embeddings,
        frame_metrics=frame_metrics,
        minimum_settled_dwell_seconds=minimum_settled_dwell_seconds,
    )
    remaining_capacity = max(0, int(max_primary_candidates) - base_count)
    if remaining_capacity > 0:
        candidates, durable_fill, eligible_durable_group_count = _fill_durable_state_capacity(
            base,
            groups,
            timestamps=timestamps,
            frame_indices=frame_indices,
            dhashes=dhashes,
            clip_embeddings=clip_embeddings,
            frame_metrics=frame_metrics,
            coverage_interval_seconds=coverage_interval_seconds,
            coverage_bounds=coverage_bounds,
            max_primary_candidates=max_primary_candidates,
        )
    else:
        candidates = base
        durable_fill = ()
        eligible_durable_group_count = 0
    metadata = {
        "base_coverage_count": int(len(coverage_pool)),
        "base_semantic_count": int(len(semantic_candidates)),
        "unique_base_primary_count": int(base_count),
        "durable_occurrence_count": int(len(occurrences)),
        "durable_group_count": int(len(groups)),
        "eligible_durable_group_count": int(eligible_durable_group_count),
        "durable_fill_count": int(len(durable_fill)),
        "final_primary_count": int(len(candidates)),
        "durable_state_group_count": int(len(groups)),
        "durable_state_fill_count": int(len(durable_fill)),
        "remaining_primary_capacity": int(remaining_capacity),
        "unused_primary_capacity": int(max(0, int(max_primary_candidates) - len(candidates))),
    }
    return PrimarySelectionResult(
        candidates=tuple(sorted(candidates, key=lambda c: (float(c.timestamp), int(c.sample_idx)))),
        durable_state_fill=durable_fill,
        metadata=metadata,
        dwell_ids=dwell_ids,
    )
