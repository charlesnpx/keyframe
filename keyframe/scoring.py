"""Deterministic allocation and representative scoring helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from collections import defaultdict
import math
import re
from typing import Any

import numpy as np
from PIL import Image

from keyframe.dedupe import (
    canonical_markers,
    has_evidence_markers,
    has_meaningful_evidence_for_retention,
    hamming,
)
from keyframe.evidence import (
    select_structured_comparator,
    structured_delta_categories,
    structured_signature_change_count,
)
from keyframe.pipeline.contracts import CandidateRecord, as_candidate_record, candidate_records
from keyframe.visual import FrameMetricTable, build_frame_metric_table, mean_abs_content_delta


def coalesce_tiny_scenes(
    scenes: Sequence[tuple[int, int]],
    timestamps: Sequence[float],
    dhashes: Sequence[int] | Mapping[int, int],
    max_scene_seconds: float = 3.0,
    boundary_hamming_threshold: int = 18,
    return_trace: bool = False,
) -> list[tuple[int, int]] | tuple[list[tuple[int, int]], dict[str, Any]]:
    """Merge tiny scene-detection fragments unless the visual boundary is large."""
    if not scenes:
        trace = {"original_scene_count": 0, "coalesced_scene_count": 0, "coalescences": []}
        return ([], trace) if return_trace else []

    merged: list[tuple[int, int]] = []
    coalescences: list[dict[str, Any]] = []
    for start, end in scenes:
        duration = float(timestamps[end]) - float(timestamps[start]) if end < len(timestamps) else 0.0
        if not merged or duration >= max_scene_seconds:
            merged.append((start, end))
            continue

        prev_start, prev_end = merged[-1]
        try:
            boundary_jump = hamming(int(dhashes[prev_end]), int(dhashes[start]))
        except (IndexError, KeyError):
            boundary_jump = boundary_hamming_threshold

        if boundary_jump >= boundary_hamming_threshold:
            merged.append((start, end))
        else:
            merged[-1] = (prev_start, end)
            coalescences.append({
                "from_scene": [int(start), int(end)],
                "into_scene": [int(prev_start), int(prev_end)],
                "result_scene": [int(prev_start), int(end)],
                "boundary_hash_jump": int(boundary_jump),
            })

    if return_trace:
        return merged, {
            "original_scene_count": len(scenes),
            "coalesced_scene_count": len(merged),
            "coalescences": coalescences,
        }
    return merged


def allocate_clusters_by_novelty(
    scenes: Sequence[tuple[int, int]],
    total_clusters: int,
    dhashes: Sequence[int] | Mapping[int, int],
    floor: int = 1,
) -> list[int]:
    """Allocate a cluster budget by scene visual novelty while summing exactly."""
    if not scenes:
        return []
    if total_clusters <= 0:
        return [0] * len(scenes)

    min_budget = min(floor, total_clusters // len(scenes)) if total_clusters < floor * len(scenes) else floor
    allocs = [min_budget] * len(scenes)
    remaining = total_clusters - sum(allocs)
    if remaining <= 0:
        return allocs

    novelty: list[float] = []
    for start, end in scenes:
        distances = []
        for idx in range(start, end):
            try:
                distances.append(hamming(int(dhashes[idx]), int(dhashes[idx + 1])))
            except (IndexError, KeyError):
                continue
        novelty.append(sum(distances) / len(distances) if distances else 0.0)

    if sum(novelty) <= 0:
        weights = [max(1, end - start + 1) for start, end in scenes]
    else:
        weights = novelty

    raw = [remaining * w / sum(weights) for w in weights]
    extras = [int(x) for x in raw]
    allocs = [a + e for a, e in zip(allocs, extras)]
    leftover = total_clusters - sum(allocs)

    order = sorted(range(len(scenes)), key=lambda i: (raw[i] - extras[i], weights[i]), reverse=True)
    for i in order[:leftover]:
        allocs[i] += 1
    return allocs


def candidate_budget_for_scenes(base_clusters: int, scene_count: int, multiplier: int = 2) -> int:
    """Allow scene-heavy videos a bounded candidate budget above the base cluster count."""
    if base_clusters <= 0 or scene_count <= 0:
        return 0
    return max(base_clusters, min(scene_count, base_clusters * multiplier))


def _records(candidates: Sequence[Mapping[str, Any] | CandidateRecord]) -> tuple[CandidateRecord, ...]:
    return candidate_records(candidates)


def _record_tokens(candidate: CandidateRecord, *, rescue: bool = False) -> set[str]:
    if rescue and candidate.evidence.rescue_tokens:
        return set(candidate.evidence.rescue_tokens)
    if candidate.evidence.rescue_tokens:
        return set(candidate.evidence.rescue_tokens)
    return set(candidate.evidence.ocr_tokens)


def score_candidate_for_rep(
    candidate: CandidateRecord,
    image: Any | None = None,
    transcript_density: float = 0.0,
    end_of_dwell_bonus: float | None = None,
) -> float:
    """Score a candidate for representative selection."""
    sharpness = candidate.visual.sharpness
    if sharpness is None and image is not None:
        from keyframe.visual import laplacian_sharpness

        sharpness = laplacian_sharpness(image)

    sharpness = float(sharpness or 0.0)
    normalized_sharpness = min(sharpness / 1000.0, 1.5)
    transcript_bonus = min(max(float(transcript_density or 0.0), 0.0), 1.0) * 0.75

    if end_of_dwell_bonus is None:
        end_of_dwell_bonus = float(candidate.selection.end_of_dwell_bonus or 0.0)
    dwell_bonus = min(max(float(end_of_dwell_bonus), 0.0), 1.0) * 0.5
    return normalized_sharpness + transcript_bonus + dwell_bonus


def _normalize(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return [0.0 for _ in values]
    return [(float(value) - lo) / (hi - lo) for value in values]


def proxy_content_scores(frames: Sequence[Image.Image]) -> list[dict[str, float]]:
    """Compute normalized proxy content scores for every sampled frame."""
    sample_idxs = list(range(len(frames)))
    table = build_frame_metric_table(frames, [float(idx) for idx in sample_idxs], sample_idxs)
    return table.to_proxy_rows()


def assign_dwell_ids(dhashes: Sequence[int] | Mapping[int, int], hamming_threshold: int = 6) -> list[int]:
    """Assign stable visual dwell ids based on adjacent dHash continuity."""
    if isinstance(dhashes, Mapping):
        keys = sorted(int(k) for k in dhashes)
        values = [int(dhashes[k]) for k in keys]
    else:
        values = [int(v) for v in dhashes]
    if not values:
        return []
    dwell_ids = [0]
    current = 0
    for previous, value in zip(values, values[1:]):
        if hamming(previous, value) > hamming_threshold:
            current += 1
        dwell_ids.append(current)
    return dwell_ids


def rescue_window_seconds(timestamps: Sequence[float]) -> float:
    diffs = [
        float(b) - float(a)
        for a, b in zip(timestamps, timestamps[1:])
        if float(b) > float(a)
    ]
    sample_interval = float(np.median(diffs)) if diffs else 0.5
    return max(20.0, 8.0 * sample_interval)


def assign_temporal_window_ids(
    timestamps: Sequence[float],
    sample_scenes: Mapping[int, int] | None,
    *,
    window_seconds: float | None = None,
) -> list[int | None]:
    if not timestamps:
        return []
    window_seconds = float(window_seconds or rescue_window_seconds(timestamps))
    scene_starts: dict[int, float] = {}
    if sample_scenes:
        for sample_idx, scene_id in sample_scenes.items():
            idx = int(sample_idx)
            if 0 <= idx < len(timestamps):
                scene_starts[int(scene_id)] = min(
                    scene_starts.get(int(scene_id), float(timestamps[idx])),
                    float(timestamps[idx]),
                )

    window_ids: list[int | None] = []
    for sample_idx, timestamp in enumerate(timestamps):
        scene_id = sample_scenes.get(sample_idx) if sample_scenes else None
        if scene_id is None:
            window_ids.append(int(float(timestamp) // window_seconds))
            continue
        start = scene_starts.get(int(scene_id), float(timestamp))
        window_ids.append(int(max(0.0, float(timestamp) - start) // window_seconds))
    return window_ids


def build_rescue_shortlist(
    frames: Sequence[Image.Image] | None,
    timestamps: Sequence[float],
    frame_indices: Sequence[int],
    candidates: Sequence[Mapping[str, Any] | CandidateRecord],
    pass1_clusters: int,
    *,
    sample_clusters: Mapping[int, int] | None = None,
    sample_scenes: Mapping[int, int] | None = None,
    frame_metrics: FrameMetricTable | None = None,
    frame_count: int | None = None,
    dhashes: Sequence[int] | Mapping[int, int] | None = None,
) -> tuple[
    tuple[CandidateRecord, ...],
    list[dict[str, float]],
    int,
    int,
    int,
    int,
    int,
    dict[str, Any],
]:
    """Rank non-selected sampled frames for bounded OCR rescue."""
    candidates = _records(candidates)
    sample_count = len(frames) if frames is not None else int(frame_count or 0)
    if sample_count != len(timestamps) or sample_count != len(frame_indices):
        raise ValueError("rescue frame metadata must have matching lengths")
    if frame_metrics is None and frames is None:
        raise ValueError("rescue shortlist requires frame metrics when frames are not retained")
    proxy_rows = frame_metrics.to_proxy_rows() if frame_metrics is not None else proxy_content_scores(frames)
    candidate_idxs = {int(c.sample_idx) for c in candidates}
    eligible_mask = np.ones((sample_count,), dtype=bool)
    for idx in candidate_idxs:
        if 0 <= idx < len(eligible_mask):
            eligible_mask[idx] = False
    duration_seconds = max(timestamps) - min(timestamps) if timestamps else 0.0
    duration_floor = min(8, math.ceil(float(duration_seconds) / 90.0))
    rescue_budget = max(3, round(pass1_clusters * 0.35), duration_floor)
    scores = [row["proxy_content_score"] for row in proxy_rows]
    tau_proxy = float(np.percentile(scores, 75)) if scores else 0.0
    legacy_cap = min(60, rescue_budget * 4)
    proposal_decisions: list[dict[str, Any]] = []

    window_seconds = rescue_window_seconds(timestamps)
    temporal_window_ids = assign_temporal_window_ids(
        timestamps,
        sample_scenes,
        window_seconds=window_seconds,
    )

    if frame_metrics is not None:
        content_deltas = [float(value) for value in frame_metrics.content_prev_delta]
    else:
        content_deltas = [0.0 for _ in range(sample_count)]
        for idx in range(1, sample_count):
            content_deltas[idx] = mean_abs_content_delta(frames[idx - 1], frames[idx])

    def hash_at(sample_idx: int) -> int | None:
        if dhashes is None:
            return None
        if isinstance(dhashes, Mapping):
            value = dhashes.get(int(sample_idx))
        else:
            value = (
                dhashes[int(sample_idx)]
                if 0 <= int(sample_idx) < len(dhashes)
                else None
            )
        return int(value) if value is not None else None

    def path_content_delta(left: int, right: int) -> float:
        start, end = sorted((int(left), int(right)))
        if start == end:
            return 0.0
        segment = content_deltas[start + 1 : end + 1]
        return max(segment, default=float("inf"))

    ranked: list[dict[str, Any]] = []
    for sample_idx, metrics in enumerate(proxy_rows):
        previous_delta = content_deltas[sample_idx] if sample_idx < len(content_deltas) else 0.0
        next_delta = content_deltas[sample_idx + 1] if sample_idx + 1 < len(content_deltas) else 0.0
        metrics["content_area_delta_score"] = float(max(previous_delta, next_delta) / 255.0)
        metrics["content_area_previous_delta"] = float(previous_delta)
        metrics["content_area_next_delta"] = float(next_delta)
        if not eligible_mask[sample_idx]:
            continue
        row = {
            "sample_idx": int(sample_idx),
            "frame_idx": int(frame_indices[sample_idx]),
            "timestamp": float(timestamps[sample_idx]),
            "clip_cluster": sample_clusters.get(sample_idx) if sample_clusters else None,
            "scene_id": sample_scenes.get(sample_idx) if sample_scenes else None,
            "temporal_window_id": temporal_window_ids[sample_idx] if sample_idx < len(temporal_window_ids) else None,
            "temporal_window_seconds": window_seconds,
            "clip_cluster_size": 1,
            "cluster_role": "rescue",
            "proxy_content_score": float(metrics["proxy_content_score"]),
            "content_area_delta_score": float(metrics["content_area_delta_score"]),
            "rescue_priority": 0,
        }
        if frame_metrics is not None and frame_metrics.has_sample(sample_idx):
            row["sharpness"] = float(frame_metrics.sharpness[sample_idx])
        ranked.append(row)

    ranked.sort(
        key=lambda row: (
            float(row["proxy_content_score"]),
            -float(row["timestamp"]),
            -int(row["sample_idx"]),
        ),
        reverse=True,
    )

    by_time_window: dict[tuple[Any, int], list[dict[str, Any]]] = defaultdict(list)
    by_scene: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in ranked:
        scene_id = row.get("scene_id")
        window_id = int(row.get("temporal_window_id") or 0)
        by_time_window[(scene_id, window_id)].append(row)
        if scene_id is not None:
            by_scene[scene_id].append(row)

    temporal_window_count = len(by_time_window)
    scene_count = len(by_scene)
    rescue_ocr_cap = min(
        96,
        max(
            rescue_budget * 4,
            2 * temporal_window_count + scene_count,
            40,
        ),
    )
    reserved_proposal_capacity = min(
        rescue_ocr_cap,
        2 * temporal_window_count + scene_count,
    )

    def add_to_lane(
        lane: list[dict[str, Any]],
        selected: set[int],
        row: Mapping[str, Any],
        proposal_lane: str,
        cap: int | None = None,
    ) -> bool:
        if cap is not None and len(lane) >= cap:
            return False
        sample_idx = int(row["sample_idx"])
        if sample_idx in selected:
            return False
        lane.append({**dict(row), "proposal_lane": proposal_lane})
        selected.add(sample_idx)
        return True

    rows_by_idx = {int(row["sample_idx"]): row for row in ranked}
    selected_by_idx = {int(candidate.sample_idx): candidate for candidate in candidates}
    meaningful_deltas = [
        float(delta)
        for delta in content_deltas
        if float(delta) >= 3.0
    ]
    transition_content_threshold = max(
        3.0,
        float(np.percentile(meaningful_deltas, 70))
        if meaningful_deltas
        else 3.0,
    )

    def context_for(sample_idx: int) -> tuple[Any, int]:
        scene_id = sample_scenes.get(sample_idx) if sample_scenes else None
        window_id = (
            int(temporal_window_ids[sample_idx] or 0)
            if sample_idx < len(temporal_window_ids)
            else 0
        )
        return scene_id, window_id

    def local_side_coverage(sample_idx: int) -> dict[str, Any] | None:
        exact = selected_by_idx.get(int(sample_idx))
        if exact is not None:
            return {
                "coverage_reason": "exact_selected_sample",
                "covering_sample_idx": int(exact.sample_idx),
                "covering_timestamp": float(exact.timestamp),
                "dhash_distance": 0,
                "content_path_delta": 0.0,
            }
        scene_id = sample_scenes.get(sample_idx) if sample_scenes else None
        sample_timestamp = float(timestamps[sample_idx])
        sample_hash = hash_at(sample_idx)
        for selected in sorted(
            candidates,
            key=lambda candidate: (
                abs(float(candidate.timestamp) - sample_timestamp),
                int(candidate.frame_idx),
                int(candidate.sample_idx),
            ),
        ):
            if selected.temporal.scene_id != scene_id:
                continue
            if abs(float(selected.timestamp) - sample_timestamp) > 2.25:
                continue
            selected_hash = hash_at(int(selected.sample_idx))
            hash_distance = (
                hamming(sample_hash, selected_hash)
                if sample_hash is not None and selected_hash is not None
                else None
            )
            content_delta = path_content_delta(
                sample_idx,
                int(selected.sample_idx),
            )
            if (
                (hash_distance is not None and hash_distance <= 6)
                or content_delta <= 2.5
            ):
                return {
                    "coverage_reason": (
                        "local_selected_dhash"
                        if hash_distance is not None and hash_distance <= 6
                        else "local_selected_content_delta"
                    ),
                    "covering_sample_idx": int(selected.sample_idx),
                    "covering_timestamp": float(selected.timestamp),
                    "dhash_distance": hash_distance,
                    "content_path_delta": float(content_delta),
                }
        return None

    transition_queues: dict[
        tuple[Any, int],
        list[dict[str, Any]],
    ] = defaultdict(list)
    if sample_count >= 2:
        for boundary_idx in range(1, sample_count):
            content_delta = float(content_deltas[boundary_idx])
            next_delta = (
                float(content_deltas[boundary_idx + 1])
                if boundary_idx + 1 < sample_count
                else 0.0
            )
            text_band_delta = abs(
                float(proxy_rows[boundary_idx].get("textline_score", 0.0))
                - float(
                    proxy_rows[boundary_idx - 1].get(
                        "textline_score",
                        0.0,
                    )
                )
            )
            left_hash = hash_at(boundary_idx - 1)
            right_hash = hash_at(boundary_idx)
            dhash_distance = (
                hamming(left_hash, right_hash)
                if left_hash is not None and right_hash is not None
                else None
            )
            local_peak = (
                content_delta >= transition_content_threshold
                and content_delta >= next_delta
            )
            settled_transition = (
                content_delta >= transition_content_threshold
                and next_delta <= max(2.0, content_delta * 0.60)
            )
            predicates = tuple(
                predicate
                for predicate, met in (
                    ("local_peak", local_peak),
                    ("settled_transition", settled_transition),
                    ("text_band", text_band_delta >= 0.20),
                )
                if met
            )
            distinct_reasons = tuple(
                reason
                for reason, met in (
                    (
                        "content_threshold",
                        content_delta >= transition_content_threshold,
                    ),
                    ("text_band_threshold", text_band_delta >= 0.20),
                    (
                        "dhash_threshold",
                        dhash_distance is not None
                        and dhash_distance >= 12,
                    ),
                )
                if met
            )
            boundary_detail = {
                "boundary_sample_idx": int(boundary_idx),
                "boundary_timestamp": float(timestamps[boundary_idx]),
                "content_delta": content_delta,
                "content_threshold": transition_content_threshold,
                "text_band_delta": text_band_delta,
                "dhash_distance": dhash_distance,
                "predicates": list(predicates),
                "distinct_reasons": list(distinct_reasons),
            }
            if not predicates or not distinct_reasons:
                proposal_decisions.append(
                    {
                        **boundary_detail,
                        "decision": "transition_rejected",
                        "reason": (
                            "transition_predicate_not_met"
                            if not predicates
                            else "transition_not_distinct"
                        ),
                    }
                )
                continue

            proposal_decisions.append(
                {
                    **boundary_detail,
                    "decision": "transition_qualified",
                    "reason": "distinct_transition",
                }
            )
            for side, anchor_idx, candidate_range in (
                ("pre", boundary_idx - 1, range(boundary_idx - 1, -1, -1)),
                ("post", boundary_idx, range(boundary_idx, sample_count)),
            ):
                coverage = local_side_coverage(anchor_idx)
                if coverage is not None:
                    proposal_decisions.append(
                        {
                            **boundary_detail,
                            "decision": "transition_side_rejected",
                            "reason": "locally_covered",
                            "transition_side": side,
                            "anchor_sample_idx": int(anchor_idx),
                            **coverage,
                        }
                    )
                    continue

                anchor_context = context_for(anchor_idx)
                pool = [
                    sample_idx
                    for sample_idx in candidate_range
                    if eligible_mask[sample_idx]
                    and sample_idx in rows_by_idx
                    and context_for(sample_idx) == anchor_context
                ]
                if not pool:
                    proposal_decisions.append(
                        {
                            **boundary_detail,
                            "decision": "transition_side_rejected",
                            "reason": "no_eligible_side_sample",
                            "transition_side": side,
                            "anchor_sample_idx": int(anchor_idx),
                        }
                    )
                    continue

                sharpness_by_idx = {
                    sample_idx: float(
                        rows_by_idx[sample_idx].get("sharpness", 0.0)
                    )
                    for sample_idx in pool
                }
                best_sharpness = max(sharpness_by_idx.values(), default=0.0)
                sharpness_floor = 0.5 * best_sharpness
                sharp_pool = [
                    sample_idx
                    for sample_idx in pool
                    if sharpness_by_idx[sample_idx] >= sharpness_floor
                ]
                chosen_idx = min(
                    sharp_pool,
                    key=lambda sample_idx: (
                        abs(
                            float(timestamps[sample_idx])
                            - float(timestamps[boundary_idx])
                        ),
                        -sharpness_by_idx[sample_idx],
                        int(frame_indices[sample_idx]),
                        int(sample_idx),
                    ),
                )
                row = {
                    **dict(rows_by_idx[chosen_idx]),
                    "proposal_lane": "transition",
                    "transition_side": side,
                    "transition_boundary_sample_idx": int(boundary_idx),
                    "transition_boundary_timestamp": float(
                        timestamps[boundary_idx]
                    ),
                    "transition_boundary_content_delta": content_delta,
                    "transition_boundary_text_band_delta": text_band_delta,
                    "transition_boundary_dhash_distance": dhash_distance,
                    "transition_boundary_predicates": predicates,
                }
                transition_queues[context_for(chosen_idx)].append(row)
                proposal_decisions.append(
                    {
                        **boundary_detail,
                        "decision": "transition_side_proposed",
                        "reason": "nearest_eligible_side_sample",
                        "transition_side": side,
                        "anchor_sample_idx": int(anchor_idx),
                        "sample_idx": int(chosen_idx),
                        "timestamp": float(timestamps[chosen_idx]),
                        "sharpness": sharpness_by_idx[chosen_idx],
                        "sharpness_floor": sharpness_floor,
                        "sharpness_floor_rejected_count": (
                            len(pool) - len(sharp_pool)
                        ),
                    }
                )

    def build_legacy_proxy_lane() -> list[dict[str, Any]]:
        lane: list[dict[str, Any]] = []
        selected: set[int] = set()
        global_quota = min(legacy_cap, max(rescue_budget, legacy_cap // 4))
        for row in ranked:
            if float(row["proxy_content_score"]) >= tau_proxy:
                add_to_lane(lane, selected, row, "legacy_proxy", legacy_cap)
            if len(lane) >= global_quota:
                break

        for key in sorted(
            by_time_window,
            key=lambda item: (
                -max(float(row["proxy_content_score"]) for row in by_time_window[item]),
                -len(by_time_window[item]),
            ),
        ):
            if len(lane) >= legacy_cap:
                break
            added_in_window: list[float] = []
            for row in by_time_window[key]:
                ts = float(row["timestamp"])
                if any(abs(ts - existing_ts) < 2.25 for existing_ts in added_in_window):
                    continue
                if add_to_lane(lane, selected, row, "legacy_proxy", legacy_cap):
                    added_in_window.append(ts)
                if len(added_in_window) >= 3 or len(lane) >= legacy_cap:
                    break

        for scene_id in sorted(by_scene, key=lambda value: int(value)):
            if len(lane) >= legacy_cap:
                break
            add_to_lane(lane, selected, by_scene[scene_id][0], "legacy_proxy", legacy_cap)

        for row in ranked:
            if len(lane) >= legacy_cap:
                break
            add_to_lane(lane, selected, row, "legacy_proxy", legacy_cap)
        return lane

    def window_sort_key(key: tuple[Any, int]) -> tuple[float, int, int]:
        scene_id, window_id = key
        first_ts = min(float(row["timestamp"]) for row in by_time_window[key])
        scene_sort = int(scene_id) if scene_id is not None else -1
        return first_ts, scene_sort, int(window_id)

    window_keys = sorted(by_time_window, key=window_sort_key)
    all_timestamps_by_window: dict[
        tuple[Any, int],
        list[float],
    ] = defaultdict(list)
    for sample_idx, timestamp in enumerate(timestamps):
        all_timestamps_by_window[context_for(sample_idx)].append(
            float(timestamp)
        )
    selected_timestamps_by_window: dict[
        tuple[Any, int],
        list[float],
    ] = defaultdict(list)
    for candidate in candidates:
        selected_timestamps_by_window[
            context_for(int(candidate.sample_idx))
        ].append(float(candidate.timestamp))

    for queue in transition_queues.values():
        queue.sort(
            key=lambda row: (
                float(row["transition_boundary_timestamp"]),
                0 if row["transition_side"] == "pre" else 1,
                float(row["timestamp"]),
                int(row["frame_idx"]),
                int(row["sample_idx"]),
            )
        )

    def coverage_order(
        key: tuple[Any, int],
    ) -> list[dict[str, Any]]:
        rows = list(by_time_window[key])
        if not rows:
            return []
        window_timestamps = all_timestamps_by_window.get(key) or [
            float(row["timestamp"]) for row in rows
        ]
        start = min(window_timestamps)
        end = max(window_timestamps)
        midpoint = (start + end) / 2.0
        transitions = transition_queues.get(key, ())
        if transitions and end > start:
            transition_ts = float(
                transitions[0]["transition_boundary_timestamp"]
            )
            selected_in_window = selected_timestamps_by_window.get(key, ())
            selected_anchor = (
                float(np.median(selected_in_window))
                if selected_in_window
                else None
            )
            transition_is_early = transition_ts <= midpoint
            boundary_hash_distance = transitions[0].get(
                "transition_boundary_dhash_distance"
            )
            strongly_distinct_transition = (
                (
                    boundary_hash_distance is not None
                    and int(boundary_hash_distance) >= 12
                )
                or float(
                    transitions[0].get(
                        "transition_boundary_text_band_delta",
                        0.0,
                    )
                )
                >= 0.20
            )
            selection_brackets_transition = (
                strongly_distinct_transition
                and selected_anchor is not None
                and (
                    (
                        transition_is_early
                        and selected_anchor > transition_ts
                    )
                    or (
                        not transition_is_early
                        and selected_anchor < transition_ts
                    )
                )
            )
            if (
                not strongly_distinct_transition
                and selected_anchor is not None
            ):
                fraction = (
                    0.8
                    if selected_anchor <= midpoint
                    else 0.2
                )
                target = start + fraction * (end - start)
            elif selection_brackets_transition:
                target = (transition_ts + selected_anchor) / 2.0
            else:
                boundary_idx = int(
                    transitions[0][
                        "transition_boundary_sample_idx"
                    ]
                )
                delayed_tail_idxs = (
                    [
                        sample_idx
                        for sample_idx in range(
                            boundary_idx + 2,
                            sample_count,
                        )
                        if context_for(sample_idx) == key
                        and float(timestamps[sample_idx])
                        <= transition_ts + 2.25
                        and (
                            float(content_deltas[sample_idx]) > 2.5
                            or (
                                hash_at(sample_idx - 1) is not None
                                and hash_at(sample_idx) is not None
                                and hamming(
                                    hash_at(sample_idx - 1),
                                    hash_at(sample_idx),
                                )
                                > 6
                            )
                        )
                    ]
                    if strongly_distinct_transition
                    else []
                )
                if delayed_tail_idxs:
                    target = min(
                        end,
                        float(timestamps[delayed_tail_idxs[-1]])
                        + 2.25,
                    )
                else:
                    fraction = 0.8 if transition_is_early else 0.2
                    target = start + fraction * (end - start)
        elif selected_timestamps_by_window.get(key) and end > start:
            selected_anchor = float(
                np.median(selected_timestamps_by_window[key])
            )
            fraction = 0.8 if selected_anchor <= midpoint else 0.2
            target = start + fraction * (end - start)
        else:
            target = midpoint

        first = min(
            rows,
            key=lambda row: (
                abs(float(row["timestamp"]) - target),
                -float(row.get("proxy_content_score", 0.0)),
                -float(row.get("sharpness", 0.0)),
                int(row["frame_idx"]),
                int(row["sample_idx"]),
            ),
        )
        ordered = [first]
        remaining = [
            row
            for row in rows
            if int(row["sample_idx"]) != int(first["sample_idx"])
        ]
        while remaining:
            row = max(
                remaining,
                key=lambda candidate: (
                    min(
                        abs(
                            float(candidate["timestamp"])
                            - float(selected["timestamp"])
                        )
                        for selected in ordered
                    ),
                    float(candidate.get("proxy_content_score", 0.0)),
                    float(candidate.get("sharpness", 0.0)),
                    -float(candidate["timestamp"]),
                    -int(candidate["frame_idx"]),
                    -int(candidate["sample_idx"]),
                ),
            )
            ordered.append(row)
            remaining.remove(row)
        return ordered

    coverage_queues: dict[
        tuple[Any, int],
        list[dict[str, Any]],
    ] = {
        key: [
            {**dict(row), "proposal_lane": "temporal_coverage"}
            for row in coverage_order(key)
        ]
        for key in window_keys
    }

    scene_window_counts: dict[Any, int] = defaultdict(int)
    for scene_id, _window_id in by_time_window:
        if scene_id is not None:
            scene_window_counts[scene_id] += 1

    ordered_scene_ids = sorted(
        by_scene,
        key=lambda scene_id: (
            min(float(row["timestamp"]) for row in by_scene[scene_id]),
            int(scene_id),
        ),
    )
    scene_queues: dict[Any, list[dict[str, Any]]] = {
        scene_id: [
            {**dict(row), "proposal_lane": "scene_coverage"}
            for row in by_scene[scene_id]
        ]
        for scene_id in ordered_scene_ids
        if scene_window_counts.get(scene_id, 0) > 1
    }

    legacy_lane = build_legacy_proxy_lane()
    legacy_sample_idxs = {
        int(row["sample_idx"]) for row in legacy_lane
    }
    final_rows: list[dict[str, Any]] = []
    final_selected: set[int] = set()
    queue_positions: dict[tuple[str, Any], int] = defaultdict(int)

    def add_final(
        row: Mapping[str, Any],
        *,
        allocation_phase: str,
    ) -> bool:
        sample_idx = int(row["sample_idx"])
        lane = str(row.get("proposal_lane") or "global_backfill")
        if len(final_rows) >= rescue_ocr_cap:
            proposal_decisions.append(
                {
                    "decision": "quota_rejected",
                    "reason": "rescue_ocr_cap_exhausted",
                    "allocation_phase": allocation_phase,
                    "proposal_lane": lane,
                    "sample_idx": sample_idx,
                    "timestamp": float(row["timestamp"]),
                }
            )
            return False
        if sample_idx in final_selected:
            proposal_decisions.append(
                {
                    "decision": "quota_rejected",
                    "reason": "sample_already_allocated",
                    "allocation_phase": allocation_phase,
                    "proposal_lane": lane,
                    "sample_idx": sample_idx,
                    "timestamp": float(row["timestamp"]),
                }
            )
            return False
        final_rows.append(dict(row))
        final_selected.add(sample_idx)
        proposal_decisions.append(
            {
                "decision": "quota_allocated",
                "reason": allocation_phase,
                "allocation_phase": allocation_phase,
                "proposal_lane": lane,
                "sample_idx": sample_idx,
                "timestamp": float(row["timestamp"]),
                "allocation_position": len(final_rows),
            }
        )
        return True

    def take_from_queue(
        queue_name: str,
        queue_key: Any,
        queue: Sequence[Mapping[str, Any]],
        *,
        allocation_phase: str,
        protect_legacy_capacity: bool = False,
    ) -> bool:
        position_key = (queue_name, queue_key)
        while queue_positions[position_key] < len(queue):
            position = queue_positions[position_key]
            queue_positions[position_key] += 1
            row = queue[position]
            sample_idx = int(row["sample_idx"])
            if (
                protect_legacy_capacity
                and sample_idx not in final_selected
                and sample_idx not in legacy_sample_idxs
            ):
                remaining_legacy = (
                    legacy_sample_idxs - final_selected
                )
                if (
                    len(final_rows) + len(remaining_legacy)
                    >= rescue_ocr_cap
                ):
                    proposal_decisions.append(
                        {
                            "decision": "quota_rejected",
                            "reason": (
                                "legacy_proxy_capacity_reserved"
                            ),
                            "allocation_phase": allocation_phase,
                            "proposal_lane": queue_name,
                            "sample_idx": sample_idx,
                            "timestamp": float(row["timestamp"]),
                            "reserved_legacy_slots": len(
                                remaining_legacy
                            ),
                        }
                    )
                    continue
            if add_final(
                row,
                allocation_phase=allocation_phase,
            ):
                return True
            if len(final_rows) >= rescue_ocr_cap:
                return False
        proposal_decisions.append(
            {
                "decision": "quota_unavailable",
                "reason": "lane_queue_exhausted",
                "allocation_phase": allocation_phase,
                "proposal_lane": queue_name,
                "queue_key": list(queue_key)
                if isinstance(queue_key, tuple)
                else queue_key,
            }
        )
        return False

    # The first reserved round is temporal and alternates transition/coverage
    # for each window before any window receives a second same-lane slot.
    for key in window_keys:
        if len(final_rows) >= reserved_proposal_capacity:
            break
        take_from_queue(
            "transition",
            key,
            transition_queues.get(key, ()),
            allocation_phase="reserved_first_transition",
        )
        if len(final_rows) >= reserved_proposal_capacity:
            break
        take_from_queue(
            "temporal_coverage",
            key,
            coverage_queues.get(key, ()),
            allocation_phase="reserved_first_temporal_coverage",
        )

    # The remaining reserved scene portion receives one deterministic
    # opportunity per eligible multi-window scene.
    for scene_id, queue in scene_queues.items():
        if len(final_rows) >= reserved_proposal_capacity:
            break
        selected_scene_timestamps = [
            float(row["timestamp"])
            for row in final_rows
            if row.get("scene_id") == scene_id
        ]
        if selected_scene_timestamps:
            queue = sorted(
                queue,
                key=lambda row: (
                    min(
                        abs(
                            float(row["timestamp"])
                            - selected_timestamp
                        )
                        for selected_timestamp in selected_scene_timestamps
                    ),
                    float(row.get("proxy_content_score", 0.0)),
                    float(row.get("sharpness", 0.0)),
                    -float(row["timestamp"]),
                    -int(row["frame_idx"]),
                    -int(row["sample_idx"]),
                ),
                reverse=True,
            )
            scene_queues[scene_id] = queue
        take_from_queue(
            "scene_coverage",
            scene_id,
            queue,
            allocation_phase="reserved_scene_coverage",
        )

    def round_robin_backfill(
        queue_name: str,
        queues: Mapping[Any, Sequence[Mapping[str, Any]]],
        *,
        allocation_phase: str,
        protect_legacy_capacity: bool = False,
    ) -> None:
        while len(final_rows) < rescue_ocr_cap:
            progress = False
            for key in queues:
                if len(final_rows) >= rescue_ocr_cap:
                    break
                before = len(final_rows)
                take_from_queue(
                    queue_name,
                    key,
                    queues[key],
                    allocation_phase=allocation_phase,
                    protect_legacy_capacity=(
                        protect_legacy_capacity
                    ),
                )
                progress = progress or len(final_rows) > before
            if not progress:
                break

    round_robin_backfill(
        "transition",
        {key: transition_queues.get(key, ()) for key in window_keys},
        allocation_phase="transition_backfill",
    )
    round_robin_backfill(
        "temporal_coverage",
        coverage_queues,
        allocation_phase="multi_window_backfill",
        protect_legacy_capacity=True,
    )
    round_robin_backfill(
        "scene_coverage",
        scene_queues,
        allocation_phase="multi_scene_backfill",
        protect_legacy_capacity=True,
    )

    for row in legacy_lane:
        if len(final_rows) >= rescue_ocr_cap:
            break
        add_final(row, allocation_phase="legacy_proxy_backfill")

    for row in ranked:
        if len(final_rows) >= rescue_ocr_cap:
            break
        add_final(
            {**dict(row), "proposal_lane": "global_backfill"},
            allocation_phase="global_backfill",
        )

    if len(final_rows) >= rescue_ocr_cap:
        for queue_name, queues in (
            ("transition", transition_queues),
            ("temporal_coverage", coverage_queues),
            ("scene_coverage", scene_queues),
        ):
            for key, queue in queues.items():
                position = queue_positions[(queue_name, key)]
                for row in queue[position:]:
                    proposal_decisions.append(
                        {
                            "decision": "quota_rejected",
                            "reason": "rescue_ocr_cap_exhausted",
                            "allocation_phase": "unallocated_queue_tail",
                            "proposal_lane": queue_name,
                            "sample_idx": int(row["sample_idx"]),
                            "timestamp": float(row["timestamp"]),
                        }
                    )

    final_sample_idxs = {int(row["sample_idx"]) for row in final_rows}
    legacy_proxy_dropped_count = sum(
        1 for row in legacy_lane if int(row["sample_idx"]) not in final_sample_idxs
    )

    return tuple(
        as_candidate_record(row, origin="rescue_shortlist")
        for row in final_rows[:rescue_ocr_cap]
    ), proxy_rows, rescue_budget, rescue_ocr_cap, temporal_window_count, scene_count, legacy_proxy_dropped_count, {
        "reserved_proposal_capacity": int(reserved_proposal_capacity),
        "transition_content_threshold": float(
            transition_content_threshold
        ),
        "proposal_decisions": tuple(proposal_decisions),
    }


def _marker_signature(tokens: set[str]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    markers = canonical_markers(tokens)
    return tuple((key, tuple(sorted(values))) for key, values in sorted(markers.items()) if values)


def _has_marker_signature(tokens: set[str]) -> bool:
    return bool(_marker_signature(tokens))


def _jaccard(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _clip_cosine(embeddings: Any | None, idx_a: int, idx_b: int) -> float:
    if embeddings is None:
        return 0.0
    try:
        return float(np.dot(embeddings[idx_a], embeddings[idx_b]))
    except Exception:
        return 0.0


def _content_reference_tokens(tokens: set[str]) -> set[str]:
    reference = {"figma", "pdf", "source", "mockup", "design", "document", "doc", "file"}
    return {
        token
        for token in tokens
        if token in reference or "." in token or "_" in token or "-" in token
    }


FORM_STATE_TOKENS = {
    "approved",
    "approval",
    "approve",
    "completed",
    "complete",
    "date",
    "draft",
    "error",
    "field",
    "mandatory",
    "please",
    "required",
    "selection",
    "signed",
    "status",
    "validation",
    "workflow",
}
DATE_VALUE_RE = re.compile(r"^\d{1,2}[a-z]{3}\d{4}$", re.IGNORECASE)


def _has_form_state_delta(rescue_tokens: set[str], candidate_tokens: set[str]) -> bool:
    gained = rescue_tokens - candidate_tokens
    if {"please", "selection"} <= rescue_tokens and not {"please", "selection"} <= candidate_tokens:
        return True
    if any(DATE_VALUE_RE.match(token) for token in gained):
        return True
    return len(gained & FORM_STATE_TOKENS) >= 2


def _marker_redundant(rescue_tokens: set[str], candidate_tokens: set[str]) -> bool:
    return _marker_equivalent(candidate_tokens, rescue_tokens) and not _has_form_state_delta(
        rescue_tokens,
        candidate_tokens,
    )


def _clip_token_redundant(
    rescue_tokens: set[str],
    candidate_tokens: set[str],
    clip_embeddings: Any | None,
    rescue_idx: int,
    candidate_idx: int,
) -> bool:
    if _has_form_state_delta(rescue_tokens, candidate_tokens):
        return False
    return (
        _clip_cosine(clip_embeddings, rescue_idx, candidate_idx) >= 0.93
        and _jaccard(rescue_tokens, candidate_tokens) >= 0.7
    )


def _strictly_subsumes_tokens(rescue_tokens: set[str], primary_tokens: set[str]) -> bool:
    if not primary_tokens:
        return bool(rescue_tokens)
    if _has_form_state_delta(primary_tokens, rescue_tokens):
        return False
    primary_markers = canonical_markers(primary_tokens)
    rescue_markers = canonical_markers(rescue_tokens)
    marker_subsumed = all(
        primary_markers[key] <= rescue_markers[key]
        for key in ("page", "option", "section", "status")
    )
    token_subsumed = primary_tokens <= rescue_tokens
    return marker_subsumed and token_subsumed and rescue_tokens != primary_tokens


def _marker_equivalent(tokens_a: set[str], tokens_b: set[str]) -> bool:
    signature_a = _marker_signature(tokens_a)
    return bool(signature_a) and signature_a == _marker_signature(tokens_b)


def has_local_equivalent_coverage(
    rescue: Mapping[str, Any] | CandidateRecord,
    candidates: Sequence[Mapping[str, Any] | CandidateRecord],
    dwell_ids: Sequence[int],
    *,
    tolerance: float = 2.25,
) -> bool:
    rescue = as_candidate_record(rescue)
    records = _records(candidates)
    rescue_scene = rescue.temporal.scene_id
    rescue_tokens = _record_tokens(rescue, rescue=True)
    for candidate in records:
        if candidate.temporal.scene_id != rescue_scene:
            continue
        candidate_tokens = _record_tokens(candidate, rescue=True)
        if not _marker_redundant(rescue_tokens, candidate_tokens):
            continue
        near_time = abs(float(candidate.timestamp) - float(rescue.timestamp)) <= tolerance
        if near_time:
            return True
    return False


def _nearest_same_scene_delta(rescue: CandidateRecord, candidates: Sequence[CandidateRecord]) -> float:
    rescue_scene = rescue.temporal.scene_id
    deltas = [
        abs(float(candidate.timestamp) - float(rescue.timestamp))
        for candidate in candidates
        if rescue_scene is not None and candidate.temporal.scene_id == rescue_scene
    ]
    return min(deltas, default=float("inf"))


def _nearest_candidate_delta(rescue: CandidateRecord, candidates: Sequence[CandidateRecord]) -> float:
    deltas = [abs(float(candidate.timestamp) - float(rescue.timestamp)) for candidate in candidates]
    return min(deltas, default=float("inf"))


StructuredDiversityBucket = (
    tuple[str, int, int]
    | tuple[str, int, int, int]
)
PromotionLaneCounts = Mapping[str, int]
_SETTLED_CONTENT_AREA_DELTA_SCORE = 2.5 / 255.0


def _structured_diversity_bucket(
    candidate: CandidateRecord,
) -> StructuredDiversityBucket | None:
    scene_id = candidate.temporal.scene_id
    if scene_id is None:
        return None
    dwell_id = candidate.temporal.dwell_id
    window_id = candidate.temporal.temporal_window_id
    if dwell_id is not None and window_id is not None:
        return (
            "dwell_window",
            int(scene_id),
            int(dwell_id),
            int(window_id),
        )
    if dwell_id is not None:
        return ("dwell", int(scene_id), int(dwell_id))
    if window_id is not None:
        return ("window", int(scene_id), int(window_id))
    return ("scene", int(scene_id), int(scene_id))


def _represented_structured_diversity_buckets(
    candidates: Sequence[CandidateRecord],
) -> set[StructuredDiversityBucket]:
    return {
        bucket
        for candidate in candidates
        if candidate.selection.rescue_reason == "structured_delta"
        and (bucket := _structured_diversity_bucket(candidate)) is not None
    }


def _represented_ordinary_scenes(
    candidates: Sequence[CandidateRecord],
) -> set[int]:
    return {
        int(candidate.temporal.scene_id)
        for candidate in candidates
        if candidate.selection.rescue_origin
        and _promotion_lane(
            str(candidate.selection.rescue_reason or "")
        )
        == "ordinary"
        and candidate.temporal.scene_id is not None
    }


def _promotion_lane(reason: str) -> str:
    if reason == "structured_delta":
        return "structured"
    if reason == "transition":
        return "transition"
    return "ordinary"


def _promotion_lane_counts(
    candidates: Sequence[CandidateRecord],
) -> dict[str, int]:
    counts = {
        "structured": 0,
        "transition": 0,
        "ordinary": 0,
    }
    for candidate in candidates:
        if not candidate.selection.rescue_origin:
            continue
        counts[_promotion_lane(
            str(candidate.selection.rescue_reason or "")
        )] += 1
    return counts


def _additive_priority_components(
    rescue: CandidateRecord,
    primary: CandidateRecord | None,
    reason: str,
    candidates: Sequence[CandidateRecord],
    represented_structured_buckets: set[StructuredDiversityBucket] | None = None,
    represented_ordinary_scenes: set[int] | None = None,
    promotion_lane_counts: PromotionLaneCounts | None = None,
) -> dict[str, Any]:
    rescue_tokens = _record_tokens(rescue, rescue=True)
    primary_tokens = _record_tokens(primary, rescue=True) if primary is not None else set()
    rescue_markers = canonical_markers(rescue_tokens)
    primary_markers = canonical_markers(primary_tokens)
    marker_gain = sum(
        len(rescue_markers[key] - primary_markers[key])
        for key in ("page", "option", "section", "status")
    )
    token_gain = len(rescue_tokens - primary_tokens)
    form_state_delta = _has_form_state_delta(rescue_tokens, primary_tokens)
    reason_priority = {
        "structured_delta": 6,
        "transition": 5,
        "evidence_marker": 4,
        "temporal_coverage": 3,
        "content_reference": 2,
        "token_gain": 1,
    }.get(reason, 0)
    temporal_gap = _nearest_same_scene_delta(rescue, candidates)
    if math.isinf(temporal_gap):
        temporal_gap = _nearest_candidate_delta(rescue, candidates)
    if math.isinf(temporal_gap):
        temporal_gap = 0.0
    categories = set(rescue.selection.structured_delta_categories)
    structured_delta = bool(categories)
    transition_proposal = (
        not structured_delta
        and rescue.selection.proposal_lane == "transition"
    )
    diversity_bucket = (
        _structured_diversity_bucket(rescue)
        if structured_delta
        else None
    )
    represented_structured_buckets = (
        represented_structured_buckets or set()
    )
    lane = _promotion_lane(reason)
    represented_ordinary_scenes = (
        represented_ordinary_scenes or set()
    )
    ordinary_scene_id = (
        int(rescue.temporal.scene_id)
        if lane == "ordinary"
        and rescue.temporal.scene_id is not None
        else None
    )
    lane_counts = (
        promotion_lane_counts
        if promotion_lane_counts is not None
        else _promotion_lane_counts(candidates)
    )
    settled_coverage = (
        rescue.selection.proposal_lane
        in {"temporal_coverage", "scene_coverage"}
        and float(
            rescue.selection.content_area_delta_score or 0.0
        )
        <= _SETTLED_CONTENT_AREA_DELTA_SCORE
    )
    page_signature_count = sum(
        1
        for signature in rescue.evidence.field_signature
        if signature.startswith("page:")
    )
    return {
        "priority_tier": (
            2
            if structured_delta
            else 1
            if transition_proposal
            else 0
        ),
        "structured_diversity_bucket": (
            list(diversity_bucket)
            if diversity_bucket is not None
            else None
        ),
        "structured_diversity_novel": bool(
            diversity_bucket is not None
            and diversity_bucket not in represented_structured_buckets
        ),
        "promotion_lane": lane,
        "promotion_lane_round": int(lane_counts.get(lane, 0)),
        "ordinary_scene_novel": bool(
            ordinary_scene_id is not None
            and ordinary_scene_id
            not in represented_ordinary_scenes
        ),
        "settled_coverage": bool(settled_coverage),
        "page_boundary_evidence": page_signature_count >= 2,
        "structured_blank_populated": (
            "blank_populated" in categories
        ),
        "structured_status_date": bool(
            {"status", "date"} & categories
        ),
        "structured_same_label_value": (
            "same_label_value" in categories
        ),
        "structured_page_section": bool(
            {"page", "section"} & categories
        ),
        "structured_changed_signature_count": int(
            rescue.selection.structured_changed_signature_count or 0
        ),
        "content_area_delta_score": float(
            rescue.selection.content_area_delta_score or 0.0
        ),
        "reason_priority": int(reason_priority),
        "marker_gain": int(marker_gain),
        "form_state_delta": bool(form_state_delta),
        "token_gain": int(token_gain),
        "temporal_gap": float(temporal_gap),
        "proxy_content_score": float(rescue.selection.proxy_content_score or 0.0),
    }


def _additive_priority_key(
    rescue: CandidateRecord,
    primary: CandidateRecord | None,
    reason: str,
    candidates: Sequence[CandidateRecord],
    represented_structured_buckets: set[StructuredDiversityBucket] | None = None,
    represented_ordinary_scenes: set[int] | None = None,
    promotion_lane_counts: PromotionLaneCounts | None = None,
) -> tuple[float, ...]:
    components = _additive_priority_components(
        rescue,
        primary,
        reason,
        candidates,
        represented_structured_buckets,
        represented_ordinary_scenes,
        promotion_lane_counts,
    )
    return (
        -float(components["promotion_lane_round"]),
        float(components["priority_tier"]),
        float(components["structured_diversity_novel"]),
        float(components["structured_blank_populated"]),
        float(components["structured_status_date"]),
        float(components["structured_same_label_value"]),
        float(components["structured_page_section"]),
        (
            float(components["page_boundary_evidence"])
            if components["priority_tier"] == 2
            else 0.0
        ),
        (
            float(components["settled_coverage"])
            if components["priority_tier"] == 2
            else 0.0
        ),
        float(components["structured_changed_signature_count"]),
        (
            float(components["content_area_delta_score"])
            if components["priority_tier"] == 2
            else -float(components["content_area_delta_score"])
            if components["priority_tier"] == 1
            else 0.0
        ),
        (
            float(components["proxy_content_score"])
            if components["priority_tier"]
            else 0.0
        ),
        (
            float(components["ordinary_scene_novel"])
            if not components["priority_tier"]
            else 0.0
        ),
        (
            float(components["form_state_delta"])
            if not components["priority_tier"]
            else 0.0
        ),
        (
            float(components["page_boundary_evidence"])
            if not components["priority_tier"]
            else 0.0
        ),
        (
            float(components["marker_gain"])
            if not components["priority_tier"]
            else 0.0
        ),
        (
            float(components["temporal_gap"])
            if not components["priority_tier"]
            else 0.0
        ),
        (
            float(components["form_state_delta"])
            if components["priority_tier"]
            else 0.0
        ),
        float(components["reason_priority"]),
        (
            float(components["marker_gain"])
            if components["priority_tier"]
            else 0.0
        ),
        float(components["token_gain"]),
        (
            float(components["temporal_gap"])
            if components["priority_tier"]
            else 0.0
        ),
        (
            float(components["proxy_content_score"])
            if not components["priority_tier"]
            else 0.0
        ),
        -float(rescue.timestamp),
        -float(rescue.sample_idx),
    )


def _with_structured_comparison(
    rescue: CandidateRecord,
    candidates: Sequence[CandidateRecord],
) -> tuple[CandidateRecord, CandidateRecord | None]:
    comparator = select_structured_comparator(rescue, candidates)
    if comparator is None:
        return rescue.with_selection(
            structured_delta_categories=(),
            structured_comparator_sample_idx=None,
            structured_comparator_timestamp=None,
            structured_changed_signature_count=0,
        ), None

    categories = structured_delta_categories(
        rescue.evidence.field_signature,
        comparator.evidence.field_signature,
    )
    return rescue.with_selection(
        structured_delta_categories=categories,
        structured_comparator_sample_idx=int(comparator.sample_idx),
        structured_comparator_timestamp=float(comparator.timestamp),
        structured_changed_signature_count=(
            structured_signature_change_count(
                rescue.evidence.field_signature,
                comparator.evidence.field_signature,
            )
        ),
    ), comparator


def _primary_for_rescue(
    rescue: CandidateRecord,
    candidates: Sequence[CandidateRecord],
    *,
    same_cluster: bool,
) -> CandidateRecord | None:
    if same_cluster:
        cluster = rescue.visual.clip_cluster
        pool = [c for c in candidates if c.visual.clip_cluster == cluster]
    else:
        scene = rescue.temporal.scene_id
        pool = [c for c in candidates if scene is not None and c.temporal.scene_id == scene]
    if not pool:
        return None
    return min(pool, key=lambda c: abs(float(c.timestamp) - float(rescue.timestamp)))


def _rescue_reason(
    rescue: CandidateRecord,
    primary: CandidateRecord,
    dwell_ids: Sequence[int],
    candidates: Sequence[CandidateRecord],
) -> str | None:
    rescue_tokens = _record_tokens(rescue, rescue=True)
    primary_tokens = _record_tokens(primary, rescue=True)
    rescue_score = float(rescue.selection.proxy_content_score or 0.0)
    primary_score = float(primary.selection.proxy_content_score or 0.0)
    nearest_same_scene_dt = _nearest_same_scene_delta(rescue, candidates)

    evidence_rescue = (
        _has_marker_signature(rescue_tokens)
        and (
            not has_evidence_markers(primary_tokens)
            or _has_form_state_delta(rescue_tokens, primary_tokens)
        )
        and (
            has_meaningful_evidence_for_retention(primary_tokens, rescue_tokens)
            or _has_form_state_delta(rescue_tokens, primary_tokens)
        )
        and not has_local_equivalent_coverage(rescue, candidates, dwell_ids)
    )
    if evidence_rescue:
        return "evidence_marker"

    has_content_reference = bool(_content_reference_tokens(rescue_tokens))
    token_gain = len(rescue_tokens - primary_tokens) >= 3 and len(rescue_tokens) >= 4
    clears_margin = rescue_score >= primary_score * 1.3 and rescue_score - primary_score >= 0.15
    if clears_margin and has_content_reference:
        return "content_reference"
    if clears_margin and token_gain:
        return "token_gain"
    temporally_distinct = nearest_same_scene_dt > 2.25
    evidence_or_dense_reference = (
        has_content_reference
        or len(rescue_tokens) >= 20
        or (
            _has_marker_signature(rescue_tokens)
            and not has_local_equivalent_coverage(rescue, candidates, dwell_ids)
        )
    )
    if temporally_distinct and evidence_or_dense_reference:
        return "temporal_coverage"
    return None


def _as_promoted_rescue(
    rescue: CandidateRecord,
    primary: CandidateRecord | None,
    *,
    origin: str,
    reason: str,
    priority: int,
    next_cluster: int,
) -> CandidateRecord:
    clip_cluster = rescue.visual.clip_cluster
    clip_cluster_size = rescue.visual.clip_cluster_size
    if primary is not None:
        clip_cluster = primary.visual.clip_cluster if primary.visual.clip_cluster is not None else clip_cluster
        clip_cluster_size = primary.visual.clip_cluster_size if primary.visual.clip_cluster_size is not None else clip_cluster_size
    elif clip_cluster is None:
        clip_cluster = next_cluster
    roles = set(rescue.lineage.lineage_roles)
    roles.add("rescue")
    return (
        rescue.with_visual(
            clip_cluster=clip_cluster,
            clip_cluster_size=clip_cluster_size or 1,
            cluster_role="rescue",
        )
        .with_selection(
            candidate_score=float(rescue.selection.proxy_content_score or 0.0),
            rescue_origin=origin,
            rescue_reason=reason,
            rescue_priority=int(priority),
            retention_reason=rescue.selection.retention_reason or "none",
        )
        .with_lineage(lineage_roles=tuple(sorted(roles)))
    )


def _temporally_local(
    rescue: CandidateRecord,
    candidate: CandidateRecord,
    *,
    tolerance: float = 2.25,
) -> bool:
    return (
        candidate.temporal.scene_id == rescue.temporal.scene_id
        and abs(float(candidate.timestamp) - float(rescue.timestamp)) <= tolerance
    )


def _nearest_competing_candidate(
    rescue: CandidateRecord,
    candidates: Sequence[CandidateRecord],
) -> CandidateRecord | None:
    if not candidates:
        return None
    rescue_scene = rescue.temporal.scene_id
    return min(
        candidates,
        key=lambda candidate: (
            0 if rescue_scene is not None and candidate.temporal.scene_id == rescue_scene else 1,
            abs(float(candidate.timestamp) - float(rescue.timestamp)),
            int(candidate.sample_idx),
        ),
    )


def _preflight_rejection_detail(
    rescue: CandidateRecord,
    candidates: Sequence[CandidateRecord],
    dwell_ids: Sequence[int],
    *,
    clip_embeddings: Any | None,
) -> dict[str, Any]:
    structured_comparator = select_structured_comparator(
        rescue,
        candidates,
    )
    if rescue.selection.structured_delta_categories:
        return {
            "eligible": True,
            "reason": "structured_delta",
            "rejection_branch": None,
            "rejection_reason": None,
            "competing_candidate": structured_comparator,
            "local_equivalent_coverage": False,
        }
    if rescue.selection.proposal_lane == "transition":
        return {
            "eligible": True,
            "reason": "transition",
            "rejection_branch": None,
            "rejection_reason": None,
            "competing_candidate": structured_comparator
            or _nearest_competing_candidate(rescue, candidates),
            "local_equivalent_coverage": False,
        }

    rescue_tokens = _record_tokens(rescue, rescue=True)
    rescue_idx = int(rescue.sample_idx)
    nearest = _nearest_competing_candidate(rescue, candidates)
    local_equivalent_coverage = has_local_equivalent_coverage(rescue, candidates, dwell_ids)

    for candidate in candidates:
        if not _temporally_local(rescue, candidate):
            continue
        candidate_tokens = _record_tokens(candidate, rescue=True)
        if _marker_redundant(rescue_tokens, candidate_tokens):
            return {
                "eligible": False,
                "reason": None,
                "rejection_branch": "redundancy",
                "rejection_reason": "temporally_local_marker_equivalent",
                "competing_candidate": candidate,
                "local_equivalent_coverage": local_equivalent_coverage,
            }
        if _clip_token_redundant(
            rescue_tokens,
            candidate_tokens,
            clip_embeddings,
            rescue_idx,
            int(candidate.sample_idx),
        ):
            return {
                "eligible": False,
                "reason": None,
                "rejection_branch": "redundancy",
                "rejection_reason": "temporally_local_clip_token_similarity",
                "competing_candidate": candidate,
                "local_equivalent_coverage": local_equivalent_coverage,
            }

    primary = _primary_for_rescue(rescue, candidates, same_cluster=False) or (candidates[0] if candidates else None)
    if primary:
        reason = _rescue_reason(rescue, primary, dwell_ids, candidates)
        if reason is None:
            return {
                "eligible": False,
                "reason": None,
                "rejection_branch": "rescue_reason",
                "rejection_reason": "no_rescue_reason",
                "competing_candidate": primary,
                "local_equivalent_coverage": local_equivalent_coverage,
            }
        if (
            reason == "evidence_marker"
            and _has_marker_signature(rescue_tokens)
            and local_equivalent_coverage
        ):
            return {
                "eligible": False,
                "reason": reason,
                "rejection_branch": "local_equivalent_coverage",
                "rejection_reason": "evidence_marker_local_equivalent_coverage",
                "competing_candidate": primary,
                "local_equivalent_coverage": local_equivalent_coverage,
            }
    else:
        reason = "content_reference" if _content_reference_tokens(rescue_tokens) else "evidence_marker"

    return {
        "eligible": True,
        "reason": reason,
        "rejection_branch": None,
        "rejection_reason": None,
        "competing_candidate": primary or nearest,
        "local_equivalent_coverage": local_equivalent_coverage,
    }


def rescue_promotion_preflight_report(
    base_candidates: tuple[CandidateRecord, ...],
    rescue_shortlist: tuple[CandidateRecord, ...],
    current_promoted: tuple[CandidateRecord, ...],
    dwell_ids: Sequence[int],
    rescue_budget: int,
    clip_embeddings: Any | None,
) -> dict[str, Any]:
    """Classify unpromoted rescue candidates without changing selection behavior."""
    base_candidates = _records(base_candidates)
    rescue_shortlist = _records(rescue_shortlist)
    current_promoted = _records(current_promoted)
    base_idxs = {int(candidate.sample_idx) for candidate in base_candidates}
    current_by_idx = {int(candidate.sample_idx): candidate for candidate in current_promoted}
    base_candidate_count = len(base_candidates)
    current_post_rescue_count = len(current_promoted)
    max_post_rescue_count = base_candidate_count + int(rescue_budget)
    additive_output_headroom = max(0, max_post_rescue_count - current_post_rescue_count)
    current_rescue_count = sum(1 for candidate in current_promoted if candidate.selection.rescue_origin)

    rows: list[dict[str, Any]] = []
    eligible_pending: list[
        tuple[
            CandidateRecord,
            CandidateRecord | None,
            str,
            dict[str, Any],
            dict[str, Any],
        ]
    ] = []
    represented_structured_buckets = (
        _represented_structured_diversity_buckets(current_promoted)
    )
    represented_ordinary_scenes = _represented_ordinary_scenes(
        current_promoted
    )
    promotion_lane_counts = _promotion_lane_counts(current_promoted)

    for rescue in rescue_shortlist:
        rescue, _structured_comparator = _with_structured_comparison(
            rescue,
            current_promoted,
        )
        rescue_idx = int(rescue.sample_idx)
        current = current_by_idx.get(rescue_idx)
        if rescue_idx in base_idxs or (current is not None and not current.selection.rescue_origin):
            status = "already_selected"
            outcome = "already_selected"
        elif current is not None and current.selection.rescue_origin:
            status = "already_promoted"
            outcome = "already_promoted"
        else:
            status = "unpromoted"
            outcome = None

        competing = _nearest_competing_candidate(rescue, current_promoted)
        local_equivalent_coverage = has_local_equivalent_coverage(rescue, current_promoted, dwell_ids)
        detail: dict[str, Any] = {
            "eligible": False,
            "reason": None,
            "rejection_branch": None,
            "rejection_reason": None,
            "competing_candidate": competing,
            "local_equivalent_coverage": local_equivalent_coverage,
        }
        if status == "unpromoted":
            detail = _preflight_rejection_detail(
                rescue,
                current_promoted,
                dwell_ids,
                clip_embeddings=clip_embeddings,
            )
            if detail["eligible"]:
                outcome = "eligible_pending"
            else:
                outcome = "predicate_rejected"

        competing = detail.get("competing_candidate") or competing
        competing_tokens = _record_tokens(competing, rescue=True) if competing is not None else set()
        rescue_tokens = _record_tokens(rescue, rescue=True)
        row = {
            "sample_idx": rescue_idx,
            "timestamp": float(rescue.timestamp),
            "origin": rescue.origin,
            "proxy_content_score": (
                float(rescue.selection.proxy_content_score)
                if rescue.selection.proxy_content_score is not None
                else None
            ),
            "current_status": status,
            "current_rescue_origin": current.selection.rescue_origin if current is not None else None,
            "current_rescue_reason": current.selection.rescue_reason if current is not None else None,
            "phase_a_eligible": bool(status == "unpromoted" and detail["eligible"]),
            "phase_a_rank": None,
            "above_additive_headroom_cut": None,
            "outcome": outcome,
            "binding_budget": "none",
            "rejection_branch": detail.get("rejection_branch"),
            "rejection_reason": detail.get("rejection_reason"),
            "nearest_competing_candidate_sample_idx": (
                int(competing.sample_idx) if competing is not None else None
            ),
            "nearest_competing_candidate_timestamp": (
                float(competing.timestamp) if competing is not None else None
            ),
            "token_jaccard": float(_jaccard(rescue_tokens, competing_tokens)) if competing is not None else None,
            "marker_equivalent": (
                bool(_marker_equivalent(rescue_tokens, competing_tokens)) if competing is not None else False
            ),
            "local_equivalent_coverage": bool(detail.get("local_equivalent_coverage", False)),
            "structured_delta_categories": list(
                rescue.selection.structured_delta_categories
            ),
            "structured_comparator_sample_idx": (
                rescue.selection.structured_comparator_sample_idx
            ),
            "structured_comparator_timestamp": (
                rescue.selection.structured_comparator_timestamp
            ),
            "structured_changed_signature_count": (
                rescue.selection.structured_changed_signature_count
            ),
        }
        if row["phase_a_eligible"]:
            reason = str(detail.get("reason") or "")
            primary = detail.get("competing_candidate")
            priority_components = _additive_priority_components(
                rescue,
                primary,
                reason,
                current_promoted,
                represented_structured_buckets,
                represented_ordinary_scenes,
                promotion_lane_counts,
            )
            row.update(priority_components)
            eligible_pending.append((
                rescue,
                primary,
                reason,
                row,
                detail,
            ))
        rows.append(row)

    predicted_ordered_eligible: list[dict[str, Any]] = []
    rank = 0
    while eligible_pending:
        best_index = max(
            range(len(eligible_pending)),
            key=lambda index: _additive_priority_key(
                eligible_pending[index][0],
                eligible_pending[index][1],
                eligible_pending[index][2],
                current_promoted,
                represented_structured_buckets,
                represented_ordinary_scenes,
                promotion_lane_counts,
            ),
        )
        rescue, primary, reason, row, detail = eligible_pending.pop(
            best_index
        )
        row.update(
            _additive_priority_components(
                rescue,
                primary,
                reason,
                current_promoted,
                represented_structured_buckets,
                represented_ordinary_scenes,
                promotion_lane_counts,
            )
        )
        rank += 1
        above_headroom = rank <= additive_output_headroom
        row["phase_a_rank"] = rank
        row["above_additive_headroom_cut"] = above_headroom
        row["outcome"] = "eligible_above_headroom" if above_headroom else "eligible_below_headroom"
        row["binding_budget"] = "additive_output_headroom"
        predicted_ordered_eligible.append({
            "sample_idx": int(row["sample_idx"]),
            "timestamp": float(row["timestamp"]),
            "phase_a_rank": rank,
            "above_additive_headroom_cut": above_headroom,
            "reason": detail.get("reason"),
        })
        if reason == "structured_delta":
            diversity_bucket = _structured_diversity_bucket(rescue)
            if diversity_bucket is not None:
                represented_structured_buckets.add(diversity_bucket)
        if _promotion_lane(reason) == "ordinary":
            scene_id = rescue.temporal.scene_id
            if scene_id is not None:
                represented_ordinary_scenes.add(int(scene_id))
        promotion_lane_counts[_promotion_lane(reason)] += 1

    return {
        "rescue_budget": int(rescue_budget),
        "base_candidate_count": base_candidate_count,
        "current_post_rescue_count": current_post_rescue_count,
        "max_post_rescue_count": max_post_rescue_count,
        "additive_output_headroom": additive_output_headroom,
        "current_rescue_count": current_rescue_count,
        "eligible_below_headroom_count": sum(
            1 for row in rows if row.get("outcome") == "eligible_below_headroom"
        ),
        "predicted_ordered_eligible": predicted_ordered_eligible,
        "candidate_rows": rows,
    }


def promote_rescue_candidates(
    candidates: Sequence[Mapping[str, Any] | CandidateRecord],
    rescue_shortlist: Sequence[Mapping[str, Any] | CandidateRecord],
    dwell_ids: Sequence[int],
    *,
    rescue_budget: int,
    clip_embeddings: Any | None = None,
) -> tuple[CandidateRecord, ...]:
    """Promote OCR-bearing rescue frames by bounded swap/additive rules."""
    promoted = list(_records(candidates))
    rescue_shortlist = _records(rescue_shortlist)
    if rescue_budget <= 0 or not rescue_shortlist:
        return tuple(sorted(promoted, key=lambda c: (float(c.timestamp), int(c.sample_idx))))

    next_cluster = max((int(c.visual.clip_cluster) for c in promoted if c.visual.clip_cluster is not None), default=-1) + 1
    used_idxs = {int(c.sample_idx) for c in promoted}
    consumed = 0
    priority = 0

    def maybe_swap(rescue: CandidateRecord, *, same_cluster: bool, origin: str) -> bool:
        nonlocal consumed, priority
        primary = _primary_for_rescue(rescue, promoted, same_cluster=same_cluster)
        if primary is None:
            return False
        rescue_tokens = _record_tokens(rescue, rescue=True)
        primary_tokens = _record_tokens(primary, rescue=True)
        if _has_form_state_delta(rescue_tokens, primary_tokens):
            return False
        if not _strictly_subsumes_tokens(rescue_tokens, primary_tokens):
            return False
        if (
            _temporally_local(rescue, primary)
            and _clip_token_redundant(
                rescue_tokens,
                primary_tokens,
                clip_embeddings,
                int(rescue.sample_idx),
                int(primary.sample_idx),
            )
        ):
            return False
        reason = _rescue_reason(rescue, primary, dwell_ids, promoted)
        if reason is None:
            return False
        if reason == "temporal_coverage":
            return False
        rescue_idx = int(rescue.sample_idx)
        primary_idx = int(primary.sample_idx)
        for i, candidate in enumerate(promoted):
            if int(candidate.sample_idx) == primary_idx:
                priority += 1
                promoted[i] = _as_promoted_rescue(
                    rescue,
                    primary,
                    origin=origin,
                    reason=reason,
                    priority=priority,
                    next_cluster=next_cluster,
                )
                used_idxs.discard(primary_idx)
                used_idxs.add(rescue_idx)
                consumed += 1
                return True
        return False

    def additive_candidate(
        rescue: CandidateRecord,
        represented_structured_buckets: set[
            StructuredDiversityBucket
        ],
        represented_ordinary_scenes: set[int],
        promotion_lane_counts: PromotionLaneCounts,
    ) -> tuple[
        tuple[float, ...],
        CandidateRecord,
        CandidateRecord | None,
        str,
    ] | None:
        rescue, structured_comparator = _with_structured_comparison(
            rescue,
            promoted,
        )
        rescue_idx = int(rescue.sample_idx)
        if rescue_idx in used_idxs:
            return None
        structured_delta = bool(
            rescue.selection.structured_delta_categories
        )
        transition_proposal = (
            not structured_delta
            and rescue.selection.proposal_lane == "transition"
        )
        rescue_tokens = _record_tokens(rescue, rescue=True)
        redundant = False
        if not structured_delta and not transition_proposal:
            for candidate in promoted:
                temporally_local = _temporally_local(rescue, candidate)
                candidate_tokens = _record_tokens(candidate, rescue=True)
                if temporally_local and _marker_redundant(
                    rescue_tokens,
                    candidate_tokens,
                ):
                    redundant = True
                    break
                if temporally_local and _clip_token_redundant(
                    rescue_tokens,
                    candidate_tokens,
                    clip_embeddings,
                    rescue_idx,
                    int(candidate.sample_idx),
                ):
                    redundant = True
                    break
        if redundant:
            return None

        primary = (
            structured_comparator
            if structured_delta
            else None
        )
        if primary is None:
            primary = _primary_for_rescue(
                rescue,
                promoted,
                same_cluster=False,
            ) or (promoted[0] if promoted else None)
        if structured_delta:
            reason = "structured_delta"
        elif transition_proposal:
            reason = "transition"
        elif primary:
            reason = _rescue_reason(rescue, primary, dwell_ids, promoted)
            if reason is None:
                return None
            if (
                reason == "evidence_marker"
                and _has_marker_signature(rescue_tokens)
                and has_local_equivalent_coverage(rescue, promoted, dwell_ids)
            ):
                return None
        else:
            reason = "content_reference" if _content_reference_tokens(rescue_tokens) else "evidence_marker"
        return (
            _additive_priority_key(
                rescue,
                primary if primary else None,
                reason,
                promoted,
                represented_structured_buckets,
                represented_ordinary_scenes,
                promotion_lane_counts,
            ),
            rescue,
            primary if primary else None,
            reason,
        )

    while consumed < rescue_budget:
        represented_structured_buckets = (
            _represented_structured_diversity_buckets(promoted)
        )
        represented_ordinary_scenes = _represented_ordinary_scenes(
            promoted
        )
        promotion_lane_counts = _promotion_lane_counts(promoted)
        additive_candidates = [
            candidate
            for rescue in rescue_shortlist
            if (
                candidate := additive_candidate(
                    rescue,
                    represented_structured_buckets,
                    represented_ordinary_scenes,
                    promotion_lane_counts,
                )
            )
            is not None
        ]
        if not additive_candidates:
            break
        _priority_key, rescue, primary, reason = max(
            additive_candidates,
            key=lambda item: item[0],
        )
        rescue_idx = int(rescue.sample_idx)
        priority += 1
        row = _as_promoted_rescue(
            rescue,
            primary,
            origin="additive_rescue",
            reason=reason,
            priority=priority,
            next_cluster=next_cluster,
        )
        if primary is None:
            next_cluster += 1
        promoted.append(row)
        used_idxs.add(rescue_idx)
        consumed += 1

    for rescue in rescue_shortlist:
        if consumed >= rescue_budget:
            break
        if int(rescue.sample_idx) in used_idxs:
            continue
        if maybe_swap(rescue, same_cluster=True, origin="same_cluster_swap"):
            continue

    for rescue in rescue_shortlist:
        if consumed >= rescue_budget:
            break
        if int(rescue.sample_idx) in used_idxs:
            continue
        if maybe_swap(rescue, same_cluster=False, origin="same_scene_generic_primary_swap"):
            continue

    return tuple(sorted(promoted, key=lambda c: (float(c.timestamp), int(c.sample_idx))))
