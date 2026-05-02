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
    frames: Sequence[Image.Image],
    timestamps: Sequence[float],
    frame_indices: Sequence[int],
    candidates: Sequence[Mapping[str, Any] | CandidateRecord],
    pass1_clusters: int,
    *,
    sample_clusters: Mapping[int, int] | None = None,
    sample_scenes: Mapping[int, int] | None = None,
    frame_metrics: FrameMetricTable | None = None,
) -> tuple[tuple[CandidateRecord, ...], list[dict[str, float]], int, int, int, int, int]:
    """Rank non-selected sampled frames for bounded OCR rescue."""
    candidates = _records(candidates)
    proxy_rows = frame_metrics.to_proxy_rows() if frame_metrics is not None else proxy_content_scores(frames)
    candidate_idxs = {int(c.sample_idx) for c in candidates}
    eligible_mask = np.ones((len(frames),), dtype=bool)
    for idx in candidate_idxs:
        if 0 <= idx < len(eligible_mask):
            eligible_mask[idx] = False
    duration_seconds = max(timestamps) - min(timestamps) if timestamps else 0.0
    duration_floor = min(8, math.ceil(float(duration_seconds) / 90.0))
    rescue_budget = max(3, round(pass1_clusters * 0.35), duration_floor)
    scores = [row["proxy_content_score"] for row in proxy_rows]
    tau_proxy = float(np.percentile(scores, 75)) if scores else 0.0
    legacy_cap = min(60, rescue_budget * 4)

    window_seconds = rescue_window_seconds(timestamps)
    temporal_window_ids = assign_temporal_window_ids(
        timestamps,
        sample_scenes,
        window_seconds=window_seconds,
    )

    if frame_metrics is not None:
        content_deltas = [float(value) for value in frame_metrics.content_prev_delta]
    else:
        content_deltas = [0.0 for _ in frames]
        for idx in range(1, len(frames)):
            content_deltas[idx] = mean_abs_content_delta(frames[idx - 1], frames[idx])

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

    def build_content_area_delta_lane() -> list[dict[str, Any]]:
        lane: list[dict[str, Any]] = []
        selected: set[int] = set()
        if len(frames) < 3:
            return lane
        meaningful = [delta for delta in content_deltas if delta >= 3.0]
        if not meaningful:
            return lane
        threshold = max(3.0, float(np.percentile(meaningful, 70)))
        rows_by_idx = {int(row["sample_idx"]): row for row in ranked}
        scored: list[tuple[float, dict[str, Any]]] = []
        for idx in range(1, len(frames) - 1):
            if not eligible_mask[idx] or idx not in rows_by_idx:
                continue
            prev_delta = float(content_deltas[idx])
            next_delta = float(content_deltas[idx + 1])
            local_peak = prev_delta >= threshold and prev_delta >= next_delta
            settled_after_transition = prev_delta >= threshold and next_delta <= max(2.0, prev_delta * 0.60)
            text_band_change = abs(
                float(proxy_rows[idx].get("textline_score", 0.0))
                - float(proxy_rows[idx - 1].get("textline_score", 0.0))
            )
            if not (local_peak or settled_after_transition or text_band_change >= 0.20):
                continue
            row = rows_by_idx[idx]
            score = (
                float(row.get("content_area_delta_score", 0.0))
                + 0.25 * text_band_change
                + 0.10 * float(row.get("proxy_content_score", 0.0))
            )
            scored.append((score, row))

        scored.sort(
            key=lambda item: (
                item[0],
                -float(item[1]["timestamp"]),
                -int(item[1]["sample_idx"]),
            ),
            reverse=True,
        )
        cap = min(rescue_ocr_cap, max(rescue_budget * 2, temporal_window_count + scene_count, 8))
        for _score, row in scored:
            add_to_lane(lane, selected, row, "content_area_delta", cap)
        return lane

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

    def window_sort_key(key: tuple[Any, int]) -> tuple[int, int, float]:
        scene_id, window_id = key
        first_ts = min(float(row["timestamp"]) for row in by_time_window[key])
        scene_sort = int(scene_id) if scene_id is not None else -1
        return scene_sort, int(window_id), first_ts

    window_keys = sorted(by_time_window, key=window_sort_key)
    coverage_lane: list[dict[str, Any]] = []
    coverage_selected: set[int] = set()
    selected_ts_by_window: dict[tuple[Any, int], list[float]] = defaultdict(list)

    for key in window_keys:
        row = by_time_window[key][0]
        if add_to_lane(coverage_lane, coverage_selected, row, "temporal_coverage"):
            selected_ts_by_window[key].append(float(row["timestamp"]))

    for key in window_keys:
        for row in by_time_window[key]:
            sample_idx = int(row["sample_idx"])
            if sample_idx in coverage_selected:
                continue
            ts = float(row["timestamp"])
            if any(abs(ts - existing_ts) < 2.25 for existing_ts in selected_ts_by_window[key]):
                continue
            if add_to_lane(coverage_lane, coverage_selected, row, "temporal_coverage"):
                selected_ts_by_window[key].append(ts)
            break

    scene_window_counts: dict[Any, int] = defaultdict(int)
    for scene_id, _window_id in by_time_window:
        if scene_id is not None:
            scene_window_counts[scene_id] += 1

    for scene_id in sorted(by_scene, key=lambda value: int(value)):
        if scene_window_counts.get(scene_id, 0) <= 1:
            continue
        for row in by_scene[scene_id]:
            if add_to_lane(coverage_lane, coverage_selected, row, "scene_coverage"):
                break

    content_delta_lane = build_content_area_delta_lane()
    legacy_lane = build_legacy_proxy_lane()
    final_rows: list[dict[str, Any]] = []
    final_selected: set[int] = set()

    def add_final(row: Mapping[str, Any]) -> bool:
        return add_to_lane(
            final_rows,
            final_selected,
            row,
            str(row.get("proposal_lane") or "global_backfill"),
            rescue_ocr_cap,
        )

    for lane_rows in (content_delta_lane, legacy_lane, coverage_lane):
        for row in lane_rows:
            add_final(row)

    for row in ranked:
        if len(final_rows) >= rescue_ocr_cap:
            break
        add_final({**dict(row), "proposal_lane": "global_backfill"})

    final_sample_idxs = {int(row["sample_idx"]) for row in final_rows}
    legacy_proxy_dropped_count = sum(
        1 for row in legacy_lane if int(row["sample_idx"]) not in final_sample_idxs
    )

    return tuple(
        as_candidate_record(row, origin="rescue_shortlist")
        for row in final_rows[:rescue_ocr_cap]
    ), proxy_rows, rescue_budget, rescue_ocr_cap, temporal_window_count, scene_count, legacy_proxy_dropped_count


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


def _additive_priority_components(
    rescue: CandidateRecord,
    primary: CandidateRecord | None,
    reason: str,
    candidates: Sequence[CandidateRecord],
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
    return {
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
) -> tuple[float, ...]:
    components = _additive_priority_components(rescue, primary, reason, candidates)
    return (
        float(components["form_state_delta"]),
        float(components["reason_priority"]),
        float(components["marker_gain"]),
        float(components["token_gain"]),
        float(components["temporal_gap"]),
        float(components["proxy_content_score"]),
        -float(rescue.timestamp),
        -float(rescue.sample_idx),
    )


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
    eligible_pending: list[tuple[tuple[float, ...], dict[str, Any], dict[str, Any]]] = []

    for rescue in rescue_shortlist:
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
        }
        if row["phase_a_eligible"]:
            reason = str(detail.get("reason") or "")
            primary = detail.get("competing_candidate")
            priority_components = _additive_priority_components(rescue, primary, reason, current_promoted)
            row.update(priority_components)
            eligible_pending.append((
                _additive_priority_key(rescue, primary, reason, current_promoted),
                row,
                detail,
            ))
        rows.append(row)

    predicted_ordered_eligible: list[dict[str, Any]] = []
    eligible_pending.sort(key=lambda item: item[0], reverse=True)
    for rank, (_priority_key, row, detail) in enumerate(eligible_pending, start=1):
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

    additive_candidates: list[tuple[tuple[float, ...], CandidateRecord, CandidateRecord | None, str]] = []
    for rescue in rescue_shortlist:
        if consumed >= rescue_budget:
            break
        rescue_idx = int(rescue.sample_idx)
        if rescue_idx in used_idxs:
            continue
        rescue_tokens = _record_tokens(rescue, rescue=True)
        redundant = False
        for candidate in promoted:
            temporally_local = _temporally_local(rescue, candidate)
            candidate_tokens = _record_tokens(candidate, rescue=True)
            if temporally_local and _marker_redundant(rescue_tokens, candidate_tokens):
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
            continue

        primary = _primary_for_rescue(rescue, promoted, same_cluster=False) or (promoted[0] if promoted else None)
        if primary:
            reason = _rescue_reason(rescue, primary, dwell_ids, promoted)
            if reason is None:
                continue
            if (
                reason == "evidence_marker"
                and _has_marker_signature(rescue_tokens)
                and has_local_equivalent_coverage(rescue, promoted, dwell_ids)
            ):
                continue
        else:
            reason = "content_reference" if _content_reference_tokens(rescue_tokens) else "evidence_marker"
        additive_candidates.append((
            _additive_priority_key(rescue, primary if primary else None, reason, promoted),
            rescue,
            primary if primary else None,
            reason,
        ))

    additive_candidates.sort(key=lambda item: item[0], reverse=True)
    for _priority_key, rescue, primary, reason in additive_candidates:
        if consumed >= rescue_budget:
            break
        rescue_idx = int(rescue.sample_idx)
        if rescue_idx in used_idxs:
            continue
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
