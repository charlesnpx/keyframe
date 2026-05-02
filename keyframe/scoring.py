"""Deterministic allocation and representative scoring helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from collections import defaultdict
import math
from typing import Any

import numpy as np
from PIL import Image

from keyframe.dedupe import (
    canonical_markers,
    has_evidence_markers,
    has_meaningful_evidence_for_retention,
    hamming,
)


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


def score_candidate_for_rep(
    candidate: Mapping[str, Any],
    image: Any | None = None,
    transcript_density: float = 0.0,
    end_of_dwell_bonus: float | None = None,
) -> float:
    """Score a candidate for representative selection."""
    sharpness = candidate.get("sharpness")
    if sharpness is None and image is not None:
        from keyframe.frames import _laplacian_sharpness

        sharpness = _laplacian_sharpness(image)

    sharpness = float(sharpness or 0.0)
    normalized_sharpness = min(sharpness / 1000.0, 1.5)
    transcript_bonus = min(max(float(transcript_density or 0.0), 0.0), 1.0) * 0.75

    if end_of_dwell_bonus is None:
        end_of_dwell_bonus = float(candidate.get("end_of_dwell_bonus", 0.0) or 0.0)
    dwell_bonus = min(max(float(end_of_dwell_bonus), 0.0), 1.0) * 0.5
    return normalized_sharpness + transcript_bonus + dwell_bonus


def _sobel_edges(gray_array: np.ndarray) -> np.ndarray:
    padded = np.pad(gray_array.astype(np.float32), 1, mode="edge")
    gx = (
        -padded[:-2, :-2] + padded[:-2, 2:]
        - 2 * padded[1:-1, :-2] + 2 * padded[1:-1, 2:]
        - padded[2:, :-2] + padded[2:, 2:]
    )
    gy = (
        -padded[:-2, :-2] - 2 * padded[:-2, 1:-1] - padded[:-2, 2:]
        + padded[2:, :-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:]
    )
    return np.hypot(gx, gy)


def proxy_frame_components(image: Image.Image) -> dict[str, float]:
    """Return raw cheap content metrics for all-sampled-frame rescue ranking."""
    gray = image.convert("L").resize((160, 90), Image.Resampling.LANCZOS)
    arr = np.asarray(gray, dtype=np.float32)
    edges = _sobel_edges(arr)
    total = max(arr.size, 1)
    dark_ratio = float(np.count_nonzero(arr <= 24) / total)
    bright_ratio = float(np.count_nonzero(arr >= 232) / total)

    if edges.size:
        threshold = max(float(np.percentile(edges, 75)), 12.0)
        edge_mask = edges >= threshold
        band_density = []
        for start in range(0, edge_mask.shape[0], 3):
            band = edge_mask[start:start + 3, :]
            if band.size:
                band_density.append(float(np.count_nonzero(band) / band.size))
        textline_score = sum(1 for density in band_density if density >= 0.08) / max(len(band_density), 1)
        edge_score = float(edges.mean())
    else:
        textline_score = 0.0
        edge_score = 0.0

    histogram = gray.histogram()
    entropy = 0.0
    for count in histogram:
        if not count:
            continue
        p = count / total
        entropy -= p * math.log2(p)

    return {
        "textline_score": float(textline_score),
        "edge_score": float(edge_score),
        "entropy": float(entropy),
        "dark_ratio": dark_ratio,
        "bright_ratio": bright_ratio,
    }


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
    components = [proxy_frame_components(frame) for frame in frames]
    norm_textline = _normalize([c["textline_score"] for c in components])
    norm_edge = _normalize([c["edge_score"] for c in components])
    norm_entropy = _normalize([c["entropy"] for c in components])

    scored: list[dict[str, float]] = []
    for comp, textline, edge, entropy in zip(components, norm_textline, norm_edge, norm_entropy):
        blank_penalty = (
            0.5 * max(0.0, comp["dark_ratio"] - 0.7)
            + 0.5 * max(0.0, comp["bright_ratio"] - 0.7)
        )
        score = 0.45 * textline + 0.30 * edge + 0.25 * entropy - blank_penalty
        row = dict(comp)
        row["proxy_content_score"] = min(max(float(score), 0.0), 1.0)
        row["normalized_textline_score"] = float(textline)
        row["normalized_edge_score"] = float(edge)
        row["normalized_entropy"] = float(entropy)
        row["blank_penalty"] = float(blank_penalty)
        scored.append(row)
    return scored


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
    candidates: Sequence[Mapping[str, Any]],
    pass1_clusters: int,
    *,
    sample_clusters: Mapping[int, int] | None = None,
    sample_scenes: Mapping[int, int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, float]], int]:
    """Rank non-selected sampled frames for bounded OCR rescue."""
    proxy_rows = proxy_content_scores(frames)
    candidate_idxs = {int(c["sample_idx"]) for c in candidates}
    rescue_budget = max(3, round(pass1_clusters * 0.35))
    scores = [row["proxy_content_score"] for row in proxy_rows]
    tau_proxy = float(np.percentile(scores, 75)) if scores else 0.0
    cap = min(60, rescue_budget * 4)

    window_seconds = rescue_window_seconds(timestamps)
    temporal_window_ids = assign_temporal_window_ids(
        timestamps,
        sample_scenes,
        window_seconds=window_seconds,
    )

    ranked: list[dict[str, Any]] = []
    for sample_idx, metrics in enumerate(proxy_rows):
        if sample_idx in candidate_idxs:
            continue
        ranked.append({
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
            "rescue_priority": 0,
            **metrics,
        })

    ranked.sort(
        key=lambda row: (
            float(row["proxy_content_score"]),
            -float(row["timestamp"]),
        ),
        reverse=True,
    )

    shortlist: list[dict[str, Any]] = []
    selected: set[int] = set()

    def add(row: Mapping[str, Any]) -> None:
        if len(shortlist) >= cap:
            return
        sample_idx = int(row["sample_idx"])
        if sample_idx in selected:
            return
        shortlist.append(dict(row))
        selected.add(sample_idx)

    global_quota = min(cap, max(rescue_budget, cap // 4))
    for row in ranked:
        if float(row["proxy_content_score"]) >= tau_proxy:
            add(row)
        if len(shortlist) >= global_quota:
            break

    by_time_window: dict[tuple[Any, int], list[dict[str, Any]]] = defaultdict(list)
    for row in ranked:
        scene_id = row.get("scene_id")
        window_id = int(row.get("temporal_window_id") or 0)
        by_time_window[(scene_id, window_id)].append(row)

    for key in sorted(
        by_time_window,
        key=lambda item: (
            -max(float(row["proxy_content_score"]) for row in by_time_window[item]),
            -len(by_time_window[item]),
        ),
    ):
        if len(shortlist) >= cap:
            break
        added_in_window: list[float] = []
        for row in by_time_window[key]:
            ts = float(row["timestamp"])
            if any(abs(ts - existing_ts) < 2.25 for existing_ts in added_in_window):
                continue
            before = len(shortlist)
            add(row)
            if len(shortlist) > before:
                added_in_window.append(ts)
            if len(added_in_window) >= 3 or len(shortlist) >= cap:
                break

    all_scenes = set(sample_scenes.values()) if sample_scenes else {
        row.get("scene_id") for row in ranked if row.get("scene_id") is not None
    }
    by_scene: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in ranked:
        scene_id = row.get("scene_id")
        if scene_id in all_scenes:
            by_scene[scene_id].append(row)

    for scene_id in sorted(by_scene, key=lambda value: -1 if value is None else int(value)):
        if len(shortlist) >= cap:
            break
        add(by_scene[scene_id][0])

    for row in ranked:
        if len(shortlist) >= cap:
            break
        add(row)

    return shortlist[:cap], proxy_rows, rescue_budget


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


def _same_marker_covered(
    rescue: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    dwell_ids: Sequence[int],
    *,
    scene_only: bool = False,
) -> bool:
    rescue_tokens = set(rescue.get("rescue_tokens", rescue.get("ocr_tokens", [])))
    signature = _marker_signature(rescue_tokens)
    if not signature:
        return False
    rescue_idx = int(rescue["sample_idx"])
    rescue_dwell = dwell_ids[rescue_idx] if 0 <= rescue_idx < len(dwell_ids) else None
    rescue_scene = rescue.get("scene_id")
    for candidate in candidates:
        candidate_tokens = set(candidate.get("rescue_tokens", candidate.get("ocr_tokens", [])))
        if _marker_signature(candidate_tokens) != signature:
            continue
        if scene_only:
            if rescue_scene is not None and candidate.get("scene_id") == rescue_scene:
                return True
            continue
        candidate_idx = int(candidate["sample_idx"])
        candidate_dwell = dwell_ids[candidate_idx] if 0 <= candidate_idx < len(dwell_ids) else None
        if rescue_dwell is not None and candidate_dwell == rescue_dwell:
            return True
        if abs(float(rescue["timestamp"]) - float(candidate["timestamp"])) <= 2.0:
            return True
    return False


def _marker_equivalent(tokens_a: set[str], tokens_b: set[str]) -> bool:
    signature_a = _marker_signature(tokens_a)
    return bool(signature_a) and signature_a == _marker_signature(tokens_b)


def _candidate_dwell_id(candidate: Mapping[str, Any], dwell_ids: Sequence[int]) -> int | None:
    if candidate.get("dwell_id") is not None:
        return int(candidate["dwell_id"])
    sample_idx = int(candidate["sample_idx"])
    if 0 <= sample_idx < len(dwell_ids):
        return int(dwell_ids[sample_idx])
    return None


def has_local_equivalent_coverage(
    rescue: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    dwell_ids: Sequence[int],
    *,
    tolerance: float = 2.25,
) -> bool:
    rescue_scene = rescue.get("scene_id")
    rescue_tokens = set(rescue.get("rescue_tokens", rescue.get("ocr_tokens", [])))
    rescue_dwell = _candidate_dwell_id(rescue, dwell_ids)
    rescue_window = rescue.get("temporal_window_id")
    for candidate in candidates:
        if candidate.get("scene_id") != rescue_scene:
            continue
        candidate_tokens = set(candidate.get("rescue_tokens", candidate.get("ocr_tokens", [])))
        if not _marker_equivalent(candidate_tokens, rescue_tokens):
            continue
        candidate_dwell = _candidate_dwell_id(candidate, dwell_ids)
        same_dwell = rescue_dwell is not None and candidate_dwell == rescue_dwell
        near_time = abs(float(candidate["timestamp"]) - float(rescue["timestamp"])) <= tolerance
        same_window = (
            rescue_window is not None
            and candidate.get("temporal_window_id") is not None
            and int(candidate["temporal_window_id"]) == int(rescue_window)
        )
        if same_dwell or near_time or same_window:
            return True
    return False


def _primary_for_rescue(
    rescue: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    same_cluster: bool,
) -> dict[str, Any] | None:
    if same_cluster:
        cluster = rescue.get("clip_cluster")
        pool = [dict(c) for c in candidates if c.get("clip_cluster") == cluster]
    else:
        scene = rescue.get("scene_id")
        pool = [dict(c) for c in candidates if scene is not None and c.get("scene_id") == scene]
    if not pool:
        return None
    return min(pool, key=lambda c: abs(float(c.get("timestamp", 0.0)) - float(rescue.get("timestamp", 0.0))))


def _rescue_reason(
    rescue: Mapping[str, Any],
    primary: Mapping[str, Any],
    dwell_ids: Sequence[int],
    candidates: Sequence[Mapping[str, Any]],
) -> str | None:
    rescue_tokens = set(rescue.get("rescue_tokens", rescue.get("ocr_tokens", [])))
    primary_tokens = set(primary.get("rescue_tokens", primary.get("ocr_tokens", [])))
    rescue_score = float(rescue.get("proxy_content_score", 0.0) or 0.0)
    primary_score = float(primary.get("proxy_content_score", 0.0) or 0.0)
    rescue_scene = rescue.get("scene_id")
    same_scene_deltas = [
        abs(float(candidate.get("timestamp", 0.0)) - float(rescue.get("timestamp", 0.0)))
        for candidate in candidates
        if rescue_scene is not None and candidate.get("scene_id") == rescue_scene
    ]
    nearest_same_scene_dt = min(same_scene_deltas, default=float("inf"))

    evidence_rescue = (
        _has_marker_signature(rescue_tokens)
        and not has_evidence_markers(primary_tokens)
        and has_meaningful_evidence_for_retention(primary_tokens, rescue_tokens)
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
    rescue: Mapping[str, Any],
    primary: Mapping[str, Any] | None,
    *,
    origin: str,
    reason: str,
    priority: int,
    next_cluster: int,
) -> dict[str, Any]:
    row = dict(rescue)
    if primary is not None:
        row["clip_cluster"] = primary.get("clip_cluster", row.get("clip_cluster", next_cluster))
        row["clip_cluster_size"] = primary.get("clip_cluster_size", row.get("clip_cluster_size", 1))
    elif row.get("clip_cluster") is None:
        row["clip_cluster"] = next_cluster
    row["cluster_role"] = "rescue"
    row["candidate_score"] = float(row.get("proxy_content_score", 0.0) or 0.0)
    row["rescue_origin"] = origin
    row["rescue_reason"] = reason
    row["rescue_priority"] = int(priority)
    row.setdefault("retention_reason", "none")
    roles = set(row.get("lineage_roles", []))
    roles.add("rescue")
    row["lineage_roles"] = sorted(roles)
    return row


def promote_rescue_candidates(
    candidates: Sequence[Mapping[str, Any]],
    rescue_shortlist: Sequence[Mapping[str, Any]],
    dwell_ids: Sequence[int],
    *,
    rescue_budget: int,
    clip_embeddings: Any | None = None,
) -> list[dict[str, Any]]:
    """Promote OCR-bearing rescue frames by bounded swap/additive rules."""
    promoted = [dict(c) for c in candidates]
    if rescue_budget <= 0 or not rescue_shortlist:
        return sorted(promoted, key=lambda c: (float(c.get("timestamp", 0.0)), int(c.get("sample_idx", 0))))

    next_cluster = max((int(c.get("clip_cluster", -1)) for c in promoted if c.get("clip_cluster") is not None), default=-1) + 1
    used_idxs = {int(c["sample_idx"]) for c in promoted}
    consumed = 0
    priority = 0

    def maybe_swap(rescue: Mapping[str, Any], *, same_cluster: bool, origin: str) -> bool:
        nonlocal consumed, priority
        primary = _primary_for_rescue(rescue, promoted, same_cluster=same_cluster)
        if primary is None:
            return False
        rescue_tokens = set(rescue.get("rescue_tokens", rescue.get("ocr_tokens", [])))
        primary_tokens = set(primary.get("rescue_tokens", primary.get("ocr_tokens", [])))
        temporally_local = (
            primary.get("scene_id") == rescue.get("scene_id")
            and (
                abs(float(primary.get("timestamp", 0.0)) - float(rescue.get("timestamp", 0.0))) <= 2.25
                or (
                    rescue.get("temporal_window_id") is not None
                    and primary.get("temporal_window_id") is not None
                    and int(primary["temporal_window_id"]) == int(rescue["temporal_window_id"])
                )
                or (
                    _candidate_dwell_id(primary, dwell_ids) is not None
                    and _candidate_dwell_id(primary, dwell_ids) == _candidate_dwell_id(rescue, dwell_ids)
                )
            )
        )
        if (
            temporally_local
            and _clip_cosine(clip_embeddings, int(rescue["sample_idx"]), int(primary["sample_idx"])) >= 0.93
            and _jaccard(rescue_tokens, primary_tokens) >= 0.7
        ):
            return False
        reason = _rescue_reason(rescue, primary, dwell_ids, promoted)
        if reason is None:
            return False
        if reason == "temporal_coverage":
            return False
        rescue_idx = int(rescue["sample_idx"])
        primary_idx = int(primary["sample_idx"])
        for i, candidate in enumerate(promoted):
            if int(candidate["sample_idx"]) == primary_idx:
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

    for rescue in rescue_shortlist:
        if consumed >= rescue_budget:
            break
        if int(rescue["sample_idx"]) in used_idxs:
            continue
        if maybe_swap(rescue, same_cluster=True, origin="same_cluster_swap"):
            continue

    for rescue in rescue_shortlist:
        if consumed >= rescue_budget:
            break
        if int(rescue["sample_idx"]) in used_idxs:
            continue
        if maybe_swap(rescue, same_cluster=False, origin="same_scene_generic_primary_swap"):
            continue

    for rescue in rescue_shortlist:
        if consumed >= rescue_budget:
            break
        rescue_idx = int(rescue["sample_idx"])
        if rescue_idx in used_idxs:
            continue
        rescue_tokens = set(rescue.get("rescue_tokens", rescue.get("ocr_tokens", [])))
        redundant = False
        for candidate in promoted:
            temporally_local = (
                candidate.get("scene_id") == rescue.get("scene_id")
                and (
                    abs(float(candidate.get("timestamp", 0.0)) - float(rescue.get("timestamp", 0.0))) <= 2.25
                    or (
                        rescue.get("temporal_window_id") is not None
                        and candidate.get("temporal_window_id") is not None
                        and int(candidate["temporal_window_id"]) == int(rescue["temporal_window_id"])
                    )
                    or (
                        _candidate_dwell_id(candidate, dwell_ids) is not None
                        and _candidate_dwell_id(candidate, dwell_ids) == _candidate_dwell_id(rescue, dwell_ids)
                    )
                )
            )
            if temporally_local and _marker_equivalent(
                set(candidate.get("rescue_tokens", candidate.get("ocr_tokens", []))),
                rescue_tokens,
            ):
                redundant = True
                break
            candidate_tokens = set(candidate.get("rescue_tokens", candidate.get("ocr_tokens", [])))
            if (
                temporally_local
                and _clip_cosine(clip_embeddings, rescue_idx, int(candidate["sample_idx"])) >= 0.93
                and _jaccard(rescue_tokens, candidate_tokens) >= 0.7
            ):
                redundant = True
                break
        if redundant:
            continue

        primary = _primary_for_rescue(rescue, promoted, same_cluster=False) or (promoted[0] if promoted else {})
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
        priority += 1
        row = _as_promoted_rescue(
            rescue,
            primary if primary else None,
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

    return sorted(promoted, key=lambda c: (float(c.get("timestamp", 0.0)), int(c.get("sample_idx", 0))))
