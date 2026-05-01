"""Deterministic allocation and representative scoring helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from keyframe.dedupe import hamming


def coalesce_tiny_scenes(
    scenes: Sequence[tuple[int, int]],
    timestamps: Sequence[float],
    dhashes: Sequence[int] | Mapping[int, int],
    max_scene_seconds: float = 3.0,
    boundary_hamming_threshold: int = 18,
) -> list[tuple[int, int]]:
    """Merge tiny scene-detection fragments unless the visual boundary is large."""
    if not scenes:
        return []

    merged: list[tuple[int, int]] = []
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
    normalized_sharpness = min(sharpness / 1000.0, 3.0)
    transcript_bonus = min(max(float(transcript_density or 0.0), 0.0), 1.0) * 0.75

    if end_of_dwell_bonus is None:
        end_of_dwell_bonus = float(candidate.get("end_of_dwell_bonus", 0.0) or 0.0)
    dwell_bonus = min(max(float(end_of_dwell_bonus), 0.0), 1.0) * 0.5
    return normalized_sharpness + transcript_bonus + dwell_bonus
