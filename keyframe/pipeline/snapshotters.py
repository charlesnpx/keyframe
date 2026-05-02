from __future__ import annotations

from collections import Counter
from dataclasses import is_dataclass, asdict
from typing import Any, Mapping

from keyframe.pipeline.contracts import (
    CandidateBatch,
    FeatureOutput,
    ProposalOutput,
    SampleTable,
    SamplingOutput,
    TemporalOutput,
)


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _jsonish(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonish(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonish(v) for v in value]
    if isinstance(value, set):
        return sorted(_jsonish(v) for v in value)
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {
            "shape": [int(x) for x in value.shape],
            "dtype": str(value.dtype),
        }
    if is_dataclass(value):
        return _jsonish(asdict(value))
    return str(value)


def _count_duplicates(values: list[Any]) -> int:
    counts = Counter(value for value in values if value is not None)
    return sum(1 for count in counts.values() if count > 1)


def _candidate_id(stage: str, cand: Mapping[str, Any], index: int) -> str:
    sample_idx = _safe_int(cand.get("sample_idx"))
    timestamp = _safe_float(cand.get("timestamp"))
    if sample_idx is not None:
        return f"{stage}:{sample_idx}"
    if timestamp is not None:
        return f"{stage}:{timestamp:.3f}:{index}"
    return f"{stage}:unknown:{index}"


def _candidate_row(stage: str, cand: Mapping[str, Any], index: int) -> dict[str, Any]:
    row = {
        "candidate_id": str(cand.get("candidate_id") or _candidate_id(stage, cand, index)),
        "sample_idx": _safe_int(cand.get("sample_idx")),
        "frame_idx": _safe_int(cand.get("frame_idx")),
        "timestamp": _safe_float(cand.get("timestamp")),
        "origin": cand.get("origin"),
        "cluster_role": cand.get("cluster_role"),
        "scene_id": _safe_int(cand.get("scene_id")),
        "clip_cluster": _safe_int(cand.get("clip_cluster")),
        "dwell_id": _safe_int(cand.get("dwell_id")),
        "temporal_window_id": _safe_int(cand.get("temporal_window_id")),
        "rescue_origin": cand.get("rescue_origin"),
        "rescue_reason": cand.get("rescue_reason"),
        "retention_reason": cand.get("retention_reason", "none"),
        "retention_candidate_reason": cand.get("retention_candidate_reason"),
        "retention_rejected_reason": cand.get("retention_rejected_reason"),
        "proxy_content_score": _safe_float(cand.get("proxy_content_score")),
        "rescue_priority": _safe_int(cand.get("rescue_priority")),
        "merged_timestamps": [
            ts for ts in (_safe_float(v) for v in cand.get("merged_timestamps", []))
            if ts is not None
        ],
        "merged_from_sample_idxs": [
            idx for idx in (_safe_int(v) for v in cand.get("merged_from_sample_idxs", []))
            if idx is not None
        ],
        "ocr_text_present": bool(cand.get("ocr_text")),
        "ocr_token_count": _safe_int(cand.get("cleaned_token_count")) or len(cand.get("ocr_tokens", []) or []),
    }
    return row


def _candidate_violations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for row in rows:
        cid = row["candidate_id"]
        if row["timestamp"] is None:
            violations.append({"candidate_id": cid, "reason": "missing_timestamp"})
        if row["sample_idx"] is None:
            violations.append({"candidate_id": cid, "reason": "missing_sample_idx"})
        if row["merged_from_sample_idxs"] and not row["merged_timestamps"]:
            violations.append({"candidate_id": cid, "reason": "merged_idxs_without_timestamps"})
        if row["merged_from_sample_idxs"] and row["sample_idx"] is not None:
            if row["sample_idx"] not in row["merged_from_sample_idxs"]:
                violations.append({"candidate_id": cid, "reason": "sample_idx_missing_from_merged_idxs"})
        if row["merged_timestamps"] and row["timestamp"] is not None:
            if not any(abs(row["timestamp"] - ts) < 1e-6 for ts in row["merged_timestamps"]):
                violations.append({"candidate_id": cid, "reason": "timestamp_missing_from_merged_timestamps"})
        if row["rescue_origin"] and not row["rescue_reason"]:
            violations.append({"candidate_id": cid, "reason": "rescue_origin_without_reason"})
        score = row["proxy_content_score"]
        if row["rescue_origin"] and (score is None or score < 0.0 or score > 1.0):
            violations.append({"candidate_id": cid, "reason": "invalid_rescue_proxy_score"})
    return violations


def snapshot_candidate_batch(stage: str, batch: CandidateBatch | list[dict[str, Any]]) -> dict[str, Any]:
    candidates = batch.candidates if isinstance(batch, CandidateBatch) else batch
    rows = [_candidate_row(stage, cand, index) for index, cand in enumerate(candidates)]
    sample_idxs = [row["sample_idx"] for row in rows]
    timestamps = [row["timestamp"] for row in rows]
    return {
        "stage": stage,
        "candidate_count": len(rows),
        "duplicate_sample_idx_count": _count_duplicates(sample_idxs),
        "duplicate_timestamp_count": _count_duplicates(timestamps),
        "candidates": rows,
        "integrity_violations": _candidate_violations(rows),
    }


class SnapshotterRegistry:
    def snapshot(self, stage: str, value: Any) -> dict[str, Any]:
        if isinstance(value, CandidateBatch):
            return snapshot_candidate_batch(stage, value)
        if isinstance(value, ProposalOutput):
            return {
                "stage": stage,
                "candidate_count": len(value.candidates),
                "rescue_shortlist_count": len(value.rescue_shortlist),
                "proxy_row_count": len(value.proxy_rows),
                "rescue_budget": int(value.rescue_budget),
            }
        if isinstance(value, SamplingOutput):
            return self.snapshot(stage, value.samples)
        if isinstance(value, SampleTable):
            timestamps = [_safe_float(v) for v in value.timestamps]
            timestamps = [v for v in timestamps if v is not None]
            samples = [
                {
                    "sample_idx": index,
                    "frame_idx": _safe_int(frame_idx),
                    "timestamp": _safe_float(timestamp),
                    "origin": "sampled_frame",
                    "merged_timestamps": [],
                    "merged_from_sample_idxs": [],
                }
                for index, (timestamp, frame_idx) in enumerate(zip(value.timestamps, value.frame_indices))
            ]
            return {
                "stage": stage,
                "sample_count": len(value.timestamps),
                "timestamp_min": min(timestamps) if timestamps else None,
                "timestamp_max": max(timestamps) if timestamps else None,
                "timestamps": timestamps,
                "frame_indices": [_safe_int(v) for v in value.frame_indices],
                "samples": samples,
            }
        if isinstance(value, FeatureOutput):
            return {
                "stage": stage,
                "dhash_count": len(value.dhashes),
                "clip_embeddings": _jsonish(value.clip_embeddings),
            }
        if isinstance(value, TemporalOutput):
            return {
                "stage": stage,
                "scene_count": len(value.scenes),
                "scenes": _jsonish(value.scenes),
                "scene_coalescence": _jsonish(value.scene_coalescence),
                "cluster_allocs": _jsonish(value.cluster_allocs),
                "sample_cluster_count": len(value.sample_clusters),
                "sample_scene_count": len(value.sample_scenes),
                "sample_temporal_window_count": len(value.sample_temporal_windows),
                "sample_context": [
                    {
                        "sample_idx": int(sample_idx),
                        "scene_id": _safe_int(value.sample_scenes.get(sample_idx)),
                        "clip_cluster": _safe_int(value.sample_clusters.get(sample_idx)),
                        "temporal_window_id": _safe_int(value.sample_temporal_windows.get(sample_idx)),
                    }
                    for sample_idx in sorted(
                        set(value.sample_scenes) | set(value.sample_clusters) | set(value.sample_temporal_windows)
                    )
                ],
            }
        if isinstance(value, list) and all(isinstance(item, Mapping) for item in value):
            return snapshot_candidate_batch(stage, value)
        return {
            "stage": stage,
            "value": _jsonish(value),
        }
