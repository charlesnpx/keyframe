"""Deterministic manifest writer for extracted key frames."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def transcript_window(segments: list[dict[str, Any]] | None, timestamp: float, radius: float = 5.0) -> str:
    if not segments:
        return ""
    start = timestamp - radius
    end = timestamp + radius
    texts = [
        str(seg.get("text", "")).strip()
        for seg in segments
        if float(seg.get("end", 0.0)) >= start and float(seg.get("start", 0.0)) <= end
    ]
    return " ".join(t for t in texts if t)


def screen_type(tokens: list[str], caption: str = "") -> str:
    haystack = " ".join(tokens).lower() + " " + caption.lower()
    if any(term in haystack for term in ("approve", "approval", "submit", "confirm")):
        return "approval"
    if any(term in haystack for term in ("table", "row", "column", "spreadsheet")):
        return "table"
    if any(term in haystack for term in ("form", "field", "input", "dropdown")):
        return "form"
    if tokens:
        return "text_screen"
    return "visual"


def write_manifest(
    selected: list[dict[str, Any]],
    output_dir: str | Path,
    transcript_segments: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in selected:
        timestamp = float(item.get("timestamp", 0.0))
        tokens = list(item.get("ocr_tokens", []))
        rows.append({
            "filename": Path(item.get("path", "")).name,
            "timestamp": timestamp,
            "caption": item.get("caption", ""),
            "ocr_cache_source": item.get("ocr_cache_source"),
            "ocr_tokens": tokens,
            "transcript_window": transcript_window(transcript_segments, timestamp),
            "dhash": item.get("dhash_hex") or (f"{int(item['dhash']):016x}" if "dhash" in item else None),
            "merged_from_sample_idxs": item.get("merged_from_sample_idxs", [item.get("sample_idx")]),
            "merged_timestamps": item.get("merged_timestamps", [timestamp]),
            "screen_type": screen_type(tokens, item.get("caption", "")),
            "cluster_role": item.get("cluster_role"),
            "retention_reason": item.get("retention_reason", "none"),
            "retention_reasons_seen": item.get("retention_reasons_seen", [item.get("retention_reason", "none")]),
            "retention_candidate_reason": item.get("retention_candidate_reason"),
            "retention_rejected_reason": item.get("retention_rejected_reason"),
            "rescue_origin": item.get("rescue_origin"),
            "rescue_reason": item.get("rescue_reason"),
            "rescue_origins_seen": item.get("rescue_origins_seen", [item.get("rescue_origin")] if item.get("rescue_origin") else []),
            "rescue_priorities_seen": item.get("rescue_priorities_seen", [item.get("rescue_priority")] if item.get("rescue_priority") is not None else []),
            "proxy_content_score": item.get("proxy_content_score"),
            "rescue_priority": item.get("rescue_priority"),
            "dwell_id": item.get("dwell_id"),
            "temporal_window_id": item.get("temporal_window_id"),
            "lineage_roles": item.get("lineage_roles", [item.get("cluster_role")] if item.get("cluster_role") else []),
            "raw_token_count": item.get("raw_token_count", 0),
            "filtered_token_count": item.get("filtered_token_count", 0),
            "cleaned_token_count": item.get("cleaned_token_count", len(tokens)),
            "cleaning_attrition_ratio": item.get("cleaning_attrition_ratio", 0.0),
            "low_information_filter_reason": item.get("low_information_filter_reason"),
            "dedupe_stage": item.get("dedupe_stage"),
            "merge_reason": item.get("merge_reason"),
        })

    path = out / "manifest.json"
    payload = {"schema_version": 1, "frames": rows}
    if metadata:
        payload["metadata"] = metadata
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path
