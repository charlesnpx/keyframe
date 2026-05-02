from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


DEFAULT_STAGE_ORDER = [
    "sampling",
    "proposal.pass1_primary",
    "proposal.cluster_alt",
    "proposal.rescue_shortlist",
    "evidence.rescue_ocr_batch",
    "selection.promoted_rescue_only",
    "selection.after_rescue",
    "evidence.final_ocr",
    "selection.retained_after_alt",
    "survival.after_near_dedupe",
    "survival.after_global_dedupe",
    "survival.after_low_info_filter",
    "survival.after_adjacent_dedupe",
    "survival.final_pre_cap",
    "survival.final_post_cap",
]


FAMILY_STAGE_ORDER = {
    "pass1": [
        "proposal.pass1_primary",
        "selection.after_rescue",
        "evidence.final_ocr",
        "selection.retained_after_alt",
        "survival.after_near_dedupe",
        "survival.after_global_dedupe",
        "survival.after_low_info_filter",
        "survival.after_adjacent_dedupe",
        "survival.final_pre_cap",
        "survival.final_post_cap",
    ],
    "cluster_alt": [
        "proposal.cluster_alt",
        "selection.after_rescue",
        "evidence.final_ocr",
        "selection.retained_after_alt",
        "survival.after_near_dedupe",
        "survival.after_global_dedupe",
        "survival.after_low_info_filter",
        "survival.after_adjacent_dedupe",
        "survival.final_pre_cap",
        "survival.final_post_cap",
    ],
    "rescue": [
        "proposal.rescue_shortlist",
        "evidence.rescue_ocr_batch",
        "selection.promoted_rescue_only",
        "selection.after_rescue",
        "evidence.final_ocr",
        "selection.retained_after_alt",
        "survival.after_near_dedupe",
        "survival.after_global_dedupe",
        "survival.after_low_info_filter",
        "survival.after_adjacent_dedupe",
        "survival.final_pre_cap",
        "survival.final_post_cap",
    ],
}


def load_targets(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if isinstance(payload, Mapping) and isinstance(payload.get("targets"), list):
        return [dict(item) for item in payload["targets"]]
    raise ValueError("QA target file must be a list or an object with a 'targets' list")


def _target_time(target: Mapping[str, Any]) -> float:
    return float(target["time"])


def _target_label(target: Mapping[str, Any]) -> str:
    return str(target.get("label", target["time"]))


def _target_tolerance(target: Mapping[str, Any]) -> float:
    return float(target.get("tolerance", 2.25))


def _stage_candidate_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    stages: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record.get("event") != "exit":
            continue
        payload = record.get("payload") or {}
        candidates = payload.get("candidates")
        if isinstance(candidates, list):
            stages[record["stage"]] = candidates
            continue
        samples = payload.get("samples")
        if isinstance(samples, list):
            stages[record["stage"]] = samples
    return stages


def _stage_summaries(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("event") != "exit":
            continue
        payload = record.get("payload") or {}
        stage = record["stage"]
        summary = {
            key: payload.get(key)
            for key in (
                "candidate_count",
                "duplicate_sample_idx_count",
                "duplicate_timestamp_count",
                "integrity_violations",
                "sample_count",
                "scene_count",
                "rescue_shortlist_count",
                "rescue_budget",
                "rescue_ocr_cap",
                "temporal_window_count",
                "legacy_proxy_dropped_count",
            )
            if key in payload
        }
        if summary:
            summaries[stage] = summary
    return summaries


def _promotion_preflight_payload(records: list[dict[str, Any]]) -> dict[str, Any]:
    for record in records:
        if record.get("event") != "decision":
            continue
        if record.get("stage") != "selection.rescue_promotion_preflight":
            continue
        payload = record.get("payload") or {}
        if isinstance(payload.get("value"), Mapping):
            payload = payload["value"]
        if payload.get("name") != "promotion_preflight":
            continue
        report = payload.get("payload") or {}
        return dict(report) if isinstance(report, Mapping) else {}
    return {}


def _promotion_preflight_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    report = _promotion_preflight_payload(records)
    if report:
        rows = report.get("candidate_rows")
        if isinstance(rows, list):
            return [dict(row) for row in rows if isinstance(row, Mapping)]
    return []


def _promotion_preflight_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    report = _promotion_preflight_payload(records)
    keys = (
        "rescue_budget",
        "base_candidate_count",
        "current_post_rescue_count",
        "max_post_rescue_count",
        "additive_output_headroom",
        "current_rescue_count",
        "eligible_below_headroom_count",
    )
    return {key: report.get(key) for key in keys if key in report}


def _nearest_in_stage(
    target_time: float,
    tolerance: float,
    candidates: list[Mapping[str, Any]],
) -> dict[str, Any]:
    best_any: tuple[float, Mapping[str, Any], str, float] | None = None
    best_direct_hit: tuple[float, Mapping[str, Any], str, float] | None = None
    best_lineage_hit: tuple[float, Mapping[str, Any], str, float] | None = None
    for candidate in candidates:
        direct_ts = candidate.get("timestamp")
        candidate_ts = float(direct_ts) if direct_ts is not None else None
        values: list[tuple[str, float]] = []
        if candidate_ts is not None:
            values.append(("timestamp", candidate_ts))
        for merged_ts in candidate.get("merged_timestamps", []) or []:
            try:
                values.append(("merged_timestamps", float(merged_ts)))
            except (TypeError, ValueError):
                continue
        for matched_via, ts in values:
            delta = abs(ts - target_time)
            item = (delta, candidate, matched_via, ts)
            if best_any is None or delta < best_any[0]:
                best_any = item
            if delta <= tolerance and matched_via == "timestamp":
                if best_direct_hit is None or delta < best_direct_hit[0]:
                    best_direct_hit = item
            elif delta <= tolerance and matched_via == "merged_timestamps":
                if best_lineage_hit is None or delta < best_lineage_hit[0]:
                    best_lineage_hit = item
    best = best_direct_hit or best_lineage_hit or best_any
    if best is None:
        return {"hit": False, "nearest_ts": None, "delta": None, "matched_via": "none"}
    delta, candidate, matched_via, nearest_ts = best
    return {
        "hit": delta <= tolerance,
        "nearest_ts": nearest_ts,
        "delta": delta,
        "candidate_timestamp": candidate.get("timestamp"),
        "matched_via": matched_via if delta <= tolerance else "none",
        "sample_idx": candidate.get("sample_idx"),
        "merged_from_sample_idxs": candidate.get("merged_from_sample_idxs", []),
        "origin": candidate.get("origin"),
        "cluster_role": candidate.get("cluster_role"),
        "rescue_origin": candidate.get("rescue_origin"),
        "rescue_reason": candidate.get("rescue_reason"),
        "retention_reason": candidate.get("retention_reason"),
    }


def _nearest_promotion_preflight(
    target_time: float,
    tolerance: float,
    rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    best: tuple[float, Mapping[str, Any]] | None = None
    for row in rows:
        timestamp = row.get("timestamp")
        if timestamp is None:
            continue
        try:
            delta = abs(float(timestamp) - target_time)
        except (TypeError, ValueError):
            continue
        if best is None or delta < best[0]:
            best = (delta, row)
    if best is None or best[0] > tolerance:
        return {"hit": False}
    delta, row = best
    return {
        "hit": True,
        "candidate_timestamp": row.get("timestamp"),
        "sample_idx": row.get("sample_idx"),
        "delta": delta,
        "outcome": row.get("outcome"),
        "phase_a_rank": row.get("phase_a_rank"),
        "above_additive_headroom_cut": row.get("above_additive_headroom_cut"),
        "rejection_branch": row.get("rejection_branch"),
        "rejection_reason": row.get("rejection_reason"),
        "nearest_competing_candidate_timestamp": row.get("nearest_competing_candidate_timestamp"),
        "reason_priority": row.get("reason_priority"),
        "marker_gain": row.get("marker_gain"),
        "form_state_delta": row.get("form_state_delta"),
        "token_gain": row.get("token_gain"),
        "temporal_gap": row.get("temporal_gap"),
    }


def _hit(stage_membership: Mapping[str, Mapping[str, Any]], stage: str) -> bool:
    return bool(stage_membership.get(stage, {}).get("hit"))


def _last_hit(
    stage_membership: Mapping[str, Mapping[str, Any]],
    stages: list[str],
) -> tuple[str | None, Mapping[str, Any] | None]:
    for stage in reversed(stages):
        hit = stage_membership.get(stage)
        if hit and hit.get("hit"):
            return stage, hit
    return None, None


def _best_family(stage_membership: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if _hit(stage_membership, "survival.final_post_cap"):
        hit = stage_membership["survival.final_post_cap"]
        return {
            "best_family": "final",
            "best_stage": "survival.final_post_cap",
            "best_matched_via": hit.get("matched_via"),
        }

    ranks = {stage: index for index, stage in enumerate(DEFAULT_STAGE_ORDER)}
    best: tuple[int, float, int, str, str, Mapping[str, Any]] | None = None
    family_priority = {"rescue": 0, "cluster_alt": 1, "pass1": 2}
    for family, stages in FAMILY_STAGE_ORDER.items():
        if family == "rescue":
            if not (
                _hit(stage_membership, "proposal.rescue_shortlist")
                or _hit(stage_membership, "selection.promoted_rescue_only")
            ):
                continue
            if not _hit(stage_membership, "selection.promoted_rescue_only"):
                stages = [
                    "proposal.rescue_shortlist",
                    "evidence.rescue_ocr_batch",
                ]
        elif family == "cluster_alt" and not _hit(stage_membership, "proposal.cluster_alt"):
            continue
        elif family == "pass1" and not _hit(stage_membership, "proposal.pass1_primary"):
            continue
        stage, hit = _last_hit(stage_membership, stages)
        if stage is None or hit is None:
            continue
        delta = float(hit.get("delta") if hit.get("delta") is not None else 1_000_000.0)
        matched_via_rank = 0 if hit.get("matched_via") == "timestamp" else 1
        candidate = (
            ranks.get(stage, -1),
            -delta,
            -matched_via_rank,
            family,
            stage,
            hit,
        )
        if best is None or candidate[:3] > best[:3] or (
            candidate[:3] == best[:3]
            and family_priority.get(family, 9) < family_priority.get(best[3], 9)
        ):
            best = candidate

    if best is not None:
        return {
            "best_family": best[3],
            "best_stage": best[4],
            "best_matched_via": best[5].get("matched_via"),
        }
    if _hit(stage_membership, "sampling"):
        hit = stage_membership["sampling"]
        return {
            "best_family": "sampling_only",
            "best_stage": "sampling",
            "best_matched_via": hit.get("matched_via"),
        }
    return {
        "best_family": "none",
        "best_stage": None,
        "best_matched_via": "none",
    }


def _bucket(stage_membership: Mapping[str, Mapping[str, Any]]) -> str:
    final = stage_membership.get("survival.final_post_cap", {})
    if final.get("hit"):
        return "hit_lineage_only" if final.get("matched_via") == "merged_timestamps" else "hit_direct"
    pre_cap = stage_membership.get("survival.final_pre_cap", {})
    if pre_cap.get("hit"):
        return "present_pre_cap_but_removed_by_cap_or_policy"
    if any(
        _hit(stage_membership, stage)
        for stage in (
            "survival.after_near_dedupe",
            "survival.after_global_dedupe",
            "survival.after_low_info_filter",
            "survival.after_adjacent_dedupe",
        )
    ):
        return "present_after_selection_but_removed_by_survival_policy"
    if _hit(stage_membership, "selection.retained_after_alt"):
        return "retained_after_alt_but_removed_by_survival_policy"
    if _hit(stage_membership, "selection.promoted_rescue_only"):
        return "promoted_rescue_but_not_retained_after_alt"
    if _hit(stage_membership, "proposal.rescue_shortlist"):
        if _hit(stage_membership, "evidence.rescue_ocr_batch"):
            return "rescue_ocrd_but_not_promoted"
        return "rescue_shortlisted_but_not_ocrd"
    if _hit(stage_membership, "evidence.final_ocr") or _hit(stage_membership, "selection.after_rescue"):
        return "selected_but_not_retained_after_alt"
    if _hit(stage_membership, "proposal.pass1_primary") or _hit(stage_membership, "proposal.cluster_alt"):
        return "coarse_proposal_exists_but_not_selected"
    if _hit(stage_membership, "sampling"):
        return "sampled_but_no_proposal_near_target"
    return "no_sample_near_target"


def build_debug_qa_trace(
    *,
    trace_records: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    video: str,
    stage_order: list[str] | None = None,
) -> dict[str, Any]:
    stage_order = stage_order or DEFAULT_STAGE_ORDER
    stage_records = _stage_candidate_records(trace_records)
    promotion_preflight_rows = _promotion_preflight_rows(trace_records)
    promotion_preflight_summary = _promotion_preflight_summary(trace_records)
    target_rows = []
    direct_hit_count = 0
    lineage_hit_count = 0
    miss_count = 0
    for target in targets:
        time = _target_time(target)
        tolerance = _target_tolerance(target)
        membership = {
            stage: _nearest_in_stage(time, tolerance, stage_records.get(stage, []))
            for stage in stage_order
            if stage in stage_records
        }
        bucket = _bucket(membership)
        best = _best_family(membership)
        if bucket == "hit_direct":
            direct_hit_count += 1
        elif bucket == "hit_lineage_only":
            lineage_hit_count += 1
        else:
            miss_count += 1
        final = membership.get("survival.final_post_cap", {})
        sampled = membership.get("sampling", {})
        target_rows.append({
            "label": _target_label(target),
            "time": time,
            "tolerance": tolerance,
            "anchor_tokens": target.get("anchor_tokens", []),
            "bucket": bucket,
            "best_family": best["best_family"],
            "best_stage": best["best_stage"],
            "best_matched_via": best["best_matched_via"],
            "nearest_sample_timestamp": sampled.get("nearest_ts"),
            "nearest_sample_delta": sampled.get("delta"),
            "nearest_final_timestamp": final.get("nearest_ts"),
            "nearest_final_delta": final.get("delta"),
            "promotion_preflight": _nearest_promotion_preflight(
                time,
                tolerance,
                promotion_preflight_rows,
            ),
            "stage_membership": membership,
        })
    summaries = _stage_summaries(trace_records)
    violations = []
    for summary in summaries.values():
        violations.extend(summary.get("integrity_violations", []) or [])
    return {
        "schema": 1,
        "video": video,
        "target_count": len(targets),
        "stages": stage_order,
        "targets": target_rows,
        "promotion_preflight_summary": promotion_preflight_summary,
        "fixture_summary": {
            "direct_hit_count": direct_hit_count,
            "lineage_only_hit_count": lineage_hit_count,
            "miss_count": miss_count,
        },
        "stage_summaries": summaries,
        "integrity_violations": violations,
    }


def write_debug_qa_trace(
    *,
    trace_records: list[dict[str, Any]],
    targets_path: str | Path,
    video: str,
    output_path: str | Path,
) -> Path:
    targets = load_targets(targets_path)
    payload = build_debug_qa_trace(
        trace_records=trace_records,
        targets=targets,
        video=video,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path
