import json

from keyframe.pipeline.contracts import CandidateBatch
from keyframe.pipeline.qa_targets import build_debug_qa_trace
from keyframe.pipeline.trace import SnapshotTraceSink


def test_snapshot_trace_does_not_store_mutable_references():
    sink = SnapshotTraceSink()
    batch = CandidateBatch(
        stage="test",
        candidates=[{"sample_idx": 1, "timestamp": 10.0}],
    )

    sink.exit("test", batch)
    batch.candidates[0]["timestamp"] = 999.0

    saved = sink.records[0]["payload"]["candidates"][0]
    assert saved["timestamp"] == 10.0


def test_candidate_snapshot_is_json_safe_and_reports_duplicates_and_violations():
    sink = SnapshotTraceSink()
    batch = CandidateBatch(
        stage="test",
        candidates=[
            {
                "sample_idx": 1,
                "timestamp": 10.0,
                "merged_from_sample_idxs": [1, 2],
                "rescue_origin": "additive_rescue",
                "proxy_content_score": 0.5,
            },
            {"sample_idx": 1, "timestamp": 10.0},
        ],
    )

    sink.exit("test", batch)
    json.dumps(sink.records)
    payload = sink.records[0]["payload"]

    assert payload["duplicate_sample_idx_count"] == 1
    assert payload["duplicate_timestamp_count"] == 1
    assert {"candidate_id": "test:1", "reason": "merged_idxs_without_timestamps"} in payload["integrity_violations"]
    assert {"candidate_id": "test:1", "reason": "rescue_origin_without_reason"} in payload["integrity_violations"]


def test_candidate_snapshot_includes_rescue_shortlist_metadata():
    sink = SnapshotTraceSink()
    batch = CandidateBatch(
        stage="proposal.rescue_shortlist",
        candidates=[{"sample_idx": 1, "timestamp": 10.0}],
        metadata={
            "rescue_budget": 3,
            "rescue_ocr_cap": 40,
            "temporal_window_count": 4,
            "scene_count": 2,
        },
    )

    sink.exit("proposal.rescue_shortlist", batch)
    payload = sink.records[0]["payload"]

    assert payload["candidate_count"] == 1
    assert payload["rescue_budget"] == 3
    assert payload["rescue_ocr_cap"] == 40
    assert payload["temporal_window_count"] == 4
    assert payload["scene_count"] == 2


def test_debug_qa_trace_reports_direct_and_lineage_only_hits():
    records = [
        {
            "event": "exit",
            "stage": "survival.final_post_cap",
            "payload": {
                "candidates": [
                    {
                        "sample_idx": 1,
                        "timestamp": 10.0,
                        "merged_timestamps": [20.0],
                        "merged_from_sample_idxs": [1, 2],
                    },
                    {"sample_idx": 3, "timestamp": 30.0, "merged_timestamps": []},
                ]
            },
        }
    ]

    payload = build_debug_qa_trace(
        trace_records=records,
        targets=[
            {"time": 10.0, "label": "direct", "tolerance": 2.25},
            {"time": 20.0, "label": "lineage", "tolerance": 2.25},
            {"time": 25.0, "label": "miss", "tolerance": 2.25},
        ],
        video="video.mp4",
    )

    buckets = {row["label"]: row["bucket"] for row in payload["targets"]}
    assert buckets["direct"] == "hit_direct"
    assert buckets["lineage"] == "hit_lineage_only"
    assert buckets["miss"] == "no_sample_near_target"
    assert payload["fixture_summary"] == {
        "direct_hit_count": 1,
        "lineage_only_hit_count": 1,
        "miss_count": 1,
    }


def test_debug_qa_trace_prefers_direct_hit_over_closer_lineage_hit():
    records = [
        {
            "event": "exit",
            "stage": "survival.final_post_cap",
            "payload": {
                "candidates": [
                    {"sample_idx": 1, "timestamp": 18.5, "merged_timestamps": []},
                    {
                        "sample_idx": 2,
                        "timestamp": 23.5,
                        "merged_timestamps": [20.0],
                        "merged_from_sample_idxs": [2, 3],
                    },
                ]
            },
        }
    ]

    payload = build_debug_qa_trace(
        trace_records=records,
        targets=[{"time": 20.0, "label": "direct", "tolerance": 2.25}],
        video="video.mp4",
    )

    target = payload["targets"][0]
    assert target["bucket"] == "hit_direct"
    assert target["stage_membership"]["survival.final_post_cap"]["matched_via"] == "timestamp"
    assert target["nearest_final_timestamp"] == 18.5


def test_debug_qa_trace_uses_sampling_snapshot_rows():
    records = [
        {
            "event": "exit",
            "stage": "sampling",
            "payload": {
                "samples": [
                    {
                        "sample_idx": 4,
                        "frame_idx": 40,
                        "timestamp": 20.0,
                        "origin": "sampled_frame",
                        "merged_timestamps": [],
                    }
                ]
            },
        }
    ]

    payload = build_debug_qa_trace(
        trace_records=records,
        targets=[{"time": 20.0, "label": "sampled", "tolerance": 2.25}],
        video="video.mp4",
    )

    membership = payload["targets"][0]["stage_membership"]["sampling"]
    assert membership["hit"] is True
    assert membership["sample_idx"] == 4
    assert payload["targets"][0]["nearest_sample_timestamp"] == 20.0


def test_debug_qa_trace_reports_sampled_but_no_proposal_bucket():
    records = [
        {
            "event": "exit",
            "stage": "sampling",
            "payload": {
                "samples": [
                    {
                        "sample_idx": 4,
                        "frame_idx": 40,
                        "timestamp": 20.0,
                        "origin": "sampled_frame",
                        "merged_timestamps": [],
                    }
                ]
            },
        }
    ]

    payload = build_debug_qa_trace(
        trace_records=records,
        targets=[{"time": 20.0, "label": "sampled", "tolerance": 2.25}],
        video="video.mp4",
    )

    assert payload["targets"][0]["bucket"] == "sampled_but_no_proposal_near_target"


def test_debug_qa_trace_distinguishes_rescue_ocr_and_promotion():
    records = [
        {
            "event": "exit",
            "stage": "proposal.rescue_shortlist",
            "payload": {
                "candidates": [{"sample_idx": 7, "timestamp": 42.0, "merged_timestamps": []}]
            },
        },
        {
            "event": "exit",
            "stage": "evidence.rescue_ocr_batch",
            "payload": {
                "candidates": [{"sample_idx": 7, "timestamp": 42.0, "merged_timestamps": []}]
            },
        },
        {
            "event": "exit",
            "stage": "selection.promoted_rescue_only",
            "payload": {"candidates": []},
        },
        {
            "event": "exit",
            "stage": "selection.after_rescue",
            "payload": {"candidates": []},
        },
    ]

    payload = build_debug_qa_trace(
        trace_records=records,
        targets=[{"time": 42.0, "label": "rescue", "tolerance": 2.25}],
        video="video.mp4",
    )

    target = payload["targets"][0]
    assert target["bucket"] == "rescue_ocrd_but_not_promoted"
    assert target["best_family"] == "rescue"
    assert target["best_stage"] == "evidence.rescue_ocr_batch"


def test_debug_qa_trace_attaches_promotion_preflight_row_by_tolerance():
    records = [
        {
            "event": "decision",
            "stage": "selection.rescue_promotion_preflight",
            "payload": {
                "name": "promotion_preflight",
                "payload": {
                    "candidate_rows": [
                        {
                            "sample_idx": 7,
                            "timestamp": 42.2,
                            "outcome": "eligible_above_headroom",
                            "phase_a_rank": 1,
                            "above_additive_headroom_cut": True,
                            "rejection_branch": None,
                            "rejection_reason": None,
                            "nearest_competing_candidate_timestamp": 40.0,
                        }
                    ]
                },
            },
        }
    ]

    payload = build_debug_qa_trace(
        trace_records=records,
        targets=[{"time": 42.0, "label": "rescue", "tolerance": 2.25}],
        video="video.mp4",
    )

    preflight = payload["targets"][0]["promotion_preflight"]
    assert preflight["hit"] is True
    assert preflight["candidate_timestamp"] == 42.2
    assert preflight["sample_idx"] == 7
    assert preflight["outcome"] == "eligible_above_headroom"
    assert preflight["phase_a_rank"] == 1
    assert preflight["above_additive_headroom_cut"] is True
    assert preflight["nearest_competing_candidate_timestamp"] == 40.0


def test_debug_qa_trace_reads_materialized_promotion_preflight_decision():
    sink = SnapshotTraceSink()
    sink.decision(
        "selection.rescue_promotion_preflight",
        "promotion_preflight",
        {
            "rescue_budget": 5,
            "base_candidate_count": 10,
            "current_post_rescue_count": 15,
            "max_post_rescue_count": 15,
            "additive_output_headroom": 0,
            "current_rescue_count": 5,
            "eligible_below_headroom_count": 1,
            "candidate_rows": [
                {
                    "sample_idx": 7,
                    "timestamp": 42.2,
                    "outcome": "predicate_rejected",
                    "phase_a_rank": 3,
                    "above_additive_headroom_cut": False,
                    "rejection_branch": "redundant_with_selected",
                    "rejection_reason": "near_duplicate",
                    "nearest_competing_candidate_timestamp": 40.0,
                }
            ]
        },
    )

    payload = build_debug_qa_trace(
        trace_records=sink.records,
        targets=[{"time": 42.0, "label": "rescue", "tolerance": 2.25}],
        video="video.mp4",
    )

    preflight = payload["targets"][0]["promotion_preflight"]
    assert preflight["hit"] is True
    assert preflight["candidate_timestamp"] == 42.2
    assert preflight["sample_idx"] == 7
    assert preflight["outcome"] == "predicate_rejected"
    assert preflight["phase_a_rank"] == 3
    assert preflight["above_additive_headroom_cut"] is False
    assert preflight["rejection_branch"] == "redundant_with_selected"
    assert preflight["rejection_reason"] == "near_duplicate"
    assert payload["promotion_preflight_summary"] == {
        "rescue_budget": 5,
        "base_candidate_count": 10,
        "current_post_rescue_count": 15,
        "max_post_rescue_count": 15,
        "additive_output_headroom": 0,
        "current_rescue_count": 5,
        "eligible_below_headroom_count": 1,
    }


def test_debug_qa_trace_reports_promotion_preflight_miss_when_no_row_near_target():
    records = [
        {
            "event": "decision",
            "stage": "selection.rescue_promotion_preflight",
            "payload": {
                "name": "promotion_preflight",
                "payload": {"candidate_rows": [{"sample_idx": 7, "timestamp": 100.0}]},
            },
        }
    ]

    payload = build_debug_qa_trace(
        trace_records=records,
        targets=[{"time": 42.0, "label": "rescue", "tolerance": 2.25}],
        video="video.mp4",
    )

    assert payload["targets"][0]["promotion_preflight"] == {"hit": False}


def test_debug_qa_trace_reports_rescue_shortlisted_but_not_ocrd():
    records = [
        {
            "event": "exit",
            "stage": "proposal.rescue_shortlist",
            "payload": {
                "candidates": [{"sample_idx": 8, "timestamp": 55.0, "merged_timestamps": []}]
            },
        },
        {
            "event": "exit",
            "stage": "evidence.rescue_ocr_batch",
            "payload": {"candidates": []},
        },
    ]

    payload = build_debug_qa_trace(
        trace_records=records,
        targets=[{"time": 55.0, "label": "shortlisted", "tolerance": 2.25}],
        video="video.mp4",
    )

    target = payload["targets"][0]
    assert target["bucket"] == "rescue_shortlisted_but_not_ocrd"
    assert target["best_family"] == "rescue"
    assert target["best_stage"] == "proposal.rescue_shortlist"


def test_debug_qa_trace_prefers_deepest_family_path():
    records = [
        {
            "event": "exit",
            "stage": "proposal.pass1_primary",
            "payload": {
                "candidates": [{"sample_idx": 1, "timestamp": 10.0, "merged_timestamps": []}]
            },
        },
        {
            "event": "exit",
            "stage": "proposal.rescue_shortlist",
            "payload": {
                "candidates": [{"sample_idx": 2, "timestamp": 10.1, "merged_timestamps": []}]
            },
        },
        {
            "event": "exit",
            "stage": "evidence.rescue_ocr_batch",
            "payload": {
                "candidates": [{"sample_idx": 2, "timestamp": 10.1, "merged_timestamps": []}]
            },
        },
    ]

    payload = build_debug_qa_trace(
        trace_records=records,
        targets=[{"time": 10.0, "label": "target", "tolerance": 2.25}],
        video="video.mp4",
    )

    target = payload["targets"][0]
    assert target["best_family"] == "rescue"
    assert target["best_stage"] == "evidence.rescue_ocr_batch"


def test_rescue_ocr_batch_without_shortlist_hit_does_not_imply_rescue_family():
    records = [
        {
            "event": "exit",
            "stage": "proposal.pass1_primary",
            "payload": {
                "candidates": [{"sample_idx": 1, "timestamp": 22.0, "merged_timestamps": []}]
            },
        },
        {
            "event": "exit",
            "stage": "evidence.rescue_ocr_batch",
            "payload": {
                "candidates": [{"sample_idx": 1, "timestamp": 22.0, "merged_timestamps": []}]
            },
        },
    ]

    payload = build_debug_qa_trace(
        trace_records=records,
        targets=[{"time": 22.0, "label": "primary", "tolerance": 2.25}],
        video="video.mp4",
    )

    target = payload["targets"][0]
    assert target["bucket"] == "coarse_proposal_exists_but_not_selected"
    assert target["best_family"] == "pass1"
    assert target["best_stage"] == "proposal.pass1_primary"


def test_candidate_snapshot_reports_merged_metadata_integrity():
    sink = SnapshotTraceSink()
    batch = CandidateBatch(
        stage="test",
        candidates=[
            {
                "sample_idx": 3,
                "timestamp": 30.0,
                "merged_from_sample_idxs": [4],
                "merged_timestamps": [40.0],
            }
        ],
    )

    sink.exit("test", batch)
    violations = sink.records[0]["payload"]["integrity_violations"]

    assert {"candidate_id": "test:3", "reason": "sample_idx_missing_from_merged_idxs"} in violations
    assert {"candidate_id": "test:3", "reason": "timestamp_missing_from_merged_timestamps"} in violations
