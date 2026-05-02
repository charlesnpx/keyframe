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
    assert buckets["miss"] == "no_coarse_proposal_near_target"
    assert payload["fixture_summary"] == {
        "direct_hit_count": 1,
        "lineage_only_hit_count": 1,
        "miss_count": 1,
    }


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
