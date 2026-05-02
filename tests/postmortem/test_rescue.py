import numpy as np
from PIL import Image
import json

from keyframe.frames import _comparison_primary_sample_idxs, ocr_candidates as _ocr_candidates
from keyframe.pipeline.contracts import candidate_records
from keyframe.scoring import (
    assign_dwell_ids,
    build_rescue_shortlist as _build_rescue_shortlist,
    promote_rescue_candidates as _promote_rescue_candidates,
    proxy_content_scores,
    rescue_promotion_preflight_report,
)


def _project(records):
    return [record.to_dict() for record in records]


def promote_rescue_candidates(*args, **kwargs):
    return _project(_promote_rescue_candidates(*args, **kwargs))


def build_rescue_shortlist(*args, **kwargs):
    shortlist, proxy_rows, budget = _build_rescue_shortlist(*args, **kwargs)
    return _project(shortlist), proxy_rows, budget


def ocr_candidates(*args, **kwargs):
    texts, records = _ocr_candidates(*args, **kwargs)
    return texts, _project(records)


def _cand(sample_idx, timestamp, *, scene=0, cluster=1, tokens=(), proxy=0.1, window=0):
    return {
        "sample_idx": sample_idx,
        "frame_idx": sample_idx,
        "timestamp": timestamp,
        "clip_cluster": cluster,
        "scene_id": scene,
        "temporal_window_id": window,
        "proxy_content_score": proxy,
        "ocr_tokens": list(tokens),
        "rescue_tokens": list(tokens),
    }


def _preflight(base, shortlist, current, dwell_ids, budget, embeddings=None):
    return rescue_promotion_preflight_report(
        candidate_records(base),
        candidate_records(shortlist),
        candidate_records(current),
        dwell_ids,
        budget,
        embeddings,
    )


def test_proxy_content_scores_clamp_and_no_variance_normalizes_to_zero():
    frames = [Image.new("RGB", (32, 32), "white"), Image.new("RGB", (32, 32), "white")]

    scores = proxy_content_scores(frames)

    assert [row["normalized_textline_score"] for row in scores] == [0.0, 0.0]
    assert all(0.0 <= row["proxy_content_score"] <= 1.0 for row in scores)


def test_assign_dwell_ids_groups_adjacent_similar_hashes():
    assert assign_dwell_ids([0b0000, 0b0001, 0b1111], hamming_threshold=1) == [0, 0, 1]


def test_rescue_shortlist_backfills_when_proxy_scores_are_flat(monkeypatch):
    monkeypatch.setattr(
        "keyframe.scoring.proxy_content_scores",
        lambda frames: [
            {
                "proxy_content_score": 0.0,
                "textline_score": 0.0,
                "edge_score": 0.0,
                "entropy": 0.0,
                "dark_ratio": 0.0,
                "bright_ratio": 1.0,
            }
            for _ in frames
        ],
    )
    frames = [Image.new("RGB", (8, 8), "white") for _ in range(8)]
    candidates = [{"sample_idx": 0, "timestamp": 0.0, "scene_id": 0}]

    shortlist, _proxy_rows, budget = build_rescue_shortlist(
        frames,
        [float(i) for i in range(8)],
        list(range(8)),
        candidates,
        pass1_clusters=3,
        sample_scenes={i: 0 for i in range(8)},
    )

    assert budget == 3
    assert len(shortlist) >= budget
    assert 0 not in {row["sample_idx"] for row in shortlist}


def test_rescue_shortlist_includes_per_scene_coverage(monkeypatch):
    scores = [0.99, 0.98, 0.97, 0.96, 0.10, 0.09]
    monkeypatch.setattr(
        "keyframe.scoring.proxy_content_scores",
        lambda frames: [
            {
                "proxy_content_score": score,
                "textline_score": score,
                "edge_score": score,
                "entropy": score,
                "dark_ratio": 0.0,
                "bright_ratio": 0.0,
            }
            for score in scores
        ],
    )
    frames = [Image.new("RGB", (8, 8), "white") for _ in scores]
    candidates = [
        {"sample_idx": 0, "timestamp": 0.0, "scene_id": 0},
        {"sample_idx": 5, "timestamp": 5.0, "scene_id": 1},
    ]

    shortlist, _proxy_rows, _budget = build_rescue_shortlist(
        frames,
        [float(i) for i in range(len(scores))],
        list(range(len(scores))),
        candidates,
        pass1_clusters=3,
        sample_scenes={0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1},
    )

    assert 4 in {row["sample_idx"] for row in shortlist}


def test_comparison_primary_sample_idxs_include_cluster_and_scene_primaries():
    candidates = [
        {"sample_idx": 0, "timestamp": 0.0, "clip_cluster": 1, "scene_id": 0},
        {"sample_idx": 5, "timestamp": 5.0, "clip_cluster": 2, "scene_id": 0},
        {"sample_idx": 20, "timestamp": 20.0, "clip_cluster": 3, "scene_id": 1},
    ]
    shortlist = [
        {"sample_idx": 6, "timestamp": 6.0, "clip_cluster": 2, "scene_id": 0},
        {"sample_idx": 22, "timestamp": 22.0, "clip_cluster": 4, "scene_id": 1},
    ]

    assert _comparison_primary_sample_idxs(candidates, shortlist) == {5, 20}


def test_rescue_same_cluster_swap_precedes_additive_and_respects_budget():
    candidates = [
        {
            "sample_idx": 0,
            "timestamp": 0.0,
            "clip_cluster": 3,
            "scene_id": 0,
            "cluster_role": "primary",
            "proxy_content_score": 0.1,
            "ocr_tokens": [],
        },
        {
            "sample_idx": 2,
            "timestamp": 2.0,
            "clip_cluster": 4,
            "scene_id": 0,
            "cluster_role": "single",
            "proxy_content_score": 0.1,
            "ocr_tokens": [],
        },
    ]
    shortlist = [
        {
            "sample_idx": 1,
            "frame_idx": 1,
            "timestamp": 1.0,
            "clip_cluster": 3,
            "scene_id": 0,
            "proxy_content_score": 0.8,
            "ocr_tokens": ["page1"],
            "rescue_tokens": ["page1"],
        },
        {
            "sample_idx": 3,
            "frame_idx": 3,
            "timestamp": 3.0,
            "clip_cluster": 5,
            "scene_id": 1,
            "proxy_content_score": 0.9,
            "ocr_tokens": ["page2"],
            "rescue_tokens": ["page2"],
        },
    ]

    promoted = promote_rescue_candidates(candidates, shortlist, [0, 1, 2, 3], rescue_budget=1)

    assert [row["sample_idx"] for row in promoted] == [1, 2]
    assert promoted[0]["rescue_origin"] == "same_cluster_swap"
    assert promoted[0]["rescue_priority"] == 1


def test_additive_evidence_rescue_skips_equivalent_scene_marker_coverage():
    candidates = [
        {
            "sample_idx": 0,
            "timestamp": 0.0,
            "clip_cluster": 1,
            "scene_id": 0,
            "proxy_content_score": 0.1,
            "ocr_tokens": ["page1"],
            "rescue_tokens": ["page1"],
            "temporal_window_id": 0,
        }
    ]
    shortlist = [
        {
            "sample_idx": 1,
            "frame_idx": 1,
            "timestamp": 10.0,
            "clip_cluster": 2,
            "scene_id": 0,
            "proxy_content_score": 0.8,
            "ocr_tokens": ["page1"],
            "rescue_tokens": ["page1"],
            "temporal_window_id": 0,
        }
    ]

    promoted = promote_rescue_candidates(candidates, shortlist, [0, 1], rescue_budget=3)

    assert [row["sample_idx"] for row in promoted] == [0]


def test_additive_evidence_rescue_keeps_equivalent_marker_in_different_window():
    candidates = [
        {
            "sample_idx": 0,
            "timestamp": 0.0,
            "clip_cluster": 1,
            "scene_id": 0,
            "proxy_content_score": 0.1,
            "ocr_tokens": ["page1"],
            "rescue_tokens": ["page1"],
            "temporal_window_id": 0,
        }
    ]
    shortlist = [
        {
            "sample_idx": 1,
            "frame_idx": 1,
            "timestamp": 30.0,
            "clip_cluster": 2,
            "scene_id": 0,
            "proxy_content_score": 0.8,
            "ocr_tokens": ["page1"],
            "rescue_tokens": ["page1"],
            "temporal_window_id": 1,
        }
    ]

    promoted = promote_rescue_candidates(candidates, shortlist, [0, 1], rescue_budget=3)

    assert [row["sample_idx"] for row in promoted] == [0, 1]
    assert promoted[1]["rescue_reason"] == "temporal_coverage"


def test_additive_content_reference_rescue_skips_clip_and_token_redundancy():
    candidates = [
        {
            "sample_idx": 0,
            "timestamp": 0.0,
            "clip_cluster": 1,
            "scene_id": 0,
            "proxy_content_score": 0.1,
            "ocr_tokens": ["figma", "mockup", "source"],
            "rescue_tokens": ["figma", "mockup", "source"],
            "temporal_window_id": 0,
        }
    ]
    shortlist = [
        {
            "sample_idx": 1,
            "frame_idx": 1,
            "timestamp": 10.0,
            "clip_cluster": 2,
            "scene_id": 0,
            "proxy_content_score": 0.8,
            "ocr_tokens": ["figma", "mockup", "source"],
            "rescue_tokens": ["figma", "mockup", "source"],
            "temporal_window_id": 0,
        }
    ]
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    promoted = promote_rescue_candidates(
        candidates,
        shortlist,
        [0, 1],
        rescue_budget=3,
        clip_embeddings=embeddings,
    )

    assert [row["sample_idx"] for row in promoted] == [0]


def test_temporal_coverage_rescue_adds_distinct_dense_evidence_without_swapping():
    candidates = [
        {
            "sample_idx": 0,
            "timestamp": 0.0,
            "clip_cluster": 1,
            "scene_id": 0,
            "proxy_content_score": 0.8,
            "ocr_tokens": ["page1"],
            "rescue_tokens": ["page1"],
        }
    ]
    shortlist = [
        {
            "sample_idx": 10,
            "frame_idx": 10,
            "timestamp": 10.0,
            "clip_cluster": 1,
            "scene_id": 0,
            "proxy_content_score": 0.7,
            "ocr_tokens": ["page1"],
            "rescue_tokens": [f"token{i}" for i in range(20)],
        }
    ]

    promoted = promote_rescue_candidates(candidates, shortlist, [0] * 11, rescue_budget=3)

    assert [row["sample_idx"] for row in promoted] == [0, 10]
    assert promoted[1]["rescue_origin"] == "additive_rescue"
    assert promoted[1]["rescue_reason"] == "temporal_coverage"


def test_promotion_preflight_classifies_eligible_candidate_above_headroom():
    base = [_cand(0, 0.0, tokens=["intro"])]
    shortlist = [_cand(10, 10.0, cluster=2, tokens=["page2", "alpha", "beta"], proxy=0.9, window=1)]

    report = _preflight(base, shortlist, base, [0] * 11, 1)

    row = report["candidate_rows"][0]
    assert report["additive_output_headroom"] == 1
    assert row["outcome"] == "eligible_above_headroom"
    assert row["phase_a_eligible"] is True
    assert row["phase_a_rank"] == 1
    assert row["above_additive_headroom_cut"] is True


def test_promotion_preflight_classifies_eligible_candidate_below_headroom():
    base = [_cand(0, 0.0, tokens=["intro"])]
    current = base + [
        {
            **_cand(5, 5.0, scene=1, cluster=2, tokens=["page1"], proxy=0.8),
            "rescue_origin": "additive_rescue",
            "rescue_reason": "evidence_marker",
        }
    ]
    shortlist = [_cand(10, 10.0, scene=2, cluster=3, tokens=["page2", "alpha", "beta"], proxy=0.9)]

    report = _preflight(base, shortlist, current, [0] * 11, 1)

    row = report["candidate_rows"][0]
    assert report["additive_output_headroom"] == 0
    assert row["outcome"] == "eligible_below_headroom"
    assert row["phase_a_rank"] == 1
    assert row["above_additive_headroom_cut"] is False
    assert row["binding_budget"] == "additive_output_headroom"


def test_promotion_preflight_classifies_predicate_rejected_candidate_with_branch_details():
    base = [_cand(0, 0.0, tokens=["page1"])]
    shortlist = [_cand(1, 1.0, tokens=["page1"], proxy=0.9)]

    report = _preflight(base, shortlist, base, [0, 0], 1)

    row = report["candidate_rows"][0]
    assert row["outcome"] == "predicate_rejected"
    assert row["phase_a_eligible"] is False
    assert row["phase_a_rank"] is None
    assert row["rejection_branch"] == "redundancy"
    assert row["rejection_reason"] == "temporally_local_marker_equivalent"
    assert row["nearest_competing_candidate_sample_idx"] == 0
    assert row["marker_equivalent"] is True


def test_promotion_preflight_does_not_mutate_candidate_records_and_is_json_safe():
    base_records = candidate_records([_cand(0, 0.0, tokens=["intro"])])
    shortlist_records = candidate_records([_cand(10, 10.0, cluster=2, tokens=["page2"], proxy=0.9)])
    before = [record.to_dict() for record in base_records + shortlist_records]

    report = rescue_promotion_preflight_report(
        base_records,
        shortlist_records,
        base_records,
        [0] * 11,
        1,
        None,
    )

    assert [record.to_dict() for record in base_records + shortlist_records] == before
    json.dumps(report)


def test_promotion_preflight_predicate_rejected_and_below_headroom_are_mutually_exclusive():
    base = [_cand(0, 0.0, tokens=["page1"])]
    shortlist = [_cand(1, 1.0, tokens=["page1"], proxy=0.9)]

    report = _preflight(base, shortlist, base, [0, 0], 0)

    row = report["candidate_rows"][0]
    assert row["outcome"] == "predicate_rejected"
    assert row["outcome"] != "eligible_below_headroom"
    assert row["binding_budget"] == "none"


def test_promotion_preflight_does_not_change_promotion_output():
    candidates = [_cand(0, 0.0, tokens=["intro"])]
    shortlist = [_cand(10, 10.0, cluster=2, tokens=["page2", "alpha", "beta"], proxy=0.9)]
    dwell_ids = [0] * 11

    before = _promote_rescue_candidates(candidates, shortlist, dwell_ids, rescue_budget=1)
    rescue_promotion_preflight_report(
        candidate_records(candidates),
        candidate_records(shortlist),
        before,
        dwell_ids,
        1,
        None,
    )
    after = _promote_rescue_candidates(candidates, shortlist, dwell_ids, rescue_budget=1)

    assert [record.to_dict() for record in before] == [record.to_dict() for record in after]


def test_ocr_candidates_skips_precached_ocr(monkeypatch):
    monkeypatch.setattr("keyframe.frames._ocr_apple_vision", lambda _img: (_ for _ in ()).throw(AssertionError("called")))
    candidates = [{"sample_idx": 0, "timestamp": 1.0, "ocr_text": "cached text"}]

    texts, updated = ocr_candidates(candidates, [Image.new("RGB", (8, 8), "white")])

    assert texts == ["cached text"]
    assert updated[0]["ocr_text"] == "cached text"
