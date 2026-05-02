import numpy as np
from PIL import Image

from keyframe.frames import _comparison_primary_sample_idxs, ocr_candidates
from keyframe.scoring import (
    assign_dwell_ids,
    build_rescue_shortlist,
    promote_rescue_candidates,
    proxy_content_scores,
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


def test_ocr_candidates_skips_precached_ocr(monkeypatch):
    monkeypatch.setattr("keyframe.frames._ocr_apple_vision", lambda _img: (_ for _ in ()).throw(AssertionError("called")))
    candidates = [{"sample_idx": 0, "timestamp": 1.0, "ocr_text": "cached text"}]

    texts = ocr_candidates(candidates, [Image.new("RGB", (8, 8), "white")])

    assert texts == ["cached text"]
