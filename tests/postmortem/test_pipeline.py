from tests.postmortem.checker import Status, assert_no_failures, check_fixture
from PIL import Image

from keyframe.frames import save_results
from keyframe.manifest import write_manifest


def _versioned(payload):
    return {
        "schema_version": 1,
        "token_normalization_version": 1,
        "transcript_normalization_version": 1,
        **payload,
    }


def test_fixture_checker_passes_minimal_expected_assertions():
    pre = _versioned({"samples": []})
    post = _versioned({
        "survivor_sample_idxs": [1, 3],
        "merged_from_sample_idxs": [[1, 2], [3]],
    })
    expected = _versioned({
        "must_appear_sample_idxs": [
            {"sample_idx": 1, "stage": "p1", "rationale": "decisive frame"},
        ],
        "must_collapse_pairs": [
            {"sample_idxs": [1, 2], "stage": "p1", "rationale": "duplicate neighbor"},
        ],
        "must_not_merge_pairs": [
            {"sample_idxs": [1, 3], "stage": "p2", "rationale": "different content"},
        ],
        "must_select_representative_from": [
            {"sample_idxs": [1, 2], "stage": "p1", "rationale": "either neighbor is acceptable"},
        ],
        "min_total_frames": 2,
        "max_total_frames": 3,
    })

    results = check_fixture(pre, post, expected)

    assert_no_failures(results)
    assert all(result.stage and result.rationale for result in results if result.stage != "schema")


def test_fixture_checker_hard_fails_version_mismatch():
    pre = _versioned({})
    pre["schema_version"] = 99
    post = _versioned({"survivor_sample_idxs": [], "merged_from_sample_idxs": []})
    expected = _versioned({})

    results = check_fixture(pre, post, expected)

    assert any(result.status == Status.FAIL and result.assertion == "pre_p1.schema_version" for result in results)


def test_outputs_include_additive_attribution_fields(tmp_path):
    selected = [{
        "sample_idx": 0,
        "frame_idx": 10,
        "timestamp": 1.25,
        "caption": "PDF document page",
        "caption_cluster": 0,
        "merged_from": 1,
        "merged_captions": ["PDF document page"],
        "merged_timestamps": [1.25],
        "merged_from_sample_idxs": [0],
        "clip_cluster": 4,
        "clip_cluster_size": 3,
        "cluster_role": "alt",
        "retention_reason": "evidence_asymmetry",
        "retention_reasons_seen": ["evidence_asymmetry"],
        "retention_candidate_reason": "meaningful_evidence_delta",
        "retention_rejected_reason": None,
        "rescue_origin": "additive_rescue",
        "proxy_content_score": 0.72,
        "rescue_priority": 1,
        "lineage_roles": ["primary", "alt"],
        "ocr_text": "Page 1 Approved",
        "ocr_tokens": ["page1", "approved"],
        "raw_token_count": 3,
        "filtered_token_count": 3,
        "cleaned_token_count": 2,
        "cleaning_attrition_ratio": 0.3333,
        "low_information_filter_reason": "protected_retained_evidence",
        "dedupe_stage": "union_find_merge",
        "merge_reason": "component",
    }]
    save_results(selected, [Image.new("RGB", (8, 8), "white")], tmp_path)
    write_manifest(
        selected,
        tmp_path,
        metadata={"scene_coalescence": {"original_scene_count": 2, "coalesced_scene_count": 1}},
    )

    import json

    captions = json.loads((tmp_path / "captions.json").read_text())
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    for payload in (captions[0], manifest["frames"][0]):
        assert payload["cluster_role"] == "alt"
        assert payload["retention_reason"] == "evidence_asymmetry"
        assert payload["retention_reasons_seen"] == ["evidence_asymmetry"]
        assert payload["retention_candidate_reason"] == "meaningful_evidence_delta"
        assert payload["retention_rejected_reason"] is None
        assert payload["rescue_origin"] == "additive_rescue"
        assert payload["proxy_content_score"] == 0.72
        assert payload["rescue_priority"] == 1
        assert payload["lineage_roles"] == ["primary", "alt"]
        assert payload["raw_token_count"] == 3
        assert payload["filtered_token_count"] == 3
        assert payload["cleaned_token_count"] == 2
        assert payload["low_information_filter_reason"] == "protected_retained_evidence"
        assert payload["dedupe_stage"] == "union_find_merge"
        assert payload["merge_reason"] == "component"
    assert manifest["metadata"]["scene_coalescence"]["original_scene_count"] == 2
