from tests.postmortem.checker import Status, assert_no_failures, check_fixture


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
