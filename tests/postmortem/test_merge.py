from PIL import Image

from keyframe.merge import union_find_merge


def _candidate(sample_idx, timestamp, score=1.0):
    return {
        "sample_idx": sample_idx,
        "frame_idx": sample_idx,
        "timestamp": timestamp,
        "caption": f"frame {sample_idx}",
        "candidate_score": score,
        "clip_cluster": sample_idx,
        "clip_cluster_size": 1,
    }


def _frames(n=10):
    return [Image.new("RGB", (8, 8), "white") for _ in range(n)]


def test_no_average_link_chain_through_intermediate_near_matches():
    candidates = [_candidate(0, 0), _candidate(1, 10), _candidate(2, 20)]
    tokens = [
        {"alpha", "beta", "one"},
        {"alpha", "beta", "two"},
        {"alpha", "three", "four"},
    ]

    final = union_find_merge(candidates, tokens, [True, True, True], _frames(3))

    groups = [set(c["merged_from_sample_idxs"]) for c in final]
    assert {0, 1} in groups
    assert {2} in groups


def test_populated_approval_block_does_not_merge_into_empty_template_neighbor():
    candidates = [_candidate(0, 0), _candidate(1, 2)]
    tokens = [
        {"approval", "amount", "manager", "confirmed"},
        set(),
    ]

    final = union_find_merge(candidates, tokens, [True, False], _frames(2))

    assert len(final) == 2


def test_ocr_density_asymmetry_veto_requires_two_x_and_novel_tokens():
    candidates = [_candidate(0, 0), _candidate(1, 2), _candidate(2, 4)]
    frames = _frames(3)

    blocked = union_find_merge(
        candidates[:2],
        [{"a", "b", "c", "d"}, {"a", "b"}],
        [True, True],
        frames,
    )
    allowed = union_find_merge(
        [candidates[0], candidates[2]],
        [{"a", "b", "c"}, {"a", "b", "c", "d"}],
        [True, True],
        frames,
    )

    assert len(blocked) == 2
    assert len(allowed) == 1
