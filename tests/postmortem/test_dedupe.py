from keyframe.dedupe import clean_ocr_token_sets, global_candidate_dedupe, near_time_dedupe


def test_half_second_ocr_neighbors_collapse():
    candidates = [
        {"sample_idx": 10, "timestamp": 95.49, "candidate_score": 1.0},
        {"sample_idx": 11, "timestamp": 95.99, "candidate_score": 2.0},
    ]
    tokens = [{"approve", "request", "total"}, {"approve", "request", "total"}]

    survivors = near_time_dedupe(candidates, tokens)

    assert [c["sample_idx"] for c in survivors] == [11]
    assert survivors[0]["merged_from_sample_idxs"] == [10, 11]


def test_distinct_near_time_ocr_survives():
    candidates = [
        {"sample_idx": 1, "timestamp": 1.0},
        {"sample_idx": 2, "timestamp": 1.5},
    ]
    tokens = [{"approval", "page"}, {"shipping", "address"}]

    survivors = near_time_dedupe(candidates, tokens)

    assert [c["sample_idx"] for c in survivors] == [1, 2]


def test_weak_ocr_dhash_path_collapses():
    candidates = [
        {"sample_idx": 1, "timestamp": 10.0, "candidate_score": 2.0},
        {"sample_idx": 2, "timestamp": 11.0, "candidate_score": 1.0},
    ]

    survivors = near_time_dedupe(candidates, [set(), set()], {1: 0b101010, 2: 0b101011})

    assert [c["sample_idx"] for c in survivors] == [1]
    assert survivors[0]["merged_timestamps"] == [10.0, 11.0]


def test_global_dedupe_merges_high_ocr_jaccard_far_apart():
    candidates = [
        {"sample_idx": 1, "timestamp": 10.0, "candidate_score": 1.0},
        {"sample_idx": 2, "timestamp": 200.0, "candidate_score": 2.0},
    ]
    tokens = [
        {"approval", "request", "amount", "manager", "confirmed"},
        {"approval", "request", "amount", "manager", "confirmed"},
    ]

    survivors = global_candidate_dedupe(candidates, tokens)

    assert [c["sample_idx"] for c in survivors] == [2]
    assert survivors[0]["merged_from_sample_idxs"] == [1, 2]


def test_global_dedupe_blocks_different_page_markers():
    candidates = [
        {"sample_idx": 1, "timestamp": 10.0},
        {"sample_idx": 2, "timestamp": 200.0},
    ]
    tokens = [
        {"brucepower", "page1", "component", "test"},
        {"brucepower", "page2", "component", "test"},
    ]

    survivors = global_candidate_dedupe(candidates, tokens)

    assert [c["sample_idx"] for c in survivors] == [1, 2]


def test_global_dedupe_blocks_different_status_tokens():
    candidates = [
        {"sample_idx": 1, "timestamp": 10.0},
        {"sample_idx": 2, "timestamp": 200.0},
    ]
    tokens = [
        {"approval", "request", "draft"},
        {"approval", "request", "approved"},
    ]

    survivors = global_candidate_dedupe(candidates, tokens)

    assert [c["sample_idx"] for c in survivors] == [1, 2]


def test_global_dhash_only_requires_strict_match():
    candidates = [
        {"sample_idx": 1, "timestamp": 10.0},
        {"sample_idx": 2, "timestamp": 200.0},
        {"sample_idx": 3, "timestamp": 300.0},
    ]

    survivors = global_candidate_dedupe(
        candidates,
        [set(), set(), set()],
        {1: 0b100000, 2: 0b100111, 3: 0b100001},
    )

    assert [c["sample_idx"] for c in survivors] == [1, 2]
    assert survivors[0]["merged_from_sample_idxs"] == [1, 3]


def test_global_dedupe_blocks_density_asymmetry():
    candidates = [
        {"sample_idx": 1, "timestamp": 10.0},
        {"sample_idx": 2, "timestamp": 200.0},
    ]
    tokens = [
        {"approval", "request"},
        {"approval", "request", "amount", "manager"},
    ]

    survivors = global_candidate_dedupe(candidates, tokens)

    assert [c["sample_idx"] for c in survivors] == [1, 2]


def test_clean_ocr_token_sets_drops_chrome_and_keeps_status():
    cleaned = clean_ocr_token_sets([
        {"echo", "figma", "approved", "page1", "component"},
        {"echo", "figma", "draft", "page2", "component"},
    ])

    assert cleaned == [
        {"approved", "page1"},
        {"draft", "page2"},
    ]
