from PIL import Image

from keyframe.dedupe import (
    _merge_metadata,
    _is_retained_evidence_candidate,
    adjacent_same_screen_dedupe,
    clean_ocr_token_sets,
    canonical_markers,
    filter_low_information_candidates,
    global_candidate_dedupe,
    has_meaningful_evidence_for_retention,
    is_protected_candidate,
    near_time_dedupe,
    retain_cluster_alternates,
)


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


def test_near_time_dedupe_keeps_different_page_markers():
    shared = {"brucepower", "component", "test", "document", "review", "unit", "owner", "date", "form"}
    candidates = [
        {"sample_idx": 1, "timestamp": 10.0, "candidate_score": 1.0},
        {"sample_idx": 2, "timestamp": 10.5, "candidate_score": 2.0},
    ]

    survivors = near_time_dedupe(candidates, [shared | {"page1"}, shared | {"page2"}])

    assert [c["sample_idx"] for c in survivors] == [1, 2]


def test_near_time_dedupe_keeps_different_status_tokens():
    shared = {"approval", "request", "amount", "manager", "review", "owner", "date", "form", "unit"}
    candidates = [
        {"sample_idx": 1, "timestamp": 10.0, "candidate_score": 1.0},
        {"sample_idx": 2, "timestamp": 10.5, "candidate_score": 2.0},
    ]

    survivors = near_time_dedupe(candidates, [shared | {"draft"}, shared | {"approved"}])

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
        {"approved", "page1", "component"},
        {"draft", "page2", "component"},
    ]


def test_clean_ocr_token_sets_keeps_common_content_tokens():
    cleaned = clean_ocr_token_sets(
        [
            {"approved", "page1", "component", "shared"},
            {"draft", "page2", "component", "shared"},
            {"pending", "page3", "component", "shared"},
        ],
        df_cutoff=0.1,
    )

    assert all("component" in tokens for tokens in cleaned)
    assert all("shared" in tokens for tokens in cleaned)


def test_canonical_markers_page_word_and_joined_page_match():
    assert canonical_markers({"page", "2"})["page"] == {"page2"}
    assert canonical_markers({"page2"})["page"] == {"page2"}


def test_canonical_markers_bare_numeric_is_not_page_without_page_word():
    assert canonical_markers({"2", "invoice"})["page"] == set()


def test_meaningful_evidence_retention_rules():
    assert has_meaningful_evidence_for_retention({"draft"}, {"approved"})
    assert has_meaningful_evidence_for_retention(set(), {"page1"})
    assert has_meaningful_evidence_for_retention({"page1", "option1"}, {"page2", "option2"})
    assert not has_meaningful_evidence_for_retention({"page1", "form"}, {"page2", "form"})
    assert has_meaningful_evidence_for_retention({"page1", "form"}, {"page2", "form"}, visual_delta=0.2)


def test_low_information_filter_drops_blank_avatar_only_frame():
    candidates = [
        {
            "sample_idx": 0,
            "timestamp": 35.99,
            "caption": "black background with circular shape/photo",
            "ocr_tokens": [],
        }
    ]
    frames = [Image.new("RGB", (64, 64), "black")]

    survivors = filter_low_information_candidates(candidates, frames)

    assert len(survivors) == 0


def test_low_information_filter_keeps_title_card():
    candidates = [
        {
            "sample_idx": 0,
            "timestamp": 1.0,
            "caption": "ECHO TESTING meeting title",
            "ocr_tokens": ["testing", "36381", "recorded"],
        }
    ]
    frames = [Image.new("RGB", (64, 64), "black")]

    survivors = filter_low_information_candidates(candidates, frames)

    assert [c["sample_idx"] for c in survivors] == [0]


def test_low_information_filter_keeps_document_or_app_frame():
    candidates = [
        {
            "sample_idx": 0,
            "timestamp": 12.0,
            "caption": "document shown in a software interface",
            "ocr_tokens": ["brucepower", "page1", "draft"],
        }
    ]
    frames = [Image.new("RGB", (64, 64), "white")]

    survivors = filter_low_information_candidates(candidates, frames)

    assert [c["sample_idx"] for c in survivors] == [0]


def test_low_information_filter_keeps_protected_retained_evidence_frame():
    candidates = [
        {
            "sample_idx": 0,
            "timestamp": 12.0,
            "caption": "document shown in a software interface",
            "ocr_tokens": ["page1"],
            "retention_reason": "evidence_asymmetry",
        }
    ]
    frames = [Image.new("RGB", (64, 64), "white")]

    survivors = filter_low_information_candidates(candidates, frames, min_clean_tokens=3)

    assert [c["sample_idx"] for c in survivors] == [0]
    assert survivors[0]["low_information_filter_reason"] == "protected_retained_evidence"


def test_low_information_filter_keeps_rescue_origin_as_protected():
    candidates = [
        {
            "sample_idx": 0,
            "timestamp": 12.0,
            "caption": "document shown in a software interface",
            "ocr_tokens": ["page1"],
            "rescue_origin": "additive_rescue",
        }
    ]
    frames = [Image.new("RGB", (64, 64), "white")]

    survivors = filter_low_information_candidates(candidates, frames, min_clean_tokens=3)

    assert [c["sample_idx"] for c in survivors] == [0]
    assert _is_retained_evidence_candidate(survivors[0])


def test_low_information_filter_drops_sparse_generic_screen_transition():
    candidates = [
        {
            "sample_idx": 0,
            "timestamp": 35.99,
            "caption": "screenshot of a computer screen with a white background",
            "ocr_tokens": ["dll", "iit", "cechoe", "lall", "bpvpn", "townhall"],
        }
    ]
    image = Image.new("RGB", (200, 100), "white")
    for x in range(40):
        for y in range(100):
            image.putpixel((x, y), (0, 0, 0))

    survivors = filter_low_information_candidates(candidates, [image])

    assert len(survivors) == 0


def test_low_information_filter_drops_dark_viewer_transition_with_chrome_text():
    candidates = [
        {
            "sample_idx": 0,
            "timestamp": 63.0,
            "caption": "screenshot of a computer screen with a black background and browser chrome",
            "ocr_tokens": ["townhall", "gather", "component", "counts", "file"],
        }
    ]
    image = Image.new("RGB", (200, 100), "black")
    for x in range(0, 200, 20):
        for y in range(2):
            image.putpixel((x, y), (255, 255, 255))

    survivors = filter_low_information_candidates(candidates, [image])

    assert len(survivors) == 0


def test_adjacent_same_screen_dedupe_collapses_neighboring_echo_list_states():
    shared = {"echo", "request", "case", "unit", "owner", "date", "list", "dashboard", "row", "team"}
    candidates = [
        {
            "sample_idx": 1,
            "timestamp": 65.5,
            "candidate_score": 1.0,
            "ocr_tokens": sorted(shared | {"cursor"}),
        },
        {
            "sample_idx": 2,
            "timestamp": 141.5,
            "candidate_score": 2.0,
            "ocr_tokens": sorted(shared | {"scroll"}),
        },
    ]

    survivors = adjacent_same_screen_dedupe(candidates)

    assert [c["sample_idx"] for c in survivors] == [2]
    assert survivors[0]["merged_from_sample_idxs"] == [1, 2]
    assert survivors[0]["merged_timestamps"] == [65.5, 141.5]


def test_adjacent_same_screen_dedupe_keeps_different_page_markers():
    shared = {"brucepower", "component", "test", "document", "review", "section", "unit", "owner", "date", "form"}
    candidates = [
        {"sample_idx": 1, "timestamp": 10.0, "ocr_tokens": sorted(shared | {"page1"})},
        {"sample_idx": 2, "timestamp": 20.0, "ocr_tokens": sorted(shared | {"page2"})},
    ]

    survivors = adjacent_same_screen_dedupe(candidates)

    assert [c["sample_idx"] for c in survivors] == [1, 2]


def test_adjacent_same_screen_dedupe_keeps_different_status_tokens():
    shared = {"approval", "request", "amount", "manager", "review", "owner", "date", "form", "unit", "case"}
    candidates = [
        {"sample_idx": 1, "timestamp": 10.0, "ocr_tokens": sorted(shared | {"draft"})},
        {"sample_idx": 2, "timestamp": 20.0, "ocr_tokens": sorted(shared | {"approved"})},
    ]

    survivors = adjacent_same_screen_dedupe(candidates)

    assert [c["sample_idx"] for c in survivors] == [1, 2]


def test_adjacent_same_screen_dedupe_respects_time_window():
    tokens = {"echo", "request", "case", "unit", "owner", "date", "list", "dashboard"}
    candidates = [
        {"sample_idx": 1, "timestamp": 10.0, "ocr_tokens": sorted(tokens)},
        {"sample_idx": 2, "timestamp": 101.1, "ocr_tokens": sorted(tokens)},
    ]

    survivors = adjacent_same_screen_dedupe(candidates)

    assert [c["sample_idx"] for c in survivors] == [1, 2]


def test_retain_cluster_alternates_keeps_differing_evidence():
    candidates = [
        {"sample_idx": 1, "timestamp": 10.0, "clip_cluster": 7, "cluster_role": "primary", "ocr_tokens": ["page1"]},
        {"sample_idx": 2, "timestamp": 11.0, "clip_cluster": 7, "cluster_role": "alt", "ocr_tokens": ["page2"]},
    ]

    retained = retain_cluster_alternates(candidates)

    assert [c["sample_idx"] for c in retained] == [1, 2]
    assert retained[1]["retention_reason"] == "differing_evidence"


def test_retain_cluster_alternates_keeps_evidence_asymmetry():
    candidates = [
        {"sample_idx": 1, "timestamp": 10.0, "clip_cluster": 7, "cluster_role": "primary", "ocr_tokens": ["form"]},
        {"sample_idx": 2, "timestamp": 11.0, "clip_cluster": 7, "cluster_role": "alt", "ocr_tokens": ["form", "page1"]},
    ]

    retained = retain_cluster_alternates(candidates)

    assert [c["sample_idx"] for c in retained] == [1, 2]
    assert retained[1]["retention_reason"] == "evidence_asymmetry"
    assert retained[1]["retention_candidate_reason"] in {"meaningful_evidence_delta", "raw_evidence_asymmetry"}


def test_retain_cluster_alternates_keeps_protective_caption_asymmetry():
    candidates = [
        {"sample_idx": 1, "timestamp": 10.0, "clip_cluster": 7, "cluster_role": "primary", "ocr_tokens": []},
        {
            "sample_idx": 2,
            "timestamp": 11.0,
            "clip_cluster": 7,
            "cluster_role": "alt",
            "ocr_tokens": [],
            "caption": "PDF document page in a viewer",
        },
    ]

    retained = retain_cluster_alternates(candidates)

    assert [c["sample_idx"] for c in retained] == [1, 2]
    assert retained[1]["retention_reason"] == "protective_caption_asymmetry"


def test_retain_cluster_alternates_drops_alt_without_asymmetry():
    candidates = [
        {"sample_idx": 1, "timestamp": 10.0, "clip_cluster": 7, "cluster_role": "primary", "ocr_tokens": ["form"]},
        {"sample_idx": 2, "timestamp": 11.0, "clip_cluster": 7, "cluster_role": "alt", "ocr_tokens": ["form"]},
    ]

    retained = retain_cluster_alternates(candidates)

    assert [c["sample_idx"] for c in retained] == [1]


def test_retain_cluster_alternates_records_rejection_reason():
    candidates = [
        {"sample_idx": 1, "timestamp": 10.0, "clip_cluster": 7, "cluster_role": "primary", "ocr_tokens": ["form"]},
        {"sample_idx": 2, "timestamp": 11.0, "clip_cluster": 7, "cluster_role": "alt", "ocr_tokens": ["form"]},
    ]

    retained = retain_cluster_alternates(candidates)

    assert len(retained) == 1
    assert retained[0]["sample_idx"] == 1
    assert retained[0]["timestamp"] == 10.0
    assert retained[0]["clip_cluster"] == 7
    assert retained[0]["cluster_role"] == "primary"
    assert retained[0]["ocr_tokens"] == ["form"]
    assert retained[0]["retention_reason"] == "none"
    assert retained[0]["retention_candidate_reason"] == "primary"
    assert retained[0]["retention_reasons_seen"] == ["none"]
    assert retained[0]["lineage_roles"] == ["primary"]


def test_merge_metadata_preserves_policy_and_lineage_fields():
    winner = {
        "sample_idx": 1,
        "timestamp": 10.0,
        "retention_reason": "protective_caption_asymmetry",
        "cluster_role": "primary",
    }
    loser = {
        "sample_idx": 2,
        "timestamp": 11.0,
        "retention_reason": "evidence_asymmetry",
        "cluster_role": "alt",
        "retention_reasons_seen": ["evidence_asymmetry"],
        "lineage_roles": ["alt"],
    }

    _merge_metadata(winner, loser)

    assert winner["retention_reason"] == "evidence_asymmetry"
    assert winner["retention_reasons_seen"] == ["evidence_asymmetry", "protective_caption_asymmetry"]
    assert winner["lineage_roles"] == ["alt", "primary"]


def test_merge_metadata_preserves_rescue_policy_fields():
    winner = {"sample_idx": 1, "timestamp": 10.0, "cluster_role": "primary"}
    loser = {
        "sample_idx": 2,
        "timestamp": 11.0,
        "cluster_role": "rescue",
        "rescue_origin": "additive_rescue",
        "rescue_reason": "evidence_marker",
        "rescue_priority": 2,
        "lineage_roles": ["rescue"],
    }

    _merge_metadata(winner, loser)

    assert is_protected_candidate(winner)
    assert winner["rescue_origin"] == "additive_rescue"
    assert winner["rescue_reason"] == "evidence_marker"
    assert winner["rescue_origins_seen"] == ["additive_rescue"]
    assert winner["rescue_priorities_seen"] == ["2"]
    assert winner["lineage_roles"] == ["primary", "rescue"]
