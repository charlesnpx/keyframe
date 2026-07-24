from pathlib import Path

from PIL import Image

from keyframe.dedupe import (
    _is_retained_evidence_candidate,
    adjacent_same_screen_dedupe as _adjacent_same_screen_dedupe,
    clean_ocr_token_sets,
    canonical_markers,
    content_area_duplicate_veto as _content_area_duplicate_veto,
    filter_low_information_candidates as _filter_low_information_candidates,
    global_candidate_dedupe as _global_candidate_dedupe,
    has_meaningful_evidence_for_retention,
    is_protected_candidate,
    merge_candidate_lineage,
    near_time_dedupe as _near_time_dedupe,
    retain_cluster_alternates as _retain_cluster_alternates,
)
from keyframe.evidence import (
    field_section_signatures,
    normalized_ocr_line_signatures,
    structured_delta_categories,
    structured_signature_change_count,
)
from keyframe.pipeline.contracts import as_candidate_record
from keyframe.visual import build_frame_metric_table


def _project(records):
    return [record.to_dict() for record in records]


def near_time_dedupe(*args, **kwargs):
    return _project(_near_time_dedupe(*args, **kwargs))


def global_candidate_dedupe(*args, **kwargs):
    return _project(_global_candidate_dedupe(*args, **kwargs))


def filter_low_information_candidates(*args, **kwargs):
    return _project(_filter_low_information_candidates(*args, **kwargs))


def adjacent_same_screen_dedupe(*args, **kwargs):
    return _project(_adjacent_same_screen_dedupe(*args, **kwargs))


def retain_cluster_alternates(*args, **kwargs):
    return _project(_retain_cluster_alternates(*args, **kwargs))


def content_area_duplicate_veto(*args, **kwargs):
    survivors, dropped = _content_area_duplicate_veto(*args, **kwargs)
    return _project(survivors), dropped


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
        {"browser", "figma", "approved", "page1", "component"},
        {"chrome", "figma", "draft", "page2", "component"},
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


def test_low_information_filter_uses_cached_frame_metrics_without_frame_lookup():
    candidates = [
        {
            "sample_idx": 0,
            "timestamp": 35.99,
            "caption": "black background with circular shape/photo",
            "ocr_tokens": [],
        }
    ]
    table = build_frame_metric_table([Image.new("RGB", (64, 64), "black")], [35.99], [0])

    survivors = filter_low_information_candidates(candidates, [], frame_metrics=table)

    assert len(survivors) == 0


def test_low_information_filter_keeps_meeting_title_card():
    candidates = [
        {
            "sample_idx": 0,
            "timestamp": 1.0,
            "caption": "recorded meeting title",
            "ocr_tokens": ["testing", "session", "recorded"],
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


def test_adjacent_same_screen_dedupe_collapses_neighboring_list_states():
    shared = {"request", "case", "unit", "owner", "date", "list", "dashboard", "row", "team", "queue"}
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
    tokens = {"request", "case", "unit", "owner", "date", "list", "dashboard"}
    candidates = [
        {"sample_idx": 1, "timestamp": 10.0, "ocr_tokens": sorted(tokens)},
        {"sample_idx": 2, "timestamp": 101.1, "ocr_tokens": sorted(tokens)},
    ]

    survivors = adjacent_same_screen_dedupe(candidates)

    assert [c["sample_idx"] for c in survivors] == [1, 2]


def test_content_area_duplicate_veto_drops_cursor_jitter():
    base = Image.new("RGB", (120, 80), "white")
    jitter = base.copy()
    jitter.putpixel((4, 4), (0, 0, 0))
    candidates = [
        {"sample_idx": 0, "timestamp": 10.0, "scene_id": 1, "dwell_id": 1, "ocr_tokens": ["page1", "form"]},
        {"sample_idx": 1, "timestamp": 11.0, "scene_id": 1, "dwell_id": 1, "ocr_tokens": ["page1", "form"]},
    ]

    survivors, dropped = content_area_duplicate_veto(candidates, [base, jitter])

    assert [row["sample_idx"] for row in survivors] == [0]
    assert dropped[0]["reason"] == "near_identical_content_area"


def test_content_area_duplicate_veto_uses_cached_content_metrics_without_frame_lookup():
    image = Image.new("RGB", (120, 80), "white")
    table = build_frame_metric_table([image, image.copy()], [10.0, 11.0], [0, 1])
    candidates = [
        {"sample_idx": 0, "timestamp": 10.0, "scene_id": 1, "dwell_id": 1, "ocr_tokens": ["page1", "form"]},
        {"sample_idx": 1, "timestamp": 11.0, "scene_id": 1, "dwell_id": 1, "ocr_tokens": ["page1", "form"]},
    ]

    survivors, dropped = content_area_duplicate_veto(candidates, [], frame_metrics=table)

    assert [row["sample_idx"] for row in survivors] == [0]
    assert dropped[0]["mean_abs_content_delta"] == 0.0


def test_content_area_duplicate_veto_preserves_form_state_delta():
    image = Image.new("RGB", (120, 80), "white")
    candidates = [
        {"sample_idx": 0, "timestamp": 70.0, "scene_id": 1, "dwell_id": 1, "ocr_tokens": ["completed", "status", "date", "please", "selection"]},
        {"sample_idx": 1, "timestamp": 80.0, "scene_id": 1, "dwell_id": 1, "ocr_tokens": ["completed", "status", "date", "24apr2026"]},
    ]

    survivors, dropped = content_area_duplicate_veto(candidates, [image, image.copy()])

    assert [row["sample_idx"] for row in survivors] == [0, 1]
    assert dropped == []


def test_content_area_duplicate_veto_preserves_generic_status_signature_delta():
    image = Image.new("RGB", (120, 80), "white")
    candidates = [
        {
            "sample_idx": 0,
            "timestamp": 70.0,
            "scene_id": 1,
            "dwell_id": 1,
            "ocr_tokens": ["review", "record", "status"],
            "field_signature": field_section_signatures("Status Draft"),
        },
        {
            "sample_idx": 1,
            "timestamp": 80.0,
            "scene_id": 1,
            "dwell_id": 1,
            "ocr_tokens": ["review", "record", "status"],
            "field_signature": field_section_signatures("Status Approved"),
        },
    ]

    survivors, dropped = content_area_duplicate_veto(candidates, [image, image.copy()])

    assert [row["sample_idx"] for row in survivors] == [0, 1]
    assert dropped == []


def test_content_area_duplicate_veto_preserves_generic_label_value_signature_delta():
    image = Image.new("RGB", (120, 80), "white")
    candidates = [
        {
            "sample_idx": 0,
            "timestamp": 70.0,
            "scene_id": 1,
            "dwell_id": 1,
            "ocr_tokens": ["control", "id", "record"],
            "field_signature": field_section_signatures("Control ID: 12345"),
        },
        {
            "sample_idx": 1,
            "timestamp": 80.0,
            "scene_id": 1,
            "dwell_id": 1,
            "ocr_tokens": ["control", "id", "record"],
            "field_signature": field_section_signatures("Control ID: 67890"),
        },
    ]

    survivors, dropped = content_area_duplicate_veto(candidates, [image, image.copy()])

    assert [row["sample_idx"] for row in survivors] == [0, 1]
    assert dropped == []


def test_ocr_signatures_capture_page_marker_changes():
    left_text = "Page 1"
    right_text = "Page 2"

    left_lines = normalized_ocr_line_signatures(left_text)
    right_lines = normalized_ocr_line_signatures(right_text)
    left_fields = field_section_signatures(left_text)
    right_fields = field_section_signatures(right_text)

    assert "page 1" in left_lines
    assert "page:1" in left_fields
    assert "page 2" in right_lines
    assert "page:2" in right_fields


def test_ocr_signatures_capture_date_value_appearing():
    left_fields = field_section_signatures("Review Date")
    right_fields = field_section_signatures("Review Date: 24APR2026")

    assert "date:24apr2026" not in left_fields
    assert "date:24apr2026" in right_fields
    assert left_fields != right_fields


def test_ocr_signatures_capture_status_marker_changes():
    left_fields = field_section_signatures("Status Draft")
    right_fields = field_section_signatures("Status Approved")

    assert "status:draft" in left_fields
    assert "status:approved" in right_fields
    assert left_fields != right_fields


def test_ocr_signatures_capture_generic_label_value_pair_changes():
    left_fields = field_section_signatures("Control ID: 12345")
    right_fields = field_section_signatures("Control ID: 67890")
    left_labels = {sig for sig in left_fields if sig.startswith("label:")}
    right_labels = {sig for sig in right_fields if sig.startswith("label:")}
    left_values = {sig for sig in left_fields if sig.startswith("label-value:")}
    right_values = {sig for sig in right_fields if sig.startswith("label-value:")}

    assert left_labels
    assert left_labels == right_labels
    assert left_values
    assert right_values
    assert left_values != right_values


def test_structured_delta_classifier_covers_supported_material_categories():
    cases = [
        ("Status: Draft", "Status: Approved", {"status", "same_label_value"}),
        ("Review Date:", "Review Date: 24APR2026", {"blank_populated", "date"}),
        ("Page 1", "Page 2", {"page"}),
        ("Section 1", "Section 2", {"section"}),
        ("Control ID: 12345", "Control ID: 67890", {"same_label_value"}),
        ("Field:", "Field: populated", {"blank_populated"}),
        ("Field: populated", "Field:", {"blank_populated"}),
    ]

    for left_text, right_text, expected in cases:
        left = field_section_signatures(left_text)
        right = field_section_signatures(right_text)
        assert set(structured_delta_categories(left, right)) == expected
        assert structured_signature_change_count(left, right) > 0


def test_page_and_section_categories_require_values_on_both_sides():
    no_marker = field_section_signatures("Document body")
    page = field_section_signatures("Page 1")
    section = field_section_signatures("Section 2")

    assert structured_delta_categories(no_marker, page) == ()
    assert structured_delta_categories(no_marker, section) == ()
    assert structured_signature_change_count(no_marker, page) == 0
    assert structured_signature_change_count(no_marker, section) == 0


def test_explicit_label_followed_by_value_line_records_populated_state():
    blank = field_section_signatures("Control ID:\n")
    populated = field_section_signatures("Control ID:\n12345")

    assert structured_delta_categories(blank, populated) == (
        "blank_populated",
    )
    assert any(
        signature.startswith("label-value:")
        for signature in populated
    )


def test_structured_signatures_reject_long_labels_headings_and_character_noise():
    long_label = field_section_signatures(
        "one two three four five six seven eight nine: populated"
    )
    heading = field_section_signatures("Release Checklist")
    blank = field_section_signatures("Code:")
    one_character = field_section_signatures("Code: X")

    assert long_label == ()
    assert heading == ()
    assert structured_delta_categories(blank, one_character) == ()
    assert not any(
        signature.startswith("label-value:")
        for signature in one_character
    )


def test_structured_signatures_reject_prose_clock_path_and_browser_chrome_labels():
    noisy = (
        "GIVEN: a user\n"
        "WHEN: the user saves\n"
        "THEN: the record appears\n"
        "Note: continue scrolling\n"
        "4/30/2026 12:00 AM\n"
        "< > C File C:/Users/example/document.pdf\n"
        "Ca NPX Favourites :8 Benefits C ECHO\n"
        "© AMOT-0561-CoverPage: x"
    )

    assert field_section_signatures(noisy) == ()


def test_structured_signatures_reject_uri_and_bare_clock_labels():
    noisy = (
        "https://example.com/forms/123\n"
        "12:00\n"
        "12:00 PM"
    )

    assert field_section_signatures(noisy) == ()


def test_equivalent_status_spellings_share_canonical_signatures():
    pairs = [
        ("Status: Approve", "Status: Approved"),
        ("Status: Complete", "Status: Completed"),
        ("Status Approve", "Status Approved"),
        ("Status Complete", "Status Completed"),
    ]

    for left_text, right_text in pairs:
        left = field_section_signatures(left_text)
        right = field_section_signatures(right_text)
        assert left == right
        assert structured_delta_categories(left, right) == ()


def test_structured_signatures_ignore_one_character_value_ocr_edits():
    left = field_section_signatures("Facility: Bruce Al")
    right = field_section_signatures("Facility: Bruce A")

    assert structured_delta_categories(left, right) == ()
    assert structured_signature_change_count(left, right) == 0


def test_structured_signatures_require_context_for_status_date_page_and_section():
    fields = field_section_signatures(
        "Power Pages\n"
        "Updated sections: 4, 5, 8\n"
        "The task is complete\n"
        "2026-04-29"
    )

    assert not any(
        signature.startswith(
            ("status:", "date:", "page:", "section:")
        )
        for signature in fields
    )


def test_ocr_signatures_capture_heading_and_line_novelty_without_domain_terms():
    left_text = "Release Checklist"
    right_text = "Deployment Checklist"
    left_lines = normalized_ocr_line_signatures(left_text)
    right_lines = normalized_ocr_line_signatures(right_text)
    left_fields = field_section_signatures(left_text)
    right_fields = field_section_signatures(right_text)

    assert left_lines == ("release checklist",)
    assert right_lines == ("deployment checklist",)
    assert left_fields == ()
    assert right_fields == ()


def test_ocr_signatures_capture_generic_reference_artifact_words():
    fields = field_section_signatures("Source design reference")

    assert fields == ()


def test_ocr_field_signatures_do_not_emit_fixture_domain_concepts():
    fields = field_section_signatures(
        "AMOT # Revision\nPriority Form\nRisk Justification\nOverride Justification\nSigned on Behalf"
    )

    joined = "\n".join(fields).casefold()
    for banned in ("amot", "priority", "risk", "override", "signed", "behalf"):
        assert banned not in joined


def test_structured_candidate_survives_alternate_and_visual_dedupe_stages():
    ordinary = {
        "sample_idx": 0,
        "frame_idx": 0,
        "timestamp": 0.0,
        "clip_cluster": 1,
        "cluster_role": "primary",
        "scene_id": 0,
        "dwell_id": 0,
        "ocr_tokens": ["same", "screen", "text"],
        "candidate_score": 10.0,
    }
    structured = {
        "sample_idx": 1,
        "frame_idx": 1,
        "timestamp": 1.0,
        "clip_cluster": 1,
        "cluster_role": "alt",
        "scene_id": 0,
        "dwell_id": 0,
        "ocr_tokens": ["same", "screen", "text"],
        "candidate_score": 1.0,
        "structured_delta_categories": ["same_label_value"],
        "structured_changed_signature_count": 2,
    }

    retained = retain_cluster_alternates([ordinary, structured])
    assert {row["sample_idx"] for row in retained} == {0, 1}

    near = near_time_dedupe(
        [ordinary, structured],
        [{"same", "screen", "text"}] * 2,
        [0, 0],
    )
    assert {row["sample_idx"] for row in near} == {0, 1}

    global_rows = global_candidate_dedupe(
        [ordinary, structured],
        [{"same", "screen", "text"}] * 2,
        [0, 0],
    )
    assert {row["sample_idx"] for row in global_rows} == {0, 1}

    adjacent = adjacent_same_screen_dedupe([ordinary, structured])
    assert {row["sample_idx"] for row in adjacent} == {0, 1}

    image = Image.new("RGB", (40, 40), "white")
    post_veto, dropped = content_area_duplicate_veto(
        [ordinary, structured],
        [image, image.copy()],
    )
    assert {row["sample_idx"] for row in post_veto} == {0, 1}
    assert dropped == []


def test_structured_candidate_survives_low_information_filter():
    structured = {
        "sample_idx": 0,
        "frame_idx": 0,
        "timestamp": 0.0,
        "ocr_tokens": [],
        "structured_delta_categories": ["blank_populated"],
    }

    survivors = filter_low_information_candidates(
        [structured],
        [Image.new("RGB", (40, 40), "black")],
    )

    assert [row["sample_idx"] for row in survivors] == [0]
    assert (
        survivors[0]["low_information_filter_reason"]
        == "protected_structured_delta"
    )


def test_production_code_does_not_contain_fixture_domain_terms():
    repo_root = Path(__file__).resolve().parents[2]
    banned_terms = (
        "amot",
        "echo",
        "echoteam",
        "override justification",
        "override_justification",
        "priority form",
        "priority_form",
        "risk justification",
        "risk_justification",
        "signed on behalf",
        "signed_on_behalf",
    )
    leaks = []
    for path in sorted((repo_root / "keyframe").rglob("*.py")):
        text = path.read_text(encoding="utf-8").casefold()
        for term in banned_terms:
            if term in text:
                leaks.append(f"{path.relative_to(repo_root)}: {term}")

    assert leaks == []


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

    winner = merge_candidate_lineage(
        as_candidate_record(winner),
        as_candidate_record(loser),
        stage="test",
        reason="component",
    ).to_dict()

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

    winner = merge_candidate_lineage(
        as_candidate_record(winner),
        as_candidate_record(loser),
        stage="test",
        reason="component",
    ).to_dict()

    assert is_protected_candidate(winner)
    assert winner["rescue_origin"] == "additive_rescue"
    assert winner["rescue_reason"] == "evidence_marker"
    assert winner["rescue_origins_seen"] == ["additive_rescue"]
    assert winner["rescue_priorities_seen"] == [2]
    assert winner["lineage_roles"] == ["primary", "rescue"]
