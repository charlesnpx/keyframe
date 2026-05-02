import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


FIXTURES = {
    "36381": {
        "env": "KEYFRAME_QA_FIXTURE_36381",
        "default_path": "videos/36381.mp4",
        "must_include": [
            {
                "time": 201.46,
                "label": "amot_spacing_midpage_consequence_dependencies_comments",
                "tolerance": 2.25,
                "anchor_tokens": ["Consequence of Failure", "Dependencies", "Comments", "Summary of Changes"],
            },
            {
                "time": 244.46,
                "label": "amot_spacing_page_boundary_current_state_description_to_page2",
                "tolerance": 2.25,
                "anchor_tokens": ["Current State", "Description", "Prepared By", "Consequence of Failure"],
            },
            {
                "time": 259.95,
                "label": "source_form_text_fields_for_spacing_comparison",
                "tolerance": 2.25,
                "anchor_tokens": ["Linked Risks", "Description", "Dependencies", "Risk ID"],
            },
            {
                "time": 315.0,
                "label": "priority_form_spacing_sections",
                "tolerance": 2.25,
                "anchor_tokens": ["Priority Form", "Risk Justification", "Other Considerations", "Override Justification"],
            },
        ],
        "should_include_diagnostic": [
            {"time": 45.0, "label": "spacing_design_reference_image"},
            {"time": 75.0, "label": "cover_page_spacing_top_sections"},
            {"time": 280.0, "label": "regenerated_amot_spacing_after_source_edit"},
        ],
        "source_start_seconds": 0.0,
    },
    "36380": {
        "env": "KEYFRAME_QA_FIXTURE_36380",
        "default_path": "videos/36380.mp4",
        "must_include": [
            {
                "time": 110.0,
                "label": "regular_amot_pdf_header_body_fields_wrong_location",
                "tolerance": 2.25,
                "anchor_tokens": ["AMOT # Revision", "AMOT # Revision Date", "Status", "Draft"],
            },
            {
                "time": 180.0,
                "label": "cover_page_unapproved_signed_by_should_be_blank",
                "tolerance": 2.25,
                "anchor_tokens": ["Signed on Behalf of the IMC By", "Alyssa Leon", "Approved Date"],
            },
            {
                "time": 240.0,
                "label": "priority_form_header_fields_in_body",
                "tolerance": 2.25,
                "anchor_tokens": ["Priority Form", "AMOT # Revision", "Cover Page Revision", "Status"],
            },
        ],
        "should_include_diagnostic": [
            {"time": 30.0, "label": "design_reference_header_mockup"},
            {"time": 420.0, "label": "cover_page_approved_footer_and_status_correct"},
            {"time": 550.0, "label": "cover_page_blank_strategy_fields"},
        ],
        "source_start_seconds": 0.0,
    },
    "36324": {
        "env": "KEYFRAME_QA_FIXTURE_36324",
        "default_path": "videos/36324.mp4",
        "must_include": [
            {
                "time": 55.0,
                "label": "impacted_location_completed_status_date_blank_required",
                "tolerance": 2.25,
                "anchor_tokens": [
                    "Impacted Locations",
                    "Equipment Status",
                    "Completed",
                    "Status Date",
                ],
            },
            {
                "time": 70.0,
                "label": "status_date_empty_expected_error_visible",
                "tolerance": 2.25,
                "anchor_tokens": [
                    "Completed",
                    "Please make a selection",
                    "Status Date",
                ],
            },
            {
                "time": 80.0,
                "label": "status_date_filled_error_state_persists",
                "tolerance": 2.25,
                "anchor_tokens": [
                    "Completed",
                    "24APR2026",
                    "Status Date",
                ],
            },
        ],
        "should_include_diagnostic": [
            {"time": 25.0, "label": "impacted_location_row_added_default_not_completed"},
            {"time": 90.0, "label": "filled_date_persistent_red_underline_later"},
        ],
        "source_start_seconds": 0.0,
    },
    "11111": {
        "env": "KEYFRAME_QA_FIXTURE_11111",
        "default_path": "videos/11111.mp4",
        "must_include": [
            {
                "time": 20.0,
                "label": "active_amot_0264_visible_in_list",
                "tolerance": 2.25,
                "anchor_tokens": [
                    "Existing AMOTs",
                    "ACTIVE AMOTS",
                    "0264",
                    "Draft",
                ],
            },
            {
                "time": 42.0,
                "label": "amot_0264_search_value_available",
                "tolerance": 2.25,
                "anchor_tokens": [
                    "AMOT #",
                    "0264",
                    "ACTIVE AMOTS",
                ],
            },
            {
                "time": 45.0,
                "label": "amot_0264_filter_returns_no_templates_found",
                "tolerance": 2.25,
                "anchor_tokens": [
                    "0264",
                    "No AMOT templates found",
                ],
            },
        ],
        "should_include_diagnostic": [
            {"time": 35.0, "label": "amot_filter_dropdown_open_initial_search"},
            {"time": 40.0, "label": "amot_0264_filter_dropdown_option"},
        ],
        "source_start_seconds": 0.0,
    },
}


EXPECTED_BASELINE = {
    "36381": {
        "misses": {
            "amot_spacing_midpage_consequence_dependencies_comments",
            "amot_spacing_page_boundary_current_state_description_to_page2",
            "source_form_text_fields_for_spacing_comparison",
            "priority_form_spacing_sections",
        },
        "buckets": {
            "amot_spacing_midpage_consequence_dependencies_comments": "no_coarse_proposal_near_target",
            "amot_spacing_page_boundary_current_state_description_to_page2": "no_coarse_proposal_near_target",
            "source_form_text_fields_for_spacing_comparison": "no_coarse_proposal_near_target",
            "priority_form_spacing_sections": "no_coarse_proposal_near_target",
        },
    },
    "36380": {
        "misses": {
            "regular_amot_pdf_header_body_fields_wrong_location",
            "cover_page_unapproved_signed_by_should_be_blank",
            "priority_form_header_fields_in_body",
        },
        "buckets": {
            "regular_amot_pdf_header_body_fields_wrong_location": "no_coarse_proposal_near_target",
            "cover_page_unapproved_signed_by_should_be_blank": "rescue_ocrd_but_not_promoted",
            "priority_form_header_fields_in_body": "no_coarse_proposal_near_target",
        },
    },
    "36324": {
        "misses": {
            "impacted_location_completed_status_date_blank_required",
            "status_date_empty_expected_error_visible",
        },
        "buckets": {
            "impacted_location_completed_status_date_blank_required": "rescue_ocrd_but_not_promoted",
            "status_date_empty_expected_error_visible": "rescue_ocrd_but_not_promoted",
            "status_date_filled_error_state_persists": "hit_direct",
        },
    },
    "11111": {
        "misses": set(),
        "buckets": {
            "active_amot_0264_visible_in_list": "hit_direct",
            "amot_0264_search_value_available": "hit_direct",
            "amot_0264_filter_returns_no_templates_found": "hit_direct",
        },
    },
}


def _load_local_env():
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _fixture_enabled():
    _load_local_env()
    return os.environ.get("KEYFRAME_QA_FIXTURES") == "1"


def _fixture_path(annotation):
    configured = os.environ.get(annotation["env"], annotation["default_path"])
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _manifest_timestamps(manifest):
    timestamps = []
    for frame in manifest["frames"]:
        timestamps.append(float(frame["timestamp"]))
        timestamps.extend(float(ts) for ts in frame.get("merged_timestamps", []))
    return timestamps


FIXTURE_RECALL_TOLERANCE_SECONDS = 2.25


def _target_time(target):
    return float(target["time"] if isinstance(target, dict) else target)


def _target_label(target):
    return target.get("label", str(target["time"])) if isinstance(target, dict) else str(target)


def _target_tolerance(target, default=FIXTURE_RECALL_TOLERANCE_SECONDS):
    return float(target.get("tolerance", default)) if isinstance(target, dict) else float(default)


def _recall_at_must_include(
    manifest,
    targets,
    source_start_seconds=0.0,
    tolerance=FIXTURE_RECALL_TOLERANCE_SECONDS,
):
    """Fixture-only tolerance; product target remains approximately +/-2s."""
    timestamps = _manifest_timestamps(manifest)
    hits = 0
    misses = []
    for target in targets:
        adjusted = _target_time(target) - float(source_start_seconds)
        target_tolerance = _target_tolerance(target, tolerance)
        if any(abs(ts - adjusted) <= target_tolerance for ts in timestamps):
            hits += 1
        else:
            misses.append(target)
    recall = 1.0 if not targets else hits / len(targets)
    return recall, misses


def _nearest_deltas(manifest, targets, source_start_seconds=0.0):
    timestamps = _manifest_timestamps(manifest)
    rows = []
    for target in targets:
        adjusted = _target_time(target) - float(source_start_seconds)
        nearest = min(timestamps, key=lambda ts: abs(ts - adjusted)) if timestamps else None
        rows.append({
            "target": _target_time(target),
            "label": _target_label(target),
            "nearest": nearest,
            "delta": None if nearest is None else abs(nearest - adjusted),
            "tolerance": _target_tolerance(target),
        })
    return rows


def _anchor_token_diagnostics(manifest, targets, source_start_seconds=0.0):
    frames = manifest["frames"]
    rows = []
    for target in targets:
        if not isinstance(target, dict) or not target.get("anchor_tokens") or not frames:
            continue
        adjusted = _target_time(target) - float(source_start_seconds)
        nearest = min(frames, key=lambda frame: abs(float(frame["timestamp"]) - adjusted))
        haystack = " ".join(nearest.get("ocr_tokens", [])).casefold() + " " + str(nearest.get("caption", "")).casefold()
        anchors = list(target["anchor_tokens"])
        found = [
            anchor for anchor in anchors
            if all(part.casefold() in haystack for part in str(anchor).split())
        ]
        rows.append({
            "label": _target_label(target),
            "nearest": nearest["timestamp"],
            "found": len(found),
            "total": len(anchors),
            "found_tokens": found,
        })
    return rows


def _structural_redundancy_score(manifest):
    frames = manifest["frames"]
    if len(frames) < 2:
        return 0.0
    redundant = 0
    pairs = 0
    for left, right in zip(frames, frames[1:]):
        left_tokens = set(left.get("ocr_tokens", []))
        right_tokens = set(right.get("ocr_tokens", []))
        if left_tokens or right_tokens:
            overlap = len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)
        else:
            overlap = 0.0
        close_time = abs(float(left["timestamp"]) - float(right["timestamp"])) <= 2.0
        redundant += int(close_time and overlap >= 0.9)
        pairs += 1
    return redundant / pairs


@pytest.mark.slow
@pytest.mark.skipif(not _fixture_enabled(), reason="set KEYFRAME_QA_FIXTURES=1 to run local full-video QA fixtures")
@pytest.mark.parametrize("name, annotation", FIXTURES.items())
def test_full_video_qa_fixture_recall(tmp_path, name, annotation):
    video_path = _fixture_path(annotation)
    if not video_path.exists():
        pytest.skip(f"fixture video not found: {video_path}")

    out_dir = tmp_path / name
    debug_targets_path = tmp_path / f"{name}_debug_targets.json"
    debug_targets_path.write_text(
        json.dumps({"targets": annotation["must_include"]}, indent=2),
        encoding="utf-8",
    )
    command = [
        sys.executable,
        "-m",
        "keyframe.cli",
        str(video_path),
        "--frames-only",
        "--output",
        str(out_dir),
        "--debug-qa-targets",
        str(debug_targets_path),
    ]
    subprocess.run(command, check=True)

    frames_dir = out_dir / "frames"
    captions_path = frames_dir / "captions.json"
    manifest_path = frames_dir / "manifest.json"
    debug_trace_path = frames_dir / "debug_qa_trace.json"
    captions = json.loads(captions_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pngs = sorted(frames_dir.glob("*.png"))

    assert captions
    assert len(pngs) == len(captions)
    assert len(manifest["frames"]) == len(captions)

    recall, misses = _recall_at_must_include(
        manifest,
        annotation["must_include"],
        source_start_seconds=annotation.get("source_start_seconds", 0.0),
    )
    redundancy = _structural_redundancy_score(manifest)
    rescue_metadata = manifest.get("metadata", {}).get("rescue", {})
    pre_rescue_count = int(rescue_metadata.get("pre_rescue_candidate_count", len(captions)))
    rescue_budget = int(rescue_metadata.get("rescue_budget", 0))
    rescues = [
        {
            "timestamp": frame["timestamp"],
            "rescue_origin": frame.get("rescue_origin"),
            "rescue_reason": frame.get("rescue_reason"),
            "proxy_content_score": frame.get("proxy_content_score"),
        }
        for frame in manifest["frames"]
        if frame.get("rescue_origin")
    ]
    print(
        f"{name}: recall={recall:.3f}, misses={misses}, "
        f"structural_redundancy_score={redundancy:.3f}, output_count={len(captions)}, "
        f"pre_rescue_candidate_count={pre_rescue_count}, rescue_budget={rescue_budget}, "
        f"fixture_tolerance={FIXTURE_RECALL_TOLERANCE_SECONDS}s"
    )
    print(f"{name}: selected_timestamps={[round(float(f['timestamp']), 3) for f in manifest['frames']]}")
    print(f"{name}: rescue_promotions={rescues}")
    print(
        f"{name}: nearest_deltas="
        f"{_nearest_deltas(manifest, annotation['must_include'], annotation.get('source_start_seconds', 0.0))}"
    )
    diagnostics = annotation.get("should_include_diagnostic", [])
    if diagnostics:
        print(
            f"{name}: diagnostic_nearest_deltas="
            f"{_nearest_deltas(manifest, diagnostics, annotation.get('source_start_seconds', 0.0))}"
        )
    anchor_diagnostics = _anchor_token_diagnostics(
        manifest,
        annotation["must_include"],
        annotation.get("source_start_seconds", 0.0),
    )
    if anchor_diagnostics:
        print(f"{name}: anchor_token_diagnostics={anchor_diagnostics}")
    trace = None
    if debug_trace_path.exists():
        trace = json.loads(debug_trace_path.read_text(encoding="utf-8"))
        trace_summary = [
            {
                "label": target["label"],
                "bucket": target["bucket"],
                "nearest_final_timestamp": target["nearest_final_timestamp"],
                "nearest_final_delta": target["nearest_final_delta"],
            }
            for target in trace.get("targets", [])
        ]
        print(f"{name}: debug_qa_trace={debug_trace_path}")
        print(f"{name}: debug_qa_target_summary={trace_summary}")
        if trace.get("integrity_violations"):
            print(f"{name}: debug_qa_integrity_violations={trace['integrity_violations']}")
    assert redundancy <= 0.10
    assert len(captions) <= pre_rescue_count + rescue_budget
    expected = EXPECTED_BASELINE[name]
    assert {miss["label"] for miss in misses} == expected["misses"]
    if trace is not None:
        buckets = {target["label"]: target["bucket"] for target in trace.get("targets", [])}
        for label, bucket in expected["buckets"].items():
            assert buckets[label] == bucket
