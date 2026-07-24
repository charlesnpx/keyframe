import numpy as np
import pytest
from PIL import Image
import json

from keyframe.evidence import (
    field_section_signatures,
    select_structured_comparator,
)
from keyframe.frames import (
    _comparison_primary_sample_idxs,
    attach_structured_delta_metadata,
    ocr_candidates as _ocr_candidates,
)
from keyframe.pipeline.contracts import candidate_records
from keyframe.scoring import (
    assign_dwell_ids,
    build_rescue_shortlist as _build_rescue_shortlist,
    promote_rescue_candidates as _promote_rescue_candidates,
    proxy_content_scores,
    rescue_promotion_preflight_report,
)
from keyframe.visual import (
    FrameMetricTable,
    build_compact_frame_metric_table,
    build_frame_metric_table,
)


def _project(records):
    return [record.to_dict() for record in records]


def promote_rescue_candidates(*args, **kwargs):
    return _project(_promote_rescue_candidates(*args, **kwargs))


def build_rescue_shortlist(*args, **kwargs):
    shortlist, proxy_rows, budget, *_metadata = _build_rescue_shortlist(*args, **kwargs)
    return _project(shortlist), proxy_rows, budget


def build_rescue_shortlist_with_metadata(*args, **kwargs):
    (
        shortlist,
        proxy_rows,
        budget,
        ocr_cap,
        window_count,
        scene_count,
        legacy_proxy_dropped_count,
        proposal_metadata,
    ) = _build_rescue_shortlist(*args, **kwargs)
    return (
        _project(shortlist),
        proxy_rows,
        budget,
        ocr_cap,
        window_count,
        scene_count,
        legacy_proxy_dropped_count,
        proposal_metadata,
    )


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


def _metric_table(scores, *, timestamps=None, prev_deltas=None, textline_scores=None):
    n = len(scores)
    scores = np.asarray(scores, dtype=np.float32)
    timestamps = np.asarray(timestamps if timestamps is not None else [float(i) for i in range(n)], dtype=np.float64)
    prev = np.asarray(prev_deltas if prev_deltas is not None else [0.0] * n, dtype=np.float32)
    next_delta = np.zeros((n,), dtype=np.float32)
    if n > 1:
        next_delta[:-1] = prev[1:]
    textline = np.asarray(textline_scores if textline_scores is not None else scores, dtype=np.float32)
    zeros = np.zeros((n,), dtype=np.float32)
    content_stack = np.zeros((n, 1, 1), dtype=np.float32)
    for idx, value in enumerate(np.cumsum(prev)):
        content_stack[idx, :, :] = float(value % 255.0)
    return FrameMetricTable(
        sample_idx=np.arange(n, dtype=np.int64),
        frame_idx=np.arange(n, dtype=np.int64),
        timestamp=timestamps,
        textline_score=textline,
        edge_score=scores,
        entropy=scores,
        dark_ratio=zeros.copy(),
        bright_ratio=zeros.copy(),
        normalized_textline_score=textline,
        normalized_edge_score=scores,
        normalized_entropy=scores,
        blank_penalty=zeros.copy(),
        proxy_content_score=scores,
        content_prev_delta=prev,
        content_next_delta=next_delta,
        content_area_delta_score=np.maximum(prev, next_delta) / 255.0,
        visual_stddev=np.full((n,), 32.0, dtype=np.float32),
        visual_edge_score=np.full((n,), 32.0, dtype=np.float32),
        visual_dark_ratio=zeros.copy(),
        visual_bright_ratio=zeros.copy(),
        visual_entropy=np.full((n,), 3.0, dtype=np.float32),
        visual_unique_buckets=np.full((n,), 8.0, dtype=np.float32),
        sharpness=np.linspace(10.0, 10.0 + n, n, dtype=np.float32),
        full_gray_stack=np.zeros((n, 90, 160), dtype=np.float32),
        content_gray_stack=content_stack,
    )


def test_proxy_content_scores_clamp_and_no_variance_normalizes_to_zero():
    frames = [Image.new("RGB", (32, 32), "white"), Image.new("RGB", (32, 32), "white")]

    scores = proxy_content_scores(frames)

    assert [row["normalized_textline_score"] for row in scores] == [0.0, 0.0]
    assert all(0.0 <= row["proxy_content_score"] <= 1.0 for row in scores)


def test_frame_metric_table_flat_frames_normalize_without_divide_by_zero():
    frames = [Image.new("RGB", (32, 32), "white"), Image.new("RGB", (32, 32), "black")]

    table = build_frame_metric_table(frames, [0.0, 1.0], [0, 1])

    assert table.sample_count == 2
    assert np.isfinite(table.proxy_content_score).all()
    assert np.isfinite(table.normalized_textline_score).all()
    assert all(0.0 <= float(score) <= 1.0 for score in table.proxy_content_score)


def test_frame_metric_table_proxy_rows_match_adapter():
    frames = [
        Image.new("RGB", (64, 64), "white"),
        Image.new("RGB", (64, 64), "black"),
        Image.effect_noise((64, 64), 40).convert("RGB"),
    ]

    table_rows = build_frame_metric_table(frames, [0.0, 1.0, 2.0], [0, 1, 2]).to_proxy_rows()
    adapter_rows = proxy_content_scores(frames)

    for table_row, adapter_row in zip(table_rows, adapter_rows):
        assert table_row["proxy_content_score"] == pytest.approx(adapter_row["proxy_content_score"], abs=1e-6)
        assert table_row["edge_score"] == pytest.approx(adapter_row["edge_score"], abs=1e-6)


def test_frame_metric_table_adjacent_deltas_and_cached_metrics_by_sample_idx():
    frames = [Image.new("RGB", (40, 40), "white"), Image.new("RGB", (40, 40), "black")]

    table = build_frame_metric_table(frames, [0.0, 1.0], [10, 20])

    assert table.content_prev_delta[1] > 200.0
    assert table.content_next_delta[0] > 200.0
    assert table.visual_information_for(1)["dark_ratio"] > 0.95
    assert table.sharpness_for(0) is not None


def test_assign_dwell_ids_groups_adjacent_similar_hashes():
    assert assign_dwell_ids([0b0000, 0b0001, 0b1111], hamming_threshold=1) == [0, 0, 1]


def test_rescue_shortlist_backfills_when_proxy_scores_are_flat():
    frames = [Image.new("RGB", (8, 8), "white") for _ in range(8)]
    candidates = [{"sample_idx": 0, "timestamp": 0.0, "scene_id": 0}]
    frame_metrics = _metric_table([0.0] * 8)

    shortlist, _proxy_rows, budget = build_rescue_shortlist(
        frames,
        [float(i) for i in range(8)],
        list(range(8)),
        candidates,
        pass1_clusters=3,
        sample_scenes={i: 0 for i in range(8)},
        frame_metrics=frame_metrics,
    )

    assert budget == 3
    assert len(shortlist) >= budget
    assert 0 not in {row["sample_idx"] for row in shortlist}


def test_rescue_shortlist_includes_per_scene_coverage():
    scores = [0.99, 0.98, 0.97, 0.96, 0.10, 0.09]
    frames = [Image.new("RGB", (8, 8), "white") for _ in scores]
    candidates = [
        {"sample_idx": 0, "timestamp": 0.0, "scene_id": 0},
        {"sample_idx": 5, "timestamp": 5.0, "scene_id": 1},
    ]
    frame_metrics = _metric_table(scores)

    shortlist, _proxy_rows, _budget = build_rescue_shortlist(
        frames,
        [float(i) for i in range(len(scores))],
        list(range(len(scores))),
        candidates,
        pass1_clusters=3,
        sample_scenes={0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1},
        frame_metrics=frame_metrics,
    )

    assert 4 in {row["sample_idx"] for row in shortlist}


def test_rescue_budget_duration_floor_is_bounded():
    frames = [Image.new("RGB", (8, 8), "white") for _ in range(12)]
    timestamps = [float(i * 90) for i in range(12)]
    frame_metrics = _metric_table([0.5] * 12, timestamps=timestamps)

    _shortlist, _proxy_rows, budget = build_rescue_shortlist(
        frames,
        timestamps,
        list(range(12)),
        [{"sample_idx": 0, "timestamp": 0.0, "scene_id": 0}],
        pass1_clusters=3,
        sample_scenes={i: i // 3 for i in range(12)},
        frame_metrics=frame_metrics,
    )

    assert budget == 8


def test_rescue_ocr_cap_scales_with_temporal_windows_not_only_output_budget():
    frames = [Image.new("RGB", (8, 8), "white") for _ in range(50)]
    frame_metrics = _metric_table([0.5] * 50)

    _shortlist, _proxy_rows, budget, ocr_cap, window_count, scene_count, _dropped, _metadata = build_rescue_shortlist_with_metadata(
        frames,
        [float(i) for i in range(50)],
        list(range(50)),
        [{"sample_idx": 0, "timestamp": 0.0, "scene_id": 0}],
        pass1_clusters=3,
        sample_scenes={i: 0 for i in range(50)},
        frame_metrics=frame_metrics,
    )

    assert budget == 3
    assert window_count == 3
    assert scene_count == 1
    assert ocr_cap == 40


def test_coverage_shortlist_is_monotonic_with_legacy_proxy_shortlist():
    scores = [0.99] * 20 + [0.10] * 20 + [0.01] * 5
    frames = [Image.new("RGB", (8, 8), "white") for _ in scores]
    frame_metrics = _metric_table(scores)

    shortlist, _proxy_rows, budget, ocr_cap, window_count, _scene_count, dropped, _metadata = build_rescue_shortlist_with_metadata(
        frames,
        [float(i) for i in range(len(scores))],
        list(range(len(scores))),
        [{"sample_idx": 0, "timestamp": 0.0, "scene_id": 0}],
        pass1_clusters=3,
        sample_scenes={i: 0 for i in range(len(scores))},
        frame_metrics=frame_metrics,
    )

    assert budget == 3
    assert ocr_cap == 40
    assert window_count == 3
    sample_idxs = {row["sample_idx"] for row in shortlist}
    assert set(range(1, 13)) <= sample_idxs
    assert {20, 40} <= sample_idxs
    assert dropped == 0
    lane_by_idx = {row["sample_idx"]: row["proposal_lane"] for row in shortlist}
    assert {
        lane_by_idx[idx]
        for idx in (1, 2, 3)
    } <= {"temporal_coverage", "scene_coverage"}


def test_global_high_proxy_frames_do_not_starve_later_windows():
    scores = [0.99] * 60 + [0.01] * 20
    frames = [Image.new("RGB", (8, 8), "white") for _ in scores]
    frame_metrics = _metric_table(scores)

    shortlist, _proxy_rows, _budget, ocr_cap, window_count, _scene_count, dropped, _metadata = build_rescue_shortlist_with_metadata(
        frames,
        [float(i) for i in range(len(scores))],
        list(range(len(scores))),
        [{"sample_idx": 0, "timestamp": 0.0, "scene_id": 0}],
        pass1_clusters=3,
        sample_scenes={i: 0 for i in range(len(scores))},
        frame_metrics=frame_metrics,
    )

    assert ocr_cap == 40
    assert window_count == 4
    assert 60 in {row["sample_idx"] for row in shortlist}
    assert dropped == 0


def test_rescue_candidate_records_proposal_lane_metadata():
    frames = [Image.new("RGB", (8, 8), "white") for _ in range(30)]
    frame_metrics = _metric_table([0.5] * 30)

    shortlist, *_ = build_rescue_shortlist_with_metadata(
        frames,
        [float(i) for i in range(30)],
        list(range(30)),
        [{"sample_idx": 0, "timestamp": 0.0, "scene_id": 0}],
        pass1_clusters=3,
        sample_scenes={i: 0 for i in range(30)},
        frame_metrics=frame_metrics,
    )

    assert all(row.get("proposal_lane") for row in shortlist)


def test_content_area_delta_lane_proposes_settled_transition_frame():
    frames = [Image.new("RGB", (40, 40), "white") for _ in range(8)]
    for idx in range(3, 8):
        frames[idx] = Image.new("RGB", (40, 40), "black")
    frame_metrics = _metric_table([0.1] * 8, prev_deltas=[0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 0.0])

    shortlist, *_ = build_rescue_shortlist_with_metadata(
        frames,
        [float(i) for i in range(len(frames))],
        list(range(len(frames))),
        [{"sample_idx": 0, "timestamp": 0.0, "scene_id": 0}],
        pass1_clusters=3,
        sample_scenes={i: 0 for i in range(len(frames))},
        frame_metrics=frame_metrics,
    )

    lane_by_idx = {row["sample_idx"]: row["proposal_lane"] for row in shortlist}
    assert lane_by_idx[3] == "transition"


def test_transition_proposes_nearest_pre_and_post_sides_with_boundary_provenance():
    frames = [Image.new("RGB", (8, 8), "white") for _ in range(8)]
    metrics = _metric_table(
        [0.1] * 8,
        prev_deltas=[0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0],
    )

    shortlist, *_rest, metadata = build_rescue_shortlist_with_metadata(
        frames,
        [float(i) for i in range(8)],
        list(range(8)),
        [_cand(7, 7.0)],
        pass1_clusters=3,
        sample_scenes={i: 0 for i in range(8)},
        frame_metrics=metrics,
        dhashes=[0, 0, 0, (1 << 16) - 1, (1 << 16) - 1, (1 << 16) - 1, (1 << 16) - 1, (1 << 16) - 1],
    )

    transition_rows = {
        (row["transition_side"], row["sample_idx"]): row
        for row in shortlist
        if row["proposal_lane"] == "transition"
        and row["transition_boundary_sample_idx"] == 3
    }
    assert set(transition_rows) == {("pre", 2), ("post", 3)}
    assert all(
        row["transition_boundary_timestamp"] == 3.0
        and row["transition_boundary_content_delta"] == 20.0
        for row in transition_rows.values()
    )
    decisions = metadata["proposal_decisions"]
    assert any(
        row["decision"] == "transition_qualified"
        and row["boundary_sample_idx"] == 3
        for row in decisions
    )
    assert {
        row["transition_side"]
        for row in decisions
        if row["decision"] == "transition_side_proposed"
        and row["boundary_sample_idx"] == 3
    } == {"pre", "post"}


def test_transition_qualification_accepts_text_band_predicate():
    frames = [Image.new("RGB", (8, 8), "white") for _ in range(6)]
    metrics = _metric_table(
        [0.1] * 6,
        textline_scores=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
    )

    shortlist, *_rest, metadata = build_rescue_shortlist_with_metadata(
        frames,
        [float(i) for i in range(6)],
        list(range(6)),
        [_cand(5, 5.0)],
        pass1_clusters=3,
        sample_scenes={i: 0 for i in range(6)},
        frame_metrics=metrics,
        dhashes=[0] * 6,
    )

    assert any(
        row["proposal_lane"] == "transition"
        and row["transition_boundary_sample_idx"] == 3
        for row in shortlist
    )
    qualified = next(
        row
        for row in metadata["proposal_decisions"]
        if row["decision"] == "transition_qualified"
        and row["boundary_sample_idx"] == 3
    )
    assert "text_band" in qualified["predicates"]
    assert "text_band_threshold" in qualified["distinct_reasons"]


def test_dhash_distinctness_alone_does_not_qualify_transition():
    frames = [Image.new("RGB", (8, 8), "white") for _ in range(6)]
    metrics = _metric_table([0.1] * 6, textline_scores=[0.0] * 6)

    shortlist, *_rest, metadata = build_rescue_shortlist_with_metadata(
        frames,
        [float(i) for i in range(6)],
        list(range(6)),
        [_cand(5, 5.0)],
        pass1_clusters=3,
        sample_scenes={i: 0 for i in range(6)},
        frame_metrics=metrics,
        dhashes=[
            0,
            0,
            0,
            (1 << 16) - 1,
            (1 << 16) - 1,
            (1 << 16) - 1,
        ],
    )

    assert not any(
        row["proposal_lane"] == "transition"
        and row.get("transition_boundary_sample_idx") == 3
        for row in shortlist
    )
    rejected = next(
        row
        for row in metadata["proposal_decisions"]
        if row["decision"] == "transition_rejected"
        and row["boundary_sample_idx"] == 3
    )
    assert rejected["reason"] == "transition_predicate_not_met"
    assert rejected["predicates"] == []
    assert "dhash_threshold" in rejected["distinct_reasons"]


def test_dhash_is_recorded_as_distinctness_for_qualified_transition():
    frames = [Image.new("RGB", (8, 8), "white") for _ in range(6)]
    metrics = _metric_table(
        [0.1] * 6,
        prev_deltas=[0.0, 0.0, 0.0, 20.0, 0.0, 0.0],
        textline_scores=[0.0] * 6,
    )

    _shortlist, *_rest, metadata = build_rescue_shortlist_with_metadata(
        frames,
        [float(i) for i in range(6)],
        list(range(6)),
        [_cand(5, 5.0)],
        pass1_clusters=3,
        sample_scenes={i: 0 for i in range(6)},
        frame_metrics=metrics,
        dhashes=[
            0,
            0,
            0,
            (1 << 16) - 1,
            (1 << 16) - 1,
            (1 << 16) - 1,
        ],
    )

    qualified = next(
        row
        for row in metadata["proposal_decisions"]
        if row["decision"] == "transition_qualified"
        and row["boundary_sample_idx"] == 3
    )
    assert {"local_peak", "settled_transition"} <= set(
        qualified["predicates"]
    )
    assert "dhash_threshold" in qualified["distinct_reasons"]


@pytest.mark.parametrize(
    ("boundary_idx", "selected_idx", "expected_timestamp"),
    [(3, 10, 7.0), (8, 0, 4.0)],
)
def test_temporal_coverage_fills_gap_between_transition_and_existing_selection(
    monkeypatch,
    boundary_idx,
    selected_idx,
    expected_timestamp,
):
    monkeypatch.setattr(
        "keyframe.scoring.rescue_window_seconds",
        lambda _timestamps: 20.0,
    )
    sample_count = 11
    previous_deltas = [0.0] * sample_count
    previous_deltas[boundary_idx] = 20.0
    metrics = _metric_table(
        [0.1] * sample_count,
        prev_deltas=previous_deltas,
        textline_scores=[0.0] * sample_count,
    )
    dhashes = [0] * sample_count
    dhashes[boundary_idx:] = [
        (1 << 16) - 1
    ] * (sample_count - boundary_idx)

    _shortlist, *_rest, metadata = build_rescue_shortlist_with_metadata(
        [Image.new("RGB", (8, 8), "white") for _ in range(sample_count)],
        [float(i) for i in range(sample_count)],
        list(range(sample_count)),
        [_cand(selected_idx, float(selected_idx))],
        pass1_clusters=3,
        sample_scenes={i: 0 for i in range(sample_count)},
        frame_metrics=metrics,
        dhashes=dhashes,
    )

    first_coverage = next(
        row
        for row in metadata["proposal_decisions"]
        if row["decision"] == "quota_allocated"
        and row["allocation_phase"] == "reserved_first_temporal_coverage"
    )
    assert first_coverage["timestamp"] == expected_timestamp


def test_temporal_coverage_ignores_marginal_transition_before_selection():
    sample_count = 11
    previous_deltas = [0.0] * sample_count
    previous_deltas[1] = 20.0
    metrics = _metric_table(
        [0.1] * sample_count,
        prev_deltas=previous_deltas,
        textline_scores=[0.0] * sample_count,
    )

    _shortlist, *_rest, metadata = build_rescue_shortlist_with_metadata(
        [Image.new("RGB", (8, 8), "white") for _ in range(sample_count)],
        [float(i) for i in range(sample_count)],
        list(range(sample_count)),
        [_cand(10, 10.0)],
        pass1_clusters=3,
        sample_scenes={i: 0 for i in range(sample_count)},
        frame_metrics=metrics,
        dhashes=[0] * sample_count,
    )

    first_coverage = next(
        row
        for row in metadata["proposal_decisions"]
        if row["decision"] == "quota_allocated"
        and row["allocation_phase"]
        == "reserved_first_temporal_coverage"
    )
    assert first_coverage["timestamp"] == 2.0


@pytest.mark.parametrize(
    ("selected_idx", "expected_timestamp"),
    [(10, 2.0), (0, 8.0)],
)
def test_temporal_coverage_targets_opposite_existing_selection_without_transition(
    monkeypatch,
    selected_idx,
    expected_timestamp,
):
    monkeypatch.setattr(
        "keyframe.scoring.rescue_window_seconds",
        lambda _timestamps: 20.0,
    )
    sample_count = 11
    metrics = _metric_table([0.1] * sample_count)

    _shortlist, *_rest, metadata = build_rescue_shortlist_with_metadata(
        [Image.new("RGB", (8, 8), "white") for _ in range(sample_count)],
        [float(i) for i in range(sample_count)],
        list(range(sample_count)),
        [_cand(selected_idx, float(selected_idx))],
        pass1_clusters=3,
        sample_scenes={i: 0 for i in range(sample_count)},
        frame_metrics=metrics,
    )

    first_coverage = next(
        row
        for row in metadata["proposal_decisions"]
        if row["decision"] == "quota_allocated"
        and row["allocation_phase"] == "reserved_first_temporal_coverage"
    )
    assert first_coverage["timestamp"] == expected_timestamp


def test_temporal_coverage_uses_farthest_point_order_after_target():
    sample_count = 11
    previous_deltas = [0.0] * sample_count
    previous_deltas[3] = 20.0
    metrics = _metric_table(
        [0.1] * sample_count,
        prev_deltas=previous_deltas,
        textline_scores=[0.0] * sample_count,
    )
    dhashes = [0] * sample_count
    dhashes[3:] = [
        (1 << 16) - 1
    ] * (sample_count - 3)

    _shortlist, *_rest, metadata = build_rescue_shortlist_with_metadata(
        [Image.new("RGB", (8, 8), "white") for _ in range(sample_count)],
        [float(i) for i in range(sample_count)],
        list(range(sample_count)),
        [_cand(10, 10.0)],
        pass1_clusters=3,
        sample_scenes={i: 0 for i in range(sample_count)},
        frame_metrics=metrics,
        dhashes=dhashes,
    )

    coverage_timestamps = [
        row["timestamp"]
        for row in metadata["proposal_decisions"]
        if row["decision"] == "quota_allocated"
        and row["proposal_lane"] == "temporal_coverage"
    ]
    assert coverage_timestamps[:2] == [7.0, 0.0]


def test_temporal_coverage_captures_settled_side_after_delayed_transition_tail():
    sample_count = 40
    previous_deltas = [0.0] * sample_count
    previous_deltas[3] = 20.0
    previous_deltas[5] = 5.0
    metrics = _metric_table(
        [0.1] * sample_count,
        prev_deltas=previous_deltas,
        textline_scores=[0.0] * sample_count,
    )

    _shortlist, *_rest, metadata = build_rescue_shortlist_with_metadata(
        [
            Image.new("RGB", (8, 8), "white")
            for _ in range(sample_count)
        ],
        [float(i) for i in range(sample_count)],
        list(range(sample_count)),
        [_cand(sample_count - 1, float(sample_count - 1))],
        pass1_clusters=3,
        sample_scenes={i: 0 for i in range(sample_count)},
        frame_metrics=metrics,
        dhashes=[
            0,
            0,
            0,
            (1 << 16) - 1,
            (1 << 16) - 1,
            (1 << 16) - 1,
            *([(1 << 16) - 1] * (sample_count - 6)),
        ],
    )

    first_window_coverage = next(
        row
        for row in metadata["proposal_decisions"]
        if row["decision"] == "quota_allocated"
        and row["allocation_phase"]
        == "reserved_first_temporal_coverage"
        and row["timestamp"] < 20.0
    )
    assert first_window_coverage["timestamp"] == 7.0


def test_temporal_coverage_keeps_far_side_for_marginal_content_transition():
    sample_count = 40
    previous_deltas = [0.0] * sample_count
    previous_deltas[3] = 20.0
    previous_deltas[5] = 5.0
    metrics = _metric_table(
        [0.1] * sample_count,
        prev_deltas=previous_deltas,
        textline_scores=[0.0] * sample_count,
    )

    _shortlist, *_rest, metadata = build_rescue_shortlist_with_metadata(
        [
            Image.new("RGB", (8, 8), "white")
            for _ in range(sample_count)
        ],
        [float(i) for i in range(sample_count)],
        list(range(sample_count)),
        [_cand(sample_count - 1, float(sample_count - 1))],
        pass1_clusters=3,
        sample_scenes={i: 0 for i in range(sample_count)},
        frame_metrics=metrics,
        dhashes=[0] * sample_count,
    )

    first_window_coverage = next(
        row
        for row in metadata["proposal_decisions"]
        if row["decision"] == "quota_allocated"
        and row["allocation_phase"]
        == "reserved_first_temporal_coverage"
        and row["timestamp"] < 20.0
    )
    assert first_window_coverage["timestamp"] == 15.0


def test_temporal_coverage_keeps_far_side_without_delayed_transition_tail():
    sample_count = 40
    previous_deltas = [0.0] * sample_count
    previous_deltas[3] = 20.0
    metrics = _metric_table(
        [0.1] * sample_count,
        prev_deltas=previous_deltas,
        textline_scores=[0.0] * sample_count,
    )

    _shortlist, *_rest, metadata = build_rescue_shortlist_with_metadata(
        [
            Image.new("RGB", (8, 8), "white")
            for _ in range(sample_count)
        ],
        [float(i) for i in range(sample_count)],
        list(range(sample_count)),
        [_cand(sample_count - 1, float(sample_count - 1))],
        pass1_clusters=3,
        sample_scenes={i: 0 for i in range(sample_count)},
        frame_metrics=metrics,
    )

    first_window_coverage = next(
        row
        for row in metadata["proposal_decisions"]
        if row["decision"] == "quota_allocated"
        and row["allocation_phase"]
        == "reserved_first_temporal_coverage"
        and row["timestamp"] < 20.0
    )
    assert first_window_coverage["timestamp"] == 15.0


def test_scene_coverage_maximizes_distance_from_reserved_window_coverage(
    monkeypatch,
):
    monkeypatch.setattr(
        "keyframe.scoring.rescue_window_seconds",
        lambda _timestamps: 5.0,
    )
    sample_count = 12
    metrics = _metric_table([0.1] * sample_count)
    metrics.sharpness[:] = 10.0

    _shortlist, *_rest, metadata = build_rescue_shortlist_with_metadata(
        [Image.new("RGB", (8, 8), "white") for _ in range(sample_count)],
        [float(i) for i in range(sample_count)],
        list(range(sample_count)),
        [_cand(11, 11.0)],
        pass1_clusters=3,
        sample_scenes={i: 0 for i in range(sample_count)},
        frame_metrics=metrics,
    )

    scene_coverage = next(
        row
        for row in metadata["proposal_decisions"]
        if row["decision"] == "quota_allocated"
        and row["allocation_phase"] == "reserved_scene_coverage"
    )
    assert scene_coverage["timestamp"] == 0.0


def test_single_window_scene_receives_distinct_reserved_scene_coverage():
    sample_count = 8
    metrics = _metric_table([0.1] * sample_count)

    _shortlist, *_rest, metadata = build_rescue_shortlist_with_metadata(
        [Image.new("RGB", (8, 8), "white") for _ in range(sample_count)],
        [float(i) for i in range(sample_count)],
        list(range(sample_count)),
        [_cand(7, 7.0)],
        pass1_clusters=3,
        sample_scenes={i: 0 for i in range(sample_count)},
        frame_metrics=metrics,
    )

    temporal = next(
        row
        for row in metadata["proposal_decisions"]
        if row["decision"] == "quota_allocated"
        and row["allocation_phase"]
        == "reserved_first_temporal_coverage"
    )
    scene = next(
        row
        for row in metadata["proposal_decisions"]
        if row["decision"] == "quota_allocated"
        and row["allocation_phase"] == "reserved_scene_coverage"
    )
    assert scene["sample_idx"] != temporal["sample_idx"]


def test_scene_reservation_prefers_next_distinct_transition_boundary():
    sample_count = 12
    previous_deltas = [0.0] * sample_count
    previous_deltas[3] = 20.0
    previous_deltas[8] = 20.0
    metrics = _metric_table(
        [0.1] * sample_count,
        prev_deltas=previous_deltas,
        textline_scores=[0.0] * sample_count,
    )
    changed_hash = (1 << 16) - 1
    dhashes = [
        *([0] * 3),
        *([changed_hash] * 5),
        *([0] * 4),
    ]

    _shortlist, *_rest, metadata = build_rescue_shortlist_with_metadata(
        [
            Image.new("RGB", (8, 8), "white")
            for _ in range(sample_count)
        ],
        [float(i) for i in range(sample_count)],
        list(range(sample_count)),
        [_cand(0, 0.0)],
        pass1_clusters=3,
        sample_scenes={i: 0 for i in range(sample_count)},
        frame_metrics=metrics,
        dhashes=dhashes,
    )

    temporal = next(
        row
        for row in metadata["proposal_decisions"]
        if row["decision"] == "quota_allocated"
        and row["allocation_phase"]
        == "reserved_first_temporal_coverage"
    )
    scene = next(
        row
        for row in metadata["proposal_decisions"]
        if row["decision"] == "quota_allocated"
        and row["allocation_phase"] == "reserved_scene_coverage"
    )

    assert scene["sample_idx"] != temporal["sample_idx"]
    assert scene["proposal_lane"] == "transition"
    assert scene["sample_idx"] == 7


def test_transition_side_rejects_exact_selected_local_coverage():
    frames = [Image.new("RGB", (8, 8), "white") for _ in range(7)]
    metrics = _metric_table(
        [0.1] * 7,
        prev_deltas=[0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0],
    )

    shortlist, *_rest, metadata = build_rescue_shortlist_with_metadata(
        frames,
        [float(i) for i in range(7)],
        list(range(7)),
        [_cand(2, 2.0)],
        pass1_clusters=3,
        sample_scenes={i: 0 for i in range(7)},
        frame_metrics=metrics,
        dhashes=[0, 0, 0, (1 << 16) - 1, (1 << 16) - 1, (1 << 16) - 1, (1 << 16) - 1],
    )

    assert not any(
        row["proposal_lane"] == "transition"
        and row.get("transition_side") == "pre"
        and row.get("transition_boundary_sample_idx") == 3
        for row in shortlist
    )
    rejection = next(
        row
        for row in metadata["proposal_decisions"]
        if row["decision"] == "transition_side_rejected"
        and row["boundary_sample_idx"] == 3
        and row["transition_side"] == "pre"
    )
    assert rejection["reason"] == "locally_covered"
    assert rejection["coverage_reason"] == "exact_selected_sample"
    assert rejection["covering_sample_idx"] == 2


def test_transition_side_local_coverage_uses_endpoint_content_delta():
    sample_count = 6
    metrics = _metric_table(
        [0.1] * sample_count,
        prev_deltas=[0.0, 2.0, 2.0, 20.0, 0.0, 0.0],
        textline_scores=[0.0] * sample_count,
    )

    shortlist, *_ = build_rescue_shortlist_with_metadata(
        None,
        [float(i) for i in range(sample_count)],
        list(range(sample_count)),
        [_cand(0, 0.0)],
        pass1_clusters=3,
        sample_scenes={i: 0 for i in range(sample_count)},
        frame_metrics=metrics,
        frame_count=sample_count,
        dhashes=[
            0,
            0,
            (1 << 16) - 1,
            (1 << 16) - 1,
            (1 << 16) - 1,
            (1 << 16) - 1,
        ],
    )

    assert {
        row["sample_idx"]
        for row in shortlist
        if row["proposal_lane"] == "transition"
        and row["transition_boundary_sample_idx"] == 3
    } == {2, 3}


def test_transition_side_unavailable_endpoint_delta_is_json_safe():
    sample_count = 6
    metrics = _metric_table(
        [0.1] * sample_count,
        prev_deltas=[0.0, 2.0, 2.0, 20.0, 0.0, 0.0],
        textline_scores=[0.0] * sample_count,
    )
    metrics.content_gray_stack = np.empty((0, 1, 1), dtype=np.float32)

    _shortlist, *_rest, metadata = build_rescue_shortlist_with_metadata(
        None,
        [float(i) for i in range(sample_count)],
        list(range(sample_count)),
        [_cand(0, 0.0)],
        pass1_clusters=3,
        sample_scenes={i: 0 for i in range(sample_count)},
        frame_metrics=metrics,
        frame_count=sample_count,
        dhashes=[0, 0, 0, (1 << 16) - 1, 0, 0],
    )

    rejection = next(
        row
        for row in metadata["proposal_decisions"]
        if row["decision"] == "transition_side_rejected"
        and row["boundary_sample_idx"] == 3
        and row["transition_side"] == "pre"
    )
    assert rejection["coverage_reason"] == "local_selected_dhash"
    assert rejection["content_endpoint_delta"] is None
    json.dumps(metadata, allow_nan=False)


def test_compact_metrics_preserve_endpoint_content_coverage():
    sample_count = 6
    descriptors = np.full(
        (sample_count, 2, 2),
        100.0,
        dtype=np.float32,
    )
    descriptors[2] = 102.0
    rows = [
        {
            "textline_score": 0.0,
            "edge_score": 0.1,
            "entropy": 0.1,
            "dark_ratio": 0.0,
            "bright_ratio": 0.0,
        }
        for _ in range(sample_count)
    ]
    metrics = build_compact_frame_metric_table(
        rows,
        timestamps=[float(i) for i in range(sample_count)],
        frame_indices=list(range(sample_count)),
        content_prev_delta=[0.0, 2.0, 2.0, 20.0, 0.0, 0.0],
        content_next_delta=[2.0, 2.0, 20.0, 0.0, 0.0, 0.0],
        content_endpoint_descriptors=descriptors,
    )

    _shortlist, *_rest, metadata = build_rescue_shortlist_with_metadata(
        None,
        [float(i) for i in range(sample_count)],
        list(range(sample_count)),
        [_cand(0, 0.0)],
        pass1_clusters=3,
        sample_scenes={i: 0 for i in range(sample_count)},
        frame_metrics=metrics,
        frame_count=sample_count,
        dhashes=[
            0,
            0,
            (1 << 16) - 1,
            (1 << 16) - 1,
            (1 << 16) - 1,
            (1 << 16) - 1,
        ],
    )

    rejection = next(
        row
        for row in metadata["proposal_decisions"]
        if row["decision"] == "transition_side_rejected"
        and row["boundary_sample_idx"] == 3
        and row["transition_side"] == "pre"
    )
    assert rejection["coverage_reason"] == (
        "local_selected_content_delta"
    )
    assert rejection["content_endpoint_delta"] == pytest.approx(2.0)


def test_transition_side_applies_relative_sharpness_floor_before_distance():
    frames = [Image.new("RGB", (8, 8), "white") for _ in range(8)]
    metrics = _metric_table(
        [0.1] * 8,
        prev_deltas=[0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0],
    )
    metrics.sharpness[:] = [5.0, 100.0, 20.0, 10.0, 10.0, 10.0, 10.0, 10.0]

    shortlist, *_ = build_rescue_shortlist_with_metadata(
        frames,
        [float(i) for i in range(8)],
        list(range(8)),
        [_cand(7, 7.0)],
        pass1_clusters=3,
        sample_scenes={i: 0 for i in range(8)},
        frame_metrics=metrics,
        dhashes=[0, 0, 0, (1 << 16) - 1, (1 << 16) - 1, (1 << 16) - 1, (1 << 16) - 1, (1 << 16) - 1],
    )

    pre = next(
        row
        for row in shortlist
        if row["proposal_lane"] == "transition"
        and row["transition_side"] == "pre"
        and row["transition_boundary_sample_idx"] == 3
    )
    assert pre["sample_idx"] == 1


def test_transition_side_ties_use_higher_sharpness_then_source_index():
    frames = [Image.new("RGB", (8, 8), "white") for _ in range(8)]
    timestamps = [0.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    metrics = _metric_table(
        [0.1] * 8,
        timestamps=timestamps,
        prev_deltas=[0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0],
    )
    metrics.sharpness[:] = [10.0, 20.0, 30.0, 10.0, 10.0, 10.0, 10.0, 10.0]

    higher_sharpness, *_ = build_rescue_shortlist_with_metadata(
        frames,
        timestamps,
        [0, 10, 20, 30, 40, 50, 60, 70],
        [_cand(7, 7.0)],
        pass1_clusters=3,
        sample_scenes={i: 0 for i in range(8)},
        frame_metrics=metrics,
    )
    pre = next(
        row
        for row in higher_sharpness
        if row["proposal_lane"] == "transition"
        and row["transition_side"] == "pre"
        and row["transition_boundary_sample_idx"] == 3
    )
    assert pre["sample_idx"] == 2

    metrics.sharpness[1] = metrics.sharpness[2]
    source_tie, *_ = build_rescue_shortlist_with_metadata(
        frames,
        timestamps,
        [0, 20, 10, 30, 40, 50, 60, 70],
        [_cand(7, 7.0)],
        pass1_clusters=3,
        sample_scenes={i: 0 for i in range(8)},
        frame_metrics=metrics,
    )
    pre = next(
        row
        for row in source_tie
        if row["proposal_lane"] == "transition"
        and row["transition_side"] == "pre"
        and row["transition_boundary_sample_idx"] == 3
    )
    assert pre["sample_idx"] == 2


def test_reserved_capacity_round_robin_is_temporal_and_traces_cap_exhaustion(
    monkeypatch,
):
    monkeypatch.setattr(
        "keyframe.scoring.rescue_window_seconds",
        lambda _timestamps: 1.0,
    )
    sample_count = 120
    frames = [
        Image.new("RGB", (8, 8), "white")
        for _ in range(sample_count)
    ]
    metrics = _metric_table([0.1] * sample_count)

    shortlist, _rows, _budget, ocr_cap, window_count, scene_count, _dropped, metadata = (
        build_rescue_shortlist_with_metadata(
            frames,
            [float(i) for i in range(sample_count)],
            list(range(sample_count)),
            [_cand(sample_count - 1, float(sample_count - 1))],
            pass1_clusters=3,
            sample_scenes={i: 0 for i in range(sample_count)},
            frame_metrics=metrics,
        )
    )

    assert ocr_cap == 96
    assert window_count == sample_count - 1
    assert scene_count == 1
    assert metadata["reserved_proposal_capacity"] == 96
    assert len(shortlist) == ocr_cap
    allocations = [
        row
        for row in metadata["proposal_decisions"]
        if row["decision"] == "quota_allocated"
    ]
    assert [row["timestamp"] for row in allocations] == [
        float(i) for i in range(ocr_cap)
    ]
    assert all(
        row["allocation_phase"] == "reserved_first_temporal_coverage"
        for row in allocations
    )
    assert any(
        row["decision"] == "quota_rejected"
        and row["reason"] == "rescue_ocr_cap_exhausted"
        for row in metadata["proposal_decisions"]
    )


def test_reserved_scene_coverage_precedes_multi_window_and_legacy_backfill():
    frames = [Image.new("RGB", (8, 8), "white") for _ in range(45)]
    metrics = _metric_table([0.1] * 45)

    _shortlist, *_rest, metadata = build_rescue_shortlist_with_metadata(
        frames,
        [float(i) for i in range(45)],
        list(range(45)),
        [_cand(44, 44.0)],
        pass1_clusters=3,
        sample_scenes={i: 0 for i in range(45)},
        frame_metrics=metrics,
    )

    phases = [
        row["allocation_phase"]
        for row in metadata["proposal_decisions"]
        if row["decision"] == "quota_allocated"
    ]
    assert phases.count("reserved_first_temporal_coverage") == 3
    assert "reserved_scene_coverage" in phases
    scene_position = phases.index("reserved_scene_coverage")
    later_positions = [
        phases.index(phase)
        for phase in ("multi_window_backfill", "legacy_proxy_backfill")
        if phase in phases
    ]
    assert later_positions
    assert scene_position < min(later_positions)


def test_rescue_shortlist_order_is_deterministic_when_scores_tie():
    frames = [Image.new("RGB", (8, 8), "white") for _ in range(45)]
    frame_metrics = _metric_table([0.5] * 45)
    kwargs = {
        "sample_scenes": {i: 0 for i in range(45)},
        "frame_metrics": frame_metrics,
    }

    first, *_ = build_rescue_shortlist_with_metadata(
        frames,
        [float(i) for i in range(45)],
        list(range(45)),
        [{"sample_idx": 0, "timestamp": 0.0, "scene_id": 0}],
        pass1_clusters=3,
        **kwargs,
    )
    second, *_ = build_rescue_shortlist_with_metadata(
        frames,
        [float(i) for i in range(45)],
        list(range(45)),
        [{"sample_idx": 0, "timestamp": 0.0, "scene_id": 0}],
        pass1_clusters=3,
        **kwargs,
    )

    assert [row["sample_idx"] for row in first] == [row["sample_idx"] for row in second]


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


def test_structured_comparator_prefers_scene_then_dwell_window_distance_and_source():
    rescue = {
        **_cand(50, 10.0, scene=4, window=3),
        "frame_idx": 500,
        "dwell_id": 8,
    }
    candidates = [
        {
            **_cand(1, 9.9, scene=4, window=3),
            "frame_idx": 1,
            "dwell_id": 7,
        },
        {
            **_cand(2, 100.0, scene=4, window=2),
            "frame_idx": 2,
            "dwell_id": 8,
        },
        {
            **_cand(3, 12.0, scene=4, window=3),
            "frame_idx": 30,
            "dwell_id": 8,
        },
        {
            **_cand(4, 8.0, scene=4, window=3),
            "frame_idx": 20,
            "dwell_id": 8,
        },
        {
            **_cand(5, 10.0, scene=9, window=3),
            "frame_idx": 0,
            "dwell_id": 8,
        },
    ]

    comparator = select_structured_comparator(rescue, candidates)

    assert comparator is not None
    assert comparator.sample_idx == 4


def test_comparison_ocr_selection_uses_exact_structured_comparator_only():
    candidates = [
        {
            **_cand(1, 9.0, scene=0, cluster=1, window=0),
            "dwell_id": 1,
        },
        {
            **_cand(2, 8.0, scene=0, cluster=2, window=1),
            "dwell_id": 2,
        },
        {
            **_cand(3, 20.0, scene=1, cluster=3, window=0),
            "dwell_id": 3,
        },
    ]
    shortlist = [
        {
            **_cand(10, 10.0, scene=0, cluster=2, window=0),
            "dwell_id": 1,
        },
        {
            **_cand(21, 21.0, scene=1, cluster=4, window=0),
            "dwell_id": 3,
        },
    ]

    assert _comparison_primary_sample_idxs(candidates, shortlist) == {1, 3}


def test_structured_delta_metadata_records_categories_and_comparator_identity():
    candidates = candidate_records(
        [
            {
                **_cand(1, 9.0, scene=0, window=0),
                "dwell_id": 1,
                "field_signature": field_section_signatures(
                    "Status: Draft"
                ),
            }
        ]
    )
    shortlist = candidate_records(
        [
            {
                **_cand(2, 10.0, scene=0, window=0),
                "dwell_id": 1,
                "field_signature": field_section_signatures(
                    "Status: Approved"
                ),
            }
        ]
    )

    updated = attach_structured_delta_metadata(shortlist, candidates)

    assert updated[0].selection.structured_delta_categories == (
        "status",
        "same_label_value",
    )
    assert updated[0].selection.structured_comparator_sample_idx == 1
    assert updated[0].selection.structured_comparator_timestamp == 9.0
    assert updated[0].selection.structured_changed_signature_count > 0


def test_structured_blank_populated_delta_outranks_other_structured_and_proxy():
    candidates = [
        {
            **_cand(0, 0.0, scene=0),
            "field_signature": field_section_signatures("Field:"),
        },
        {
            **_cand(10, 10.0, scene=1),
            "field_signature": field_section_signatures(
                "Status: Draft"
            ),
        },
    ]
    shortlist = [
        {
            **_cand(1, 1.0, scene=0, cluster=2, proxy=0.01),
            "field_signature": field_section_signatures(
                "Field: populated"
            ),
        },
        {
            **_cand(11, 11.0, scene=1, cluster=3, proxy=0.99),
            "field_signature": field_section_signatures(
                "Status: Approved"
            ),
        },
    ]

    promoted = promote_rescue_candidates(
        candidates,
        shortlist,
        [0] * 12,
        rescue_budget=1,
    )

    assert [row["sample_idx"] for row in promoted] == [0, 1, 10]
    rescued = next(row for row in promoted if row.get("rescue_origin"))
    assert rescued["rescue_reason"] == "structured_delta"
    assert rescued["structured_delta_categories"] == [
        "blank_populated"
    ]


def test_structured_promotion_recompares_after_each_rescue_slot():
    candidates = [
        {
            **_cand(0, 0.0, scene=0, window=0),
            "dwell_id": 1,
            "field_signature": field_section_signatures(
                "Status: Draft"
            ),
        }
    ]
    shortlist = [
        {
            **_cand(1, 1.0, scene=0, cluster=2, window=0, proxy=0.9),
            "dwell_id": 1,
            "field_signature": field_section_signatures(
                "Status: Approved"
            ),
        },
        {
            **_cand(2, 2.0, scene=0, cluster=3, window=0, proxy=0.8),
            "dwell_id": 1,
            "field_signature": field_section_signatures(
                "Status: Approved"
            ),
        },
    ]

    promoted = promote_rescue_candidates(
        candidates,
        shortlist,
        [1, 1, 1],
        rescue_budget=2,
    )

    assert [row["sample_idx"] for row in promoted] == [0, 1]
    rescued = next(row for row in promoted if row.get("rescue_origin"))
    assert rescued["structured_delta_categories"] == [
        "status",
        "same_label_value",
    ]


def test_structured_promotion_gives_distinct_dwell_an_opportunity_before_repeat():
    candidates = [
        {
            **_cand(0, 0.0, scene=0, window=0),
            "dwell_id": 1,
            "field_signature": field_section_signatures("Field:"),
        },
        {
            **_cand(10, 10.0, scene=1, window=1),
            "dwell_id": 2,
            "field_signature": field_section_signatures("Page: 12"),
        },
    ]
    same_dwell = [
        {
            **_cand(1, 1.0, scene=0, cluster=2, window=0, proxy=0.9),
            "dwell_id": 1,
            "field_signature": field_section_signatures(
                "Field: populated"
            ),
        },
        {
            **_cand(2, 2.0, scene=0, cluster=3, window=0, proxy=0.8),
            "dwell_id": 1,
            "field_signature": field_section_signatures(
                "Field: another value"
            ),
        },
    ]
    other_dwell = {
        **_cand(11, 11.0, scene=1, cluster=4, window=1, proxy=0.1),
        "dwell_id": 2,
        "field_signature": field_section_signatures("Page: 34"),
    }

    promoted = promote_rescue_candidates(
        candidates,
        [*same_dwell, other_dwell],
        [1] * 12,
        rescue_budget=2,
    )
    reverse_promoted = promote_rescue_candidates(
        candidates,
        [other_dwell, *reversed(same_dwell)],
        [1] * 12,
        rescue_budget=2,
    )

    assert [row["sample_idx"] for row in promoted] == [0, 1, 10, 11]
    assert [row["sample_idx"] for row in reverse_promoted] == [
        0,
        1,
        10,
        11,
    ]


def test_structured_promotion_distinguishes_windows_within_a_long_dwell():
    candidates = [
        {
            **_cand(0, 0.0, scene=0, window=0),
            "dwell_id": 1,
            "field_signature": field_section_signatures("Field:"),
        }
    ]
    first_window = [
        {
            **_cand(1, 1.0, scene=0, cluster=2, window=0, proxy=0.9),
            "dwell_id": 1,
            "field_signature": field_section_signatures(
                "Field: populated"
            ),
        },
        {
            **_cand(2, 2.0, scene=0, cluster=3, window=0, proxy=0.99),
            "dwell_id": 1,
            "field_signature": field_section_signatures(
                "Field: another value"
            ),
        },
    ]
    later_window = {
        **_cand(10, 10.0, scene=0, cluster=4, window=1, proxy=0.01),
        "dwell_id": 1,
        "field_signature": field_section_signatures(
            "Field: reviewed"
        ),
    }

    promoted = promote_rescue_candidates(
        candidates,
        [*first_window, later_window],
        [1] * 11,
        rescue_budget=2,
    )
    reverse_promoted = promote_rescue_candidates(
        candidates,
        [later_window, *reversed(first_window)],
        [1] * 11,
        rescue_budget=2,
    )

    promoted_idxs = {
        row["sample_idx"]
        for row in promoted
        if row.get("rescue_origin")
    }
    reverse_promoted_idxs = {
        row["sample_idx"]
        for row in reverse_promoted
        if row.get("rescue_origin")
    }

    assert promoted_idxs == reverse_promoted_idxs
    assert 10 in promoted_idxs
    assert len(promoted_idxs & {1, 2}) == 1


def test_promotion_round_robin_resumes_for_repeated_structured_label():
    candidates = [
        {
            **_cand(0, 0.0, scene=0, window=0),
            "dwell_id": 1,
            "field_signature": field_section_signatures("Field:"),
        }
    ]
    shortlist = [
        {
            **_cand(1, 1.0, scene=0, cluster=2, window=0, proxy=0.9),
            "dwell_id": 1,
            "field_signature": field_section_signatures(
                "Field: populated"
            ),
        },
        {
            **_cand(2, 2.0, scene=0, cluster=3, window=0, proxy=0.8),
            "dwell_id": 1,
            "field_signature": field_section_signatures(
                "Field: another value"
            ),
        },
        {
            **_cand(10, 10.0, scene=1, cluster=4, proxy=0.99),
            "proposal_lane": "transition",
        },
        {
            **_cand(
                20,
                20.0,
                scene=2,
                cluster=5,
                tokens=["page1", "approved", "cover", "pdf"],
                proxy=1.0,
            ),
            "proposal_lane": "legacy_proxy",
        },
    ]

    promoted = promote_rescue_candidates(
        candidates,
        shortlist,
        [1] * 21,
        rescue_budget=4,
    )

    rescued = sorted(
        (
            row["rescue_priority"],
            row["sample_idx"],
            row["rescue_reason"],
        )
        for row in promoted
        if row.get("rescue_origin")
    )
    assert rescued == [
        (1, 1, "structured_delta"),
        (2, 10, "transition"),
        (3, 20, "evidence_marker"),
        (4, 2, "structured_delta"),
    ]


def test_independent_changed_form_labels_exhaust_small_budget_first():
    candidates = [
        {
            **_cand(0, 0.0, scene=0),
            "dwell_id": 0,
            "field_signature": field_section_signatures(
                "Control ID:\nStatus:"
            ),
        },
    ]
    shortlist = [
        {
            **_cand(1, 1.0, scene=0, cluster=2, proxy=0.9),
            "dwell_id": 0,
            "field_signature": field_section_signatures(
                "Control ID: 12345\nStatus:"
            ),
        },
        {
            **_cand(2, 2.0, scene=0, cluster=3, proxy=0.8),
            "dwell_id": 0,
            "field_signature": field_section_signatures(
                "Control ID:\nStatus: Approved"
            ),
        },
        {
            **_cand(10, 10.0, scene=1, cluster=4, proxy=0.01),
            "dwell_id": 1,
            "proposal_lane": "transition",
        },
    ]

    promoted = promote_rescue_candidates(
        candidates,
        shortlist,
        list(range(11)),
        rescue_budget=2,
    )

    assert {
        row["sample_idx"]
        for row in promoted
        if row.get("rescue_origin")
    } == {1, 2}


def test_noisy_prose_value_change_does_not_outrank_stable_field_value():
    candidates = [
        {
            **_cand(0, 0.0, scene=0),
            "dwell_id": 0,
            "field_signature": field_section_signatures(
                "Owner: Naygen"
            ),
        },
        {
            **_cand(10, 10.0, scene=1),
            "dwell_id": 1,
            "field_signature": field_section_signatures(
                "Control ID: Unlink"
            ),
        },
    ]
    shortlist = [
        {
            **_cand(1, 1.0, scene=0, cluster=2, proxy=1.0),
            "dwell_id": 0,
            "field_signature": field_section_signatures(
                "Owner: Naveen"
            ),
        },
        {
            **_cand(11, 11.0, scene=1, cluster=3, proxy=0.1),
            "dwell_id": 1,
            "field_signature": field_section_signatures(
                "Control ID: 67890"
            ),
        },
    ]

    promoted = promote_rescue_candidates(
        candidates,
        shortlist,
        list(range(12)),
        rescue_budget=1,
    )

    assert {
        row["sample_idx"]
        for row in promoted
        if row.get("rescue_origin")
    } == {11}


def test_structured_lane_wins_first_round_over_transition_and_ordinary():
    candidates = [
        {
            **_cand(0, 0.0, scene=0),
            "field_signature": field_section_signatures(
                "Status: Draft"
            ),
        }
    ]
    shortlist = [
        {
            **_cand(1, 1.0, scene=0, cluster=2, proxy=0.01),
            "field_signature": field_section_signatures(
                "Status: Approved"
            ),
        },
        {
            **_cand(10, 10.0, scene=1, cluster=3, proxy=1.0),
            "proposal_lane": "transition",
        },
        {
            **_cand(
                20,
                20.0,
                scene=2,
                cluster=4,
                tokens=["page1", "approved", "cover", "pdf"],
                proxy=1.0,
            ),
            "proposal_lane": "legacy_proxy",
        },
    ]

    promoted = promote_rescue_candidates(
        candidates,
        shortlist,
        [0] * 21,
        rescue_budget=1,
    )

    rescued = next(
        row for row in promoted if row.get("rescue_origin")
    )
    assert rescued["sample_idx"] == 1
    assert rescued["rescue_reason"] == "structured_delta"


def test_structured_page_lane_prefers_settled_coverage_representative():
    candidates = [
        {
            **_cand(0, 0.0, scene=0),
            "field_signature": field_section_signatures("Page 2"),
        }
    ]
    shortlist = [
        {
            **_cand(1, 1.0, scene=0, cluster=2, proxy=0.5),
            "field_signature": field_section_signatures("Page 4"),
            "proposal_lane": "transition",
            "content_area_delta_score": 0.4,
        },
        {
            **_cand(10, 10.0, scene=0, cluster=3, proxy=0.4),
            "field_signature": field_section_signatures("Page 6"),
            "proposal_lane": "temporal_coverage",
            "content_area_delta_score": 0.001,
        },
    ]

    promoted = promote_rescue_candidates(
        candidates,
        shortlist,
        [0] * 11,
        rescue_budget=1,
    )

    rescued = next(
        row for row in promoted if row.get("rescue_origin")
    )
    assert rescued["sample_idx"] == 10
    assert rescued["rescue_reason"] == "structured_delta"


def test_structured_page_lane_prefers_multi_page_boundary_evidence():
    candidates = [
        {
            **_cand(0, 0.0, scene=0),
            "field_signature": ["page:2", "page:3"],
        }
    ]
    shortlist = [
        {
            **_cand(10, 10.0, scene=0, cluster=2, proxy=0.9),
            "field_signature": ["page:1"],
            "proposal_lane": "temporal_coverage",
            "content_area_delta_score": 0.001,
        },
        {
            **_cand(20, 20.0, scene=0, cluster=3, proxy=0.5),
            "field_signature": ["page:1", "page:2"],
            "proposal_lane": "temporal_coverage",
            "content_area_delta_score": 0.001,
        },
    ]

    promoted = promote_rescue_candidates(
        candidates,
        shortlist,
        [0] * 21,
        rescue_budget=1,
    )

    rescued = next(
        row for row in promoted if row.get("rescue_origin")
    )
    assert rescued["sample_idx"] == 20
    assert rescued["rescue_reason"] == "structured_delta"


def test_transition_lane_prefers_settled_qualified_side():
    candidates = [_cand(0, 0.0, scene=0)]
    shortlist = [
        {
            **_cand(1, 1.0, scene=1, cluster=2, proxy=1.0),
            "proposal_lane": "transition",
            "content_area_delta_score": 0.4,
        },
        {
            **_cand(10, 10.0, scene=2, cluster=3, proxy=0.5),
            "proposal_lane": "transition",
            "content_area_delta_score": 0.05,
        },
    ]

    promoted = promote_rescue_candidates(
        candidates,
        shortlist,
        [0] * 11,
        rescue_budget=1,
    )

    rescued = next(
        row for row in promoted if row.get("rescue_origin")
    )
    assert rescued["sample_idx"] == 10
    assert rescued["rescue_reason"] == "transition"


def test_transition_proposal_outranks_ordinary_evidence_after_structured_tier():
    candidates = [_cand(0, 0.0, scene=0, tokens=["intro"], proxy=0.1)]
    shortlist = [
        {
            **_cand(
                10,
                10.0,
                scene=1,
                cluster=2,
                tokens=["page1", "approved", "cover", "pdf"],
                proxy=0.99,
            ),
            "proposal_lane": "legacy_proxy",
        },
        {
            **_cand(
                20,
                20.0,
                scene=2,
                cluster=3,
                tokens=(),
                proxy=0.01,
            ),
            "proposal_lane": "transition",
            "content_area_delta_score": 0.1,
        },
    ]

    promoted = promote_rescue_candidates(
        candidates,
        shortlist,
        [0] * 21,
        rescue_budget=1,
    )

    assert [row["sample_idx"] for row in promoted] == [0, 20]
    assert promoted[1]["rescue_reason"] == "transition"


def test_ordinary_lane_prefers_settled_multi_page_boundary_evidence():
    candidates = [
        _cand(
            0,
            0.0,
            scene=0,
            tokens=["intro"],
            proxy=0.1,
        )
    ]
    shortlist = [
        {
            **_cand(
                10,
                10.0,
                scene=1,
                cluster=2,
                tokens=[
                    "page1",
                    "status",
                    "draft",
                    "approved",
                    "workflow",
                ],
                proxy=0.99,
            ),
            "field_signature": ["page:1"],
            "proposal_lane": "temporal_coverage",
            "content_area_delta_score": 0.001,
        },
        {
            **_cand(
                20,
                20.0,
                scene=2,
                cluster=3,
                tokens=[
                    "page1",
                    "page2",
                    "signed",
                    "date",
                ],
                proxy=0.5,
            ),
            "field_signature": ["page:1", "page:2"],
            "proposal_lane": "temporal_coverage",
            "content_area_delta_score": 0.001,
        },
    ]

    promoted = promote_rescue_candidates(
        candidates,
        shortlist,
        [0] * 21,
        rescue_budget=1,
    )

    rescued = next(
        row for row in promoted if row.get("rescue_origin")
    )
    assert rescued["sample_idx"] == 20
    assert rescued["rescue_reason"] == "evidence_marker"


def test_ordinary_lane_prefers_material_form_state_before_page_boundary():
    candidates = [
        _cand(
            0,
            0.0,
            scene=0,
            tokens=["intro"],
            proxy=0.1,
        )
    ]
    form_state = {
        **_cand(
            10,
            10.0,
            scene=1,
            cluster=2,
            tokens=["status", "draft", "approved", "workflow"],
            proxy=0.5,
        ),
        "field_signature": ["page:1"],
        "proposal_lane": "temporal_coverage",
        "content_area_delta_score": 0.001,
    }
    page_boundary = {
        **_cand(
            20,
            20.0,
            scene=2,
            cluster=3,
            tokens=["page1", "page2", "cover", "pdf"],
            proxy=0.9,
        ),
        "field_signature": ["page:1", "page:2"],
        "proposal_lane": "temporal_coverage",
        "content_area_delta_score": 0.001,
    }

    promoted = promote_rescue_candidates(
        candidates,
        [page_boundary, form_state],
        [0] * 21,
        rescue_budget=1,
    )

    rescued = next(
        row for row in promoted if row.get("rescue_origin")
    )
    assert rescued["sample_idx"] == 10
    assert rescued["rescue_reason"] == "evidence_marker"


def test_ordinary_lane_prefers_larger_uncovered_gap_when_evidence_ties():
    candidates = [
        _cand(
            20,
            20.0,
            scene=1,
            tokens=["intro"],
            proxy=0.1,
        )
    ]
    far_coverage = {
        **_cand(
            0,
            0.0,
            scene=1,
            cluster=2,
            tokens=[f"dense{i}" for i in range(20)],
            proxy=0.1,
        ),
        "field_signature": ["page:1", "page:2"],
        "proposal_lane": "temporal_coverage",
        "content_area_delta_score": 0.001,
    }
    close_coverage = {
        **_cand(
            15,
            15.0,
            scene=1,
            cluster=3,
            tokens=[f"other{i}" for i in range(20)],
            proxy=0.1,
        ),
        "field_signature": ["page:1", "page:2"],
        "proposal_lane": "temporal_coverage",
        "content_area_delta_score": 0.001,
    }

    promoted = promote_rescue_candidates(
        candidates,
        [close_coverage, far_coverage],
        [0] * 21,
        rescue_budget=1,
    )

    rescued = next(
        row for row in promoted if row.get("rescue_origin")
    )
    assert rescued["sample_idx"] == 0
    assert rescued["rescue_reason"] == "temporal_coverage"


def test_ordinary_lane_prefers_new_scene_before_repeating_page_boundary_scene():
    candidates = [
        _cand(
            0,
            0.0,
            scene=0,
            tokens=["intro"],
            proxy=0.1,
        )
    ]
    multi_page = {
        **_cand(
            10,
            10.0,
            scene=1,
            cluster=2,
            tokens=["page1", "page2", "signed", "date"],
            proxy=0.9,
        ),
        "field_signature": ["page:1", "page:2"],
        "proposal_lane": "temporal_coverage",
        "content_area_delta_score": 0.001,
    }
    repeated_scene = {
        **_cand(
            20,
            20.0,
            scene=1,
            cluster=3,
            tokens=[
                "page1",
                "page2",
                "signed",
                "date",
                "details",
            ],
            proxy=0.8,
        ),
        "field_signature": ["page:1", "page:2"],
        "proposal_lane": "temporal_coverage",
        "content_area_delta_score": 0.001,
    }
    new_scene = {
        **_cand(
            30,
            30.0,
            scene=2,
            cluster=4,
            tokens=["page1", "status", "draft", "priority"],
            proxy=0.5,
        ),
        "field_signature": ["page:1"],
        "proposal_lane": "temporal_coverage",
        "content_area_delta_score": 0.001,
    }

    promoted = promote_rescue_candidates(
        candidates,
        [multi_page, repeated_scene, new_scene],
        [0] * 31,
        rescue_budget=2,
    )

    promoted_idxs = {
        row["sample_idx"]
        for row in promoted
        if row.get("rescue_origin")
    }
    assert 30 in promoted_idxs
    assert len(promoted_idxs & {10, 20}) == 1


def test_additive_evidence_promotes_before_non_subsuming_swap():
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

    assert [row["sample_idx"] for row in promoted] == [0, 2, 3]
    assert promoted[2]["rescue_origin"] == "additive_rescue"
    assert promoted[2]["rescue_priority"] == 1


def test_non_subsuming_same_cluster_candidate_does_not_swap_or_add():
    candidates = [
        {
            "sample_idx": 0,
            "timestamp": 0.0,
            "clip_cluster": 3,
            "scene_id": 0,
            "cluster_role": "primary",
            "proxy_content_score": 0.1,
            "ocr_tokens": ["page1", "status"],
            "rescue_tokens": ["page1", "status"],
        },
    ]
    non_subsuming = [
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
    ]

    not_swapped = promote_rescue_candidates(candidates, non_subsuming, [0, 1, 2], rescue_budget=1)

    assert [row["sample_idx"] for row in not_swapped] == [0]


def test_same_window_marker_equivalent_does_not_reject_outside_tolerance():
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

    assert [row["sample_idx"] for row in promoted] == [0, 1]
    assert promoted[1]["rescue_reason"] == "temporal_coverage"


def test_same_dwell_marker_equivalent_does_not_reject_outside_tolerance():
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
            "temporal_window_id": 1,
        }
    ]

    promoted = promote_rescue_candidates(candidates, shortlist, [0, 0], rescue_budget=3)

    assert [row["sample_idx"] for row in promoted] == [0, 1]
    assert promoted[1]["rescue_reason"] == "temporal_coverage"


def test_form_state_token_gain_overrides_marker_equivalence():
    candidates = [
        _cand(130, 65.0, tokens=["page1", "completed", "status"], proxy=0.1),
    ]
    shortlist = [
        _cand(
            142,
            66.0,
            tokens=["page1", "completed", "status", "date", "please", "selection", "required"],
            proxy=0.9,
        ),
    ]

    promoted = promote_rescue_candidates(candidates, shortlist, [0] * 143, rescue_budget=3)

    assert [row["sample_idx"] for row in promoted] == [130, 142]
    assert promoted[1]["rescue_origin"] == "additive_rescue"


def test_status_date_error_and_filled_date_are_distinct_states():
    candidates = [
        _cand(
            142,
            70.0,
            tokens=["page1", "completed", "status", "date", "please", "selection"],
            proxy=0.4,
        ),
    ]
    shortlist = [
        _cand(
            164,
            71.0,
            tokens=["page1", "completed", "status", "date", "24apr2026"],
            proxy=0.9,
        ),
    ]

    promoted = promote_rescue_candidates(candidates, shortlist, [0] * 165, rescue_budget=3)

    assert [row["sample_idx"] for row in promoted] == [142, 164]


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
            "timestamp": 1.0,
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


def test_eligible_evidence_candidates_are_ranked_before_generic_candidates():
    candidates = [_cand(0, 0.0, scene=0, cluster=1, tokens=["intro"], proxy=0.1)]
    shortlist = [
        _cand(10, 10.0, scene=1, cluster=2, tokens=[f"generic{i}" for i in range(20)], proxy=0.99),
        _cand(20, 20.0, scene=2, cluster=3, tokens=["page1", "signed", "approved", "date", "cover", "pdf"], proxy=0.5),
    ]

    promoted = promote_rescue_candidates(candidates, shortlist, [0] * 21, rescue_budget=1)

    assert [row["sample_idx"] for row in promoted] == [0, 20]
    assert promoted[1]["rescue_reason"] == "evidence_marker"


def test_preflight_uses_same_additive_priority_as_promotion():
    base = [_cand(0, 0.0, scene=0, cluster=1, tokens=["intro"], proxy=0.1)]
    shortlist = [
        _cand(10, 10.0, scene=1, cluster=2, tokens=[f"generic{i}" for i in range(20)], proxy=0.99),
        _cand(20, 20.0, scene=2, cluster=3, tokens=["page1", "signed", "approved", "date", "cover", "pdf"], proxy=0.5),
    ]

    report = _preflight(base, shortlist, base, [0] * 21, 1)
    rows = {row["sample_idx"]: row for row in report["candidate_rows"]}

    assert rows[20]["phase_a_rank"] == 1
    assert rows[20]["outcome"] == "eligible_above_headroom"
    assert rows[10]["phase_a_rank"] == 2
    assert rows[10]["outcome"] == "eligible_below_headroom"
    assert rows[20]["reason_priority"] > rows[10]["reason_priority"]


def test_preflight_replays_dynamic_promotion_order():
    def with_fields(candidate, text):
        return {
            **candidate,
            "field_signature": list(
                field_section_signatures(text)
            ),
        }

    base = [
        with_fields(
            _cand(0, 0.0, scene=0, tokens=["status", "draft"]),
            "Status: Draft",
        )
    ]
    shortlist = [
        with_fields(
            _cand(
                1,
                1.0,
                scene=0,
                tokens=["status", "approved"],
                proxy=0.9,
            ),
            "Status: Approved",
        ),
        with_fields(
            _cand(
                2,
                2.0,
                scene=0,
                tokens=["status", "rejected"],
                proxy=0.8,
            ),
            "Status: Rejected",
        ),
        {
            **_cand(10, 10.0, scene=1, tokens=["transition"]),
            "proposal_lane": "transition",
        },
        _cand(
            20,
            20.0,
            scene=2,
            tokens=[f"dense{i}" for i in range(20)],
        ),
    ]
    dwell_ids = [0] * 21

    actual = _promote_rescue_candidates(
        base,
        shortlist,
        dwell_ids,
        rescue_budget=4,
    )
    actual_order = [
        record.sample_idx
        for record in sorted(
            (
                record
                for record in actual
                if record.selection.rescue_origin
            ),
            key=lambda record: int(
                record.selection.rescue_priority or 0
            ),
        )
    ]
    report = _preflight(
        base,
        shortlist,
        base,
        dwell_ids,
        4,
    )
    predicted_order = [
        row["sample_idx"]
        for row in report["predicted_ordered_eligible"]
    ]

    assert actual_order == [1, 10, 20, 2]
    assert predicted_order == actual_order


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


def test_ocr_candidates_preserves_line_boundaries_for_structured_fields(
    monkeypatch,
):
    monkeypatch.setattr("keyframe.frames._is_macos", lambda: True)
    monkeypatch.setattr(
        "keyframe.frames._ocr_apple_vision",
        lambda _img: ["Control ID:", "12345"],
    )
    candidates = [{"sample_idx": 0, "timestamp": 1.0}]

    texts, updated = ocr_candidates(
        candidates,
        [Image.new("RGB", (8, 8), "white")],
    )

    assert texts == ["Control ID:\n12345"]
    assert updated[0]["ocr_text"] == "Control ID:\n12345"
