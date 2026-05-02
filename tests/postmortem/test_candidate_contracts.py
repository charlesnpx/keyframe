from collections.abc import Mapping

import pytest
from PIL import Image

from keyframe.pipeline.contracts import CandidateBatch, CandidateRecord, candidate_to_trace_row
from keyframe.pipeline.snapshotters import snapshot_candidate_batch


def test_candidate_updates_return_new_record_and_preserve_original():
    candidate = CandidateRecord(sample_idx=3, frame_idx=300, timestamp=12.5, origin="proposal")

    updated = candidate.with_evidence(ocr_text="hello", ocr_tokens=["hello"]).with_selection(retention_reason="none")

    assert updated is not candidate
    assert candidate.evidence.ocr_text is None
    assert updated.evidence.ocr_text == "hello"
    assert updated.evidence.ocr_tokens == ("hello",)
    assert not isinstance(candidate, Mapping)
    with pytest.raises(TypeError):
        candidate["sample_idx"]  # type: ignore[index]
    assert not hasattr(candidate, "get")


def test_candidate_rejects_heavy_artifact_metadata():
    with pytest.raises(ValueError):
        CandidateRecord(sample_idx=1, frame_idx=10, timestamp=1.0).with_visual(image=object())


def test_candidate_rejects_unknown_update_fields():
    candidate = CandidateRecord(sample_idx=1, frame_idx=10, timestamp=1.0)

    with pytest.raises(TypeError):
        candidate.with_selection(nope=True)


def test_candidate_lineage_defaults_to_self_identity():
    candidate = CandidateRecord(sample_idx=4, frame_idx=40, timestamp=8.0)

    assert candidate.lineage.merged_from_sample_idxs == (4,)
    assert candidate.lineage.merged_timestamps == (8.0,)


def test_candidate_snapshot_is_materialized_json_safe_copy():
    candidate = CandidateRecord(sample_idx=2, frame_idx=20, timestamp=4.0).with_evidence(
        ocr_tokens=["alpha"],
    )
    batch = CandidateBatch(stage="test", candidates=(candidate,))

    payload = snapshot_candidate_batch("test", batch)
    updated = candidate.with_evidence(ocr_tokens=["beta"])

    assert updated.evidence.ocr_tokens == ("beta",)
    assert payload["candidates"][0]["ocr_token_count"] == 1
    assert payload["candidates"][0]["sample_idx"] == 2


def test_candidate_projection_is_json_safe():
    candidate = CandidateRecord(sample_idx=2, frame_idx=20, timestamp=4.0).with_evidence(
        ocr_tokens=["alpha"],
    )

    assert candidate_to_trace_row(candidate)["ocr_tokens"] == ["alpha"]


def test_algorithm_outputs_are_candidate_record_tuples():
    from keyframe.dedupe import near_time_dedupe
    from keyframe.merge import union_find_merge
    from keyframe.scoring import promote_rescue_candidates

    candidates = (
        CandidateRecord(sample_idx=0, frame_idx=0, timestamp=0.0).with_evidence(ocr_tokens=("same",)),
        CandidateRecord(sample_idx=1, frame_idx=1, timestamp=1.0).with_evidence(ocr_tokens=("same",)),
    )

    deduped = near_time_dedupe(candidates, [{"same"}, {"same"}])
    merged = union_find_merge(deduped, [set(c.evidence.ocr_tokens) for c in deduped], [False], [Image.new("RGB", (8, 8))])
    promoted = promote_rescue_candidates(deduped, (), [0, 1], rescue_budget=0)

    assert isinstance(deduped, tuple)
    assert isinstance(merged, tuple)
    assert isinstance(promoted, tuple)
    assert all(isinstance(candidate, CandidateRecord) for candidate in deduped + merged + promoted)
