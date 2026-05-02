import pytest

from keyframe.pipeline.contracts import CandidateBatch, CandidateRecord
from keyframe.pipeline.snapshotters import snapshot_candidate_batch


def test_candidate_updates_return_new_record_and_preserve_original():
    candidate = CandidateRecord(sample_idx=3, frame_idx=300, timestamp=12.5, origin="proposal")

    updated = candidate.with_updates(ocr_text="hello", ocr_tokens=["hello"], retention_reason="none")

    assert updated is not candidate
    assert candidate.get("ocr_text") is None
    assert updated["ocr_text"] == "hello"
    assert updated["ocr_tokens"] == ["hello"]


def test_candidate_rejects_heavy_artifact_metadata():
    with pytest.raises(ValueError):
        CandidateRecord(
            sample_idx=1,
            frame_idx=10,
            timestamp=1.0,
            metadata={"image": object()},
        )


def test_candidate_lineage_defaults_to_self_identity():
    candidate = CandidateRecord(sample_idx=4, frame_idx=40, timestamp=8.0)

    assert candidate["merged_from_sample_idxs"] == [4]
    assert candidate["merged_timestamps"] == [8.0]


def test_candidate_snapshot_is_materialized_json_safe_copy():
    candidate = CandidateRecord(sample_idx=2, frame_idx=20, timestamp=4.0).with_updates(
        ocr_tokens=["alpha"],
    )
    batch = CandidateBatch(stage="test", candidates=(candidate,))

    payload = snapshot_candidate_batch("test", batch)
    updated = candidate.with_updates(ocr_tokens=["beta"])

    assert updated["ocr_tokens"] == ["beta"]
    assert payload["candidates"][0]["ocr_token_count"] == 1
    assert payload["candidates"][0]["sample_idx"] == 2
