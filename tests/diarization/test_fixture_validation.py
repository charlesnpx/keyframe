import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from keyframe.diarization import (
    ChannelRecord,
    DatasetAccess,
    DatasetManifest,
    DatasetSplitManifest,
    ExpectedDatasetFile,
    ScoringPolicyManifest,
    build_candidate_bundle_from_recording,
    build_fixture_slice_metadata,
    merge_fixture_validation_results,
    read_recording_json,
    validate_candidate_bundle_against_reference,
    validate_canonical_reference_payload,
    validate_fixture_gate,
    validate_manifest_expected_files,
    validate_scoring_exports,
)


FIXTURE_DIR = Path("tests/diarization/fixtures")


def _payload(name):
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _scoring_policy(*, ignore_overlap=False):
    return ScoringPolicyManifest(
        policy_id="fixture-policy",
        version="1",
        description="Fixture validation policy",
        ignore_overlap=ignore_overlap,
    )


def _manifest_for_files(expected_files):
    return DatasetManifest(
        dataset_id="fixture-validation",
        name="Fixture Validation",
        role="smoke_ci",
        access=DatasetAccess(mode="local_only", redistribution="allowed"),
        license_url="https://example.com/license",
        attribution="Synthetic fixture",
        expected_files=tuple(expected_files),
        splits=(
            DatasetSplitManifest(
                split_id="fixture-smoke",
                role="smoke_ci",
                expected_file_paths=tuple(file.path for file in expected_files),
                scoring_policy_id="fixture-policy",
            ),
        ),
        scoring_policies=(_scoring_policy(),),
    )


def test_manifest_expected_file_validation_reports_missing_and_checksum_mismatch(tmp_path):
    actual = tmp_path / "actual.txt"
    actual.write_text("actual", encoding="utf-8")
    expected_files = (
        ExpectedDatasetFile(
            path="actual.txt",
            checksum_sha256="0" * 64,
            file_role="metadata",
            size_bytes=actual.stat().st_size,
        ),
        ExpectedDatasetFile(
            path="missing.txt",
            checksum_sha256=hashlib.sha256(b"missing").hexdigest(),
            file_role="metadata",
        ),
    )

    result = validate_manifest_expected_files(_manifest_for_files(expected_files), root=tmp_path)

    assert result.status == "invalid_fixture"
    assert [issue.category for issue in result.issues] == ["checksum_mismatch", "missing_file"]
    assert result.checked_files == (str(actual), str(tmp_path / "missing.txt"))


def test_valid_overlap_is_allowed_when_scoring_policy_includes_overlap():
    result = validate_canonical_reference_payload(
        _payload("overlap.json"),
        scoring_policy=_scoring_policy(ignore_overlap=False),
    )

    assert result.valid is True
    assert result.issues == ()
    assert any(item.dimension == "overlap_ratio" and item.status == "ready" for item in result.slice_metadata)


def test_overlap_is_invalid_when_policy_ignores_overlap():
    result = validate_canonical_reference_payload(
        _payload("overlap.json"),
        scoring_policy=_scoring_policy(ignore_overlap=True),
    )

    assert result.status == "invalid_fixture"
    assert [issue.category for issue in result.issues] == ["unsupported_overlap"]


def test_unflagged_cross_speaker_span_overlap_is_invalid_when_policy_ignores_overlap():
    payload = _payload("clean_two_speaker.json")
    payload["speaker_spans"][0]["end_ms"] = 600
    payload["speaker_spans"][1]["start_ms"] = 300

    result = validate_canonical_reference_payload(payload, scoring_policy=_scoring_policy(ignore_overlap=True))

    assert result.status == "invalid_fixture"
    assert [issue.category for issue in result.issues] == ["unsupported_overlap"]


def test_invalid_negative_interval_returns_invalid_fixture_not_exception():
    payload = _payload("clean_two_speaker.json")
    payload["words"][0]["start_ms"] = -1

    result = validate_canonical_reference_payload(payload, scoring_policy=_scoring_policy())

    assert result.status == "invalid_fixture"
    assert result.issues[0].category == "invalid_interval"
    assert "start_ms must be >= 0" in result.issues[0].message


def test_missing_speaker_refs_return_invalid_fixture_result():
    result = validate_canonical_reference_payload(_payload("missing_speaker.json"), scoring_policy=_scoring_policy())

    assert result.status == "invalid_fixture"
    assert [issue.category for issue in result.issues] == ["unresolved_speaker"]
    assert result.issues[0].recording_id == "fixture-missing-speaker"


def test_candidate_validation_reports_redaction_and_audio_metadata_failures():
    recording = read_recording_json(FIXTURE_DIR / "clean_two_speaker.json")
    payload = build_candidate_bundle_from_recording(
        recording,
        artifact_id="reference-fixture",
        bundle_id="candidate-fixture",
    ).to_dict()

    leaked_payload = copy.deepcopy(payload)
    leaked_payload["runtime_hints"]["speaker_ref"] = "spk-a"
    leaked = validate_candidate_bundle_against_reference(leaked_payload, recording)

    mismatch_payload = copy.deepcopy(payload)
    mismatch_payload["audio"]["duration_ms"] += 1
    mismatch_payload["runtime_hints"]["timeline"]["duration_ms"] = mismatch_payload["audio"]["duration_ms"]
    mismatch = validate_candidate_bundle_against_reference(mismatch_payload, recording)

    channel_payload = copy.deepcopy(payload)
    channel_payload["audio"]["channel_count"] = 2
    channel_payload["channels"].append({"channel_id": "ch-2"})
    channel_payload["runtime_hints"]["channel_ids"].append("ch-2")
    channel_payload["runtime_hints"]["timeline"]["channel_ids"].append("ch-2")
    channel_mismatch = validate_candidate_bundle_against_reference(channel_payload, recording)

    drift_payload = copy.deepcopy(payload)
    drift_payload["channels"][0]["channel_id"] = "wrong-channel"
    drift_payload["runtime_hints"]["channel_ids"][0] = "wrong-channel"
    drift_payload["runtime_hints"]["timeline"]["channel_ids"][0] = "wrong-channel"
    channel_drift = validate_candidate_bundle_against_reference(drift_payload, recording)

    timeline_payload = copy.deepcopy(payload)
    timeline_payload["runtime_hints"]["timeline"]["timeline_id"] = "wrong-timeline"
    timeline_drift = validate_candidate_bundle_against_reference(timeline_payload, recording)

    transform_payload = copy.deepcopy(payload)
    transform_payload["runtime_hints"]["timeline"]["transform_chain_id"] = "wrong-transform"
    transform_drift = validate_candidate_bundle_against_reference(transform_payload, recording)

    assert leaked.status == "invalid_fixture"
    assert leaked.issues[0].category == "reference_leakage"
    assert mismatch.issues[0].category == "audio_metadata_mismatch"
    assert channel_mismatch.issues[0].category == "audio_metadata_mismatch"
    assert channel_drift.issues[0].category == "audio_metadata_mismatch"
    assert timeline_drift.issues[0].category == "audio_metadata_mismatch"
    assert transform_drift.issues[0].category == "audio_metadata_mismatch"


def test_fixture_gate_allows_mono_mix_when_enabled_for_multichannel_reference():
    recording = replace(
        read_recording_json(FIXTURE_DIR / "clean_two_speaker.json"),
        channels=(ChannelRecord("ch-1", name="mixed"), ChannelRecord("ch-2", name="room")),
    )
    payload = build_candidate_bundle_from_recording(
        recording,
        artifact_id="reference-fixture",
        bundle_id="candidate-fixture",
    ).to_dict()
    payload["audio"]["channel_count"] = 1
    payload["channels"] = [{"channel_id": "mono-mix"}]
    payload["runtime_hints"]["channel_ids"] = ["mono-mix"]
    payload["runtime_hints"]["timeline"]["channel_ids"] = ["mono-mix"]
    payload["runtime_hints"]["timeline"]["transform_chain_id"] = "identity-mono-mix"

    rejected = validate_fixture_gate(candidate_payloads=((payload, recording),))
    accepted = validate_fixture_gate(candidate_payloads=((payload, recording),), allow_mono_mix=True)

    assert rejected.status == "invalid_fixture"
    assert rejected.issues[0].category == "audio_metadata_mismatch"
    assert accepted.status == "valid"
    assert accepted.issues == ()

    payload["channels"] = [{"channel_id": "wrong-channel"}]
    payload["runtime_hints"]["channel_ids"] = ["wrong-channel"]
    payload["runtime_hints"]["timeline"]["channel_ids"] = ["wrong-channel"]
    wrong_mono_id = validate_fixture_gate(candidate_payloads=((payload, recording),), allow_mono_mix=True)

    assert wrong_mono_id.status == "invalid_fixture"
    assert wrong_mono_id.issues[0].category == "audio_metadata_mismatch"


def test_fixture_gate_aggregates_slice_support_across_canonical_payloads():
    first = _payload("clean_two_speaker.json")
    second = copy.deepcopy(first)
    second["recording_id"] = "fixture-clean-two-speaker-copy"
    second["original_audio_id"] = "fixture-clean-two-speaker-copy-original"
    second["canonical_audio_id"] = "fixture-clean-two-speaker-copy-canonical"
    second["timeline_id"] = "fixture-clean-two-speaker-copy-timeline"

    result = validate_fixture_gate(
        canonical_payloads=(first, second),
        scoring_policy=_scoring_policy(),
        minimum_slice_support=2,
    )

    speaker_count_slices = [
        item for item in result.slice_metadata if item.dimension == "speaker_count" and item.value == "2"
    ]
    assert result.status == "valid"
    assert len(speaker_count_slices) == 1
    assert speaker_count_slices[0].status == "ready"
    assert speaker_count_slices[0].support_count == 2
    assert speaker_count_slices[0].recording_ids == (
        "fixture-clean-two-speaker",
        "fixture-clean-two-speaker-copy",
    )


def test_fixture_gate_returns_invalid_fixture_for_duplicate_canonical_recording_ids():
    payload = _payload("clean_two_speaker.json")

    result = validate_fixture_gate(
        canonical_payloads=(payload, copy.deepcopy(payload)),
        scoring_policy=_scoring_policy(),
        minimum_slice_support=1,
    )

    assert result.status == "invalid_fixture"
    assert result.issues[0].category == "schema_validation"
    assert "duplicate canonical recording_id" in result.issues[0].message


def test_merge_fixture_validation_results_aggregates_slice_support():
    first_payload = _payload("clean_two_speaker.json")
    second_payload = copy.deepcopy(first_payload)
    second_payload["recording_id"] = "fixture-clean-two-speaker-copy"
    second_payload["original_audio_id"] = "fixture-clean-two-speaker-copy-original"
    second_payload["canonical_audio_id"] = "fixture-clean-two-speaker-copy-canonical"
    second_payload["timeline_id"] = "fixture-clean-two-speaker-copy-timeline"

    first = validate_canonical_reference_payload(
        first_payload,
        scoring_policy=_scoring_policy(),
        minimum_slice_support=2,
    )
    second = validate_canonical_reference_payload(
        second_payload,
        scoring_policy=_scoring_policy(),
        minimum_slice_support=2,
    )

    merged = merge_fixture_validation_results(first, second)
    speaker_count_slices = [
        item for item in merged.slice_metadata if item.dimension == "speaker_count" and item.value == "2"
    ]

    assert merged.status == "valid"
    assert len(speaker_count_slices) == 1
    assert speaker_count_slices[0].status == "ready"
    assert speaker_count_slices[0].support_count == 2


def test_missing_scoring_exports_are_invalid_fixture_results(tmp_path):
    rttm = tmp_path / "missing.rttm"

    result = validate_scoring_exports(artifact_paths={"rttm": str(rttm)})

    assert result.status == "invalid_fixture"
    assert [issue.category for issue in result.issues] == ["missing_scoring_export", "missing_scoring_export"]
    assert result.issues[0].message == "missing scoring export file: rttm"
    assert result.issues[1].message == "missing scoring export path: uem"


def test_sparse_slices_emit_validator_status_output_for_reports():
    recording = read_recording_json(FIXTURE_DIR / "clean_two_speaker.json")

    slices = build_fixture_slice_metadata((recording,), minimum_support=2)

    assert {item.dimension for item in slices} == {
        "channel_mode",
        "duration_bucket",
        "known_count_mode",
        "overlap_ratio",
        "speaker_count",
        "speech_ratio",
    }
    assert {item.status for item in slices} == {"insufficient_support"}
    assert all(item.support_count == 1 for item in slices)


def test_fixture_gate_aggregates_invalid_fixture_issues_and_slice_metadata():
    result = validate_fixture_gate(
        canonical_payloads=(_payload("missing_speaker.json"),),
        scoring_policy=_scoring_policy(),
        minimum_slice_support=2,
    )

    assert result.status == "invalid_fixture"
    assert result.issues[0].category == "unresolved_speaker"
    assert result.slice_metadata
