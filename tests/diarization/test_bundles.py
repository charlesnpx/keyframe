import pytest

from keyframe.diarization import (
    CandidateBundle,
    ReferenceBundle,
    ValidationError,
    build_candidate_bundle,
    build_candidate_bundle_from_recording,
    read_recording_json,
    validate_candidate_bundle_payload,
)


FORBIDDEN_KEYS = {
    "canonical_audio_id",
    "corpus_identity",
    "corpus_speaker_id",
    "cross_recording_identity",
    "display_label",
    "evaluator_speaker_map",
    "global_identity",
    "local_audio_sha256",
    "oracle",
    "oracle_metadata",
    "original_audio_id",
    "participant_id",
    "reference_speaker_id",
    "role",
    "role_label",
    "speaker_ref",
    "voice_profile",
}


def _recording():
    return read_recording_json("tests/diarization/fixtures/clean_two_speaker.json")


def _walk_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def _assert_no_forbidden_fields(payload):
    assert FORBIDDEN_KEYS.isdisjoint(set(_walk_keys(payload)))


def test_reference_bundle_contains_evaluator_only_data():
    recording = _recording()

    reference = ReferenceBundle.from_recording(
        recording,
        artifact_id="reference-fixture",
        local_audio_sha256="local-cache-hash",
        oracle_metadata={"corpus_identity": "ami-meeting-a"},
    )

    payload = reference.to_evaluator_dict()
    assert payload["recording"]["speakers"][0]["speaker_ref"] == "spk-a"
    assert payload["evaluator_speaker_map"] == {"spk-a": "spk-a", "spk-b": "spk-b"}
    assert payload["artifact"]["timeline"]["original_audio_id"] == recording.original_audio_id
    assert payload["artifact"]["timeline"]["local_audio_sha256"] == "local-cache-hash"
    assert payload["oracle_metadata"]["corpus_identity"] == "ami-meeting-a"


def test_product_realistic_candidate_bundle_exposes_only_runtime_audio_and_channel_inputs():
    bundle = build_candidate_bundle_from_recording(
        _recording(),
        artifact_id="reference-fixture",
        bundle_id="candidate-fixture",
        mode="product_realistic",
        local_audio_sha256="local-cache-hash",
    )
    payload = bundle.to_dict()

    assert bundle.product_quality_reportable is True
    assert payload["mode"] == "product_realistic"
    assert payload["audio"] == {
        "channel_count": 1,
        "duration_ms": 1200,
        "sample_rate_hz": 16000,
        "time_basis": "canonical_ms",
    }
    assert payload["channels"] == [{"channel_id": "ch-1"}]
    assert payload["runtime_hints"]["channel_ids"] == ["ch-1"]
    assert payload["runtime_hints"]["mode_supports_speaker_identity"] is False
    _assert_no_forbidden_fields(payload)


def test_authenticated_track_metadata_mode_includes_track_names_but_not_speaker_identity():
    bundle = build_candidate_bundle_from_recording(
        _recording(),
        artifact_id="reference-fixture",
        bundle_id="candidate-fixture",
        mode="authenticated_track_metadata",
    )

    payload = bundle.to_dict()
    assert payload["mode"] == "authenticated_track_metadata"
    assert payload["channels"] == [{"channel_id": "ch-1", "track_name": "mixed"}]
    _assert_no_forbidden_fields(payload)


def test_oracle_diagnostic_bundle_is_explicitly_non_reportable():
    reference = ReferenceBundle.from_recording(_recording(), artifact_id="reference-fixture")

    bundle = build_candidate_bundle(reference, bundle_id="diagnostic-fixture", mode="oracle_diagnostic")

    payload = bundle.to_dict()
    assert payload["mode"] == "oracle_diagnostic"
    assert payload["oracle_diagnostic"] is True
    assert payload["product_quality_reportable"] is False
    _assert_no_forbidden_fields(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"runtime_hints": {"speaker_ref": "spk-a"}},
        {"channels": [{"channel_id": "ch-1", "participant_id": "P001"}]},
        {"audio": {"original_audio_id": "fixture-original"}},
        {"oracle_metadata": {"role_label": "chair"}},
        {"runtime_hints": {"nested": {"voice_profile": "retained-profile"}}},
    ],
)
def test_candidate_bundle_validation_rejects_oracle_and_identity_leaks(payload):
    with pytest.raises(ValidationError, match="forbidden in candidate bundles"):
        validate_candidate_bundle_payload(payload)


def test_candidate_bundle_constructor_rejects_leaked_fields():
    with pytest.raises(ValidationError, match="forbidden in candidate bundles"):
        CandidateBundle(
            bundle_id="candidate-fixture",
            mode="product_realistic",
            audio={"duration_ms": 1200, "sample_rate_hz": 16000, "original_audio_id": "leak"},
            channels=({"channel_id": "ch-1"},),
            runtime_hints={},
        )


def test_oracle_diagnostic_label_cannot_be_misreported_as_product_quality():
    with pytest.raises(ValidationError, match="oracle diagnostic bundles must be explicitly labeled"):
        CandidateBundle(
            bundle_id="candidate-fixture",
            mode="oracle_diagnostic",
            audio={"duration_ms": 1200, "sample_rate_hz": 16000},
            channels=({"channel_id": "ch-1"},),
            runtime_hints={},
            oracle_diagnostic=False,
        )

    with pytest.raises(ValidationError, match="product-quality bundles cannot be labeled oracle diagnostic"):
        CandidateBundle(
            bundle_id="candidate-fixture",
            mode="product_realistic",
            audio={"duration_ms": 1200, "sample_rate_hz": 16000},
            channels=({"channel_id": "ch-1"},),
            runtime_hints={},
            oracle_diagnostic=True,
        )
