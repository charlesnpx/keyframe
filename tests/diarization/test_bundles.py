import math

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


def _candidate_audio():
    return {
        "channel_count": 1,
        "duration_ms": 1200,
        "sample_rate_hz": 16000,
        "time_basis": "canonical_ms",
    }


def _candidate_runtime_hints():
    return {
        "channel_ids": ["ch-1"],
        "mode_supports_speaker_identity": False,
        "timeline": {
            "channel_ids": ["ch-1"],
            "duration_ms": 1200,
            "sample_rate_hz": 16000,
            "time_basis": "canonical_ms",
            "timeline_id": "timeline-fixture",
            "transform_chain_id": "transform-chain-fixture",
        },
    }


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


def test_candidate_bundle_metadata_is_immutable_after_validation():
    bundle = build_candidate_bundle_from_recording(
        _recording(),
        artifact_id="reference-fixture",
        bundle_id="candidate-fixture",
    )

    with pytest.raises(TypeError):
        bundle.runtime_hints["speaker_ref"] = "spk-a"
    with pytest.raises(TypeError):
        bundle.audio["original_audio_id"] = "leak"
    with pytest.raises(TypeError):
        bundle.channels[0]["participant_id"] = "P001"

    payload = bundle.to_dict()
    payload["runtime_hints"]["speaker_ref"] = "mutated-copy"
    assert "speaker_ref" not in bundle.to_dict()["runtime_hints"]


@pytest.mark.parametrize("reserved_key", ["channel_ids", "mode_supports_speaker_identity", "timeline"])
def test_runtime_hint_overrides_cannot_corrupt_generated_metadata(reserved_key):
    with pytest.raises(ValidationError, match="cannot override generated runtime metadata"):
        build_candidate_bundle_from_recording(
            _recording(),
            artifact_id="reference-fixture",
            bundle_id="candidate-fixture",
            runtime_hints={reserved_key: "override"},
        )


@pytest.mark.parametrize("runtime_hints", [[], "", False])
def test_runtime_hint_builder_rejects_falsey_non_object_values(runtime_hints):
    with pytest.raises(ValidationError, match="runtime_hints must be an object"):
        build_candidate_bundle_from_recording(
            _recording(),
            artifact_id="reference-fixture",
            bundle_id="candidate-fixture",
            runtime_hints=runtime_hints,
        )


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


def test_serialized_candidate_payload_validation_enforces_oracle_reportability():
    payload = build_candidate_bundle_from_recording(
        _recording(),
        artifact_id="reference-fixture",
        bundle_id="diagnostic-fixture",
        mode="oracle_diagnostic",
    ).to_dict()

    payload["product_quality_reportable"] = True
    with pytest.raises(ValidationError, match="oracle diagnostic bundles must be non-reportable"):
        validate_candidate_bundle_payload(payload)

    payload["product_quality_reportable"] = False
    payload["oracle_diagnostic"] = False
    with pytest.raises(ValidationError, match="oracle diagnostic bundles must be explicitly labeled"):
        validate_candidate_bundle_payload(payload)

    payload.pop("oracle_diagnostic")
    with pytest.raises(ValidationError, match="candidate_bundle.oracle_diagnostic must be a boolean"):
        validate_candidate_bundle_payload(payload)

    payload = build_candidate_bundle_from_recording(
        _recording(),
        artifact_id="reference-fixture",
        bundle_id="candidate-fixture",
    ).to_dict()
    payload["oracle_diagnostic"] = True
    with pytest.raises(ValidationError, match="product-quality bundles cannot be labeled oracle diagnostic"):
        validate_candidate_bundle_payload(payload)


def test_serialized_candidate_payload_validation_rejects_unsupported_modes():
    payload = build_candidate_bundle_from_recording(
        _recording(),
        artifact_id="reference-fixture",
        bundle_id="candidate-fixture",
    ).to_dict()
    payload["mode"] = "unsupported"

    with pytest.raises(ValidationError, match="candidate_bundle.mode is not supported"):
        validate_candidate_bundle_payload(payload)


@pytest.mark.parametrize(
    ("key", "expected_message"),
    [
        ("bundle_id", "candidate_bundle.bundle_id is required"),
        ("audio", "candidate_bundle.audio must be an object"),
        ("mode", "candidate_bundle.mode is required"),
        ("oracle_diagnostic", "candidate_bundle.oracle_diagnostic must be a boolean"),
        ("product_quality_reportable", "candidate_bundle.product_quality_reportable must be a boolean"),
        ("runtime_hints", "candidate_bundle.runtime_hints must be an object"),
    ],
)
def test_serialized_candidate_payload_validation_rejects_missing_required_fields(key, expected_message):
    payload = build_candidate_bundle_from_recording(
        _recording(),
        artifact_id="reference-fixture",
        bundle_id="candidate-fixture",
    ).to_dict()
    payload.pop(key)

    with pytest.raises(ValidationError, match=expected_message):
        validate_candidate_bundle_payload(payload)


def test_serialized_candidate_payload_validation_rejects_missing_channels():
    payload = build_candidate_bundle_from_recording(
        _recording(),
        artifact_id="reference-fixture",
        bundle_id="candidate-fixture",
    ).to_dict()
    payload.pop("channels")

    with pytest.raises(ValidationError, match="candidate_bundle.channels must be an iterable"):
        validate_candidate_bundle_payload(payload)


@pytest.mark.parametrize(
    ("audio", "expected_message"),
    [
        ({}, "candidate_bundle.audio.channel_count must be an integer"),
        (
            {
                "channel_count": 2,
                "duration_ms": 1200,
                "sample_rate_hz": 16000,
                "time_basis": "canonical_ms",
            },
            "candidate_bundle.audio.channel_count must match channels",
        ),
    ],
)
def test_serialized_candidate_payload_validation_rejects_invalid_audio_metadata(audio, expected_message):
    payload = build_candidate_bundle_from_recording(
        _recording(),
        artifact_id="reference-fixture",
        bundle_id="candidate-fixture",
    ).to_dict()
    payload["audio"] = audio

    with pytest.raises(ValidationError, match=expected_message):
        validate_candidate_bundle_payload(payload)


@pytest.mark.parametrize(
    ("channels", "expected_message"),
    [
        ([], "candidate_bundle.channels is required"),
        ([{"channel_id": ""}], r"candidate_bundle.channels\[0\].channel_id is required"),
        ([{"channel_id": "   "}], r"candidate_bundle.channels\[0\].channel_id is required"),
        (
            [{"channel_id": "ch-1"}, {"channel_id": " ch-1 "}],
            "duplicate candidate_bundle.channels.channel_id: ch-1",
        ),
    ],
)
def test_serialized_candidate_payload_validation_rejects_invalid_channel_payloads(channels, expected_message):
    payload = build_candidate_bundle_from_recording(
        _recording(),
        artifact_id="reference-fixture",
        bundle_id="candidate-fixture",
    ).to_dict()
    payload["channels"] = channels

    with pytest.raises(ValidationError, match=expected_message):
        validate_candidate_bundle_payload(payload)


@pytest.mark.parametrize(
    ("runtime_hints", "expected_message"),
    [
        ({}, "candidate_bundle.runtime_hints.channel_ids must be a list"),
        (
            {
                "channel_ids": ["ch-1"],
                "mode_supports_speaker_identity": True,
                "timeline": {
                    "channel_ids": ["ch-1"],
                    "duration_ms": 1200,
                    "sample_rate_hz": 16000,
                    "time_basis": "canonical_ms",
                    "timeline_id": "timeline-fixture",
                    "transform_chain_id": "transform-chain-fixture",
                },
            },
            "candidate_bundle.runtime_hints.mode_supports_speaker_identity must be false",
        ),
        (
            {
                "channel_ids": ["other-channel"],
                "mode_supports_speaker_identity": False,
                "timeline": {
                    "channel_ids": ["other-channel"],
                    "duration_ms": 1200,
                    "sample_rate_hz": 16000,
                    "time_basis": "canonical_ms",
                    "timeline_id": "timeline-fixture",
                    "transform_chain_id": "transform-chain-fixture",
                },
            },
            "candidate_bundle.runtime_hints.channel_ids must match channels",
        ),
        (
            {
                "channel_ids": ["ch-1"],
                "mode_supports_speaker_identity": False,
            },
            "candidate_bundle.runtime_hints.timeline must be an object",
        ),
        (
            {
                "channel_ids": ["ch-1"],
                "mode_supports_speaker_identity": False,
                "timeline": {"channel_ids": ["ch-1"]},
            },
            "candidate_bundle.runtime_hints.timeline.duration_ms must be an integer",
        ),
        (
            {
                "channel_ids": ["ch-1"],
                "mode_supports_speaker_identity": False,
                "timeline": {
                    "channel_ids": ["ch-1"],
                    "duration_ms": 1200,
                    "sample_rate_hz": 16000,
                    "time_basis": "canonical_ms",
                    "transform_chain_id": "transform-chain-fixture",
                },
            },
            "candidate_bundle.runtime_hints.timeline.timeline_id is required",
        ),
        (
            {
                "channel_ids": ["ch-1"],
                "mode_supports_speaker_identity": False,
                "timeline": {
                    "channel_ids": ["other-channel"],
                    "duration_ms": 1200,
                    "sample_rate_hz": 16000,
                    "time_basis": "canonical_ms",
                    "timeline_id": "timeline-fixture",
                    "transform_chain_id": "transform-chain-fixture",
                },
            },
            "candidate_bundle.runtime_hints.timeline.channel_ids must match channels",
        ),
    ],
)
def test_serialized_candidate_payload_validation_rejects_invalid_runtime_hints(
    runtime_hints,
    expected_message,
):
    payload = build_candidate_bundle_from_recording(
        _recording(),
        artifact_id="reference-fixture",
        bundle_id="candidate-fixture",
    ).to_dict()
    payload["runtime_hints"] = runtime_hints

    with pytest.raises(ValidationError, match=expected_message):
        validate_candidate_bundle_payload(payload)


def test_product_quality_serialized_payloads_must_be_reportable():
    payload = build_candidate_bundle_from_recording(
        _recording(),
        artifact_id="reference-fixture",
        bundle_id="candidate-fixture",
    ).to_dict()
    payload["product_quality_reportable"] = False

    with pytest.raises(ValidationError, match="product-quality bundles must be reportable"):
        validate_candidate_bundle_payload(payload)


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
        audio = _candidate_audio()
        audio["original_audio_id"] = "leak"
        CandidateBundle(
            bundle_id="candidate-fixture",
            mode="product_realistic",
            audio=audio,
            channels=({"channel_id": "ch-1"},),
            runtime_hints=_candidate_runtime_hints(),
        )


def test_candidate_bundle_constructor_rejects_incomplete_structural_payloads():
    with pytest.raises(ValidationError, match="candidate_bundle.audio.channel_count must be an integer"):
        CandidateBundle(
            bundle_id="candidate-fixture",
            mode="product_realistic",
            audio={"duration_ms": 1200, "sample_rate_hz": 16000},
            channels=({"channel_id": "ch-1"},),
            runtime_hints={},
        )


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_candidate_metadata_rejects_non_finite_json_numbers(bad_value):
    with pytest.raises(ValidationError, match="must be a finite JSON number"):
        runtime_hints = _candidate_runtime_hints()
        runtime_hints["score"] = bad_value
        CandidateBundle(
            bundle_id="candidate-fixture",
            mode="product_realistic",
            audio=_candidate_audio(),
            channels=({"channel_id": "ch-1"},),
            runtime_hints=runtime_hints,
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
