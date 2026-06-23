import pytest

from keyframe.diarization import (
    PreflightFeatures,
    PreflightPolicy,
    ValidationError,
    preflight_features_from_dict,
    route_preflight,
)


_FROZEN_SHA = "a" * 40


def _policy(**overrides):
    values = {
        "confident_capture_modes": ("separate_tracks",),
        "frozen_git_sha": _FROZEN_SHA,
        "max_confident_clipping_estimate": 0.05,
        "max_confident_rough_overlap_estimate": 0.35,
        "max_confident_speaker_count_hint": 6,
        "max_duration_ms": 14_400_000,
        "min_confident_speaker_count_hint": 2,
        "min_confident_speech_ratio": 0.30,
        "min_duration_ms": 30_000,
        "min_sample_rate_hz": 16_000,
        "policy_id": "launch-preflight",
        "supported_capture_modes": ("separate_tracks", "mono_mix"),
        "supported_channel_counts": (1, 2),
        "supported_codecs": ("pcm_s16le", "opus"),
        "supported_locales": ("en-US",),
        "supported_sources": ("zoom", "teams"),
        "tuned_on_splits": ("public_dev",),
        "validated_on_splits": ("public_holdout", "private_acceptance"),
        "version": "2026-06-19",
    }
    values.update(overrides)
    return PreflightPolicy(**values)


def _features(**overrides):
    values = {
        "capture_mode": "separate_tracks",
        "channel_count": 2,
        "clipping_estimate": 0.01,
        "codec": "pcm_s16le",
        "declared_locale": "en-US",
        "duration_ms": 900_000,
        "rough_overlap_estimate": 0.12,
        "sample_rate_hz": 16_000,
        "source": "zoom",
        "speaker_count_hint": 3,
        "speech_ratio": 0.62,
    }
    values.update(overrides)
    return PreflightFeatures(**values)


def test_preflight_routes_in_scope_call_to_confident_pipeline():
    decision = route_preflight(_policy(), _features())

    assert decision.route == "confident_pipeline"
    assert decision.accepted_for_pipeline is True
    assert decision.reasons == ()
    assert decision.policy_version == "2026-06-19"
    assert decision.frozen_git_sha == _FROZEN_SHA
    assert decision.tuned_on_splits == ("public_dev",)
    assert decision.validated_on_splits == ("public_holdout", "private_acceptance")
    assert decision.to_dict()["features"]["speaker_count_hint"] == 3


def test_preflight_routes_low_quality_call_to_needs_review_with_reason_codes():
    decision = route_preflight(
        _policy(),
        _features(
            clipping_estimate=0.20,
            rough_overlap_estimate=0.80,
            speech_ratio=0.10,
        ),
    )

    assert decision.route == "needs_review"
    assert decision.accepted_for_pipeline is False
    assert decision.reasons == (
        "clipping_above_confident_threshold",
        "speech_ratio_below_confident_threshold",
        "rough_overlap_above_confident_threshold",
    )


def test_preflight_routes_mono_mix_out_of_confident_scope_to_diagnostic_only():
    decision = route_preflight(
        _policy(),
        _features(capture_mode="mono_mix", channel_count=1),
    )

    assert decision.route == "diagnostic_only"
    assert decision.accepted_for_pipeline is False
    assert decision.reasons == ("capture_mode_outside_confident_scope",)


def test_preflight_routes_unknown_speaker_count_to_needs_review():
    decision = route_preflight(_policy(), _features(speaker_count_hint=None))

    assert decision.route == "needs_review"
    assert decision.reasons == ("speaker_count_hint_unknown",)


@pytest.mark.parametrize(
    ("feature_overrides", "reason"),
    [
        ({"declared_locale": "fr-CA"}, "unsupported_locale"),
        ({"source": "webex"}, "unsupported_source"),
    ],
)
def test_preflight_routes_unsupported_language_and_source_to_unsupported(feature_overrides, reason):
    decision = route_preflight(_policy(), _features(**feature_overrides))

    assert decision.route == "unsupported"
    assert reason in decision.reasons


def test_preflight_policy_rejects_unversioned_policy():
    with pytest.raises(ValidationError, match="preflight_policy.version is required"):
        _policy(version="")


def test_preflight_policy_rejects_tuned_on_holdout_policy():
    with pytest.raises(ValidationError, match="cannot tune on holdout split: public_holdout"):
        _policy(tuned_on_splits=("public_holdout",), validated_on_splits=("private_acceptance",))


def test_preflight_policy_rejects_non_frozen_git_sha():
    with pytest.raises(ValidationError, match="must be a frozen 40-character git SHA"):
        _policy(frozen_git_sha="HEAD")


def test_preflight_feature_payload_rejects_candidate_invisible_reference_fields():
    payload = _features().to_dict()
    payload["reference_speaker_id"] = "AMI-P1"

    with pytest.raises(ValidationError, match="reference_speaker_id must remain candidate-invisible"):
        preflight_features_from_dict(payload)


def test_preflight_feature_payload_uses_candidate_visible_fields_only():
    parsed = preflight_features_from_dict(_features().to_dict())

    assert parsed == _features()
