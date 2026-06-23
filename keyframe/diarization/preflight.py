"""Candidate-visible preflight routing for diarization launch scope.

The preflight router intentionally accepts only declared metadata and cheap
audio diagnostics. It must not consume reference labels, benchmark outcomes,
speaker identities, embeddings, or persisted cross-call profile material.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from keyframe.diarization.models import ValidationError
from keyframe.diarization.rendering import (
    RenderedTranscript,
    TranscriptOverlay,
    render_transcript,
)


PreflightCaptureMode = Literal["separate_tracks", "mono_mix", "authenticated_track_metadata"]
PreflightRoute = Literal["confident_pipeline", "needs_review", "diagnostic_only", "unsupported"]

ALLOWED_PREFLIGHT_CAPTURE_MODES = frozenset({"separate_tracks", "mono_mix", "authenticated_track_metadata"})
ALLOWED_PREFLIGHT_ROUTES = frozenset({"confident_pipeline", "needs_review", "diagnostic_only", "unsupported"})

_HEX_DIGITS = frozenset("0123456789abcdef")
_REFERENCE_IDENTITY_KEYS = frozenset(
    {
        "benchmark_result",
        "corpus_speaker_id",
        "corpus_speaker_ids",
        "cross_recording_identity",
        "display_label",
        "evaluator_speaker_map",
        "global_identity",
        "gold_label",
        "oracle",
        "oracle_metadata",
        "participant_id",
        "participant_ids",
        "reference_label",
        "reference_speaker_id",
        "reference_speaker_ids",
        "speaker_embedding",
        "speaker_ref",
        "speaker_refs",
        "voice_embedding",
        "voice_fingerprint",
        "voice_profile",
    }
)


@runtime_checkable
class PreflightDiagnosticFeatureProvider(Protocol):
    """Cheap feature provider for runtime routing before model invocation."""

    def collect_preflight_features(self) -> "PreflightFeatures":
        """Return candidate-visible metadata and cheap audio diagnostics."""


@dataclass(frozen=True)
class PreflightFeatures:
    """Candidate-visible inputs for routing one call through launch-scope policy."""

    declared_locale: str
    source: str
    capture_mode: PreflightCaptureMode
    channel_count: int
    duration_ms: int
    sample_rate_hz: int
    codec: str
    clipping_estimate: float
    speech_ratio: float
    rough_overlap_estimate: float
    speaker_count_hint: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "declared_locale", _require_id(self.declared_locale, "preflight.declared_locale"))
        object.__setattr__(self, "source", _require_id(self.source, "preflight.source"))
        object.__setattr__(self, "capture_mode", _validate_capture_mode(self.capture_mode, "preflight.capture_mode"))
        object.__setattr__(self, "channel_count", _positive_int(self.channel_count, "preflight.channel_count"))
        object.__setattr__(self, "duration_ms", _positive_int(self.duration_ms, "preflight.duration_ms"))
        object.__setattr__(self, "sample_rate_hz", _positive_int(self.sample_rate_hz, "preflight.sample_rate_hz"))
        object.__setattr__(self, "codec", _require_id(self.codec, "preflight.codec"))
        object.__setattr__(
            self,
            "clipping_estimate",
            _probability(self.clipping_estimate, "preflight.clipping_estimate"),
        )
        object.__setattr__(self, "speech_ratio", _probability(self.speech_ratio, "preflight.speech_ratio"))
        object.__setattr__(
            self,
            "rough_overlap_estimate",
            _probability(self.rough_overlap_estimate, "preflight.rough_overlap_estimate"),
        )
        if self.speaker_count_hint is not None:
            object.__setattr__(
                self,
                "speaker_count_hint",
                _positive_int(self.speaker_count_hint, "preflight.speaker_count_hint"),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "capture_mode": self.capture_mode,
            "channel_count": self.channel_count,
            "clipping_estimate": self.clipping_estimate,
            "codec": self.codec,
            "declared_locale": self.declared_locale,
            "duration_ms": self.duration_ms,
            "rough_overlap_estimate": self.rough_overlap_estimate,
            "sample_rate_hz": self.sample_rate_hz,
            "source": self.source,
            "speaker_count_hint": self.speaker_count_hint,
            "speech_ratio": self.speech_ratio,
        }


@dataclass(frozen=True)
class PreflightPolicy:
    """Versioned runtime routing policy with frozen tuning provenance."""

    policy_id: str
    version: str
    frozen_git_sha: str
    tuned_on_splits: tuple[str, ...]
    validated_on_splits: tuple[str, ...]
    supported_locales: tuple[str, ...]
    supported_sources: tuple[str, ...]
    supported_capture_modes: tuple[PreflightCaptureMode, ...]
    confident_capture_modes: tuple[PreflightCaptureMode, ...]
    supported_channel_counts: tuple[int, ...]
    supported_codecs: tuple[str, ...]
    min_duration_ms: int
    max_duration_ms: int
    min_sample_rate_hz: int
    max_confident_clipping_estimate: float
    min_confident_speech_ratio: float
    max_confident_rough_overlap_estimate: float
    min_confident_speaker_count_hint: int = 2
    max_confident_speaker_count_hint: int = 8
    require_speaker_count_hint_for_confident: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _require_id(self.policy_id, "preflight_policy.policy_id"))
        object.__setattr__(self, "version", _require_id(self.version, "preflight_policy.version"))
        object.__setattr__(
            self,
            "frozen_git_sha",
            _validate_frozen_git_sha(self.frozen_git_sha, "preflight_policy.frozen_git_sha"),
        )
        tuned_on = _non_empty_id_tuple(self.tuned_on_splits, "preflight_policy.tuned_on_splits")
        validated_on = _non_empty_id_tuple(self.validated_on_splits, "preflight_policy.validated_on_splits")
        if set(tuned_on) & set(validated_on):
            raise ValidationError("preflight_policy tuned_on_splits and validated_on_splits must be disjoint")
        tuned_holdout = tuple(split for split in tuned_on if _is_holdout_split(split))
        if tuned_holdout:
            raise ValidationError(f"preflight_policy cannot tune on holdout split: {tuned_holdout[0]}")
        object.__setattr__(self, "tuned_on_splits", tuned_on)
        object.__setattr__(self, "validated_on_splits", validated_on)
        object.__setattr__(
            self,
            "supported_locales",
            _non_empty_id_tuple(self.supported_locales, "preflight_policy.supported_locales"),
        )
        object.__setattr__(
            self,
            "supported_sources",
            _non_empty_id_tuple(self.supported_sources, "preflight_policy.supported_sources"),
        )
        supported_capture_modes = tuple(
            _validate_capture_mode(value, "preflight_policy.supported_capture_modes")
            for value in _sequence(self.supported_capture_modes, "preflight_policy.supported_capture_modes")
        )
        if not supported_capture_modes:
            raise ValidationError("preflight_policy.supported_capture_modes is required")
        _reject_duplicates(supported_capture_modes, "preflight_policy.supported_capture_modes")
        confident_capture_modes = tuple(
            _validate_capture_mode(value, "preflight_policy.confident_capture_modes")
            for value in _sequence(self.confident_capture_modes, "preflight_policy.confident_capture_modes")
        )
        if not confident_capture_modes:
            raise ValidationError("preflight_policy.confident_capture_modes is required")
        _reject_duplicates(confident_capture_modes, "preflight_policy.confident_capture_modes")
        unsupported_confident = set(confident_capture_modes) - set(supported_capture_modes)
        if unsupported_confident:
            raise ValidationError(
                "preflight_policy.confident_capture_modes must be included in supported_capture_modes"
            )
        object.__setattr__(self, "supported_capture_modes", supported_capture_modes)
        object.__setattr__(self, "confident_capture_modes", confident_capture_modes)
        supported_channel_counts = tuple(
            _positive_int(value, "preflight_policy.supported_channel_counts")
            for value in _sequence(self.supported_channel_counts, "preflight_policy.supported_channel_counts")
        )
        if not supported_channel_counts:
            raise ValidationError("preflight_policy.supported_channel_counts is required")
        _reject_duplicates(supported_channel_counts, "preflight_policy.supported_channel_counts")
        object.__setattr__(self, "supported_channel_counts", supported_channel_counts)
        object.__setattr__(
            self,
            "supported_codecs",
            _non_empty_id_tuple(self.supported_codecs, "preflight_policy.supported_codecs"),
        )
        object.__setattr__(self, "min_duration_ms", _positive_int(self.min_duration_ms, "preflight_policy.min_duration_ms"))
        object.__setattr__(self, "max_duration_ms", _positive_int(self.max_duration_ms, "preflight_policy.max_duration_ms"))
        if self.min_duration_ms > self.max_duration_ms:
            raise ValidationError("preflight_policy.min_duration_ms cannot exceed max_duration_ms")
        object.__setattr__(
            self,
            "min_sample_rate_hz",
            _positive_int(self.min_sample_rate_hz, "preflight_policy.min_sample_rate_hz"),
        )
        object.__setattr__(
            self,
            "max_confident_clipping_estimate",
            _probability(
                self.max_confident_clipping_estimate,
                "preflight_policy.max_confident_clipping_estimate",
            ),
        )
        object.__setattr__(
            self,
            "min_confident_speech_ratio",
            _probability(self.min_confident_speech_ratio, "preflight_policy.min_confident_speech_ratio"),
        )
        object.__setattr__(
            self,
            "max_confident_rough_overlap_estimate",
            _probability(
                self.max_confident_rough_overlap_estimate,
                "preflight_policy.max_confident_rough_overlap_estimate",
            ),
        )
        object.__setattr__(
            self,
            "min_confident_speaker_count_hint",
            _positive_int(
                self.min_confident_speaker_count_hint,
                "preflight_policy.min_confident_speaker_count_hint",
            ),
        )
        object.__setattr__(
            self,
            "max_confident_speaker_count_hint",
            _positive_int(
                self.max_confident_speaker_count_hint,
                "preflight_policy.max_confident_speaker_count_hint",
            ),
        )
        if self.min_confident_speaker_count_hint > self.max_confident_speaker_count_hint:
            raise ValidationError(
                "preflight_policy.min_confident_speaker_count_hint cannot exceed max_confident_speaker_count_hint"
            )
        object.__setattr__(
            self,
            "require_speaker_count_hint_for_confident",
            _require_bool(
                self.require_speaker_count_hint_for_confident,
                "preflight_policy.require_speaker_count_hint_for_confident",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "confident_capture_modes": list(self.confident_capture_modes),
            "frozen_git_sha": self.frozen_git_sha,
            "max_confident_clipping_estimate": self.max_confident_clipping_estimate,
            "max_confident_rough_overlap_estimate": self.max_confident_rough_overlap_estimate,
            "max_confident_speaker_count_hint": self.max_confident_speaker_count_hint,
            "max_duration_ms": self.max_duration_ms,
            "min_confident_speaker_count_hint": self.min_confident_speaker_count_hint,
            "min_confident_speech_ratio": self.min_confident_speech_ratio,
            "min_duration_ms": self.min_duration_ms,
            "min_sample_rate_hz": self.min_sample_rate_hz,
            "policy_id": self.policy_id,
            "require_speaker_count_hint_for_confident": self.require_speaker_count_hint_for_confident,
            "supported_capture_modes": list(self.supported_capture_modes),
            "supported_channel_counts": list(self.supported_channel_counts),
            "supported_codecs": list(self.supported_codecs),
            "supported_locales": list(self.supported_locales),
            "supported_sources": list(self.supported_sources),
            "tuned_on_splits": list(self.tuned_on_splits),
            "validated_on_splits": list(self.validated_on_splits),
            "version": self.version,
        }


@dataclass(frozen=True)
class PreflightRouteDecision:
    """Deterministic route plus report-safe policy provenance and reason codes."""

    route: PreflightRoute
    reasons: tuple[str, ...]
    policy_id: str
    policy_version: str
    frozen_git_sha: str
    tuned_on_splits: tuple[str, ...]
    validated_on_splits: tuple[str, ...]
    features: PreflightFeatures

    def __post_init__(self) -> None:
        route = _require_id(self.route, "preflight_decision.route")
        if route not in ALLOWED_PREFLIGHT_ROUTES:
            raise ValidationError(f"preflight_decision.route is not supported: {route}")
        object.__setattr__(self, "route", route)
        object.__setattr__(self, "reasons", _id_tuple(self.reasons, "preflight_decision.reasons"))
        if route == "confident_pipeline" and self.reasons:
            raise ValidationError("confident preflight decisions cannot include reasons")
        if route != "confident_pipeline" and not self.reasons:
            raise ValidationError("non-confident preflight decisions require reasons")
        object.__setattr__(self, "policy_id", _require_id(self.policy_id, "preflight_decision.policy_id"))
        object.__setattr__(self, "policy_version", _require_id(self.policy_version, "preflight_decision.policy_version"))
        object.__setattr__(
            self,
            "frozen_git_sha",
            _validate_frozen_git_sha(self.frozen_git_sha, "preflight_decision.frozen_git_sha"),
        )
        object.__setattr__(
            self,
            "tuned_on_splits",
            _non_empty_id_tuple(self.tuned_on_splits, "preflight_decision.tuned_on_splits"),
        )
        object.__setattr__(
            self,
            "validated_on_splits",
            _non_empty_id_tuple(self.validated_on_splits, "preflight_decision.validated_on_splits"),
        )
        if not isinstance(self.features, PreflightFeatures):
            raise ValidationError("preflight_decision.features must be PreflightFeatures")

    @property
    def accepted_for_pipeline(self) -> bool:
        return self.route == "confident_pipeline"

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted_for_pipeline": self.accepted_for_pipeline,
            "features": self.features.to_dict(),
            "frozen_git_sha": self.frozen_git_sha,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "reasons": list(self.reasons),
            "route": self.route,
            "tuned_on_splits": list(self.tuned_on_splits),
            "validated_on_splits": list(self.validated_on_splits),
        }


@dataclass(frozen=True)
class PreflightManualOverrideAudit:
    """Audit trail for a human route override that is excluded from benchmark truth."""

    override_id: str
    actor_id: str
    reason: str
    override_route: PreflightRoute
    created_at: str
    excluded_from_benchmark_truth: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "override_id", _require_id(self.override_id, "preflight_override.override_id"))
        object.__setattr__(self, "actor_id", _require_id(self.actor_id, "preflight_override.actor_id"))
        object.__setattr__(self, "reason", _require_text(self.reason, "preflight_override.reason"))
        route = _require_id(self.override_route, "preflight_override.override_route")
        if route not in ALLOWED_PREFLIGHT_ROUTES:
            raise ValidationError(f"preflight_override.override_route is not supported: {route}")
        object.__setattr__(self, "override_route", route)
        object.__setattr__(self, "created_at", _require_id(self.created_at, "preflight_override.created_at"))
        if self.excluded_from_benchmark_truth is not True:
            raise ValidationError("preflight_override.excluded_from_benchmark_truth must be true")

    def to_dict(self) -> dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "created_at": self.created_at,
            "excluded_from_benchmark_truth": self.excluded_from_benchmark_truth,
            "override_id": self.override_id,
            "override_route": self.override_route,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PreflightJobRecord:
    """Persisted route state for one runtime job or benchmark candidate."""

    job_id: str
    decision: PreflightRouteDecision
    validated_launch_scope_version: str
    manual_override: PreflightManualOverrideAudit | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "job_id", _require_id(self.job_id, "preflight_job.job_id"))
        if not isinstance(self.decision, PreflightRouteDecision):
            raise ValidationError("preflight_job.decision must be a PreflightRouteDecision")
        object.__setattr__(
            self,
            "validated_launch_scope_version",
            _require_id(
                self.validated_launch_scope_version,
                "preflight_job.validated_launch_scope_version",
            ),
        )
        if self.manual_override is not None and not isinstance(self.manual_override, PreflightManualOverrideAudit):
            raise ValidationError("preflight_job.manual_override must be PreflightManualOverrideAudit")

    @property
    def route(self) -> PreflightRoute:
        return self.decision.route

    @property
    def effective_route(self) -> PreflightRoute:
        return self.manual_override.override_route if self.manual_override is not None else self.route

    @property
    def manual_override_applied(self) -> bool:
        return self.manual_override is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "effective_route": self.effective_route,
            "features": self.decision.features.to_dict(),
            "frozen_git_sha": self.decision.frozen_git_sha,
            "job_id": self.job_id,
            "manual_override": self.manual_override.to_dict() if self.manual_override is not None else None,
            "manual_override_applied": self.manual_override_applied,
            "policy_id": self.decision.policy_id,
            "policy_version": self.decision.policy_version,
            "reasons": list(self.decision.reasons),
            "route": self.route,
            "tuned_on_splits": list(self.decision.tuned_on_splits),
            "validated_launch_scope_version": self.validated_launch_scope_version,
            "validated_on_splits": list(self.decision.validated_on_splits),
        }


@dataclass(frozen=True)
class PreflightRouteAssessment:
    """Benchmark truth row for route-level confusion reporting."""

    corpus_id: str
    branch_id: str
    recording_id: str
    predicted_route: PreflightRoute
    reference_route: PreflightRoute
    manual_override_applied: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "corpus_id", _require_id(self.corpus_id, "route_assessment.corpus_id"))
        object.__setattr__(self, "branch_id", _require_id(self.branch_id, "route_assessment.branch_id"))
        object.__setattr__(self, "recording_id", _require_id(self.recording_id, "route_assessment.recording_id"))
        object.__setattr__(
            self,
            "predicted_route",
            _validate_preflight_route(self.predicted_route, "route_assessment.predicted_route"),
        )
        object.__setattr__(
            self,
            "reference_route",
            _validate_preflight_route(self.reference_route, "route_assessment.reference_route"),
        )
        object.__setattr__(
            self,
            "manual_override_applied",
            _require_bool(self.manual_override_applied, "route_assessment.manual_override_applied"),
        )

    @property
    def counted_in_benchmark(self) -> bool:
        return not self.manual_override_applied

    @property
    def serious_failure(self) -> bool:
        return (
            self.counted_in_benchmark
            and self.predicted_route == "confident_pipeline"
            and self.reference_route != "confident_pipeline"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "corpus_id": self.corpus_id,
            "counted_in_benchmark": self.counted_in_benchmark,
            "manual_override_applied": self.manual_override_applied,
            "predicted_route": self.predicted_route,
            "recording_id": self.recording_id,
            "reference_route": self.reference_route,
            "serious_failure": self.serious_failure,
        }


@dataclass(frozen=True)
class PreflightRouteConfusionReport:
    """Route-level confusion table for benchmark and release records."""

    assessments: tuple[PreflightRouteAssessment, ...]

    def __post_init__(self) -> None:
        assessments = _tuple_of(
            self.assessments,
            PreflightRouteAssessment,
            "route_confusion.assessments",
        )
        if not assessments:
            raise ValidationError("route_confusion.assessments is required")
        seen: set[tuple[str, str, str]] = set()
        for assessment in assessments:
            key = (assessment.corpus_id, assessment.branch_id, assessment.recording_id)
            if key in seen:
                raise ValidationError(
                    "route_confusion.assessments contains duplicate recording scope: "
                    f"{assessment.corpus_id}/{assessment.branch_id}/{assessment.recording_id}"
                )
            seen.add(key)
        object.__setattr__(self, "assessments", assessments)

    @property
    def counted_assessments(self) -> tuple[PreflightRouteAssessment, ...]:
        return tuple(assessment for assessment in self.assessments if assessment.counted_in_benchmark)

    @property
    def manual_override_count(self) -> int:
        return sum(1 for assessment in self.assessments if assessment.manual_override_applied)

    @property
    def serious_failure_count(self) -> int:
        return sum(1 for assessment in self.assessments if assessment.serious_failure)

    @property
    def out_of_scope_false_confident_count(self) -> int:
        return self.serious_failure_count

    @property
    def passed(self) -> bool:
        return self.serious_failure_count == 0

    @property
    def matrix(self) -> dict[str, dict[str, int]]:
        result = {
            reference_route: {predicted_route: 0 for predicted_route in ALLOWED_PREFLIGHT_ROUTES}
            for reference_route in ALLOWED_PREFLIGHT_ROUTES
        }
        for assessment in self.counted_assessments:
            result[assessment.reference_route][assessment.predicted_route] += 1
        return {route: dict(sorted(counts.items())) for route, counts in sorted(result.items())}

    def to_dict(self) -> dict[str, Any]:
        return {
            "assessments": [assessment.to_dict() for assessment in self.assessments],
            "counted_assessment_count": len(self.counted_assessments),
            "manual_override_count": self.manual_override_count,
            "matrix": self.matrix,
            "out_of_scope_false_confident_count": self.out_of_scope_false_confident_count,
            "passed": self.passed,
            "serious_failure_count": self.serious_failure_count,
        }


def route_preflight(policy: PreflightPolicy, features: PreflightFeatures) -> PreflightRouteDecision:
    """Route one call using only candidate-visible features and policy thresholds."""

    if not isinstance(policy, PreflightPolicy):
        raise ValidationError("policy must be PreflightPolicy")
    if not isinstance(features, PreflightFeatures):
        raise ValidationError("features must be PreflightFeatures")

    unsupported_reasons: list[str] = []
    if features.declared_locale not in policy.supported_locales:
        unsupported_reasons.append("unsupported_locale")
    if features.source not in policy.supported_sources:
        unsupported_reasons.append("unsupported_source")
    if features.capture_mode not in policy.supported_capture_modes:
        unsupported_reasons.append("unsupported_capture_mode")
    if features.channel_count not in policy.supported_channel_counts:
        unsupported_reasons.append("unsupported_channel_count")
    if features.codec not in policy.supported_codecs:
        unsupported_reasons.append("unsupported_codec")
    if features.duration_ms < policy.min_duration_ms:
        unsupported_reasons.append("duration_below_minimum")
    if features.duration_ms > policy.max_duration_ms:
        unsupported_reasons.append("duration_above_maximum")
    if features.sample_rate_hz < policy.min_sample_rate_hz:
        unsupported_reasons.append("sample_rate_below_minimum")
    if unsupported_reasons:
        return _decision(policy, features, "unsupported", tuple(unsupported_reasons))

    if features.capture_mode not in policy.confident_capture_modes:
        return _decision(policy, features, "diagnostic_only", ("capture_mode_outside_confident_scope",))

    review_reasons: list[str] = []
    if policy.require_speaker_count_hint_for_confident and features.speaker_count_hint is None:
        review_reasons.append("speaker_count_hint_unknown")
    elif features.speaker_count_hint is not None and not (
        policy.min_confident_speaker_count_hint
        <= features.speaker_count_hint
        <= policy.max_confident_speaker_count_hint
    ):
        review_reasons.append("speaker_count_hint_outside_confident_scope")
    if features.clipping_estimate > policy.max_confident_clipping_estimate:
        review_reasons.append("clipping_above_confident_threshold")
    if features.speech_ratio < policy.min_confident_speech_ratio:
        review_reasons.append("speech_ratio_below_confident_threshold")
    if features.rough_overlap_estimate > policy.max_confident_rough_overlap_estimate:
        review_reasons.append("rough_overlap_above_confident_threshold")
    if review_reasons:
        return _decision(policy, features, "needs_review", tuple(review_reasons))

    return _decision(policy, features, "confident_pipeline", ())


def render_transcript_for_preflight(
    recording: Any,
    decision: PreflightRouteDecision,
    *,
    overlays: tuple[TranscriptOverlay, ...] = (),
    label_source: str = "diarization_cluster",
    max_gap_ms: int = 900,
    split_after_punctuation: bool = True,
    min_speaker_confidence: float = 0.5,
) -> RenderedTranscript:
    """Render route-aware output while preventing non-confident speaker-label promotion."""

    if not isinstance(decision, PreflightRouteDecision):
        raise ValidationError("decision must be PreflightRouteDecision")
    review_reasons = ("speaker_attribution_unavailable",) if decision.route == "needs_review" else ()
    degraded_state = None if decision.route == "confident_pipeline" else decision.route
    return render_transcript(
        recording,
        overlays=overlays,
        label_source=label_source,  # type: ignore[arg-type]
        max_gap_ms=max_gap_ms,
        split_after_punctuation=split_after_punctuation,
        degraded_state=degraded_state,
        review_reasons=review_reasons,  # type: ignore[arg-type]
        min_speaker_confidence=min_speaker_confidence,
    )


def preflight_features_from_dict(payload: object) -> PreflightFeatures:
    """Parse candidate-visible feature payloads while rejecting hidden identity inputs."""

    data = _mapping(payload, "preflight")
    _reject_reference_identity_fields(data, "preflight")
    _reject_unknown_fields(
        data,
        {
            "capture_mode",
            "channel_count",
            "clipping_estimate",
            "codec",
            "declared_locale",
            "duration_ms",
            "rough_overlap_estimate",
            "sample_rate_hz",
            "source",
            "speaker_count_hint",
            "speech_ratio",
        },
        "preflight",
    )
    return PreflightFeatures(
        declared_locale=_required(data, "declared_locale", "preflight"),
        source=_required(data, "source", "preflight"),
        capture_mode=_required(data, "capture_mode", "preflight"),
        channel_count=_required(data, "channel_count", "preflight"),
        duration_ms=_required(data, "duration_ms", "preflight"),
        sample_rate_hz=_required(data, "sample_rate_hz", "preflight"),
        codec=_required(data, "codec", "preflight"),
        clipping_estimate=_required(data, "clipping_estimate", "preflight"),
        speech_ratio=_required(data, "speech_ratio", "preflight"),
        rough_overlap_estimate=_required(data, "rough_overlap_estimate", "preflight"),
        speaker_count_hint=data.get("speaker_count_hint"),
    )


def preflight_decision_from_dict(payload: object) -> PreflightRouteDecision:
    data = _mapping(payload, "preflight_decision")
    _reject_unknown_fields(
        data,
        {
            "accepted_for_pipeline",
            "features",
            "frozen_git_sha",
            "policy_id",
            "policy_version",
            "reasons",
            "route",
            "tuned_on_splits",
            "validated_on_splits",
        },
        "preflight_decision",
    )
    decision = PreflightRouteDecision(
        route=_required(data, "route", "preflight_decision"),
        reasons=tuple(_sequence(_required(data, "reasons", "preflight_decision"), "preflight_decision.reasons")),
        policy_id=_required(data, "policy_id", "preflight_decision"),
        policy_version=_required(data, "policy_version", "preflight_decision"),
        frozen_git_sha=_required(data, "frozen_git_sha", "preflight_decision"),
        tuned_on_splits=tuple(
            _sequence(_required(data, "tuned_on_splits", "preflight_decision"), "preflight_decision.tuned_on_splits")
        ),
        validated_on_splits=tuple(
            _sequence(
                _required(data, "validated_on_splits", "preflight_decision"),
                "preflight_decision.validated_on_splits",
            )
        ),
        features=preflight_features_from_dict(_required(data, "features", "preflight_decision")),
    )
    if data.get("accepted_for_pipeline") != decision.accepted_for_pipeline:
        raise ValidationError("preflight_decision.accepted_for_pipeline must match route")
    return decision


def preflight_job_record_from_dict(payload: object) -> PreflightJobRecord:
    data = _mapping(payload, "preflight_job")
    _reject_unknown_fields(
        data,
        {
            "effective_route",
            "features",
            "frozen_git_sha",
            "job_id",
            "manual_override",
            "manual_override_applied",
            "policy_id",
            "policy_version",
            "reasons",
            "route",
            "tuned_on_splits",
            "validated_launch_scope_version",
            "validated_on_splits",
        },
        "preflight_job",
    )
    decision = PreflightRouteDecision(
        route=_required(data, "route", "preflight_job"),
        reasons=tuple(_sequence(_required(data, "reasons", "preflight_job"), "preflight_job.reasons")),
        policy_id=_required(data, "policy_id", "preflight_job"),
        policy_version=_required(data, "policy_version", "preflight_job"),
        frozen_git_sha=_required(data, "frozen_git_sha", "preflight_job"),
        tuned_on_splits=tuple(
            _sequence(_required(data, "tuned_on_splits", "preflight_job"), "preflight_job.tuned_on_splits")
        ),
        validated_on_splits=tuple(
            _sequence(_required(data, "validated_on_splits", "preflight_job"), "preflight_job.validated_on_splits")
        ),
        features=preflight_features_from_dict(_required(data, "features", "preflight_job")),
    )
    manual_override = data.get("manual_override")
    record = PreflightJobRecord(
        job_id=_required(data, "job_id", "preflight_job"),
        decision=decision,
        validated_launch_scope_version=_required(data, "validated_launch_scope_version", "preflight_job"),
        manual_override=(
            None if manual_override is None else preflight_manual_override_audit_from_dict(manual_override)
        ),
    )
    if data.get("manual_override_applied") != record.manual_override_applied:
        raise ValidationError("preflight_job.manual_override_applied must match manual_override")
    if data.get("effective_route") != record.effective_route:
        raise ValidationError("preflight_job.effective_route must match route and manual_override")
    return record


def preflight_manual_override_audit_from_dict(payload: object) -> PreflightManualOverrideAudit:
    data = _mapping(payload, "preflight_override")
    _reject_unknown_fields(
        data,
        {
            "actor_id",
            "created_at",
            "excluded_from_benchmark_truth",
            "override_id",
            "override_route",
            "reason",
        },
        "preflight_override",
    )
    return PreflightManualOverrideAudit(
        override_id=_required(data, "override_id", "preflight_override"),
        actor_id=_required(data, "actor_id", "preflight_override"),
        reason=_required(data, "reason", "preflight_override"),
        override_route=_required(data, "override_route", "preflight_override"),
        created_at=_required(data, "created_at", "preflight_override"),
        excluded_from_benchmark_truth=_required(data, "excluded_from_benchmark_truth", "preflight_override"),
    )


def build_preflight_route_confusion_report(
    assessments: tuple[PreflightRouteAssessment, ...],
) -> PreflightRouteConfusionReport:
    return PreflightRouteConfusionReport(assessments=assessments)


def preflight_route_confusion_report_from_dict(payload: object) -> PreflightRouteConfusionReport:
    data = _mapping(payload, "route_confusion")
    _reject_unknown_fields(
        data,
        {
            "assessments",
            "counted_assessment_count",
            "manual_override_count",
            "matrix",
            "out_of_scope_false_confident_count",
            "passed",
            "serious_failure_count",
        },
        "route_confusion",
    )
    report = PreflightRouteConfusionReport(
        assessments=tuple(
            _route_assessment_from_dict(item)
            for item in _sequence(_required(data, "assessments", "route_confusion"), "route_confusion.assessments")
        )
    )
    if data.get("counted_assessment_count") != len(report.counted_assessments):
        raise ValidationError("route_confusion.counted_assessment_count must match assessments")
    if data.get("manual_override_count") != report.manual_override_count:
        raise ValidationError("route_confusion.manual_override_count must match assessments")
    if data.get("serious_failure_count") != report.serious_failure_count:
        raise ValidationError("route_confusion.serious_failure_count must match assessments")
    if data.get("out_of_scope_false_confident_count") != report.out_of_scope_false_confident_count:
        raise ValidationError("route_confusion.out_of_scope_false_confident_count must match assessments")
    if data.get("passed") != report.passed:
        raise ValidationError("route_confusion.passed must match serious failures")
    if data.get("matrix") != report.matrix:
        raise ValidationError("route_confusion.matrix must match assessments")
    return report


def _decision(
    policy: PreflightPolicy,
    features: PreflightFeatures,
    route: PreflightRoute,
    reasons: tuple[str, ...],
) -> PreflightRouteDecision:
    return PreflightRouteDecision(
        route=route,
        reasons=reasons,
        policy_id=policy.policy_id,
        policy_version=policy.version,
        frozen_git_sha=policy.frozen_git_sha,
        tuned_on_splits=policy.tuned_on_splits,
        validated_on_splits=policy.validated_on_splits,
        features=features,
    )


def _route_assessment_from_dict(payload: object) -> PreflightRouteAssessment:
    data = _mapping(payload, "route_assessment")
    _reject_unknown_fields(
        data,
        {
            "branch_id",
            "corpus_id",
            "counted_in_benchmark",
            "manual_override_applied",
            "predicted_route",
            "recording_id",
            "reference_route",
            "serious_failure",
        },
        "route_assessment",
    )
    assessment = PreflightRouteAssessment(
        corpus_id=_required(data, "corpus_id", "route_assessment"),
        branch_id=_required(data, "branch_id", "route_assessment"),
        recording_id=_required(data, "recording_id", "route_assessment"),
        predicted_route=_required(data, "predicted_route", "route_assessment"),
        reference_route=_required(data, "reference_route", "route_assessment"),
        manual_override_applied=data.get("manual_override_applied", False),
    )
    if data.get("counted_in_benchmark") != assessment.counted_in_benchmark:
        raise ValidationError("route_assessment.counted_in_benchmark must match manual_override_applied")
    if data.get("serious_failure") != assessment.serious_failure:
        raise ValidationError("route_assessment.serious_failure must match routes")
    return assessment


def _mapping(payload: object, field_name: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValidationError(f"{field_name} must be an object")
    return dict(payload)


def _reject_reference_identity_fields(data: dict[str, Any], field_name: str) -> None:
    for key in data:
        if key in _REFERENCE_IDENTITY_KEYS:
            raise ValidationError(f"{field_name}.{key} must remain candidate-invisible")


def _reject_unknown_fields(data: dict[str, Any], allowed_fields: set[str], field_name: str) -> None:
    unknown = sorted(set(data) - allowed_fields)
    if unknown:
        raise ValidationError(f"{field_name} has unsupported fields: {', '.join(unknown)}")


def _required(data: dict[str, Any], key: str, field_name: str) -> Any:
    try:
        return data[key]
    except KeyError as exc:
        raise ValidationError(f"{field_name}.{key} is required") from exc


def _sequence(value: object, field_name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValidationError(f"{field_name} must be an array")
    return tuple(value)


def _id_tuple(values: object, field_name: str) -> tuple[str, ...]:
    result = tuple(_require_id(item, field_name) for item in _sequence(values, field_name))
    _reject_duplicates(result, field_name)
    return result


def _non_empty_id_tuple(values: object, field_name: str) -> tuple[str, ...]:
    result = _id_tuple(values, field_name)
    if not result:
        raise ValidationError(f"{field_name} is required")
    return result


def _reject_duplicates(values: tuple[Any, ...], field_name: str) -> None:
    seen: set[Any] = set()
    for value in values:
        if value in seen:
            raise ValidationError(f"{field_name} contains duplicate value: {value}")
        seen.add(value)


def _validate_capture_mode(value: object, field_name: str) -> PreflightCaptureMode:
    mode = _require_id(value, field_name)
    if mode not in ALLOWED_PREFLIGHT_CAPTURE_MODES:
        raise ValidationError(f"{field_name} is not supported: {mode}")
    return mode  # type: ignore[return-value]


def _validate_preflight_route(value: object, field_name: str) -> PreflightRoute:
    route = _require_id(value, field_name)
    if route not in ALLOWED_PREFLIGHT_ROUTES:
        raise ValidationError(f"{field_name} is not supported: {route}")
    return route  # type: ignore[return-value]


def _validate_frozen_git_sha(value: object, field_name: str) -> str:
    sha = _require_id(value, field_name)
    if len(sha) != 40 or any(char not in _HEX_DIGITS for char in sha):
        raise ValidationError(f"{field_name} must be a frozen 40-character git SHA")
    return sha


def _is_holdout_split(value: str) -> bool:
    split_id = value.lower()
    return "holdout" in split_id or "acceptance" in split_id


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValidationError(f"{field_name} must be > 0")
    return value


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValidationError(f"{field_name} must be a boolean")
    return value


def _tuple_of(values: object, item_type: type[Any], field_name: str) -> tuple[Any, ...]:
    result = _sequence(values, field_name)
    for index, item in enumerate(result):
        if not isinstance(item, item_type):
            raise ValidationError(f"{field_name}[{index}] must be {item_type.__name__}")
    return result


def _probability(value: object, field_name: str) -> float:
    result = _finite_number(value, field_name)
    if result < 0.0 or result > 1.0:
        raise ValidationError(f"{field_name} must be between 0 and 1")
    return result


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{field_name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValidationError(f"{field_name} must be finite")
    return result


def _require_id(value: object, field_name: str) -> str:
    if value is None:
        raise ValidationError(f"{field_name} is required")
    if not isinstance(value, str):
        raise ValidationError(f"{field_name} must be a string")
    value = value.strip()
    if not value:
        raise ValidationError(f"{field_name} is required")
    return value


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{field_name} must be a string")
    if not value.strip():
        raise ValidationError(f"{field_name} is required")
    return value
