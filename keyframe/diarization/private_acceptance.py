"""Private in-domain acceptance annotation protocol metadata.

The objects in this module describe private acceptance-set rules and aggregate
annotation quality. They intentionally avoid reference speaker IDs, voice
profiles, or any cross-call identity material.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from keyframe.diarization.models import ValidationError


PRIVATE_ACCEPTANCE_METADATA_SCHEMA_VERSION = 1

PrivateAcceptanceLabel = Literal["adjudicated", "unadjudicated_diagnostic", "reference_unstable", "no_score"]
PrivateAnnotationQualityGateStatus = Literal["passed", "failed", "unavailable"]
PrivateAcceptanceCoverageStatus = Literal["sufficient", "insufficient_acceptance_coverage", "diagnostic_only"]

ALLOWED_PRIVATE_ACCEPTANCE_LABELS = frozenset(
    {"adjudicated", "unadjudicated_diagnostic", "reference_unstable", "no_score"}
)
PRIVATE_ANNOTATION_QUALITY_GATE_STAGE = "annotation_quality_pre_model"

_REFERENCE_IDENTITY_KEYS = frozenset(
    {
        "corpus_speaker_id",
        "corpus_speaker_ids",
        "cross_recording_identity",
        "global_identity",
        "participant_id",
        "participant_ids",
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


@dataclass(frozen=True)
class PrivateAnnotationProtocol:
    """Versioned annotation rules for private in-domain acceptance sets."""

    protocol_id: str
    version: str
    transcript_normalization: str
    speaker_span_rules: tuple[str, ...]
    overlap_rules: tuple[str, ...]
    unintelligible_no_score_rules: tuple[str, ...]
    critical_span_label_rules: tuple[str, ...]
    reference_speaker_id_policy: Literal["candidate_invisible"] = "candidate_invisible"

    def __post_init__(self) -> None:
        object.__setattr__(self, "protocol_id", _require_id(self.protocol_id, "private_protocol.protocol_id"))
        object.__setattr__(self, "version", _require_id(self.version, "private_protocol.version"))
        object.__setattr__(
            self,
            "transcript_normalization",
            _require_text(self.transcript_normalization, "private_protocol.transcript_normalization"),
        )
        object.__setattr__(
            self,
            "speaker_span_rules",
            _non_empty_text_tuple(self.speaker_span_rules, "private_protocol.speaker_span_rules"),
        )
        object.__setattr__(
            self,
            "overlap_rules",
            _non_empty_text_tuple(self.overlap_rules, "private_protocol.overlap_rules"),
        )
        object.__setattr__(
            self,
            "unintelligible_no_score_rules",
            _non_empty_text_tuple(
                self.unintelligible_no_score_rules,
                "private_protocol.unintelligible_no_score_rules",
            ),
        )
        object.__setattr__(
            self,
            "critical_span_label_rules",
            _non_empty_text_tuple(self.critical_span_label_rules, "private_protocol.critical_span_label_rules"),
        )
        policy = _require_id(self.reference_speaker_id_policy, "private_protocol.reference_speaker_id_policy")
        if policy != "candidate_invisible":
            raise ValidationError("private_protocol.reference_speaker_id_policy must be candidate_invisible")
        object.__setattr__(self, "reference_speaker_id_policy", policy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "critical_span_label_rules": list(self.critical_span_label_rules),
            "overlap_rules": list(self.overlap_rules),
            "protocol_id": self.protocol_id,
            "reference_speaker_id_policy": self.reference_speaker_id_policy,
            "speaker_span_rules": list(self.speaker_span_rules),
            "transcript_normalization": self.transcript_normalization,
            "unintelligible_no_score_rules": list(self.unintelligible_no_score_rules),
            "version": self.version,
        }


@dataclass(frozen=True)
class PrivateAcceptanceSlice:
    """Aggregate annotation status for one private acceptance slice."""

    slice_id: str
    label: PrivateAcceptanceLabel
    recording_count: int
    duration_ms: int
    no_score_region_count: int = 0
    critical_span_count: int = 0
    reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "slice_id", _require_id(self.slice_id, "private_slice.slice_id"))
        object.__setattr__(self, "label", _validate_label(self.label))
        object.__setattr__(
            self,
            "recording_count",
            _non_negative_int(self.recording_count, "private_slice.recording_count"),
        )
        object.__setattr__(self, "duration_ms", _non_negative_int(self.duration_ms, "private_slice.duration_ms"))
        object.__setattr__(
            self,
            "no_score_region_count",
            _non_negative_int(self.no_score_region_count, "private_slice.no_score_region_count"),
        )
        object.__setattr__(
            self,
            "critical_span_count",
            _non_negative_int(self.critical_span_count, "private_slice.critical_span_count"),
        )
        object.__setattr__(self, "reason", _optional_text(self.reason, "private_slice.reason"))
        if self.label == "no_score" and self.no_score_region_count == 0:
            raise ValidationError("no_score private acceptance slices require no_score_region_count")
        if self.label == "reference_unstable" and self.reason is None:
            raise ValidationError("reference_unstable private acceptance slices require a reason")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "critical_span_count": self.critical_span_count,
            "duration_ms": self.duration_ms,
            "label": self.label,
            "no_score_region_count": self.no_score_region_count,
            "recording_count": self.recording_count,
            "slice_id": self.slice_id,
        }
        if self.reason is not None:
            payload["reason"] = self.reason
        return payload


@dataclass(frozen=True)
class PrivateAnnotationQualityMetrics:
    """Aggregate annotation quality metrics evaluated before model gates."""

    annotated_recording_count: int
    double_annotated_recording_count: int
    double_annotated_sample_rate: float
    agreement_metrics: dict[str, float]
    adjudication_change_rate: float
    unresolved_disagreement_rate: float

    def __post_init__(self) -> None:
        annotated = _non_negative_int(
            self.annotated_recording_count,
            "private_quality.annotated_recording_count",
        )
        double_annotated = _non_negative_int(
            self.double_annotated_recording_count,
            "private_quality.double_annotated_recording_count",
        )
        if double_annotated > annotated:
            raise ValidationError(
                "private_quality.double_annotated_recording_count cannot exceed annotated_recording_count"
            )
        object.__setattr__(self, "annotated_recording_count", annotated)
        object.__setattr__(self, "double_annotated_recording_count", double_annotated)
        sample_rate = _probability(self.double_annotated_sample_rate, "private_quality.double_annotated_sample_rate")
        expected_rate = 0.0 if annotated == 0 else double_annotated / annotated
        if not math.isclose(sample_rate, expected_rate, rel_tol=1e-9, abs_tol=1e-9):
            raise ValidationError(
                "private_quality.double_annotated_sample_rate must match "
                "double_annotated_recording_count / annotated_recording_count"
            )
        object.__setattr__(self, "double_annotated_sample_rate", sample_rate)
        object.__setattr__(
            self,
            "agreement_metrics",
            _probability_map(self.agreement_metrics, "private_quality.agreement_metrics"),
        )
        object.__setattr__(
            self,
            "adjudication_change_rate",
            _probability(self.adjudication_change_rate, "private_quality.adjudication_change_rate"),
        )
        object.__setattr__(
            self,
            "unresolved_disagreement_rate",
            _probability(self.unresolved_disagreement_rate, "private_quality.unresolved_disagreement_rate"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "adjudication_change_rate": self.adjudication_change_rate,
            "agreement_metrics": dict(sorted(self.agreement_metrics.items())),
            "annotated_recording_count": self.annotated_recording_count,
            "double_annotated_recording_count": self.double_annotated_recording_count,
            "double_annotated_sample_rate": self.double_annotated_sample_rate,
            "unresolved_disagreement_rate": self.unresolved_disagreement_rate,
        }


@dataclass(frozen=True)
class PrivateAnnotationQualityGateConfig:
    """Thresholds for blocking model gates until annotation quality is ready."""

    min_double_annotated_sample_rate: float = 0.2
    min_agreement_metrics: dict[str, float] = field(
        default_factory=lambda: {
            "overlap_agreement": 0.85,
            "speaker_span_agreement": 0.90,
            "transcript_normalization_agreement": 0.95,
        }
    )
    max_adjudication_change_rate: float = 0.15
    max_unresolved_disagreement_rate: float = 0.02

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "min_double_annotated_sample_rate",
            _probability(
                self.min_double_annotated_sample_rate,
                "private_quality_gate.min_double_annotated_sample_rate",
            ),
        )
        object.__setattr__(
            self,
            "min_agreement_metrics",
            _probability_map(self.min_agreement_metrics, "private_quality_gate.min_agreement_metrics"),
        )
        object.__setattr__(
            self,
            "max_adjudication_change_rate",
            _probability(self.max_adjudication_change_rate, "private_quality_gate.max_adjudication_change_rate"),
        )
        object.__setattr__(
            self,
            "max_unresolved_disagreement_rate",
            _probability(
                self.max_unresolved_disagreement_rate,
                "private_quality_gate.max_unresolved_disagreement_rate",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_adjudication_change_rate": self.max_adjudication_change_rate,
            "max_unresolved_disagreement_rate": self.max_unresolved_disagreement_rate,
            "min_agreement_metrics": dict(sorted(self.min_agreement_metrics.items())),
            "min_double_annotated_sample_rate": self.min_double_annotated_sample_rate,
        }


@dataclass(frozen=True)
class PrivateAnnotationQualityGateResult:
    """Pre-model gate result for private annotation quality."""

    status: PrivateAnnotationQualityGateStatus
    reasons: tuple[str, ...]
    metrics: PrivateAnnotationQualityMetrics | None
    gate_config: PrivateAnnotationQualityGateConfig
    gate_stage: str = PRIVATE_ANNOTATION_QUALITY_GATE_STAGE

    def __post_init__(self) -> None:
        status = _require_id(self.status, "private_quality_gate_result.status")
        if status not in {"passed", "failed", "unavailable"}:
            raise ValidationError(f"private_quality_gate_result.status is not supported: {status}")
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "reasons",
            tuple(_require_text(reason, "private_quality_gate_result.reasons") for reason in _sequence(self.reasons)),
        )
        if self.status == "passed" and self.reasons:
            raise ValidationError("passed private annotation quality gates cannot include reasons")
        if self.status in {"failed", "unavailable"} and not self.reasons:
            raise ValidationError("failed or unavailable private annotation quality gates require reasons")
        if self.metrics is not None and not isinstance(self.metrics, PrivateAnnotationQualityMetrics):
            raise ValidationError("private_quality_gate_result.metrics must be PrivateAnnotationQualityMetrics")
        if not isinstance(self.gate_config, PrivateAnnotationQualityGateConfig):
            raise ValidationError("private_quality_gate_result.gate_config must be PrivateAnnotationQualityGateConfig")
        if self.gate_stage != PRIVATE_ANNOTATION_QUALITY_GATE_STAGE:
            raise ValidationError("private_quality_gate_result.gate_stage must be annotation_quality_pre_model")

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    @property
    def blocks_model_gates(self) -> bool:
        return not self.passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "blocks_model_gates": self.blocks_model_gates,
            "gate_config": self.gate_config.to_dict(),
            "gate_stage": self.gate_stage,
            "metrics": self.metrics.to_dict() if self.metrics is not None else None,
            "reasons": list(self.reasons),
            "status": self.status,
        }


@dataclass(frozen=True)
class PrivateAcceptanceCoverageSliceTarget:
    """Minimum private acceptance coverage for one validated launch-scope slice."""

    slice_id: str
    capture_modes: tuple[str, ...]
    speaker_count_buckets: tuple[str, ...]
    duration_buckets: tuple[str, ...]
    overlap_ratio_buckets: tuple[str, ...]
    audio_quality_buckets: tuple[str, ...]
    platform_sources: tuple[str, ...]
    language_accent_domains: tuple[str, ...]
    min_scored_recording_count: int
    min_scored_duration_ms: int
    required: bool = True
    diagnostic_only: bool = False
    description: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "slice_id", _require_id(self.slice_id, "coverage_target.slice_id"))
        for field_name in (
            "capture_modes",
            "speaker_count_buckets",
            "duration_buckets",
            "overlap_ratio_buckets",
            "audio_quality_buckets",
            "platform_sources",
            "language_accent_domains",
        ):
            object.__setattr__(
                self,
                field_name,
                _non_empty_id_tuple(getattr(self, field_name), f"coverage_target.{field_name}"),
            )
        object.__setattr__(
            self,
            "min_scored_recording_count",
            _positive_int(self.min_scored_recording_count, "coverage_target.min_scored_recording_count"),
        )
        object.__setattr__(
            self,
            "min_scored_duration_ms",
            _positive_int(self.min_scored_duration_ms, "coverage_target.min_scored_duration_ms"),
        )
        object.__setattr__(self, "required", _require_bool(self.required, "coverage_target.required"))
        object.__setattr__(
            self,
            "diagnostic_only",
            _require_bool(self.diagnostic_only, "coverage_target.diagnostic_only"),
        )
        object.__setattr__(self, "description", _optional_text(self.description, "coverage_target.description"))
        if self.required and self.diagnostic_only:
            raise ValidationError("coverage_target cannot be both required and diagnostic_only")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "audio_quality_buckets": list(self.audio_quality_buckets),
            "capture_modes": list(self.capture_modes),
            "diagnostic_only": self.diagnostic_only,
            "duration_buckets": list(self.duration_buckets),
            "language_accent_domains": list(self.language_accent_domains),
            "min_scored_duration_ms": self.min_scored_duration_ms,
            "min_scored_recording_count": self.min_scored_recording_count,
            "overlap_ratio_buckets": list(self.overlap_ratio_buckets),
            "platform_sources": list(self.platform_sources),
            "required": self.required,
            "slice_id": self.slice_id,
            "speaker_count_buckets": list(self.speaker_count_buckets),
        }
        if self.description is not None:
            payload["description"] = self.description
        return payload


@dataclass(frozen=True)
class PrivateAcceptanceCoverageObservation:
    """Observed scored support for one private acceptance coverage slice."""

    slice_id: str
    scored_recording_count: int
    scored_duration_ms: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "slice_id", _require_id(self.slice_id, "coverage_observation.slice_id"))
        object.__setattr__(
            self,
            "scored_recording_count",
            _non_negative_int(self.scored_recording_count, "coverage_observation.scored_recording_count"),
        )
        object.__setattr__(
            self,
            "scored_duration_ms",
            _non_negative_int(self.scored_duration_ms, "coverage_observation.scored_duration_ms"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scored_duration_ms": self.scored_duration_ms,
            "scored_recording_count": self.scored_recording_count,
            "slice_id": self.slice_id,
        }


@dataclass(frozen=True)
class PrivateAcceptanceCoveragePlan:
    """Versioned private sampling and launch-scope coverage targets."""

    plan_id: str
    version: str
    targets: tuple[PrivateAcceptanceCoverageSliceTarget, ...]
    validated_scope: tuple[str, ...] = ()
    unsupported_scope: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_id", _require_id(self.plan_id, "coverage_plan.plan_id"))
        object.__setattr__(self, "version", _require_id(self.version, "coverage_plan.version"))
        targets = _tuple_of(self.targets, PrivateAcceptanceCoverageSliceTarget, "coverage_plan.targets")
        if not targets:
            raise ValidationError("coverage_plan.targets is required")
        _unique_ids(tuple(target.slice_id for target in targets), "coverage_plan.targets.slice_id")
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "validated_scope", _id_tuple(self.validated_scope, "coverage_plan.validated_scope"))
        object.__setattr__(
            self,
            "unsupported_scope",
            _id_tuple(self.unsupported_scope, "coverage_plan.unsupported_scope"),
        )
        overlap = set(self.validated_scope) & set(self.unsupported_scope)
        if overlap:
            raise ValidationError(f"coverage_plan scope cannot be both validated and unsupported: {sorted(overlap)[0]}")
        required_ids = {target.slice_id for target in targets if target.required}
        unknown_scope = (set(self.validated_scope) | set(self.unsupported_scope)) - required_ids
        if unknown_scope:
            raise ValidationError(f"coverage_plan scope references unknown required slice: {sorted(unknown_scope)[0]}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "targets": [target.to_dict() for target in self.targets],
            "unsupported_scope": list(self.unsupported_scope),
            "validated_scope": list(self.validated_scope),
            "version": self.version,
        }


@dataclass(frozen=True)
class PrivateAcceptanceCoverageSliceResult:
    """Coverage status for one launch-scope slice."""

    slice_id: str
    status: PrivateAcceptanceCoverageStatus
    required: bool
    diagnostic_only: bool
    scored_recording_count: int
    scored_duration_ms: int
    min_scored_recording_count: int
    min_scored_duration_ms: int
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "slice_id", _require_id(self.slice_id, "coverage_result.slice_id"))
        object.__setattr__(self, "status", _validate_coverage_status(self.status, "coverage_result.status"))
        object.__setattr__(self, "required", _require_bool(self.required, "coverage_result.required"))
        object.__setattr__(
            self,
            "diagnostic_only",
            _require_bool(self.diagnostic_only, "coverage_result.diagnostic_only"),
        )
        for field_name in (
            "scored_recording_count",
            "scored_duration_ms",
            "min_scored_recording_count",
            "min_scored_duration_ms",
        ):
            object.__setattr__(
                self,
                field_name,
                _non_negative_int(getattr(self, field_name), f"coverage_result.{field_name}"),
            )
        reasons = tuple(_require_text(reason, "coverage_result.reasons") for reason in _sequence(self.reasons))
        object.__setattr__(self, "reasons", reasons)
        if self.diagnostic_only != (self.status == "diagnostic_only"):
            raise ValidationError("coverage_result.diagnostic_only must match diagnostic_only status")
        below_threshold = (
            self.scored_recording_count < self.min_scored_recording_count
            or self.scored_duration_ms < self.min_scored_duration_ms
        )
        if self.status == "sufficient" and (reasons or below_threshold):
            raise ValidationError("sufficient coverage results must meet minimum coverage without reasons")
        if self.status == "insufficient_acceptance_coverage" and not below_threshold:
            raise ValidationError("insufficient coverage results must be below minimum coverage")
        if self.status != "sufficient" and not reasons:
            raise ValidationError("insufficient or diagnostic coverage results require reasons")

    @property
    def passed(self) -> bool:
        return self.status in {"sufficient", "diagnostic_only"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "diagnostic_only": self.diagnostic_only,
            "min_scored_duration_ms": self.min_scored_duration_ms,
            "min_scored_recording_count": self.min_scored_recording_count,
            "passed": self.passed,
            "reasons": list(self.reasons),
            "required": self.required,
            "scored_duration_ms": self.scored_duration_ms,
            "scored_recording_count": self.scored_recording_count,
            "slice_id": self.slice_id,
            "status": self.status,
        }


@dataclass(frozen=True)
class PrivateAcceptanceCoverageReport:
    """Release-record-ready private acceptance coverage status."""

    plan_id: str
    plan_version: str
    status: PrivateAcceptanceCoverageStatus
    slice_results: tuple[PrivateAcceptanceCoverageSliceResult, ...]
    validated_scope: tuple[str, ...]
    unsupported_scope: tuple[str, ...]
    failure_code: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_id", _require_id(self.plan_id, "coverage_report.plan_id"))
        object.__setattr__(self, "plan_version", _require_id(self.plan_version, "coverage_report.plan_version"))
        object.__setattr__(self, "status", _validate_coverage_status(self.status, "coverage_report.status"))
        results = _tuple_of(self.slice_results, PrivateAcceptanceCoverageSliceResult, "coverage_report.slice_results")
        if not results:
            raise ValidationError("coverage_report.slice_results is required")
        result_ids = tuple(result.slice_id for result in results)
        _unique_ids(result_ids, "coverage_report.slice_results.slice_id")
        object.__setattr__(self, "slice_results", results)
        validated_scope = _id_tuple(self.validated_scope, "coverage_report.validated_scope")
        object.__setattr__(self, "validated_scope", validated_scope)
        unsupported_scope = _id_tuple(self.unsupported_scope, "coverage_report.unsupported_scope")
        object.__setattr__(
            self,
            "unsupported_scope",
            unsupported_scope,
        )
        object.__setattr__(self, "failure_code", _optional_id(self.failure_code, "coverage_report.failure_code"))
        expected_validated_scope = tuple(
            result.slice_id
            for result in results
            if result.required and result.status == "sufficient"
        )
        expected_unsupported_scope = tuple(
            result.slice_id
            for result in results
            if result.required and result.status == "insufficient_acceptance_coverage"
        )
        if validated_scope != expected_validated_scope:
            raise ValidationError("coverage_report.validated_scope must match sufficient required slices")
        if unsupported_scope != expected_unsupported_scope:
            raise ValidationError("coverage_report.unsupported_scope must match insufficient required slices")
        if self.status == "diagnostic_only":
            raise ValidationError("coverage_report.status cannot be diagnostic_only")
        has_required_insufficient = any(
            result.required and result.status == "insufficient_acceptance_coverage"
            for result in results
        )
        if self.status == "sufficient" and has_required_insufficient:
            raise ValidationError("sufficient coverage reports cannot include insufficient required slices")
        if self.status == "insufficient_acceptance_coverage" and not has_required_insufficient:
            raise ValidationError("insufficient coverage reports require an insufficient required slice")
        if (
            self.status == "insufficient_acceptance_coverage"
            and self.failure_code != "insufficient_acceptance_coverage"
        ):
            raise ValidationError("insufficient coverage reports require insufficient_acceptance_coverage failure_code")
        if self.status != "insufficient_acceptance_coverage" and self.failure_code is not None:
            raise ValidationError("sufficient coverage reports cannot include a failure_code")

    @property
    def passed(self) -> bool:
        return self.status != "insufficient_acceptance_coverage"

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_code": self.failure_code,
            "passed": self.passed,
            "plan_id": self.plan_id,
            "plan_version": self.plan_version,
            "slice_results": [result.to_dict() for result in self.slice_results],
            "status": self.status,
            "unsupported_scope": list(self.unsupported_scope),
            "validated_scope": list(self.validated_scope),
        }


@dataclass(frozen=True)
class PrivateAcceptanceMetadata:
    """Manifest-safe private acceptance metadata without private annotations."""

    metadata_id: str
    protocol: PrivateAnnotationProtocol
    slices: tuple[PrivateAcceptanceSlice, ...]
    quality_metrics: PrivateAnnotationQualityMetrics | None = None
    coverage_plan: PrivateAcceptanceCoveragePlan | None = None
    schema_version: int = PRIVATE_ACCEPTANCE_METADATA_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _validate_schema_version(self.schema_version))
        object.__setattr__(self, "metadata_id", _require_id(self.metadata_id, "private_acceptance.metadata_id"))
        if not isinstance(self.protocol, PrivateAnnotationProtocol):
            raise ValidationError("private_acceptance.protocol must be a PrivateAnnotationProtocol")
        slices = _tuple_of(self.slices, PrivateAcceptanceSlice, "private_acceptance.slices")
        if not slices:
            raise ValidationError("private_acceptance.slices is required")
        _unique_ids(tuple(item.slice_id for item in slices), "private_acceptance.slices.slice_id")
        object.__setattr__(self, "slices", slices)
        if self.quality_metrics is not None and not isinstance(self.quality_metrics, PrivateAnnotationQualityMetrics):
            raise ValidationError("private_acceptance.quality_metrics must be PrivateAnnotationQualityMetrics")
        if self.coverage_plan is not None and not isinstance(self.coverage_plan, PrivateAcceptanceCoveragePlan):
            raise ValidationError("private_acceptance.coverage_plan must be PrivateAcceptanceCoveragePlan")
        if self.coverage_plan is not None:
            allowed_slice_ids = {item.slice_id for item in slices if item.label == "adjudicated"}
            required_slice_ids = {target.slice_id for target in self.coverage_plan.targets if target.required}
            unknown_required = required_slice_ids - allowed_slice_ids
            if unknown_required:
                raise ValidationError(
                    f"private_acceptance.coverage_plan requires adjudicated slice: {sorted(unknown_required)[0]}"
                )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "metadata_id": self.metadata_id,
            "protocol": self.protocol.to_dict(),
            "schema_version": self.schema_version,
            "slices": [item.to_dict() for item in self.slices],
        }
        if self.quality_metrics is not None:
            payload["quality_metrics"] = self.quality_metrics.to_dict()
        if self.coverage_plan is not None:
            payload["coverage_plan"] = self.coverage_plan.to_dict()
        return payload


def evaluate_private_annotation_quality(
    metadata: PrivateAcceptanceMetadata,
    gate_config: PrivateAnnotationQualityGateConfig | None = None,
) -> PrivateAnnotationQualityGateResult:
    """Evaluate private annotation quality before any model gates run."""

    if not isinstance(metadata, PrivateAcceptanceMetadata):
        raise ValidationError("metadata must be PrivateAcceptanceMetadata")
    gate_config = gate_config or PrivateAnnotationQualityGateConfig()
    if not isinstance(gate_config, PrivateAnnotationQualityGateConfig):
        raise ValidationError("gate_config must be PrivateAnnotationQualityGateConfig")
    metrics = metadata.quality_metrics
    if metrics is None:
        return PrivateAnnotationQualityGateResult(
            status="unavailable",
            reasons=("private annotation quality metrics are unavailable",),
            metrics=None,
            gate_config=gate_config,
        )

    reasons: list[str] = []
    if metrics.double_annotated_sample_rate < gate_config.min_double_annotated_sample_rate:
        reasons.append("double_annotated_sample_rate below threshold")
    for metric_name, threshold in sorted(gate_config.min_agreement_metrics.items()):
        value = metrics.agreement_metrics.get(metric_name)
        if value is None:
            reasons.append(f"agreement metric missing: {metric_name}")
        elif value < threshold:
            reasons.append(f"agreement metric below threshold: {metric_name}")
    if metrics.adjudication_change_rate > gate_config.max_adjudication_change_rate:
        reasons.append("adjudication_change_rate above threshold")
    if metrics.unresolved_disagreement_rate > gate_config.max_unresolved_disagreement_rate:
        reasons.append("unresolved_disagreement_rate above threshold")

    return PrivateAnnotationQualityGateResult(
        status="failed" if reasons else "passed",
        reasons=tuple(reasons),
        metrics=metrics,
        gate_config=gate_config,
    )


def evaluate_private_acceptance_coverage(
    metadata: PrivateAcceptanceMetadata,
    observations: tuple[PrivateAcceptanceCoverageObservation, ...],
) -> PrivateAcceptanceCoverageReport:
    """Evaluate private acceptance coverage against the validated launch scope."""

    if not isinstance(metadata, PrivateAcceptanceMetadata):
        raise ValidationError("metadata must be PrivateAcceptanceMetadata")
    if metadata.coverage_plan is None:
        raise ValidationError("metadata.coverage_plan is required")
    plan = metadata.coverage_plan
    observations = _tuple_of(observations, PrivateAcceptanceCoverageObservation, "coverage_observations")
    target_ids = {target.slice_id for target in plan.targets}
    observed_by_slice: dict[str, PrivateAcceptanceCoverageObservation] = {}
    for observation in observations:
        if observation.slice_id not in target_ids:
            raise ValidationError(f"coverage observation references unknown slice_id: {observation.slice_id}")
        if observation.slice_id in observed_by_slice:
            raise ValidationError(f"duplicate coverage observation slice_id: {observation.slice_id}")
        observed_by_slice[observation.slice_id] = observation

    results = tuple(
        _coverage_result_for_target(target, observed_by_slice.get(target.slice_id))
        for target in plan.targets
    )
    has_insufficient = any(result.status == "insufficient_acceptance_coverage" for result in results if result.required)
    return PrivateAcceptanceCoverageReport(
        plan_id=plan.plan_id,
        plan_version=plan.version,
        status="insufficient_acceptance_coverage" if has_insufficient else "sufficient",
        slice_results=results,
        validated_scope=tuple(
            result.slice_id
            for result in results
            if result.required and result.status == "sufficient"
        ),
        unsupported_scope=tuple(
            result.slice_id
            for result in results
            if result.required and result.status == "insufficient_acceptance_coverage"
        ),
        failure_code="insufficient_acceptance_coverage" if has_insufficient else None,
    )


def private_acceptance_metadata_to_dict(metadata: PrivateAcceptanceMetadata) -> dict[str, Any]:
    if not isinstance(metadata, PrivateAcceptanceMetadata):
        raise ValidationError("metadata must be PrivateAcceptanceMetadata")
    return metadata.to_dict()


def private_acceptance_metadata_from_dict(payload: object) -> PrivateAcceptanceMetadata:
    data = _mapping(payload, "private_acceptance")
    _reject_reference_identity_fields(data, "private_acceptance")
    _reject_unknown_fields(
        data,
        {"coverage_plan", "metadata_id", "protocol", "quality_metrics", "schema_version", "slices"},
        "private_acceptance",
    )
    return PrivateAcceptanceMetadata(
        schema_version=_required(data, "schema_version", "private_acceptance"),
        metadata_id=_required(data, "metadata_id", "private_acceptance"),
        protocol=_protocol_from_dict(_required(data, "protocol", "private_acceptance")),
        slices=tuple(_slice_from_dict(item) for item in _sequence(_required(data, "slices", "private_acceptance"))),
        quality_metrics=(
            None
            if data.get("quality_metrics") is None
            else _quality_metrics_from_dict(data.get("quality_metrics"))
        ),
        coverage_plan=(
            None if data.get("coverage_plan") is None else _coverage_plan_from_dict(data.get("coverage_plan"))
        ),
    )


def private_acceptance_metadata_json_dumps(metadata: PrivateAcceptanceMetadata) -> str:
    payload = private_acceptance_metadata_to_dict(metadata)
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def private_acceptance_metadata_json_loads(text: str) -> PrivateAcceptanceMetadata:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"private acceptance metadata JSON is invalid: {exc.msg}") from exc
    return private_acceptance_metadata_from_dict(payload)


def read_private_acceptance_metadata_json(path: str | Path) -> PrivateAcceptanceMetadata:
    return private_acceptance_metadata_json_loads(Path(path).read_text(encoding="utf-8"))


def write_private_acceptance_metadata_json(path: str | Path, metadata: PrivateAcceptanceMetadata) -> None:
    Path(path).write_text(private_acceptance_metadata_json_dumps(metadata), encoding="utf-8", newline="\n")


def _protocol_from_dict(payload: object) -> PrivateAnnotationProtocol:
    data = _mapping(payload, "private_protocol")
    _reject_reference_identity_fields(data, "private_protocol")
    _reject_unknown_fields(
        data,
        {
            "critical_span_label_rules",
            "overlap_rules",
            "protocol_id",
            "reference_speaker_id_policy",
            "speaker_span_rules",
            "transcript_normalization",
            "unintelligible_no_score_rules",
            "version",
        },
        "private_protocol",
    )
    return PrivateAnnotationProtocol(
        protocol_id=_required(data, "protocol_id", "private_protocol"),
        version=_required(data, "version", "private_protocol"),
        transcript_normalization=_required(data, "transcript_normalization", "private_protocol"),
        speaker_span_rules=tuple(_sequence(_required(data, "speaker_span_rules", "private_protocol"))),
        overlap_rules=tuple(_sequence(_required(data, "overlap_rules", "private_protocol"))),
        unintelligible_no_score_rules=tuple(
            _sequence(_required(data, "unintelligible_no_score_rules", "private_protocol"))
        ),
        critical_span_label_rules=tuple(_sequence(_required(data, "critical_span_label_rules", "private_protocol"))),
        reference_speaker_id_policy=data.get("reference_speaker_id_policy", "candidate_invisible"),
    )


def _slice_from_dict(payload: object) -> PrivateAcceptanceSlice:
    data = _mapping(payload, "private_slice")
    _reject_reference_identity_fields(data, "private_slice")
    _reject_unknown_fields(
        data,
        {
            "critical_span_count",
            "duration_ms",
            "label",
            "no_score_region_count",
            "reason",
            "recording_count",
            "slice_id",
        },
        "private_slice",
    )
    return PrivateAcceptanceSlice(
        slice_id=_required(data, "slice_id", "private_slice"),
        label=_required(data, "label", "private_slice"),
        recording_count=_required(data, "recording_count", "private_slice"),
        duration_ms=_required(data, "duration_ms", "private_slice"),
        no_score_region_count=data.get("no_score_region_count", 0),
        critical_span_count=data.get("critical_span_count", 0),
        reason=data.get("reason"),
    )


def _quality_metrics_from_dict(payload: object) -> PrivateAnnotationQualityMetrics:
    data = _mapping(payload, "private_quality")
    _reject_reference_identity_fields(data, "private_quality")
    _reject_unknown_fields(
        data,
        {
            "adjudication_change_rate",
            "agreement_metrics",
            "annotated_recording_count",
            "double_annotated_recording_count",
            "double_annotated_sample_rate",
            "unresolved_disagreement_rate",
        },
        "private_quality",
    )
    return PrivateAnnotationQualityMetrics(
        annotated_recording_count=_required(data, "annotated_recording_count", "private_quality"),
        double_annotated_recording_count=_required(data, "double_annotated_recording_count", "private_quality"),
        double_annotated_sample_rate=_required(data, "double_annotated_sample_rate", "private_quality"),
        agreement_metrics=_required(data, "agreement_metrics", "private_quality"),
        adjudication_change_rate=_required(data, "adjudication_change_rate", "private_quality"),
        unresolved_disagreement_rate=_required(data, "unresolved_disagreement_rate", "private_quality"),
    )


def _coverage_plan_from_dict(payload: object) -> PrivateAcceptanceCoveragePlan:
    data = _mapping(payload, "coverage_plan")
    _reject_reference_identity_fields(data, "coverage_plan")
    _reject_unknown_fields(
        data,
        {"plan_id", "targets", "unsupported_scope", "validated_scope", "version"},
        "coverage_plan",
    )
    return PrivateAcceptanceCoveragePlan(
        plan_id=_required(data, "plan_id", "coverage_plan"),
        version=_required(data, "version", "coverage_plan"),
        targets=tuple(
            _coverage_target_from_dict(item)
            for item in _sequence(_required(data, "targets", "coverage_plan"))
        ),
        validated_scope=tuple(_sequence(data.get("validated_scope", ()))),
        unsupported_scope=tuple(_sequence(data.get("unsupported_scope", ()))),
    )


def _coverage_target_from_dict(payload: object) -> PrivateAcceptanceCoverageSliceTarget:
    data = _mapping(payload, "coverage_target")
    _reject_reference_identity_fields(data, "coverage_target")
    _reject_unknown_fields(
        data,
        {
            "audio_quality_buckets",
            "capture_modes",
            "description",
            "diagnostic_only",
            "duration_buckets",
            "language_accent_domains",
            "min_scored_duration_ms",
            "min_scored_recording_count",
            "overlap_ratio_buckets",
            "platform_sources",
            "required",
            "slice_id",
            "speaker_count_buckets",
        },
        "coverage_target",
    )
    return PrivateAcceptanceCoverageSliceTarget(
        slice_id=_required(data, "slice_id", "coverage_target"),
        capture_modes=tuple(_sequence(_required(data, "capture_modes", "coverage_target"))),
        speaker_count_buckets=tuple(_sequence(_required(data, "speaker_count_buckets", "coverage_target"))),
        duration_buckets=tuple(_sequence(_required(data, "duration_buckets", "coverage_target"))),
        overlap_ratio_buckets=tuple(_sequence(_required(data, "overlap_ratio_buckets", "coverage_target"))),
        audio_quality_buckets=tuple(_sequence(_required(data, "audio_quality_buckets", "coverage_target"))),
        platform_sources=tuple(_sequence(_required(data, "platform_sources", "coverage_target"))),
        language_accent_domains=tuple(_sequence(_required(data, "language_accent_domains", "coverage_target"))),
        min_scored_recording_count=_required(data, "min_scored_recording_count", "coverage_target"),
        min_scored_duration_ms=_required(data, "min_scored_duration_ms", "coverage_target"),
        required=data.get("required", True),
        diagnostic_only=data.get("diagnostic_only", False),
        description=data.get("description"),
    )


def _coverage_result_for_target(
    target: PrivateAcceptanceCoverageSliceTarget,
    observation: PrivateAcceptanceCoverageObservation | None,
) -> PrivateAcceptanceCoverageSliceResult:
    scored_recording_count = 0 if observation is None else observation.scored_recording_count
    scored_duration_ms = 0 if observation is None else observation.scored_duration_ms
    if target.diagnostic_only:
        return PrivateAcceptanceCoverageSliceResult(
            slice_id=target.slice_id,
            status="diagnostic_only",
            required=target.required,
            diagnostic_only=True,
            scored_recording_count=scored_recording_count,
            scored_duration_ms=scored_duration_ms,
            min_scored_recording_count=target.min_scored_recording_count,
            min_scored_duration_ms=target.min_scored_duration_ms,
            reasons=("diagnostic slice is not promoted through private acceptance protocol",),
        )

    reasons: list[str] = []
    if scored_recording_count < target.min_scored_recording_count:
        reasons.append("scored_recording_count below threshold")
    if scored_duration_ms < target.min_scored_duration_ms:
        reasons.append("scored_duration_ms below threshold")
    return PrivateAcceptanceCoverageSliceResult(
        slice_id=target.slice_id,
        status="insufficient_acceptance_coverage" if reasons else "sufficient",
        required=target.required,
        diagnostic_only=False,
        scored_recording_count=scored_recording_count,
        scored_duration_ms=scored_duration_ms,
        min_scored_recording_count=target.min_scored_recording_count,
        min_scored_duration_ms=target.min_scored_duration_ms,
        reasons=tuple(reasons),
    )


def _validate_schema_version(value: object) -> int:
    version = _positive_int(value, "private_acceptance.schema_version")
    if version != PRIVATE_ACCEPTANCE_METADATA_SCHEMA_VERSION:
        raise ValidationError(f"private_acceptance schema version is not supported: {version}")
    return version


def _validate_label(value: object) -> PrivateAcceptanceLabel:
    label = _require_id(value, "private_slice.label")
    if label not in ALLOWED_PRIVATE_ACCEPTANCE_LABELS:
        raise ValidationError(f"private acceptance label is not supported: {label}")
    return label  # type: ignore[return-value]


def _validate_coverage_status(value: object, field_name: str) -> PrivateAcceptanceCoverageStatus:
    status = _require_id(value, field_name)
    if status not in {"sufficient", "insufficient_acceptance_coverage", "diagnostic_only"}:
        raise ValidationError(f"{field_name} is not supported: {status}")
    return status  # type: ignore[return-value]


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{context} must be an object")
    return dict(value)


def _required(data: dict[str, Any], field_name: str, context: str) -> object:
    if field_name not in data:
        raise ValidationError(f"{context}.{field_name} is required")
    return data[field_name]


def _reject_unknown_fields(data: dict[str, Any], allowed_fields: set[str], context: str) -> None:
    unknown = sorted(set(data) - allowed_fields)
    if unknown:
        raise ValidationError(f"{context} has unsupported fields: {', '.join(unknown)}")


def _reject_reference_identity_fields(value: object, context: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in _REFERENCE_IDENTITY_KEYS:
                raise ValidationError(f"{context}.{key} must remain candidate-invisible")
            _reject_reference_identity_fields(item, f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_reference_identity_fields(item, f"{context}[{index}]")


def _tuple_of(values: object, item_type: type, field_name: str) -> tuple[Any, ...]:
    try:
        result = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValidationError(f"{field_name} must be an iterable") from exc
    for index, item in enumerate(result):
        if not isinstance(item, item_type):
            raise ValidationError(f"{field_name}[{index}] must be a {item_type.__name__}")
    return result


def _sequence(value: object) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValidationError("private acceptance sequence fields must be arrays")
    return tuple(value)


def _unique_ids(values: tuple[str, ...], field_name: str) -> None:
    seen: set[str] = set()
    for value in values:
        value = _require_id(value, field_name)
        if value in seen:
            raise ValidationError(f"duplicate {field_name}: {value}")
        seen.add(value)


def _non_empty_text_tuple(values: object, field_name: str) -> tuple[str, ...]:
    result = tuple(_require_text(item, field_name) for item in _sequence(values))
    if not result:
        raise ValidationError(f"{field_name} is required")
    return result


def _non_empty_id_tuple(values: object, field_name: str) -> tuple[str, ...]:
    result = _id_tuple(values, field_name)
    if not result:
        raise ValidationError(f"{field_name} is required")
    return result


def _id_tuple(values: object, field_name: str) -> tuple[str, ...]:
    return tuple(_require_id(item, field_name) for item in _sequence(values))


def _probability_map(value: object, field_name: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValidationError(f"{field_name} must be an object")
    if not value:
        raise ValidationError(f"{field_name} is required")
    result: dict[str, float] = {}
    for key, item in value.items():
        metric_name = _require_id(key, f"{field_name}.key")
        if metric_name in _REFERENCE_IDENTITY_KEYS:
            raise ValidationError(f"{field_name}.{metric_name} must remain candidate-invisible")
        result[metric_name] = _probability(item, f"{field_name}.{metric_name}")
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


def _positive_int(value: object, field_name: str) -> int:
    result = _non_negative_int(value, field_name)
    if result == 0:
        raise ValidationError(f"{field_name} must be > 0")
    return result


def _non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{field_name} must be an integer")
    if value < 0:
        raise ValidationError(f"{field_name} must be >= 0")
    return value


def _require_id(value: object, field_name: str) -> str:
    if value is None:
        raise ValidationError(f"{field_name} is required")
    if not isinstance(value, str):
        raise ValidationError(f"{field_name} must be a string")
    value = value.strip()
    if not value:
        raise ValidationError(f"{field_name} is required")
    return value


def _optional_id(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_id(value, field_name)


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValidationError(f"{field_name} must be a boolean")
    return value


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{field_name} must be a string")
    if not value.strip():
        raise ValidationError(f"{field_name} is required")
    return value


def _optional_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_text(value, field_name)
