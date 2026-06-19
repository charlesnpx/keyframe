"""Benchmark report models, regression gates, and review-signal calibration."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from keyframe.diarization.evaluator import (
    DiarizationEvaluationResult,
    DiarizationRecordingMetricRow,
    DiarizationSliceMetricRow,
)
from keyframe.diarization.models import ValidationError


BENCHMARK_REPORT_SCHEMA_VERSION = 1

BenchmarkReportStatus = Literal["passed", "failed"]
BenchmarkReportScopeType = Literal["corpus", "branch", "recording", "slice"]
RegressionBudgetKind = Literal[
    "metric",
    "overlap",
    "boundary",
    "false_confidence",
    "human_review_burden",
    "deterministic_rendering",
]
RegressionGateStatus = Literal["passed", "failed", "unavailable"]
UncertaintyStatus = Literal["available", "unavailable"]
ReviewSignalSeverity = Literal["serious", "minor"]
CriticalSpanPolicyScope = Literal["diagnostic_fixture"]
MetricDirection = Literal["higher_is_better", "lower_is_better"]

_REVIEW_SIGNAL_METRIC_NAMES = ("precision", "recall", "false_confident_rate", "over_flag_rate", "coverage")


@dataclass(frozen=True)
class UncertaintyInterval:
    """A conservative interval for either paired deltas or point scores."""

    status: UncertaintyStatus
    confidence_level: float = 0.95
    lower: float | None = None
    upper: float | None = None
    basis: str = "unavailable"
    reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "status",
            _validate_choice(self.status, {"available", "unavailable"}, "uncertainty.status"),
        )
        object.__setattr__(
            self,
            "confidence_level",
            _validate_probability(self.confidence_level, "uncertainty.confidence_level", allow_zero=False),
        )
        object.__setattr__(self, "basis", _require_id(self.basis, "uncertainty.basis"))
        object.__setattr__(self, "reason", _optional_text(self.reason, "uncertainty.reason"))
        lower = _optional_finite_number(self.lower, "uncertainty.lower")
        upper = _optional_finite_number(self.upper, "uncertainty.upper")
        if self.status == "available":
            if lower is None or upper is None:
                raise ValidationError("available uncertainty intervals require lower and upper")
            if lower > upper:
                raise ValidationError("uncertainty.lower must be <= uncertainty.upper")
            if self.reason is not None:
                raise ValidationError("available uncertainty intervals cannot include a reason")
        else:
            if lower is not None or upper is not None:
                raise ValidationError("unavailable uncertainty intervals cannot include bounds")
            if self.reason is None:
                raise ValidationError("unavailable uncertainty intervals require a reason")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def to_dict(self) -> dict[str, Any]:
        return {
            "basis": self.basis,
            "confidence_level": self.confidence_level,
            "lower": self.lower,
            "reason": self.reason,
            "status": self.status,
            "upper": self.upper,
        }


@dataclass(frozen=True)
class RegressionGateResult:
    """Gate outcome for one metric summary."""

    status: RegressionGateStatus
    budget_id: str | None = None
    reasons: tuple[str, ...] = ()
    thresholds: dict[str, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "status",
            _validate_choice(self.status, {"passed", "failed", "unavailable"}, "gate.status"),
        )
        object.__setattr__(self, "budget_id", _optional_id(self.budget_id, "gate.budget_id"))
        object.__setattr__(self, "reasons", _tuple_of_text(self.reasons, "gate.reasons"))
        thresholds = _validate_number_map(self.thresholds or {}, "gate.thresholds")
        object.__setattr__(self, "thresholds", thresholds)
        if self.status == "passed" and self.reasons:
            raise ValidationError("passed regression gates cannot include failure reasons")
        if self.status in {"failed", "unavailable"} and not self.reasons:
            raise ValidationError("failed or unavailable regression gates require reasons")

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget_id": self.budget_id,
            "reasons": list(self.reasons),
            "status": self.status,
            "thresholds": dict(self.thresholds or {}),
        }


@dataclass(frozen=True)
class BenchmarkRegressionBudget:
    """Metric or slice-specific threshold used to gate regressions."""

    budget_id: str
    metric_name: str
    budget_kind: RegressionBudgetKind = "metric"
    direction: MetricDirection = "higher_is_better"
    max_regression_delta: float | None = None
    min_point_score: float | None = None
    max_point_score: float | None = None
    scope_type: BenchmarkReportScopeType | None = None
    scope_id: str | None = None
    slice_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "budget_id", _require_id(self.budget_id, "regression_budget.budget_id"))
        object.__setattr__(self, "metric_name", _require_id(self.metric_name, "regression_budget.metric_name"))
        object.__setattr__(
            self,
            "direction",
            _validate_choice(
                self.direction,
                {"higher_is_better", "lower_is_better"},
                "regression_budget.direction",
            ),
        )
        object.__setattr__(
            self,
            "budget_kind",
            _validate_choice(
                self.budget_kind,
                {
                    "metric",
                    "overlap",
                    "boundary",
                    "false_confidence",
                    "human_review_burden",
                    "deterministic_rendering",
                },
                "regression_budget.budget_kind",
            ),
        )
        max_regression_delta = _optional_finite_number(
            self.max_regression_delta,
            "regression_budget.max_regression_delta",
        )
        min_point_score = _optional_finite_number(self.min_point_score, "regression_budget.min_point_score")
        max_point_score = _optional_finite_number(self.max_point_score, "regression_budget.max_point_score")
        if max_regression_delta is None and min_point_score is None and max_point_score is None:
            raise ValidationError("regression budgets require a regression or point-score threshold")
        if max_regression_delta is not None and max_regression_delta < 0:
            raise ValidationError("regression_budget.max_regression_delta must be non-negative")
        object.__setattr__(self, "max_regression_delta", max_regression_delta)
        object.__setattr__(self, "min_point_score", min_point_score)
        object.__setattr__(self, "max_point_score", max_point_score)
        if self.scope_type is not None:
            object.__setattr__(
                self,
                "scope_type",
                _validate_choice(
                    self.scope_type,
                    {"corpus", "branch", "recording", "slice"},
                    "regression_budget.scope_type",
                ),
            )
        object.__setattr__(self, "scope_id", _optional_id(self.scope_id, "regression_budget.scope_id"))
        object.__setattr__(self, "slice_id", _optional_id(self.slice_id, "regression_budget.slice_id"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget_id": self.budget_id,
            "budget_kind": self.budget_kind,
            "direction": self.direction,
            "max_point_score": self.max_point_score,
            "max_regression_delta": self.max_regression_delta,
            "metric_name": self.metric_name,
            "min_point_score": self.min_point_score,
            "scope_id": self.scope_id,
            "scope_type": self.scope_type,
            "slice_id": self.slice_id,
        }


@dataclass(frozen=True)
class BenchmarkGateConfig:
    """Regression budget collection for benchmark reports."""

    budgets: tuple[BenchmarkRegressionBudget, ...] = ()

    def __post_init__(self) -> None:
        budgets = _tuple_of(self.budgets, BenchmarkRegressionBudget, "gate_config.budgets")
        seen: set[str] = set()
        for budget in budgets:
            if budget.budget_id in seen:
                raise ValidationError(f"gate_config.budgets contains duplicate budget_id: {budget.budget_id}")
            seen.add(budget.budget_id)
        object.__setattr__(self, "budgets", budgets)

    def to_dict(self) -> dict[str, Any]:
        return {"budgets": [budget.to_dict() for budget in self.budgets]}


@dataclass(frozen=True)
class BenchmarkEvaluationCase:
    """One evaluated recording for one corpus and benchmark branch."""

    corpus_id: str
    branch_id: str
    evaluation: DiarizationEvaluationResult
    baseline_evaluation: DiarizationEvaluationResult | None = None
    scored_duration_ms: int = 0
    scored_words: int = 0
    scored_speaker_turns: int = 0
    slice_scored_words: Mapping[str, int] | None = None
    slice_scored_speaker_turns: Mapping[str, int] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "corpus_id", _require_id(self.corpus_id, "benchmark_case.corpus_id"))
        object.__setattr__(self, "branch_id", _require_id(self.branch_id, "benchmark_case.branch_id"))
        if not isinstance(self.evaluation, DiarizationEvaluationResult):
            raise ValidationError("benchmark_case.evaluation must be a DiarizationEvaluationResult")
        if self.baseline_evaluation is not None:
            if not isinstance(self.baseline_evaluation, DiarizationEvaluationResult):
                raise ValidationError("benchmark_case.baseline_evaluation must be a DiarizationEvaluationResult")
            if self.baseline_evaluation.recording_id != self.evaluation.recording_id:
                raise ValidationError("benchmark_case.baseline_evaluation recording_id must match evaluation")
            if self.baseline_evaluation.scoring_policy != self.evaluation.scoring_policy:
                raise ValidationError("benchmark_case.baseline_evaluation scoring_policy must match evaluation")
        object.__setattr__(
            self,
            "scored_duration_ms",
            _non_negative_int(self.scored_duration_ms, "benchmark_case.scored_duration_ms"),
        )
        object.__setattr__(self, "scored_words", _non_negative_int(self.scored_words, "benchmark_case.scored_words"))
        object.__setattr__(
            self,
            "scored_speaker_turns",
            _non_negative_int(self.scored_speaker_turns, "benchmark_case.scored_speaker_turns"),
        )
        object.__setattr__(
            self,
            "slice_scored_words",
            _validate_int_map(self.slice_scored_words or {}, "benchmark_case.slice_scored_words"),
        )
        object.__setattr__(
            self,
            "slice_scored_speaker_turns",
            _validate_int_map(
                self.slice_scored_speaker_turns or {},
                "benchmark_case.slice_scored_speaker_turns",
            ),
        )


@dataclass(frozen=True)
class BenchmarkMetricResult:
    """Aggregated report metric for a corpus, branch, recording, or slice."""

    scope_type: BenchmarkReportScopeType
    scope_id: str
    metric_name: str
    point_score: float
    sample_count: int
    scored_duration_ms: int
    scored_words: int
    scored_speaker_turns: int
    gate: RegressionGateResult
    uncertainty: UncertaintyInterval
    baseline_score: float | None = None
    paired_delta: float | None = None
    corpus_id: str | None = None
    branch_id: str | None = None
    recording_id: str | None = None
    slice_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scope_type",
            _validate_choice(
                self.scope_type,
                {"corpus", "branch", "recording", "slice"},
                "metric_result.scope_type",
            ),
        )
        object.__setattr__(self, "scope_id", _require_id(self.scope_id, "metric_result.scope_id"))
        object.__setattr__(self, "metric_name", _require_id(self.metric_name, "metric_result.metric_name"))
        object.__setattr__(self, "point_score", _finite_number(self.point_score, "metric_result.point_score"))
        object.__setattr__(
            self,
            "baseline_score",
            _optional_finite_number(self.baseline_score, "metric_result.baseline_score"),
        )
        object.__setattr__(
            self,
            "paired_delta",
            _optional_finite_number(self.paired_delta, "metric_result.paired_delta"),
        )
        object.__setattr__(self, "sample_count", _positive_int(self.sample_count, "metric_result.sample_count"))
        object.__setattr__(
            self,
            "scored_duration_ms",
            _non_negative_int(self.scored_duration_ms, "metric_result.scored_duration_ms"),
        )
        object.__setattr__(self, "scored_words", _non_negative_int(self.scored_words, "metric_result.scored_words"))
        object.__setattr__(
            self,
            "scored_speaker_turns",
            _non_negative_int(self.scored_speaker_turns, "metric_result.scored_speaker_turns"),
        )
        if not isinstance(self.gate, RegressionGateResult):
            raise ValidationError("metric_result.gate must be a RegressionGateResult")
        if not isinstance(self.uncertainty, UncertaintyInterval):
            raise ValidationError("metric_result.uncertainty must be an UncertaintyInterval")
        object.__setattr__(self, "corpus_id", _optional_id(self.corpus_id, "metric_result.corpus_id"))
        object.__setattr__(self, "branch_id", _optional_id(self.branch_id, "metric_result.branch_id"))
        object.__setattr__(self, "recording_id", _optional_id(self.recording_id, "metric_result.recording_id"))
        object.__setattr__(self, "slice_id", _optional_id(self.slice_id, "metric_result.slice_id"))
        _validate_metric_result_scope_identity(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_score": self.baseline_score,
            "branch_id": self.branch_id,
            "corpus_id": self.corpus_id,
            "gate": self.gate.to_dict(),
            "metric_name": self.metric_name,
            "paired_delta": self.paired_delta,
            "point_score": self.point_score,
            "recording_id": self.recording_id,
            "sample_count": self.sample_count,
            "scope_id": self.scope_id,
            "scope_type": self.scope_type,
            "scored_duration_ms": self.scored_duration_ms,
            "scored_speaker_turns": self.scored_speaker_turns,
            "scored_words": self.scored_words,
            "slice_id": self.slice_id,
            "uncertainty": self.uncertainty.to_dict(),
        }


@dataclass(frozen=True)
class ReviewSignalSpan:
    """One canonical span-level review signal and reference label."""

    signal_id: str
    corpus_id: str
    branch_id: str
    recording_id: str
    start_ms: int
    end_ms: int
    severity: ReviewSignalSeverity
    reference_review_required: bool
    predicted_review_required: bool | None
    labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "signal_id", _require_id(self.signal_id, "review_signal.signal_id"))
        object.__setattr__(self, "corpus_id", _require_id(self.corpus_id, "review_signal.corpus_id"))
        object.__setattr__(self, "branch_id", _require_id(self.branch_id, "review_signal.branch_id"))
        object.__setattr__(self, "recording_id", _require_id(self.recording_id, "review_signal.recording_id"))
        start_ms = _non_negative_int(self.start_ms, "review_signal.start_ms")
        end_ms = _positive_int(self.end_ms, "review_signal.end_ms")
        if end_ms <= start_ms:
            raise ValidationError("review_signal.end_ms must be greater than start_ms")
        object.__setattr__(self, "start_ms", start_ms)
        object.__setattr__(self, "end_ms", end_ms)
        object.__setattr__(
            self,
            "severity",
            _validate_choice(self.severity, {"serious", "minor"}, "review_signal.severity"),
        )
        object.__setattr__(
            self,
            "reference_review_required",
            _require_bool(self.reference_review_required, "review_signal.reference_review_required"),
        )
        if self.predicted_review_required is not None:
            object.__setattr__(
                self,
                "predicted_review_required",
                _require_bool(self.predicted_review_required, "review_signal.predicted_review_required"),
            )
        object.__setattr__(self, "labels", _tuple_of_text(self.labels, "review_signal.labels"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "corpus_id": self.corpus_id,
            "end_ms": self.end_ms,
            "labels": list(self.labels),
            "predicted_review_required": self.predicted_review_required,
            "recording_id": self.recording_id,
            "reference_review_required": self.reference_review_required,
            "severity": self.severity,
            "signal_id": self.signal_id,
            "start_ms": self.start_ms,
        }


@dataclass(frozen=True)
class ReviewSignalSeverityBreakdown:
    """Confusion-matrix and rates for one review-signal severity."""

    severity: ReviewSignalSeverity
    total: int
    assessed: int
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    precision: float | None
    recall: float | None
    false_confident_rate: float | None
    over_flag_rate: float | None
    coverage: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "severity",
            _validate_choice(self.severity, {"serious", "minor"}, "severity_breakdown.severity"),
        )
        count_fields = ("total", "assessed", "true_positive", "false_positive", "false_negative", "true_negative")
        for field_name in count_fields:
            object.__setattr__(
                self,
                field_name,
                _non_negative_int(getattr(self, field_name), f"severity_breakdown.{field_name}"),
            )
        for field_name in ("precision", "recall", "false_confident_rate", "over_flag_rate"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _validate_probability(value, f"severity_breakdown.{field_name}"))
        object.__setattr__(self, "coverage", _validate_probability(self.coverage, "severity_breakdown.coverage"))
        _validate_review_signal_summary(
            total=self.total,
            assessed=self.assessed,
            true_positive=self.true_positive,
            false_positive=self.false_positive,
            false_negative=self.false_negative,
            true_negative=self.true_negative,
            precision=self.precision,
            recall=self.recall,
            false_confident_rate=self.false_confident_rate,
            over_flag_rate=self.over_flag_rate,
            coverage=self.coverage,
            context=f"severity_breakdown.{self.severity}",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "assessed": self.assessed,
            "coverage": self.coverage,
            "false_confident_rate": self.false_confident_rate,
            "false_negative": self.false_negative,
            "false_positive": self.false_positive,
            "over_flag_rate": self.over_flag_rate,
            "precision": self.precision,
            "recall": self.recall,
            "severity": self.severity,
            "total": self.total,
            "true_negative": self.true_negative,
            "true_positive": self.true_positive,
        }


@dataclass(frozen=True)
class ReviewSignalCalibration:
    """Aggregate review-signal calibration for span-level signals."""

    total: int
    assessed: int
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    precision: float | None
    recall: float | None
    false_confident_rate: float | None
    over_flag_rate: float | None
    coverage: float
    serious: ReviewSignalSeverityBreakdown
    minor: ReviewSignalSeverityBreakdown

    def __post_init__(self) -> None:
        count_fields = ("total", "assessed", "true_positive", "false_positive", "false_negative", "true_negative")
        for field_name in count_fields:
            object.__setattr__(
                self,
                field_name,
                _non_negative_int(getattr(self, field_name), f"review_calibration.{field_name}"),
            )
        for field_name in ("precision", "recall", "false_confident_rate", "over_flag_rate"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _validate_probability(value, f"review_calibration.{field_name}"))
        object.__setattr__(self, "coverage", _validate_probability(self.coverage, "review_calibration.coverage"))
        if not isinstance(self.serious, ReviewSignalSeverityBreakdown):
            raise ValidationError("review_calibration.serious must be a ReviewSignalSeverityBreakdown")
        if not isinstance(self.minor, ReviewSignalSeverityBreakdown):
            raise ValidationError("review_calibration.minor must be a ReviewSignalSeverityBreakdown")
        if self.serious.severity != "serious":
            raise ValidationError("review_calibration.serious must contain serious breakdown")
        if self.minor.severity != "minor":
            raise ValidationError("review_calibration.minor must contain minor breakdown")
        _validate_review_signal_summary(
            total=self.total,
            assessed=self.assessed,
            true_positive=self.true_positive,
            false_positive=self.false_positive,
            false_negative=self.false_negative,
            true_negative=self.true_negative,
            precision=self.precision,
            recall=self.recall,
            false_confident_rate=self.false_confident_rate,
            over_flag_rate=self.over_flag_rate,
            coverage=self.coverage,
            context="review_calibration",
        )
        _validate_review_signal_breakdown_totals(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "assessed": self.assessed,
            "coverage": self.coverage,
            "false_confident_rate": self.false_confident_rate,
            "false_negative": self.false_negative,
            "false_positive": self.false_positive,
            "minor": self.minor.to_dict(),
            "over_flag_rate": self.over_flag_rate,
            "precision": self.precision,
            "recall": self.recall,
            "serious": self.serious.to_dict(),
            "total": self.total,
            "true_negative": self.true_negative,
            "true_positive": self.true_positive,
        }


@dataclass(frozen=True)
class ReviewSignalScopeCalibration:
    """Review-signal calibration for one report scope."""

    scope_type: BenchmarkReportScopeType
    scope_id: str
    calibration: ReviewSignalCalibration
    scored_duration_ms: int = 0
    corpus_id: str | None = None
    branch_id: str | None = None
    recording_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scope_type",
            _validate_choice(
                self.scope_type,
                {"corpus", "branch", "recording"},
                "review_scope_calibration.scope_type",
            ),
        )
        object.__setattr__(self, "scope_id", _require_id(self.scope_id, "review_scope_calibration.scope_id"))
        object.__setattr__(self, "corpus_id", _optional_id(self.corpus_id, "review_scope_calibration.corpus_id"))
        object.__setattr__(self, "branch_id", _optional_id(self.branch_id, "review_scope_calibration.branch_id"))
        object.__setattr__(
            self,
            "recording_id",
            _optional_id(self.recording_id, "review_scope_calibration.recording_id"),
        )
        if not isinstance(self.calibration, ReviewSignalCalibration):
            raise ValidationError("review_scope_calibration.calibration must be a ReviewSignalCalibration")
        object.__setattr__(
            self,
            "scored_duration_ms",
            _non_negative_int(self.scored_duration_ms, "review_scope_calibration.scored_duration_ms"),
        )
        _validate_review_signal_scope_identity(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "calibration": self.calibration.to_dict(),
            "corpus_id": self.corpus_id,
            "recording_id": self.recording_id,
            "scored_duration_ms": self.scored_duration_ms,
            "scope_id": self.scope_id,
            "scope_type": self.scope_type,
        }


@dataclass(frozen=True)
class CriticalSpanPolicyDefinition:
    """Versioned diagnostic policy hook for critical-span scoring."""

    policy_id: str
    version: str
    description: str
    critical_severities: tuple[ReviewSignalSeverity, ...] = ("serious",)
    critical_labels: tuple[str, ...] = ()
    minimum_recall: float = 1.0
    scope: CriticalSpanPolicyScope = "diagnostic_fixture"

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _require_id(self.policy_id, "critical_span_policy.policy_id"))
        object.__setattr__(self, "version", _require_id(self.version, "critical_span_policy.version"))
        object.__setattr__(self, "description", _require_text(self.description, "critical_span_policy.description"))
        object.__setattr__(
            self,
            "critical_severities",
            tuple(
                _validate_choice(severity, {"serious", "minor"}, "critical_span_policy.critical_severities")
                for severity in self.critical_severities
            ),
        )
        object.__setattr__(
            self,
            "critical_labels",
            _tuple_of_text(self.critical_labels, "critical_span_policy.critical_labels"),
        )
        object.__setattr__(
            self,
            "minimum_recall",
            _validate_probability(self.minimum_recall, "critical_span_policy.minimum_recall"),
        )
        object.__setattr__(
            self,
            "scope",
            _validate_choice(self.scope, {"diagnostic_fixture"}, "critical_span_policy.scope"),
        )
        if not self.critical_severities and not self.critical_labels:
            raise ValidationError("critical span policies require severities or labels")

    def to_dict(self) -> dict[str, Any]:
        return {
            "critical_labels": list(self.critical_labels),
            "critical_severities": list(self.critical_severities),
            "description": self.description,
            "minimum_recall": self.minimum_recall,
            "policy_id": self.policy_id,
            "scope": self.scope,
            "version": self.version,
        }


@dataclass(frozen=True)
class CriticalSpanDiagnosticScore:
    """Diagnostic-only critical-span score for synthetic fixtures."""

    status: RegressionGateStatus
    policy_id: str
    version: str
    critical_span_count: int
    detected_critical_span_count: int
    missed_critical_span_count: int
    recall: float | None
    failures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "status",
            _validate_choice(self.status, {"passed", "failed", "unavailable"}, "critical_span_score.status"),
        )
        object.__setattr__(self, "policy_id", _require_id(self.policy_id, "critical_span_score.policy_id"))
        object.__setattr__(self, "version", _require_id(self.version, "critical_span_score.version"))
        for field_name in ("critical_span_count", "detected_critical_span_count", "missed_critical_span_count"):
            object.__setattr__(
                self,
                field_name,
                _non_negative_int(getattr(self, field_name), f"critical_span_score.{field_name}"),
            )
        if self.recall is not None:
            object.__setattr__(self, "recall", _validate_probability(self.recall, "critical_span_score.recall"))
        object.__setattr__(self, "failures", _tuple_of_text(self.failures, "critical_span_score.failures"))
        if self.status == "passed" and self.failures:
            raise ValidationError("passed critical-span scores cannot include failures")
        if self.status in {"failed", "unavailable"} and not self.failures:
            raise ValidationError("failed or unavailable critical-span scores require failures")

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "critical_span_count": self.critical_span_count,
            "detected_critical_span_count": self.detected_critical_span_count,
            "failures": list(self.failures),
            "missed_critical_span_count": self.missed_critical_span_count,
            "policy_id": self.policy_id,
            "recall": self.recall,
            "status": self.status,
            "version": self.version,
        }


@dataclass(frozen=True)
class BenchmarkReport:
    """Complete first-pass benchmark report."""

    report_id: str
    status: BenchmarkReportStatus
    gate_config: BenchmarkGateConfig
    corpus_results: tuple[BenchmarkMetricResult, ...] = ()
    branch_results: tuple[BenchmarkMetricResult, ...] = ()
    recording_results: tuple[BenchmarkMetricResult, ...] = ()
    slice_results: tuple[BenchmarkMetricResult, ...] = ()
    review_signal_calibration: ReviewSignalCalibration | None = None
    review_signal_scope_calibrations: tuple[ReviewSignalScopeCalibration, ...] = ()
    critical_span_policy: CriticalSpanPolicyDefinition | None = None
    critical_span_diagnostic: CriticalSpanDiagnosticScore | None = None
    schema_version: int = BENCHMARK_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "report_id", _require_id(self.report_id, "benchmark_report.report_id"))
        object.__setattr__(
            self,
            "status",
            _validate_choice(self.status, {"passed", "failed"}, "benchmark_report.status"),
        )
        object.__setattr__(
            self,
            "schema_version",
            _validate_report_schema_version(self.schema_version),
        )
        if not isinstance(self.gate_config, BenchmarkGateConfig):
            raise ValidationError("benchmark_report.gate_config must be a BenchmarkGateConfig")
        for field_name, scope_type in (
            ("corpus_results", "corpus"),
            ("branch_results", "branch"),
            ("recording_results", "recording"),
            ("slice_results", "slice"),
        ):
            results = _tuple_of(getattr(self, field_name), BenchmarkMetricResult, f"benchmark_report.{field_name}")
            if any(result.scope_type != scope_type for result in results):
                raise ValidationError(f"benchmark_report.{field_name} must contain only {scope_type} results")
            object.__setattr__(self, field_name, results)
        if self.review_signal_calibration is not None and not isinstance(
            self.review_signal_calibration,
            ReviewSignalCalibration,
        ):
            raise ValidationError("benchmark_report.review_signal_calibration must be a ReviewSignalCalibration")
        object.__setattr__(
            self,
            "review_signal_scope_calibrations",
            _tuple_of(
                self.review_signal_scope_calibrations,
                ReviewSignalScopeCalibration,
                "benchmark_report.review_signal_scope_calibrations",
            ),
        )
        if self.critical_span_policy is not None and not isinstance(
            self.critical_span_policy,
            CriticalSpanPolicyDefinition,
        ):
            raise ValidationError("benchmark_report.critical_span_policy must be a CriticalSpanPolicyDefinition")
        if self.critical_span_diagnostic is not None and not isinstance(
            self.critical_span_diagnostic,
            CriticalSpanDiagnosticScore,
        ):
            raise ValidationError("benchmark_report.critical_span_diagnostic must be a CriticalSpanDiagnosticScore")
        metric_results = self.metric_results
        if not metric_results:
            raise ValidationError("benchmark_report requires at least one metric result")
        _validate_unique_metric_results(metric_results)
        _validate_serialized_gates(self.gate_config, metric_results)
        _validate_serialized_review_signal_metric_results(
            self.review_signal_calibration,
            self.review_signal_scope_calibrations,
            self.gate_config,
            metric_results,
        )
        has_failure = _has_failed_gate(metric_results, self.critical_span_diagnostic)
        if self.status == "passed" and has_failure:
            raise ValidationError("passed benchmark reports cannot include failed gates")
        if self.status == "failed" and not has_failure:
            raise ValidationError("failed benchmark reports must include failed gates")
        _validate_serialized_critical_span_diagnostic(
            self.critical_span_policy,
            self.critical_span_diagnostic,
        )

    @property
    def metric_results(self) -> tuple[BenchmarkMetricResult, ...]:
        return self.corpus_results + self.branch_results + self.recording_results + self.slice_results

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch_results": [result.to_dict() for result in self.branch_results],
            "corpus_results": [result.to_dict() for result in self.corpus_results],
            "critical_span_diagnostic": (
                self.critical_span_diagnostic.to_dict() if self.critical_span_diagnostic is not None else None
            ),
            "critical_span_policy": (
                self.critical_span_policy.to_dict() if self.critical_span_policy is not None else None
            ),
            "gate_config": self.gate_config.to_dict(),
            "recording_results": [result.to_dict() for result in self.recording_results],
            "report_id": self.report_id,
            "review_signal_calibration": (
                self.review_signal_calibration.to_dict() if self.review_signal_calibration is not None else None
            ),
            "review_signal_scope_calibrations": [
                calibration.to_dict() for calibration in self.review_signal_scope_calibrations
            ],
            "schema_version": self.schema_version,
            "slice_results": [result.to_dict() for result in self.slice_results],
            "status": self.status,
        }


@dataclass(frozen=True)
class _MetricObservation:
    corpus_id: str
    branch_id: str
    recording_id: str
    metric_name: str
    point_score: float
    baseline_score: float | None
    scored_duration_ms: int
    scored_words: int
    scored_speaker_turns: int
    source_scope: Literal["recording", "slice"]
    slice_id: str | None = None


def build_benchmark_report(
    report_id: str,
    cases: tuple[BenchmarkEvaluationCase, ...],
    *,
    gate_config: BenchmarkGateConfig | None = None,
    review_signals: tuple[ReviewSignalSpan, ...] = (),
    critical_span_policy: CriticalSpanPolicyDefinition | None = None,
) -> BenchmarkReport:
    """Build a comparable benchmark report from evaluated recording cases."""

    report_id = _require_id(report_id, "report_id")
    cases = _tuple_of(cases, BenchmarkEvaluationCase, "cases")
    gate_config = gate_config or BenchmarkGateConfig()
    if not isinstance(gate_config, BenchmarkGateConfig):
        raise ValidationError("gate_config must be a BenchmarkGateConfig")
    review_signals = _tuple_of(review_signals, ReviewSignalSpan, "review_signals")
    _validate_review_signals_match_cases(cases, review_signals)
    observations = tuple(observation for case in cases for observation in _observations_from_case(case))
    if not observations:
        raise ValidationError("benchmark reports require at least one scored metric observation")
    corpus_results = _metric_results_for_scope(observations, "corpus", gate_config)
    branch_results = _metric_results_for_scope(observations, "branch", gate_config)
    recording_results = _metric_results_for_scope(observations, "recording", gate_config)
    slice_results = _metric_results_for_scope(observations, "slice", gate_config)
    review_calibration = calibrate_review_signals(review_signals) if review_signals else None
    review_scope_calibrations = _review_signal_scope_calibrations(review_signals) if review_calibration else ()
    review_metric_results = _review_signal_metric_results(review_scope_calibrations, gate_config)
    corpus_results = _sort_metric_results(
        corpus_results + tuple(result for result in review_metric_results if result.scope_type == "corpus")
    )
    branch_results = _sort_metric_results(
        branch_results + tuple(result for result in review_metric_results if result.scope_type == "branch")
    )
    recording_results = _sort_metric_results(
        recording_results + tuple(result for result in review_metric_results if result.scope_type == "recording")
    )
    all_results = corpus_results + branch_results + recording_results + slice_results
    _validate_all_budgets_matched(gate_config, all_results)
    critical_score = (
        score_diagnostic_critical_spans(review_signals, critical_span_policy)
        if critical_span_policy is not None
        else None
    )
    status: BenchmarkReportStatus = "failed" if _has_failed_gate(
        all_results,
        critical_score,
    ) else "passed"
    return BenchmarkReport(
        report_id=report_id,
        status=status,
        gate_config=gate_config,
        corpus_results=corpus_results,
        branch_results=branch_results,
        recording_results=recording_results,
        slice_results=slice_results,
        review_signal_calibration=review_calibration,
        review_signal_scope_calibrations=review_scope_calibrations,
        critical_span_policy=critical_span_policy,
        critical_span_diagnostic=critical_score,
    )


def calibrate_review_signals(signals: tuple[ReviewSignalSpan, ...]) -> ReviewSignalCalibration:
    """Compute precision, recall, false-confident, over-flag, and severity breakdowns."""

    signals = _tuple_of(signals, ReviewSignalSpan, "review_signals")
    serious = _review_signal_breakdown(tuple(signal for signal in signals if signal.severity == "serious"), "serious")
    minor = _review_signal_breakdown(tuple(signal for signal in signals if signal.severity == "minor"), "minor")
    aggregate = _review_signal_counts(signals)
    return ReviewSignalCalibration(
        total=aggregate["total"],
        assessed=aggregate["assessed"],
        true_positive=aggregate["tp"],
        false_positive=aggregate["fp"],
        false_negative=aggregate["fn"],
        true_negative=aggregate["tn"],
        precision=_rate(aggregate["tp"], aggregate["tp"] + aggregate["fp"]),
        recall=_rate(aggregate["tp"], aggregate["tp"] + aggregate["fn"]),
        false_confident_rate=_rate(aggregate["fn"], aggregate["tp"] + aggregate["fn"]),
        over_flag_rate=_rate(aggregate["fp"], aggregate["fp"] + aggregate["tn"]),
        coverage=_rate(aggregate["assessed"], aggregate["total"]) or 0.0,
        serious=serious,
        minor=minor,
    )


def score_diagnostic_critical_spans(
    signals: tuple[ReviewSignalSpan, ...],
    policy: CriticalSpanPolicyDefinition,
) -> CriticalSpanDiagnosticScore:
    """Score diagnostic critical-span hooks against a versioned policy definition."""

    signals = _tuple_of(signals, ReviewSignalSpan, "review_signals")
    if not isinstance(policy, CriticalSpanPolicyDefinition):
        raise ValidationError("critical_span_policy must be a CriticalSpanPolicyDefinition")
    critical = tuple(signal for signal in signals if _is_critical_span(signal, policy))
    if not critical:
        return CriticalSpanDiagnosticScore(
            status="unavailable",
            policy_id=policy.policy_id,
            version=policy.version,
            critical_span_count=0,
            detected_critical_span_count=0,
            missed_critical_span_count=0,
            recall=None,
            failures=("no diagnostic critical spans matched policy",),
        )
    detected = sum(1 for signal in critical if signal.predicted_review_required is True)
    missed = len(critical) - detected
    recall = _round_metric(detected / len(critical))
    failures = ()
    status: RegressionGateStatus = "passed"
    if recall < policy.minimum_recall:
        status = "failed"
        failures = (f"critical-span recall {recall} below minimum {policy.minimum_recall}",)
    return CriticalSpanDiagnosticScore(
        status=status,
        policy_id=policy.policy_id,
        version=policy.version,
        critical_span_count=len(critical),
        detected_critical_span_count=detected,
        missed_critical_span_count=missed,
        recall=recall,
        failures=failures,
    )


def benchmark_report_json_dumps(report: BenchmarkReport) -> str:
    """Serialize a benchmark report to stable JSON."""

    if not isinstance(report, BenchmarkReport):
        raise ValidationError("report must be a BenchmarkReport")
    return json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def benchmark_report_json_loads(text: str) -> BenchmarkReport:
    """Load a benchmark report from JSON emitted by benchmark_report_json_dumps."""

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"benchmark report JSON is invalid: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValidationError("benchmark report JSON must be an object")
    return benchmark_report_from_dict(payload)


def benchmark_report_to_markdown(report: BenchmarkReport) -> str:
    """Render a compact Markdown report for human review."""

    if not isinstance(report, BenchmarkReport):
        raise ValidationError("report must be a BenchmarkReport")
    lines = [
        f"# Benchmark Report: {report.report_id}",
        "",
        f"Status: {report.status}",
        "",
    ]
    for title, results in (
        ("Corpus Results", report.corpus_results),
        ("Branch Results", report.branch_results),
        ("Recording Results", report.recording_results),
        ("Slice Results", report.slice_results),
    ):
        lines.extend(_markdown_metric_table(title, results))
    if report.review_signal_calibration is not None:
        calibration = report.review_signal_calibration
        lines.extend(
            [
                "## Review Signals",
                "",
                f"- Precision: {_format_optional_float(calibration.precision)}",
                f"- Recall: {_format_optional_float(calibration.recall)}",
                f"- False-confident rate: {_format_optional_float(calibration.false_confident_rate)}",
                f"- Over-flag rate: {_format_optional_float(calibration.over_flag_rate)}",
                f"- Coverage: {_format_optional_float(calibration.coverage)}",
                (
                    f"- Serious spans: TP {calibration.serious.true_positive}, "
                    f"FP {calibration.serious.false_positive}, "
                    f"FN {calibration.serious.false_negative}, "
                    f"TN {calibration.serious.true_negative}"
                ),
                (
                    f"- Minor spans: TP {calibration.minor.true_positive}, "
                    f"FP {calibration.minor.false_positive}, "
                    f"FN {calibration.minor.false_negative}, "
                    f"TN {calibration.minor.true_negative}"
                ),
                "",
            ]
        )
    if report.critical_span_diagnostic is not None:
        diagnostic = report.critical_span_diagnostic
        lines.extend(
            [
                "## Critical Span Diagnostic",
                "",
                f"- Policy: {diagnostic.policy_id}@{diagnostic.version}",
                f"- Status: {diagnostic.status}",
                f"- Recall: {_format_optional_float(diagnostic.recall)}",
                f"- Missed critical spans: {diagnostic.missed_critical_span_count}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def read_benchmark_report_json(path: str | Path) -> BenchmarkReport:
    return benchmark_report_json_loads(Path(path).read_text(encoding="utf-8"))


def write_benchmark_report_json(path: str | Path, report: BenchmarkReport) -> None:
    Path(path).write_text(benchmark_report_json_dumps(report), encoding="utf-8", newline="\n")


def write_benchmark_report_markdown(path: str | Path, report: BenchmarkReport) -> None:
    Path(path).write_text(benchmark_report_to_markdown(report), encoding="utf-8", newline="\n")


def benchmark_report_from_dict(data: Mapping[str, Any]) -> BenchmarkReport:
    if not isinstance(data, Mapping):
        raise ValidationError("benchmark report data must be an object")
    gate_config = _gate_config_from_dict(_required(data, "gate_config", "benchmark_report"))
    report = BenchmarkReport(
        report_id=_required(data, "report_id", "benchmark_report"),
        status=_required(data, "status", "benchmark_report"),
        gate_config=gate_config,
        corpus_results=tuple(
            _metric_result_from_dict(item) for item in _sequence(data.get("corpus_results", ()))
        ),
        branch_results=tuple(
            _metric_result_from_dict(item) for item in _sequence(data.get("branch_results", ()))
        ),
        recording_results=tuple(
            _metric_result_from_dict(item) for item in _sequence(data.get("recording_results", ()))
        ),
        slice_results=tuple(
            _metric_result_from_dict(item) for item in _sequence(data.get("slice_results", ()))
        ),
        review_signal_calibration=(
            _review_calibration_from_dict(data["review_signal_calibration"])
            if data.get("review_signal_calibration") is not None
            else None
        ),
        review_signal_scope_calibrations=tuple(
            _review_scope_calibration_from_dict(item)
            for item in _sequence(data.get("review_signal_scope_calibrations", ()))
        ),
        critical_span_policy=(
            _critical_span_policy_from_dict(data["critical_span_policy"])
            if data.get("critical_span_policy") is not None
            else None
        ),
        critical_span_diagnostic=(
            _critical_span_score_from_dict(data["critical_span_diagnostic"])
            if data.get("critical_span_diagnostic") is not None
            else None
        ),
        schema_version=_required(data, "schema_version", "benchmark_report"),
    )
    _validate_serialized_gates(gate_config, report.metric_results)
    _validate_serialized_critical_span_diagnostic(
        report.critical_span_policy,
        report.critical_span_diagnostic,
    )
    return report


def _observations_from_case(case: BenchmarkEvaluationCase) -> tuple[_MetricObservation, ...]:
    baseline_recording_metrics = (
        next(
            (row.metrics for row in case.baseline_evaluation.recording_metrics if row.status == "scored"),
            {},
        )
        if case.baseline_evaluation is not None
        else {}
    )
    observations: list[_MetricObservation] = []
    for row in case.evaluation.recording_metrics:
        if row.status != "scored":
            continue
        observations.extend(
            _observations_from_recording_row(
                case,
                row,
                baseline_recording_metrics,
            )
        )
    baseline_slice_metrics = _baseline_slice_metrics(case.baseline_evaluation)
    for row in case.evaluation.slice_metrics:
        if row.status != "scored":
            continue
        observations.extend(_observations_from_slice_row(case, row, baseline_slice_metrics.get(row.slice_id, {})))
    return tuple(observations)


def _observations_from_recording_row(
    case: BenchmarkEvaluationCase,
    row: DiarizationRecordingMetricRow,
    baseline_metrics: Mapping[str, Any],
) -> tuple[_MetricObservation, ...]:
    return tuple(
        _MetricObservation(
            corpus_id=case.corpus_id,
            branch_id=case.branch_id,
            recording_id=row.recording_id,
            metric_name=metric_name,
            point_score=point_score,
            baseline_score=_optional_metric_number(baseline_metrics, metric_name),
            scored_duration_ms=case.scored_duration_ms,
            scored_words=case.scored_words,
            scored_speaker_turns=case.scored_speaker_turns,
            source_scope="recording",
        )
        for metric_name, point_score in _numeric_metric_items(row.metrics)
    )


def _observations_from_slice_row(
    case: BenchmarkEvaluationCase,
    row: DiarizationSliceMetricRow,
    baseline_metrics: Mapping[str, Any],
) -> tuple[_MetricObservation, ...]:
    return tuple(
        _MetricObservation(
            corpus_id=case.corpus_id,
            branch_id=case.branch_id,
            recording_id=row.recording_id,
            metric_name=metric_name,
            point_score=point_score,
            baseline_score=_optional_metric_number(baseline_metrics, metric_name),
            scored_duration_ms=row.support_ms,
            scored_words=case.slice_scored_words.get(row.slice_id, 0),  # type: ignore[union-attr]
            scored_speaker_turns=case.slice_scored_speaker_turns.get(row.slice_id, 0),  # type: ignore[union-attr]
            source_scope="slice",
            slice_id=row.slice_id,
        )
        for metric_name, point_score in _numeric_metric_items(row.metrics)
    )


def _baseline_slice_metrics(evaluation: DiarizationEvaluationResult | None) -> dict[str, Mapping[str, Any]]:
    if evaluation is None:
        return {}
    return {row.slice_id: row.metrics for row in evaluation.slice_metrics if row.status == "scored"}


def _metric_results_for_scope(
    observations: tuple[_MetricObservation, ...],
    scope_type: BenchmarkReportScopeType,
    gate_config: BenchmarkGateConfig,
) -> tuple[BenchmarkMetricResult, ...]:
    grouped: dict[tuple[str, ...], list[_MetricObservation]] = {}
    for observation in observations:
        if scope_type == "slice":
            if observation.source_scope != "slice" or observation.slice_id is None:
                continue
            key = (
                observation.corpus_id,
                observation.branch_id,
                observation.slice_id,
                observation.metric_name,
            )
        else:
            if observation.source_scope != "recording":
                continue
            if scope_type == "corpus":
                key = (observation.corpus_id, observation.metric_name)
            elif scope_type == "branch":
                key = (observation.corpus_id, observation.branch_id, observation.metric_name)
            else:
                key = (
                    observation.corpus_id,
                    observation.branch_id,
                    observation.recording_id,
                    observation.metric_name,
                )
        grouped.setdefault(key, []).append(observation)

    results = [
        _metric_result_from_observations(scope_type, key, tuple(items), gate_config)
        for key, items in grouped.items()
    ]
    return _sort_metric_results(tuple(results))


def _sort_metric_results(results: tuple[BenchmarkMetricResult, ...]) -> tuple[BenchmarkMetricResult, ...]:
    return tuple(sorted(results, key=lambda item: (item.scope_type, item.scope_id, item.metric_name)))


def _review_signal_scope_calibrations(
    signals: tuple[ReviewSignalSpan, ...],
) -> tuple[ReviewSignalScopeCalibration, ...]:
    groups: dict[tuple[BenchmarkReportScopeType, tuple[str, ...]], list[ReviewSignalSpan]] = {}
    for signal in signals:
        groups.setdefault(("corpus", (signal.corpus_id,)), []).append(signal)
        groups.setdefault(("branch", (signal.corpus_id, signal.branch_id)), []).append(signal)
        groups.setdefault(("recording", (signal.corpus_id, signal.branch_id, signal.recording_id)), []).append(signal)

    calibrations: list[ReviewSignalScopeCalibration] = []
    for (scope_type, key), grouped_signals in groups.items():
        scope_id, corpus_id, branch_id, recording_id = _review_signal_scope_fields(scope_type, key)
        calibrations.append(
            ReviewSignalScopeCalibration(
                scope_type=scope_type,
                scope_id=scope_id,
                calibration=calibrate_review_signals(tuple(grouped_signals)),
                scored_duration_ms=sum(signal.end_ms - signal.start_ms for signal in grouped_signals),
                corpus_id=corpus_id,
                branch_id=branch_id,
                recording_id=recording_id,
            )
        )
    return tuple(
        sorted(
            calibrations,
            key=lambda item: (item.scope_type, item.scope_id),
        )
    )


def _review_signal_metric_results(
    scope_calibrations: tuple[ReviewSignalScopeCalibration, ...],
    gate_config: BenchmarkGateConfig,
) -> tuple[BenchmarkMetricResult, ...]:
    results: list[BenchmarkMetricResult] = []
    for scope_calibration in scope_calibrations:
        calibration = scope_calibration.calibration
        for metric_name, point_score in _review_signal_metric_values(scope_calibration.calibration):
            result_without_gate = BenchmarkMetricResult(
                scope_type=scope_calibration.scope_type,
                scope_id=scope_calibration.scope_id,
                metric_name=metric_name,
                point_score=point_score,
                sample_count=calibration.total,
                scored_duration_ms=scope_calibration.scored_duration_ms,
                scored_words=0,
                scored_speaker_turns=0,
                gate=RegressionGateResult(status="unavailable", reasons=("gate not evaluated",)),
                uncertainty=UncertaintyInterval(
                    status="unavailable",
                    basis="review_signal_metric",
                    reason="review-signal metrics do not have paired samples",
                ),
                corpus_id=scope_calibration.corpus_id,
                branch_id=scope_calibration.branch_id,
                recording_id=scope_calibration.recording_id,
            )
            results.append(
                BenchmarkMetricResult(
                    scope_type=result_without_gate.scope_type,
                    scope_id=result_without_gate.scope_id,
                    metric_name=result_without_gate.metric_name,
                    point_score=result_without_gate.point_score,
                    sample_count=result_without_gate.sample_count,
                    scored_duration_ms=result_without_gate.scored_duration_ms,
                    scored_words=result_without_gate.scored_words,
                    scored_speaker_turns=result_without_gate.scored_speaker_turns,
                    gate=_evaluate_gate(result_without_gate, gate_config),
                    uncertainty=result_without_gate.uncertainty,
                    corpus_id=result_without_gate.corpus_id,
                    branch_id=result_without_gate.branch_id,
                    recording_id=result_without_gate.recording_id,
                )
            )
    return _sort_metric_results(tuple(results))


def _review_signal_scope_fields(
    scope_type: BenchmarkReportScopeType,
    key: tuple[str, ...],
) -> tuple[str, str, str | None, str | None]:
    if scope_type == "corpus":
        (corpus_id,) = key
        return corpus_id, corpus_id, None, None
    if scope_type == "branch":
        corpus_id, branch_id = key
        return f"{corpus_id}/{branch_id}", corpus_id, branch_id, None
    if scope_type == "recording":
        corpus_id, branch_id, recording_id = key
        return f"{corpus_id}/{branch_id}/{recording_id}", corpus_id, branch_id, recording_id
    raise ValidationError(f"review-signal scope is not supported: {scope_type}")


def _review_signal_metric_values(calibration: ReviewSignalCalibration) -> tuple[tuple[str, float], ...]:
    return tuple(
        (metric_name, value)
        for metric_name in _REVIEW_SIGNAL_METRIC_NAMES
        if (value := getattr(calibration, metric_name)) is not None
    )


def _metric_result_from_observations(
    scope_type: BenchmarkReportScopeType,
    key: tuple[str, ...],
    observations: tuple[_MetricObservation, ...],
    gate_config: BenchmarkGateConfig,
) -> BenchmarkMetricResult:
    weighted_scores = tuple((item.point_score, _observation_weight(item)) for item in observations)
    point_score = _weighted_mean(weighted_scores)
    weighted_baseline_scores = tuple(
        (item.baseline_score, _observation_weight(item))
        for item in observations
        if item.baseline_score is not None
    )
    baseline_score = (
        _weighted_mean(weighted_baseline_scores)
        if len(weighted_baseline_scores) == len(observations)
        else None
    )
    paired_delta = point_score - baseline_score if baseline_score is not None else None
    paired_deltas = tuple(
        item.point_score - item.baseline_score
        for item in observations
        if item.baseline_score is not None
    )
    metric_name = observations[0].metric_name
    scope_id, corpus_id, branch_id, recording_id, slice_id = _scope_fields(scope_type, key)
    result_without_gate = BenchmarkMetricResult(
        scope_type=scope_type,
        scope_id=scope_id,
        metric_name=metric_name,
        point_score=_round_metric(point_score),
        baseline_score=_round_optional_metric(baseline_score),
        paired_delta=_round_optional_metric(paired_delta),
        sample_count=len(observations),
        scored_duration_ms=sum(item.scored_duration_ms for item in observations),
        scored_words=sum(item.scored_words for item in observations),
        scored_speaker_turns=sum(item.scored_speaker_turns for item in observations),
        gate=RegressionGateResult(status="unavailable", reasons=("gate not evaluated",)),
        uncertainty=(
            _uncertainty_interval(paired_deltas, basis="paired_delta")
            if len(paired_deltas) == len(observations)
            else _uncertainty_interval(tuple(item.point_score for item in observations), basis="point_score")
        ),
        corpus_id=corpus_id,
        branch_id=branch_id,
        recording_id=recording_id,
        slice_id=slice_id,
    )
    gate = _evaluate_gate(result_without_gate, gate_config)
    return BenchmarkMetricResult(
        scope_type=result_without_gate.scope_type,
        scope_id=result_without_gate.scope_id,
        metric_name=result_without_gate.metric_name,
        point_score=result_without_gate.point_score,
        sample_count=result_without_gate.sample_count,
        scored_duration_ms=result_without_gate.scored_duration_ms,
        scored_words=result_without_gate.scored_words,
        scored_speaker_turns=result_without_gate.scored_speaker_turns,
        gate=gate,
        uncertainty=result_without_gate.uncertainty,
        baseline_score=result_without_gate.baseline_score,
        paired_delta=result_without_gate.paired_delta,
        corpus_id=result_without_gate.corpus_id,
        branch_id=result_without_gate.branch_id,
        recording_id=result_without_gate.recording_id,
        slice_id=result_without_gate.slice_id,
    )


def _scope_fields(
    scope_type: BenchmarkReportScopeType,
    key: tuple[str, ...],
) -> tuple[str, str | None, str | None, str | None, str | None]:
    if scope_type == "corpus":
        corpus_id, _metric_name = key
        return corpus_id, corpus_id, None, None, None
    if scope_type == "branch":
        corpus_id, branch_id, _metric_name = key
        return f"{corpus_id}/{branch_id}", corpus_id, branch_id, None, None
    if scope_type == "recording":
        corpus_id, branch_id, recording_id, _metric_name = key
        return f"{corpus_id}/{branch_id}/{recording_id}", corpus_id, branch_id, recording_id, None
    corpus_id, branch_id, slice_id, _metric_name = key
    return f"{corpus_id}/{branch_id}/{slice_id}", corpus_id, branch_id, None, slice_id


def _evaluate_gate(result: BenchmarkMetricResult, gate_config: BenchmarkGateConfig) -> RegressionGateResult:
    budget = _matching_budget(result, gate_config)
    if budget is None:
        return RegressionGateResult(status="unavailable", reasons=("no regression budget configured",))
    reasons: list[str] = []
    thresholds: dict[str, float] = {}
    if budget.min_point_score is not None:
        thresholds["min_point_score"] = budget.min_point_score
        if result.point_score < budget.min_point_score:
            reasons.append(f"point score {result.point_score} below minimum {budget.min_point_score}")
    if budget.max_point_score is not None:
        thresholds["max_point_score"] = budget.max_point_score
        if result.point_score > budget.max_point_score:
            reasons.append(f"point score {result.point_score} above maximum {budget.max_point_score}")
    if budget.max_regression_delta is not None:
        thresholds["max_regression_delta"] = budget.max_regression_delta
        if result.paired_delta is None:
            if reasons:
                return RegressionGateResult(
                    status="failed",
                    budget_id=budget.budget_id,
                    reasons=tuple([*reasons, "paired delta unavailable for regression budget"]),
                    thresholds=thresholds,
                )
            return RegressionGateResult(
                status="unavailable",
                budget_id=budget.budget_id,
                reasons=("paired delta unavailable for regression budget",),
                thresholds=thresholds,
            )
        if budget.direction == "higher_is_better" and result.paired_delta < -budget.max_regression_delta:
            reasons.append(
                f"paired delta {result.paired_delta} exceeds regression budget {budget.max_regression_delta}"
            )
        if budget.direction == "lower_is_better" and result.paired_delta > budget.max_regression_delta:
            reasons.append(
                f"paired delta {result.paired_delta} exceeds regression budget {budget.max_regression_delta}"
            )
    if reasons:
        return RegressionGateResult(
            status="failed",
            budget_id=budget.budget_id,
            reasons=tuple(reasons),
            thresholds=thresholds,
        )
    return RegressionGateResult(status="passed", budget_id=budget.budget_id, thresholds=thresholds)


def _matching_budget(
    result: BenchmarkMetricResult,
    gate_config: BenchmarkGateConfig,
) -> BenchmarkRegressionBudget | None:
    matches = [
        budget
        for budget in gate_config.budgets
        if _budget_matches_result(budget, result)
    ]
    if not matches:
        return None
    return max(matches, key=_budget_specificity)


def _budget_matches_result(budget: BenchmarkRegressionBudget, result: BenchmarkMetricResult) -> bool:
    return (
        budget.metric_name == result.metric_name
        and (budget.scope_type is None or budget.scope_type == result.scope_type)
        and (budget.scope_id is None or budget.scope_id == result.scope_id)
        and (budget.slice_id is None or budget.slice_id == result.slice_id)
    )


def _budget_specificity(budget: BenchmarkRegressionBudget) -> tuple[bool, bool, bool, int]:
    fields = (budget.scope_type, budget.scope_id, budget.slice_id)
    return (
        budget.scope_id is not None,
        budget.slice_id is not None,
        budget.scope_type is not None,
        sum(value is not None for value in fields),
    )


def _validate_all_budgets_matched(
    gate_config: BenchmarkGateConfig,
    results: tuple[BenchmarkMetricResult, ...],
) -> None:
    matched_budget_ids = {
        result.gate.budget_id
        for result in results
        if result.gate.budget_id is not None
    }
    unmatched_budget_ids = {
        budget.budget_id
        for budget in gate_config.budgets
        if budget.budget_id not in matched_budget_ids
        and not _budget_is_shadowed_by_more_specific_match(budget, gate_config, results)
    }
    if unmatched_budget_ids:
        raise ValidationError(
            "regression budgets did not match any metric result: "
            + ", ".join(sorted(unmatched_budget_ids))
        )


def _budget_is_shadowed_by_more_specific_match(
    budget: BenchmarkRegressionBudget,
    gate_config: BenchmarkGateConfig,
    results: tuple[BenchmarkMetricResult, ...],
) -> bool:
    covered_results = tuple(result for result in results if _budget_matches_result(budget, result))
    if not covered_results:
        return False
    return all(
        (selected := _matching_budget(result, gate_config)) is not None
        and selected.budget_id != budget.budget_id
        and _budget_specificity(selected) > _budget_specificity(budget)
        for result in covered_results
    )


def _validate_serialized_gates(
    gate_config: BenchmarkGateConfig,
    results: tuple[BenchmarkMetricResult, ...],
) -> None:
    _validate_all_budgets_matched(gate_config, results)
    for result in results:
        expected_gate = _evaluate_gate(result, gate_config)
        if result.gate.to_dict() != expected_gate.to_dict():
            raise ValidationError(
                "metric_result gate does not match regression budget: "
                f"{result.scope_id}/{result.metric_name}"
            )


def _validate_review_signal_summary(
    *,
    total: int,
    assessed: int,
    true_positive: int,
    false_positive: int,
    false_negative: int,
    true_negative: int,
    precision: float | None,
    recall: float | None,
    false_confident_rate: float | None,
    over_flag_rate: float | None,
    coverage: float,
    context: str,
) -> None:
    if assessed != true_positive + false_positive + false_negative + true_negative:
        raise ValidationError(f"{context}.assessed must match confusion-matrix counts")
    if assessed > total:
        raise ValidationError(f"{context}.assessed must be <= total")
    expected_rates = {
        "precision": _rate(true_positive, true_positive + false_positive),
        "recall": _rate(true_positive, true_positive + false_negative),
        "false_confident_rate": _rate(false_negative, true_positive + false_negative),
        "over_flag_rate": _rate(false_positive, false_positive + true_negative),
        "coverage": _rate(assessed, total) or 0.0,
    }
    actual_rates = {
        "precision": precision,
        "recall": recall,
        "false_confident_rate": false_confident_rate,
        "over_flag_rate": over_flag_rate,
        "coverage": coverage,
    }
    for field_name, expected_value in expected_rates.items():
        if actual_rates[field_name] != expected_value:
            raise ValidationError(f"{context}.{field_name} must match confusion-matrix counts")


def _validate_review_signal_breakdown_totals(calibration: ReviewSignalCalibration) -> None:
    breakdowns = (calibration.serious, calibration.minor)
    for field_name in ("total", "assessed", "true_positive", "false_positive", "false_negative", "true_negative"):
        breakdown_total = sum(getattr(breakdown, field_name) for breakdown in breakdowns)
        if getattr(calibration, field_name) != breakdown_total:
            raise ValidationError(f"review_calibration.{field_name} must match severity breakdown totals")


def _validate_metric_result_scope_identity(result: BenchmarkMetricResult) -> None:
    if result.scope_type == "corpus":
        if result.corpus_id is None:
            raise ValidationError("corpus metric results require corpus_id")
        if result.branch_id is not None or result.recording_id is not None or result.slice_id is not None:
            raise ValidationError("corpus metric results cannot include branch, recording, or slice ids")
        expected_scope_id = result.corpus_id
    elif result.scope_type == "branch":
        if result.corpus_id is None or result.branch_id is None:
            raise ValidationError("branch metric results require corpus_id and branch_id")
        if result.recording_id is not None or result.slice_id is not None:
            raise ValidationError("branch metric results cannot include recording or slice ids")
        expected_scope_id = f"{result.corpus_id}/{result.branch_id}"
    elif result.scope_type == "recording":
        if result.corpus_id is None or result.branch_id is None or result.recording_id is None:
            raise ValidationError("recording metric results require corpus_id, branch_id, and recording_id")
        if result.slice_id is not None:
            raise ValidationError("recording metric results cannot include slice_id")
        expected_scope_id = f"{result.corpus_id}/{result.branch_id}/{result.recording_id}"
    else:
        if result.corpus_id is None or result.branch_id is None or result.slice_id is None:
            raise ValidationError("slice metric results require corpus_id, branch_id, and slice_id")
        if result.recording_id is not None:
            raise ValidationError("slice metric results cannot include recording_id")
        expected_scope_id = f"{result.corpus_id}/{result.branch_id}/{result.slice_id}"
    if result.scope_id != expected_scope_id:
        raise ValidationError("metric_result.scope_id must match its scope identifiers")


def _validate_unique_metric_results(results: tuple[BenchmarkMetricResult, ...]) -> None:
    seen: set[tuple[BenchmarkReportScopeType, str, str]] = set()
    for result in results:
        key = _metric_result_key(result)
        if key in seen:
            raise ValidationError(f"duplicate metric result: {result.scope_id}/{result.metric_name}")
        seen.add(key)


def _validate_review_signals_match_cases(
    cases: tuple[BenchmarkEvaluationCase, ...],
    signals: tuple[ReviewSignalSpan, ...],
) -> None:
    allowed = {
        (case.corpus_id, case.branch_id, case.evaluation.recording_id)
        for case in cases
    }
    for signal in signals:
        if (signal.corpus_id, signal.branch_id, signal.recording_id) not in allowed:
            raise ValidationError("review_signals must match evaluated benchmark cases")


def _validate_review_signal_scope_identity(scope_calibration: ReviewSignalScopeCalibration) -> None:
    if scope_calibration.scope_type == "corpus":
        if scope_calibration.corpus_id is None:
            raise ValidationError("review_scope_calibration.corpus_id is required for corpus scopes")
        if scope_calibration.branch_id is not None or scope_calibration.recording_id is not None:
            raise ValidationError("corpus review-signal scopes cannot include branch or recording ids")
        expected_scope_id = scope_calibration.corpus_id
    elif scope_calibration.scope_type == "branch":
        if scope_calibration.corpus_id is None or scope_calibration.branch_id is None:
            raise ValidationError("review_scope_calibration branch scopes require corpus_id and branch_id")
        if scope_calibration.recording_id is not None:
            raise ValidationError("branch review-signal scopes cannot include recording_id")
        expected_scope_id = f"{scope_calibration.corpus_id}/{scope_calibration.branch_id}"
    else:
        if (
            scope_calibration.corpus_id is None
            or scope_calibration.branch_id is None
            or scope_calibration.recording_id is None
        ):
            raise ValidationError(
                "review_scope_calibration recording scopes require corpus_id, branch_id, and recording_id"
            )
        expected_scope_id = (
            f"{scope_calibration.corpus_id}/{scope_calibration.branch_id}/{scope_calibration.recording_id}"
        )
    if scope_calibration.scope_id != expected_scope_id:
        raise ValidationError("review_scope_calibration.scope_id must match its scope identifiers")


def _validate_serialized_review_signal_metric_results(
    calibration: ReviewSignalCalibration | None,
    scope_calibrations: tuple[ReviewSignalScopeCalibration, ...],
    gate_config: BenchmarkGateConfig,
    results: tuple[BenchmarkMetricResult, ...],
) -> None:
    if calibration is None:
        if scope_calibrations:
            raise ValidationError("review_signal_scope_calibrations require review_signal_calibration")
        if any(_is_serialized_review_signal_metric_result(result) for result in results):
            raise ValidationError("review-signal metric results require review_signal_calibration")
        return
    if not scope_calibrations:
        raise ValidationError("review_signal_calibration requires review_signal_scope_calibrations")
    _validate_review_signal_scope_calibrations(calibration, scope_calibrations)
    expected = _review_signal_metric_results(scope_calibrations, gate_config)
    evaluated_scopes = {
        (result.scope_type, result.scope_id)
        for result in results
        if result.scope_type in {"corpus", "branch", "recording"}
        and not _is_serialized_review_signal_metric_result(result)
    }
    for scope_calibration in scope_calibrations:
        if (scope_calibration.scope_type, scope_calibration.scope_id) not in evaluated_scopes:
            raise ValidationError("review_signal_scope_calibrations must match evaluated metric scopes")
    expected_by_key = {_metric_result_key(result): result.to_dict() for result in expected}
    actual_review_signal_results = tuple(
        result
        for result in results
        if _is_serialized_review_signal_metric_result(result)
    )
    actual_by_key: dict[tuple[BenchmarkReportScopeType, str, str], dict[str, Any]] = {}
    for result in actual_review_signal_results:
        key = _metric_result_key(result)
        if key in actual_by_key:
            raise ValidationError("duplicate review-signal metric result")
        actual_by_key[key] = result.to_dict()
    if set(actual_by_key) != set(expected_by_key):
        raise ValidationError("review-signal metric results must match review_signal_scope_calibrations")
    for key, expected_payload in expected_by_key.items():
        if actual_by_key[key] != expected_payload:
            raise ValidationError("review-signal metric result does not match review_signal_scope_calibrations")


def _validate_review_signal_scope_calibrations(
    calibration: ReviewSignalCalibration,
    scope_calibrations: tuple[ReviewSignalScopeCalibration, ...],
) -> None:
    seen: set[tuple[BenchmarkReportScopeType, str]] = set()
    by_scope_type: dict[BenchmarkReportScopeType, list[ReviewSignalScopeCalibration]] = {}
    for scope_calibration in scope_calibrations:
        key = (scope_calibration.scope_type, scope_calibration.scope_id)
        if key in seen:
            raise ValidationError(f"duplicate review_signal_scope_calibration: {scope_calibration.scope_id}")
        seen.add(key)
        by_scope_type.setdefault(scope_calibration.scope_type, []).append(scope_calibration)
    for scope_type in ("corpus", "branch", "recording"):
        scoped = tuple(by_scope_type.get(scope_type, ()))
        if not scoped:
            raise ValidationError(f"review_signal_scope_calibrations require at least one {scope_type} scope")
        _validate_review_signal_scope_totals(calibration, scoped, scope_type)


def _validate_review_signal_scope_totals(
    calibration: ReviewSignalCalibration,
    scope_calibrations: tuple[ReviewSignalScopeCalibration, ...],
    scope_type: BenchmarkReportScopeType,
) -> None:
    for field_name in ("total", "assessed", "true_positive", "false_positive", "false_negative", "true_negative"):
        scoped_total = sum(getattr(item.calibration, field_name) for item in scope_calibrations)
        if getattr(calibration, field_name) != scoped_total:
            raise ValidationError(f"review_calibration.{field_name} must match {scope_type} scope totals")


def _metric_result_key(result: BenchmarkMetricResult) -> tuple[BenchmarkReportScopeType, str, str]:
    return result.scope_type, result.scope_id, result.metric_name


def _is_serialized_review_signal_metric_result(result: BenchmarkMetricResult) -> bool:
    return (
        result.uncertainty.basis == "review_signal_metric"
        or result.metric_name in {"false_confident_rate", "over_flag_rate"}
    )


def _validate_serialized_critical_span_diagnostic(
    policy: CriticalSpanPolicyDefinition | None,
    diagnostic: CriticalSpanDiagnosticScore | None,
) -> None:
    if policy is None:
        if diagnostic is not None:
            raise ValidationError("critical_span_diagnostic requires a critical_span_policy")
        return
    if diagnostic is None:
        raise ValidationError("critical_span_policy requires critical_span_diagnostic")
    if diagnostic.policy_id != policy.policy_id or diagnostic.version != policy.version:
        raise ValidationError("critical_span_diagnostic policy identity must match critical_span_policy")
    if (
        diagnostic.detected_critical_span_count + diagnostic.missed_critical_span_count
        != diagnostic.critical_span_count
    ):
        raise ValidationError("critical_span_diagnostic counts must add up to critical_span_count")
    expected_recall = (
        None
        if diagnostic.critical_span_count == 0
        else _round_metric(diagnostic.detected_critical_span_count / diagnostic.critical_span_count)
    )
    if diagnostic.recall != expected_recall:
        raise ValidationError("critical_span_diagnostic recall must match detected critical spans")
    expected_status: RegressionGateStatus
    if diagnostic.critical_span_count == 0:
        expected_status = "unavailable"
    elif expected_recall is not None and expected_recall < policy.minimum_recall:
        expected_status = "failed"
    else:
        expected_status = "passed"
    if diagnostic.status != expected_status:
        raise ValidationError("critical_span_diagnostic status must match critical span policy")


def _uncertainty_interval(values: tuple[float, ...], *, basis: str) -> UncertaintyInterval:
    if len(values) < 2:
        return UncertaintyInterval(
            status="unavailable",
            basis=basis,
            reason="requires at least two paired samples",
        )
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    stderr = math.sqrt(variance) / math.sqrt(len(values))
    half_width = 1.96 * stderr
    return UncertaintyInterval(
        status="available",
        lower=_round_metric(mean - half_width),
        upper=_round_metric(mean + half_width),
        basis=basis,
    )


def _review_signal_breakdown(
    signals: tuple[ReviewSignalSpan, ...],
    severity: ReviewSignalSeverity,
) -> ReviewSignalSeverityBreakdown:
    counts = _review_signal_counts(signals)
    return ReviewSignalSeverityBreakdown(
        severity=severity,
        total=counts["total"],
        assessed=counts["assessed"],
        true_positive=counts["tp"],
        false_positive=counts["fp"],
        false_negative=counts["fn"],
        true_negative=counts["tn"],
        precision=_rate(counts["tp"], counts["tp"] + counts["fp"]),
        recall=_rate(counts["tp"], counts["tp"] + counts["fn"]),
        false_confident_rate=_rate(counts["fn"], counts["tp"] + counts["fn"]),
        over_flag_rate=_rate(counts["fp"], counts["fp"] + counts["tn"]),
        coverage=_rate(counts["assessed"], counts["total"]) or 0.0,
    )


def _review_signal_counts(signals: tuple[ReviewSignalSpan, ...]) -> dict[str, int]:
    total = len(signals)
    assessed = sum(1 for signal in signals if signal.predicted_review_required is not None)
    true_positive = sum(
        1
        for signal in signals
        if signal.predicted_review_required is True and signal.reference_review_required is True
    )
    false_positive = sum(
        1
        for signal in signals
        if signal.predicted_review_required is True and signal.reference_review_required is False
    )
    false_negative = sum(
        1
        for signal in signals
        if signal.predicted_review_required is False and signal.reference_review_required is True
    )
    true_negative = sum(
        1
        for signal in signals
        if signal.predicted_review_required is False and signal.reference_review_required is False
    )
    return {
        "assessed": assessed,
        "fn": false_negative,
        "fp": false_positive,
        "tn": true_negative,
        "total": total,
        "tp": true_positive,
    }


def _is_critical_span(signal: ReviewSignalSpan, policy: CriticalSpanPolicyDefinition) -> bool:
    if not signal.reference_review_required:
        return False
    severity_match = signal.severity in policy.critical_severities
    label_match = bool(set(signal.labels).intersection(policy.critical_labels))
    return severity_match or label_match


def _has_failed_gate(
    results: tuple[BenchmarkMetricResult, ...],
    critical_score: CriticalSpanDiagnosticScore | None,
) -> bool:
    if any(
        result.gate.status == "failed" or (result.gate.budget_id is not None and result.gate.status == "unavailable")
        for result in results
    ):
        return True
    return critical_score is not None and critical_score.status != "passed"


def _markdown_metric_table(title: str, results: tuple[BenchmarkMetricResult, ...]) -> list[str]:
    lines = [f"## {title}", ""]
    if not results:
        lines.extend(["No results.", ""])
        return lines
    lines.append("| Scope | Metric | Point | Delta | Gate | Samples | Duration ms | Words | Turns |")
    lines.append("| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |")
    for result in results:
        lines.append(
            "| "
            + " | ".join(
                (
                    result.scope_id,
                    result.metric_name,
                    _format_optional_float(result.point_score),
                    _format_optional_float(result.paired_delta),
                    result.gate.status,
                    str(result.sample_count),
                    str(result.scored_duration_ms),
                    str(result.scored_words),
                    str(result.scored_speaker_turns),
                )
            )
            + " |"
        )
    lines.append("")
    return lines


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6g}"


def _metric_result_from_dict(data: Mapping[str, Any]) -> BenchmarkMetricResult:
    return BenchmarkMetricResult(
        scope_type=_required(data, "scope_type", "metric_result"),
        scope_id=_required(data, "scope_id", "metric_result"),
        metric_name=_required(data, "metric_name", "metric_result"),
        point_score=_required(data, "point_score", "metric_result"),
        sample_count=_required(data, "sample_count", "metric_result"),
        scored_duration_ms=_required(data, "scored_duration_ms", "metric_result"),
        scored_words=_required(data, "scored_words", "metric_result"),
        scored_speaker_turns=_required(data, "scored_speaker_turns", "metric_result"),
        gate=_gate_result_from_dict(_required(data, "gate", "metric_result")),
        uncertainty=_uncertainty_from_dict(_required(data, "uncertainty", "metric_result")),
        baseline_score=data.get("baseline_score"),
        paired_delta=data.get("paired_delta"),
        corpus_id=data.get("corpus_id"),
        branch_id=data.get("branch_id"),
        recording_id=data.get("recording_id"),
        slice_id=data.get("slice_id"),
    )


def _gate_config_from_dict(data: Mapping[str, Any]) -> BenchmarkGateConfig:
    return BenchmarkGateConfig(
        budgets=tuple(_budget_from_dict(item) for item in _sequence(_required(data, "budgets", "gate_config")))
    )


def _budget_from_dict(data: Mapping[str, Any]) -> BenchmarkRegressionBudget:
    return BenchmarkRegressionBudget(
        budget_id=_required(data, "budget_id", "regression_budget"),
        metric_name=_required(data, "metric_name", "regression_budget"),
        budget_kind=data.get("budget_kind", "metric"),
        direction=data.get("direction", "higher_is_better"),
        max_regression_delta=data.get("max_regression_delta"),
        min_point_score=data.get("min_point_score"),
        max_point_score=data.get("max_point_score"),
        scope_type=data.get("scope_type"),
        scope_id=data.get("scope_id"),
        slice_id=data.get("slice_id"),
    )


def _gate_result_from_dict(data: Mapping[str, Any]) -> RegressionGateResult:
    return RegressionGateResult(
        status=_required(data, "status", "gate"),
        budget_id=data.get("budget_id"),
        reasons=tuple(_sequence(data.get("reasons", ()))),
        thresholds=data.get("thresholds", {}),
    )


def _uncertainty_from_dict(data: Mapping[str, Any]) -> UncertaintyInterval:
    return UncertaintyInterval(
        status=_required(data, "status", "uncertainty"),
        confidence_level=data.get("confidence_level", 0.95),
        lower=data.get("lower"),
        upper=data.get("upper"),
        basis=data.get("basis", "unavailable"),
        reason=data.get("reason"),
    )


def _review_calibration_from_dict(data: Mapping[str, Any]) -> ReviewSignalCalibration:
    return ReviewSignalCalibration(
        total=_required(data, "total", "review_calibration"),
        assessed=_required(data, "assessed", "review_calibration"),
        true_positive=_required(data, "true_positive", "review_calibration"),
        false_positive=_required(data, "false_positive", "review_calibration"),
        false_negative=_required(data, "false_negative", "review_calibration"),
        true_negative=_required(data, "true_negative", "review_calibration"),
        precision=data.get("precision"),
        recall=data.get("recall"),
        false_confident_rate=data.get("false_confident_rate"),
        over_flag_rate=data.get("over_flag_rate"),
        coverage=_required(data, "coverage", "review_calibration"),
        serious=_severity_breakdown_from_dict(_required(data, "serious", "review_calibration")),
        minor=_severity_breakdown_from_dict(_required(data, "minor", "review_calibration")),
    )


def _review_scope_calibration_from_dict(data: Mapping[str, Any]) -> ReviewSignalScopeCalibration:
    return ReviewSignalScopeCalibration(
        scope_type=_required(data, "scope_type", "review_scope_calibration"),
        scope_id=_required(data, "scope_id", "review_scope_calibration"),
        calibration=_review_calibration_from_dict(_required(data, "calibration", "review_scope_calibration")),
        scored_duration_ms=data.get("scored_duration_ms", 0),
        corpus_id=data.get("corpus_id"),
        branch_id=data.get("branch_id"),
        recording_id=data.get("recording_id"),
    )


def _severity_breakdown_from_dict(data: Mapping[str, Any]) -> ReviewSignalSeverityBreakdown:
    return ReviewSignalSeverityBreakdown(
        severity=_required(data, "severity", "severity_breakdown"),
        total=_required(data, "total", "severity_breakdown"),
        assessed=_required(data, "assessed", "severity_breakdown"),
        true_positive=_required(data, "true_positive", "severity_breakdown"),
        false_positive=_required(data, "false_positive", "severity_breakdown"),
        false_negative=_required(data, "false_negative", "severity_breakdown"),
        true_negative=_required(data, "true_negative", "severity_breakdown"),
        precision=data.get("precision"),
        recall=data.get("recall"),
        false_confident_rate=data.get("false_confident_rate"),
        over_flag_rate=data.get("over_flag_rate"),
        coverage=_required(data, "coverage", "severity_breakdown"),
    )


def _critical_span_policy_from_dict(data: Mapping[str, Any]) -> CriticalSpanPolicyDefinition:
    return CriticalSpanPolicyDefinition(
        policy_id=_required(data, "policy_id", "critical_span_policy"),
        version=_required(data, "version", "critical_span_policy"),
        description=_required(data, "description", "critical_span_policy"),
        critical_severities=tuple(_sequence(data.get("critical_severities", ("serious",)))),
        critical_labels=tuple(_sequence(data.get("critical_labels", ()))),
        minimum_recall=data.get("minimum_recall", 1.0),
        scope=data.get("scope", "diagnostic_fixture"),
    )


def _critical_span_score_from_dict(data: Mapping[str, Any]) -> CriticalSpanDiagnosticScore:
    return CriticalSpanDiagnosticScore(
        status=_required(data, "status", "critical_span_score"),
        policy_id=_required(data, "policy_id", "critical_span_score"),
        version=_required(data, "version", "critical_span_score"),
        critical_span_count=_required(data, "critical_span_count", "critical_span_score"),
        detected_critical_span_count=_required(data, "detected_critical_span_count", "critical_span_score"),
        missed_critical_span_count=_required(data, "missed_critical_span_count", "critical_span_score"),
        recall=data.get("recall"),
        failures=tuple(_sequence(data.get("failures", ()))),
    )


def _numeric_metric_items(metrics: Mapping[str, Any]) -> tuple[tuple[str, float], ...]:
    return tuple(
        (str(key), _finite_number(value, f"metric.{key}"))
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )


def _optional_metric_number(metrics: Mapping[str, Any], metric_name: str) -> float | None:
    if metric_name not in metrics:
        return None
    value = metrics[metric_name]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return _finite_number(value, f"baseline_metric.{metric_name}")


def _mean(values: tuple[float, ...]) -> float:
    if not values:
        raise ValidationError("cannot average an empty metric series")
    return sum(values) / len(values)


def _weighted_mean(values: tuple[tuple[float, int], ...]) -> float:
    if not values:
        raise ValidationError("cannot average an empty metric series")
    total_weight = sum(weight for _value, weight in values)
    if total_weight <= 0:
        return _mean(tuple(value for value, _weight in values))
    return sum(value * weight for value, weight in values) / total_weight


def _observation_weight(observation: _MetricObservation) -> int:
    if observation.scored_duration_ms > 0:
        return observation.scored_duration_ms
    if observation.scored_words > 0:
        return observation.scored_words
    if observation.scored_speaker_turns > 0:
        return observation.scored_speaker_turns
    return 1


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return _round_metric(numerator / denominator)


def _round_optional_metric(value: float | None) -> float | None:
    return None if value is None else _round_metric(value)


def _round_metric(value: float) -> float:
    _finite_number(value, "metric")
    return round(value, 6)


def _required(data: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in data:
        raise ValidationError(f"{context}.{key} is required")
    return data[key]


def _validate_report_schema_version(value: object) -> int:
    version = _positive_int(value, "benchmark_report.schema_version")
    if version != BENCHMARK_REPORT_SCHEMA_VERSION:
        raise ValidationError(f"benchmark_report.schema_version is not supported: {version}")
    return version


def _sequence(value: object) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValidationError("expected a JSON array")
    return tuple(value)


def _tuple_of(values: object, item_type: type[Any], field_name: str) -> tuple[Any, ...]:
    try:
        items = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValidationError(f"{field_name} must be an iterable") from exc
    for index, item in enumerate(items):
        if not isinstance(item, item_type):
            raise ValidationError(f"{field_name}[{index}] must be a {item_type.__name__}")
    return items


def _tuple_of_text(values: object, field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValidationError(f"{field_name} must be an iterable of strings")
    try:
        items = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValidationError(f"{field_name} must be an iterable") from exc
    return tuple(_require_text(item, f"{field_name}[{index}]") for index, item in enumerate(items))


def _validate_choice(value: object, allowed: set[str], field_name: str) -> Any:
    value = _require_text(value, field_name)
    if value not in allowed:
        raise ValidationError(f"{field_name} is not supported: {value}")
    return value


def _require_id(value: object, field_name: str) -> str:
    text = _require_text(value, field_name)
    normalized = text
    for character in ("_", "-", ".", ":", "/"):
        normalized = normalized.replace(character, "")
    if not normalized.isalnum():
        raise ValidationError(f"{field_name} must be an identifier")
    return text


def _optional_id(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_id(value, field_name)


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field_name} is required")
    return value


def _optional_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_text(value, field_name)


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValidationError(f"{field_name} must be a boolean")
    return value


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValidationError(f"{field_name} must be positive")
    return value


def _non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{field_name} must be an integer")
    if value < 0:
        raise ValidationError(f"{field_name} must be non-negative")
    return value


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValidationError(f"{field_name} must be a finite number")
    return float(value)


def _optional_finite_number(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    return _finite_number(value, field_name)


def _validate_probability(value: object, field_name: str, *, allow_zero: bool = True) -> float:
    number = _finite_number(value, field_name)
    if (number < 0 or number > 1) or (not allow_zero and number == 0):
        raise ValidationError(f"{field_name} must be between 0 and 1")
    return number


def _validate_number_map(values: Mapping[str, Any], field_name: str) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise ValidationError(f"{field_name} must be an object")
    return {str(key): _finite_number(value, f"{field_name}.{key}") for key, value in values.items()}


def _validate_int_map(values: Mapping[str, Any], field_name: str) -> dict[str, int]:
    if not isinstance(values, Mapping):
        raise ValidationError(f"{field_name} must be an object")
    return {str(key): _non_negative_int(value, f"{field_name}.{key}") for key, value in values.items()}
