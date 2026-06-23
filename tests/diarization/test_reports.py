import json

import pytest

from keyframe.diarization import (
    BENCHMARK_REPORT_SCHEMA_VERSION,
    BenchmarkEvaluationCase,
    BenchmarkGateConfig,
    BenchmarkMetricResult,
    BenchmarkRegressionBudget,
    BenchmarkReport,
    CriticalSpanPolicyDefinition,
    DiarizationEvaluationResult,
    DiarizationRecordingMetricRow,
    DiarizationSliceMetricRow,
    EvaluationInterval,
    EvaluationSliceDefinition,
    PreflightRouteAssessment,
    RegressionGateResult,
    ReviewSignalSpan,
    UncertaintyInterval,
    benchmark_report_json_dumps,
    benchmark_report_json_loads,
    benchmark_report_to_markdown,
    build_benchmark_report as _build_benchmark_report,
    calibrate_review_signals,
    read_benchmark_report_json,
    score_diagnostic_critical_spans,
    write_benchmark_report_json,
    write_benchmark_report_markdown,
)
from keyframe.diarization.models import ValidationError


def _evaluation(
    output_id,
    recording_id,
    *,
    recording_der,
    overlap_der,
    boundary_accuracy=0.95,
    scored_interval_ms=1_000,
    include_scored_interval_metric=True,
):
    slices = (
        EvaluationSliceDefinition(
            slice_id="overlap:true",
            dimension="overlap",
            value="true",
            status="ready",
            support_ms=400,
            minimum_support_ms=1,
            intervals=(EvaluationInterval(100, 500, "ch-1"),),
        ),
        EvaluationSliceDefinition(
            slice_id="speaker_change_boundary:within_collar",
            dimension="speaker_change_boundary",
            value="within_collar",
            status="ready",
            support_ms=120,
            minimum_support_ms=1,
            intervals=(EvaluationInterval(450, 570, "ch-1"),),
        ),
    )
    recording_metrics = (
        DiarizationRecordingMetricRow(
            recording_id=recording_id,
            output_id=output_id,
            policy_id="diagnostic-diarization-v1",
            status="scored",
            metrics=_recording_metrics(
                recording_der,
                scored_interval_ms=scored_interval_ms,
                include_scored_interval_metric=include_scored_interval_metric,
            ),
            speaker_mapping={},
        ),
    )
    slice_metrics = (
        DiarizationSliceMetricRow(
            recording_id=recording_id,
            output_id=output_id,
            policy_id="diagnostic-diarization-v1",
            slice_id="overlap:true",
            dimension="overlap",
            value="true",
            status="scored",
            support_ms=400,
            minimum_support_ms=1,
            metrics={
                "diarization_error_rate": overlap_der,
                "speaker_label_accuracy": 1.0 - overlap_der,
            },
        ),
        DiarizationSliceMetricRow(
            recording_id=recording_id,
            output_id=output_id,
            policy_id="diagnostic-diarization-v1",
            slice_id="speaker_change_boundary:within_collar",
            dimension="speaker_change_boundary",
            value="within_collar",
            status="scored",
            support_ms=120,
            minimum_support_ms=1,
            metrics={
                "boundary_accuracy": boundary_accuracy,
            },
        ),
    )
    return DiarizationEvaluationResult(
        recording_id=recording_id,
        output_id=output_id,
        scoring_policy={"policy_id": "diagnostic-diarization-v1", "version": "1"},
        speaker_mapping={},
        slices=slices,
        recording_metrics=recording_metrics,
        slice_metrics=slice_metrics,
        reference_artifact={"artifact_id": f"{recording_id}-reference", "artifact_kind": "reference"},
        candidate_artifact={"artifact_id": output_id, "artifact_kind": "candidate"},
    )


def _case(recording_id, *, current_der, baseline_der, current_overlap_der, baseline_overlap_der):
    return _case_with_support(
        recording_id,
        current_der=current_der,
        baseline_der=baseline_der,
        current_overlap_der=current_overlap_der,
        baseline_overlap_der=baseline_overlap_der,
    )


def _case_with_support(
    recording_id,
    *,
    current_der,
    baseline_der,
    current_overlap_der,
    baseline_overlap_der,
    scored_interval_ms=1_000,
):
    return BenchmarkEvaluationCase(
        corpus_id="ami-smoke",
        branch_id="separate-tracks",
        evaluation=_evaluation(
            f"{recording_id}-current",
            recording_id,
            recording_der=current_der,
            overlap_der=current_overlap_der,
            scored_interval_ms=scored_interval_ms,
            include_scored_interval_metric=False,
        ),
        baseline_evaluation=_evaluation(
            f"{recording_id}-baseline",
            recording_id,
            recording_der=baseline_der,
            overlap_der=baseline_overlap_der,
            scored_interval_ms=scored_interval_ms,
            include_scored_interval_metric=False,
        ),
        scored_duration_ms=scored_interval_ms,
        scored_words=20,
        scored_speaker_turns=4,
        slice_scored_words={
            "overlap:true": 7,
            "speaker_change_boundary:within_collar": 3,
        },
        slice_scored_speaker_turns={
            "overlap:true": 2,
            "speaker_change_boundary:within_collar": 1,
        },
    )


def _cases_for_signals():
    return (
        _case("rec-1", current_der=0.05, baseline_der=0.05, current_overlap_der=0.10, baseline_overlap_der=0.10),
        _case("rec-2", current_der=0.05, baseline_der=0.05, current_overlap_der=0.10, baseline_overlap_der=0.10),
    )


def build_benchmark_report(report_id, cases, **kwargs):
    cases = tuple(cases)
    if "route_assessments" not in kwargs and cases:
        kwargs["route_assessments"] = _route_assessments_for_cases(cases)
    return _build_benchmark_report(report_id, cases, **kwargs)


def _route_assessments_for_cases(cases):
    return tuple(
        PreflightRouteAssessment(
            corpus_id=case.corpus_id,
            branch_id=case.branch_id,
            recording_id=case.evaluation.recording_id,
            predicted_route="confident_pipeline",
            reference_route="confident_pipeline",
        )
        for case in cases
    )


def _recording_metrics(recording_der, *, scored_interval_ms, include_scored_interval_metric):
    metrics = {
        "diarization_error_rate": recording_der,
        "speaker_label_accuracy": 1.0 - recording_der,
    }
    if include_scored_interval_metric:
        metrics["scored_interval_ms"] = scored_interval_ms
    return metrics


def _signals():
    return (
        ReviewSignalSpan(
            signal_id="serious-tp",
            corpus_id="ami-smoke",
            branch_id="separate-tracks",
            recording_id="rec-1",
            start_ms=0,
            end_ms=100,
            severity="serious",
            reference_review_required=True,
            predicted_review_required=True,
            labels=("critical",),
        ),
        ReviewSignalSpan(
            signal_id="serious-fn",
            corpus_id="ami-smoke",
            branch_id="separate-tracks",
            recording_id="rec-1",
            start_ms=100,
            end_ms=200,
            severity="serious",
            reference_review_required=True,
            predicted_review_required=False,
            labels=("critical",),
        ),
        ReviewSignalSpan(
            signal_id="minor-fp",
            corpus_id="ami-smoke",
            branch_id="separate-tracks",
            recording_id="rec-2",
            start_ms=0,
            end_ms=100,
            severity="minor",
            reference_review_required=False,
            predicted_review_required=True,
        ),
        ReviewSignalSpan(
            signal_id="minor-tn",
            corpus_id="ami-smoke",
            branch_id="separate-tracks",
            recording_id="rec-2",
            start_ms=100,
            end_ms=200,
            severity="minor",
            reference_review_required=False,
            predicted_review_required=False,
        ),
        ReviewSignalSpan(
            signal_id="minor-unassessed",
            corpus_id="ami-smoke",
            branch_id="separate-tracks",
            recording_id="rec-2",
            start_ms=200,
            end_ms=300,
            severity="minor",
            reference_review_required=True,
            predicted_review_required=None,
        ),
    )


def test_benchmark_report_includes_required_scopes_and_metric_fields():
    report = build_benchmark_report(
        "story-21-report",
        (
            _case("rec-1", current_der=0.05, baseline_der=0.06, current_overlap_der=0.10, baseline_overlap_der=0.11),
            _case("rec-2", current_der=0.04, baseline_der=0.05, current_overlap_der=0.12, baseline_overlap_der=0.12),
        ),
        gate_config=BenchmarkGateConfig(
            budgets=(
                BenchmarkRegressionBudget(
                    budget_id="branch-der",
                    metric_name="diarization_error_rate",
                    direction="lower_is_better",
                    max_regression_delta=0.01,
                ),
            )
        ),
    )

    assert report.status == "passed"
    assert report.corpus_results
    assert report.branch_results
    assert report.recording_results
    assert report.slice_results
    branch_der = next(
        result
        for result in report.branch_results
        if result.metric_name == "diarization_error_rate"
    )
    assert branch_der.point_score == 0.045
    assert branch_der.paired_delta == -0.01
    assert branch_der.uncertainty.status == "available"
    assert branch_der.scored_duration_ms == 2_000
    assert branch_der.scored_words == 40
    assert branch_der.scored_speaker_turns == 8
    assert branch_der.gate.status == "passed"


def test_aggregate_metrics_are_weighted_by_scored_support():
    report = build_benchmark_report(
        "weighted-aggregate",
        (
            _case_with_support(
                "short-bad",
                current_der=1.0,
                baseline_der=0.0,
                current_overlap_der=0.0,
                baseline_overlap_der=0.0,
                scored_interval_ms=100,
            ),
            _case_with_support(
                "long-good",
                current_der=0.0,
                baseline_der=0.0,
                current_overlap_der=0.0,
                baseline_overlap_der=0.0,
                scored_interval_ms=900,
            ),
        ),
    )

    branch_der = next(
        result
        for result in report.branch_results
        if result.metric_name == "diarization_error_rate"
    )
    assert branch_der.point_score == 0.1
    assert branch_der.paired_delta == 0.1
    assert branch_der.scored_duration_ms == 1_000


def test_unscored_recording_baseline_does_not_provide_regression_delta():
    base_case = _case(
        "rec-1",
        current_der=0.105,
        baseline_der=0.10,
        current_overlap_der=0.10,
        baseline_overlap_der=0.10,
    )
    baseline = base_case.baseline_evaluation
    assert baseline is not None
    unscored_baseline = DiarizationEvaluationResult(
        recording_id=baseline.recording_id,
        output_id=baseline.output_id,
        scoring_policy=baseline.scoring_policy,
        speaker_mapping=baseline.speaker_mapping,
        slices=baseline.slices,
        recording_metrics=(
            DiarizationRecordingMetricRow(
                recording_id=baseline.recording_id,
                output_id=baseline.output_id,
                policy_id="diagnostic-diarization-v1",
                status="insufficient_support",
                metrics={
                    "diarization_error_rate": 0.10,
                    "speaker_label_accuracy": 0.90,
                },
                speaker_mapping={},
            ),
        ),
        slice_metrics=baseline.slice_metrics,
        reference_artifact=baseline.reference_artifact,
        candidate_artifact=baseline.candidate_artifact,
    )
    case = BenchmarkEvaluationCase(
        corpus_id=base_case.corpus_id,
        branch_id=base_case.branch_id,
        evaluation=base_case.evaluation,
        baseline_evaluation=unscored_baseline,
        scored_duration_ms=base_case.scored_duration_ms,
        scored_words=base_case.scored_words,
        scored_speaker_turns=base_case.scored_speaker_turns,
        slice_scored_words=base_case.slice_scored_words,
        slice_scored_speaker_turns=base_case.slice_scored_speaker_turns,
    )
    report = build_benchmark_report(
        "unscored-recording-baseline",
        (case,),
        gate_config=BenchmarkGateConfig(
            budgets=(
                BenchmarkRegressionBudget(
                    budget_id="branch-der",
                    metric_name="diarization_error_rate",
                    direction="lower_is_better",
                    max_regression_delta=0.01,
                    scope_type="branch",
                    scope_id="ami-smoke/separate-tracks",
                ),
            )
        ),
    )

    branch_der = next(result for result in report.branch_results if result.metric_name == "diarization_error_rate")
    assert report.status == "failed"
    assert branch_der.baseline_score is None
    assert branch_der.paired_delta is None
    assert branch_der.gate.status == "unavailable"
    assert branch_der.gate.budget_id == "branch-der"


def test_baseline_reference_artifact_must_match_evaluation():
    base_case = _case(
        "rec-1",
        current_der=0.05,
        baseline_der=0.05,
        current_overlap_der=0.10,
        baseline_overlap_der=0.10,
    )
    baseline = base_case.baseline_evaluation
    assert baseline is not None
    mismatched_baseline = DiarizationEvaluationResult(
        recording_id=baseline.recording_id,
        output_id=baseline.output_id,
        scoring_policy=baseline.scoring_policy,
        speaker_mapping=baseline.speaker_mapping,
        slices=baseline.slices,
        recording_metrics=baseline.recording_metrics,
        slice_metrics=baseline.slice_metrics,
        reference_artifact={"artifact_id": "different-reference", "artifact_kind": "reference"},
        candidate_artifact=baseline.candidate_artifact,
    )

    with pytest.raises(ValidationError, match="reference_artifact must match evaluation"):
        BenchmarkEvaluationCase(
            corpus_id=base_case.corpus_id,
            branch_id=base_case.branch_id,
            evaluation=base_case.evaluation,
            baseline_evaluation=mismatched_baseline,
            scored_duration_ms=base_case.scored_duration_ms,
            scored_words=base_case.scored_words,
            scored_speaker_turns=base_case.scored_speaker_turns,
            slice_scored_words=base_case.slice_scored_words,
            slice_scored_speaker_turns=base_case.slice_scored_speaker_turns,
        )


def test_slice_specific_regression_budget_can_pass_and_fail():
    pass_report = build_benchmark_report(
        "slice-pass",
        (
            _case("rec-1", current_der=0.05, baseline_der=0.05, current_overlap_der=0.105, baseline_overlap_der=0.10),
        ),
        gate_config=BenchmarkGateConfig(
            budgets=(
                BenchmarkRegressionBudget(
                    budget_id="overlap-der",
                    metric_name="diarization_error_rate",
                    budget_kind="overlap",
                    direction="lower_is_better",
                    max_regression_delta=0.01,
                    scope_type="slice",
                    slice_id="overlap:true",
                ),
            )
        ),
    )
    fail_report = build_benchmark_report(
        "slice-fail",
        (
            _case("rec-1", current_der=0.05, baseline_der=0.05, current_overlap_der=0.13, baseline_overlap_der=0.10),
        ),
        gate_config=pass_report.gate_config,
    )

    pass_overlap = next(
        result
        for result in pass_report.slice_results
        if result.slice_id == "overlap:true" and result.metric_name == "diarization_error_rate"
    )
    fail_overlap = next(
        result
        for result in fail_report.slice_results
        if result.slice_id == "overlap:true" and result.metric_name == "diarization_error_rate"
    )
    assert pass_overlap.gate.status == "passed"
    assert fail_report.status == "failed"
    assert fail_overlap.gate.status == "failed"
    assert "paired delta" in fail_overlap.gate.reasons[0]


def test_slice_specific_budget_overrides_broad_slice_budget_independent_of_order():
    report = build_benchmark_report(
        "slice-override",
        (
            _case("rec-1", current_der=0.05, baseline_der=0.05, current_overlap_der=0.105, baseline_overlap_der=0.10),
        ),
        gate_config=BenchmarkGateConfig(
            budgets=(
                BenchmarkRegressionBudget(
                    budget_id="all-slices",
                    metric_name="diarization_error_rate",
                    budget_kind="overlap",
                    direction="lower_is_better",
                    max_regression_delta=0.001,
                    scope_type="slice",
                ),
                BenchmarkRegressionBudget(
                    budget_id="overlap-specific",
                    metric_name="diarization_error_rate",
                    budget_kind="overlap",
                    direction="lower_is_better",
                    max_regression_delta=0.01,
                    scope_type="slice",
                    slice_id="overlap:true",
                ),
            )
        ),
    )

    overlap = next(
        result
        for result in report.slice_results
        if result.slice_id == "overlap:true" and result.metric_name == "diarization_error_rate"
    )
    assert report.status == "passed"
    assert overlap.gate.status == "passed"
    assert overlap.gate.budget_id == "overlap-specific"


def test_configured_regression_budget_without_baseline_fails_report():
    report = build_benchmark_report(
        "missing-baseline",
        (
            BenchmarkEvaluationCase(
                corpus_id="ami-smoke",
                branch_id="separate-tracks",
                evaluation=_evaluation(
                    "rec-1-current",
                    "rec-1",
                    recording_der=0.05,
                    overlap_der=0.10,
                ),
                baseline_evaluation=None,
                scored_duration_ms=1_000,
                scored_words=20,
                scored_speaker_turns=4,
            ),
        ),
        gate_config=BenchmarkGateConfig(
            budgets=(
                BenchmarkRegressionBudget(
                    budget_id="branch-der",
                    metric_name="diarization_error_rate",
                    direction="lower_is_better",
                    max_regression_delta=0.01,
                ),
            )
        ),
    )

    branch_der = next(
        result
        for result in report.branch_results
        if result.metric_name == "diarization_error_rate"
    )
    assert report.status == "failed"
    assert branch_der.gate.status == "unavailable"
    assert branch_der.gate.budget_id == "branch-der"
    assert branch_der.scored_duration_ms == 1_000


def test_combined_point_and_regression_budget_preserves_point_failure_without_baseline():
    report = build_benchmark_report(
        "combined-budget-without-baseline",
        (
            BenchmarkEvaluationCase(
                corpus_id="ami-smoke",
                branch_id="separate-tracks",
                evaluation=_evaluation(
                    "rec-1-current",
                    "rec-1",
                    recording_der=0.05,
                    overlap_der=0.10,
                ),
                baseline_evaluation=None,
                scored_duration_ms=1_000,
                scored_words=20,
                scored_speaker_turns=4,
            ),
        ),
        gate_config=BenchmarkGateConfig(
            budgets=(
                BenchmarkRegressionBudget(
                    budget_id="branch-der",
                    metric_name="diarization_error_rate",
                    direction="lower_is_better",
                    max_point_score=0.04,
                    max_regression_delta=0.01,
                ),
            )
        ),
    )

    branch_der = next(
        result
        for result in report.branch_results
        if result.metric_name == "diarization_error_rate"
    )
    assert report.status == "failed"
    assert branch_der.gate.status == "failed"
    assert branch_der.gate.reasons == (
        "point score 0.05 above maximum 0.04",
        "paired delta unavailable for regression budget",
    )


def test_report_requires_at_least_one_scored_metric_observation():
    with pytest.raises(ValidationError, match="route_assessments are required"):
        build_benchmark_report("empty-report", ())


def test_current_report_builder_requires_route_assessments():
    with pytest.raises(ValidationError, match="route_assessments are required"):
        _build_benchmark_report(
            "missing-route-assessments",
            (
                _case("rec-1", current_der=0.05, baseline_der=0.05, current_overlap_der=0.10, baseline_overlap_der=0.10),
            ),
            route_assessments=(),
        )


def test_configured_regression_budget_must_match_a_metric_result():
    with pytest.raises(ValidationError, match="regression budgets did not match"):
        build_benchmark_report(
            "unmatched-budget",
            (
                _case(
                    "rec-1",
                    current_der=0.05,
                    baseline_der=0.05,
                    current_overlap_der=0.10,
                    baseline_overlap_der=0.10,
                ),
            ),
            gate_config=BenchmarkGateConfig(
                budgets=(
                    BenchmarkRegressionBudget(
                        budget_id="missing-metric",
                        metric_name="false_confident_rate",
                        budget_kind="false_confidence",
                        min_point_score=0.9,
                    ),
                )
            ),
        )


def test_paired_baseline_must_match_recording_and_scoring_policy():
    current = _evaluation("rec-1-current", "rec-1", recording_der=0.05, overlap_der=0.10)
    different_recording = _evaluation("rec-2-baseline", "rec-2", recording_der=0.05, overlap_der=0.10)
    different_policy = DiarizationEvaluationResult(
        recording_id="rec-1",
        output_id="rec-1-baseline",
        scoring_policy={"policy_id": "product-transcript-v1", "version": "1"},
        speaker_mapping={},
        slices=current.slices,
        recording_metrics=current.recording_metrics,
        slice_metrics=current.slice_metrics,
        reference_artifact=current.reference_artifact,
        candidate_artifact=current.candidate_artifact,
    )

    with pytest.raises(ValidationError, match="recording_id must match"):
        BenchmarkEvaluationCase(
            corpus_id="ami-smoke",
            branch_id="separate-tracks",
            evaluation=current,
            baseline_evaluation=different_recording,
        )
    with pytest.raises(ValidationError, match="scoring_policy must match"):
        BenchmarkEvaluationCase(
            corpus_id="ami-smoke",
            branch_id="separate-tracks",
            evaluation=current,
            baseline_evaluation=different_policy,
        )


def test_report_json_round_trip_and_markdown_emitters(tmp_path):
    report = build_benchmark_report(
        "emitters",
        (
            _case("rec-1", current_der=0.05, baseline_der=0.05, current_overlap_der=0.10, baseline_overlap_der=0.10),
        ),
    )

    text = benchmark_report_json_dumps(report)
    loaded = benchmark_report_json_loads(text)
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    write_benchmark_report_json(json_path, report)
    write_benchmark_report_markdown(markdown_path, report)

    assert loaded.to_dict() == report.to_dict()
    assert read_benchmark_report_json(json_path).to_dict() == report.to_dict()
    assert benchmark_report_json_dumps(report) == text
    markdown = benchmark_report_to_markdown(report)
    assert "# Benchmark Report: emitters" in markdown
    assert "## Corpus Results" in markdown
    assert "## Slice Results" in markdown
    assert markdown_path.read_text(encoding="utf-8") == markdown


def test_report_json_rejects_unsupported_schema_versions():
    report = build_benchmark_report(
        "unsupported-schema",
        (
            _case("rec-1", current_der=0.05, baseline_der=0.05, current_overlap_der=0.10, baseline_overlap_der=0.10),
        ),
    )
    payload = report.to_dict()
    payload["schema_version"] = 999

    with pytest.raises(ValidationError, match="schema_version is not supported"):
        benchmark_report_json_loads(json.dumps(payload))


def test_report_json_rejects_status_that_conflicts_with_failed_gates():
    report = build_benchmark_report(
        "tampered-status",
        (
            _case("rec-1", current_der=0.05, baseline_der=0.05, current_overlap_der=0.13, baseline_overlap_der=0.10),
        ),
        gate_config=BenchmarkGateConfig(
            budgets=(
                BenchmarkRegressionBudget(
                    budget_id="overlap-der",
                    metric_name="diarization_error_rate",
                    budget_kind="overlap",
                    direction="lower_is_better",
                    max_regression_delta=0.01,
                    scope_type="slice",
                    slice_id="overlap:true",
                ),
            )
        ),
    )
    payload = report.to_dict()
    payload["status"] = "passed"

    with pytest.raises(ValidationError, match="passed benchmark reports cannot include failed gates"):
        benchmark_report_json_loads(json.dumps(payload))


def test_report_json_rejects_serialized_gate_that_conflicts_with_budget():
    report = build_benchmark_report(
        "tampered-gate",
        (
            _case("rec-1", current_der=0.05, baseline_der=0.05, current_overlap_der=0.13, baseline_overlap_der=0.10),
        ),
        gate_config=BenchmarkGateConfig(
            budgets=(
                BenchmarkRegressionBudget(
                    budget_id="overlap-der",
                    metric_name="diarization_error_rate",
                    budget_kind="overlap",
                    direction="lower_is_better",
                    max_regression_delta=0.01,
                    scope_type="slice",
                    slice_id="overlap:true",
                ),
            )
        ),
    )
    payload = report.to_dict()
    payload["status"] = "passed"
    for result in payload["slice_results"]:
        if result["slice_id"] == "overlap:true" and result["metric_name"] == "diarization_error_rate":
            result["gate"] = {
                "budget_id": "overlap-der",
                "reasons": [],
                "status": "passed",
                "thresholds": {"max_regression_delta": 0.01},
            }
            break

    with pytest.raises(ValidationError, match="metric_result gate does not match regression budget"):
        benchmark_report_json_loads(json.dumps(payload))


def test_report_constructor_rejects_metric_gate_that_conflicts_with_budget():
    gate_config = BenchmarkGateConfig(
        budgets=(
            BenchmarkRegressionBudget(
                budget_id="der-budget",
                metric_name="diarization_error_rate",
                direction="lower_is_better",
                max_regression_delta=0.01,
            ),
        )
    )
    forged_result = BenchmarkMetricResult(
        scope_type="corpus",
        scope_id="ami-smoke",
        corpus_id="ami-smoke",
        metric_name="diarization_error_rate",
        point_score=0.15,
        baseline_score=0.10,
        paired_delta=0.05,
        sample_count=1,
        scored_duration_ms=1_000,
        scored_words=10,
        scored_speaker_turns=2,
        gate=RegressionGateResult(
            status="passed",
            budget_id="der-budget",
            thresholds={"max_regression_delta": 0.01},
        ),
        uncertainty=UncertaintyInterval(
            status="unavailable",
            basis="paired_delta",
            reason="requires at least two paired samples",
        ),
    )

    with pytest.raises(ValidationError, match="metric_result gate does not match regression budget"):
        BenchmarkReport(
            report_id="forged-gate",
            status="passed",
            gate_config=gate_config,
            corpus_results=(forged_result,),
            schema_version=1,
        )


def test_report_json_rejects_tampered_review_signal_calibration_rate():
    report = build_benchmark_report(
        "tampered-review-calibration-rate",
        _cases_for_signals(),
        review_signals=_signals(),
    )
    payload = report.to_dict()
    payload["review_signal_calibration"]["recall"] = 1.0

    with pytest.raises(ValidationError, match="review_calibration.recall"):
        benchmark_report_json_loads(json.dumps(payload))


def test_report_json_rejects_tampered_review_signal_breakdown_totals():
    report = build_benchmark_report(
        "tampered-review-calibration-breakdown",
        _cases_for_signals(),
        review_signals=_signals(),
    )
    payload = report.to_dict()
    payload["review_signal_calibration"]["minor"].update(
        {
            "assessed": 0,
            "coverage": 0.0,
            "false_confident_rate": None,
            "false_negative": 0,
            "false_positive": 0,
            "over_flag_rate": None,
            "precision": None,
            "recall": None,
            "total": 0,
            "true_negative": 0,
            "true_positive": 0,
        }
    )

    with pytest.raises(ValidationError, match="review_calibration.total"):
        benchmark_report_json_loads(json.dumps(payload))


def test_report_json_rejects_tampered_critical_span_diagnostic():
    policy = CriticalSpanPolicyDefinition(
        policy_id="critical-span-diagnostic",
        version="v1",
        description="Synthetic diagnostic hook for serious review spans.",
        critical_severities=("serious",),
        minimum_recall=0.5,
    )
    report = build_benchmark_report(
        "tampered-critical-span",
        _cases_for_signals(),
        review_signals=_signals(),
        critical_span_policy=policy,
    )
    payload = report.to_dict()
    payload["critical_span_diagnostic"]["recall"] = 0.0

    with pytest.raises(ValidationError, match="critical_span_diagnostic recall"):
        benchmark_report_json_loads(json.dumps(payload))


def test_gate_config_rejects_duplicate_budget_ids():
    with pytest.raises(ValidationError, match="duplicate budget_id"):
        BenchmarkGateConfig(
            budgets=(
                BenchmarkRegressionBudget(
                    budget_id="duplicate",
                    metric_name="diarization_error_rate",
                    max_regression_delta=0.01,
                ),
                BenchmarkRegressionBudget(
                    budget_id="duplicate",
                    metric_name="speaker_label_accuracy",
                    max_regression_delta=0.01,
                ),
            )
        )


def test_review_signal_calibration_reports_serious_and_minor_breakdowns():
    calibration = calibrate_review_signals(_signals())

    assert calibration.total == 5
    assert calibration.assessed == 4
    assert calibration.coverage == 0.8
    assert calibration.precision == 0.5
    assert calibration.recall == 0.5
    assert calibration.false_confident_rate == 0.5
    assert calibration.over_flag_rate == 0.5
    assert calibration.serious.true_positive == 1
    assert calibration.serious.false_negative == 1
    assert calibration.serious.recall == 0.5
    assert calibration.minor.false_positive == 1
    assert calibration.minor.true_negative == 1


def test_review_signal_budgets_gate_serialized_review_metric_results():
    gate_config = BenchmarkGateConfig(
        budgets=(
            BenchmarkRegressionBudget(
                budget_id="false-confident-budget",
                metric_name="false_confident_rate",
                budget_kind="false_confidence",
                direction="lower_is_better",
                max_point_score=0.25,
                scope_type="branch",
                scope_id="ami-smoke/separate-tracks",
            ),
            BenchmarkRegressionBudget(
                budget_id="review-burden-budget",
                metric_name="over_flag_rate",
                budget_kind="human_review_burden",
                direction="lower_is_better",
                max_point_score=0.75,
                scope_type="branch",
                scope_id="ami-smoke/separate-tracks",
            ),
        )
    )
    report = build_benchmark_report(
        "review-signal-gates",
        _cases_for_signals(),
        gate_config=gate_config,
        review_signals=_signals(),
    )

    false_confident = next(result for result in report.branch_results if result.metric_name == "false_confident_rate")
    over_flag = next(result for result in report.branch_results if result.metric_name == "over_flag_rate")

    assert report.status == "failed"
    assert report.review_signal_scope_calibrations
    assert false_confident.point_score == 0.5
    assert false_confident.gate.status == "failed"
    assert false_confident.gate.budget_id == "false-confident-budget"
    assert over_flag.point_score == 0.5
    assert over_flag.gate.status == "passed"
    assert over_flag.gate.budget_id == "review-burden-budget"
    assert benchmark_report_json_loads(json.dumps(report.to_dict())).to_dict() == report.to_dict()


def test_review_signals_must_match_evaluated_cases():
    with pytest.raises(ValidationError, match="review_signals must match evaluated benchmark cases"):
        build_benchmark_report(
            "mismatched-review-signal-scope",
            (
                _case(
                    "rec-1",
                    current_der=0.05,
                    baseline_der=0.05,
                    current_overlap_der=0.10,
                    baseline_overlap_der=0.10,
                ),
            ),
            review_signals=(
                ReviewSignalSpan(
                    signal_id="wrong-scope",
                    corpus_id="ami-smoke",
                    branch_id="other-branch",
                    recording_id="rec-99",
                    start_ms=0,
                    end_ms=100,
                    severity="minor",
                    reference_review_required=False,
                    predicted_review_required=False,
                ),
            ),
        )


def test_report_json_rejects_tampered_review_signal_metric_gate():
    report = build_benchmark_report(
        "tampered-review-signal-gate",
        _cases_for_signals(),
        gate_config=BenchmarkGateConfig(
            budgets=(
                BenchmarkRegressionBudget(
                    budget_id="false-confident-budget",
                    metric_name="false_confident_rate",
                    budget_kind="false_confidence",
                    direction="lower_is_better",
                    max_point_score=0.25,
                    scope_type="branch",
                    scope_id="ami-smoke/separate-tracks",
                ),
            )
        ),
        review_signals=_signals(),
    )
    payload = report.to_dict()
    payload["status"] = "passed"
    for result in payload["branch_results"]:
        if result["metric_name"] == "false_confident_rate":
            result["point_score"] = 0.1
            result["gate"] = {
                "budget_id": "false-confident-budget",
                "reasons": [],
                "status": "passed",
                "thresholds": {"max_point_score": 0.25},
            }
            break

    with pytest.raises(ValidationError, match="review-signal metric result"):
        benchmark_report_json_loads(json.dumps(payload))


def test_report_json_rejects_extra_review_signal_metric_result():
    report = build_benchmark_report(
        "extra-review-signal-metric",
        (
            _case("rec-1", current_der=0.05, baseline_der=0.05, current_overlap_der=0.10, baseline_overlap_der=0.10),
        ),
        review_signals=(
            ReviewSignalSpan(
                signal_id="minor-tn",
                corpus_id="ami-smoke",
                branch_id="separate-tracks",
                recording_id="rec-1",
                start_ms=0,
                end_ms=100,
                severity="minor",
                reference_review_required=False,
                predicted_review_required=False,
            ),
        ),
    )
    payload = report.to_dict()
    over_flag = next(result for result in payload["branch_results"] if result["metric_name"] == "over_flag_rate")
    extra_precision = dict(over_flag)
    extra_precision.update(
        {
            "gate": {
                "budget_id": None,
                "reasons": ["no regression budget configured"],
                "status": "unavailable",
                "thresholds": {},
            },
            "metric_name": "precision",
            "point_score": 1.0,
        }
    )
    payload["branch_results"].append(extra_precision)

    with pytest.raises(ValidationError, match="review-signal metric results"):
        benchmark_report_json_loads(json.dumps(payload))


def test_report_json_rejects_review_signal_scope_not_backed_by_evaluated_metrics():
    report = build_benchmark_report(
        "moved-review-signal-scope",
        _cases_for_signals(),
        review_signals=_signals(),
    )
    payload = report.to_dict()
    for scope_calibration in payload["review_signal_scope_calibrations"]:
        if scope_calibration["scope_type"] == "recording" and scope_calibration["recording_id"] == "rec-2":
            scope_calibration["recording_id"] = "rec-99"
            scope_calibration["scope_id"] = "ami-smoke/separate-tracks/rec-99"
            break
    for result in payload["recording_results"]:
        if result["recording_id"] == "rec-2" and result["metric_name"] in {
            "coverage",
            "false_confident_rate",
            "over_flag_rate",
        }:
            result["recording_id"] = "rec-99"
            result["scope_id"] = "ami-smoke/separate-tracks/rec-99"

    with pytest.raises(ValidationError, match="review_signal_scope_calibrations must match evaluated metric scopes"):
        benchmark_report_json_loads(json.dumps(payload))


def test_report_json_rejects_missing_review_signal_scope_calibrations():
    report = build_benchmark_report(
        "missing-review-signal-scopes",
        _cases_for_signals(),
        review_signals=_signals(),
    )
    payload = report.to_dict()
    payload["review_signal_scope_calibrations"] = [
        item for item in payload["review_signal_scope_calibrations"] if item["scope_type"] == "corpus"
    ]

    with pytest.raises(ValidationError, match="at least one branch scope"):
        benchmark_report_json_loads(json.dumps(payload))


def test_report_json_rejects_review_signal_metric_without_calibration():
    report = build_benchmark_report(
        "forged-review-metric-without-calibration",
        (
            _case("rec-1", current_der=0.05, baseline_der=0.05, current_overlap_der=0.10, baseline_overlap_der=0.10),
        ),
    )
    payload = report.to_dict()
    forged = dict(payload["branch_results"][0])
    forged.update(
        {
            "metric_name": "false_confident_rate",
            "point_score": 0.0,
        }
    )
    payload["branch_results"].append(forged)

    with pytest.raises(ValidationError, match="require review_signal_calibration"):
        benchmark_report_json_loads(json.dumps(payload))


def test_report_json_rejects_precision_metric_without_calibration():
    report = build_benchmark_report(
        "forged-precision-without-calibration",
        (
            _case("rec-1", current_der=0.05, baseline_der=0.05, current_overlap_der=0.10, baseline_overlap_der=0.10),
        ),
    )
    payload = report.to_dict()
    forged = dict(payload["branch_results"][0])
    forged.update(
        {
            "metric_name": "precision",
            "point_score": 1.0,
        }
    )
    payload["branch_results"].append(forged)

    with pytest.raises(ValidationError, match="require review_signal_calibration"):
        benchmark_report_json_loads(json.dumps(payload))


def test_report_json_rejects_slice_review_signal_metric_without_calibration():
    report = build_benchmark_report(
        "forged-slice-review-metric-without-calibration",
        (
            _case("rec-1", current_der=0.05, baseline_der=0.05, current_overlap_der=0.10, baseline_overlap_der=0.10),
        ),
    )
    payload = report.to_dict()
    forged = dict(payload["slice_results"][0])
    forged.update(
        {
            "gate": {
                "budget_id": None,
                "reasons": ["no regression budget configured"],
                "status": "unavailable",
                "thresholds": {},
            },
            "metric_name": "false_confident_rate",
            "point_score": 0.0,
            "uncertainty": {
                "basis": "review_signal_metric",
                "confidence_level": 0.95,
                "lower": None,
                "reason": "review-signal metrics do not have paired samples",
                "status": "unavailable",
                "upper": None,
            },
        }
    )
    payload["slice_results"].append(forged)

    with pytest.raises(ValidationError, match="require review_signal_calibration"):
        benchmark_report_json_loads(json.dumps(payload))


def test_report_json_rejects_metric_result_in_wrong_section():
    report = build_benchmark_report(
        "wrong-result-section",
        (
            _case("rec-1", current_der=0.05, baseline_der=0.05, current_overlap_der=0.10, baseline_overlap_der=0.10),
        ),
    )
    payload = report.to_dict()
    payload["corpus_results"].append(payload["branch_results"][0])

    with pytest.raises(ValidationError, match="corpus_results must contain only corpus results"):
        benchmark_report_json_loads(json.dumps(payload))


def test_report_json_rejects_metric_result_scope_identifier_mismatch():
    report = build_benchmark_report(
        "wrong-scope-identity",
        (
            _case("rec-1", current_der=0.05, baseline_der=0.05, current_overlap_der=0.10, baseline_overlap_der=0.10),
        ),
    )
    payload = report.to_dict()
    payload["branch_results"][0]["branch_id"] = "different-branch"

    with pytest.raises(ValidationError, match="metric_result.scope_id"):
        benchmark_report_json_loads(json.dumps(payload))


def test_report_json_rejects_duplicate_metric_result_keys():
    report = build_benchmark_report(
        "duplicate-metric-result",
        (
            _case("rec-1", current_der=0.05, baseline_der=0.05, current_overlap_der=0.10, baseline_overlap_der=0.10),
        ),
    )
    payload = report.to_dict()
    duplicate = dict(payload["branch_results"][0])
    duplicate["point_score"] = duplicate["point_score"] + 0.1
    payload["branch_results"].append(duplicate)

    with pytest.raises(ValidationError, match="duplicate metric result"):
        benchmark_report_json_loads(json.dumps(payload))


def test_diagnostic_critical_span_policy_hook_scores_synthetic_spans():
    policy = CriticalSpanPolicyDefinition(
        policy_id="critical-span-diagnostic",
        version="v1",
        description="Synthetic diagnostic hook for serious review spans.",
        critical_severities=("serious",),
        critical_labels=("critical",),
        minimum_recall=1.0,
    )

    score = score_diagnostic_critical_spans(_signals(), policy)
    passing_score = score_diagnostic_critical_spans(
        (
            ReviewSignalSpan(
                signal_id="serious-tp",
                corpus_id="ami-smoke",
                branch_id="separate-tracks",
                recording_id="rec-1",
                start_ms=0,
                end_ms=100,
                severity="serious",
                reference_review_required=True,
                predicted_review_required=True,
                labels=("critical",),
            ),
        ),
        policy,
    )

    assert score.status == "failed"
    assert score.critical_span_count == 2
    assert score.detected_critical_span_count == 1
    assert score.missed_critical_span_count == 1
    assert score.recall == 0.5
    assert passing_score.status == "passed"
    assert passing_score.recall == 1.0


def test_review_signal_labels_reject_bare_strings():
    with pytest.raises(ValidationError, match="review_signal.labels"):
        ReviewSignalSpan(
            signal_id="string-labels",
            corpus_id="ami-smoke",
            branch_id="separate-tracks",
            recording_id="rec-1",
            start_ms=0,
            end_ms=100,
            severity="serious",
            reference_review_required=True,
            predicted_review_required=True,
            labels="critical",
        )


def test_report_embeds_review_signal_and_critical_span_diagnostics():
    policy = CriticalSpanPolicyDefinition(
        policy_id="critical-span-diagnostic",
        version="v1",
        description="Synthetic diagnostic hook for serious review spans.",
        critical_severities=("serious",),
        minimum_recall=0.5,
    )

    report = build_benchmark_report(
        "review-diagnostics",
        _cases_for_signals(),
        review_signals=_signals(),
        critical_span_policy=policy,
    )

    assert report.status == "passed"
    assert report.review_signal_calibration is not None
    assert report.review_signal_calibration.serious.false_negative == 1
    assert report.critical_span_policy == policy
    assert report.critical_span_diagnostic is not None
    assert report.critical_span_diagnostic.status == "passed"
    assert "Critical Span Diagnostic" in benchmark_report_to_markdown(report)


def test_report_fails_on_out_of_scope_false_confident_routing():
    report = build_benchmark_report(
        "route-confusion",
        (
            _case("rec-1", current_der=0.05, baseline_der=0.05, current_overlap_der=0.10, baseline_overlap_der=0.10),
        ),
        route_assessments=(
            PreflightRouteAssessment(
                corpus_id="ami-smoke",
                branch_id="separate-tracks",
                recording_id="rec-1",
                predicted_route="confident_pipeline",
                reference_route="diagnostic_only",
            ),
        ),
    )

    assert report.status == "failed"
    assert report.route_confusion is not None
    assert report.route_confusion.out_of_scope_false_confident_count == 1
    assert report.route_confusion.matrix["diagnostic_only"]["confident_pipeline"] == 1
    assert "Route Confusion" in benchmark_report_to_markdown(report)
    assert benchmark_report_json_loads(json.dumps(report.to_dict())).to_dict() == report.to_dict()


def test_report_schema_version_one_without_route_confusion_stays_readable():
    report = build_benchmark_report(
        "legacy-report",
        (
            _case("rec-1", current_der=0.05, baseline_der=0.05, current_overlap_der=0.10, baseline_overlap_der=0.10),
        ),
    )
    payload = report.to_dict()
    assert payload["schema_version"] == BENCHMARK_REPORT_SCHEMA_VERSION
    payload["schema_version"] = 1
    payload.pop("route_confusion")

    loaded = benchmark_report_json_loads(json.dumps(payload))

    assert loaded.schema_version == 1
    assert loaded.route_confusion is None
    assert "route_confusion" not in loaded.to_dict()


def test_report_schema_version_one_rejects_route_confusion():
    report = build_benchmark_report(
        "route-confusion-v1",
        (
            _case("rec-1", current_der=0.05, baseline_der=0.05, current_overlap_der=0.10, baseline_overlap_der=0.10),
        ),
        route_assessments=(
            PreflightRouteAssessment(
                corpus_id="ami-smoke",
                branch_id="separate-tracks",
                recording_id="rec-1",
                predicted_route="confident_pipeline",
                reference_route="confident_pipeline",
            ),
        ),
    )
    payload = report.to_dict()
    payload["schema_version"] = 1

    with pytest.raises(ValidationError, match="route_confusion requires schema_version 2"):
        benchmark_report_json_loads(json.dumps(payload))


def test_route_assessments_must_cover_every_evaluated_case():
    with pytest.raises(ValidationError, match="route_assessments must cover every evaluated benchmark case"):
        build_benchmark_report(
            "incomplete-route-confusion",
            _cases_for_signals(),
            route_assessments=(
                PreflightRouteAssessment(
                    corpus_id="ami-smoke",
                    branch_id="separate-tracks",
                    recording_id="rec-1",
                    predicted_route="confident_pipeline",
                    reference_route="confident_pipeline",
                ),
            ),
        )


def test_report_excludes_manual_route_overrides_from_confusion_metrics():
    report = build_benchmark_report(
        "route-confusion-manual",
        (
            _case("rec-1", current_der=0.05, baseline_der=0.05, current_overlap_der=0.10, baseline_overlap_der=0.10),
        ),
        route_assessments=(
            PreflightRouteAssessment(
                corpus_id="ami-smoke",
                branch_id="separate-tracks",
                recording_id="rec-1",
                predicted_route="confident_pipeline",
                reference_route="diagnostic_only",
                manual_override_applied=True,
            ),
        ),
    )

    assert report.status == "passed"
    assert report.route_confusion is not None
    assert report.route_confusion.manual_override_count == 1
    assert report.route_confusion.out_of_scope_false_confident_count == 0
    assert report.route_confusion.matrix["diagnostic_only"]["confident_pipeline"] == 0


def test_report_fails_when_configured_critical_span_policy_has_no_diagnostic_support():
    policy = CriticalSpanPolicyDefinition(
        policy_id="critical-span-diagnostic",
        version="v1",
        description="Synthetic diagnostic hook for serious review spans.",
        critical_severities=("serious",),
        minimum_recall=1.0,
    )

    report = build_benchmark_report(
        "unsupported-critical-spans",
        (
            _case("rec-1", current_der=0.05, baseline_der=0.05, current_overlap_der=0.10, baseline_overlap_der=0.10),
        ),
        review_signals=(
            ReviewSignalSpan(
                signal_id="minor-only",
                corpus_id="ami-smoke",
                branch_id="separate-tracks",
                recording_id="rec-1",
                start_ms=0,
                end_ms=100,
                severity="minor",
                reference_review_required=True,
                predicted_review_required=True,
            ),
        ),
        critical_span_policy=policy,
    )

    assert report.status == "failed"
    assert report.critical_span_diagnostic is not None
    assert report.critical_span_diagnostic.status == "unavailable"
