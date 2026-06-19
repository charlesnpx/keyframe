from pathlib import Path

from keyframe.diarization import (
    BenchmarkEvaluationCase,
    BenchmarkGateConfig,
    BenchmarkRegressionBudget,
    CriticalSpanPolicyDefinition,
    DiarizationEvaluationResult,
    DiarizationRecordingMetricRow,
    DiarizationSliceMetricRow,
    EvaluationInterval,
    EvaluationSliceDefinition,
    ReviewSignalSpan,
    benchmark_report_json_dumps,
    benchmark_report_json_loads,
    benchmark_report_to_markdown,
    build_benchmark_report,
    calibrate_review_signals,
    read_benchmark_report_json,
    score_diagnostic_critical_spans,
    write_benchmark_report_json,
    write_benchmark_report_markdown,
)


def _evaluation(
    output_id,
    recording_id,
    *,
    recording_der,
    overlap_der,
    boundary_accuracy=0.95,
    scored_interval_ms=1_000,
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
            metrics={
                "diarization_error_rate": recording_der,
                "speaker_label_accuracy": 1.0 - recording_der,
                "scored_interval_ms": scored_interval_ms,
            },
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
    return BenchmarkEvaluationCase(
        corpus_id="ami-smoke",
        branch_id="separate-tracks",
        evaluation=_evaluation(
            f"{recording_id}-current",
            recording_id,
            recording_der=current_der,
            overlap_der=current_overlap_der,
        ),
        baseline_evaluation=_evaluation(
            f"{recording_id}-baseline",
            recording_id,
            recording_der=baseline_der,
            overlap_der=baseline_overlap_der,
        ),
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
        (
            _case("rec-1", current_der=0.05, baseline_der=0.05, current_overlap_der=0.10, baseline_overlap_der=0.10),
        ),
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
