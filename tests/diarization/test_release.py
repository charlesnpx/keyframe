import pytest

from keyframe.diarization import (
    BenchmarkGateConfig,
    BenchmarkMetricResult,
    BenchmarkRegressionBudget,
    BenchmarkReport,
    BranchAcceptanceRecord,
    EngineConfigMetadata,
    PreflightPolicy,
    PreflightRouteAssessment,
    PrivateAnnotationProtocol,
    RegressionGateResult,
    ReleaseCandidateRecord,
    ReleaseGoldenTestResult,
    ReleaseMustNotHaveChecks,
    ReleaseRevalidationReport,
    ReleaseRouteState,
    UncertaintyInterval,
    ValidationError,
    build_preflight_route_confusion_report,
    check_release_runtime_config,
    benchmark_report_from_dict,
    read_dataset_manifest_json,
    release_expected_runtime_config,
    release_revalidation_report,
    release_revalidation_summary,
    release_record_content_hash,
    release_record_from_dict,
    release_record_json_dumps,
    release_record_json_loads,
    validate_release_runtime_output,
)


def _release(**overrides):
    values = {
        "release_candidate_id": "release-2026-06-23",
        "git_sha": "b" * 40,
        "dataset_snapshots": (_manifest().to_dict(),),
        "private_acceptance_split_id": "private_acceptance",
        "annotation_protocol": _annotation_protocol(),
        "scoring_policies": _manifest().scoring_policies,
        "preflight_policy": _preflight_policy(),
        "route_state": _route_state(),
        "branch_decisions": (_branch_decision(),),
        "engine_configs": (_engine_config(),),
        "validated_scope": ("separate_tracks_2_speaker",),
        "unsupported_scope": ("mono_mix_high_overlap",),
        "governance_decision": "approve_confident_labels",
        "benchmark_reports": (_benchmark_report(),),
        "golden_tests": (_golden_test(),),
        "approval_status": "approved",
        "must_not_have_checks": _must_not_have_checks(),
    }
    values.update(overrides)
    return ReleaseCandidateRecord(**values)


def _manifest():
    return read_dataset_manifest_json("keyframe/diarization/dataset_manifests/ami.json")


def _annotation_protocol():
    return PrivateAnnotationProtocol(
        protocol_id="private-annotation",
        version="2026-06-23",
        transcript_normalization="lowercase punctuation-stripped words",
        speaker_span_rules=("mark contiguous same-speaker spans",),
        overlap_rules=("label overlaps explicitly",),
        unintelligible_no_score_rules=("exclude unintelligible no-score spans",),
        critical_span_label_rules=("mark speaker-change critical spans",),
    )


def _preflight_policy():
    return PreflightPolicy(
        policy_id="launch-preflight",
        version="2026-06-23",
        frozen_git_sha="c" * 40,
        tuned_on_splits=("public_dev",),
        validated_on_splits=("public_holdout", "private_acceptance"),
        supported_locales=("en-US",),
        supported_sources=("zoom", "teams"),
        supported_capture_modes=("separate_tracks", "mono_mix"),
        confident_capture_modes=("separate_tracks",),
        supported_channel_counts=(1, 2),
        supported_codecs=("pcm_s16le", "opus"),
        min_duration_ms=30_000,
        max_duration_ms=14_400_000,
        min_sample_rate_hz=16_000,
        max_confident_clipping_estimate=0.05,
        min_confident_speech_ratio=0.30,
        max_confident_rough_overlap_estimate=0.35,
        min_confident_speaker_count_hint=2,
        max_confident_speaker_count_hint=6,
        require_speaker_count_hint_for_confident=True,
    )


def _route_state():
    return ReleaseRouteState(
        policy_id="launch-preflight",
        policy_version="2026-06-23",
        route="confident_pipeline",
        effective_route="confident_pipeline",
        validated_launch_scope_version="private-coverage-v1@2026-06-23",
    )


def _branch_decision():
    return BranchAcceptanceRecord(
        branch_id="separate_tracks",
        decision="accept_complex_branch",
        quality_delta=0.02,
        false_confidence_delta=-0.01,
        review_burden_delta=-0.03,
        quality_gate_passed=True,
        false_confidence_gate_passed=True,
        review_burden_gate_passed=True,
        governance_delta={"provider_contract": "pinned"},
        private_coverage_ready=True,
    )


def _branch_decision_with(**overrides):
    values = _branch_decision().to_dict()
    values.update(overrides)
    enforced_gates = values.pop("enforced_gates")
    non_enforced_fields = values.pop("non_enforced_fields")
    values["quality_gate_passed"] = enforced_gates["quality"]
    values["false_confidence_gate_passed"] = enforced_gates["false_confidence"]
    values["review_burden_gate_passed"] = enforced_gates["review_burden"]
    values["latency_delta_ms"] = non_enforced_fields["latency_delta_ms"]
    values["cost_delta"] = non_enforced_fields["cost_delta"]
    values["job_failure_delta"] = non_enforced_fields["job_failure_delta"]
    values["retry_delta"] = non_enforced_fields["retry_delta"]
    values["governance_delta"] = non_enforced_fields["governance_delta"]
    return BranchAcceptanceRecord(**values)


def _engine_config(**overrides):
    values = {
        "adapter_id": "release-engine",
        "provider": "self-hosted",
        "model_name": "whisperx+pyannote",
        "model_version": "2026-06",
        "config_id": "release-config-001",
        "parameters": {
            "model_governance": {
                "checkpoint": "pyannote/speaker-diarization-3.1",
                "package_versions": {"pyannote.audio": "3.1.0", "whisperx": "3.2.0"},
                "runtime_config": {"cache_root": "/models/local"},
            }
        },
    }
    values.update(overrides)
    return EngineConfigMetadata(**values)


def _benchmark_report():
    budget = BenchmarkRegressionBudget(
        budget_id="max-der",
        metric_name="diarization_error_rate",
        direction="lower_is_better",
        max_point_score=0.10,
    )
    metric = BenchmarkMetricResult(
        scope_type="corpus",
        scope_id="ami",
        metric_name="diarization_error_rate",
        point_score=0.04,
        sample_count=1,
        scored_duration_ms=1_000,
        scored_words=10,
        scored_speaker_turns=2,
        gate=RegressionGateResult(
            "passed",
            budget_id="max-der",
            thresholds={"max_point_score": 0.10},
        ),
        uncertainty=UncertaintyInterval("unavailable", reason="single deterministic fixture"),
        corpus_id="ami",
    )
    route_confusion = build_preflight_route_confusion_report(
        (
            PreflightRouteAssessment(
                corpus_id="ami",
                branch_id="separate_tracks",
                recording_id="rec-1",
                predicted_route="confident_pipeline",
                reference_route="confident_pipeline",
            ),
        )
    )
    return BenchmarkReport(
        report_id="benchmark-report-001",
        status="passed",
        gate_config=BenchmarkGateConfig((budget,)),
        corpus_results=(metric,),
        route_confusion=route_confusion,
    )


def _golden_test(**overrides):
    values = {
        "test_id": "rendered-transcript-golden",
        "status": "passed",
        "artifact_hash": "a" * 64,
        "metadata": {"fixture": "clean_two_speaker"},
    }
    values.update(overrides)
    return ReleaseGoldenTestResult(**values)


def _must_not_have_checks(**overrides):
    values = {
        "no_cross_call_speaker_ids": True,
        "no_retained_voice_profiles_or_embeddings": True,
        "no_reference_speaker_ids_in_runtime_output": True,
        "no_unpinned_model_or_provider_config": True,
        "no_tuned_on_holdout_result": True,
        "evidence": {"identity_scope": "session_local"},
    }
    values.update(overrides)
    return ReleaseMustNotHaveChecks(**values)


def test_release_record_hash_round_trip_and_runtime_match_enable_confident_labels():
    record = _release()

    assert record.content_hash == release_record_content_hash(record)
    loaded = release_record_json_loads(release_record_json_dumps(record))
    assert loaded.to_dict() == record.to_dict()

    check = check_release_runtime_config(record, release_expected_runtime_config(record))

    assert check.status == "approved"
    assert check.confident_speaker_attribution_enabled is True
    assert check.audit_events == ()


def test_release_record_includes_rollback_and_revalidation_metadata():
    record = _release(
        rollback_release_candidate_id="release-2026-06-22",
        revalidate_on=("model_version", "scoring_policy", "governance_retention"),
        emergency_degraded_route="needs_review",
    )
    payload = record.to_dict()

    assert payload["rollback_release_candidate_id"] == "release-2026-06-22"
    assert payload["revalidate_on"] == ["model_version", "scoring_policy", "governance_retention"]
    assert payload["emergency_degraded_route"] == "needs_review"
    assert payload["degraded_transcript_output_allowed"] is True
    assert release_record_from_dict(payload).to_dict() == payload


def test_release_record_validates_rollback_and_revalidation_metadata():
    with pytest.raises(ValidationError, match="must differ from release_candidate_id"):
        _release(rollback_release_candidate_id="release-2026-06-23")

    with pytest.raises(ValidationError, match="release.revalidate_on is required"):
        _release(revalidate_on=())

    with pytest.raises(ValidationError, match="contains duplicate"):
        _release(revalidate_on=("model_version", "model_version"))

    with pytest.raises(ValidationError, match="is not supported"):
        _release(revalidate_on=("unsupported_trigger",))

    with pytest.raises(ValidationError, match="diagnostic_only or needs_review"):
        _release(emergency_degraded_route="confident_pipeline")


def test_release_loader_requires_persisted_content_hash():
    payload = _release().to_dict()
    payload.pop("content_hash")

    with pytest.raises(ValidationError, match="release.content_hash is required"):
        release_record_from_dict(payload)


def test_release_record_payload_is_immutable_after_hashing():
    engine_config = _engine_config()
    record = _release(engine_configs=(engine_config,))
    original_hash = record.content_hash

    with pytest.raises(TypeError):
        record.dataset_snapshots[0]["dataset_id"] = "mutated"
    with pytest.raises(TypeError):
        record.engine_configs[0].parameters["model_governance"] = {}
    with pytest.raises(TypeError):
        record.engine_configs[0].parameters["model_governance"]["package_versions"]["pyannote.audio"] = "mutated"

    engine_config.parameters["model_governance"]["package_versions"]["pyannote.audio"] = "mutated-source"
    payload = record.to_dict()
    payload["dataset_snapshots"][0]["dataset_id"] = "mutated-copy"
    payload["engine_configs"][0]["parameters"]["model_governance"]["package_versions"][
        "pyannote.audio"
    ] = "mutated-copy"

    assert record.content_hash == original_hash
    assert release_record_content_hash(record) == original_hash
    assert record.to_dict()["dataset_snapshots"][0]["dataset_id"] == "ami"
    assert (
        record.engine_configs[0].parameters["model_governance"]["package_versions"]["pyannote.audio"]
        == "3.1.0"
    )
    assert (
        record.to_dict()["engine_configs"][0]["parameters"]["model_governance"]["package_versions"][
            "pyannote.audio"
        ]
        == "3.1.0"
    )


def test_release_loader_rejects_tampered_branch_acceptance_payload():
    payload = _release().to_dict()
    payload["branch_decisions"][0]["non_enforced_fields"]["cost_delta"] = 12.34

    with pytest.raises(ValidationError, match="non_enforced_fields must match"):
        release_record_from_dict(payload)

    payload = _release().to_dict()
    payload["branch_decisions"][0]["enforced_gates"]["hidden_gate"] = True

    with pytest.raises(ValidationError, match="branch_acceptance.enforced_gates has unsupported fields"):
        release_record_from_dict(payload)


def test_runtime_config_mismatch_disables_confident_labels_and_emits_audit_event():
    record = _release()
    active_payload = release_expected_runtime_config(record).to_dict()
    active_payload["preflight_policy_version"] = "2026-06-01"

    check = check_release_runtime_config(record, active_payload)

    assert check.status == "degraded"
    assert check.confident_speaker_attribution_enabled is False
    assert check.degraded_route == "diagnostic_only"
    assert check.degraded_transcript_output_allowed is True
    assert [event.code for event in check.audit_events] == ["preflight_policy_revalidation_required"]
    assert check.audit_events[0].actual["trigger"] == "preflight_policy"


def test_engine_fingerprint_includes_pinned_package_versions():
    record = _release()
    active_payload = release_expected_runtime_config(record).to_dict()
    drifted = _engine_config(
        parameters={
            "model_governance": {
                "checkpoint": "pyannote/speaker-diarization-3.1",
                "package_versions": {"pyannote.audio": "3.1.1", "whisperx": "3.2.0"},
                "runtime_config": {"cache_root": "/models/local"},
            }
        }
    )
    drifted_record = _release(engine_configs=(drifted,))
    active_payload["engine_config_ids"] = release_expected_runtime_config(drifted_record).engine_config_ids

    check = check_release_runtime_config(record, active_payload)

    assert check.status == "degraded"
    assert [event.code for event in check.audit_events] == [
        "model_version_revalidation_required",
        "provider_contract_revalidation_required",
        "governance_retention_revalidation_required",
    ]


@pytest.mark.parametrize(
    ("trigger", "runtime_field", "replacement"),
    (
        ("model_version", "engine_config_ids", {"release-engine": "model-drift"}),
        ("provider_contract", "engine_config_ids", {"release-engine": "provider-contract-drift"}),
        ("scoring_policy", "scoring_policy_versions", {"ami-diarization-v1": "2026-07-01"}),
        ("preflight_policy", "preflight_policy_version", "2026-07-01"),
        ("canonical_transform", "dataset_snapshot_ids", {"ami": "canonical-transform-drift"}),
        ("launch_scope", "validated_scope", ["separate_tracks_2_speaker", "mono_mix_low_overlap"]),
        ("annotation_protocol", "annotation_protocol_version", "private-annotation@2026-07-01"),
        ("governance_retention", "governance_decision", "degraded_only"),
    ),
)
def test_release_revalidation_report_identifies_each_trigger(trigger, runtime_field, replacement):
    record = _release(
        rollback_release_candidate_id="release-2026-06-22",
        revalidate_on=(trigger,),
    )
    active_payload = release_expected_runtime_config(record).to_dict()
    active_payload[runtime_field] = replacement

    report = release_revalidation_report(record, active_payload)
    summary = release_revalidation_summary(record, active_payload)

    assert isinstance(report, ReleaseRevalidationReport)
    assert report.requires_revalidation is True
    assert report.rollback_release_candidate_id == "release-2026-06-22"
    assert report.triggers == (trigger,)
    assert report.emergency_degraded_route == "diagnostic_only"
    assert report.degraded_transcript_output_allowed is True
    assert [event.code for event in report.audit_events] == [f"{trigger}_revalidation_required"]
    assert report.audit_events[0].actual["runtime_field"] == runtime_field
    assert trigger in summary
    assert "rollback=release-2026-06-22" in summary


def test_release_revalidation_report_allows_clean_runtime_config():
    record = _release(rollback_release_candidate_id="release-2026-06-22")
    active_config = release_expected_runtime_config(record)

    report = release_revalidation_report(record, active_config)

    assert report.requires_revalidation is False
    assert report.triggers == ()
    assert report.audit_events == ()
    assert release_revalidation_summary(record, active_config) == "release-2026-06-23: no revalidation required"


def test_emergency_degradation_can_route_to_needs_review_and_preserve_degraded_output():
    record = _release(
        rollback_release_candidate_id="release-2026-06-22",
        revalidate_on=("scoring_policy",),
        emergency_degraded_route="needs_review",
    )
    active_payload = release_expected_runtime_config(record).to_dict()
    active_payload["scoring_policy_versions"] = {"ami-diarization-v1": "2026-07-01"}

    check = check_release_runtime_config(record, active_payload)
    report = release_revalidation_report(record, active_payload)

    assert check.status == "degraded"
    assert check.confident_speaker_attribution_enabled is False
    assert check.degraded_route == "needs_review"
    assert check.degraded_transcript_output_allowed is True
    assert report.emergency_degraded_route == "needs_review"
    assert report.triggers == ("scoring_policy",)


def test_self_hosted_release_requires_package_version_pins():
    missing_packages = _engine_config(parameters={"model_governance": {"checkpoint": "local-model"}})

    with pytest.raises(ValidationError, match="pin package_versions"):
        _release(engine_configs=(missing_packages,))

    malformed_packages = _engine_config(
        parameters={"model_governance": {"package_versions": ["pyannote.audio==3.1.0"]}}
    )

    with pytest.raises(ValidationError, match="package_versions must be an object"):
        _release(engine_configs=(malformed_packages,))

    non_string_package = _engine_config(
        parameters={"model_governance": {"package_versions": {"pyannote.audio": 3.1}}}
    )

    with pytest.raises(ValidationError, match="package_versions.pyannote.audio must be a string"):
        _release(engine_configs=(non_string_package,))


def test_hosted_release_requires_well_formed_provider_pins():
    missing_governance = _engine_config(provider="aws_transcribe", parameters={})

    with pytest.raises(ValidationError, match="include hosted_provider_governance"):
        _release(engine_configs=(missing_governance,))

    missing_pinning = _engine_config(
        provider="hosted-provider",
        parameters={"hosted_provider_governance": {"model_version": "diarize-2026-06"}},
    )

    with pytest.raises(ValidationError, match="pin model_version and version_pinning"):
        _release(engine_configs=(missing_pinning,))

    malformed_pinning = _engine_config(
        provider="hosted-provider",
        parameters={
            "hosted_provider_governance": {
                "model_version": "diarize-2026-06",
                "version_pinning": {"contract": "2026-06"},
            }
        },
    )

    with pytest.raises(ValidationError, match="version_pinning must be a string"):
        _release(engine_configs=(malformed_pinning,))


def test_runtime_schema_fingerprint_is_derived_from_release_artifacts():
    legacy_payload = _benchmark_report().to_dict()
    legacy_payload["schema_version"] = 1
    legacy_payload.pop("route_confusion")
    record = _release(benchmark_reports=(benchmark_report_from_dict(legacy_payload),))

    runtime_config = release_expected_runtime_config(record)

    assert runtime_config.schema_versions["benchmark_report"] == 1
    assert runtime_config.schema_versions["dataset_manifest"] == _manifest().schema_version
    assert runtime_config.schema_versions["release_record"] == record.schema_version


def test_invalid_active_runtime_metadata_degrades_with_audit_event():
    check = check_release_runtime_config(_release(), {"git_sha": "not-a-runtime-config"})

    assert check.status == "degraded"
    assert check.confident_speaker_attribution_enabled is False
    assert check.degraded_route == "diagnostic_only"
    assert [event.code for event in check.audit_events] == ["runtime_config_invalid"]


def test_malformed_active_runtime_metadata_keys_degrade_with_audit_event():
    check = check_release_runtime_config(_release(), {1: "not-a-runtime-config"})

    assert check.status == "degraded"
    assert check.confident_speaker_attribution_enabled is False
    assert check.degraded_route == "diagnostic_only"
    assert [event.code for event in check.audit_events] == ["runtime_config_invalid"]
    assert "field names must be strings" in check.audit_events[0].actual


def test_invalid_release_runtime_metadata_degrades_with_audit_event():
    legacy_payload = _benchmark_report().to_dict()
    legacy_payload["report_id"] = "benchmark-report-legacy"
    legacy_payload["schema_version"] = 1
    legacy_payload.pop("route_confusion")
    mixed_schema_record = _release(
        benchmark_reports=(_benchmark_report(), benchmark_report_from_dict(legacy_payload))
    )
    active_payload = release_expected_runtime_config(_release()).to_dict()

    check = check_release_runtime_config(mixed_schema_record, active_payload)

    assert check.status == "degraded"
    assert check.confident_speaker_attribution_enabled is False
    assert check.degraded_route == "diagnostic_only"
    assert [event.code for event in check.audit_events] == ["release_runtime_config_invalid"]
    assert "release.benchmark_reports.schema_version" in check.audit_events[0].actual


def test_release_loader_requires_every_must_not_have_check():
    payload = _release().to_dict()
    payload["must_not_have_checks"].pop("no_cross_call_speaker_ids")

    with pytest.raises(ValidationError, match="must_not_have.no_cross_call_speaker_ids is required"):
        release_record_from_dict(payload)


def test_approved_release_rejects_failed_must_not_have_checks():
    with pytest.raises(ValidationError, match="all must-not-have checks"):
        _release(must_not_have_checks=_must_not_have_checks(no_tuned_on_holdout_result=False))


def test_approved_release_rejects_unpinned_engine_config():
    unpinned = _engine_config(model_version=None, config_id=None, parameters={})

    with pytest.raises(ValidationError, match="pin model_version or config_id"):
        _release(engine_configs=(unpinned,))


def test_approved_release_rejects_unaccepted_or_private_coverage_gap_branch_decisions():
    coverage_gap = _branch_decision_with(
        decision="needs_more_private_coverage",
        private_coverage_ready=False,
    )

    with pytest.raises(ValidationError, match="accepted branch decisions"):
        _release(branch_decisions=(coverage_gap,))

    degraded_only = _branch_decision_with(decision="ship_degraded_only")

    with pytest.raises(ValidationError, match="accepted branch decisions"):
        _release(branch_decisions=(degraded_only,))

    accepted_without_private_coverage = _branch_decision_with(private_coverage_ready=False)

    with pytest.raises(ValidationError, match="private coverage ready"):
        _release(branch_decisions=(accepted_without_private_coverage,))


def test_release_rejects_forbidden_cross_call_identity_metadata():
    with pytest.raises(ValidationError, match="voice_profile must not persist"):
        _golden_test(metadata={"voice_profile": "do-not-store"})

    with pytest.raises(ValidationError, match="voice_profile must not persist"):
        _must_not_have_checks(evidence={"voice_profile": "do-not-store"})

    with pytest.raises(ValidationError, match="embedding must not persist"):
        _must_not_have_checks(evidence={"embedding": "do-not-store"})

    with pytest.raises(ValidationError, match="reference_speaker_id must not persist"):
        validate_release_runtime_output({"words": [{"reference_speaker_id": "AMI-P1"}]})

    with pytest.raises(ValidationError, match="embeddings must not persist"):
        validate_release_runtime_output({"diagnostics": {"embeddings": [0.1, 0.2]}})
