import pytest

from keyframe.diarization import (
    PRIVATE_ACCEPTANCE_METADATA_SCHEMA_VERSION,
    PRIVATE_ANNOTATION_QUALITY_GATE_STAGE,
    DatasetAccess,
    DatasetManifest,
    DatasetSplitManifest,
    ExpectedDatasetFile,
    PrivateAcceptanceMetadata,
    PrivateAcceptanceSlice,
    PrivateAnnotationProtocol,
    PrivateAnnotationQualityGateConfig,
    PrivateAnnotationQualityMetrics,
    ScoringPolicyManifest,
    ValidationError,
    dataset_manifest_json_dumps,
    dataset_manifest_json_loads,
    evaluate_private_annotation_quality,
    private_acceptance_metadata_from_dict,
    private_acceptance_metadata_json_dumps,
    private_acceptance_metadata_json_loads,
)


_DEFAULT_QUALITY_METRICS = object()


def _protocol():
    return PrivateAnnotationProtocol(
        protocol_id="private-call-annotation",
        version="2026-06-19",
        transcript_normalization="casefold punctuation, normalize filler tokens, preserve unintelligible markers",
        speaker_span_rules=(
            "mark contiguous same-speaker spans on canonical milliseconds",
            "split spans at speaker changes and long silences",
        ),
        overlap_rules=(
            "mark overlapping speech when two speakers are active in the same interval",
            "score overlapped words only when the policy explicitly enables overlap scoring",
        ),
        unintelligible_no_score_rules=(
            "mark unintelligible speech as no_score instead of forcing transcript text",
            "exclude no_score regions from product acceptance gates",
        ),
        critical_span_label_rules=(
            "flag missed action items, commitments, and names as critical spans",
            "adjudicate critical-span disagreements before model gates",
        ),
    )


def _quality_metrics(**overrides):
    values = {
        "adjudication_change_rate": 0.04,
        "agreement_metrics": {
            "overlap_agreement": 0.91,
            "speaker_span_agreement": 0.94,
            "transcript_normalization_agreement": 0.98,
        },
        "annotated_recording_count": 20,
        "double_annotated_recording_count": 6,
        "double_annotated_sample_rate": 0.30,
        "unresolved_disagreement_rate": 0.01,
    }
    values.update(overrides)
    return PrivateAnnotationQualityMetrics(**values)


def _metadata(quality_metrics=_DEFAULT_QUALITY_METRICS):
    if quality_metrics is _DEFAULT_QUALITY_METRICS:
        quality_metrics = _quality_metrics()
    return PrivateAcceptanceMetadata(
        metadata_id="private-acceptance-v1",
        protocol=_protocol(),
        slices=(
            PrivateAcceptanceSlice(
                slice_id="adjudicated-core",
                label="adjudicated",
                recording_count=12,
                duration_ms=1_800_000,
                critical_span_count=6,
            ),
            PrivateAcceptanceSlice(
                slice_id="diagnostic-unadjudicated",
                label="unadjudicated_diagnostic",
                recording_count=2,
                duration_ms=300_000,
                reason="diagnostic slice excluded from product acceptance gates until adjudicated",
            ),
            PrivateAcceptanceSlice(
                slice_id="unintelligible-no-score",
                label="no_score",
                recording_count=1,
                duration_ms=90_000,
                no_score_region_count=4,
                reason="contains unintelligible regions that should not be forced into labels",
            ),
            PrivateAcceptanceSlice(
                slice_id="reference-needs-stabilization",
                label="reference_unstable",
                recording_count=1,
                duration_ms=120_000,
                reason="annotators disagreed on the reference transcript boundary",
            ),
        ),
        quality_metrics=quality_metrics,
    )


def _private_manifest():
    return DatasetManifest(
        dataset_id="private-call-acceptance",
        name="Private Call Acceptance Placeholder",
        role="private_in_domain_acceptance",
        access=DatasetAccess(mode="local_only", redistribution="forbidden"),
        license_url="https://example.com/private-acceptance-license",
        attribution="Private in-domain acceptance metadata placeholder",
        expected_files=(
            ExpectedDatasetFile(
                path="private/acceptance_manifest.json",
                checksum_sha256="0" * 64,
                file_role="manifest",
                size_bytes=0,
            ),
        ),
        splits=(
            DatasetSplitManifest(
                split_id="private-acceptance",
                role="private_in_domain_acceptance",
                expected_file_paths=("private/acceptance_manifest.json",),
                scoring_policy_id="private-product-v1",
                recording_ids=("private-call-001",),
            ),
        ),
        scoring_policies=(
            ScoringPolicyManifest(
                policy_id="private-product-v1",
                version="1",
                description="Private product transcript acceptance policy",
                policy_kind="product_transcript",
                collar_ms=0,
                score_overlap=False,
                channel_mode="rendered_transcript",
                speaker_count_mode="session_local",
                text_normalization="casefold_punctuation",
                metric_set=("word_speaker_label_accuracy", "turn_speaker_label_accuracy"),
            ),
        ),
        private_acceptance=_metadata(),
    )


def test_private_acceptance_manifest_round_trips_synthetic_annotation_labels():
    manifest = _private_manifest()
    text = dataset_manifest_json_dumps(manifest)

    loaded = dataset_manifest_json_loads(text)

    assert loaded.private_acceptance is not None
    assert loaded.private_acceptance.schema_version == PRIVATE_ACCEPTANCE_METADATA_SCHEMA_VERSION
    assert loaded.private_acceptance.protocol.reference_speaker_id_policy == "candidate_invisible"
    assert [item.label for item in loaded.private_acceptance.slices] == [
        "adjudicated",
        "unadjudicated_diagnostic",
        "no_score",
        "reference_unstable",
    ]
    assert loaded.to_dict() == manifest.to_dict()


def test_private_acceptance_metadata_json_round_trip_is_stable():
    metadata = _metadata()
    text = private_acceptance_metadata_json_dumps(metadata)

    loaded = private_acceptance_metadata_json_loads(text)

    assert loaded.to_dict() == metadata.to_dict()
    assert '"schema_version": 1' in text


def test_private_acceptance_rejects_reference_speaker_identity_fields():
    payload = _metadata().to_dict()
    payload["slices"][0]["reference_speaker_ids"] = ["AMI-P1"]

    with pytest.raises(ValidationError, match="reference_speaker_ids must remain candidate-invisible"):
        private_acceptance_metadata_from_dict(payload)


def test_private_annotation_quality_rejects_contradictory_double_annotation_rate():
    with pytest.raises(
        ValidationError,
        match="double_annotated_sample_rate must match double_annotated_recording_count",
    ):
        _quality_metrics(
            annotated_recording_count=20,
            double_annotated_recording_count=0,
            double_annotated_sample_rate=0.30,
        )


def test_private_annotation_quality_rejects_identity_metric_keys_from_direct_constructor():
    with pytest.raises(ValidationError, match="agreement_metrics.speaker_ref must remain candidate-invisible"):
        _quality_metrics(agreement_metrics={"speaker_ref": 0.99})


def test_private_acceptance_slice_validation_for_no_score_and_reference_unstable_labels():
    with pytest.raises(ValidationError, match="no_score.*require.*no_score_region_count"):
        PrivateAcceptanceSlice(
            slice_id="bad-no-score",
            label="no_score",
            recording_count=1,
            duration_ms=1_000,
        )

    with pytest.raises(ValidationError, match="reference_unstable.*require.*reason"):
        PrivateAcceptanceSlice(
            slice_id="bad-reference",
            label="reference_unstable",
            recording_count=1,
            duration_ms=1_000,
        )


def test_private_annotation_quality_gate_blocks_low_quality_data_before_model_gates():
    metadata = _metadata(
        quality_metrics=_quality_metrics(
            adjudication_change_rate=0.30,
            agreement_metrics={
                "overlap_agreement": 0.70,
                "speaker_span_agreement": 0.80,
                "transcript_normalization_agreement": 0.90,
            },
            double_annotated_recording_count=1,
            double_annotated_sample_rate=0.05,
            unresolved_disagreement_rate=0.08,
        )
    )

    result = evaluate_private_annotation_quality(metadata)

    assert result.status == "failed"
    assert result.gate_stage == PRIVATE_ANNOTATION_QUALITY_GATE_STAGE
    assert result.blocks_model_gates is True
    assert "double_annotated_sample_rate below threshold" in result.reasons
    assert "agreement metric below threshold: speaker_span_agreement" in result.reasons
    assert "adjudication_change_rate above threshold" in result.reasons
    assert "unresolved_disagreement_rate above threshold" in result.reasons
    assert result.to_dict()["blocks_model_gates"] is True


def test_private_annotation_quality_gate_reports_unavailable_before_model_gates():
    result = evaluate_private_annotation_quality(_metadata(quality_metrics=None), PrivateAnnotationQualityGateConfig())

    assert result.status == "unavailable"
    assert result.blocks_model_gates is True
    assert result.metrics is None
    assert result.reasons == ("private annotation quality metrics are unavailable",)
