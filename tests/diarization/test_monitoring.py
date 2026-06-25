import pytest

from keyframe.diarization import (
    MONITORING_RECORD_SCHEMA_VERSION,
    MonitoringAggregate,
    MonitoringPromotionState,
    MonitoringRecord,
    ValidationError,
    advance_monitoring_promotion_state,
    aggregate_monitoring_records,
    monitoring_record_from_dict,
    monitoring_record_json_dumps,
    monitoring_record_json_loads,
)


def _timeline_metadata(**overrides):
    values = {
        "channel_ids": ["ch-1"],
        "duration_ms": 120_000,
        "sample_rate_hz": 16_000,
        "time_basis": "canonical_ms",
        "timeline_id": "timeline-session-001",
        "transform_chain_id": "identity",
    }
    values.update(overrides)
    return values


def _record(**overrides):
    values = {
        "monitoring_record_id": "monitoring-session-001",
        "release_candidate_id": "release-2026-06-24",
        "monitoring_policy_version": "monitoring-v1",
        "release_record_schema_version": 2,
        "engine_config_versions": {"release-engine": "config-001"},
        "preflight_policy_id": "launch-preflight",
        "preflight_policy_version": "2026-06-24",
        "route": "confident_pipeline",
        "confident_speaker_attribution_enabled": True,
        "canonical_timeline_metadata": _timeline_metadata(),
        "edit_operation_counts": {"merge_speakers": 1, "rename_label": 2},
        "review_time_ms": 45_000,
        "retention_class": "diagnostic_30d",
        "expires_at": "2026-07-24T00:00:00Z",
    }
    values.update(overrides)
    return MonitoringRecord(**values)


def test_monitoring_record_round_trips_versions_route_timeline_review_and_retention():
    record = _record()

    payload = record.to_dict()
    loaded = monitoring_record_json_loads(monitoring_record_json_dumps(record))

    assert payload["schema_version"] == MONITORING_RECORD_SCHEMA_VERSION
    assert payload["monitoring_policy_version"] == "monitoring-v1"
    assert payload["release_record_schema_version"] == 2
    assert payload["engine_config_versions"] == {"release-engine": "config-001"}
    assert payload["preflight_policy_version"] == "2026-06-24"
    assert payload["route"] == "confident_pipeline"
    assert payload["confident_speaker_attribution_enabled"] is True
    assert payload["degraded_route"] is None
    assert payload["canonical_timeline_metadata"]["timeline_id"] == "timeline-session-001"
    assert payload["edit_operation_counts"] == {"merge_speakers": 1, "rename_label": 2}
    assert payload["review_time_ms"] == 45_000
    assert payload["retention_class"] == "diagnostic_30d"
    assert payload["expires_at"] == "2026-07-24T00:00:00Z"
    assert loaded.to_dict() == payload
    assert monitoring_record_from_dict(payload).to_dict() == payload


def test_monitoring_record_rejects_malformed_payload_field_names():
    payload = _record().to_dict()
    payload[1] = "not-a-monitoring-field"

    with pytest.raises(ValidationError, match="monitoring field names must be strings"):
        monitoring_record_from_dict(payload)


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("raw_audio_path", "/tmp/raw.wav"),
        ("voice_profile", "do-not-store"),
        ("Voice_Profile", "do-not-store"),
        ("embedding", [0.1, 0.2]),
        ("audio_fingerprint", "do-not-store"),
        ("audioFingerprint", "do-not-store"),
        ("speakerEmbedding", [0.1, 0.2]),
        ("cross_session_speaker_key", "speaker-global-1"),
        ("crossSessionSpeakerKey", "speaker-global-1"),
        ("cross_recording_identity", "speaker-global-1"),
        ("crossRecordingIdentity", "speaker-global-1"),
        ("global_identity", "speaker-global-1"),
        ("globalIdentity", "speaker-global-1"),
        ("identity_profile", "speaker-global-1"),
        ("identityProfile", "speaker-global-1"),
        ("profile_id", "speaker-global-1"),
        ("profileId", "speaker-global-1"),
        ("corpus_identity", "speaker-global-1"),
        ("source_speaker_id", "speaker-global-1"),
        ("evaluator_speaker_map", {"cluster-a": "AMI-P1"}),
        ("oracle_metadata", {"speaker": "AMI-P1"}),
        ("display_label", "Speaker 1"),
        ("gold_label", "AMI-P1"),
        ("reference_label", "AMI-P1"),
        ("participantEmail", "speaker@example.com"),
        ("participantName", "Speaker One"),
        ("userId", "user-001"),
        ("customerId", "customer-001"),
    ),
)
def test_monitoring_records_reject_sensitive_identity_or_audio_fields(field_name, value):
    metadata = _timeline_metadata(extra={field_name: value})

    with pytest.raises(ValidationError, match=field_name):
        _record(canonical_timeline_metadata=metadata)


def test_monitoring_records_reject_unsupported_timeline_metadata_fields():
    with pytest.raises(ValidationError, match="monitoring.timeline has unsupported fields: processing_notes"):
        _record(canonical_timeline_metadata=_timeline_metadata(processing_notes="manual notes"))


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("engine_config_versions", {"audio_fingerprint": "config-001"}),
        ("engine_config_versions", {"audioFingerprint": "config-001"}),
        ("engine_config_versions", {"audio_id": "config-001"}),
        ("engine_config_versions", {"audioId": "config-001"}),
        ("engine_config_versions", {"recordingId": "config-001"}),
        ("engine_config_versions", {"audio_hash": "config-001"}),
        ("engine_config_versions", {"audioHash": "config-001"}),
        ("engine_config_versions", {"recording_hash": "config-001"}),
        ("engine_config_versions", {"recordingHash": "config-001"}),
        ("engine_config_versions", {"recordingSha256": "config-001"}),
        ("engine_config_versions", {"audioGuid": "config-001"}),
        ("engine_config_versions", {"recordingUuid": "config-001"}),
        ("engine_config_versions", {"session_id": "config-001"}),
        ("engine_config_versions", {"sessionId": "config-001"}),
        ("engine_config_versions", {"callId": "config-001"}),
        ("engine_config_versions", {"meeting_id": "config-001"}),
        ("engine_config_versions", {"conversationId": "config-001"}),
        ("engine_config_versions", {"sid": "config-001"}),
        ("engine_config_versions", {"sessId": "config-001"}),
        ("engine_config_versions", {"cid": "config-001"}),
        ("engine_config_versions", {"mtgId": "config-001"}),
        ("engine_config_versions", {"convId": "config-001"}),
        ("engine_config_versions", {"sidValue": "config-001"}),
        ("engine_config_versions", {"sid_hash": "config-001"}),
        ("engine_config_versions", {"cidValue": "config-001"}),
        ("engine_config_versions", {"mtgIdValue": "config-001"}),
        ("engine_config_versions", {"convIdHash": "config-001"}),
        ("engine_config_versions", {"sidHashValue": "config-001"}),
        ("engine_config_versions", {"sidId": "config-001"}),
        ("engine_config_versions", {"cid_hash_value": "config-001"}),
        ("engine_config_versions", {"mtgGuid": "config-001"}),
        ("engine_config_versions", {"mtgUuid": "config-001"}),
        ("engine_config_versions", {"mtgToken": "config-001"}),
        ("engine_config_versions", {"convGuid": "config-001"}),
        ("engine_config_versions", {"convUuid": "config-001"}),
        ("engine_config_versions", {"convToken": "config-001"}),
        ("engine_config_versions", {"sessGuid": "config-001"}),
        ("engine_config_versions", {"candidateSidHashValue": "config-001"}),
        ("engine_config_versions", {"sourceMtgGuid": "config-001"}),
        ("engine_config_versions", {"candidateConvToken": "config-001"}),
        ("engine_config_versions", {"monitorSidId": "config-001"}),
        ("engine_config_versions", {"candidateconvtoken": "config-001"}),
        ("engine_config_versions", {"contactEmail": "config-001"}),
        ("engine_config_versions", {"participantEmailAddress": "config-001"}),
        ("engine_config_versions", {"customerAccountId": "config-001"}),
        ("engine_config_versions", {"accountIdentifier": "config-001"}),
        ("engine_config_versions", {"crossCallSpeakerKey": "config-001"}),
        ("engine_config_versions", {"crossRecordingSpeakerKey": "config-001"}),
        ("engine_config_versions", {"sourceSpeakerKey": "config-001"}),
        ("engine_config_versions", {"speakerKey": "config-001"}),
        ("engine_config_versions", {"crossCallSpeakerUuid": "config-001"}),
        ("engine_config_versions", {"crossRecordingSpeakerGuid": "config-001"}),
        ("engine_config_versions", {"sourceSpeakerUuid": "config-001"}),
        ("engine_config_versions", {"sessionUuid": "config-001"}),
        ("engine_config_versions", {"userUuid": "config-001"}),
        ("edit_operation_counts", {"cross_session_speaker_key": 1}),
        ("edit_operation_counts", {"crossSessionSpeakerKey": 1}),
    ),
)
def test_monitoring_records_reject_sensitive_map_keys(field_name, value):
    with pytest.raises(ValidationError, match="must not persist sensitive monitoring identity material"):
        _record(**{field_name: value})


def test_monitoring_records_allow_non_identifier_guidance_keys():
    record = _record(
        engine_config_versions={
            "releaseGuideVersion": "config-001",
            "guidanceVersion": "config-002",
            "sourceGuideline": "config-003",
        }
    )

    assert record.to_dict()["engine_config_versions"] == {
        "guidanceVersion": "config-002",
        "releaseGuideVersion": "config-001",
        "sourceGuideline": "config-003",
    }


def test_monitoring_records_reject_unknown_edit_operation_keys():
    with pytest.raises(ValidationError, match="monitoring.edit_operation_counts.custom_edit is not supported"):
        _record(edit_operation_counts={"custom_edit": 1})


def test_monitoring_record_maps_are_immutable_after_validation():
    record = _record()

    with pytest.raises(TypeError):
        record.engine_config_versions["audioFingerprint"] = "config-002"

    with pytest.raises(TypeError):
        record.edit_operation_counts["sessionSpecificEdit"] = 1


def test_monitoring_record_loader_requires_explicit_promotion_state():
    payload = _record().to_dict()
    payload.pop("promotion_state")

    with pytest.raises(ValidationError, match="monitoring.promotion_state is required"):
        monitoring_record_from_dict(payload)

    payload["promotion_state"] = {}

    with pytest.raises(ValidationError, match="promotion.consent_granted is required"):
        monitoring_record_from_dict(payload)


def test_monitoring_record_loader_rejects_persisted_promotion_notes():
    payload = _record().to_dict()
    payload["promotion_state"]["note"] = "speaker@example.com"

    with pytest.raises(ValidationError, match="promotion has unsupported fields: note"):
        monitoring_record_from_dict(payload)


def test_monitoring_aggregate_rejects_sensitive_edit_total_keys():
    with pytest.raises(ValidationError, match="must not persist sensitive monitoring identity material"):
        MonitoringAggregate(
            generated_at="2026-07-01T00:00:00Z",
            total_records=0,
            active_record_count=0,
            expired_record_count=0,
            route_counts={},
            degraded_record_count=0,
            edit_operation_totals={"speakerEmbedding": 1},
            review_time_ms_total=0,
        )


def test_monitoring_aggregate_rejects_unknown_route_and_edit_total_keys():
    with pytest.raises(ValidationError, match="monitoring_aggregate.route_counts.session-route is not supported"):
        MonitoringAggregate(
            generated_at="2026-07-01T00:00:00Z",
            total_records=0,
            active_record_count=0,
            expired_record_count=0,
            route_counts={"session-route": 1},
            degraded_record_count=0,
            edit_operation_totals={},
            review_time_ms_total=0,
        )

    with pytest.raises(ValidationError, match="monitoring_aggregate.edit_operation_totals.sessionSpecificEdit"):
        MonitoringAggregate(
            generated_at="2026-07-01T00:00:00Z",
            total_records=0,
            active_record_count=0,
            expired_record_count=0,
            route_counts={},
            degraded_record_count=0,
            edit_operation_totals={"sessionSpecificEdit": 1},
            review_time_ms_total=0,
        )


def test_monitoring_aggregate_rejects_inconsistent_redaction_counts():
    with pytest.raises(ValidationError, match="route_counts must sum to total_records"):
        MonitoringAggregate(
            generated_at="2026-07-01T00:00:00Z",
            total_records=1,
            active_record_count=0,
            expired_record_count=1,
            route_counts={},
            degraded_record_count=0,
            edit_operation_totals={},
            review_time_ms_total=0,
        )

    with pytest.raises(ValidationError, match="degraded_record_count must match"):
        MonitoringAggregate(
            generated_at="2026-07-01T00:00:00Z",
            total_records=1,
            active_record_count=0,
            expired_record_count=1,
            route_counts={"needs_review": 1},
            degraded_record_count=0,
            edit_operation_totals={},
            review_time_ms_total=0,
        )

    with pytest.raises(ValidationError, match="active_monitoring_record_ids must match"):
        MonitoringAggregate(
            generated_at="2026-07-01T00:00:00Z",
            total_records=1,
            active_record_count=0,
            expired_record_count=1,
            route_counts={"confident_pipeline": 1},
            degraded_record_count=0,
            edit_operation_totals={},
            review_time_ms_total=0,
            active_monitoring_record_ids=("monitoring-expired",),
        )

    with pytest.raises(ValidationError, match="active_timeline_ids must match"):
        MonitoringAggregate(
            generated_at="2026-07-01T00:00:00Z",
            total_records=1,
            active_record_count=0,
            expired_record_count=1,
            route_counts={"confident_pipeline": 1},
            degraded_record_count=0,
            edit_operation_totals={},
            review_time_ms_total=0,
            active_timeline_ids=("timeline-expired",),
        )


def test_monitoring_aggregate_maps_are_immutable_after_validation():
    aggregate = aggregate_monitoring_records((_record(),), as_of="2026-07-01T00:00:00Z")

    with pytest.raises(TypeError):
        aggregate.route_counts["session-route"] = 1

    with pytest.raises(TypeError):
        aggregate.edit_operation_totals["speakerEmbedding"] = 1


def test_monitoring_records_require_expiry_and_valid_degraded_state():
    with pytest.raises(ValidationError, match="monitoring.expires_at"):
        _record(expires_at=None)

    with pytest.raises(ValidationError, match="cannot enable confident_speaker_attribution"):
        _record(route="needs_review", confident_speaker_attribution_enabled=True)

    with pytest.raises(ValidationError, match="confident_pipeline monitoring records require"):
        _record(
            route="confident_pipeline",
            confident_speaker_attribution_enabled=False,
            degraded_route="needs_review",
        )

    with pytest.raises(ValidationError, match="degraded monitoring records require degraded_route"):
        _record(route="needs_review", confident_speaker_attribution_enabled=False)

    with pytest.raises(ValidationError, match="degraded_route must match route"):
        _record(
            route="needs_review",
            confident_speaker_attribution_enabled=False,
            degraded_route="diagnostic_only",
        )

    degraded = _record(
        route="needs_review",
        confident_speaker_attribution_enabled=False,
        degraded_route="needs_review",
    )
    unsupported = _record(
        route="unsupported",
        confident_speaker_attribution_enabled=False,
        degraded_route="unsupported",
    )

    assert degraded.to_dict()["degraded_route"] == "needs_review"
    assert unsupported.to_dict()["degraded_route"] == "unsupported"


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"timeline_id": 123}, "monitoring.timeline.timeline_id must be a string"),
        ({"transform_chain_id": ""}, "monitoring.timeline.transform_chain_id is required"),
        ({"duration_ms": -1}, "monitoring.timeline.duration_ms must be greater than 0"),
        ({"sample_rate_hz": "bad"}, "monitoring.timeline.sample_rate_hz must be an integer"),
        ({"time_basis": "wall_clock"}, "monitoring.timeline.time_basis is not supported"),
    ),
)
def test_monitoring_records_validate_required_timeline_metadata_shape(override, message):
    with pytest.raises(ValidationError, match=message):
        _record(canonical_timeline_metadata=_timeline_metadata(**override))


def test_monitoring_records_reject_missing_required_timeline_metadata_fields():
    metadata = _timeline_metadata()
    metadata.pop("time_basis")
    metadata.pop("duration_ms")

    with pytest.raises(ValidationError, match="monitoring.timeline missing required fields: duration_ms, time_basis"):
        _record(canonical_timeline_metadata=metadata)


def test_monitoring_records_normalize_required_timeline_metadata_ids():
    payload = _record(
        canonical_timeline_metadata=_timeline_metadata(
            channel_ids=(" ch-1 ",),
            time_basis=" canonical_ms ",
            timeline_id=" timeline-session-001 ",
            transform_chain_id=" identity ",
        )
    ).to_dict()

    assert payload["canonical_timeline_metadata"] == {
        "channel_ids": ["ch-1"],
        "duration_ms": 120_000,
        "sample_rate_hz": 16_000,
        "time_basis": "canonical_ms",
        "timeline_id": "timeline-session-001",
        "transform_chain_id": "identity",
    }


def test_monitoring_promotion_requires_consent_access_split_annotation_and_adjudication():
    state = MonitoringPromotionState()

    with pytest.raises(ValidationError, match="requires consent and access check"):
        advance_monitoring_promotion_state(state, "assign_split", split_id="private_acceptance")

    state = advance_monitoring_promotion_state(state, "grant_consent")
    state = advance_monitoring_promotion_state(state, "verify_access")
    state = advance_monitoring_promotion_state(state, "assign_split", split_id="private_acceptance")

    with pytest.raises(ValidationError, match="adjudication requires annotation"):
        MonitoringPromotionState(consent_granted=True, access_checked=True, split_id="private_acceptance", adjudicated=True)

    with pytest.raises(ValidationError, match="annotation requires annotation_protocol_version"):
        MonitoringPromotionState(
            consent_granted=True,
            access_checked=True,
            split_id="private_acceptance",
            annotated=True,
        )

    state = advance_monitoring_promotion_state(
        state,
        "mark_annotated",
        annotation_protocol_version="private-annotation@2026-06-24",
    )
    assert advance_monitoring_promotion_state(state, "assign_split", split_id="private_acceptance") == state

    with pytest.raises(ValidationError, match="cannot change after annotation or adjudication"):
        advance_monitoring_promotion_state(state, "assign_split", split_id="private_holdout")

    with pytest.raises(ValidationError, match="annotation_protocol_version cannot change"):
        advance_monitoring_promotion_state(
            state,
            "mark_annotated",
            annotation_protocol_version="private-annotation@2026-06-25",
        )

    with pytest.raises(ValidationError, match="adjudication requires adjudication_id"):
        MonitoringPromotionState(
            consent_granted=True,
            access_checked=True,
            split_id="private_acceptance",
            annotated=True,
            annotation_protocol_version="private-annotation@2026-06-24",
            adjudicated=True,
        )

    state = advance_monitoring_promotion_state(state, "mark_adjudicated", adjudication_id="adjudication-001")
    assert advance_monitoring_promotion_state(state, "mark_adjudicated", adjudication_id="adjudication-001") == state

    with pytest.raises(ValidationError, match="adjudication_id cannot change"):
        advance_monitoring_promotion_state(state, "mark_adjudicated", adjudication_id="adjudication-002")

    assert state.eligible_for_private_fixture is True
    assert _record(promotion_state=state).to_dict()["promotion_state"]["eligible_for_private_fixture"] is True


def test_monitoring_aggregation_redacts_expired_session_local_identifiers():
    active = _record(
        monitoring_record_id="monitoring-active",
        canonical_timeline_metadata=_timeline_metadata(timeline_id="timeline-active"),
        edit_operation_counts={"rename_label": 2},
        review_time_ms=10_000,
        expires_at="2026-07-24T00:00:00Z",
    )
    expired = _record(
        monitoring_record_id="monitoring-expired",
        route="needs_review",
        confident_speaker_attribution_enabled=False,
        degraded_route="needs_review",
        canonical_timeline_metadata=_timeline_metadata(timeline_id="timeline-expired"),
        edit_operation_counts={"merge_speakers": 1},
        review_time_ms=20_000,
        expires_at="2026-06-24T00:00:00Z",
    )

    aggregate = aggregate_monitoring_records((active, expired), as_of="2026-07-01T00:00:00Z")
    payload = aggregate.to_dict()

    assert payload["total_records"] == 2
    assert payload["active_record_count"] == 1
    assert payload["expired_record_count"] == 1
    assert payload["degraded_record_count"] == 1
    assert payload["route_counts"] == {"confident_pipeline": 1, "needs_review": 1}
    assert payload["edit_operation_totals"] == {"merge_speakers": 1, "rename_label": 2}
    assert payload["review_time_ms_total"] == 30_000
    assert payload["active_monitoring_record_ids"] == ["monitoring-active"]
    assert payload["active_timeline_ids"] == ["timeline-active"]
    assert "monitoring-expired" not in str(payload)
    assert "timeline-expired" not in str(payload)


def test_monitoring_aggregation_allows_shared_active_timeline_ids():
    first = _record(
        monitoring_record_id="monitoring-active-1",
        canonical_timeline_metadata=_timeline_metadata(timeline_id="timeline-shared"),
    )
    second = _record(
        monitoring_record_id="monitoring-active-2",
        canonical_timeline_metadata=_timeline_metadata(timeline_id="timeline-shared"),
    )

    aggregate = aggregate_monitoring_records((first, second), as_of="2026-07-01T00:00:00Z")
    payload = aggregate.to_dict()

    assert payload["active_record_count"] == 2
    assert payload["active_monitoring_record_ids"] == ["monitoring-active-1", "monitoring-active-2"]
    assert payload["active_timeline_ids"] == ["timeline-shared", "timeline-shared"]
