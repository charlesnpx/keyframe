"""Immutable release records and runtime compatibility checks."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from keyframe.diarization.adapters import BENCHMARK_RUN_RECORD_SCHEMA_VERSION
from keyframe.diarization.engines import EngineConfigMetadata
from keyframe.diarization.manifests import ScoringPolicyManifest, dataset_manifest_from_dict, scoring_policy_from_dict
from keyframe.diarization.models import ValidationError
from keyframe.diarization.pipelines import BranchAcceptanceRecord
from keyframe.diarization.preflight import (
    ALLOWED_PREFLIGHT_ROUTES,
    PreflightJobRecord,
    PreflightPolicy,
    PreflightRoute,
)
from keyframe.diarization.private_acceptance import PrivateAnnotationProtocol
from keyframe.diarization.reports import (
    BenchmarkReport,
    benchmark_report_from_dict,
)


RELEASE_RECORD_SCHEMA_VERSION = 1
SUPPORTED_RELEASE_RECORD_SCHEMA_VERSIONS = frozenset({1})

ReleaseApprovalStatus = Literal["pending", "approved", "rejected"]
ReleaseGovernanceDecision = Literal["approve_confident_labels", "degraded_only", "reject"]
ReleaseRuntimeCheckStatus = Literal["approved", "degraded"]
ReleaseGoldenTestStatus = Literal["passed", "failed"]

_APPROVAL_STATUSES = frozenset({"pending", "approved", "rejected"})
_GOVERNANCE_DECISIONS = frozenset({"approve_confident_labels", "degraded_only", "reject"})
_RUNTIME_CHECK_STATUSES = frozenset({"approved", "degraded"})
_GOLDEN_TEST_STATUSES = frozenset({"passed", "failed"})
_HEX_DIGITS = frozenset("0123456789abcdef")
_FORBIDDEN_RUNTIME_IDENTITY_KEYS = frozenset(
    {
        "canonical_audio_id",
        "corpus_speaker_id",
        "corpus_speaker_ids",
        "cross_call_speaker_id",
        "cross_recording_identity",
        "cross_session_speaker_key",
        "global_identity",
        "identity_profile",
        "local_audio_sha256",
        "original_audio_id",
        "participant_id",
        "participant_ids",
        "profile_id",
        "reference_speaker_id",
        "reference_speaker_ids",
        "retained_voice_profile",
        "speaker_embedding",
        "speaker_embeddings",
        "speaker_profile",
        "voice_embedding",
        "voice_embeddings",
        "voice_fingerprint",
        "voice_fingerprints",
        "voice_profile",
        "voice_profiles",
    }
)
_HOSTED_PROVIDER_CONFIG_PROVIDERS = frozenset({"aws_transcribe", "google_speech", "deepgram"})


@dataclass(frozen=True)
class ReleaseMustNotHaveChecks:
    """Required release gates for privacy, provenance, and holdout hygiene."""

    no_cross_call_speaker_ids: bool
    no_retained_voice_profiles_or_embeddings: bool
    no_reference_speaker_ids_in_runtime_output: bool
    no_unpinned_model_or_provider_config: bool
    no_tuned_on_holdout_result: bool
    evidence: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "no_cross_call_speaker_ids",
            "no_retained_voice_profiles_or_embeddings",
            "no_reference_speaker_ids_in_runtime_output",
            "no_unpinned_model_or_provider_config",
            "no_tuned_on_holdout_result",
        ):
            object.__setattr__(self, field_name, _require_bool(getattr(self, field_name), f"must_not_have.{field_name}"))
        evidence = _validate_string_map(self.evidence, "must_not_have.evidence")
        _reject_forbidden_runtime_identity_fields(evidence, "must_not_have.evidence")
        object.__setattr__(self, "evidence", _freeze_json(evidence))

    @property
    def passed(self) -> bool:
        return (
            self.no_cross_call_speaker_ids
            and self.no_retained_voice_profiles_or_embeddings
            and self.no_reference_speaker_ids_in_runtime_output
            and self.no_unpinned_model_or_provider_config
            and self.no_tuned_on_holdout_result
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence": dict(self.evidence),
            "no_cross_call_speaker_ids": self.no_cross_call_speaker_ids,
            "no_reference_speaker_ids_in_runtime_output": self.no_reference_speaker_ids_in_runtime_output,
            "no_retained_voice_profiles_or_embeddings": self.no_retained_voice_profiles_or_embeddings,
            "no_tuned_on_holdout_result": self.no_tuned_on_holdout_result,
            "no_unpinned_model_or_provider_config": self.no_unpinned_model_or_provider_config,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class ReleaseGoldenTestResult:
    """Golden test evidence captured in an immutable release candidate."""

    test_id: str
    status: ReleaseGoldenTestStatus
    artifact_hash: str | None = None
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "test_id", _require_id(self.test_id, "golden_test.test_id"))
        object.__setattr__(
            self,
            "status",
            _validate_choice(self.status, _GOLDEN_TEST_STATUSES, "golden_test.status"),
        )
        object.__setattr__(self, "artifact_hash", _optional_sha256(self.artifact_hash, "golden_test.artifact_hash"))
        object.__setattr__(self, "reason", _optional_text(self.reason, "golden_test.reason"))
        metadata = _validate_metadata(self.metadata, "golden_test.metadata")
        _reject_forbidden_runtime_identity_fields(metadata, "golden_test.metadata")
        object.__setattr__(self, "metadata", _freeze_json(metadata))
        if self.status == "passed" and self.reason is not None:
            raise ValidationError("passed golden tests cannot include a reason")
        if self.status == "failed" and self.reason is None:
            raise ValidationError("failed golden tests require a reason")

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_hash": self.artifact_hash,
            "metadata": _thaw_json(self.metadata),
            "passed": self.passed,
            "reason": self.reason,
            "status": self.status,
            "test_id": self.test_id,
        }


@dataclass(frozen=True)
class ReleaseRouteState:
    """Route fields copied from persisted preflight job evidence."""

    policy_id: str
    policy_version: str
    route: PreflightRoute
    effective_route: PreflightRoute
    validated_launch_scope_version: str
    manual_override_applied: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _require_id(self.policy_id, "route_state.policy_id"))
        object.__setattr__(self, "policy_version", _require_id(self.policy_version, "route_state.policy_version"))
        object.__setattr__(self, "route", _validate_preflight_route(self.route, "route_state.route"))
        object.__setattr__(
            self,
            "effective_route",
            _validate_preflight_route(self.effective_route, "route_state.effective_route"),
        )
        object.__setattr__(
            self,
            "validated_launch_scope_version",
            _require_id(self.validated_launch_scope_version, "route_state.validated_launch_scope_version"),
        )
        object.__setattr__(
            self,
            "manual_override_applied",
            _require_bool(self.manual_override_applied, "route_state.manual_override_applied"),
        )
        if not self.manual_override_applied and self.effective_route != self.route:
            raise ValidationError("route_state.effective_route must match route without manual override")

    @classmethod
    def from_preflight_job(cls, preflight: PreflightJobRecord) -> "ReleaseRouteState":
        if not isinstance(preflight, PreflightJobRecord):
            raise ValidationError("preflight must be a PreflightJobRecord")
        return cls(
            policy_id=preflight.decision.policy_id,
            policy_version=preflight.decision.policy_version,
            route=preflight.route,
            effective_route=preflight.effective_route,
            validated_launch_scope_version=preflight.validated_launch_scope_version,
            manual_override_applied=preflight.manual_override_applied,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "effective_route": self.effective_route,
            "manual_override_applied": self.manual_override_applied,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "route": self.route,
            "validated_launch_scope_version": self.validated_launch_scope_version,
        }


@dataclass(frozen=True)
class ReleaseRuntimeConfig:
    """Runtime fingerprint that must match an approved release before confident labels are enabled."""

    git_sha: str
    schema_versions: dict[str, int]
    dataset_snapshot_ids: dict[str, str]
    private_acceptance_split_id: str
    annotation_protocol_version: str
    scoring_policy_versions: dict[str, str]
    preflight_policy_id: str
    preflight_policy_version: str
    route: PreflightRoute
    branch_decisions: dict[str, str]
    engine_config_ids: dict[str, str]
    validated_scope: tuple[str, ...]
    unsupported_scope: tuple[str, ...]
    governance_decision: ReleaseGovernanceDecision

    def __post_init__(self) -> None:
        object.__setattr__(self, "git_sha", _validate_git_sha(self.git_sha, "runtime_config.git_sha"))
        object.__setattr__(
            self,
            "schema_versions",
            _freeze_json(_validate_int_map(self.schema_versions, "runtime_config.schema_versions")),
        )
        object.__setattr__(
            self,
            "dataset_snapshot_ids",
            _freeze_json(_validate_string_map(self.dataset_snapshot_ids, "runtime_config.dataset_snapshot_ids")),
        )
        object.__setattr__(
            self,
            "private_acceptance_split_id",
            _require_id(self.private_acceptance_split_id, "runtime_config.private_acceptance_split_id"),
        )
        object.__setattr__(
            self,
            "annotation_protocol_version",
            _require_id(self.annotation_protocol_version, "runtime_config.annotation_protocol_version"),
        )
        object.__setattr__(
            self,
            "scoring_policy_versions",
            _freeze_json(_validate_string_map(self.scoring_policy_versions, "runtime_config.scoring_policy_versions")),
        )
        object.__setattr__(
            self,
            "preflight_policy_id",
            _require_id(self.preflight_policy_id, "runtime_config.preflight_policy_id"),
        )
        object.__setattr__(
            self,
            "preflight_policy_version",
            _require_id(self.preflight_policy_version, "runtime_config.preflight_policy_version"),
        )
        object.__setattr__(self, "route", _validate_preflight_route(self.route, "runtime_config.route"))
        object.__setattr__(
            self,
            "branch_decisions",
            _freeze_json(_validate_string_map(self.branch_decisions, "runtime_config.branch_decisions")),
        )
        object.__setattr__(
            self,
            "engine_config_ids",
            _freeze_json(_validate_string_map(self.engine_config_ids, "runtime_config.engine_config_ids")),
        )
        object.__setattr__(self, "validated_scope", _id_tuple(self.validated_scope, "runtime_config.validated_scope"))
        object.__setattr__(
            self,
            "unsupported_scope",
            _id_tuple(self.unsupported_scope, "runtime_config.unsupported_scope"),
        )
        if set(self.validated_scope) & set(self.unsupported_scope):
            raise ValidationError("runtime_config scope cannot be both validated and unsupported")
        object.__setattr__(
            self,
            "governance_decision",
            _validate_choice(
                self.governance_decision,
                _GOVERNANCE_DECISIONS,
                "runtime_config.governance_decision",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "annotation_protocol_version": self.annotation_protocol_version,
            "branch_decisions": dict(self.branch_decisions),
            "dataset_snapshot_ids": dict(self.dataset_snapshot_ids),
            "engine_config_ids": dict(self.engine_config_ids),
            "git_sha": self.git_sha,
            "governance_decision": self.governance_decision,
            "preflight_policy_id": self.preflight_policy_id,
            "preflight_policy_version": self.preflight_policy_version,
            "private_acceptance_split_id": self.private_acceptance_split_id,
            "route": self.route,
            "schema_versions": dict(self.schema_versions),
            "scoring_policy_versions": dict(self.scoring_policy_versions),
            "unsupported_scope": list(self.unsupported_scope),
            "validated_scope": list(self.validated_scope),
        }


@dataclass(frozen=True)
class ReleaseRuntimeAuditEvent:
    """Runtime event emitted when confident attribution is disabled."""

    code: str
    message: str
    expected: Any = None
    actual: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _require_id(self.code, "runtime_audit.code"))
        object.__setattr__(self, "message", _require_text(self.message, "runtime_audit.message"))
        _validate_json_value(self.expected, "runtime_audit.expected")
        _validate_json_value(self.actual, "runtime_audit.actual")
        object.__setattr__(self, "expected", _freeze_json(self.expected))
        object.__setattr__(self, "actual", _freeze_json(self.actual))

    def to_dict(self) -> dict[str, Any]:
        return {
            "actual": _thaw_json(self.actual),
            "code": self.code,
            "expected": _thaw_json(self.expected),
            "message": self.message,
        }


@dataclass(frozen=True)
class ReleaseRuntimeCheck:
    """Result of checking active runtime metadata against an immutable release."""

    status: ReleaseRuntimeCheckStatus
    confident_speaker_attribution_enabled: bool
    degraded_route: PreflightRoute | None
    audit_events: tuple[ReleaseRuntimeAuditEvent, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "status",
            _validate_choice(self.status, _RUNTIME_CHECK_STATUSES, "runtime_check.status"),
        )
        object.__setattr__(
            self,
            "confident_speaker_attribution_enabled",
            _require_bool(
                self.confident_speaker_attribution_enabled,
                "runtime_check.confident_speaker_attribution_enabled",
            ),
        )
        if self.degraded_route is not None:
            object.__setattr__(
                self,
                "degraded_route",
                _validate_preflight_route(self.degraded_route, "runtime_check.degraded_route"),
            )
        object.__setattr__(
            self,
            "audit_events",
            _tuple_of(self.audit_events, ReleaseRuntimeAuditEvent, "runtime_check.audit_events"),
        )
        if self.status == "approved" and (not self.confident_speaker_attribution_enabled or self.audit_events):
            raise ValidationError("approved runtime checks cannot include audit events or disabled attribution")
        if self.status == "degraded" and (self.confident_speaker_attribution_enabled or not self.audit_events):
            raise ValidationError("degraded runtime checks require disabled attribution and audit events")

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_events": [event.to_dict() for event in self.audit_events],
            "confident_speaker_attribution_enabled": self.confident_speaker_attribution_enabled,
            "degraded_route": self.degraded_route,
            "status": self.status,
        }


@dataclass(frozen=True)
class ReleaseCandidateRecord:
    """Content-addressed release candidate for enabling confident speaker labels."""

    release_candidate_id: str
    git_sha: str
    dataset_snapshots: tuple[dict[str, Any], ...]
    private_acceptance_split_id: str
    annotation_protocol: PrivateAnnotationProtocol
    scoring_policies: tuple[ScoringPolicyManifest, ...]
    preflight_policy: PreflightPolicy
    route_state: ReleaseRouteState
    branch_decisions: tuple[BranchAcceptanceRecord, ...]
    engine_configs: tuple[EngineConfigMetadata, ...]
    validated_scope: tuple[str, ...]
    unsupported_scope: tuple[str, ...]
    governance_decision: ReleaseGovernanceDecision
    benchmark_reports: tuple[BenchmarkReport, ...]
    golden_tests: tuple[ReleaseGoldenTestResult, ...]
    approval_status: ReleaseApprovalStatus
    must_not_have_checks: ReleaseMustNotHaveChecks
    schema_version: int = RELEASE_RECORD_SCHEMA_VERSION
    content_hash: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _validate_release_schema_version(self.schema_version))
        object.__setattr__(self, "release_candidate_id", _require_id(self.release_candidate_id, "release.candidate_id"))
        object.__setattr__(self, "git_sha", _validate_git_sha(self.git_sha, "release.git_sha"))
        snapshots = _dataset_snapshots(self.dataset_snapshots)
        object.__setattr__(self, "dataset_snapshots", snapshots)
        object.__setattr__(
            self,
            "private_acceptance_split_id",
            _require_id(self.private_acceptance_split_id, "release.private_acceptance_split_id"),
        )
        if not isinstance(self.annotation_protocol, PrivateAnnotationProtocol):
            raise ValidationError("release.annotation_protocol must be PrivateAnnotationProtocol")
        scoring_policies = _tuple_of(self.scoring_policies, ScoringPolicyManifest, "release.scoring_policies")
        if not scoring_policies:
            raise ValidationError("release.scoring_policies is required")
        _reject_duplicates(tuple(policy.policy_id for policy in scoring_policies), "release.scoring_policies.policy_id")
        object.__setattr__(self, "scoring_policies", scoring_policies)
        if not isinstance(self.preflight_policy, PreflightPolicy):
            raise ValidationError("release.preflight_policy must be PreflightPolicy")
        if not isinstance(self.route_state, ReleaseRouteState):
            raise ValidationError("release.route_state must be ReleaseRouteState")
        if self.route_state.policy_id != self.preflight_policy.policy_id:
            raise ValidationError("release.route_state policy_id must match preflight_policy")
        if self.route_state.policy_version != self.preflight_policy.version:
            raise ValidationError("release.route_state policy_version must match preflight_policy")
        branch_decisions = _tuple_of(self.branch_decisions, BranchAcceptanceRecord, "release.branch_decisions")
        if not branch_decisions:
            raise ValidationError("release.branch_decisions is required")
        _reject_duplicates(tuple(decision.branch_id for decision in branch_decisions), "release.branch_decisions.branch_id")
        object.__setattr__(self, "branch_decisions", branch_decisions)
        engine_configs = _tuple_of(self.engine_configs, EngineConfigMetadata, "release.engine_configs")
        if not engine_configs:
            raise ValidationError("release.engine_configs is required")
        _validate_engine_configs_pinned(engine_configs)
        object.__setattr__(self, "engine_configs", engine_configs)
        validated_scope = _id_tuple(self.validated_scope, "release.validated_scope")
        unsupported_scope = _id_tuple(self.unsupported_scope, "release.unsupported_scope")
        if set(validated_scope) & set(unsupported_scope):
            raise ValidationError("release scope cannot be both validated and unsupported")
        object.__setattr__(self, "validated_scope", validated_scope)
        object.__setattr__(self, "unsupported_scope", unsupported_scope)
        object.__setattr__(
            self,
            "governance_decision",
            _validate_choice(self.governance_decision, _GOVERNANCE_DECISIONS, "release.governance_decision"),
        )
        reports = _tuple_of(self.benchmark_reports, BenchmarkReport, "release.benchmark_reports")
        if not reports:
            raise ValidationError("release.benchmark_reports is required")
        _reject_duplicates(tuple(report.report_id for report in reports), "release.benchmark_reports.report_id")
        object.__setattr__(self, "benchmark_reports", reports)
        golden_tests = _tuple_of(self.golden_tests, ReleaseGoldenTestResult, "release.golden_tests")
        if not golden_tests:
            raise ValidationError("release.golden_tests is required")
        _reject_duplicates(tuple(test.test_id for test in golden_tests), "release.golden_tests.test_id")
        object.__setattr__(self, "golden_tests", golden_tests)
        object.__setattr__(
            self,
            "approval_status",
            _validate_choice(self.approval_status, _APPROVAL_STATUSES, "release.approval_status"),
        )
        if not isinstance(self.must_not_have_checks, ReleaseMustNotHaveChecks):
            raise ValidationError("release.must_not_have_checks must be ReleaseMustNotHaveChecks")
        if self.approval_status == "approved":
            _validate_approved_release(self)
        content_hash = _optional_sha256(self.content_hash, "release.content_hash")
        payload_without_hash = _release_payload_from_fields(self)
        frozen_payload = _freeze_json(payload_without_hash)
        computed_hash = _release_payload_hash(frozen_payload)
        if content_hash is not None and content_hash != computed_hash:
            raise ValidationError("release.content_hash must match immutable release payload")
        object.__setattr__(self, "_payload_without_content_hash", frozen_payload)
        object.__setattr__(self, "content_hash", computed_hash)

    @property
    def approved_for_confident_labels(self) -> bool:
        return (
            self.approval_status == "approved"
            and self.governance_decision == "approve_confident_labels"
            and self.route_state.effective_route == "confident_pipeline"
            and self.must_not_have_checks.passed
        )

    def to_dict(self) -> dict[str, Any]:
        return _release_payload(self, include_content_hash=True)


def release_expected_runtime_config(record: ReleaseCandidateRecord) -> ReleaseRuntimeConfig:
    if not isinstance(record, ReleaseCandidateRecord):
        raise ValidationError("record must be ReleaseCandidateRecord")
    payload = _release_payload(record, include_content_hash=False)
    return ReleaseRuntimeConfig(
        git_sha=payload["git_sha"],
        schema_versions={
            "benchmark_report": _single_schema_version(
                tuple(report["schema_version"] for report in payload["benchmark_reports"]),
                "release.benchmark_reports.schema_version",
            ),
            "benchmark_run_record": BENCHMARK_RUN_RECORD_SCHEMA_VERSION,
            "dataset_manifest": _single_schema_version(
                tuple(snapshot["schema_version"] for snapshot in payload["dataset_snapshots"]),
                "release.dataset_snapshots.schema_version",
            ),
            "release_record": payload["schema_version"],
        },
        dataset_snapshot_ids={
            snapshot["dataset_id"]: _release_payload_hash(snapshot) for snapshot in payload["dataset_snapshots"]
        },
        private_acceptance_split_id=payload["private_acceptance_split_id"],
        annotation_protocol_version=(
            f"{payload['annotation_protocol']['protocol_id']}@{payload['annotation_protocol']['version']}"
        ),
        scoring_policy_versions={policy["policy_id"]: policy["version"] for policy in payload["scoring_policies"]},
        preflight_policy_id=payload["preflight_policy"]["policy_id"],
        preflight_policy_version=payload["preflight_policy"]["version"],
        route=payload["route_state"]["effective_route"],
        branch_decisions={decision["branch_id"]: decision["decision"] for decision in payload["branch_decisions"]},
        engine_config_ids={
            config["adapter_id"]: _release_payload_hash(config) for config in payload["engine_configs"]
        },
        validated_scope=tuple(payload["validated_scope"]),
        unsupported_scope=tuple(payload["unsupported_scope"]),
        governance_decision=payload["governance_decision"],
    )


def check_release_runtime_config(
    record: ReleaseCandidateRecord,
    active_config: ReleaseRuntimeConfig | Mapping[str, Any],
) -> ReleaseRuntimeCheck:
    """Fail closed unless active runtime metadata exactly matches an approved release."""

    if not isinstance(record, ReleaseCandidateRecord):
        raise ValidationError("record must be ReleaseCandidateRecord")
    if isinstance(active_config, Mapping):
        try:
            active_config = release_runtime_config_from_dict(active_config)
        except ValidationError as exc:
            return ReleaseRuntimeCheck(
                status="degraded",
                confident_speaker_attribution_enabled=False,
                degraded_route="diagnostic_only",
                audit_events=(
                    ReleaseRuntimeAuditEvent(
                        code="runtime_config_invalid",
                        message="Active runtime metadata is invalid.",
                        expected="valid runtime_config payload",
                        actual=str(exc),
                    ),
                ),
            )
    if not isinstance(active_config, ReleaseRuntimeConfig):
        raise ValidationError("active_config must be ReleaseRuntimeConfig")
    try:
        expected = release_expected_runtime_config(record)
    except ValidationError as exc:
        return ReleaseRuntimeCheck(
            status="degraded",
            confident_speaker_attribution_enabled=False,
            degraded_route="diagnostic_only",
            audit_events=(
                ReleaseRuntimeAuditEvent(
                    code="release_runtime_config_invalid",
                    message="Release runtime metadata is invalid.",
                    expected="valid release runtime_config payload",
                    actual=str(exc),
                ),
            ),
        )
    events: list[ReleaseRuntimeAuditEvent] = []
    if record.approval_status != "approved":
        events.append(
            ReleaseRuntimeAuditEvent(
                code="release_not_approved",
                message="Release is not approved for confident speaker attribution.",
                expected="approved",
                actual=record.approval_status,
            )
        )
    if record.governance_decision != "approve_confident_labels":
        events.append(
            ReleaseRuntimeAuditEvent(
                code="governance_decision_not_confident",
                message="Release governance does not approve confident speaker labels.",
                expected="approve_confident_labels",
                actual=record.governance_decision,
            )
        )
    if record.route_state.effective_route != "confident_pipeline":
        events.append(
            ReleaseRuntimeAuditEvent(
                code="route_not_confident",
                message="Release route is not confident_pipeline.",
                expected="confident_pipeline",
                actual=record.route_state.effective_route,
            )
        )
    expected_payload = expected.to_dict()
    active_payload = active_config.to_dict()
    for key, expected_value in expected_payload.items():
        actual_value = active_payload.get(key)
        if actual_value != expected_value:
            events.append(
                ReleaseRuntimeAuditEvent(
                    code=f"{key}_mismatch",
                    message=f"Active runtime {key} does not match the approved release.",
                    expected=expected_value,
                    actual=actual_value,
                )
            )
    if events:
        return ReleaseRuntimeCheck(
            status="degraded",
            confident_speaker_attribution_enabled=False,
            degraded_route=(
                record.route_state.effective_route
                if record.route_state.effective_route != "confident_pipeline"
                else "diagnostic_only"
            ),
            audit_events=tuple(events),
        )
    return ReleaseRuntimeCheck(
        status="approved",
        confident_speaker_attribution_enabled=True,
        degraded_route=None,
        audit_events=(),
    )


def validate_release_runtime_output(payload: Mapping[str, Any]) -> None:
    """Reject runtime output that contains persisted identity material."""

    data = _validate_metadata(dict(payload), "runtime_output")
    _reject_forbidden_runtime_identity_fields(data, "runtime_output")


def release_record_content_hash(record: ReleaseCandidateRecord) -> str:
    if not isinstance(record, ReleaseCandidateRecord):
        raise ValidationError("record must be ReleaseCandidateRecord")
    return _release_payload_hash(_release_payload(record, include_content_hash=False))


def release_record_json_dumps(record: ReleaseCandidateRecord) -> str:
    if not isinstance(record, ReleaseCandidateRecord):
        raise ValidationError("record must be ReleaseCandidateRecord")
    return json.dumps(record.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def release_record_json_loads(text: str) -> ReleaseCandidateRecord:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"release record JSON is invalid: {exc.msg}") from exc
    return release_record_from_dict(payload)


def release_record_from_dict(payload: Mapping[str, Any]) -> ReleaseCandidateRecord:
    data = _mapping(payload, "release")
    _reject_unknown_fields(
        data,
        {
            "annotation_protocol",
            "approval_status",
            "benchmark_reports",
            "branch_decisions",
            "content_hash",
            "dataset_snapshots",
            "engine_configs",
            "git_sha",
            "golden_tests",
            "governance_decision",
            "must_not_have_checks",
            "preflight_policy",
            "private_acceptance_split_id",
            "release_candidate_id",
            "route_state",
            "schema_version",
            "scoring_policies",
            "unsupported_scope",
            "validated_scope",
        },
        "release",
    )
    return ReleaseCandidateRecord(
        schema_version=_required(data, "schema_version", "release"),
        release_candidate_id=_required(data, "release_candidate_id", "release"),
        git_sha=_required(data, "git_sha", "release"),
        dataset_snapshots=tuple(_sequence(_required(data, "dataset_snapshots", "release"), "release.dataset_snapshots")),
        private_acceptance_split_id=_required(data, "private_acceptance_split_id", "release"),
        annotation_protocol=_private_annotation_protocol_from_dict(_required(data, "annotation_protocol", "release")),
        scoring_policies=tuple(
            scoring_policy_from_dict(item)
            for item in _sequence(_required(data, "scoring_policies", "release"), "release.scoring_policies")
        ),
        preflight_policy=_preflight_policy_from_dict(_required(data, "preflight_policy", "release")),
        route_state=release_route_state_from_dict(_required(data, "route_state", "release")),
        branch_decisions=tuple(
            _branch_acceptance_from_dict(item)
            for item in _sequence(_required(data, "branch_decisions", "release"), "release.branch_decisions")
        ),
        engine_configs=tuple(
            _engine_config_from_dict(item)
            for item in _sequence(_required(data, "engine_configs", "release"), "release.engine_configs")
        ),
        validated_scope=tuple(_sequence(_required(data, "validated_scope", "release"), "release.validated_scope")),
        unsupported_scope=tuple(_sequence(_required(data, "unsupported_scope", "release"), "release.unsupported_scope")),
        governance_decision=_required(data, "governance_decision", "release"),
        benchmark_reports=tuple(
            benchmark_report_from_dict(item)
            for item in _sequence(_required(data, "benchmark_reports", "release"), "release.benchmark_reports")
        ),
        golden_tests=tuple(
            release_golden_test_result_from_dict(item)
            for item in _sequence(_required(data, "golden_tests", "release"), "release.golden_tests")
        ),
        approval_status=_required(data, "approval_status", "release"),
        must_not_have_checks=release_must_not_have_checks_from_dict(
            _required(data, "must_not_have_checks", "release")
        ),
        content_hash=data.get("content_hash"),
    )


def release_runtime_config_from_dict(payload: Mapping[str, Any]) -> ReleaseRuntimeConfig:
    data = _mapping(payload, "runtime_config")
    _reject_unknown_fields(
        data,
        {
            "annotation_protocol_version",
            "branch_decisions",
            "dataset_snapshot_ids",
            "engine_config_ids",
            "git_sha",
            "governance_decision",
            "preflight_policy_id",
            "preflight_policy_version",
            "private_acceptance_split_id",
            "route",
            "schema_versions",
            "scoring_policy_versions",
            "unsupported_scope",
            "validated_scope",
        },
        "runtime_config",
    )
    return ReleaseRuntimeConfig(
        git_sha=_required(data, "git_sha", "runtime_config"),
        schema_versions=_required(data, "schema_versions", "runtime_config"),
        dataset_snapshot_ids=_required(data, "dataset_snapshot_ids", "runtime_config"),
        private_acceptance_split_id=_required(data, "private_acceptance_split_id", "runtime_config"),
        annotation_protocol_version=_required(data, "annotation_protocol_version", "runtime_config"),
        scoring_policy_versions=_required(data, "scoring_policy_versions", "runtime_config"),
        preflight_policy_id=_required(data, "preflight_policy_id", "runtime_config"),
        preflight_policy_version=_required(data, "preflight_policy_version", "runtime_config"),
        route=_required(data, "route", "runtime_config"),
        branch_decisions=_required(data, "branch_decisions", "runtime_config"),
        engine_config_ids=_required(data, "engine_config_ids", "runtime_config"),
        validated_scope=tuple(_sequence(_required(data, "validated_scope", "runtime_config"), "runtime_config.validated_scope")),
        unsupported_scope=tuple(
            _sequence(_required(data, "unsupported_scope", "runtime_config"), "runtime_config.unsupported_scope")
        ),
        governance_decision=_required(data, "governance_decision", "runtime_config"),
    )


def release_route_state_from_dict(payload: Mapping[str, Any]) -> ReleaseRouteState:
    data = _mapping(payload, "route_state")
    _reject_unknown_fields(
        data,
        {
            "effective_route",
            "manual_override_applied",
            "policy_id",
            "policy_version",
            "route",
            "validated_launch_scope_version",
        },
        "route_state",
    )
    return ReleaseRouteState(
        policy_id=_required(data, "policy_id", "route_state"),
        policy_version=_required(data, "policy_version", "route_state"),
        route=_required(data, "route", "route_state"),
        effective_route=_required(data, "effective_route", "route_state"),
        validated_launch_scope_version=_required(data, "validated_launch_scope_version", "route_state"),
        manual_override_applied=_required(data, "manual_override_applied", "route_state"),
    )


def release_golden_test_result_from_dict(payload: Mapping[str, Any]) -> ReleaseGoldenTestResult:
    data = _mapping(payload, "golden_test")
    _reject_unknown_fields(
        data,
        {"artifact_hash", "metadata", "passed", "reason", "status", "test_id"},
        "golden_test",
    )
    result = ReleaseGoldenTestResult(
        test_id=_required(data, "test_id", "golden_test"),
        status=_required(data, "status", "golden_test"),
        artifact_hash=data.get("artifact_hash"),
        reason=data.get("reason"),
        metadata=data.get("metadata", {}),
    )
    if data.get("passed") != result.passed:
        raise ValidationError("golden_test.passed must match status")
    return result


def release_must_not_have_checks_from_dict(payload: Mapping[str, Any]) -> ReleaseMustNotHaveChecks:
    data = _mapping(payload, "must_not_have")
    _reject_unknown_fields(
        data,
        {
            "evidence",
            "no_cross_call_speaker_ids",
            "no_reference_speaker_ids_in_runtime_output",
            "no_retained_voice_profiles_or_embeddings",
            "no_tuned_on_holdout_result",
            "no_unpinned_model_or_provider_config",
            "passed",
        },
        "must_not_have",
    )
    checks = ReleaseMustNotHaveChecks(
        no_cross_call_speaker_ids=_required(data, "no_cross_call_speaker_ids", "must_not_have"),
        no_retained_voice_profiles_or_embeddings=_required(
            data,
            "no_retained_voice_profiles_or_embeddings",
            "must_not_have",
        ),
        no_reference_speaker_ids_in_runtime_output=_required(
            data,
            "no_reference_speaker_ids_in_runtime_output",
            "must_not_have",
        ),
        no_unpinned_model_or_provider_config=_required(
            data,
            "no_unpinned_model_or_provider_config",
            "must_not_have",
        ),
        no_tuned_on_holdout_result=_required(data, "no_tuned_on_holdout_result", "must_not_have"),
        evidence=data.get("evidence", {}),
    )
    if data.get("passed") != checks.passed:
        raise ValidationError("must_not_have.passed must match checks")
    return checks


def read_release_record_json(path: str | Path) -> ReleaseCandidateRecord:
    return release_record_json_loads(Path(path).read_text(encoding="utf-8"))


def write_release_record_json(path: str | Path, record: ReleaseCandidateRecord) -> None:
    Path(path).write_text(release_record_json_dumps(record), encoding="utf-8", newline="\n")


def _release_payload(record: ReleaseCandidateRecord, *, include_content_hash: bool) -> dict[str, Any]:
    payload_without_hash = getattr(record, "_payload_without_content_hash", None)
    if payload_without_hash is None:
        payload = _release_payload_from_fields(record)
    else:
        payload = _thaw_json(payload_without_hash)
    if include_content_hash:
        payload["content_hash"] = record.content_hash
    return payload


def _release_payload_from_fields(record: ReleaseCandidateRecord) -> dict[str, Any]:
    payload = {
        "annotation_protocol": record.annotation_protocol.to_dict(),
        "approval_status": record.approval_status,
        "benchmark_reports": [report.to_dict() for report in record.benchmark_reports],
        "branch_decisions": [decision.to_dict() for decision in record.branch_decisions],
        "dataset_snapshots": [_thaw_json(snapshot) for snapshot in record.dataset_snapshots],
        "engine_configs": [config.to_dict() for config in record.engine_configs],
        "git_sha": record.git_sha,
        "golden_tests": [test.to_dict() for test in record.golden_tests],
        "governance_decision": record.governance_decision,
        "must_not_have_checks": record.must_not_have_checks.to_dict(),
        "preflight_policy": record.preflight_policy.to_dict(),
        "private_acceptance_split_id": record.private_acceptance_split_id,
        "release_candidate_id": record.release_candidate_id,
        "route_state": record.route_state.to_dict(),
        "schema_version": record.schema_version,
        "scoring_policies": [policy.to_dict() for policy in record.scoring_policies],
        "unsupported_scope": list(record.unsupported_scope),
        "validated_scope": list(record.validated_scope),
    }
    return payload


def _validate_approved_release(record: ReleaseCandidateRecord) -> None:
    if record.governance_decision != "approve_confident_labels":
        raise ValidationError("approved releases require approve_confident_labels governance")
    if record.route_state.effective_route != "confident_pipeline":
        raise ValidationError("approved releases require confident_pipeline route")
    if not record.must_not_have_checks.passed:
        raise ValidationError("approved releases require all must-not-have checks to pass")
    if any(not report.passed for report in record.benchmark_reports):
        raise ValidationError("approved releases cannot include failed benchmark reports")
    if any(not test.passed for test in record.golden_tests):
        raise ValidationError("approved releases cannot include failed golden tests")
    if any(not decision.enforced_gates_passed for decision in record.branch_decisions):
        raise ValidationError("approved releases cannot include failed branch gates")


def _dataset_snapshots(values: object) -> tuple[Mapping[str, Any], ...]:
    snapshots = tuple(
        _freeze_json(dataset_manifest_from_dict(item).to_dict())
        for item in _sequence(values, "release.dataset_snapshots")
    )
    if not snapshots:
        raise ValidationError("release.dataset_snapshots is required")
    _reject_duplicates(tuple(snapshot["dataset_id"] for snapshot in snapshots), "release.dataset_snapshots.dataset_id")
    return snapshots


def _validate_engine_configs_pinned(configs: tuple[EngineConfigMetadata, ...]) -> None:
    _reject_duplicates(tuple(config.adapter_id for config in configs), "release.engine_configs.adapter_id")
    for config in configs:
        _reject_forbidden_runtime_identity_fields(config.parameters, "release.engine_config.parameters")
        if config.model_version is None and config.config_id is None:
            raise ValidationError("release engine configs must pin model_version or config_id")
        if config.provider == "self-hosted":
            model = config.parameters.get("model_governance")
            if not isinstance(model, Mapping):
                raise ValidationError("self-hosted release configs must include model_governance")
            package_versions_payload = model.get("package_versions")
            if package_versions_payload is None:
                raise ValidationError("self-hosted release configs must pin package_versions")
            package_versions = _validate_string_map(
                package_versions_payload,
                "release.engine_config.model_governance.package_versions",
            )
            if not package_versions:
                raise ValidationError("self-hosted release configs must pin package_versions")
        hosted = config.parameters.get("hosted_provider_governance")
        if hosted is None and config.provider in _HOSTED_PROVIDER_CONFIG_PROVIDERS:
            raise ValidationError("hosted provider release configs must include hosted_provider_governance")
        if isinstance(hosted, Mapping):
            model_version = _optional_text(
                hosted.get("model_version"),
                "release.engine_config.hosted_provider_governance.model_version",
            )
            version_pinning = _optional_text(
                hosted.get("version_pinning"),
                "release.engine_config.hosted_provider_governance.version_pinning",
            )
            if model_version is None or version_pinning is None:
                raise ValidationError("hosted provider release configs must pin model_version and version_pinning")
        elif hosted is not None:
            raise ValidationError("hosted provider release governance must be an object")


def _release_payload_hash(payload: Mapping[str, Any]) -> str:
    text = json.dumps(_thaw_json(payload), ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _single_schema_version(values: tuple[object, ...], field_name: str) -> int:
    versions = tuple(_require_int(value, field_name) for value in values)
    if not versions:
        raise ValidationError(f"{field_name} is required")
    unique_versions = set(versions)
    if len(unique_versions) != 1:
        raise ValidationError(f"{field_name} must be consistent across release artifacts")
    return versions[0]


def _preflight_policy_from_dict(payload: object) -> PreflightPolicy:
    data = _mapping(payload, "preflight_policy")
    _reject_unknown_fields(
        data,
        {
            "confident_capture_modes",
            "frozen_git_sha",
            "max_confident_clipping_estimate",
            "max_confident_rough_overlap_estimate",
            "max_confident_speaker_count_hint",
            "max_duration_ms",
            "min_confident_speaker_count_hint",
            "min_confident_speech_ratio",
            "min_duration_ms",
            "min_sample_rate_hz",
            "policy_id",
            "require_speaker_count_hint_for_confident",
            "supported_capture_modes",
            "supported_channel_counts",
            "supported_codecs",
            "supported_locales",
            "supported_sources",
            "tuned_on_splits",
            "validated_on_splits",
            "version",
        },
        "preflight_policy",
    )
    return PreflightPolicy(
        policy_id=_required(data, "policy_id", "preflight_policy"),
        version=_required(data, "version", "preflight_policy"),
        frozen_git_sha=_required(data, "frozen_git_sha", "preflight_policy"),
        tuned_on_splits=tuple(_sequence(_required(data, "tuned_on_splits", "preflight_policy"), "preflight_policy.tuned_on_splits")),
        validated_on_splits=tuple(
            _sequence(_required(data, "validated_on_splits", "preflight_policy"), "preflight_policy.validated_on_splits")
        ),
        supported_locales=tuple(
            _sequence(_required(data, "supported_locales", "preflight_policy"), "preflight_policy.supported_locales")
        ),
        supported_sources=tuple(
            _sequence(_required(data, "supported_sources", "preflight_policy"), "preflight_policy.supported_sources")
        ),
        supported_capture_modes=tuple(
            _sequence(
                _required(data, "supported_capture_modes", "preflight_policy"),
                "preflight_policy.supported_capture_modes",
            )
        ),
        confident_capture_modes=tuple(
            _sequence(
                _required(data, "confident_capture_modes", "preflight_policy"),
                "preflight_policy.confident_capture_modes",
            )
        ),
        supported_channel_counts=tuple(
            _sequence(
                _required(data, "supported_channel_counts", "preflight_policy"),
                "preflight_policy.supported_channel_counts",
            )
        ),
        supported_codecs=tuple(
            _sequence(_required(data, "supported_codecs", "preflight_policy"), "preflight_policy.supported_codecs")
        ),
        min_duration_ms=_required(data, "min_duration_ms", "preflight_policy"),
        max_duration_ms=_required(data, "max_duration_ms", "preflight_policy"),
        min_sample_rate_hz=_required(data, "min_sample_rate_hz", "preflight_policy"),
        max_confident_clipping_estimate=_required(data, "max_confident_clipping_estimate", "preflight_policy"),
        min_confident_speech_ratio=_required(data, "min_confident_speech_ratio", "preflight_policy"),
        max_confident_rough_overlap_estimate=_required(data, "max_confident_rough_overlap_estimate", "preflight_policy"),
        min_confident_speaker_count_hint=_required(data, "min_confident_speaker_count_hint", "preflight_policy"),
        max_confident_speaker_count_hint=_required(data, "max_confident_speaker_count_hint", "preflight_policy"),
        require_speaker_count_hint_for_confident=_required(
            data,
            "require_speaker_count_hint_for_confident",
            "preflight_policy",
        ),
    )


def _private_annotation_protocol_from_dict(payload: object) -> PrivateAnnotationProtocol:
    data = _mapping(payload, "private_protocol")
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
    _reject_forbidden_runtime_identity_fields(data, "private_protocol")
    return PrivateAnnotationProtocol(
        protocol_id=_required(data, "protocol_id", "private_protocol"),
        version=_required(data, "version", "private_protocol"),
        transcript_normalization=_required(data, "transcript_normalization", "private_protocol"),
        speaker_span_rules=tuple(
            _sequence(_required(data, "speaker_span_rules", "private_protocol"), "private_protocol.speaker_span_rules")
        ),
        overlap_rules=tuple(_sequence(_required(data, "overlap_rules", "private_protocol"), "private_protocol.overlap_rules")),
        unintelligible_no_score_rules=tuple(
            _sequence(
                _required(data, "unintelligible_no_score_rules", "private_protocol"),
                "private_protocol.unintelligible_no_score_rules",
            )
        ),
        critical_span_label_rules=tuple(
            _sequence(
                _required(data, "critical_span_label_rules", "private_protocol"),
                "private_protocol.critical_span_label_rules",
            )
        ),
        reference_speaker_id_policy=data.get("reference_speaker_id_policy", "candidate_invisible"),
    )


def _engine_config_from_dict(payload: object) -> EngineConfigMetadata:
    data = _mapping(payload, "engine_config")
    _reject_unknown_fields(
        data,
        {"adapter_id", "config_id", "model_name", "model_version", "parameters", "provider"},
        "engine_config",
    )
    return EngineConfigMetadata(
        adapter_id=_required(data, "adapter_id", "engine_config"),
        provider=_required(data, "provider", "engine_config"),
        model_name=_required(data, "model_name", "engine_config"),
        model_version=data.get("model_version"),
        config_id=data.get("config_id"),
        parameters=data.get("parameters", {}),
    )


def _branch_acceptance_from_dict(payload: object) -> BranchAcceptanceRecord:
    data = _mapping(payload, "branch_acceptance")
    _reject_unknown_fields(
        data,
        {
            "branch_id",
            "cost_delta",
            "decision",
            "enforced_gates",
            "false_confidence_delta",
            "governance_delta",
            "job_failure_delta",
            "latency_delta_ms",
            "non_enforced_fields",
            "private_coverage_ready",
            "quality_delta",
            "retry_delta",
            "review_burden_delta",
        },
        "branch_acceptance",
    )
    enforced_gates = _mapping(_required(data, "enforced_gates", "branch_acceptance"), "branch_acceptance.enforced_gates")
    _reject_unknown_fields(
        enforced_gates,
        {"false_confidence", "quality", "review_burden"},
        "branch_acceptance.enforced_gates",
    )
    record = BranchAcceptanceRecord(
        branch_id=_required(data, "branch_id", "branch_acceptance"),
        decision=_required(data, "decision", "branch_acceptance"),
        quality_delta=_required(data, "quality_delta", "branch_acceptance"),
        false_confidence_delta=_required(data, "false_confidence_delta", "branch_acceptance"),
        review_burden_delta=_required(data, "review_burden_delta", "branch_acceptance"),
        quality_gate_passed=_required(enforced_gates, "quality", "branch_acceptance.enforced_gates"),
        false_confidence_gate_passed=_required(enforced_gates, "false_confidence", "branch_acceptance.enforced_gates"),
        review_burden_gate_passed=_required(enforced_gates, "review_burden", "branch_acceptance.enforced_gates"),
        latency_delta_ms=data.get("latency_delta_ms"),
        cost_delta=data.get("cost_delta"),
        job_failure_delta=data.get("job_failure_delta"),
        retry_delta=data.get("retry_delta"),
        governance_delta=data.get("governance_delta", {}),
        private_coverage_ready=_required(data, "private_coverage_ready", "branch_acceptance"),
    )
    expected_non_enforced = record.to_dict()["non_enforced_fields"]
    if data.get("non_enforced_fields", expected_non_enforced) != expected_non_enforced:
        raise ValidationError("branch_acceptance.non_enforced_fields must match branch fields")
    return record


def _validate_release_schema_version(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError("release.schema_version must be an integer")
    if value not in SUPPORTED_RELEASE_RECORD_SCHEMA_VERSIONS:
        raise ValidationError(f"release.schema_version is not supported: {value}")
    return value


def _validate_preflight_route(value: object, field_name: str) -> PreflightRoute:
    route = _require_id(value, field_name)
    if route not in ALLOWED_PREFLIGHT_ROUTES:
        raise ValidationError(f"{field_name} is not supported: {route}")
    return route  # type: ignore[return-value]


def _validate_git_sha(value: object, field_name: str) -> str:
    sha = _require_id(value, field_name)
    if len(sha) != 40 or any(char not in _HEX_DIGITS for char in sha):
        raise ValidationError(f"{field_name} must be a frozen 40-character git SHA")
    return sha


def _optional_sha256(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    text = _require_id(value, field_name)
    if len(text) != 64 or any(char not in _HEX_DIGITS for char in text):
        raise ValidationError(f"{field_name} must be a lowercase sha256 hex digest")
    return text


def _validate_choice(value: object, choices: frozenset[str], field_name: str) -> str:
    text = _require_id(value, field_name)
    if text not in choices:
        raise ValidationError(f"{field_name} is not supported: {text}")
    return text


def _id_tuple(values: object, field_name: str) -> tuple[str, ...]:
    result = tuple(_require_id(item, field_name) for item in _sequence(values, field_name))
    _reject_duplicates(result, field_name)
    return result


def _validate_string_map(value: object, field_name: str) -> dict[str, str]:
    data = _mapping(value, field_name)
    return {
        _require_id(key, f"{field_name}.key"): _require_id(item, f"{field_name}.{key}")
        for key, item in data.items()
    }


def _validate_int_map(value: object, field_name: str) -> dict[str, int]:
    data = _mapping(value, field_name)
    return {_require_id(key, f"{field_name}.key"): _require_int(item, f"{field_name}.{key}") for key, item in data.items()}


def _validate_metadata(value: object, field_name: str) -> dict[str, Any]:
    data = _mapping(value, field_name)
    _validate_json_value(data, field_name)
    return data


def _reject_forbidden_runtime_identity_fields(value: object, field_name: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = _require_id(key, f"{field_name}.key")
            if key_text in _FORBIDDEN_RUNTIME_IDENTITY_KEYS:
                raise ValidationError(f"{field_name}.{key_text} must not persist cross-call identity material")
            _reject_forbidden_runtime_identity_fields(item, f"{field_name}.{key_text}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_forbidden_runtime_identity_fields(item, f"{field_name}[{index}]")


def _validate_json_value(value: object, field_name: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_id(key, f"{field_name}.key")
            _validate_json_value(item, f"{field_name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{field_name}[{index}]")
    elif isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError(f"{field_name} must be finite")
    else:
        raise ValidationError(f"{field_name} must be JSON-serializable")


def _freeze_json(value: object) -> object:
    _validate_json_value(value, "release.payload")
    if isinstance(value, Mapping):
        return MappingProxyType(
            {_require_id(key, "release.payload.key"): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    if isinstance(value, list):
        return [_thaw_json(item) for item in value]
    return value


def _tuple_of(values: object, item_type: type[Any], field_name: str) -> tuple[Any, ...]:
    result = _sequence(values, field_name)
    for index, item in enumerate(result):
        if not isinstance(item, item_type):
            raise ValidationError(f"{field_name}[{index}] must be {item_type.__name__}")
    return result


def _sequence(value: object, field_name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValidationError(f"{field_name} must be an array")
    return tuple(value)


def _mapping(payload: object, field_name: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValidationError(f"{field_name} must be an object")
    return dict(payload)


def _required(data: Mapping[str, Any], key: str, field_name: str) -> Any:
    try:
        return data[key]
    except KeyError as exc:
        raise ValidationError(f"{field_name}.{key} is required") from exc


def _reject_unknown_fields(data: Mapping[str, Any], allowed_fields: set[str], field_name: str) -> None:
    unknown = sorted(set(data) - allowed_fields)
    if unknown:
        raise ValidationError(f"{field_name} has unsupported fields: {', '.join(unknown)}")


def _reject_duplicates(values: tuple[Any, ...], field_name: str) -> None:
    seen: set[Any] = set()
    for value in values:
        if value in seen:
            raise ValidationError(f"{field_name} contains duplicate value: {value}")
        seen.add(value)


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


def _optional_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_text(value, field_name)


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValidationError(f"{field_name} must be a boolean")
    return value


def _require_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{field_name} must be an integer")
    return value
