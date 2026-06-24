"""Privacy-preserving post-release diarization monitoring records."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from keyframe.diarization.models import ValidationError
from keyframe.diarization.preflight import ALLOWED_PREFLIGHT_ROUTES, PreflightRoute


MONITORING_RECORD_SCHEMA_VERSION = 1
SUPPORTED_MONITORING_RECORD_SCHEMA_VERSIONS = frozenset({1})

MonitoringRetentionClass = Literal["ephemeral", "diagnostic_30d", "private_candidate"]
MonitoringPromotionStep = Literal[
    "grant_consent",
    "verify_access",
    "assign_split",
    "mark_annotated",
    "mark_adjudicated",
]

_RETENTION_CLASSES = frozenset({"ephemeral", "diagnostic_30d", "private_candidate"})
_DEGRADED_ROUTES = frozenset({"diagnostic_only", "needs_review", "unsupported"})
_TIME_BASES = frozenset({"canonical_ms", "chunk_relative_ms", "sample_index", "frame_index"})
_FORBIDDEN_MONITORING_KEYS = frozenset(
    {
        "audio_bytes",
        "audio_fingerprint",
        "audio_fingerprints",
        "audio_sha256",
        "account_id",
        "account_ids",
        "benchmark_result",
        "canonical_audio_id",
        "canonical_audio_sha256",
        "corpus_identity",
        "corpus_recording_id",
        "corpus_speaker_id",
        "corpus_speaker_ids",
        "cross_call_speaker_id",
        "cross_recording_identity",
        "cross_recording_speaker_id",
        "cross_session_speaker_key",
        "cross_session_speaker_keys",
        "customer_id",
        "customer_ids",
        "display_label",
        "display_labels",
        "email",
        "emails",
        "embedding",
        "embeddings",
        "evaluator_speaker_map",
        "global_identity",
        "global_speaker_id",
        "gold_label",
        "gold_labels",
        "identity_profile",
        "local_audio_sha256",
        "oracle",
        "oracle_metadata",
        "original_audio_id",
        "original_audio_sha256",
        "participant_id",
        "participant_ids",
        "participant_email",
        "participant_emails",
        "participant_name",
        "participant_names",
        "profile_id",
        "profile_ids",
        "raw_audio",
        "raw_audio_bytes",
        "raw_audio_path",
        "reference_label",
        "reference_labels",
        "reference_speaker_id",
        "reference_speaker_ids",
        "retained_voice_profile",
        "source_speaker_id",
        "source_speaker_ids",
        "speaker_embedding",
        "speaker_embeddings",
        "speaker_id",
        "speaker_ids",
        "speaker_profile",
        "speaker_ref",
        "speaker_refs",
        "transform_config_hash",
        "user_id",
        "user_ids",
        "voice_embedding",
        "voice_embeddings",
        "voice_fingerprint",
        "voice_fingerprints",
        "voice_profile",
        "voice_profiles",
    }
)
_FORBIDDEN_MONITORING_KEY_ALIASES = frozenset(
    "".join(char for char in key.casefold() if char.isalnum()) for key in _FORBIDDEN_MONITORING_KEYS
)
_FORBIDDEN_MONITORING_KEY_FRAGMENTS = frozenset(
    {
        "account",
        "contact",
        "customer",
        "email",
        "fingerprint",
        "identifier",
        "identity",
        "participant",
        "profile",
        "speakerkey",
        "userid",
    }
)
_EDIT_OPERATION_TYPES = frozenset(
    {
        "assign_span",
        "mark_overlap",
        "mark_uncertain",
        "merge_speakers",
        "rename_label",
        "split_speaker",
    }
)
_REQUIRED_TIMELINE_FIELDS = frozenset(
    {"duration_ms", "sample_rate_hz", "time_basis", "timeline_id", "transform_chain_id"}
)
_ALLOWED_TIMELINE_FIELDS = _REQUIRED_TIMELINE_FIELDS | frozenset({"channel_ids"})


@dataclass(frozen=True)
class MonitoringPromotionState:
    """Promotion metadata for a telemetry record without automatic fixture inclusion."""

    consent_granted: bool = False
    access_checked: bool = False
    split_id: str | None = None
    annotation_protocol_version: str | None = None
    annotated: bool = False
    adjudicated: bool = False
    adjudication_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "consent_granted", _require_bool(self.consent_granted, "promotion.consent_granted"))
        object.__setattr__(self, "access_checked", _require_bool(self.access_checked, "promotion.access_checked"))
        object.__setattr__(self, "split_id", _optional_id(self.split_id, "promotion.split_id"))
        object.__setattr__(
            self,
            "annotation_protocol_version",
            _optional_id(self.annotation_protocol_version, "promotion.annotation_protocol_version"),
        )
        object.__setattr__(self, "annotated", _require_bool(self.annotated, "promotion.annotated"))
        object.__setattr__(self, "adjudicated", _require_bool(self.adjudicated, "promotion.adjudicated"))
        object.__setattr__(self, "adjudication_id", _optional_id(self.adjudication_id, "promotion.adjudication_id"))
        if self.split_id is not None and not (self.consent_granted and self.access_checked):
            raise ValidationError("promotion split assignment requires consent and access check")
        if self.annotation_protocol_version is not None and not self.annotated:
            raise ValidationError("promotion annotation_protocol_version requires annotated")
        if self.annotated and self.split_id is None:
            raise ValidationError("promotion annotation requires split assignment")
        if self.annotated and self.annotation_protocol_version is None:
            raise ValidationError("promotion annotation requires annotation_protocol_version")
        if self.adjudicated and not self.annotated:
            raise ValidationError("promotion adjudication requires annotation")
        if self.adjudication_id is not None and not self.adjudicated:
            raise ValidationError("promotion adjudication_id requires adjudicated")
        if self.adjudicated and self.adjudication_id is None:
            raise ValidationError("promotion adjudication requires adjudication_id")

    @property
    def eligible_for_private_fixture(self) -> bool:
        return (
            self.consent_granted
            and self.access_checked
            and self.split_id is not None
            and self.annotated
            and self.annotation_protocol_version is not None
            and self.adjudicated
            and self.adjudication_id is not None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "access_checked": self.access_checked,
            "adjudicated": self.adjudicated,
            "adjudication_id": self.adjudication_id,
            "annotated": self.annotated,
            "annotation_protocol_version": self.annotation_protocol_version,
            "consent_granted": self.consent_granted,
            "eligible_for_private_fixture": self.eligible_for_private_fixture,
            "split_id": self.split_id,
        }


@dataclass(frozen=True)
class MonitoringRecord:
    """Normalized post-release telemetry without cross-session identity material."""

    monitoring_record_id: str
    release_candidate_id: str
    monitoring_policy_version: str
    release_record_schema_version: int
    engine_config_versions: dict[str, str]
    preflight_policy_id: str
    preflight_policy_version: str
    route: PreflightRoute
    confident_speaker_attribution_enabled: bool
    canonical_timeline_metadata: dict[str, Any]
    edit_operation_counts: dict[str, int]
    retention_class: MonitoringRetentionClass
    expires_at: str
    degraded_route: PreflightRoute | None = None
    review_time_ms: int | None = None
    promotion_state: MonitoringPromotionState = field(default_factory=MonitoringPromotionState)
    schema_version: int = MONITORING_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _validate_schema_version(self.schema_version))
        object.__setattr__(self, "monitoring_record_id", _require_id(self.monitoring_record_id, "monitoring.record_id"))
        object.__setattr__(self, "release_candidate_id", _require_id(self.release_candidate_id, "monitoring.release_candidate_id"))
        object.__setattr__(
            self,
            "monitoring_policy_version",
            _require_id(self.monitoring_policy_version, "monitoring.policy_version"),
        )
        object.__setattr__(
            self,
            "release_record_schema_version",
            _positive_int(self.release_record_schema_version, "monitoring.release_record_schema_version"),
        )
        engine_versions = _validate_string_map(
            self.engine_config_versions,
            "monitoring.engine_config_versions",
            reject_sensitive_keys=True,
        )
        if not engine_versions:
            raise ValidationError("monitoring.engine_config_versions is required")
        object.__setattr__(self, "engine_config_versions", MappingProxyType(engine_versions))
        object.__setattr__(self, "preflight_policy_id", _require_id(self.preflight_policy_id, "monitoring.preflight_policy_id"))
        object.__setattr__(
            self,
            "preflight_policy_version",
            _require_id(self.preflight_policy_version, "monitoring.preflight_policy_version"),
        )
        object.__setattr__(self, "route", _validate_route(self.route, "monitoring.route"))
        object.__setattr__(
            self,
            "confident_speaker_attribution_enabled",
            _require_bool(
                self.confident_speaker_attribution_enabled,
                "monitoring.confident_speaker_attribution_enabled",
            ),
        )
        if self.route == "confident_pipeline" and not self.confident_speaker_attribution_enabled:
            raise ValidationError("confident_pipeline monitoring records require confident_speaker_attribution_enabled")
        if self.route != "confident_pipeline" and self.confident_speaker_attribution_enabled:
            raise ValidationError("degraded monitoring records cannot enable confident_speaker_attribution")
        if self.degraded_route is not None:
            degraded_route = _validate_route(self.degraded_route, "monitoring.degraded_route")
            if degraded_route not in _DEGRADED_ROUTES:
                raise ValidationError("monitoring.degraded_route must be diagnostic_only, needs_review, or unsupported")
            object.__setattr__(self, "degraded_route", degraded_route)
        if self.confident_speaker_attribution_enabled and self.degraded_route is not None:
            raise ValidationError("confident monitoring records cannot include degraded_route")
        if not self.confident_speaker_attribution_enabled and self.degraded_route is None:
            raise ValidationError("degraded monitoring records require degraded_route")
        if self.degraded_route is not None and self.degraded_route != self.route:
            raise ValidationError("monitoring.degraded_route must match route for degraded records")
        timeline = _monitoring_safe_metadata(self.canonical_timeline_metadata, "monitoring.timeline")
        missing_timeline_fields = _REQUIRED_TIMELINE_FIELDS - set(timeline)
        if missing_timeline_fields:
            raise ValidationError(
                "monitoring.timeline missing required fields: " + ", ".join(sorted(missing_timeline_fields))
            )
        _validate_required_timeline_metadata(timeline)
        object.__setattr__(self, "canonical_timeline_metadata", timeline)
        object.__setattr__(
            self,
            "edit_operation_counts",
            MappingProxyType(
                _validate_non_negative_int_map(
                    self.edit_operation_counts,
                    "monitoring.edit_operation_counts",
                    allowed_keys=_EDIT_OPERATION_TYPES,
                    reject_sensitive_keys=True,
                )
            ),
        )
        if self.review_time_ms is not None:
            object.__setattr__(self, "review_time_ms", _non_negative_int(self.review_time_ms, "monitoring.review_time_ms"))
        retention_class = _require_id(self.retention_class, "monitoring.retention_class")
        if retention_class not in _RETENTION_CLASSES:
            raise ValidationError(f"monitoring.retention_class is not supported: {retention_class}")
        object.__setattr__(self, "retention_class", retention_class)
        _parse_timestamp(self.expires_at, "monitoring.expires_at")
        object.__setattr__(self, "expires_at", _require_text(self.expires_at, "monitoring.expires_at"))
        if not isinstance(self.promotion_state, MonitoringPromotionState):
            raise ValidationError("monitoring.promotion_state must be MonitoringPromotionState")

    def is_expired(self, as_of: str | datetime) -> bool:
        return _parse_timestamp(self.expires_at, "monitoring.expires_at") <= _parse_timestamp(as_of, "as_of")

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_timeline_metadata": _thaw_json(self.canonical_timeline_metadata),
            "confident_speaker_attribution_enabled": self.confident_speaker_attribution_enabled,
            "degraded_route": self.degraded_route,
            "edit_operation_counts": dict(sorted(self.edit_operation_counts.items())),
            "engine_config_versions": dict(sorted(self.engine_config_versions.items())),
            "expires_at": self.expires_at,
            "monitoring_policy_version": self.monitoring_policy_version,
            "monitoring_record_id": self.monitoring_record_id,
            "preflight_policy_id": self.preflight_policy_id,
            "preflight_policy_version": self.preflight_policy_version,
            "promotion_state": self.promotion_state.to_dict(),
            "release_candidate_id": self.release_candidate_id,
            "release_record_schema_version": self.release_record_schema_version,
            "retention_class": self.retention_class,
            "review_time_ms": self.review_time_ms,
            "route": self.route,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True)
class MonitoringAggregate:
    """Aggregate monitoring summary that redacts expired session-local identifiers."""

    generated_at: str
    total_records: int
    active_record_count: int
    expired_record_count: int
    route_counts: dict[str, int]
    degraded_record_count: int
    edit_operation_totals: dict[str, int]
    review_time_ms_total: int
    active_monitoring_record_ids: tuple[str, ...] = ()
    active_timeline_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _parse_timestamp(self.generated_at, "monitoring_aggregate.generated_at")
        object.__setattr__(self, "total_records", _non_negative_int(self.total_records, "monitoring_aggregate.total_records"))
        object.__setattr__(
            self,
            "active_record_count",
            _non_negative_int(self.active_record_count, "monitoring_aggregate.active_record_count"),
        )
        object.__setattr__(
            self,
            "expired_record_count",
            _non_negative_int(self.expired_record_count, "monitoring_aggregate.expired_record_count"),
        )
        if self.total_records != self.active_record_count + self.expired_record_count:
            raise ValidationError("monitoring_aggregate record counts must add to total_records")
        object.__setattr__(
            self,
            "route_counts",
            MappingProxyType(
                _validate_non_negative_int_map(
                    self.route_counts,
                    "monitoring_aggregate.route_counts",
                    allowed_keys=ALLOWED_PREFLIGHT_ROUTES,
                )
            ),
        )
        object.__setattr__(
            self,
            "degraded_record_count",
            _non_negative_int(self.degraded_record_count, "monitoring_aggregate.degraded_record_count"),
        )
        if sum(self.route_counts.values()) != self.total_records:
            raise ValidationError("monitoring_aggregate.route_counts must sum to total_records")
        degraded_route_count = sum(
            count for route, count in self.route_counts.items() if route != "confident_pipeline"
        )
        if self.degraded_record_count != degraded_route_count:
            raise ValidationError("monitoring_aggregate.degraded_record_count must match degraded route counts")
        object.__setattr__(
            self,
            "edit_operation_totals",
            MappingProxyType(
                _validate_non_negative_int_map(
                    self.edit_operation_totals,
                    "monitoring_aggregate.edit_operation_totals",
                    allowed_keys=_EDIT_OPERATION_TYPES,
                    reject_sensitive_keys=True,
                )
            ),
        )
        object.__setattr__(
            self,
            "review_time_ms_total",
            _non_negative_int(self.review_time_ms_total, "monitoring_aggregate.review_time_ms_total"),
        )
        object.__setattr__(
            self,
            "active_monitoring_record_ids",
            _id_tuple(self.active_monitoring_record_ids, "monitoring_aggregate.active_monitoring_record_ids"),
        )
        object.__setattr__(
            self,
            "active_timeline_ids",
            _id_tuple(self.active_timeline_ids, "monitoring_aggregate.active_timeline_ids", reject_duplicates=False),
        )
        if len(self.active_monitoring_record_ids) != self.active_record_count:
            raise ValidationError("monitoring_aggregate.active_monitoring_record_ids must match active_record_count")
        if len(self.active_timeline_ids) != self.active_record_count:
            raise ValidationError("monitoring_aggregate.active_timeline_ids must match active_record_count")

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_monitoring_record_ids": list(self.active_monitoring_record_ids),
            "active_record_count": self.active_record_count,
            "active_timeline_ids": list(self.active_timeline_ids),
            "degraded_record_count": self.degraded_record_count,
            "edit_operation_totals": dict(sorted(self.edit_operation_totals.items())),
            "expired_record_count": self.expired_record_count,
            "generated_at": self.generated_at,
            "review_time_ms_total": self.review_time_ms_total,
            "route_counts": dict(sorted(self.route_counts.items())),
            "total_records": self.total_records,
        }


def advance_monitoring_promotion_state(
    state: MonitoringPromotionState,
    step: MonitoringPromotionStep,
    *,
    split_id: str | None = None,
    annotation_protocol_version: str | None = None,
    adjudication_id: str | None = None,
) -> MonitoringPromotionState:
    if not isinstance(state, MonitoringPromotionState):
        raise ValidationError("promotion state must be MonitoringPromotionState")
    step = _require_id(step, "promotion.step")
    if step == "grant_consent":
        return replace(state, consent_granted=True)
    if step == "verify_access":
        return replace(state, access_checked=True)
    if step == "assign_split":
        split_id = _require_id(split_id, "promotion.split_id")
        if state.split_id is not None and state.split_id != split_id and (state.annotated or state.adjudicated):
            raise ValidationError("promotion split assignment cannot change after annotation or adjudication")
        return replace(state, split_id=split_id)
    if step == "mark_annotated":
        annotation_protocol_version = _require_id(
            annotation_protocol_version,
            "promotion.annotation_protocol_version",
        )
        if (
            state.annotation_protocol_version is not None
            and state.annotation_protocol_version != annotation_protocol_version
        ):
            raise ValidationError("promotion annotation_protocol_version cannot change after annotation")
        return replace(
            state,
            annotated=True,
            annotation_protocol_version=annotation_protocol_version,
        )
    if step == "mark_adjudicated":
        adjudication_id = _require_id(adjudication_id, "promotion.adjudication_id")
        if state.adjudication_id is not None and state.adjudication_id != adjudication_id:
            raise ValidationError("promotion adjudication_id cannot change after adjudication")
        return replace(state, adjudicated=True, adjudication_id=adjudication_id)
    raise ValidationError(f"promotion step is not supported: {step}")


def aggregate_monitoring_records(records: tuple[MonitoringRecord, ...], *, as_of: str | datetime) -> MonitoringAggregate:
    as_of_text = _timestamp_text(as_of)
    active_record_ids: list[str] = []
    active_timeline_ids: list[str] = []
    route_counts: dict[str, int] = {}
    edit_totals: dict[str, int] = {}
    degraded_count = 0
    review_time_total = 0
    expired_count = 0
    for record in _tuple_of(records, MonitoringRecord, "monitoring.records"):
        route_counts[record.route] = route_counts.get(record.route, 0) + 1
        if not record.confident_speaker_attribution_enabled:
            degraded_count += 1
        for operation, count in record.edit_operation_counts.items():
            edit_totals[operation] = edit_totals.get(operation, 0) + count
        if record.review_time_ms is not None:
            review_time_total += record.review_time_ms
        if record.is_expired(as_of):
            expired_count += 1
            continue
        active_record_ids.append(record.monitoring_record_id)
        timeline_id = record.canonical_timeline_metadata.get("timeline_id")
        if isinstance(timeline_id, str):
            active_timeline_ids.append(timeline_id)
    return MonitoringAggregate(
        generated_at=as_of_text,
        total_records=len(records),
        active_record_count=len(active_record_ids),
        expired_record_count=expired_count,
        route_counts=route_counts,
        degraded_record_count=degraded_count,
        edit_operation_totals=edit_totals,
        review_time_ms_total=review_time_total,
        active_monitoring_record_ids=tuple(active_record_ids),
        active_timeline_ids=tuple(active_timeline_ids),
    )


def monitoring_record_from_dict(payload: Mapping[str, Any]) -> MonitoringRecord:
    data = _mapping(payload, "monitoring")
    _reject_unknown_fields(
        data,
        {
            "canonical_timeline_metadata",
            "confident_speaker_attribution_enabled",
            "degraded_route",
            "edit_operation_counts",
            "engine_config_versions",
            "expires_at",
            "monitoring_policy_version",
            "monitoring_record_id",
            "preflight_policy_id",
            "preflight_policy_version",
            "promotion_state",
            "release_candidate_id",
            "release_record_schema_version",
            "retention_class",
            "review_time_ms",
            "route",
            "schema_version",
        },
        "monitoring",
    )
    return MonitoringRecord(
        schema_version=_required(data, "schema_version", "monitoring"),
        monitoring_record_id=_required(data, "monitoring_record_id", "monitoring"),
        release_candidate_id=_required(data, "release_candidate_id", "monitoring"),
        monitoring_policy_version=_required(data, "monitoring_policy_version", "monitoring"),
        release_record_schema_version=_required(data, "release_record_schema_version", "monitoring"),
        engine_config_versions=_required(data, "engine_config_versions", "monitoring"),
        preflight_policy_id=_required(data, "preflight_policy_id", "monitoring"),
        preflight_policy_version=_required(data, "preflight_policy_version", "monitoring"),
        route=_required(data, "route", "monitoring"),
        confident_speaker_attribution_enabled=_required(
            data,
            "confident_speaker_attribution_enabled",
            "monitoring",
        ),
        degraded_route=data.get("degraded_route"),
        canonical_timeline_metadata=_required(data, "canonical_timeline_metadata", "monitoring"),
        edit_operation_counts=_required(data, "edit_operation_counts", "monitoring"),
        review_time_ms=data.get("review_time_ms"),
        retention_class=_required(data, "retention_class", "monitoring"),
        expires_at=_required(data, "expires_at", "monitoring"),
        promotion_state=monitoring_promotion_state_from_dict(_required(data, "promotion_state", "monitoring")),
    )


def monitoring_promotion_state_from_dict(payload: Mapping[str, Any]) -> MonitoringPromotionState:
    data = _mapping(payload, "promotion")
    _reject_unknown_fields(
        data,
        {
            "access_checked",
            "adjudicated",
            "adjudication_id",
            "annotated",
            "annotation_protocol_version",
            "consent_granted",
            "eligible_for_private_fixture",
            "split_id",
        },
        "promotion",
    )
    state = MonitoringPromotionState(
        consent_granted=_required(data, "consent_granted", "promotion"),
        access_checked=_required(data, "access_checked", "promotion"),
        split_id=_required(data, "split_id", "promotion"),
        annotation_protocol_version=_required(data, "annotation_protocol_version", "promotion"),
        annotated=_required(data, "annotated", "promotion"),
        adjudicated=_required(data, "adjudicated", "promotion"),
        adjudication_id=_required(data, "adjudication_id", "promotion"),
    )
    if "eligible_for_private_fixture" in data and data["eligible_for_private_fixture"] != state.eligible_for_private_fixture:
        raise ValidationError("promotion.eligible_for_private_fixture must match promotion gates")
    return state


def monitoring_record_json_dumps(record: MonitoringRecord) -> str:
    if not isinstance(record, MonitoringRecord):
        raise ValidationError("record must be MonitoringRecord")
    return json.dumps(record.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def monitoring_record_json_loads(text: str) -> MonitoringRecord:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"monitoring record JSON is invalid: {exc.msg}") from exc
    return monitoring_record_from_dict(payload)


def read_monitoring_record_json(path: str | Path) -> MonitoringRecord:
    return monitoring_record_json_loads(Path(path).read_text(encoding="utf-8"))


def write_monitoring_record_json(path: str | Path, record: MonitoringRecord) -> None:
    Path(path).write_text(monitoring_record_json_dumps(record), encoding="utf-8", newline="\n")


def _monitoring_safe_metadata(value: object, field_name: str) -> dict[str, Any]:
    data = _mapping(value, field_name)
    _validate_json_value(data, field_name)
    _reject_forbidden_monitoring_fields(data, field_name)
    _reject_unsupported_timeline_fields(data, field_name)
    return _freeze_json(data)  # type: ignore[return-value]


def _validate_required_timeline_metadata(timeline: Mapping[str, Any]) -> None:
    _require_id(timeline["timeline_id"], "monitoring.timeline.timeline_id")
    _require_id(timeline["transform_chain_id"], "monitoring.timeline.transform_chain_id")
    _positive_int(timeline["duration_ms"], "monitoring.timeline.duration_ms")
    _positive_int(timeline["sample_rate_hz"], "monitoring.timeline.sample_rate_hz")
    time_basis = _require_id(timeline["time_basis"], "monitoring.timeline.time_basis")
    if time_basis not in _TIME_BASES:
        raise ValidationError(f"monitoring.timeline.time_basis is not supported: {time_basis}")
    if "channel_ids" in timeline:
        _id_tuple(timeline["channel_ids"], "monitoring.timeline.channel_ids")


def _reject_unsupported_timeline_fields(timeline: Mapping[str, Any], field_name: str) -> None:
    unknown = sorted(set(timeline) - _ALLOWED_TIMELINE_FIELDS)
    if unknown:
        raise ValidationError(f"{field_name} has unsupported fields: {', '.join(unknown)}")


def _reject_forbidden_monitoring_fields(value: object, field_name: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = _monitoring_safe_key(key, field_name)
            _reject_forbidden_monitoring_fields(item, f"{field_name}.{key_text}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_forbidden_monitoring_fields(item, f"{field_name}[{index}]")


def _monitoring_safe_key(key: object, field_name: str) -> str:
    key_text = _require_id(key, f"{field_name}.key")
    if _monitoring_key_is_forbidden(key_text):
        raise ValidationError(f"{field_name}.{key_text} must not persist sensitive monitoring identity material")
    return key_text


def _monitoring_key_is_forbidden(key: str) -> bool:
    alias = _monitoring_key_alias(key)
    return (
        alias in _FORBIDDEN_MONITORING_KEY_ALIASES
        or _monitoring_key_has_forbidden_identifier_suffix(alias)
        or any(fragment in alias for fragment in _FORBIDDEN_MONITORING_KEY_FRAGMENTS)
    )


def _monitoring_key_has_forbidden_identifier_suffix(alias: str) -> bool:
    return alias.endswith(("guid", "uuid")) and (
        alias in {"guid", "uuid"}
        or "speaker" in alias
        or "session" in alias
        or "user" in alias
        or "participant" in alias
        or "customer" in alias
        or "account" in alias
    )


def _monitoring_key_alias(key: str) -> str:
    return "".join(char for char in key.casefold() if char.isalnum())


def _validate_schema_version(value: object) -> int:
    version = _positive_int(value, "monitoring.schema_version")
    if version not in SUPPORTED_MONITORING_RECORD_SCHEMA_VERSIONS:
        raise ValidationError(f"monitoring.schema_version is not supported: {version}")
    return version


def _validate_route(value: object, field_name: str) -> PreflightRoute:
    route = _require_id(value, field_name)
    if route not in ALLOWED_PREFLIGHT_ROUTES:
        raise ValidationError(f"{field_name} is not supported: {route}")
    return route  # type: ignore[return-value]


def _validate_string_map(value: object, field_name: str, *, reject_sensitive_keys: bool = False) -> dict[str, str]:
    data = _mapping(value, field_name)
    result: dict[str, str] = {}
    for key, item in data.items():
        key_text = (
            _monitoring_safe_key(key, field_name)
            if reject_sensitive_keys
            else _require_id(key, f"{field_name}.key")
        )
        result[key_text] = _require_id(item, f"{field_name}.{key_text}")
    return result


def _validate_non_negative_int_map(
    value: object,
    field_name: str,
    *,
    allowed_keys: frozenset[str] | None = None,
    reject_sensitive_keys: bool = False,
) -> dict[str, int]:
    data = _mapping(value, field_name)
    result: dict[str, int] = {}
    for key, item in data.items():
        key_text = (
            _monitoring_safe_key(key, field_name)
            if reject_sensitive_keys
            else _require_id(key, f"{field_name}.key")
        )
        if allowed_keys is not None and key_text not in allowed_keys:
            raise ValidationError(f"{field_name}.{key_text} is not supported")
        result[key_text] = _non_negative_int(item, f"{field_name}.{key_text}")
    return result


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
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
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
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
        raise ValidationError(f"{field_name} must be an array")
    result = tuple(values)
    for index, item in enumerate(result):
        if not isinstance(item, item_type):
            raise ValidationError(f"{field_name}[{index}] must be {item_type.__name__}")
    return result


def _id_tuple(values: object, field_name: str, *, reject_duplicates: bool = True) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
        raise ValidationError(f"{field_name} must be an array")
    result = tuple(_require_id(item, f"{field_name}[{index}]") for index, item in enumerate(values))
    if reject_duplicates:
        _reject_duplicates(result, field_name)
    return result


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
    field_names = set()
    for key in data:
        if not isinstance(key, str):
            raise ValidationError(f"{field_name} field names must be strings")
        field_names.add(key)
    unknown = sorted(field_names - allowed_fields)
    if unknown:
        raise ValidationError(f"{field_name} has unsupported fields: {', '.join(unknown)}")


def _reject_duplicates(values: tuple[Any, ...], field_name: str) -> None:
    seen: set[Any] = set()
    for value in values:
        if value in seen:
            raise ValidationError(f"{field_name} contains duplicate value: {value}")
        seen.add(value)


def _parse_timestamp(value: object, field_name: str) -> datetime:
    text = _require_text(value, field_name) if not isinstance(value, datetime) else value.isoformat()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValidationError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _timestamp_text(value: str | datetime) -> str:
    parsed = _parse_timestamp(value, "as_of")
    return parsed.isoformat().replace("+00:00", "Z")


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


def _positive_int(value: object, field_name: str) -> int:
    value = _require_int(value, field_name)
    if value <= 0:
        raise ValidationError(f"{field_name} must be greater than 0")
    return value


def _non_negative_int(value: object, field_name: str) -> int:
    value = _require_int(value, field_name)
    if value < 0:
        raise ValidationError(f"{field_name} must be >= 0")
    return value


def _require_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{field_name} must be an integer")
    return value
