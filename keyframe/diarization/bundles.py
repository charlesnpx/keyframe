"""Reference and redacted candidate bundle contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from keyframe.diarization.models import CanonicalRecording, ValidationError
from keyframe.diarization.provenance import AudioTimelineProvenance, NormalizedArtifactProvenance


CandidateBundleMode = Literal["product_realistic", "oracle_diagnostic", "authenticated_track_metadata"]
_ALLOWED_BUNDLE_MODES = frozenset(
    {
        "product_realistic",
        "oracle_diagnostic",
        "authenticated_track_metadata",
    }
)
_FORBIDDEN_CANDIDATE_FIELDS = frozenset(
    {
        "canonical_audio_id",
        "corpus_identity",
        "corpus_speaker_id",
        "cross_recording_identity",
        "display_label",
        "evaluator_speaker_map",
        "global_identity",
        "local_audio_sha256",
        "oracle",
        "oracle_metadata",
        "original_audio_id",
        "participant_id",
        "reference_speaker_id",
        "role",
        "role_label",
        "speaker_ref",
        "voice_profile",
    }
)
_ORACLE_ONLY_MODES = frozenset({"oracle_diagnostic"})
_RESERVED_RUNTIME_HINT_KEYS = frozenset({"channel_ids", "mode_supports_speaker_identity", "timeline"})


@dataclass(frozen=True)
class ReferenceBundle:
    """Evaluator-only bundle containing canonical reference data and mappings."""

    recording: CanonicalRecording
    artifact: NormalizedArtifactProvenance
    evaluator_speaker_map: dict[str, str]
    oracle_metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.recording, CanonicalRecording):
            raise ValidationError("reference_bundle.recording must be a CanonicalRecording")
        if not isinstance(self.artifact, NormalizedArtifactProvenance):
            raise ValidationError("reference_bundle.artifact must be a NormalizedArtifactProvenance")
        self.artifact.timeline.assert_consistent_with_recording(self.recording)
        object.__setattr__(
            self,
            "evaluator_speaker_map",
            _validate_string_map(self.evaluator_speaker_map, "reference_bundle.evaluator_speaker_map"),
        )
        object.__setattr__(
            self,
            "oracle_metadata",
            _validate_optional_metadata(self.oracle_metadata, "reference_bundle.oracle_metadata"),
        )

    @classmethod
    def from_recording(
        cls,
        recording: CanonicalRecording,
        *,
        artifact_id: str,
        evaluator_speaker_map: dict[str, str] | None = None,
        local_audio_sha256: str | None = None,
        oracle_metadata: dict[str, Any] | None = None,
    ) -> ReferenceBundle:
        return cls(
            recording=recording,
            artifact=NormalizedArtifactProvenance.from_recording(
                recording,
                artifact_id=artifact_id,
                artifact_kind="reference",
                local_audio_sha256=local_audio_sha256,
            ),
            evaluator_speaker_map=evaluator_speaker_map or {
                speaker.speaker_ref: speaker.speaker_ref for speaker in recording.speakers
            },
            oracle_metadata=oracle_metadata,
        )

    def to_evaluator_dict(self) -> dict[str, Any]:
        return {
            "artifact": self.artifact.to_integrity_dict(),
            "evaluator_speaker_map": dict(self.evaluator_speaker_map),
            "oracle_metadata": None if self.oracle_metadata is None else dict(self.oracle_metadata),
            "recording": self.recording.to_dict(),
        }


@dataclass(frozen=True)
class CandidateBundle:
    """Candidate-visible bundle with physically redacted runtime inputs."""

    bundle_id: str
    mode: CandidateBundleMode
    audio: dict[str, Any]
    channels: tuple[dict[str, Any], ...]
    runtime_hints: dict[str, Any]
    oracle_diagnostic: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "bundle_id", _require_id(self.bundle_id, "candidate_bundle.bundle_id"))
        mode = _validate_mode(self.mode)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "oracle_diagnostic", _validate_bool(self.oracle_diagnostic, "oracle_diagnostic"))
        if mode in _ORACLE_ONLY_MODES and not self.oracle_diagnostic:
            raise ValidationError("oracle diagnostic bundles must be explicitly labeled")
        if mode not in _ORACLE_ONLY_MODES and self.oracle_diagnostic:
            raise ValidationError("product-quality bundles cannot be labeled oracle diagnostic")
        audio, channels, runtime_hints = _validate_candidate_payload_parts(
            self.audio,
            self.channels,
            self.runtime_hints,
        )
        object.__setattr__(self, "audio", _freeze_metadata(audio))
        object.__setattr__(self, "channels", tuple(_freeze_metadata(channel) for channel in channels))
        object.__setattr__(self, "runtime_hints", _freeze_metadata(runtime_hints))
        self.validate()

    @property
    def product_quality_reportable(self) -> bool:
        return not self.oracle_diagnostic

    def validate(self) -> None:
        _reject_forbidden_fields(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "audio": _thaw_metadata(self.audio),
            "bundle_id": self.bundle_id,
            "channels": [_thaw_metadata(channel) for channel in self.channels],
            "mode": self.mode,
            "oracle_diagnostic": self.oracle_diagnostic,
            "product_quality_reportable": self.product_quality_reportable,
            "runtime_hints": _thaw_metadata(self.runtime_hints),
        }


def build_candidate_bundle(
    reference: ReferenceBundle,
    *,
    bundle_id: str,
    mode: CandidateBundleMode = "product_realistic",
    runtime_hints: dict[str, Any] | None = None,
) -> CandidateBundle:
    if not isinstance(reference, ReferenceBundle):
        raise ValidationError("reference must be a ReferenceBundle")
    mode = _validate_mode(mode)
    audio = _audio_payload(reference.artifact.timeline)
    channels = _channel_payloads(reference, mode)
    extra_hints = _validate_metadata(
        {} if runtime_hints is None else runtime_hints,
        "candidate_bundle.runtime_hints",
    )
    if _RESERVED_RUNTIME_HINT_KEYS.intersection(extra_hints):
        raise ValidationError("candidate_bundle.runtime_hints cannot override generated runtime metadata")
    hints = _default_runtime_hints(reference.artifact.timeline)
    hints.update(extra_hints)
    return CandidateBundle(
        bundle_id=bundle_id,
        mode=mode,
        audio=audio,
        channels=channels,
        runtime_hints=hints,
        oracle_diagnostic=mode == "oracle_diagnostic",
    )


def build_candidate_bundle_from_recording(
    recording: CanonicalRecording,
    *,
    artifact_id: str,
    bundle_id: str,
    mode: CandidateBundleMode = "product_realistic",
    local_audio_sha256: str | None = None,
    runtime_hints: dict[str, Any] | None = None,
) -> CandidateBundle:
    reference = ReferenceBundle.from_recording(
        recording,
        artifact_id=artifact_id,
        local_audio_sha256=local_audio_sha256,
    )
    return build_candidate_bundle(reference, bundle_id=bundle_id, mode=mode, runtime_hints=runtime_hints)


def validate_candidate_bundle_payload(payload: dict[str, Any]) -> None:
    data = _validate_metadata(payload, "candidate_bundle")
    _reject_forbidden_fields(data)

    mode = _validate_mode(data.get("mode"))
    oracle_diagnostic = _validate_bool(data.get("oracle_diagnostic"), "candidate_bundle.oracle_diagnostic")
    product_quality_reportable = _validate_bool(
        data.get("product_quality_reportable"),
        "candidate_bundle.product_quality_reportable",
    )

    _require_id(data.get("bundle_id"), "candidate_bundle.bundle_id")
    _validate_candidate_payload_parts(data.get("audio"), data.get("channels"), data.get("runtime_hints"))

    if mode == "oracle_diagnostic":
        if oracle_diagnostic is not True:
            raise ValidationError("oracle diagnostic bundles must be explicitly labeled")
        if product_quality_reportable is not False:
            raise ValidationError("oracle diagnostic bundles must be non-reportable")
    else:
        if oracle_diagnostic is True:
            raise ValidationError("product-quality bundles cannot be labeled oracle diagnostic")
        if product_quality_reportable is not True:
            raise ValidationError("product-quality bundles must be reportable")


def _audio_payload(timeline: AudioTimelineProvenance) -> dict[str, Any]:
    return {
        "channel_count": len(timeline.channel_ids),
        "duration_ms": timeline.duration_ms,
        "sample_rate_hz": timeline.sample_rate_hz,
        "time_basis": timeline.time_basis,
    }


def _channel_payloads(reference: ReferenceBundle, mode: CandidateBundleMode) -> tuple[dict[str, Any], ...]:
    if mode == "authenticated_track_metadata":
        payloads = []
        for channel in reference.recording.channels:
            payload: dict[str, Any] = {"channel_id": channel.channel_id}
            if channel.name is not None:
                payload["track_name"] = channel.name
            payloads.append(payload)
        return tuple(payloads)
    return tuple({"channel_id": channel.channel_id} for channel in reference.recording.channels)


def _default_runtime_hints(timeline: AudioTimelineProvenance) -> dict[str, Any]:
    return {
        "channel_ids": list(timeline.channel_ids),
        "mode_supports_speaker_identity": False,
        "timeline": timeline.to_rendered_transcript_metadata(),
    }


def _reject_forbidden_fields(payload: object, path: str = "candidate_bundle") -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in _FORBIDDEN_CANDIDATE_FIELDS:
                raise ValidationError(f"{path}.{key} is forbidden in candidate bundles")
            _reject_forbidden_fields(value, f"{path}.{key}")
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            _reject_forbidden_fields(value, f"{path}[{index}]")


def _validate_channel_payloads(value: object) -> tuple[dict[str, Any], ...]:
    try:
        channels = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValidationError("candidate_bundle.channels must be an iterable") from exc
    if not channels:
        raise ValidationError("candidate_bundle.channels is required")
    result = []
    seen: set[str] = set()
    for index, channel in enumerate(channels):
        payload = _validate_metadata(channel, f"candidate_bundle.channels[{index}]")
        channel_id = _require_id(payload.get("channel_id"), f"candidate_bundle.channels[{index}].channel_id")
        payload["channel_id"] = channel_id
        if channel_id in seen:
            raise ValidationError(f"duplicate candidate_bundle.channels.channel_id: {channel_id}")
        seen.add(channel_id)
        result.append(payload)
    return tuple(result)


def _validate_candidate_payload_parts(
    audio_value: object,
    channels_value: object,
    runtime_hints_value: object,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], dict[str, Any]]:
    audio = _validate_audio_payload(audio_value)
    channels = _validate_channel_payloads(channels_value)
    runtime_hints = _validate_runtime_hints_payload(runtime_hints_value, channels)
    timeline = runtime_hints["timeline"]

    if audio["channel_count"] != len(channels):
        raise ValidationError("candidate_bundle.audio.channel_count must match channels")
    if timeline["channel_ids"] != runtime_hints["channel_ids"]:
        raise ValidationError("candidate_bundle.runtime_hints.timeline.channel_ids must match channels")
    if timeline["duration_ms"] != audio["duration_ms"]:
        raise ValidationError("candidate_bundle.runtime_hints.timeline.duration_ms must match audio")
    if timeline["sample_rate_hz"] != audio["sample_rate_hz"]:
        raise ValidationError("candidate_bundle.runtime_hints.timeline.sample_rate_hz must match audio")
    if timeline["time_basis"] != audio["time_basis"]:
        raise ValidationError("candidate_bundle.runtime_hints.timeline.time_basis must match audio")

    return audio, channels, runtime_hints


def _validate_audio_payload(value: object) -> dict[str, Any]:
    payload = _validate_metadata(value, "candidate_bundle.audio")
    payload["channel_count"] = _require_positive_int(
        payload.get("channel_count"),
        "candidate_bundle.audio.channel_count",
    )
    payload["duration_ms"] = _require_positive_int(
        payload.get("duration_ms"),
        "candidate_bundle.audio.duration_ms",
    )
    payload["sample_rate_hz"] = _require_positive_int(
        payload.get("sample_rate_hz"),
        "candidate_bundle.audio.sample_rate_hz",
    )
    payload["time_basis"] = _validate_time_basis(payload.get("time_basis"), "candidate_bundle.audio.time_basis")
    return payload


def _validate_runtime_hints_payload(
    value: object,
    channels: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    payload = _validate_metadata(value, "candidate_bundle.runtime_hints")
    channel_ids = [channel["channel_id"] for channel in channels]
    payload["channel_ids"] = _validate_channel_ids_list(
        payload.get("channel_ids"),
        "candidate_bundle.runtime_hints.channel_ids",
    )
    if payload["channel_ids"] != channel_ids:
        raise ValidationError("candidate_bundle.runtime_hints.channel_ids must match channels")
    if payload.get("mode_supports_speaker_identity") is not False:
        raise ValidationError("candidate_bundle.runtime_hints.mode_supports_speaker_identity must be false")

    timeline = _validate_metadata(payload.get("timeline"), "candidate_bundle.runtime_hints.timeline")
    timeline["channel_ids"] = _validate_channel_ids_list(
        timeline.get("channel_ids"),
        "candidate_bundle.runtime_hints.timeline.channel_ids",
    )
    timeline["duration_ms"] = _require_positive_int(
        timeline.get("duration_ms"),
        "candidate_bundle.runtime_hints.timeline.duration_ms",
    )
    timeline["sample_rate_hz"] = _require_positive_int(
        timeline.get("sample_rate_hz"),
        "candidate_bundle.runtime_hints.timeline.sample_rate_hz",
    )
    timeline["time_basis"] = _validate_time_basis(
        timeline.get("time_basis"),
        "candidate_bundle.runtime_hints.timeline.time_basis",
    )
    timeline["timeline_id"] = _require_id(
        timeline.get("timeline_id"),
        "candidate_bundle.runtime_hints.timeline.timeline_id",
    )
    timeline["transform_chain_id"] = _require_id(
        timeline.get("transform_chain_id"),
        "candidate_bundle.runtime_hints.timeline.transform_chain_id",
    )
    payload["timeline"] = timeline
    return payload


def _validate_channel_ids_list(value: object, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValidationError(f"{field_name} must be a list")
    result = []
    for index, item in enumerate(value):
        result.append(_require_id(item, f"{field_name}[{index}]"))
    return result


def _validate_metadata(value: object, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{field_name} must be an object")
    return {key: _validate_metadata_value(key, item, field_name) for key, item in value.items()}


def _validate_optional_metadata(value: object, field_name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    return _validate_metadata(value, field_name)


def _validate_metadata_value(key: object, value: object, field_name: str) -> Any:
    if not isinstance(key, str):
        raise ValidationError(f"{field_name} field names must be strings")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError(f"{field_name}.{key} must be a finite JSON number")
        return value
    if isinstance(value, list):
        return [_validate_metadata_value(key, item, field_name) for item in value]
    if isinstance(value, dict):
        return _validate_metadata(value, f"{field_name}.{key}")
    raise ValidationError(f"{field_name}.{key} must be JSON-compatible")


def _freeze_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_metadata(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_metadata(item) for item in value)
    return value


def _thaw_metadata(value: Any) -> Any:
    if isinstance(value, MappingProxyType):
        return {key: _thaw_metadata(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_metadata(item) for item in value]
    return value


def _validate_string_map(value: object, field_name: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValidationError(f"{field_name} must be an object")
    result: dict[str, str] = {}
    for key, item in value.items():
        result[_require_id(key, f"{field_name}.key")] = _require_id(item, f"{field_name}.{key}")
    return result


def _validate_mode(value: object) -> CandidateBundleMode:
    value = _require_id(value, "candidate_bundle.mode")
    if value not in _ALLOWED_BUNDLE_MODES:
        raise ValidationError(f"candidate_bundle.mode is not supported: {value}")
    return value  # type: ignore[return-value]


def _validate_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValidationError(f"{field_name} must be a boolean")
    return value


def _require_positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValidationError(f"{field_name} must be greater than 0")
    return value


def _validate_time_basis(value: object, field_name: str) -> str:
    value = _require_id(value, field_name)
    if value not in {"canonical_ms", "chunk_relative_ms", "sample_index", "frame_index"}:
        raise ValidationError(f"{field_name} is not supported: {value}")
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
