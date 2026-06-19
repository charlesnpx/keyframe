"""Audio timeline and transform provenance for diarization artifacts."""

from __future__ import annotations

import math
import hashlib
import json
import shlex
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from keyframe.diarization.models import CanonicalRecording, TimeBasis, ValidationError


ArtifactKind = Literal["asr", "diarization", "reference", "candidate", "fixture"]
TransformType = Literal["identity", "resample", "channel_map", "chunk", "offset_map", "other"]
GainPolicy = Literal["none", "preserve", "normalize_peak", "normalize_lufs"]
DownmixPolicy = Literal["none", "mono_mix", "average", "first_channel", "custom"]

_ALLOWED_ARTIFACT_KINDS = frozenset({"asr", "diarization", "reference", "candidate", "fixture"})
_ALLOWED_DOWNMIX_POLICIES = frozenset({"none", "mono_mix", "average", "first_channel", "custom"})
_ALLOWED_GAIN_POLICIES = frozenset({"none", "preserve", "normalize_peak", "normalize_lufs"})
_ALLOWED_TIME_BASES = frozenset({"canonical_ms", "chunk_relative_ms", "sample_index", "frame_index"})
_ALLOWED_TRANSFORM_TYPES = frozenset({"identity", "resample", "channel_map", "chunk", "offset_map", "other"})
_LOCAL_ONLY_KEYS = frozenset({"original_audio_id", "canonical_audio_id", "local_audio_sha256"})
_MILLISECOND_TIME_BASES = frozenset({"canonical_ms", "chunk_relative_ms"})
_HEX_DIGITS = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class AudioChannelMapping:
    """Deterministic channel mapping used to produce canonical benchmark audio."""

    output_channel_id: str
    source_channel_ids: tuple[str, ...]
    weights: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_channel_id", _require_id(self.output_channel_id, "channel_mapping.output"))
        source_channel_ids = _unique_ids(self.source_channel_ids, "channel_mapping.source_channel_ids")
        object.__setattr__(self, "source_channel_ids", source_channel_ids)
        weights = _validate_weights(self.weights, "channel_mapping.weights")
        if weights and len(weights) != len(source_channel_ids):
            raise ValidationError("channel_mapping.weights must match source_channel_ids")
        object.__setattr__(self, "weights", weights)

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_channel_id": self.output_channel_id,
            "source_channel_ids": list(self.source_channel_ids),
            "weights": list(self.weights),
        }


@dataclass(frozen=True)
class AudioTransformConfig:
    """Hashable audio transform configuration for canonical benchmark assets."""

    tool_name: str
    tool_version: str
    normalized_command: str | Sequence[str]
    channel_mapping: tuple[AudioChannelMapping, ...]
    gain_policy: GainPolicy = "preserve"
    downmix_policy: DownmixPolicy = "none"
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_name", _require_id(self.tool_name, "audio_transform.tool_name"))
        object.__setattr__(self, "tool_version", _require_id(self.tool_version, "audio_transform.tool_version"))
        object.__setattr__(
            self,
            "normalized_command",
            normalize_transform_command(self.normalized_command),
        )
        object.__setattr__(
            self,
            "channel_mapping",
            _as_tuple_of(self.channel_mapping, AudioChannelMapping, "audio_transform.channel_mapping"),
        )
        if not self.channel_mapping:
            raise ValidationError("audio_transform.channel_mapping is required")
        gain_policy = _require_id(self.gain_policy, "audio_transform.gain_policy")
        if gain_policy not in _ALLOWED_GAIN_POLICIES:
            raise ValidationError(f"audio_transform.gain_policy is not supported: {gain_policy}")
        object.__setattr__(self, "gain_policy", gain_policy)
        downmix_policy = _require_id(self.downmix_policy, "audio_transform.downmix_policy")
        if downmix_policy not in _ALLOWED_DOWNMIX_POLICIES:
            raise ValidationError(f"audio_transform.downmix_policy is not supported: {downmix_policy}")
        object.__setattr__(self, "downmix_policy", downmix_policy)
        object.__setattr__(self, "parameters", _validate_parameters(self.parameters, "audio_transform.parameters"))

    @property
    def config_hash(self) -> str:
        return hash_audio_transform_config(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel_mapping": [mapping.to_dict() for mapping in self.channel_mapping],
            "downmix_policy": self.downmix_policy,
            "gain_policy": self.gain_policy,
            "normalized_command": self.normalized_command,
            "parameters": dict(self.parameters),
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
        }


@dataclass(frozen=True)
class AudioTransformManifest:
    """Integrity-only manifest for one canonical audio transform/cache entry."""

    transform_id: str
    branch_id: str
    original_audio_id: str
    canonical_audio_id: str
    original_audio_sha256: str
    canonical_audio_sha256: str
    transform_config_hash: str
    config: AudioTransformConfig

    def __post_init__(self) -> None:
        object.__setattr__(self, "transform_id", _require_id(self.transform_id, "audio_transform_manifest.transform_id"))
        object.__setattr__(self, "branch_id", _require_id(self.branch_id, "audio_transform_manifest.branch_id"))
        object.__setattr__(
            self,
            "original_audio_id",
            _require_id(self.original_audio_id, "audio_transform_manifest.original_audio_id"),
        )
        object.__setattr__(
            self,
            "canonical_audio_id",
            _require_id(self.canonical_audio_id, "audio_transform_manifest.canonical_audio_id"),
        )
        object.__setattr__(
            self,
            "original_audio_sha256",
            _validate_sha256(self.original_audio_sha256, "audio_transform_manifest.original_audio_sha256"),
        )
        object.__setattr__(
            self,
            "canonical_audio_sha256",
            _validate_sha256(self.canonical_audio_sha256, "audio_transform_manifest.canonical_audio_sha256"),
        )
        if not isinstance(self.config, AudioTransformConfig):
            raise ValidationError("audio_transform_manifest.config must be an AudioTransformConfig")
        expected_config_hash = hash_audio_transform_config(self.config)
        object.__setattr__(
            self,
            "transform_config_hash",
            _validate_sha256(self.transform_config_hash, "audio_transform_manifest.transform_config_hash"),
        )
        if self.transform_config_hash != expected_config_hash:
            raise ValidationError("audio_transform_manifest.transform_config_hash does not match config")
        expected_transform_id = _content_addressed_transform_id(
            branch_id=self.branch_id,
            original_audio_sha256=self.original_audio_sha256,
            canonical_audio_sha256=self.canonical_audio_sha256,
            transform_config_hash=self.transform_config_hash,
        )
        if self.transform_id != expected_transform_id:
            raise ValidationError("audio_transform_manifest.transform_id does not match content address")

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "canonical_audio_id": self.canonical_audio_id,
            "canonical_audio_sha256": self.canonical_audio_sha256,
            "config": self.config.to_dict(),
            "original_audio_id": self.original_audio_id,
            "original_audio_sha256": self.original_audio_sha256,
            "transform_config_hash": self.transform_config_hash,
            "transform_id": self.transform_id,
        }

    def assert_consistent_with_timeline(self, timeline: "AudioTimelineProvenance") -> None:
        if not isinstance(timeline, AudioTimelineProvenance):
            raise ValidationError("timeline must be an AudioTimelineProvenance")
        if self.original_audio_id != timeline.original_audio_id:
            raise ValidationError("transform original_audio_id conflicts with timeline")
        if self.canonical_audio_id != timeline.canonical_audio_id:
            raise ValidationError("transform canonical_audio_id conflicts with timeline")
        output_channel_ids = tuple(mapping.output_channel_id for mapping in self.config.channel_mapping)
        if output_channel_ids != timeline.channel_ids:
            raise ValidationError("transform channel mapping conflicts with timeline channel_ids")
        if timeline.local_audio_sha256 is not None and timeline.local_audio_sha256 != self.canonical_audio_sha256:
            raise ValidationError("transform canonical_audio_sha256 conflicts with timeline local_audio_sha256")


def normalize_transform_command(command: str | Sequence[str]) -> str:
    """Normalize a shell-style transform command for stable config hashing."""

    if isinstance(command, str):
        tokens = shlex.split(command)
    elif isinstance(command, Sequence):
        tokens = []
        for index, token in enumerate(command):
            if not isinstance(token, str) or not token.strip():
                raise ValidationError(f"audio_transform.normalized_command[{index}] must be a non-empty string")
            tokens.append(token)
    else:
        raise ValidationError("audio_transform.normalized_command must be a string or sequence")
    if not tokens:
        raise ValidationError("audio_transform.normalized_command is required")
    return shlex.join(tokens)


def sha256_bytes(payload: bytes) -> str:
    if not isinstance(payload, bytes):
        raise ValidationError("payload must be bytes")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_audio_transform_config(config: AudioTransformConfig) -> str:
    if not isinstance(config, AudioTransformConfig):
        raise ValidationError("config must be an AudioTransformConfig")
    payload = json.dumps(config.to_dict(), ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_audio_transform_manifest(
    *,
    branch_id: str,
    original_audio_id: str,
    canonical_audio_id: str,
    original_audio_sha256: str,
    canonical_audio_sha256: str,
    config: AudioTransformConfig,
) -> AudioTransformManifest:
    branch_id = _require_id(branch_id, "audio_transform_manifest.branch_id")
    original_audio_sha256 = _validate_sha256(original_audio_sha256, "audio_transform_manifest.original_audio_sha256")
    canonical_audio_sha256 = _validate_sha256(
        canonical_audio_sha256,
        "audio_transform_manifest.canonical_audio_sha256",
    )
    config_hash = hash_audio_transform_config(config)
    transform_id = _content_addressed_transform_id(
        branch_id=branch_id,
        original_audio_sha256=original_audio_sha256,
        canonical_audio_sha256=canonical_audio_sha256,
        transform_config_hash=config_hash,
    )
    return AudioTransformManifest(
        transform_id=transform_id,
        branch_id=branch_id,
        original_audio_id=original_audio_id,
        canonical_audio_id=canonical_audio_id,
        original_audio_sha256=original_audio_sha256,
        canonical_audio_sha256=canonical_audio_sha256,
        transform_config_hash=config_hash,
        config=config,
    )


def build_mono_mix_transform_manifest(
    *,
    branch_id: str,
    original_audio_id: str,
    canonical_audio_id: str,
    original_audio_sha256: str,
    canonical_audio_sha256: str,
    source_channel_ids: tuple[str, ...],
    tool_name: str,
    tool_version: str,
    command: str | Sequence[str],
    gain_policy: GainPolicy = "preserve",
) -> AudioTransformManifest:
    source_channel_ids = _unique_ids(source_channel_ids, "audio_transform.source_channel_ids")
    weight = round(1.0 / len(source_channel_ids), 12)
    config = AudioTransformConfig(
        tool_name=tool_name,
        tool_version=tool_version,
        normalized_command=command,
        channel_mapping=(
            AudioChannelMapping(
                output_channel_id="mono-mix",
                source_channel_ids=source_channel_ids,
                weights=tuple(weight for _ in source_channel_ids),
            ),
        ),
        gain_policy=gain_policy,
        downmix_policy="mono_mix",
    )
    return build_audio_transform_manifest(
        branch_id=branch_id,
        original_audio_id=original_audio_id,
        canonical_audio_id=canonical_audio_id,
        original_audio_sha256=original_audio_sha256,
        canonical_audio_sha256=canonical_audio_sha256,
        config=config,
    )


@dataclass(frozen=True)
class TransformStep:
    """One local audio normalization transform in a chain."""

    step_id: str
    transform_type: TransformType
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_id", _require_id(self.step_id, "transform_step.step_id"))
        transform_type = _require_id(self.transform_type, "transform_step.transform_type")
        if transform_type not in _ALLOWED_TRANSFORM_TYPES:
            raise ValidationError(f"transform_step.transform_type is not supported: {transform_type}")
        object.__setattr__(self, "transform_type", transform_type)
        object.__setattr__(self, "parameters", _validate_parameters(self.parameters, "transform_step.parameters"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameters": dict(self.parameters),
            "step_id": self.step_id,
            "transform_type": self.transform_type,
        }


@dataclass(frozen=True)
class TransformChain:
    """A stable local transform-chain identifier plus auditable steps."""

    transform_chain_id: str
    steps: tuple[TransformStep, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "transform_chain_id", _require_id(self.transform_chain_id, "transform_chain_id"))
        object.__setattr__(self, "steps", _as_tuple_of(self.steps, TransformStep, "transform_chain.steps"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [step.to_dict() for step in self.steps],
            "transform_chain_id": self.transform_chain_id,
        }


@dataclass(frozen=True)
class AudioTimelineProvenance:
    """Local benchmark provenance for one normalized audio timeline."""

    original_audio_id: str
    canonical_audio_id: str
    timeline_id: str
    transform_chain_id: str
    sample_rate_hz: int
    duration_ms: int
    channel_ids: tuple[str, ...]
    time_basis: TimeBasis = "canonical_ms"
    local_audio_sha256: str | None = None
    transform_manifest: AudioTransformManifest | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "original_audio_id", _require_id(self.original_audio_id, "original_audio_id"))
        object.__setattr__(self, "canonical_audio_id", _require_id(self.canonical_audio_id, "canonical_audio_id"))
        object.__setattr__(self, "timeline_id", _require_id(self.timeline_id, "timeline_id"))
        object.__setattr__(self, "transform_chain_id", _require_id(self.transform_chain_id, "transform_chain_id"))
        object.__setattr__(self, "sample_rate_hz", _require_positive_int(self.sample_rate_hz, "sample_rate_hz"))
        object.__setattr__(self, "duration_ms", _require_positive_int(self.duration_ms, "duration_ms"))
        object.__setattr__(self, "channel_ids", _unique_ids(self.channel_ids, "channel_ids"))
        object.__setattr__(self, "time_basis", _validate_time_basis(self.time_basis, "time_basis"))
        object.__setattr__(self, "local_audio_sha256", _optional_id(self.local_audio_sha256, "local_audio_sha256"))
        if self.transform_manifest is not None:
            if not isinstance(self.transform_manifest, AudioTransformManifest):
                raise ValidationError("transform_manifest must be an AudioTransformManifest")
            self.transform_manifest.assert_consistent_with_timeline(self)

    @classmethod
    def from_recording(
        cls,
        recording: CanonicalRecording,
        *,
        local_audio_sha256: str | None = None,
        transform_manifest: AudioTransformManifest | None = None,
    ) -> AudioTimelineProvenance:
        if not isinstance(recording, CanonicalRecording):
            raise ValidationError("recording must be a CanonicalRecording")
        return cls(
            original_audio_id=recording.original_audio_id,
            canonical_audio_id=recording.canonical_audio_id,
            timeline_id=recording.timeline_id,
            transform_chain_id=recording.transform_chain_id,
            sample_rate_hz=recording.sample_rate_hz,
            duration_ms=recording.duration_ms,
            channel_ids=tuple(channel.channel_id for channel in recording.channels),
            time_basis=recording.time_basis,
            local_audio_sha256=local_audio_sha256,
            transform_manifest=transform_manifest,
        )

    def to_integrity_dict(self) -> dict[str, Any]:
        """Return local-only integrity/cache provenance."""

        payload: dict[str, Any] = {
            "canonical_audio_id": self.canonical_audio_id,
            "channel_ids": list(self.channel_ids),
            "duration_ms": self.duration_ms,
            "local_audio_sha256": self.local_audio_sha256,
            "original_audio_id": self.original_audio_id,
            "sample_rate_hz": self.sample_rate_hz,
            "time_basis": self.time_basis,
            "timeline_id": self.timeline_id,
            "transform_chain_id": self.transform_chain_id,
            "transform_manifest": None,
        }
        if self.transform_manifest is not None:
            payload["transform_manifest"] = self.transform_manifest.to_dict()
        return payload

    def to_rendered_transcript_metadata(self) -> dict[str, Any]:
        """Return transcript-safe metadata without local audio IDs or hashes."""

        return _without_local_audio_identity(
            {
                "channel_ids": list(self.channel_ids),
                "duration_ms": self.duration_ms,
                "sample_rate_hz": self.sample_rate_hz,
                "time_basis": self.time_basis,
                "timeline_id": self.timeline_id,
                "transform_chain_id": self.transform_chain_id,
            }
        )

    def to_monitoring_metadata(self) -> dict[str, Any]:
        """Return monitoring-safe metadata without local audio IDs or hashes."""

        return self.to_rendered_transcript_metadata()

    def to_cross_session_linking_metadata(self) -> dict[str, Any]:
        """Return metadata safe for aggregate linking without local audio identity."""

        return {
            "channel_count": len(self.channel_ids),
            "sample_rate_hz": self.sample_rate_hz,
            "time_basis": self.time_basis,
        }

    def assert_consistent_with_recording(self, recording: CanonicalRecording) -> None:
        expected = AudioTimelineProvenance.from_recording(recording, local_audio_sha256=self.local_audio_sha256)
        _validate_shared_audio_metadata(self, expected)
        if self.timeline_id != expected.timeline_id:
            raise ValidationError("recording timeline_id conflicts with provenance")
        if self.transform_chain_id != expected.transform_chain_id:
            raise ValidationError("recording transform_chain_id conflicts with provenance")
        if self.time_basis != expected.time_basis:
            raise ValidationError("recording time_basis conflicts with provenance")

    def to_canonical_ms(
        self,
        value: object,
        *,
        chunk_start_ms: int | None = None,
        frame_rate_fps: float | None = None,
    ) -> int:
        value = _require_non_negative_int(value, "timestamp")
        if self.time_basis == "canonical_ms":
            return value
        if self.time_basis == "chunk_relative_ms":
            if chunk_start_ms is None:
                raise ValidationError("chunk_start_ms is required for chunk_relative_ms conversion")
            return _require_non_negative_int(chunk_start_ms, "chunk_start_ms") + value
        if self.time_basis == "sample_index":
            return round(value * 1000 / self.sample_rate_hz)
        if self.time_basis == "frame_index":
            frame_rate = _require_positive_finite_number(frame_rate_fps, "frame_rate_fps")
            return round(value * 1000 / frame_rate)
        raise ValidationError(f"time_basis is not supported: {self.time_basis}")


@dataclass(frozen=True)
class NormalizedArtifactProvenance:
    """Provenance wrapper for ASR, diarization, fixture, or candidate artifacts."""

    artifact_id: str
    artifact_kind: ArtifactKind
    timeline: AudioTimelineProvenance

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", _require_id(self.artifact_id, "artifact_id"))
        artifact_kind = _require_id(self.artifact_kind, "artifact_kind")
        if artifact_kind not in _ALLOWED_ARTIFACT_KINDS:
            raise ValidationError(f"artifact_kind is not supported: {artifact_kind}")
        object.__setattr__(self, "artifact_kind", artifact_kind)
        if not isinstance(self.timeline, AudioTimelineProvenance):
            raise ValidationError("timeline must be an AudioTimelineProvenance")

    @classmethod
    def from_recording(
        cls,
        recording: CanonicalRecording,
        *,
        artifact_id: str,
        artifact_kind: ArtifactKind = "fixture",
        local_audio_sha256: str | None = None,
    ) -> NormalizedArtifactProvenance:
        return cls(
            artifact_id=artifact_id,
            artifact_kind=artifact_kind,
            timeline=AudioTimelineProvenance.from_recording(recording, local_audio_sha256=local_audio_sha256),
        )

    def to_integrity_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_kind": self.artifact_kind,
            "timeline": self.timeline.to_integrity_dict(),
        }

    def to_rendered_transcript_metadata(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_kind": self.artifact_kind,
            "timeline": self.timeline.to_rendered_transcript_metadata(),
        }

    def to_monitoring_metadata(self) -> dict[str, Any]:
        return {
            "artifact_kind": self.artifact_kind,
            "timeline": self.timeline.to_monitoring_metadata(),
        }

    def to_cross_session_linking_metadata(self) -> dict[str, Any]:
        return {
            "artifact_kind": self.artifact_kind,
            "timeline": self.timeline.to_cross_session_linking_metadata(),
        }


@dataclass(frozen=True)
class OffsetMapSegment:
    """A constant-offset source-to-target timeline segment."""

    source_start_ms: int
    source_end_ms: int
    target_start_ms: int
    target_end_ms: int

    def __post_init__(self) -> None:
        source_start_ms, source_end_ms = _validate_interval(
            self.source_start_ms,
            self.source_end_ms,
            "offset_map_segment.source",
        )
        target_start_ms, target_end_ms = _validate_interval(
            self.target_start_ms,
            self.target_end_ms,
            "offset_map_segment.target",
        )
        if source_end_ms - source_start_ms != target_end_ms - target_start_ms:
            raise ValidationError("offset map segments must preserve duration")
        object.__setattr__(self, "source_start_ms", source_start_ms)
        object.__setattr__(self, "source_end_ms", source_end_ms)
        object.__setattr__(self, "target_start_ms", target_start_ms)
        object.__setattr__(self, "target_end_ms", target_end_ms)

    @property
    def offset_ms(self) -> int:
        return self.target_start_ms - self.source_start_ms

    def contains_source_ms(self, source_ms: int) -> bool:
        return self.source_start_ms <= source_ms < self.source_end_ms

    def convert_source_ms(self, source_ms: int) -> int:
        if not self.contains_source_ms(source_ms):
            raise ValidationError("source timestamp is outside offset map segment")
        return source_ms + self.offset_ms


@dataclass(frozen=True)
class TimelineOffsetMap:
    """Validated hook for merging artifacts from offset timelines."""

    offset_map_id: str
    source_timeline_id: str
    target_timeline_id: str
    source_transform_chain_id: str
    target_transform_chain_id: str
    source_time_basis: TimeBasis
    target_time_basis: TimeBasis
    segments: tuple[OffsetMapSegment, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "offset_map_id", _require_id(self.offset_map_id, "offset_map_id"))
        object.__setattr__(self, "source_timeline_id", _require_id(self.source_timeline_id, "source_timeline_id"))
        object.__setattr__(self, "target_timeline_id", _require_id(self.target_timeline_id, "target_timeline_id"))
        object.__setattr__(
            self,
            "source_transform_chain_id",
            _require_id(self.source_transform_chain_id, "source_transform_chain_id"),
        )
        object.__setattr__(
            self,
            "target_transform_chain_id",
            _require_id(self.target_transform_chain_id, "target_transform_chain_id"),
        )
        object.__setattr__(
            self,
            "source_time_basis",
            _validate_millisecond_time_basis(self.source_time_basis, "source_time_basis"),
        )
        object.__setattr__(
            self,
            "target_time_basis",
            _validate_millisecond_time_basis(self.target_time_basis, "target_time_basis"),
        )
        segments = _as_tuple_of(self.segments, OffsetMapSegment, "offset_map.segments")
        if not segments:
            raise ValidationError("offset_map.segments is required")
        _validate_non_overlapping_segments(segments)
        object.__setattr__(self, "segments", segments)

    def validate_for(self, source: AudioTimelineProvenance, target: AudioTimelineProvenance) -> None:
        _validate_shared_audio_metadata(source, target, require_same_duration=False)
        if self.source_timeline_id != source.timeline_id:
            raise ValidationError("offset map source_timeline_id does not match source timeline")
        if self.target_timeline_id != target.timeline_id:
            raise ValidationError("offset map target_timeline_id does not match target timeline")
        if self.source_transform_chain_id != source.transform_chain_id:
            raise ValidationError("offset map source_transform_chain_id does not match source timeline")
        if self.target_transform_chain_id != target.transform_chain_id:
            raise ValidationError("offset map target_transform_chain_id does not match target timeline")
        if self.source_time_basis != source.time_basis:
            raise ValidationError("offset map source_time_basis does not match source timeline")
        if self.target_time_basis != target.time_basis:
            raise ValidationError("offset map target_time_basis does not match target timeline")
        for segment in self.segments:
            if segment.source_end_ms > source.duration_ms:
                raise ValidationError("offset map segment exceeds source duration")
            if segment.target_end_ms > target.duration_ms:
                raise ValidationError("offset map segment exceeds target duration")
        _validate_source_coverage(self.segments, source.duration_ms)

    def convert_source_ms(self, source_ms: object) -> int:
        source_ms = _require_non_negative_int(source_ms, "source_ms")
        for segment in self.segments:
            if segment.contains_source_ms(source_ms):
                return segment.convert_source_ms(source_ms)
        terminal_segment = max(self.segments, key=lambda segment: segment.source_end_ms)
        if source_ms == terminal_segment.source_end_ms:
            return terminal_segment.target_end_ms
        raise ValidationError("source_ms is not covered by offset map")


@dataclass(frozen=True)
class TimelineMergeValidation:
    """Result of a validated ASR/diarization timeline merge preflight."""

    source_timeline_id: str
    target_timeline_id: str
    offset_map_id: str | None
    direct_timeline_match: bool


def validate_timeline_merge(
    source: AudioTimelineProvenance | NormalizedArtifactProvenance,
    target: AudioTimelineProvenance | NormalizedArtifactProvenance,
    *,
    offset_map: TimelineOffsetMap | None = None,
) -> TimelineMergeValidation:
    source_timeline = _coerce_timeline(source, "source")
    target_timeline = _coerce_timeline(target, "target")

    direct_match = (
        source_timeline.timeline_id == target_timeline.timeline_id
        and source_timeline.transform_chain_id == target_timeline.transform_chain_id
        and source_timeline.time_basis == target_timeline.time_basis
    )
    if direct_match:
        _validate_shared_audio_metadata(source_timeline, target_timeline)
        return TimelineMergeValidation(
            source_timeline_id=source_timeline.timeline_id,
            target_timeline_id=target_timeline.timeline_id,
            offset_map_id=None,
            direct_timeline_match=True,
        )
    if offset_map is None:
        _validate_shared_audio_metadata(source_timeline, target_timeline)
        raise ValidationError("timeline mismatch requires a validated offset map")
    offset_map.validate_for(source_timeline, target_timeline)
    return TimelineMergeValidation(
        source_timeline_id=source_timeline.timeline_id,
        target_timeline_id=target_timeline.timeline_id,
        offset_map_id=offset_map.offset_map_id,
        direct_timeline_match=False,
    )


def boundary_shift_degrades_scoring(
    reference_start_ms: object,
    candidate_start_ms: object,
    *,
    tolerance_ms: int = 250,
) -> bool:
    """Sentinel hook for boundary scoring regressions caused by timeline shifts."""

    reference_start_ms = _require_non_negative_int(reference_start_ms, "reference_start_ms")
    candidate_start_ms = _require_non_negative_int(candidate_start_ms, "candidate_start_ms")
    tolerance_ms = _require_non_negative_int(tolerance_ms, "tolerance_ms")
    return abs(candidate_start_ms - reference_start_ms) > tolerance_ms


def _coerce_timeline(
    value: AudioTimelineProvenance | NormalizedArtifactProvenance,
    field_name: str,
) -> AudioTimelineProvenance:
    if isinstance(value, NormalizedArtifactProvenance):
        return value.timeline
    if isinstance(value, AudioTimelineProvenance):
        return value
    raise ValidationError(f"{field_name} must include AudioTimelineProvenance")


def _content_addressed_transform_id(
    *,
    branch_id: str,
    original_audio_sha256: str,
    canonical_audio_sha256: str,
    transform_config_hash: str,
) -> str:
    payload = json.dumps(
        {
            "branch_id": branch_id,
            "canonical_audio_sha256": canonical_audio_sha256,
            "original_audio_sha256": original_audio_sha256,
            "transform_config_hash": transform_config_hash,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"audio-transform:{branch_id}:{digest[:24]}"


def _validate_shared_audio_metadata(
    left: AudioTimelineProvenance,
    right: AudioTimelineProvenance,
    *,
    require_same_duration: bool = True,
) -> None:
    if left.original_audio_id != right.original_audio_id:
        raise ValidationError("original_audio_id conflicts between artifacts")
    if left.canonical_audio_id != right.canonical_audio_id:
        raise ValidationError("canonical_audio_id conflicts between artifacts")
    if left.sample_rate_hz != right.sample_rate_hz:
        raise ValidationError("sample_rate_hz conflicts between artifacts")
    if require_same_duration and left.duration_ms != right.duration_ms:
        raise ValidationError("duration_ms conflicts between artifacts")
    if left.channel_ids != right.channel_ids:
        raise ValidationError("channel layout conflicts between artifacts")


def _validate_parameters(parameters: object, field_name: str) -> MappingProxyType[str, Any]:
    if not isinstance(parameters, dict):
        raise ValidationError(f"{field_name} must be an object")
    clean: dict[str, Any] = {}
    for key, value in parameters.items():
        if not isinstance(key, str):
            raise ValidationError(f"{field_name} field names must be strings")
        if value is None or isinstance(value, (str, bool, int)):
            clean[key] = value
            continue
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValidationError(f"{field_name}.{key} must be a finite JSON number")
            clean[key] = value
            continue
        raise ValidationError(f"{field_name}.{key} must be a JSON scalar")
    return MappingProxyType(clean)


def _validate_weights(values: object, field_name: str) -> tuple[float, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes, bytearray)):
        raise ValidationError(f"{field_name} must be an iterable")
    try:
        items = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValidationError(f"{field_name} must be an iterable") from exc
    weights: list[float] = []
    for index, value in enumerate(items):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValidationError(f"{field_name}[{index}] must be a number")
        value = float(value)
        if not math.isfinite(value):
            raise ValidationError(f"{field_name}[{index}] must be finite")
        weights.append(value)
    return tuple(weights)


def _validate_sha256(value: object, field_name: str) -> str:
    value = _require_id(value, field_name)
    if len(value) != 64 or any(char not in _HEX_DIGITS for char in value):
        raise ValidationError(f"{field_name} must be a lowercase sha256 hex digest")
    return value


def _without_local_audio_identity(payload: dict[str, Any]) -> dict[str, Any]:
    for key in _LOCAL_ONLY_KEYS:
        payload.pop(key, None)
    return payload


def _validate_non_overlapping_segments(segments: tuple[OffsetMapSegment, ...]) -> None:
    _validate_non_overlapping_ranges(
        tuple((segment.source_start_ms, segment.source_end_ms) for segment in segments),
        "offset_map source segments",
    )
    _validate_non_overlapping_ranges(
        tuple((segment.target_start_ms, segment.target_end_ms) for segment in segments),
        "offset_map target segments",
    )


def _validate_non_overlapping_ranges(ranges: tuple[tuple[int, int], ...], field_name: str) -> None:
    previous_end = -1
    for start_ms, end_ms in sorted(ranges):
        if start_ms < previous_end:
            raise ValidationError(f"{field_name} must not overlap")
        previous_end = end_ms


def _validate_source_coverage(segments: tuple[OffsetMapSegment, ...], source_duration_ms: int) -> None:
    expected_start_ms = 0
    for segment in sorted(segments, key=lambda segment: segment.source_start_ms):
        if segment.source_start_ms != expected_start_ms:
            raise ValidationError("offset map must cover the full source timeline")
        expected_start_ms = segment.source_end_ms
    if expected_start_ms != source_duration_ms:
        raise ValidationError("offset map must cover the full source timeline")


def _validate_interval(start_ms: object, end_ms: object, field_name: str) -> tuple[int, int]:
    start_ms = _require_non_negative_int(start_ms, f"{field_name}_start_ms")
    end_ms = _require_non_negative_int(end_ms, f"{field_name}_end_ms")
    if end_ms <= start_ms:
        raise ValidationError(f"{field_name}_end_ms must be greater than start_ms")
    return start_ms, end_ms


def _as_tuple_of(values: object, item_type: type[Any], field_name: str) -> tuple[Any, ...]:
    try:
        items = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValidationError(f"{field_name} must be an iterable") from exc
    for index, item in enumerate(items):
        if not isinstance(item, item_type):
            raise ValidationError(f"{field_name}[{index}] must be a {item_type.__name__}")
    return items


def _unique_ids(values: object, field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValidationError(f"{field_name} must be an iterable")
    try:
        items = tuple(_require_id(value, field_name) for value in values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValidationError(f"{field_name} must be an iterable") from exc
    if not items:
        raise ValidationError(f"{field_name} is required")
    seen: set[str] = set()
    for item in items:
        if item in seen:
            raise ValidationError(f"duplicate {field_name}: {item}")
        seen.add(item)
    return items


def _validate_time_basis(value: object, field_name: str) -> TimeBasis:
    value = _require_id(value, field_name)
    if value not in _ALLOWED_TIME_BASES:
        raise ValidationError(f"{field_name} is not supported: {value}")
    return value  # type: ignore[return-value]


def _validate_millisecond_time_basis(value: object, field_name: str) -> TimeBasis:
    value = _validate_time_basis(value, field_name)
    if value not in _MILLISECOND_TIME_BASES:
        raise ValidationError(f"offset map {field_name} must be millisecond-based")
    return value


def _optional_id(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_id(value, field_name)


def _require_id(value: object, field_name: str) -> str:
    if value is None:
        raise ValidationError(f"{field_name} is required")
    if not isinstance(value, str):
        raise ValidationError(f"{field_name} must be a string")
    value = value.strip()
    if not value:
        raise ValidationError(f"{field_name} is required")
    return value


def _require_non_negative_int(value: object, field_name: str) -> int:
    value = _require_int(value, field_name)
    if value < 0:
        raise ValidationError(f"{field_name} must be >= 0")
    return value


def _require_positive_int(value: object, field_name: str) -> int:
    value = _require_int(value, field_name)
    if value <= 0:
        raise ValidationError(f"{field_name} must be greater than 0")
    return value


def _require_positive_finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{field_name} must be a number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValidationError(f"{field_name} must be greater than 0")
    return value


def _require_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{field_name} must be an integer")
    return value
