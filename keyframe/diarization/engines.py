"""Engine output contracts for diarization benchmark candidates."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from keyframe.diarization.models import CanonicalWord, SpeakerSpan, ValidationError
from keyframe.diarization.provenance import NormalizedArtifactProvenance, TimelineOffsetMap, validate_timeline_merge


EngineOutputKind = Literal["word_spans"]
EngineContractStatus = Literal["valid", "invalid"]
EngineRuntimeSupportStatus = Literal["available", "unsupported"]
HostedProviderKind = Literal["aws_transcribe", "google_speech", "deepgram"]

_ALLOWED_OUTPUT_KINDS = frozenset({"word_spans"})
_ALLOWED_STATUSES = frozenset({"valid", "invalid"})
_ALLOWED_RUNTIME_SUPPORT_STATUSES = frozenset({"available", "unsupported"})
_ALLOWED_HOSTED_PROVIDERS = frozenset({"aws_transcribe", "google_speech", "deepgram"})
_DEFAULT_WHISPERX_DEPENDENCIES = {
    "pyannote.audio": "pyannote.audio",
    "whisperx": "whisperx",
}
_WORD_DERIVED_TURN_GAP_MS = 900


@runtime_checkable
class DiarizationEngineAdapter(Protocol):
    """Protocol for normalizing saved engine/provider outputs."""

    adapter_id: str

    def normalize_raw_output(
        self,
        raw_output: dict[str, Any],
        *,
        artifact: NormalizedArtifactProvenance,
        source_artifact: NormalizedArtifactProvenance | None = None,
        transform_offset_map: TimelineOffsetMap | None = None,
    ) -> "NormalizedEngineOutput":
        """Convert one saved raw engine output into the canonical candidate contract."""

    def describe_config(self) -> "EngineConfigMetadata":
        """Return stable model/config metadata for the engine invocation."""

    def validate_contract(self, output: "NormalizedEngineOutput") -> "EngineContractValidation":
        """Fail closed when output is insufficient for diarization scoring."""


@dataclass(frozen=True)
class EngineConfigMetadata:
    """Model/config metadata attached to normalized engine outputs."""

    adapter_id: str
    provider: str
    model_name: str
    model_version: str | None = None
    config_id: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "adapter_id", _require_id(self.adapter_id, "engine_config.adapter_id"))
        object.__setattr__(self, "provider", _require_id(self.provider, "engine_config.provider"))
        object.__setattr__(self, "model_name", _require_id(self.model_name, "engine_config.model_name"))
        object.__setattr__(
            self,
            "model_version",
            _optional_text(self.model_version, "engine_config.model_version"),
        )
        object.__setattr__(self, "config_id", _optional_text(self.config_id, "engine_config.config_id"))
        object.__setattr__(self, "parameters", _validate_metadata(self.parameters, "engine_config.parameters"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelArtifactGovernance:
    """Model artifact and runtime access metadata for optional self-hosted engines."""

    checkpoint: str | None = None
    package_versions: dict[str, str] = field(default_factory=dict)
    runtime_config: dict[str, Any] = field(default_factory=dict)
    accepted_terms: tuple[str, ...] = ()
    registry_source: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint", _optional_text(self.checkpoint, "model_governance.checkpoint"))
        object.__setattr__(
            self,
            "package_versions",
            _validate_string_map(self.package_versions, "model_governance.package_versions"),
        )
        object.__setattr__(
            self,
            "runtime_config",
            _validate_metadata(self.runtime_config, "model_governance.runtime_config"),
        )
        object.__setattr__(self, "accepted_terms", _tuple_of_text(self.accepted_terms, "model_governance.terms"))
        object.__setattr__(
            self,
            "registry_source",
            _optional_text(self.registry_source, "model_governance.registry_source"),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "accepted_terms": list(self.accepted_terms),
            "package_versions": dict(self.package_versions),
            "runtime_config": dict(self.runtime_config),
        }
        if self.checkpoint is not None:
            payload["checkpoint"] = self.checkpoint
        if self.registry_source is not None:
            payload["registry_source"] = self.registry_source
        return payload


@dataclass(frozen=True)
class HostedProviderGovernance:
    """Governance metadata for saved hosted provider outputs."""

    provider: HostedProviderKind
    region: str | None = None
    model_version: str | None = None
    version_pinning: str | None = None
    retention_policy: str | None = None
    raw_json_export: str | None = None
    terms_constraints: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        provider = _require_id(self.provider, "hosted_governance.provider")
        if provider not in _ALLOWED_HOSTED_PROVIDERS:
            raise ValidationError(f"hosted_governance.provider is not supported: {provider}")
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "region", _optional_text(self.region, "hosted_governance.region"))
        object.__setattr__(
            self,
            "model_version",
            _optional_text(self.model_version, "hosted_governance.model_version"),
        )
        object.__setattr__(
            self,
            "version_pinning",
            _optional_text(self.version_pinning, "hosted_governance.version_pinning"),
        )
        object.__setattr__(
            self,
            "retention_policy",
            _optional_text(self.retention_policy, "hosted_governance.retention_policy"),
        )
        object.__setattr__(
            self,
            "raw_json_export",
            _optional_text(self.raw_json_export, "hosted_governance.raw_json_export"),
        )
        object.__setattr__(
            self,
            "terms_constraints",
            _tuple_of_text(self.terms_constraints, "hosted_governance.terms_constraints"),
        )
        object.__setattr__(self, "parameters", _validate_metadata(self.parameters, "hosted_governance.parameters"))

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "parameters": dict(self.parameters),
            "provider": self.provider,
            "terms_constraints": list(self.terms_constraints),
        }
        if self.region is not None:
            payload["region"] = self.region
        if self.model_version is not None:
            payload["model_version"] = self.model_version
        if self.version_pinning is not None:
            payload["version_pinning"] = self.version_pinning
        if self.retention_policy is not None:
            payload["retention_policy"] = self.retention_policy
        if self.raw_json_export is not None:
            payload["raw_json_export"] = self.raw_json_export
        return payload


@dataclass(frozen=True)
class EngineRuntimeStatus:
    """Preflight status for optional local engine execution."""

    adapter_id: str
    status: EngineRuntimeSupportStatus
    missing_packages: tuple[str, ...] = ()
    package_versions: dict[str, str] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()
    requires_model_access: bool = False
    requires_gpu: bool = False
    cache_root: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "adapter_id", _require_id(self.adapter_id, "runtime_status.adapter_id"))
        status = _require_id(self.status, "runtime_status.status")
        if status not in _ALLOWED_RUNTIME_SUPPORT_STATUSES:
            raise ValidationError(f"runtime_status.status is not supported: {status}")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "missing_packages", _tuple_of_text(self.missing_packages, "runtime_status.missing"))
        object.__setattr__(
            self,
            "package_versions",
            _validate_string_map(self.package_versions, "runtime_status.package_versions"),
        )
        object.__setattr__(self, "reasons", _tuple_of_text(self.reasons, "runtime_status.reasons"))
        object.__setattr__(
            self,
            "requires_model_access",
            _require_bool(self.requires_model_access, "runtime_status.requires_model_access"),
        )
        object.__setattr__(
            self,
            "requires_gpu",
            _require_bool(self.requires_gpu, "runtime_status.requires_gpu"),
        )
        object.__setattr__(self, "cache_root", _optional_text(self.cache_root, "runtime_status.cache_root"))
        if self.status == "available" and (self.missing_packages or self.reasons):
            raise ValidationError("available runtime status cannot include missing packages or reasons")
        if self.status == "unsupported" and not (self.missing_packages or self.reasons):
            raise ValidationError("unsupported runtime status requires missing packages or reasons")

    @property
    def available(self) -> bool:
        return self.status == "available"

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "cache_root": self.cache_root,
            "missing_packages": list(self.missing_packages),
            "package_versions": dict(self.package_versions),
            "reasons": list(self.reasons),
            "requires_gpu": self.requires_gpu,
            "requires_model_access": self.requires_model_access,
            "status": self.status,
        }


@dataclass(frozen=True)
class RawSpeakerEvidence:
    """Raw speaker evidence preserved without promoting it to a display label."""

    raw_speaker_id: str
    speaker_ref: str
    channel_id: str | None = None
    source_field: str = "speaker"

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_speaker_id", _require_id(self.raw_speaker_id, "speaker_evidence.raw_speaker_id"))
        object.__setattr__(self, "speaker_ref", _require_id(self.speaker_ref, "speaker_evidence.speaker_ref"))
        object.__setattr__(self, "channel_id", _optional_id(self.channel_id, "speaker_evidence.channel_id"))
        object.__setattr__(self, "source_field", _require_id(self.source_field, "speaker_evidence.source_field"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NormalizedEngineOutput:
    """Canonical candidate output normalized from one engine/provider payload."""

    output_id: str
    output_kind: EngineOutputKind
    artifact: NormalizedArtifactProvenance
    config: EngineConfigMetadata
    words: tuple[CanonicalWord, ...]
    speaker_spans: tuple[SpeakerSpan, ...]
    raw_speaker_evidence: tuple[RawSpeakerEvidence, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_id", _require_id(self.output_id, "engine_output.output_id"))
        output_kind = _require_id(self.output_kind, "engine_output.output_kind")
        if output_kind not in _ALLOWED_OUTPUT_KINDS:
            raise ValidationError(f"engine_output.output_kind is not supported: {output_kind}")
        object.__setattr__(self, "output_kind", output_kind)
        if not isinstance(self.artifact, NormalizedArtifactProvenance):
            raise ValidationError("engine_output.artifact must be a NormalizedArtifactProvenance")
        if not isinstance(self.config, EngineConfigMetadata):
            raise ValidationError("engine_output.config must be an EngineConfigMetadata")
        object.__setattr__(self, "words", _tuple_of(self.words, CanonicalWord, "engine_output.words"))
        object.__setattr__(
            self,
            "speaker_spans",
            _tuple_of(self.speaker_spans, SpeakerSpan, "engine_output.speaker_spans"),
        )
        object.__setattr__(
            self,
            "raw_speaker_evidence",
            _tuple_of(self.raw_speaker_evidence, RawSpeakerEvidence, "engine_output.raw_speaker_evidence"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact": self.artifact.to_integrity_dict(),
            "config": self.config.to_dict(),
            "output_id": self.output_id,
            "output_kind": self.output_kind,
            "raw_speaker_evidence": [item.to_dict() for item in self.raw_speaker_evidence],
            "speaker_spans": [span.to_dict() for span in self.speaker_spans],
            "words": [word.to_dict() for word in self.words],
        }


@dataclass(frozen=True)
class EngineContractValidation:
    """Validation result for normalized engine output contracts."""

    status: EngineContractStatus
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        status = _require_id(self.status, "engine_contract.status")
        if status not in _ALLOWED_STATUSES:
            raise ValidationError(f"engine_contract.status is not supported: {status}")
        object.__setattr__(self, "status", status)
        errors = _tuple_of_text(self.errors, "engine_contract.errors")
        object.__setattr__(self, "errors", errors)
        if self.status == "valid" and errors:
            raise ValidationError("valid engine contract results cannot include errors")
        if self.status == "invalid" and not errors:
            raise ValidationError("invalid engine contract results must include errors")

    @property
    def valid(self) -> bool:
        return self.status == "valid"

    def to_dict(self) -> dict[str, Any]:
        return {"errors": list(self.errors), "status": self.status}


class CannedJsonEngineAdapter:
    """Adapter for small saved JSON outputs used by engine contract tests."""

    def __init__(self, config: EngineConfigMetadata) -> None:
        if not isinstance(config, EngineConfigMetadata):
            raise ValidationError("config must be an EngineConfigMetadata")
        self._config = config
        self.adapter_id = config.adapter_id

    def describe_config(self) -> EngineConfigMetadata:
        return self._config

    def normalize_raw_output(
        self,
        raw_output: dict[str, Any],
        *,
        artifact: NormalizedArtifactProvenance,
        source_artifact: NormalizedArtifactProvenance | None = None,
        transform_offset_map: TimelineOffsetMap | None = None,
    ) -> NormalizedEngineOutput:
        if not isinstance(artifact, NormalizedArtifactProvenance):
            raise ValidationError("artifact must be a NormalizedArtifactProvenance")
        active_offset_map = None
        if transform_offset_map is not None and source_artifact is None:
            raise ValidationError("transform_offset_map requires source_artifact")
        if source_artifact is not None:
            merge_validation = validate_timeline_merge(source_artifact, artifact, offset_map=transform_offset_map)
            if merge_validation.offset_map_id is not None:
                active_offset_map = transform_offset_map
        payload = _validate_metadata(raw_output, "raw_engine_output")
        output_id = _require_id(payload.get("output_id"), "raw_engine_output.output_id")
        segments = _sequence(payload.get("segments", ()), "raw_engine_output.segments")
        if not segments:
            raise ValidationError("raw_engine_output.segments is required")

        words: list[CanonicalWord] = []
        speaker_spans: list[SpeakerSpan] = []
        raw_speaker_evidence: list[RawSpeakerEvidence] = []
        speaker_refs: dict[tuple[str | None, str], str] = {}
        final_event_ids: set[str] = set()
        seen_word_keys: set[tuple[str, int, int, str, str | None]] = set()
        for index, segment in enumerate(segments):
            segment_payload = _validate_metadata(segment, f"raw_engine_output.segments[{index}]")
            event_status = _event_status(segment_payload)
            event_id = _optional_id(segment_payload.get("event_id"), f"raw_engine_output.segments[{index}].event_id")
            if event_status == "partial":
                continue
            if event_status == "final" and event_id is not None:
                if event_id in final_event_ids:
                    continue
                final_event_ids.add(event_id)
            if "words" not in segment_payload:
                raise ValidationError("raw_engine_output.segments[].words is required")
            channel_id = _optional_id(segment_payload.get("channel_id"), f"raw_engine_output.segments[{index}].channel_id")
            raw_speaker_id = _optional_id(
                segment_payload.get("speaker_id"),
                f"raw_engine_output.segments[{index}].speaker_id",
            )
            segment_words = _sequence(segment_payload.get("words"), f"raw_engine_output.segments[{index}].words")
            if not segment_words:
                raise ValidationError("raw_engine_output.segments[].words is required")
            speaker_ref = None
            if raw_speaker_id is not None:
                speaker_ref = _speaker_ref_for(channel_id, raw_speaker_id, speaker_refs)
                raw_speaker_evidence.append(
                    RawSpeakerEvidence(
                        raw_speaker_id=raw_speaker_id,
                        speaker_ref=speaker_ref,
                        channel_id=channel_id,
                    )
                )
            span_start_ms: int | None = None
            span_end_ms: int | None = None
            for word_index, word in enumerate(segment_words):
                word_payload = _validate_metadata(
                    word,
                    f"raw_engine_output.segments[{index}].words[{word_index}]",
                )
                start_ms, end_ms = _canonical_word_interval(
                    word_payload,
                    segment_payload=segment_payload,
                    artifact=artifact,
                    offset_map=active_offset_map,
                    field_name=f"raw_engine_output.segments[{index}].words[{word_index}]",
                )
                if end_ms <= start_ms:
                    raise ValidationError("raw_engine_output word end_ms must be greater than start_ms")
                text = _require_text(word_payload.get("text"), "raw_engine_output.word.text")
                word_speaker_id = _optional_id(
                    word_payload.get("speaker_id", raw_speaker_id),
                    f"raw_engine_output.segments[{index}].words[{word_index}].speaker_id",
                )
                word_speaker_ref = None
                if word_speaker_id is not None:
                    word_speaker_ref = _speaker_ref_for(channel_id, word_speaker_id, speaker_refs)
                    if word_speaker_ref != speaker_ref:
                        raw_speaker_evidence.append(
                            RawSpeakerEvidence(
                                raw_speaker_id=word_speaker_id,
                                speaker_ref=word_speaker_ref,
                                channel_id=channel_id,
                            )
                        )
                word_key = (channel_id or "", start_ms, end_ms, text, word_speaker_ref)
                if word_key in seen_word_keys:
                    continue
                seen_word_keys.add(word_key)
                span_start_ms = start_ms if span_start_ms is None else min(span_start_ms, start_ms)
                span_end_ms = end_ms if span_end_ms is None else max(span_end_ms, end_ms)
                words.append(
                    CanonicalWord(
                        word_id=_stable_word_id(output_id, len(words)),
                        text=text,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        speaker_ref=word_speaker_ref,
                        channel_id=channel_id,
                        text_confidence=_optional_confidence(
                            word_payload.get("text_confidence"),
                            "raw_engine_output.word.text_confidence",
                        ),
                        speaker_confidence=_optional_confidence(
                            word_payload.get("speaker_confidence"),
                            "raw_engine_output.word.speaker_confidence",
                        ),
                    )
                )
            if speaker_ref is not None and span_start_ms is not None and span_end_ms is not None:
                speaker_spans.append(
                    SpeakerSpan(
                        span_id=_stable_span_id(output_id, len(speaker_spans)),
                        speaker_ref=speaker_ref,
                        start_ms=span_start_ms,
                        end_ms=span_end_ms,
                        channel_id=channel_id,
                        confidence=_optional_confidence(
                            segment_payload.get("speaker_confidence"),
                            f"raw_engine_output.segments[{index}].speaker_confidence",
                        ),
                    )
                )

        output = NormalizedEngineOutput(
            output_id=output_id,
            output_kind="word_spans",
            artifact=artifact,
            config=self.describe_config(),
            words=tuple(words),
            speaker_spans=tuple(speaker_spans),
            raw_speaker_evidence=_dedupe_evidence(tuple(raw_speaker_evidence)),
        )
        validation = self.validate_contract(output)
        if not validation.valid:
            raise ValidationError("; ".join(validation.errors))
        return output

    def validate_contract(self, output: NormalizedEngineOutput) -> EngineContractValidation:
        if not isinstance(output, NormalizedEngineOutput):
            raise ValidationError("output must be a NormalizedEngineOutput")
        errors: list[str] = []
        if not output.words:
            errors.append("engine output must include word-level timestamps")
        if not any(word.speaker_ref is not None for word in output.words) and not output.speaker_spans:
            errors.append("engine output must include speaker attribution")
        if output.words and not any(word.channel_id is not None for word in output.words):
            errors.append("engine output must preserve channel evidence")
        return EngineContractValidation("invalid" if errors else "valid", errors=tuple(errors))


class SelfHostedWhisperXPyannoteAdapter:
    """Adapter for saved WhisperX ASR/alignment and pyannote diarization outputs."""

    adapter_id = "self-hosted-whisperx-pyannote"

    def __init__(
        self,
        governance: ModelArtifactGovernance,
        *,
        model_version: str | None = None,
        config_id: str | None = None,
    ) -> None:
        if not isinstance(governance, ModelArtifactGovernance):
            raise ValidationError("governance must be a ModelArtifactGovernance")
        self._governance = governance
        self._config = EngineConfigMetadata(
            adapter_id=self.adapter_id,
            provider="self-hosted",
            model_name="whisperx+pyannote",
            model_version=model_version,
            config_id=config_id,
            parameters={
                "model_governance": governance.to_dict(),
                "normalization": {
                    "diarization_source": "pyannote",
                    "input_format": "whisperx_json",
                    "time_unit": "seconds_or_ms",
                },
            },
        )

    def describe_config(self) -> EngineConfigMetadata:
        return self._config

    def runtime_preflight(
        self,
        *,
        dependency_modules: Mapping[str, str] | None = None,
        cache_root: str | None = None,
    ) -> EngineRuntimeStatus:
        dependencies = _DEFAULT_WHISPERX_DEPENDENCIES if dependency_modules is None else dict(dependency_modules)
        missing: list[str] = []
        package_versions = dict(self._governance.package_versions)
        for module_name, distribution_name in dependencies.items():
            if _module_available(module_name):
                version = _distribution_version(distribution_name)
                if version is not None:
                    package_versions.setdefault(distribution_name, version)
            else:
                missing.append(distribution_name)

        runtime_config = self._governance.runtime_config
        configured_cache_root = cache_root or _optional_text(
            runtime_config.get("cache_root"),
            "model_governance.runtime_config.cache_root",
        )
        allow_download = _runtime_bool(runtime_config, "allow_download")
        requires_gpu = _runtime_bool(runtime_config, "requires_gpu")
        requires_model_access = self._governance.checkpoint is not None or self._governance.registry_source is not None
        reasons: list[str] = []
        if missing:
            reasons.append(
                "Install optional runtime packages before local execution: " + ", ".join(sorted(missing))
            )
        if requires_model_access and not self._governance.accepted_terms:
            reasons.append("Record accepted model terms or use a local checkpoint before runtime execution")
        if not configured_cache_root and not allow_download:
            reasons.append("Set model_governance.runtime_config.cache_root or allow_download=true for local execution")

        status: EngineRuntimeSupportStatus = "unsupported" if missing or reasons else "available"
        return EngineRuntimeStatus(
            adapter_id=self.adapter_id,
            cache_root=configured_cache_root,
            missing_packages=tuple(sorted(missing)),
            package_versions=package_versions,
            reasons=tuple(reasons),
            requires_gpu=requires_gpu,
            requires_model_access=requires_model_access,
            status=status,
        )

    def normalize_raw_output(
        self,
        raw_output: dict[str, Any],
        *,
        artifact: NormalizedArtifactProvenance,
        source_artifact: NormalizedArtifactProvenance | None = None,
        transform_offset_map: TimelineOffsetMap | None = None,
    ) -> NormalizedEngineOutput:
        if not isinstance(artifact, NormalizedArtifactProvenance):
            raise ValidationError("artifact must be a NormalizedArtifactProvenance")
        active_offset_map = None
        if transform_offset_map is not None and source_artifact is None:
            raise ValidationError("transform_offset_map requires source_artifact")
        if source_artifact is not None:
            merge_validation = validate_timeline_merge(source_artifact, artifact, offset_map=transform_offset_map)
            if merge_validation.offset_map_id is not None:
                active_offset_map = transform_offset_map

        payload = _validate_metadata(raw_output, "whisperx_output")
        output_id = _optional_id(payload.get("output_id"), "whisperx_output.output_id") or (
            f"{artifact.artifact_id}:whisperx-pyannote"
        )
        speaker_refs: dict[tuple[str | None, str], str] = {}
        raw_speaker_evidence: list[RawSpeakerEvidence] = []
        diarization_rows = _whisper_diarization_rows(
            payload,
            artifact=artifact,
            offset_map=active_offset_map,
            speaker_refs=speaker_refs,
            raw_speaker_evidence=raw_speaker_evidence,
        )
        words = _whisper_words(
            payload,
            output_id=output_id,
            artifact=artifact,
            offset_map=active_offset_map,
            diarization_rows=diarization_rows,
            speaker_refs=speaker_refs,
            raw_speaker_evidence=raw_speaker_evidence,
        )
        if diarization_rows:
            speaker_spans = tuple(
                SpeakerSpan(
                    span_id=_stable_span_id(output_id, index),
                    speaker_ref=row["speaker_ref"],
                    start_ms=row["start_ms"],
                    end_ms=row["end_ms"],
                    channel_id=row["channel_id"],
                    confidence=row["confidence"],
                )
                for index, row in enumerate(diarization_rows)
            )
        else:
            speaker_spans = _spans_from_words(output_id, words)

        output = NormalizedEngineOutput(
            output_id=output_id,
            output_kind="word_spans",
            artifact=artifact,
            config=self.describe_config(),
            words=tuple(words),
            speaker_spans=speaker_spans,
            raw_speaker_evidence=_dedupe_evidence(tuple(raw_speaker_evidence)),
        )
        validation = self.validate_contract(output)
        if not validation.valid:
            raise ValidationError("; ".join(validation.errors))
        return output

    def validate_contract(self, output: NormalizedEngineOutput) -> EngineContractValidation:
        if not isinstance(output, NormalizedEngineOutput):
            raise ValidationError("output must be a NormalizedEngineOutput")
        errors: list[str] = []
        if not output.words:
            errors.append("whisperx output must include word-level timestamps")
        if not any(word.speaker_ref is not None for word in output.words) and not output.speaker_spans:
            errors.append("whisperx output must include speaker attribution")
        if output.words and not any(word.channel_id is not None for word in output.words):
            errors.append("whisperx output must preserve channel evidence")
        return EngineContractValidation("invalid" if errors else "valid", errors=tuple(errors))


class HostedProviderJsonAdapter:
    """Provider-neutral adapter for saved AWS, Google, and Deepgram-style JSON outputs."""

    def __init__(self, governance: HostedProviderGovernance, *, config_id: str | None = None) -> None:
        if not isinstance(governance, HostedProviderGovernance):
            raise ValidationError("governance must be a HostedProviderGovernance")
        self._governance = governance
        self.adapter_id = f"hosted-provider-{governance.provider}"
        self._config = EngineConfigMetadata(
            adapter_id=self.adapter_id,
            provider=governance.provider,
            model_name=f"{governance.provider}-hosted-transcription",
            model_version=governance.model_version,
            config_id=config_id,
            parameters={"hosted_provider_governance": governance.to_dict()},
        )

    def describe_config(self) -> EngineConfigMetadata:
        return self._config

    def normalize_raw_output(
        self,
        raw_output: dict[str, Any],
        *,
        artifact: NormalizedArtifactProvenance,
        source_artifact: NormalizedArtifactProvenance | None = None,
        transform_offset_map: TimelineOffsetMap | None = None,
    ) -> NormalizedEngineOutput:
        if not isinstance(artifact, NormalizedArtifactProvenance):
            raise ValidationError("artifact must be a NormalizedArtifactProvenance")
        active_offset_map = None
        if transform_offset_map is not None and source_artifact is None:
            raise ValidationError("transform_offset_map requires source_artifact")
        if source_artifact is not None:
            merge_validation = validate_timeline_merge(source_artifact, artifact, offset_map=transform_offset_map)
            if merge_validation.offset_map_id is not None:
                active_offset_map = transform_offset_map
        payload = _validate_metadata(raw_output, "hosted_provider_output")
        output_id = _optional_id(payload.get("output_id"), "hosted_provider_output.output_id") or (
            f"{artifact.artifact_id}:{self._governance.provider}"
        )
        speaker_refs: dict[tuple[str | None, str], str] = {}
        raw_speaker_evidence: list[RawSpeakerEvidence] = []
        if self._governance.provider == "aws_transcribe":
            words = _aws_provider_words(
                payload,
                output_id=output_id,
                artifact=artifact,
                offset_map=active_offset_map,
                speaker_refs=speaker_refs,
                raw_speaker_evidence=raw_speaker_evidence,
            )
        elif self._governance.provider == "google_speech":
            words = _google_provider_words(
                payload,
                output_id=output_id,
                artifact=artifact,
                offset_map=active_offset_map,
                speaker_refs=speaker_refs,
                raw_speaker_evidence=raw_speaker_evidence,
            )
        elif self._governance.provider == "deepgram":
            words = _deepgram_provider_words(
                payload,
                output_id=output_id,
                artifact=artifact,
                offset_map=active_offset_map,
                speaker_refs=speaker_refs,
                raw_speaker_evidence=raw_speaker_evidence,
            )
        else:
            raise ValidationError(f"hosted provider is not supported: {self._governance.provider}")

        output = NormalizedEngineOutput(
            output_id=output_id,
            output_kind="word_spans",
            artifact=artifact,
            config=self.describe_config(),
            words=words,
            speaker_spans=_spans_from_words(output_id, words),
            raw_speaker_evidence=_dedupe_evidence(tuple(raw_speaker_evidence)),
        )
        validation = self.validate_contract(output)
        if not validation.valid:
            raise ValidationError("; ".join(validation.errors))
        return output

    def validate_contract(self, output: NormalizedEngineOutput) -> EngineContractValidation:
        if not isinstance(output, NormalizedEngineOutput):
            raise ValidationError("output must be a NormalizedEngineOutput")
        errors: list[str] = []
        if not output.words:
            errors.append("hosted provider output must include word-level timestamps")
        if not any(word.speaker_ref is not None for word in output.words):
            errors.append("hosted provider output must include speaker metadata")
        if output.words and not any(word.channel_id is not None for word in output.words):
            errors.append("hosted provider output must preserve channel evidence")
        return EngineContractValidation("invalid" if errors else "valid", errors=tuple(errors))


def _speaker_ref_for(
    channel_id: str | None,
    raw_speaker_id: str,
    speaker_refs: dict[tuple[str | None, str], str],
) -> str:
    key = (channel_id, raw_speaker_id)
    if key not in speaker_refs:
        channel_part = "no_channel" if channel_id is None else _sanitize_ref_part(channel_id)
        speaker_part = _sanitize_ref_part(raw_speaker_id)
        base_ref = f"engine:{channel_part}:{speaker_part}"
        speaker_ref = base_ref
        suffix = 2
        used_refs = set(speaker_refs.values())
        while speaker_ref in used_refs:
            speaker_ref = f"{base_ref}-{suffix}"
            suffix += 1
        speaker_refs[key] = speaker_ref
    return speaker_refs[key]


def _event_status(segment_payload: dict[str, Any]) -> str:
    status = segment_payload.get("event_status", segment_payload.get("status", "final"))
    status = _require_id(status, "raw_engine_output.segment.event_status")
    if status not in {"partial", "final"}:
        raise ValidationError(f"raw_engine_output.segment.event_status is not supported: {status}")
    return status


def _canonical_word_interval(
    word_payload: dict[str, Any],
    *,
    segment_payload: dict[str, Any],
    artifact: NormalizedArtifactProvenance,
    offset_map: TimelineOffsetMap | None,
    field_name: str,
) -> tuple[int, int]:
    time_basis = _raw_time_basis(word_payload, segment_payload, artifact)
    start = _raw_timestamp(word_payload, "start", field_name)
    end = _raw_timestamp(word_payload, "end", field_name)
    if end <= start:
        raise ValidationError("raw_engine_output word end must be greater than start")
    if time_basis == "canonical_ms":
        start_ms = int(start)
        end_ms = int(end)
    elif time_basis == "chunk_relative_ms":
        chunk_start_ms = _chunk_start_ms(word_payload, segment_payload, offset_map, field_name)
        start_ms = int(start + chunk_start_ms)
        end_ms = int(end + chunk_start_ms)
    elif time_basis == "sample_index":
        start_ms = round(start * 1000 / artifact.timeline.sample_rate_hz)
        end_ms = round(end * 1000 / artifact.timeline.sample_rate_hz)
    elif time_basis == "frame_index":
        frame_rate = _frame_rate(word_payload, segment_payload, field_name)
        start_ms = round(start * 1000 / frame_rate)
        end_ms = round(end * 1000 / frame_rate)
    else:
        raise ValidationError(f"raw_engine_output time_basis is not supported: {time_basis}")

    if offset_map is not None:
        return offset_map.convert_source_ms(start_ms), offset_map.convert_source_ms(end_ms)
    return start_ms, end_ms


def _raw_time_basis(
    word_payload: dict[str, Any],
    segment_payload: dict[str, Any],
    artifact: NormalizedArtifactProvenance,
) -> str:
    value = word_payload.get("time_basis", segment_payload.get("time_basis", artifact.timeline.time_basis))
    value = _require_id(value, "raw_engine_output.time_basis")
    if value not in {"canonical_ms", "chunk_relative_ms", "sample_index", "frame_index"}:
        raise ValidationError(f"raw_engine_output.time_basis is not supported: {value}")
    return value


def _raw_timestamp(word_payload: dict[str, Any], name: str, field_name: str) -> int:
    key = f"{name}_ms"
    if key not in word_payload:
        key = f"{name}"
    return _require_non_negative_int(word_payload.get(key), f"{field_name}.{key}")


def _chunk_start_ms(
    word_payload: dict[str, Any],
    segment_payload: dict[str, Any],
    offset_map: TimelineOffsetMap | None,
    field_name: str,
) -> int:
    value = word_payload.get("chunk_start_ms", segment_payload.get("chunk_start_ms"))
    if value is not None:
        return _require_non_negative_int(value, f"{field_name}.chunk_start_ms")
    if offset_map is not None and offset_map.source_time_basis == "chunk_relative_ms":
        return 0
    raise ValidationError("chunk_relative_ms requires chunk_start_ms or a chunk-relative transform_offset_map")


def _frame_rate(word_payload: dict[str, Any], segment_payload: dict[str, Any], field_name: str) -> float:
    value = word_payload.get("frame_rate_fps", segment_payload.get("frame_rate_fps"))
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{field_name}.frame_rate_fps must be a number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValidationError(f"{field_name}.frame_rate_fps must be greater than 0")
    return value


def _stable_word_id(output_id: str, index: int) -> str:
    return f"{output_id}:word:{index + 1:06d}"


def _stable_span_id(output_id: str, index: int) -> str:
    return f"{output_id}:span:{index + 1:06d}"


def _sanitize_ref_part(value: str) -> str:
    result = []
    for char in value.strip().lower():
        result.append(char if char.isalnum() else "-")
    return "".join(result).strip("-") or "unknown"


def _dedupe_evidence(values: tuple[RawSpeakerEvidence, ...]) -> tuple[RawSpeakerEvidence, ...]:
    seen: set[tuple[str, str | None, str]] = set()
    result: list[RawSpeakerEvidence] = []
    for value in values:
        key = (value.raw_speaker_id, value.channel_id, value.speaker_ref)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return tuple(result)


def _aws_provider_words(
    payload: dict[str, Any],
    *,
    output_id: str,
    artifact: NormalizedArtifactProvenance,
    offset_map: TimelineOffsetMap | None,
    speaker_refs: dict[tuple[str | None, str], str],
    raw_speaker_evidence: list[RawSpeakerEvidence],
) -> tuple[CanonicalWord, ...]:
    results = _validate_metadata(payload.get("results"), "hosted_provider_output.results")
    items = _sequence(results.get("items"), "hosted_provider_output.results.items")
    speaker_labels = _aws_speaker_labels_by_interval(results, offset_map=offset_map)
    words: list[CanonicalWord] = []
    for index, item in enumerate(items):
        context = f"hosted_provider_output.results.items[{index}]"
        value = _validate_metadata(item, context)
        if value.get("type", "pronunciation") != "pronunciation":
            continue
        channel_id = _provider_channel_id(payload, value, artifact, context)
        start_ms, end_ms = _provider_interval_ms(value, context, offset_map)
        text = _aws_word_text(value, context)
        raw_speaker_id = _provider_speaker_id(
            value,
            context,
            keys=("speaker_label", "speaker", "speaker_id"),
        ) or speaker_labels.get((start_ms, end_ms))
        speaker_ref = _provider_speaker_ref(
            raw_speaker_id,
            channel_id=channel_id,
            speaker_refs=speaker_refs,
            raw_speaker_evidence=raw_speaker_evidence,
            source_field="speaker_label",
        )
        words.append(
            CanonicalWord(
                word_id=_stable_word_id(output_id, len(words)),
                text=text,
                start_ms=start_ms,
                end_ms=end_ms,
                speaker_ref=speaker_ref,
                channel_id=channel_id,
                text_confidence=_aws_word_confidence(value, context),
            )
        )
    return tuple(words)


def _google_provider_words(
    payload: dict[str, Any],
    *,
    output_id: str,
    artifact: NormalizedArtifactProvenance,
    offset_map: TimelineOffsetMap | None,
    speaker_refs: dict[tuple[str | None, str], str],
    raw_speaker_evidence: list[RawSpeakerEvidence],
) -> tuple[CanonicalWord, ...]:
    results = _sequence(payload.get("results"), "hosted_provider_output.results")
    word_rows: dict[tuple[str, int, int], dict[str, Any]] = {}
    word_order: list[tuple[str, int, int]] = []
    for result_index, result_item in enumerate(results):
        result = _validate_metadata(result_item, f"hosted_provider_output.results[{result_index}]")
        event_status = _provider_event_status(result)
        if event_status == "partial":
            continue
        alternatives = _sequence(
            result.get("alternatives"),
            f"hosted_provider_output.results[{result_index}].alternatives",
        )
        if not alternatives:
            raise ValidationError("hosted_provider_output.results[].alternatives is required")
        alternative = _validate_metadata(
            alternatives[0],
            f"hosted_provider_output.results[{result_index}].alternatives[0]",
        )
        alternative_confidence = _first_confidence(
            alternative,
            ("confidence",),
            f"hosted_provider_output.results[{result_index}].alternatives[0]",
        )
        word_items = _sequence(
            alternative.get("words"),
            f"hosted_provider_output.results[{result_index}].alternatives[0].words",
        )
        for word_index, item in enumerate(word_items):
            context = f"hosted_provider_output.results[{result_index}].alternatives[0].words[{word_index}]"
            value = _validate_metadata(item, context)
            channel_id = _provider_channel_id(payload, result, artifact, context)
            start_ms, end_ms = _provider_interval_ms(value, context, offset_map)
            raw_speaker_id = _provider_speaker_id(value, context, keys=("speakerTag", "speaker_tag", "speaker"))
            speaker_ref = _provider_speaker_ref(
                raw_speaker_id,
                channel_id=channel_id,
                speaker_refs=speaker_refs,
                raw_speaker_evidence=raw_speaker_evidence,
                source_field="speakerTag",
            )
            word_confidence = _first_confidence(value, ("confidence", "text_confidence"), context)
            key = (channel_id, start_ms, end_ms)
            if key not in word_rows:
                word_order.append(key)
            word_rows[key] = {
                "channel_id": channel_id,
                "end_ms": end_ms,
                "speaker_confidence": _first_confidence(value, ("speaker_confidence",), context),
                "speaker_ref": speaker_ref,
                "start_ms": start_ms,
                "text": _require_text(value.get("word"), f"{context}.word"),
                "text_confidence": word_confidence if word_confidence is not None else alternative_confidence,
            }
    return tuple(
        CanonicalWord(word_id=_stable_word_id(output_id, index), **word_rows[key])
        for index, key in enumerate(word_order)
    )


def _deepgram_provider_words(
    payload: dict[str, Any],
    *,
    output_id: str,
    artifact: NormalizedArtifactProvenance,
    offset_map: TimelineOffsetMap | None,
    speaker_refs: dict[tuple[str | None, str], str],
    raw_speaker_evidence: list[RawSpeakerEvidence],
) -> tuple[CanonicalWord, ...]:
    results = _validate_metadata(payload.get("results"), "hosted_provider_output.results")
    channels = _sequence(results.get("channels"), "hosted_provider_output.results.channels")
    words: list[CanonicalWord] = []
    for channel_index, channel_item in enumerate(channels):
        channel = _validate_metadata(channel_item, f"hosted_provider_output.results.channels[{channel_index}]")
        channel_id = _provider_channel_id(
            payload,
            channel,
            artifact,
            f"hosted_provider_output.results.channels[{channel_index}]",
            provider_channel_index=channel_index,
        )
        alternatives = _sequence(
            channel.get("alternatives"),
            f"hosted_provider_output.results.channels[{channel_index}].alternatives",
        )
        if not alternatives:
            raise ValidationError("hosted_provider_output.results.channels[].alternatives is required")
        alternative = _validate_metadata(
            alternatives[0],
            f"hosted_provider_output.results.channels[{channel_index}].alternatives[0]",
        )
        word_items = _sequence(
            alternative.get("words"),
            f"hosted_provider_output.results.channels[{channel_index}].alternatives[0].words",
        )
        for word_index, item in enumerate(word_items):
            context = f"hosted_provider_output.results.channels[{channel_index}].alternatives[0].words[{word_index}]"
            value = _validate_metadata(item, context)
            start_ms, end_ms = _provider_interval_ms(value, context, offset_map)
            raw_speaker_id = _provider_speaker_id(value, context, keys=("speaker", "speaker_id"))
            speaker_ref = _provider_speaker_ref(
                raw_speaker_id,
                channel_id=channel_id,
                speaker_refs=speaker_refs,
                raw_speaker_evidence=raw_speaker_evidence,
                source_field="speaker",
            )
            words.append(
                CanonicalWord(
                    word_id=_stable_word_id(output_id, len(words)),
                    text=_require_text(value.get("word"), f"{context}.word"),
                    start_ms=start_ms,
                    end_ms=end_ms,
                    speaker_ref=speaker_ref,
                    channel_id=channel_id,
                    text_confidence=_first_confidence(value, ("confidence",), context),
                    speaker_confidence=_first_confidence(value, ("speaker_confidence",), context),
                )
            )
    return tuple(words)


def _aws_word_text(payload: dict[str, Any], context: str) -> str:
    return _require_text(_aws_word_alternative(payload, context).get("content"), f"{context}.alternatives[0].content")


def _aws_word_confidence(payload: dict[str, Any], context: str) -> float | None:
    alternative = _aws_word_alternative(payload, context)
    if "confidence" not in alternative:
        return None
    return _provider_confidence(alternative.get("confidence"), f"{context}.alternatives[0].confidence")


def _aws_word_alternative(payload: dict[str, Any], context: str) -> dict[str, Any]:
    alternatives = _sequence(payload.get("alternatives"), f"{context}.alternatives")
    if not alternatives:
        raise ValidationError(f"{context}.alternatives is required")
    return _validate_metadata(alternatives[0], f"{context}.alternatives[0]")


def _aws_speaker_labels_by_interval(
    results: dict[str, Any],
    *,
    offset_map: TimelineOffsetMap | None,
) -> dict[tuple[int, int], str]:
    value = results.get("speaker_labels")
    if value is None:
        return {}
    speaker_labels = _validate_metadata(value, "hosted_provider_output.results.speaker_labels")
    labels: dict[tuple[int, int], str] = {}
    segments = _sequence(
        speaker_labels.get("segments", ()),
        "hosted_provider_output.results.speaker_labels.segments",
    )
    for segment_index, segment_item in enumerate(segments):
        segment = _validate_metadata(
            segment_item,
            f"hosted_provider_output.results.speaker_labels.segments[{segment_index}]",
        )
        items = _sequence(
            segment.get("items", ()),
            f"hosted_provider_output.results.speaker_labels.segments[{segment_index}].items",
        )
        for item_index, item in enumerate(items):
            context = (
                "hosted_provider_output.results.speaker_labels"
                f".segments[{segment_index}].items[{item_index}]"
            )
            payload = _validate_metadata(item, context)
            start_ms, end_ms = _provider_interval_ms(payload, context, offset_map)
            speaker_label = _require_id(payload.get("speaker_label"), f"{context}.speaker_label")
            labels[(start_ms, end_ms)] = speaker_label
    return labels


def _provider_channel_id(
    root_payload: dict[str, Any],
    payload: dict[str, Any],
    artifact: NormalizedArtifactProvenance,
    context: str,
    *,
    provider_channel_index: int | None = None,
) -> str:
    value = payload.get("channel_id")
    if value is not None:
        return _require_id(str(value), f"{context}.channel_id")
    if "channelTag" in payload:
        index = _provider_channel_ordinal(payload.get("channelTag"), f"{context}.channelTag") - 1
        return _artifact_channel_id(artifact, index, f"{context}.channelTag")
    if "channel_label" in payload:
        return _aws_channel_label_to_artifact_channel(payload.get("channel_label"), artifact, context)
    if provider_channel_index is not None:
        return _artifact_channel_id(artifact, provider_channel_index, context)
    value = root_payload.get("channel_id")
    if value is not None:
        return _require_id(str(value), "hosted_provider_output.channel_id")
    if len(artifact.timeline.channel_ids) == 1:
        return artifact.timeline.channel_ids[0]
    raise ValidationError(f"{context}.channel_id is required for multi-channel artifacts")


def _provider_channel_ordinal(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValidationError(f"{field_name} must be greater than 0")
    return value


def _artifact_channel_id(artifact: NormalizedArtifactProvenance, index: int, field_name: str) -> str:
    if index < 0 or index >= len(artifact.timeline.channel_ids):
        raise ValidationError(f"{field_name} does not map to an artifact channel")
    return artifact.timeline.channel_ids[index]


def _aws_channel_label_to_artifact_channel(
    value: object,
    artifact: NormalizedArtifactProvenance,
    context: str,
) -> str:
    label = _require_id(str(value) if value is not None else None, f"{context}.channel_label")
    if label in artifact.timeline.channel_ids:
        return label
    if label.startswith("ch_"):
        suffix = label.removeprefix("ch_")
        try:
            index = int(suffix)
        except ValueError as exc:
            raise ValidationError(f"{context}.channel_label does not map to an artifact channel") from exc
        return _artifact_channel_id(artifact, index, f"{context}.channel_label")
    raise ValidationError(f"{context}.channel_label does not map to an artifact channel")


def _provider_interval_ms(
    payload: dict[str, Any],
    context: str,
    offset_map: TimelineOffsetMap | None,
) -> tuple[int, int]:
    start_ms = _provider_timestamp_ms(payload, ("start_ms", "startTime", "start_time", "start"), context)
    end_ms = _provider_timestamp_ms(payload, ("end_ms", "endTime", "end_time", "end"), context)
    if end_ms <= start_ms:
        raise ValidationError(f"{context}.end must be greater than start")
    if offset_map is not None:
        return offset_map.convert_source_ms(start_ms), offset_map.convert_source_ms(end_ms)
    return start_ms, end_ms


def _provider_timestamp_ms(payload: dict[str, Any], keys: tuple[str, ...], context: str) -> int:
    for key in keys:
        if key in payload:
            value = payload.get(key)
            if key.endswith("_ms"):
                return _non_negative_time_ms(value, f"{context}.{key}", multiplier=1.0)
            return _hosted_time_to_ms(value, f"{context}.{key}")
    raise ValidationError(f"{context}.{keys[-1]} is required")


def _hosted_time_to_ms(value: object, field_name: str) -> int:
    if isinstance(value, str):
        value = value.strip()
        if value.endswith("s"):
            value = value[:-1]
        try:
            parsed = float(value)
        except ValueError as exc:
            raise ValidationError(f"{field_name} must be a number") from exc
        return _non_negative_time_ms(parsed, field_name, multiplier=1000.0)
    return _non_negative_time_ms(value, field_name, multiplier=1000.0)


def _provider_speaker_id(payload: dict[str, Any], context: str, *, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if key in payload:
            return _optional_id(str(payload.get(key)) if payload.get(key) is not None else None, f"{context}.{key}")
    return None


def _provider_confidence(value: object, field_name: str) -> float | None:
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError as exc:
            raise ValidationError(f"{field_name} must be a number") from exc
    return _optional_confidence(value, field_name)


def _provider_speaker_ref(
    raw_speaker_id: str | None,
    *,
    channel_id: str,
    speaker_refs: dict[tuple[str | None, str], str],
    raw_speaker_evidence: list[RawSpeakerEvidence],
    source_field: str,
) -> str | None:
    if raw_speaker_id is None:
        return None
    speaker_ref = _speaker_ref_for(channel_id, raw_speaker_id, speaker_refs)
    raw_speaker_evidence.append(
        RawSpeakerEvidence(
            raw_speaker_id=raw_speaker_id,
            speaker_ref=speaker_ref,
            channel_id=channel_id,
            source_field=source_field,
        )
    )
    return speaker_ref


def _provider_event_status(payload: dict[str, Any]) -> str:
    is_final = payload.get("isFinal")
    if is_final is None:
        return _event_status(payload)
    return "final" if _require_bool(is_final, "hosted_provider_output.result.isFinal") else "partial"


def _whisper_diarization_rows(
    payload: dict[str, Any],
    *,
    artifact: NormalizedArtifactProvenance,
    offset_map: TimelineOffsetMap | None,
    speaker_refs: dict[tuple[str | None, str], str],
    raw_speaker_evidence: list[RawSpeakerEvidence],
) -> tuple[dict[str, Any], ...]:
    value = payload.get("diarization", payload.get("speaker_spans", ()))
    if value is None:
        return ()
    rows = []
    for index, item in enumerate(_sequence(value, "whisperx_output.diarization")):
        row = _validate_metadata(item, f"whisperx_output.diarization[{index}]")
        channel_id = _whisper_channel_id(payload, row, artifact, f"whisperx_output.diarization[{index}]")
        raw_speaker_id = _whisper_speaker_id(row, f"whisperx_output.diarization[{index}]", required=True)
        start_ms, end_ms = _whisper_interval_ms(row, f"whisperx_output.diarization[{index}]")
        if offset_map is not None:
            start_ms = offset_map.convert_source_ms(start_ms)
            end_ms = offset_map.convert_source_ms(end_ms)
        speaker_ref = _speaker_ref_for(channel_id, raw_speaker_id, speaker_refs)
        raw_speaker_evidence.append(
            RawSpeakerEvidence(
                raw_speaker_id=raw_speaker_id,
                speaker_ref=speaker_ref,
                channel_id=channel_id,
                source_field=_whisper_speaker_source_field(row),
            )
        )
        rows.append(
            {
                "channel_id": channel_id,
                "confidence": _first_confidence(
                    row,
                    ("speaker_confidence", "confidence", "score"),
                    f"whisperx_output.diarization[{index}]",
                ),
                "end_ms": end_ms,
                "raw_speaker_id": raw_speaker_id,
                "speaker_ref": speaker_ref,
                "start_ms": start_ms,
            }
        )
    return tuple(rows)


def _whisper_words(
    payload: dict[str, Any],
    *,
    output_id: str,
    artifact: NormalizedArtifactProvenance,
    offset_map: TimelineOffsetMap | None,
    diarization_rows: tuple[dict[str, Any], ...],
    speaker_refs: dict[tuple[str | None, str], str],
    raw_speaker_evidence: list[RawSpeakerEvidence],
) -> tuple[CanonicalWord, ...]:
    words: list[CanonicalWord] = []
    seen_word_keys: set[tuple[str, int, int, str, str | None]] = set()
    for segment_index, segment_payload in enumerate(_whisper_segments(payload)):
        segment = _validate_metadata(segment_payload, f"whisperx_output.segments[{segment_index}]")
        event_status = _event_status(segment)
        if event_status == "partial":
            continue
        channel_id = _whisper_channel_id(payload, segment, artifact, f"whisperx_output.segments[{segment_index}]")
        segment_speaker_id = _whisper_speaker_id(segment, f"whisperx_output.segments[{segment_index}]")
        word_items = _sequence(segment.get("words"), f"whisperx_output.segments[{segment_index}].words")
        if not word_items:
            raise ValidationError("whisperx_output.segments[].words is required")
        for word_index, word_item in enumerate(word_items):
            context = f"whisperx_output.segments[{segment_index}].words[{word_index}]"
            word_payload = _validate_metadata(word_item, context)
            start_ms, end_ms = _whisper_interval_ms(word_payload, context)
            if offset_map is not None:
                start_ms = offset_map.convert_source_ms(start_ms)
                end_ms = offset_map.convert_source_ms(end_ms)
            text = _whisper_word_text(word_payload, context)
            raw_speaker_id = _whisper_speaker_id(word_payload, context) or segment_speaker_id
            if raw_speaker_id is None:
                raw_speaker_id = _speaker_at_ms(
                    diarization_rows,
                    channel_id=channel_id,
                    midpoint_ms=(start_ms + end_ms) // 2,
                )
            speaker_ref = None
            if raw_speaker_id is not None:
                speaker_ref = _speaker_ref_for(channel_id, raw_speaker_id, speaker_refs)
                raw_speaker_evidence.append(
                    RawSpeakerEvidence(
                        raw_speaker_id=raw_speaker_id,
                        speaker_ref=speaker_ref,
                        channel_id=channel_id,
                        source_field=_whisper_speaker_source_field(word_payload, segment),
                    )
                )
            word_key = (channel_id or "", start_ms, end_ms, text, speaker_ref)
            if word_key in seen_word_keys:
                continue
            seen_word_keys.add(word_key)
            words.append(
                CanonicalWord(
                    word_id=_stable_word_id(output_id, len(words)),
                    text=text,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    speaker_ref=speaker_ref,
                    channel_id=channel_id,
                    text_confidence=_first_confidence(
                        word_payload,
                        ("text_confidence", "score", "probability"),
                        context,
                    ),
                    speaker_confidence=_first_confidence(
                        word_payload,
                        ("speaker_confidence", "speaker_score"),
                        context,
                    ),
                )
            )
    return tuple(words)


def _whisper_segments(payload: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    if "segments" in payload:
        return _sequence(payload.get("segments"), "whisperx_output.segments")
    if "word_segments" in payload:
        return ({"words": payload["word_segments"]},)
    raise ValidationError("whisperx_output.segments or word_segments is required")


def _whisper_channel_id(
    root_payload: dict[str, Any],
    payload: dict[str, Any],
    artifact: NormalizedArtifactProvenance,
    context: str,
) -> str:
    value = payload.get("channel_id", root_payload.get("channel_id"))
    if value is not None:
        return _require_id(value, f"{context}.channel_id")
    if len(artifact.timeline.channel_ids) == 1:
        return artifact.timeline.channel_ids[0]
    raise ValidationError(f"{context}.channel_id is required for multi-channel artifacts")


def _whisper_speaker_id(payload: dict[str, Any], context: str, *, required: bool = False) -> str | None:
    for key in ("speaker", "speaker_id", "label"):
        if key in payload:
            return _optional_id(payload.get(key), f"{context}.{key}")
    if required:
        raise ValidationError(f"{context}.speaker is required")
    return None


def _whisper_speaker_source_field(*payloads: dict[str, Any]) -> str:
    for payload in payloads:
        for key in ("speaker", "speaker_id", "label"):
            if key in payload:
                return key
    return "speaker"


def _whisper_interval_ms(payload: dict[str, Any], context: str) -> tuple[int, int]:
    start_ms = _whisper_timestamp_ms(payload, "start", context)
    end_ms = _whisper_timestamp_ms(payload, "end", context)
    if end_ms <= start_ms:
        raise ValidationError(f"{context}.end must be greater than start")
    return start_ms, end_ms


def _whisper_timestamp_ms(payload: dict[str, Any], name: str, context: str) -> int:
    ms_key = f"{name}_ms"
    if ms_key in payload:
        return _non_negative_time_ms(payload.get(ms_key), f"{context}.{ms_key}", multiplier=1.0)
    if name in payload:
        return _non_negative_time_ms(payload.get(name), f"{context}.{name}", multiplier=1000.0)
    raise ValidationError(f"{context}.{name} is required")


def _non_negative_time_ms(value: object, field_name: str, *, multiplier: float) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{field_name} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise ValidationError(f"{field_name} must be finite")
    if value < 0:
        raise ValidationError(f"{field_name} must be >= 0")
    return round(value * multiplier)


def _whisper_word_text(payload: dict[str, Any], context: str) -> str:
    if "word" in payload:
        return _require_text(payload.get("word"), f"{context}.word")
    return _require_text(payload.get("text"), f"{context}.text")


def _speaker_at_ms(
    diarization_rows: tuple[dict[str, Any], ...],
    *,
    channel_id: str,
    midpoint_ms: int,
) -> str | None:
    for row in diarization_rows:
        if row["channel_id"] == channel_id and row["start_ms"] <= midpoint_ms < row["end_ms"]:
            return row["raw_speaker_id"]
    return None


def _spans_from_words(output_id: str, words: tuple[CanonicalWord, ...]) -> tuple[SpeakerSpan, ...]:
    spans: list[SpeakerSpan] = []
    words_by_channel: dict[str | None, list[CanonicalWord]] = {}
    for word in words:
        words_by_channel.setdefault(word.channel_id, []).append(word)

    for channel_id, channel_words in words_by_channel.items():
        active_speaker: str | None = None
        active_start_ms: int | None = None
        active_end_ms: int | None = None

        def flush_active_span() -> None:
            if active_speaker is not None and active_start_ms is not None and active_end_ms is not None:
                spans.append(
                    SpeakerSpan(
                        span_id=_stable_span_id(output_id, len(spans)),
                        speaker_ref=active_speaker,
                        start_ms=active_start_ms,
                        end_ms=active_end_ms,
                        channel_id=channel_id,
                    )
                )

        for word in sorted(channel_words, key=lambda item: (item.start_ms, item.end_ms)):
            if word.speaker_ref is None:
                flush_active_span()
                active_speaker = None
                active_start_ms = None
                active_end_ms = None
                continue
            if (
                word.speaker_ref != active_speaker
                or active_end_ms is None
                or word.start_ms - active_end_ms > _WORD_DERIVED_TURN_GAP_MS
            ):
                flush_active_span()
                active_speaker = word.speaker_ref
                active_start_ms = word.start_ms
                active_end_ms = word.end_ms
            else:
                active_end_ms = max(active_end_ms, word.end_ms)

        flush_active_span()
    return tuple(spans)


def _first_confidence(payload: dict[str, Any], keys: tuple[str, ...], context: str) -> float | None:
    for key in keys:
        if key in payload:
            return _optional_confidence(payload.get(key), f"{context}.{key}")
    return None


def _runtime_bool(runtime_config: dict[str, Any], key: str) -> bool:
    if key not in runtime_config:
        return False
    return _require_bool(runtime_config[key], f"model_governance.runtime_config.{key}")


def _validate_metadata(value: object, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{field_name} must be an object")
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValidationError(f"{field_name} field names must be strings")
        result[key] = _validate_metadata_value(key, item, field_name)
    return result


def _validate_metadata_value(key: object, value: object, field_name: str) -> Any:
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


def _sequence(value: object, field_name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValidationError(f"{field_name} must be an array")
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
    return tuple(_require_text(value, field_name) for value in _sequence(values, field_name))


def _validate_string_map(value: object, field_name: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValidationError(f"{field_name} must be an object")
    result: dict[str, str] = {}
    for key, item in value.items():
        key = _require_id(key, f"{field_name}.key")
        result[key] = _require_text(item, f"{field_name}.{key}")
    return result


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValidationError(f"{field_name} must be a boolean")
    return value


def _optional_confidence(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{field_name} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise ValidationError(f"{field_name} must be finite")
    if not 0.0 <= value <= 1.0:
        raise ValidationError(f"{field_name} must be between 0.0 and 1.0")
    return value


def _require_non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{field_name} must be an integer")
    if value < 0:
        raise ValidationError(f"{field_name} must be >= 0")
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


def _optional_id(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_id(value, field_name)


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{field_name} must be a string")
    value = value.strip()
    if not value:
        raise ValidationError(f"{field_name} is required")
    return value


def _optional_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_text(value, field_name)


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def _distribution_version(distribution_name: str) -> str | None:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None
