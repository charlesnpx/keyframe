"""Engine output contracts for diarization benchmark candidates."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from keyframe.diarization.models import CanonicalWord, SpeakerSpan, ValidationError
from keyframe.diarization.provenance import NormalizedArtifactProvenance


EngineOutputKind = Literal["word_spans"]
EngineContractStatus = Literal["valid", "invalid"]

_ALLOWED_OUTPUT_KINDS = frozenset({"word_spans"})
_ALLOWED_STATUSES = frozenset({"valid", "invalid"})


@runtime_checkable
class DiarizationEngineAdapter(Protocol):
    """Protocol for normalizing saved engine/provider outputs."""

    adapter_id: str

    def normalize_raw_output(
        self,
        raw_output: dict[str, Any],
        *,
        artifact: NormalizedArtifactProvenance,
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
    ) -> NormalizedEngineOutput:
        if not isinstance(artifact, NormalizedArtifactProvenance):
            raise ValidationError("artifact must be a NormalizedArtifactProvenance")
        payload = _validate_metadata(raw_output, "raw_engine_output")
        output_id = _require_id(payload.get("output_id"), "raw_engine_output.output_id")
        segments = _sequence(payload.get("segments", ()), "raw_engine_output.segments")
        if not segments:
            raise ValidationError("raw_engine_output.segments is required")

        words: list[CanonicalWord] = []
        speaker_spans: list[SpeakerSpan] = []
        raw_speaker_evidence: list[RawSpeakerEvidence] = []
        speaker_refs: dict[tuple[str | None, str], str] = {}
        for index, segment in enumerate(segments):
            segment_payload = _validate_metadata(segment, f"raw_engine_output.segments[{index}]")
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
                start_ms = _require_non_negative_int(
                    word_payload.get("start_ms"),
                    f"raw_engine_output.segments[{index}].words[{word_index}].start_ms",
                )
                end_ms = _require_non_negative_int(
                    word_payload.get("end_ms"),
                    f"raw_engine_output.segments[{index}].words[{word_index}].end_ms",
                )
                if end_ms <= start_ms:
                    raise ValidationError("raw_engine_output word end_ms must be greater than start_ms")
                span_start_ms = start_ms if span_start_ms is None else min(span_start_ms, start_ms)
                span_end_ms = end_ms if span_end_ms is None else max(span_end_ms, end_ms)
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
                words.append(
                    CanonicalWord(
                        word_id=_stable_word_id(output_id, len(words)),
                        text=_require_text(word_payload.get("text"), "raw_engine_output.word.text"),
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
