"""Branch-specific diarization pipeline orchestration.

The branch pipeline layer keeps speaker evidence scoped to one benchmark run.
It merges saved engine outputs into evaluator-ready candidate artifacts without
creating persisted voice profiles, cross-call identities, or reusable speaker
fingerprints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from keyframe.diarization.adapters import (
    BenchmarkExecutionMode,
    BenchmarkRunRecord,
    DatasetCacheConfig,
    create_benchmark_run_record,
)
from keyframe.diarization.bundles import (
    CandidateBundle,
    validate_candidate_bundle_payload,
)
from keyframe.diarization.engines import (
    EngineConfigMetadata,
    NormalizedEngineOutput,
)
from keyframe.diarization.evaluator import DiarizationEvaluationResult
from keyframe.diarization.models import (
    CanonicalRecording,
    CanonicalWord,
    ChannelRecord,
    DisplayLabel,
    SpeakerRecord,
    SpeakerSpan,
    ValidationError,
)
from keyframe.diarization.provenance import AudioTimelineProvenance, NormalizedArtifactProvenance
from keyframe.diarization.rendering import (
    RenderedTranscript,
    RenderedTurn,
    RenderedWord,
    render_transcript,
)
from keyframe.diarization.reports import BenchmarkEvaluationCase
from keyframe.diarization.manifests import DatasetManifest


PipelineBranchId = Literal["separate_tracks", "authenticated_track_metadata"]
PIPELINE_BRANCH_IDS: tuple[PipelineBranchId, ...] = ("separate_tracks", "authenticated_track_metadata")

_PIPELINE_BRANCH_IDS = frozenset(PIPELINE_BRANCH_IDS)
_FORBIDDEN_PIPELINE_FIELDS = frozenset(
    {
        "canonical_audio_id",
        "corpus_identity",
        "corpus_speaker_id",
        "cross_recording_identity",
        "display_label",
        "embedding",
        "evaluator_speaker_map",
        "global_identity",
        "identity_profile",
        "local_audio_sha256",
        "oracle",
        "oracle_metadata",
        "original_audio_id",
        "participant_id",
        "profile_id",
        "reference_speaker_id",
        "role",
        "role_label",
        "speaker_embedding",
        "speaker_ref",
        "voice_embedding",
        "voice_fingerprint",
        "voice_profile",
    }
)
_TRACK_METADATA_CHANNEL_FIELDS = frozenset({"channel_id", "track_name"})
_PRODUCT_CHANNEL_FIELDS = frozenset({"channel_id"})


@dataclass(frozen=True)
class BranchPipelineResult:
    """Evaluator-ready output plus report-safe branch provenance."""

    branch_id: PipelineBranchId
    candidate_bundle_id: str
    output: NormalizedEngineOutput
    recording: CanonicalRecording
    engine_output_ids: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "branch_id", _validate_pipeline_branch_id(self.branch_id))
        object.__setattr__(
            self,
            "candidate_bundle_id",
            _require_id(self.candidate_bundle_id, "pipeline_result.candidate_bundle_id"),
        )
        if not isinstance(self.output, NormalizedEngineOutput):
            raise ValidationError("pipeline_result.output must be a NormalizedEngineOutput")
        if not isinstance(self.recording, CanonicalRecording):
            raise ValidationError("pipeline_result.recording must be a CanonicalRecording")
        self.output.artifact.timeline.assert_consistent_with_recording(self.recording)
        object.__setattr__(
            self,
            "engine_output_ids",
            _unique_tuple_of_ids(self.engine_output_ids, "pipeline_result.engine_output_ids"),
        )
        metadata = _validate_json_object(self.metadata, "pipeline_result.metadata")
        _reject_forbidden_pipeline_fields(metadata, "pipeline_result.metadata")
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, Any]:
        """Return report-safe branch metadata without local audio or speaker identity."""

        return {
            "artifact": self.output.artifact.to_rendered_transcript_metadata(),
            "branch_id": self.branch_id,
            "candidate_bundle_id": self.candidate_bundle_id,
            "engine_output_ids": list(self.engine_output_ids),
            "metadata": _thaw_json(self.metadata),
            "output_id": self.output.output_id,
            "recording_id": self.recording.recording_id,
        }


def run_separate_track_branch(
    candidate_bundle: CandidateBundle,
    engine_outputs: tuple[NormalizedEngineOutput, ...],
    *,
    output_id: str | None = None,
) -> BranchPipelineResult:
    """Merge ASR-per-track outputs on the canonical timeline."""

    return _run_track_merge_branch(
        "separate_tracks",
        candidate_bundle,
        engine_outputs,
        output_id=output_id,
    )


def run_authenticated_track_metadata_branch(
    candidate_bundle: CandidateBundle,
    engine_outputs: tuple[NormalizedEngineOutput, ...],
    *,
    output_id: str | None = None,
) -> BranchPipelineResult:
    """Merge per-track outputs when benchmark mode explicitly exposes track names."""

    return _run_track_merge_branch(
        "authenticated_track_metadata",
        candidate_bundle,
        engine_outputs,
        output_id=output_id,
    )


def render_branch_transcript(
    result: BranchPipelineResult,
    *,
    candidate_bundle: CandidateBundle | None = None,
    **render_options: Any,
) -> RenderedTranscript:
    """Render a branch transcript, applying track labels only for metadata mode."""

    if not isinstance(result, BranchPipelineResult):
        raise ValidationError("result must be a BranchPipelineResult")
    transcript = render_transcript(result.recording, **render_options)
    if result.branch_id != "authenticated_track_metadata":
        if candidate_bundle is not None:
            validate_pipeline_branch_candidate_inputs(result.branch_id, candidate_bundle)
        return transcript
    if candidate_bundle is None:
        raise ValidationError("authenticated_track_metadata rendering requires the candidate bundle")
    validate_pipeline_branch_candidate_inputs("authenticated_track_metadata", candidate_bundle)
    return _render_with_track_metadata(transcript, _track_labels_by_channel(candidate_bundle))


def validate_pipeline_branch_payload(payload: dict[str, Any]) -> None:
    """Validate a serialized candidate-visible branch invocation payload."""

    data = _validate_json_object(payload, "pipeline_branch_payload")
    _reject_forbidden_pipeline_fields(data, "pipeline_branch_payload")
    branch_id = _validate_pipeline_branch_id(data.get("branch_id"))
    candidate_bundle_payload = data.get("candidate_bundle")
    if not isinstance(candidate_bundle_payload, dict):
        raise ValidationError("pipeline_branch_payload.candidate_bundle must be an object")
    validate_pipeline_branch_candidate_inputs(branch_id, candidate_bundle_payload)


def validate_pipeline_branch_candidate_inputs(
    branch_id: PipelineBranchId,
    candidate_bundle: CandidateBundle | dict[str, Any],
) -> None:
    """Apply branch-specific candidate-visible input policy."""

    branch_id = _validate_pipeline_branch_id(branch_id)
    payload = _candidate_bundle_payload(candidate_bundle)
    mode = _require_id(payload.get("mode"), "candidate_bundle.mode")
    channels = _candidate_channels(payload)
    if branch_id == "separate_tracks":
        if mode != "product_realistic":
            raise ValidationError("separate_tracks branch requires product_realistic candidate bundle mode")
        _validate_channel_field_whitelist(channels, _PRODUCT_CHANNEL_FIELDS, "separate_tracks")
        return
    if mode != "authenticated_track_metadata":
        raise ValidationError(
            "authenticated_track_metadata branch requires authenticated_track_metadata candidate bundle mode"
        )
    _validate_channel_field_whitelist(channels, _TRACK_METADATA_CHANNEL_FIELDS, "authenticated_track_metadata")
    _track_labels_by_channel_payload(channels)


def create_pipeline_branch_run_record(
    *,
    run_id: str,
    manifest: DatasetManifest,
    split_id: str,
    branch_id: PipelineBranchId,
    artifact_root: str | Path,
    cache: DatasetCacheConfig,
    tuned_split_ids: tuple[str, ...] = (),
    evaluated_split_ids: tuple[str, ...] = (),
    execution_mode: BenchmarkExecutionMode = "default_no_network",
    derived_artifacts: dict[str, str] | None = None,
) -> BenchmarkRunRecord:
    """Create a run record whose branch field is a validated pipeline branch ID."""

    branch_id = _validate_pipeline_branch_id(branch_id)
    return create_benchmark_run_record(
        run_id=run_id,
        manifest=manifest,
        split_id=split_id,
        branch=branch_id,
        artifact_root=artifact_root,
        cache=cache,
        tuned_split_ids=tuned_split_ids,
        evaluated_split_ids=evaluated_split_ids,
        execution_mode=execution_mode,
        derived_artifacts=derived_artifacts,
    )


def build_pipeline_branch_evaluation_case(
    *,
    corpus_id: str,
    result: BranchPipelineResult,
    evaluation: DiarizationEvaluationResult,
    baseline_evaluation: DiarizationEvaluationResult | None = None,
    scored_duration_ms: int = 0,
    scored_words: int = 0,
    scored_speaker_turns: int = 0,
    slice_scored_words: dict[str, int] | None = None,
    slice_scored_speaker_turns: dict[str, int] | None = None,
) -> BenchmarkEvaluationCase:
    """Attach the pipeline branch ID to a report evaluation case."""

    if not isinstance(result, BranchPipelineResult):
        raise ValidationError("result must be a BranchPipelineResult")
    if not isinstance(evaluation, DiarizationEvaluationResult):
        raise ValidationError("evaluation must be a DiarizationEvaluationResult")
    if evaluation.output_id != result.output.output_id:
        raise ValidationError("evaluation.output_id must match pipeline result output_id")
    return BenchmarkEvaluationCase(
        corpus_id=corpus_id,
        branch_id=result.branch_id,
        evaluation=evaluation,
        baseline_evaluation=baseline_evaluation,
        scored_duration_ms=scored_duration_ms,
        scored_words=scored_words,
        scored_speaker_turns=scored_speaker_turns,
        slice_scored_words=slice_scored_words,
        slice_scored_speaker_turns=slice_scored_speaker_turns,
    )


def _run_track_merge_branch(
    branch_id: PipelineBranchId,
    candidate_bundle: CandidateBundle,
    engine_outputs: tuple[NormalizedEngineOutput, ...],
    *,
    output_id: str | None,
) -> BranchPipelineResult:
    branch_id = _validate_pipeline_branch_id(branch_id)
    if not isinstance(candidate_bundle, CandidateBundle):
        raise ValidationError("candidate_bundle must be a CandidateBundle")
    validate_pipeline_branch_candidate_inputs(branch_id, candidate_bundle)
    outputs = _validate_track_engine_outputs(candidate_bundle, engine_outputs)
    result_output_id = _require_id(
        output_id or f"{candidate_bundle.bundle_id}:{branch_id}",
        "pipeline.output_id",
    )
    channel_payloads = tuple(candidate_bundle.to_dict()["channels"])
    merged_words, merged_spans, speaker_refs = _merge_track_outputs(branch_id, result_output_id, outputs)
    artifact = _branch_output_artifact(result_output_id, candidate_bundle, outputs)
    config = _branch_engine_config(branch_id, candidate_bundle, outputs)
    output = NormalizedEngineOutput(
        output_id=result_output_id,
        output_kind="word_spans",
        artifact=artifact,
        config=config,
        words=merged_words,
        speaker_spans=merged_spans,
        raw_speaker_evidence=(),
    )
    recording = CanonicalRecording(
        recording_id=result_output_id,
        original_audio_id=artifact.timeline.original_audio_id,
        canonical_audio_id=artifact.timeline.canonical_audio_id,
        timeline_id=artifact.timeline.timeline_id,
        duration_ms=artifact.timeline.duration_ms,
        transform_chain_id=artifact.timeline.transform_chain_id,
        sample_rate_hz=artifact.timeline.sample_rate_hz,
        time_basis=artifact.timeline.time_basis,
        channels=tuple(
            ChannelRecord(
                channel_id=channel["channel_id"],
                name=channel.get("track_name") if branch_id == "authenticated_track_metadata" else None,
            )
            for channel in channel_payloads
        ),
        speakers=tuple(SpeakerRecord(speaker_ref) for speaker_ref in speaker_refs),
        words=merged_words,
        speaker_spans=merged_spans,
        scoring_regions=(),
    )
    return BranchPipelineResult(
        branch_id=branch_id,
        candidate_bundle_id=candidate_bundle.bundle_id,
        output=output,
        recording=recording,
        engine_output_ids=tuple(output.output_id for output in outputs),
        metadata={
            "candidate_bundle_mode": candidate_bundle.mode,
            "merge_strategy": "asr_per_track_timeline_sort",
            "speaker_scope": "session_local",
        },
    )


def _merge_track_outputs(
    branch_id: PipelineBranchId,
    output_id: str,
    outputs: tuple[NormalizedEngineOutput, ...],
) -> tuple[tuple[CanonicalWord, ...], tuple[SpeakerSpan, ...], tuple[str, ...]]:
    speaker_refs: dict[tuple[str, str], str] = {}
    channel_counts: dict[str, int] = {}
    word_rows: list[tuple[int, int, int, CanonicalWord]] = []
    span_rows: list[tuple[int, int, int, SpeakerSpan]] = []

    for output_index, output in enumerate(outputs):
        channel_id = _single_output_channel(output)
        for word_index, word in enumerate(output.words):
            channel_id = _word_channel_for_track(word, output)
            speaker_ref = _mapped_speaker_ref(branch_id, channel_id, word.speaker_ref, speaker_refs, channel_counts)
            word_rows.append(
                (
                    word.start_ms,
                    output_index,
                    word_index,
                    replace(
                        word,
                        word_id="pending",
                        speaker_ref=speaker_ref,
                        channel_id=channel_id,
                        display_label=None,
                    ),
                )
            )
        for span_index, span in enumerate(output.speaker_spans):
            channel_id = _span_channel_for_track(span, output)
            speaker_ref = _mapped_speaker_ref(branch_id, channel_id, span.speaker_ref, speaker_refs, channel_counts)
            span_rows.append(
                (
                    span.start_ms,
                    output_index,
                    span_index,
                    replace(
                        span,
                        span_id="pending",
                        speaker_ref=speaker_ref,
                        channel_id=channel_id,
                    ),
                )
            )

    ordered_words = tuple(
        replace(word, word_id=_stable_word_id(output_id, index))
        for index, (*_, word) in enumerate(sorted(word_rows, key=lambda row: (row[0], row[1], row[2])))
    )
    ordered_spans = tuple(
        replace(span, span_id=_stable_span_id(output_id, index))
        for index, (*_, span) in enumerate(sorted(span_rows, key=lambda row: (row[0], row[1], row[2])))
    )
    speaker_order = _speaker_order(ordered_words, ordered_spans)
    return ordered_words, ordered_spans, speaker_order


def _validate_track_engine_outputs(
    candidate_bundle: CandidateBundle,
    engine_outputs: tuple[NormalizedEngineOutput, ...],
) -> tuple[NormalizedEngineOutput, ...]:
    try:
        outputs = tuple(engine_outputs)
    except TypeError as exc:
        raise ValidationError("engine_outputs must be an iterable") from exc
    if not outputs:
        raise ValidationError("engine_outputs is required")
    candidate_payload = candidate_bundle.to_dict()
    candidate_timeline = candidate_payload["runtime_hints"]["timeline"]
    candidate_channel_ids = tuple(candidate_payload["runtime_hints"]["channel_ids"])
    covered_channels: set[str] = set()
    first_timeline = None
    for index, output in enumerate(outputs):
        if not isinstance(output, NormalizedEngineOutput):
            raise ValidationError(f"engine_outputs[{index}] must be a NormalizedEngineOutput")
        channel_id = _single_output_channel(output)
        if channel_id not in candidate_channel_ids:
            raise ValidationError(f"engine_outputs[{index}] channel_id is not candidate-visible: {channel_id}")
        covered_channels.add(channel_id)
        timeline = output.artifact.timeline
        if first_timeline is None:
            first_timeline = timeline
        elif (
            timeline.original_audio_id != first_timeline.original_audio_id
            or timeline.canonical_audio_id != first_timeline.canonical_audio_id
        ):
            raise ValidationError("engine output audio identity conflicts across tracks")
        if timeline.timeline_id != candidate_timeline["timeline_id"]:
            raise ValidationError("engine output timeline_id conflicts with candidate bundle")
        if timeline.transform_chain_id != candidate_timeline["transform_chain_id"]:
            raise ValidationError("engine output transform_chain_id conflicts with candidate bundle")
        if timeline.duration_ms != candidate_timeline["duration_ms"]:
            raise ValidationError("engine output duration_ms conflicts with candidate bundle")
        if timeline.sample_rate_hz != candidate_timeline["sample_rate_hz"]:
            raise ValidationError("engine output sample_rate_hz conflicts with candidate bundle")
        if timeline.time_basis != candidate_timeline["time_basis"]:
            raise ValidationError("engine output time_basis conflicts with candidate bundle")
        for word in output.words:
            _word_channel_for_track(word, output)
        for span in output.speaker_spans:
            _span_channel_for_track(span, output)
    missing = tuple(channel_id for channel_id in candidate_channel_ids if channel_id not in covered_channels)
    if missing:
        raise ValidationError(f"engine_outputs missing per-track output for channel: {missing[0]}")
    return outputs


def _branch_output_artifact(
    output_id: str,
    candidate_bundle: CandidateBundle,
    outputs: tuple[NormalizedEngineOutput, ...],
) -> NormalizedArtifactProvenance:
    candidate_payload = candidate_bundle.to_dict()
    candidate_timeline = candidate_payload["runtime_hints"]["timeline"]
    first_timeline = outputs[0].artifact.timeline
    return NormalizedArtifactProvenance(
        artifact_id=f"{output_id}:artifact",
        artifact_kind="candidate",
        timeline=AudioTimelineProvenance(
            original_audio_id=first_timeline.original_audio_id,
            canonical_audio_id=first_timeline.canonical_audio_id,
            timeline_id=candidate_timeline["timeline_id"],
            transform_chain_id=candidate_timeline["transform_chain_id"],
            sample_rate_hz=candidate_timeline["sample_rate_hz"],
            duration_ms=candidate_timeline["duration_ms"],
            channel_ids=tuple(candidate_payload["runtime_hints"]["channel_ids"]),
            time_basis=candidate_timeline["time_basis"],
        ),
    )


def _branch_engine_config(
    branch_id: PipelineBranchId,
    candidate_bundle: CandidateBundle,
    outputs: tuple[NormalizedEngineOutput, ...],
) -> EngineConfigMetadata:
    return EngineConfigMetadata(
        adapter_id=f"pipeline-{branch_id}",
        provider="keyframe",
        model_name=f"{branch_id}-timeline-merge",
        parameters={
            "branch_id": branch_id,
            "candidate_bundle_id": candidate_bundle.bundle_id,
            "engine_output_ids": [output.output_id for output in outputs],
            "merge_strategy": "asr_per_track_timeline_sort",
            "speaker_scope": "session_local",
        },
    )


def _render_with_track_metadata(
    transcript: RenderedTranscript,
    labels_by_channel: dict[str, DisplayLabel],
) -> RenderedTranscript:
    return replace(
        transcript,
        turns=tuple(_rendered_turn_with_channel_label(turn, labels_by_channel) for turn in transcript.turns),
        words=tuple(_rendered_word_with_channel_label(word, labels_by_channel) for word in transcript.words),
    )


def _rendered_turn_with_channel_label(
    turn: RenderedTurn,
    labels_by_channel: dict[str, DisplayLabel],
) -> RenderedTurn:
    if turn.channel_id is None:
        raise ValidationError("authenticated track metadata turns require channel_id")
    label = labels_by_channel.get(turn.channel_id)
    if label is None:
        raise ValidationError(f"missing track metadata for channel: {turn.channel_id}")
    return replace(turn, label=label.label, display_label=label)


def _rendered_word_with_channel_label(
    word: RenderedWord,
    labels_by_channel: dict[str, DisplayLabel],
) -> RenderedWord:
    if word.channel_id is None:
        raise ValidationError("authenticated track metadata words require channel_id")
    label = labels_by_channel.get(word.channel_id)
    if label is None:
        raise ValidationError(f"missing track metadata for channel: {word.channel_id}")
    return replace(word, label=label.label, display_label=label)


def _candidate_bundle_payload(candidate_bundle: CandidateBundle | dict[str, Any]) -> dict[str, Any]:
    if isinstance(candidate_bundle, CandidateBundle):
        payload = candidate_bundle.to_dict()
    elif isinstance(candidate_bundle, dict):
        payload = _validate_json_object(candidate_bundle, "candidate_bundle")
        validate_candidate_bundle_payload(payload)
    else:
        raise ValidationError("candidate_bundle must be a CandidateBundle or payload object")
    _reject_forbidden_pipeline_fields(payload, "candidate_bundle")
    return payload


def _candidate_channels(payload: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    value = payload.get("channels")
    if not isinstance(value, list):
        raise ValidationError("candidate_bundle.channels must be a list")
    return tuple(_validate_json_object(channel, "candidate_bundle.channels[]") for channel in value)


def _validate_channel_field_whitelist(
    channels: tuple[dict[str, Any], ...],
    allowed_fields: frozenset[str],
    branch_id: str,
) -> None:
    for channel in channels:
        unknown = sorted(set(channel) - allowed_fields)
        if unknown:
            raise ValidationError(f"{branch_id} branch channel metadata is not permitted: {unknown[0]}")


def _track_labels_by_channel(candidate_bundle: CandidateBundle) -> dict[str, DisplayLabel]:
    payload = candidate_bundle.to_dict()
    return _track_labels_by_channel_payload(tuple(payload["channels"]))


def _track_labels_by_channel_payload(channels: tuple[dict[str, Any], ...]) -> dict[str, DisplayLabel]:
    result: dict[str, DisplayLabel] = {}
    seen_labels: set[str] = set()
    for channel in channels:
        channel_id = _require_id(channel.get("channel_id"), "candidate_bundle.channels[].channel_id")
        track_name = _require_text(channel.get("track_name"), "candidate_bundle.channels[].track_name")
        if track_name in seen_labels:
            raise ValidationError(f"duplicate authenticated track metadata label: {track_name}")
        seen_labels.add(track_name)
        result[channel_id] = DisplayLabel(
            label=track_name,
            source="channel_metadata",
            scope="recording",
            confidence=1.0,
            source_ref=channel_id,
        )
    return result


def _single_output_channel(output: NormalizedEngineOutput) -> str:
    channel_ids = output.artifact.timeline.channel_ids
    if len(channel_ids) != 1:
        raise ValidationError("separate-track pipeline requires one channel per engine output")
    return channel_ids[0]


def _word_channel_for_track(word: CanonicalWord, output: NormalizedEngineOutput) -> str:
    output_channel = _single_output_channel(output)
    if word.channel_id is None:
        return output_channel
    if word.channel_id != output_channel:
        raise ValidationError("engine output word channel_id conflicts with per-track artifact channel")
    return word.channel_id


def _span_channel_for_track(span: SpeakerSpan, output: NormalizedEngineOutput) -> str:
    output_channel = _single_output_channel(output)
    if span.channel_id is None:
        return output_channel
    if span.channel_id != output_channel:
        raise ValidationError("engine output span channel_id conflicts with per-track artifact channel")
    return span.channel_id


def _mapped_speaker_ref(
    branch_id: PipelineBranchId,
    channel_id: str,
    speaker_ref: str | None,
    speaker_refs: dict[tuple[str, str], str],
    channel_counts: dict[str, int],
) -> str | None:
    if speaker_ref is None:
        return None
    key = (channel_id, speaker_ref)
    if key not in speaker_refs:
        channel_counts[channel_id] = channel_counts.get(channel_id, 0) + 1
        speaker_refs[key] = f"{branch_id}:{_sanitize_ref_part(channel_id)}:speaker_{channel_counts[channel_id]}"
    return speaker_refs[key]


def _speaker_order(
    words: tuple[CanonicalWord, ...],
    spans: tuple[SpeakerSpan, ...],
) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for speaker_ref in (
        *(word.speaker_ref for word in words if word.speaker_ref is not None),
        *(span.speaker_ref for span in spans),
    ):
        if speaker_ref in seen:
            continue
        seen.add(speaker_ref)
        result.append(speaker_ref)
    return tuple(result)


def _stable_word_id(output_id: str, index: int) -> str:
    return f"{output_id}:word:{index + 1:06d}"


def _stable_span_id(output_id: str, index: int) -> str:
    return f"{output_id}:span:{index + 1:06d}"


def _sanitize_ref_part(value: str) -> str:
    result = []
    for char in value.strip().lower():
        result.append(char if char.isalnum() else "-")
    return "".join(result).strip("-") or "unknown"


def _reject_forbidden_pipeline_fields(payload: object, path: str) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in _FORBIDDEN_PIPELINE_FIELDS:
                raise ValidationError(f"{path}.{key} is forbidden in pipeline branch inputs")
            _reject_forbidden_pipeline_fields(value, f"{path}.{key}")
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            _reject_forbidden_pipeline_fields(value, f"{path}[{index}]")


def _validate_pipeline_branch_id(value: object) -> PipelineBranchId:
    value = _require_id(value, "pipeline_branch.branch_id")
    if value not in _PIPELINE_BRANCH_IDS:
        raise ValidationError(f"pipeline branch is not supported: {value}")
    return value  # type: ignore[return-value]


def _unique_tuple_of_ids(values: object, field_name: str) -> tuple[str, ...]:
    try:
        result = tuple(_require_id(value, field_name) for value in values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValidationError(f"{field_name} must be an iterable") from exc
    if not result:
        raise ValidationError(f"{field_name} is required")
    seen: set[str] = set()
    for value in result:
        if value in seen:
            raise ValidationError(f"{field_name} contains duplicate value: {value}")
        seen.add(value)
    return result


def _validate_json_object(value: object, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{field_name} must be an object")
    return {key: _validate_json_value(key, item, field_name) for key, item in value.items()}


def _validate_json_value(key: object, value: object, field_name: str) -> Any:
    if not isinstance(key, str):
        raise ValidationError(f"{field_name} field names must be strings")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError(f"{field_name}.{key} must be a finite JSON number")
        return value
    if isinstance(value, list):
        return [_validate_json_value(key, item, field_name) for item in value]
    if isinstance(value, dict):
        return _validate_json_object(value, f"{field_name}.{key}")
    raise ValidationError(f"{field_name}.{key} must be JSON-compatible")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    if isinstance(value, list):
        return [_thaw_json(item) for item in value]
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


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{field_name} must be a string")
    if not value.strip():
        raise ValidationError(f"{field_name} is required")
    return value
