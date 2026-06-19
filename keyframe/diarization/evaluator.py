"""Central diarization evaluator with reference-derived score slices."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

from keyframe.diarization.bundles import ReferenceBundle
from keyframe.diarization.engines import NormalizedEngineOutput
from keyframe.diarization.manifests import (
    ScoringPolicyManifest,
    default_scoring_policy,
    scoring_policy_report_provenance,
)
from keyframe.diarization.models import CanonicalRecording, CanonicalWord, SpeakerSpan, ValidationError


EvaluationSliceStatus = Literal["ready", "insufficient_support"]
EvaluationMetricStatus = Literal["scored", "insufficient_support"]

_SLICE_STATUSES = frozenset({"ready", "insufficient_support"})
_METRIC_STATUSES = frozenset({"scored", "insufficient_support"})
_SHORT_TURN_MAX_MS = 1_500
_LONG_TURN_MIN_MS = 15_000
_MAX_EXACT_ASSIGNMENT_REFERENCES = 16


@dataclass(frozen=True)
class EvaluationInterval:
    """One reference-derived interval used for score slicing."""

    start_ms: int
    end_ms: int
    channel_id: str | None = None

    def __post_init__(self) -> None:
        start_ms = _require_non_negative_int(self.start_ms, "evaluation_interval.start_ms")
        end_ms = _require_positive_int(self.end_ms, "evaluation_interval.end_ms")
        if end_ms <= start_ms:
            raise ValidationError("evaluation_interval.end_ms must be greater than start_ms")
        object.__setattr__(self, "start_ms", start_ms)
        object.__setattr__(self, "end_ms", end_ms)
        object.__setattr__(self, "channel_id", _optional_id(self.channel_id, "evaluation_interval.channel_id"))

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationSliceDefinition:
    """A fixed slice computed from reference data before candidate scoring."""

    slice_id: str
    dimension: str
    value: str
    status: EvaluationSliceStatus
    support_ms: int
    minimum_support_ms: int
    intervals: tuple[EvaluationInterval, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "slice_id", _require_id(self.slice_id, "evaluation_slice.slice_id"))
        object.__setattr__(self, "dimension", _require_id(self.dimension, "evaluation_slice.dimension"))
        object.__setattr__(self, "value", _require_id(self.value, "evaluation_slice.value"))
        status = _require_id(self.status, "evaluation_slice.status")
        if status not in _SLICE_STATUSES:
            raise ValidationError(f"evaluation_slice.status is not supported: {status}")
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "support_ms",
            _require_non_negative_int(self.support_ms, "evaluation_slice.support_ms"),
        )
        object.__setattr__(
            self,
            "minimum_support_ms",
            _require_positive_int(self.minimum_support_ms, "evaluation_slice.minimum_support_ms"),
        )
        intervals = _tuple_of(self.intervals, EvaluationInterval, "evaluation_slice.intervals")
        object.__setattr__(self, "intervals", intervals)
        if self.support_ms != _interval_support_ms(intervals):
            raise ValidationError("evaluation_slice.support_ms must match intervals")
        if self.status == "ready" and self.support_ms < self.minimum_support_ms:
            raise ValidationError("ready evaluation slices must meet minimum support")
        if self.status == "insufficient_support" and self.support_ms >= self.minimum_support_ms:
            raise ValidationError("insufficient evaluation slices must be below minimum support")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "intervals": [interval.to_dict() for interval in self.intervals],
            "minimum_support_ms": self.minimum_support_ms,
            "slice_id": self.slice_id,
            "status": self.status,
            "support_ms": self.support_ms,
            "value": self.value,
        }


@dataclass(frozen=True)
class DiarizationRecordingMetricRow:
    """Recording-level metrics for one candidate output."""

    recording_id: str
    output_id: str
    policy_id: str
    status: EvaluationMetricStatus
    metrics: dict[str, Any]
    speaker_mapping: dict[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "recording_id", _require_id(self.recording_id, "recording_metric.recording_id"))
        object.__setattr__(self, "output_id", _require_id(self.output_id, "recording_metric.output_id"))
        object.__setattr__(self, "policy_id", _require_id(self.policy_id, "recording_metric.policy_id"))
        status = _require_id(self.status, "recording_metric.status")
        if status not in _METRIC_STATUSES:
            raise ValidationError(f"recording_metric.status is not supported: {status}")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "metrics", _validate_json_object(self.metrics, "recording_metric.metrics"))
        object.__setattr__(
            self,
            "speaker_mapping",
            _validate_string_map(self.speaker_mapping, "recording_metric.speaker_mapping"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": _thaw_json_value(self.metrics),
            "output_id": self.output_id,
            "policy_id": self.policy_id,
            "recording_id": self.recording_id,
            "speaker_mapping": dict(self.speaker_mapping),
            "status": self.status,
        }


@dataclass(frozen=True)
class DiarizationSliceMetricRow:
    """Slice-level metrics for one candidate output."""

    recording_id: str
    output_id: str
    policy_id: str
    slice_id: str
    dimension: str
    value: str
    status: EvaluationMetricStatus
    support_ms: int
    minimum_support_ms: int
    metrics: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "recording_id", _require_id(self.recording_id, "slice_metric.recording_id"))
        object.__setattr__(self, "output_id", _require_id(self.output_id, "slice_metric.output_id"))
        object.__setattr__(self, "policy_id", _require_id(self.policy_id, "slice_metric.policy_id"))
        object.__setattr__(self, "slice_id", _require_id(self.slice_id, "slice_metric.slice_id"))
        object.__setattr__(self, "dimension", _require_id(self.dimension, "slice_metric.dimension"))
        object.__setattr__(self, "value", _require_id(self.value, "slice_metric.value"))
        status = _require_id(self.status, "slice_metric.status")
        if status not in _METRIC_STATUSES:
            raise ValidationError(f"slice_metric.status is not supported: {status}")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "support_ms", _require_non_negative_int(self.support_ms, "slice_metric.support_ms"))
        object.__setattr__(
            self,
            "minimum_support_ms",
            _require_positive_int(self.minimum_support_ms, "slice_metric.minimum_support_ms"),
        )
        object.__setattr__(self, "metrics", _validate_json_object(self.metrics, "slice_metric.metrics"))
        if self.status == "scored" and self.support_ms < self.minimum_support_ms:
            raise ValidationError("scored slice metrics must meet minimum support")
        if self.status == "insufficient_support" and self.support_ms >= self.minimum_support_ms:
            raise ValidationError("insufficient slice metrics must be below minimum support")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "metrics": _thaw_json_value(self.metrics),
            "minimum_support_ms": self.minimum_support_ms,
            "output_id": self.output_id,
            "policy_id": self.policy_id,
            "recording_id": self.recording_id,
            "slice_id": self.slice_id,
            "status": self.status,
            "support_ms": self.support_ms,
            "value": self.value,
        }


@dataclass(frozen=True)
class DiarizationEvaluationResult:
    """Complete score artifact for one reference/candidate pair."""

    recording_id: str
    output_id: str
    scoring_policy: dict[str, str]
    speaker_mapping: dict[str, str]
    slices: tuple[EvaluationSliceDefinition, ...]
    recording_metrics: tuple[DiarizationRecordingMetricRow, ...]
    slice_metrics: tuple[DiarizationSliceMetricRow, ...]
    reference_artifact: dict[str, Any]
    candidate_artifact: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "recording_id", _require_id(self.recording_id, "evaluation_result.recording_id"))
        object.__setattr__(self, "output_id", _require_id(self.output_id, "evaluation_result.output_id"))
        object.__setattr__(
            self,
            "scoring_policy",
            _validate_string_map(self.scoring_policy, "evaluation_result.scoring_policy"),
        )
        object.__setattr__(
            self,
            "speaker_mapping",
            _validate_string_map(self.speaker_mapping, "evaluation_result.speaker_mapping"),
        )
        object.__setattr__(
            self,
            "slices",
            _tuple_of(self.slices, EvaluationSliceDefinition, "evaluation_result.slices"),
        )
        object.__setattr__(
            self,
            "recording_metrics",
            _tuple_of(self.recording_metrics, DiarizationRecordingMetricRow, "evaluation_result.recording_metrics"),
        )
        object.__setattr__(
            self,
            "slice_metrics",
            _tuple_of(self.slice_metrics, DiarizationSliceMetricRow, "evaluation_result.slice_metrics"),
        )
        object.__setattr__(
            self,
            "reference_artifact",
            _validate_json_object(self.reference_artifact, "evaluation_result.reference_artifact"),
        )
        object.__setattr__(
            self,
            "candidate_artifact",
            _validate_json_object(self.candidate_artifact, "evaluation_result.candidate_artifact"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_artifact": _thaw_json_value(self.candidate_artifact),
            "output_id": self.output_id,
            "recording_id": self.recording_id,
            "recording_metrics": [row.to_dict() for row in self.recording_metrics],
            "reference_artifact": _thaw_json_value(self.reference_artifact),
            "scoring_policy": dict(self.scoring_policy),
            "slice_metrics": [row.to_dict() for row in self.slice_metrics],
            "slices": [item.to_dict() for item in self.slices],
            "speaker_mapping": dict(self.speaker_mapping),
        }


@dataclass(frozen=True)
class _SpanView:
    speaker_ref: str
    start_ms: int
    end_ms: int
    channel_id: str | None = None
    overlap: bool = False

    @property
    def interval(self) -> EvaluationInterval:
        return EvaluationInterval(self.start_ms, self.end_ms, self.channel_id)


def build_evaluation_slices(
    reference: ReferenceBundle | CanonicalRecording,
    *,
    scoring_policy: ScoringPolicyManifest | None = None,
    minimum_support_ms: int = 1,
) -> tuple[EvaluationSliceDefinition, ...]:
    """Compute fixed score slices from reference evidence only."""

    recording = _reference_recording(reference)
    policy = _scoring_policy(scoring_policy)
    minimum_support_ms = _require_positive_int(minimum_support_ms, "minimum_support_ms")
    scoring_intervals = _scoring_intervals(recording, policy)
    reference_spans = _spans_from_recording(recording)
    atoms = _atomic_intervals(scoring_intervals, reference_spans)
    scoreable_atoms = _policy_scoreable_reference_atoms(atoms, reference_spans, policy)

    overlap_intervals: list[EvaluationInterval] = []
    non_overlap_intervals: list[EvaluationInterval] = []
    speaker_count_intervals: dict[str, list[EvaluationInterval]] = {key: [] for key in ("0", "1", "2", "3_plus")}
    for atom in scoreable_atoms:
        active = _active_spans(reference_spans, atom)
        count = len({span.speaker_ref for span in active})
        has_overlap = count > 1 or any(span.overlap for span in active)
        if has_overlap:
            overlap_intervals.append(atom)
        elif count == 1:
            non_overlap_intervals.append(atom)
        speaker_count_intervals[_speaker_count_bucket(count)].append(atom)

    turn_duration_intervals = {
        "short": _clip_intervals_to_regions(
            _policy_slice_intervals(
                _turn_duration_intervals(reference_spans, scoring_intervals, short=True),
                policy,
            ),
            scoreable_atoms,
        ),
        "long": _clip_intervals_to_regions(
            _policy_slice_intervals(
                _turn_duration_intervals(reference_spans, scoring_intervals, short=False),
                policy,
            ),
            scoreable_atoms,
        ),
    }
    channel_mode = "mono" if len(recording.channels) <= 1 else "multichannel"
    channel_mode_intervals = {"mono": [], "multichannel": []}
    channel_mode_intervals[channel_mode] = list(scoreable_atoms)

    slices = [
        _slice("overlap", "overlap", tuple(overlap_intervals), minimum_support_ms),
        _slice("overlap", "non_overlap", tuple(non_overlap_intervals), minimum_support_ms),
        _slice(
            "speaker_change_boundary",
            "within_collar",
            _clip_intervals_to_regions(
                _policy_slice_intervals(
                    _speaker_change_boundary_intervals(reference_spans, scoring_intervals, policy.collar_ms),
                    policy,
                ),
                scoreable_atoms,
            ),
            minimum_support_ms,
        ),
        _slice("turn_duration", "short", tuple(turn_duration_intervals["short"]), minimum_support_ms),
        _slice("turn_duration", "long", tuple(turn_duration_intervals["long"]), minimum_support_ms),
    ]
    for value in ("0", "1", "2", "3_plus"):
        slices.append(_slice("speaker_count", value, tuple(speaker_count_intervals[value]), minimum_support_ms))
    for value in ("mono", "multichannel"):
        slices.append(_slice("channel_mode", value, tuple(channel_mode_intervals[value]), minimum_support_ms))
    return tuple(sorted(slices, key=lambda item: (item.dimension, item.value)))


def evaluate_diarization_candidate(
    reference: ReferenceBundle,
    candidate: NormalizedEngineOutput,
    *,
    scoring_policy: ScoringPolicyManifest | None = None,
    minimum_slice_support_ms: int = 1,
) -> DiarizationEvaluationResult:
    """Score one candidate against one reference with session-local speaker mapping."""

    if not isinstance(reference, ReferenceBundle):
        raise ValidationError("reference must be a ReferenceBundle")
    if not isinstance(candidate, NormalizedEngineOutput):
        raise ValidationError("candidate must be a NormalizedEngineOutput")
    policy = _scoring_policy(scoring_policy)
    _validate_candidate_timeline(reference.recording, candidate, policy)

    reference_spans = _spans_from_recording(reference.recording)
    candidate_spans = _spans_from_candidate(candidate)
    raw_scoring_intervals = _scoring_intervals(reference.recording, policy)
    scoring_intervals = _apply_scoring_collar(raw_scoring_intervals, reference_spans, policy.collar_ms)
    slices = build_evaluation_slices(
        reference,
        scoring_policy=policy,
        minimum_support_ms=minimum_slice_support_ms,
    )
    boundary_mapping_intervals = tuple(
        interval
        for item in slices
        if item.slice_id == "speaker_change_boundary:within_collar"
        for interval in item.intervals
    )
    mapping_intervals = _normalize_intervals((*scoring_intervals, *boundary_mapping_intervals))
    speaker_mapping = _build_speaker_mapping(reference_spans, candidate_spans, mapping_intervals, policy)
    policy_provenance = scoring_policy_report_provenance(policy)

    recording_metrics = _score_metrics(reference_spans, candidate_spans, scoring_intervals, speaker_mapping, policy)
    recording_status: EvaluationMetricStatus = (
        "scored" if recording_metrics["scored_interval_ms"] > 0 else "insufficient_support"
    )
    recording_row = DiarizationRecordingMetricRow(
        recording_id=reference.recording.recording_id,
        output_id=candidate.output_id,
        policy_id=policy.policy_id,
        status=recording_status,
        metrics=recording_metrics if recording_status == "scored" else {},
        speaker_mapping=speaker_mapping,
    )
    slice_rows: list[DiarizationSliceMetricRow] = []
    for item in slices:
        slice_scoring_intervals = (
            raw_scoring_intervals
            if item.slice_id == "speaker_change_boundary:within_collar"
            else scoring_intervals
        )
        scored_intervals = _clip_intervals_to_regions(item.intervals, slice_scoring_intervals)
        scored_support_ms = _interval_support_ms(scored_intervals)
        row_status: EvaluationMetricStatus = (
            "scored"
            if item.status == "ready" and scored_support_ms >= item.minimum_support_ms
            else "insufficient_support"
        )
        slice_rows.append(
            DiarizationSliceMetricRow(
                recording_id=reference.recording.recording_id,
                output_id=candidate.output_id,
                policy_id=policy.policy_id,
                slice_id=item.slice_id,
                dimension=item.dimension,
                value=item.value,
                status=row_status,
                support_ms=scored_support_ms,
                minimum_support_ms=item.minimum_support_ms,
                metrics=(
                    _score_metrics(reference_spans, candidate_spans, scored_intervals, speaker_mapping, policy)
                    if row_status == "scored"
                    else {}
                ),
            ),
        )
    return DiarizationEvaluationResult(
        recording_id=reference.recording.recording_id,
        output_id=candidate.output_id,
        scoring_policy=policy_provenance,
        speaker_mapping=speaker_mapping,
        slices=slices,
        recording_metrics=(recording_row,),
        slice_metrics=tuple(slice_rows),
        reference_artifact=reference.artifact.to_integrity_dict(),
        candidate_artifact=candidate.artifact.to_integrity_dict(),
    )


def _scoring_policy(scoring_policy: ScoringPolicyManifest | None) -> ScoringPolicyManifest:
    if scoring_policy is None:
        return default_scoring_policy("diagnostic_diarization")
    if not isinstance(scoring_policy, ScoringPolicyManifest):
        raise ValidationError("scoring_policy must be a ScoringPolicyManifest")
    return scoring_policy


def _reference_recording(reference: ReferenceBundle | CanonicalRecording) -> CanonicalRecording:
    if isinstance(reference, ReferenceBundle):
        return reference.recording
    if isinstance(reference, CanonicalRecording):
        return reference
    raise ValidationError("reference must be a ReferenceBundle or CanonicalRecording")


def _validate_candidate_timeline(
    recording: CanonicalRecording,
    candidate: NormalizedEngineOutput,
    policy: ScoringPolicyManifest,
) -> None:
    candidate.artifact.timeline.assert_consistent_with_recording(recording)
    timeline = candidate.artifact.timeline
    if timeline.time_basis != "canonical_ms" or recording.time_basis != "canonical_ms":
        raise ValidationError("central evaluator requires canonical_ms timelines")
    channel_ids = set(timeline.channel_ids)
    require_channels = policy.channel_mode == "per_channel" and len(recording.channels) > 1
    for span in candidate.speaker_spans:
        if require_channels and span.channel_id is None:
            raise ValidationError("per-channel scoring requires candidate speaker span channel_id")
        if span.channel_id is not None and span.channel_id not in channel_ids:
            raise ValidationError("candidate speaker span channel_id conflicts with reference channel layout")
        if span.end_ms > recording.duration_ms:
            raise ValidationError("candidate speaker span ends after reference duration")
    for word in candidate.words:
        if (
            require_channels
            and not candidate.speaker_spans
            and word.speaker_ref is not None
            and word.channel_id is None
        ):
            raise ValidationError("per-channel scoring requires candidate word channel_id")
        if word.channel_id is not None and word.channel_id not in channel_ids:
            raise ValidationError("candidate word channel_id conflicts with reference channel layout")
        if word.end_ms > recording.duration_ms:
            raise ValidationError("candidate word ends after reference duration")


def _scoring_intervals(recording: CanonicalRecording, policy: ScoringPolicyManifest) -> tuple[EvaluationInterval, ...]:
    collapse_channels = policy.channel_mode in {"mono_mix", "rendered_transcript"}
    if policy.uem_regions == "canonical_scoring_regions" and recording.scoring_regions:
        source_intervals = tuple(
            EvaluationInterval(region.start_ms, region.end_ms, region.channel_id)
            for region in recording.scoring_regions
        )
        if collapse_channels:
            _validate_collapsible_channel_uem(source_intervals, recording)
            intervals = tuple(EvaluationInterval(item.start_ms, item.end_ms, None) for item in source_intervals)
        else:
            intervals = source_intervals
    else:
        channel_ids: tuple[str | None, ...] = (
            (None,)
            if collapse_channels
            else tuple(channel.channel_id for channel in recording.channels) or (None,)
        )
        intervals = tuple(EvaluationInterval(0, recording.duration_ms, channel_id) for channel_id in channel_ids)
    return _normalize_intervals(intervals)


def _validate_collapsible_channel_uem(
    intervals: tuple[EvaluationInterval, ...],
    recording: CanonicalRecording,
) -> None:
    channel_ids = tuple(channel.channel_id for channel in recording.channels)
    if not channel_ids:
        return

    shared_coverage = tuple(
        EvaluationInterval(interval.start_ms, interval.end_ms, None)
        for interval in intervals
        if interval.channel_id is None
    )
    expected: tuple[EvaluationInterval, ...] | None = None
    for channel_id in channel_ids:
        coverage = _normalize_intervals(
            (
                *shared_coverage,
                *(
                    EvaluationInterval(interval.start_ms, interval.end_ms, None)
                    for interval in intervals
                    if interval.channel_id == channel_id
                ),
            )
        )
        if expected is None:
            expected = coverage
        elif coverage != expected:
            raise ValidationError(
                "collapsed-channel scoring requires identical canonical scoring regions per channel"
            )


def _spans_from_recording(recording: CanonicalRecording) -> tuple[_SpanView, ...]:
    if recording.speaker_spans:
        return tuple(
            _SpanView(span.speaker_ref, span.start_ms, span.end_ms, span.channel_id, span.overlap)
            for span in recording.speaker_spans
        )
    return _spans_from_words(recording.words)


def _spans_from_candidate(candidate: NormalizedEngineOutput) -> tuple[_SpanView, ...]:
    if candidate.speaker_spans:
        return tuple(
            _SpanView(span.speaker_ref, span.start_ms, span.end_ms, span.channel_id, span.overlap)
            for span in candidate.speaker_spans
        )
    return _spans_from_words(candidate.words)


def _spans_from_words(words: tuple[CanonicalWord, ...]) -> tuple[_SpanView, ...]:
    spans = []
    for word in words:
        if word.speaker_ref is None:
            continue
        spans.append(_SpanView(word.speaker_ref, word.start_ms, word.end_ms, word.channel_id, word.overlap))
    return tuple(spans)


def _atomic_intervals(
    scoring_intervals: tuple[EvaluationInterval, ...],
    *span_groups: tuple[_SpanView, ...],
) -> tuple[EvaluationInterval, ...]:
    atoms: list[EvaluationInterval] = []
    spans = tuple(span for group in span_groups for span in group)
    for region in scoring_intervals:
        boundaries = {region.start_ms, region.end_ms}
        for span in spans:
            if not _channels_match(span.channel_id, region.channel_id):
                continue
            start_ms = max(region.start_ms, span.start_ms)
            end_ms = min(region.end_ms, span.end_ms)
            if end_ms > start_ms:
                boundaries.add(start_ms)
                boundaries.add(end_ms)
        ordered = sorted(boundaries)
        atoms.extend(
            EvaluationInterval(start_ms, end_ms, region.channel_id)
            for start_ms, end_ms in zip(ordered, ordered[1:])
            if end_ms > start_ms
        )
    return tuple(atoms)


def _active_spans(spans: tuple[_SpanView, ...], interval: EvaluationInterval) -> tuple[_SpanView, ...]:
    return tuple(
        span
        for span in spans
        if _channels_match(span.channel_id, interval.channel_id)
        and min(span.end_ms, interval.end_ms) > max(span.start_ms, interval.start_ms)
    )


def _policy_scoreable_reference_atoms(
    atoms: tuple[EvaluationInterval, ...],
    reference_spans: tuple[_SpanView, ...],
    policy: ScoringPolicyManifest,
) -> tuple[EvaluationInterval, ...]:
    if policy.score_overlap:
        return atoms
    return tuple(
        atom
        for atom in atoms
        if not _is_reference_overlap(_active_spans(reference_spans, atom))
    )


def _policy_slice_intervals(
    intervals: tuple[EvaluationInterval, ...],
    policy: ScoringPolicyManifest,
) -> tuple[EvaluationInterval, ...]:
    if policy.channel_mode not in {"mono_mix", "rendered_transcript"}:
        return intervals
    return _normalize_intervals(
        tuple(EvaluationInterval(interval.start_ms, interval.end_ms, None) for interval in intervals)
    )


def _build_speaker_mapping(
    reference_spans: tuple[_SpanView, ...],
    candidate_spans: tuple[_SpanView, ...],
    scoring_intervals: tuple[EvaluationInterval, ...],
    policy: ScoringPolicyManifest,
) -> dict[str, str]:
    weights: dict[tuple[str, str], int] = {}
    scoreable_reference_speakers: set[str] = set()
    for atom in _atomic_intervals(scoring_intervals, reference_spans, candidate_spans):
        active_reference_spans = _active_spans(reference_spans, atom)
        if not policy.score_overlap and _is_reference_overlap(active_reference_spans):
            continue
        active_reference = {span.speaker_ref for span in active_reference_spans}
        scoreable_reference_speakers.update(active_reference)
        active_candidate = {span.speaker_ref for span in _active_spans(candidate_spans, atom)}
        duration_ms = atom.duration_ms
        for candidate_ref in active_candidate:
            for reference_ref in active_reference:
                weights[(candidate_ref, reference_ref)] = weights.get((candidate_ref, reference_ref), 0) + duration_ms
    reference_speakers = tuple(sorted(scoreable_reference_speakers))
    candidate_speakers = tuple(sorted({span.speaker_ref for span in candidate_spans}))
    if len(reference_speakers) > _MAX_EXACT_ASSIGNMENT_REFERENCES:
        raise ValidationError(
            "speaker assignment requires exact maximum-weight matching; "
            "too many reference speakers for the bounded exact matcher"
        )
    return _exact_speaker_mapping(candidate_speakers, reference_speakers, weights)


def _exact_speaker_mapping(
    candidate_speakers: tuple[str, ...],
    reference_speakers: tuple[str, ...],
    weights: dict[tuple[str, str], int],
) -> dict[str, str]:
    states: dict[int, tuple[int, tuple[tuple[str, str], ...]]] = {0: (0, ())}
    for candidate_ref in candidate_speakers:
        next_states = dict(states)
        for mask, (score, assignments) in states.items():
            for ref_index, reference_ref in enumerate(reference_speakers):
                bit = 1 << ref_index
                if mask & bit:
                    continue
                weight = weights.get((candidate_ref, reference_ref), 0)
                if weight <= 0:
                    continue
                candidate_state = (
                    score + weight,
                    tuple(sorted((*assignments, (candidate_ref, reference_ref)))),
                )
                previous = next_states.get(mask | bit)
                if previous is None or _assignment_is_better(candidate_state, previous):
                    next_states[mask | bit] = candidate_state
        states = next_states
    best = max(states.values(), key=lambda item: (item[0], len(item[1]), tuple(reversed(item[1]))))
    return dict(best[1])


def _assignment_is_better(
    candidate: tuple[int, tuple[tuple[str, str], ...]],
    previous: tuple[int, tuple[tuple[str, str], ...]],
) -> bool:
    return (candidate[0], len(candidate[1]), tuple(reversed(candidate[1]))) > (
        previous[0],
        len(previous[1]),
        tuple(reversed(previous[1])),
    )


def _score_metrics(
    reference_spans: tuple[_SpanView, ...],
    candidate_spans: tuple[_SpanView, ...],
    intervals: tuple[EvaluationInterval, ...],
    speaker_mapping: dict[str, str],
    policy: ScoringPolicyManifest,
) -> dict[str, Any]:
    reference_speaker_ms = 0
    hypothesis_speaker_ms = 0
    matched_speaker_ms = 0
    scored_interval_ms = 0
    reference_speakers_scored: set[str] = set()
    candidate_speakers_scored: set[str] = set()
    mapped_candidate_speakers_scored: set[str] = set()
    for atom in _atomic_intervals(_normalize_intervals(intervals), reference_spans, candidate_spans):
        active_reference_spans = _active_spans(reference_spans, atom)
        if not policy.score_overlap and _is_reference_overlap(active_reference_spans):
            continue
        active_reference = {span.speaker_ref for span in active_reference_spans}
        active_candidate = {span.speaker_ref for span in _active_spans(candidate_spans, atom)}
        mapped_candidate = {
            speaker_mapping[candidate_ref]
            for candidate_ref in active_candidate
            if candidate_ref in speaker_mapping
        }
        reference_speakers_scored.update(active_reference)
        candidate_speakers_scored.update(active_candidate)
        mapped_candidate_speakers_scored.update(
            candidate_ref for candidate_ref in active_candidate if candidate_ref in speaker_mapping
        )
        duration_ms = atom.duration_ms
        scored_interval_ms += duration_ms
        reference_speaker_ms += duration_ms * len(active_reference)
        hypothesis_speaker_ms += duration_ms * len(active_candidate)
        matched_speaker_ms += duration_ms * len(active_reference.intersection(mapped_candidate))

    missed_speaker_ms = max(0, reference_speaker_ms - matched_speaker_ms)
    false_alarm_speaker_ms = max(0, hypothesis_speaker_ms - matched_speaker_ms)
    if reference_speaker_ms == 0:
        speaker_label_accuracy = 1.0 if hypothesis_speaker_ms == 0 else 0.0
        diarization_error_rate = 0.0 if hypothesis_speaker_ms == 0 else 1.0
    else:
        speaker_label_accuracy = matched_speaker_ms / reference_speaker_ms
        diarization_error_rate = (missed_speaker_ms + false_alarm_speaker_ms) / reference_speaker_ms
    return {
        "candidate_speaker_count": len(candidate_speakers_scored),
        "diarization_error_rate": _round_metric(diarization_error_rate),
        "false_alarm_speaker_ms": false_alarm_speaker_ms,
        "hypothesis_speaker_ms": hypothesis_speaker_ms,
        "mapped_candidate_speaker_count": len(mapped_candidate_speakers_scored),
        "matched_speaker_ms": matched_speaker_ms,
        "missed_speaker_ms": missed_speaker_ms,
        "reference_speaker_count": len(reference_speakers_scored),
        "reference_speaker_ms": reference_speaker_ms,
        "scored_interval_ms": scored_interval_ms,
        "speaker_label_accuracy": _round_metric(speaker_label_accuracy),
        "speaker_label_error_rate": _round_metric(1.0 - speaker_label_accuracy),
    }


def _apply_scoring_collar(
    intervals: tuple[EvaluationInterval, ...],
    reference_spans: tuple[_SpanView, ...],
    collar_ms: int,
) -> tuple[EvaluationInterval, ...]:
    if collar_ms <= 0:
        return intervals
    collar_regions = _reference_boundary_intervals(reference_spans, intervals, collar_ms)
    return _subtract_intervals(intervals, collar_regions)


def _reference_boundary_intervals(
    reference_spans: tuple[_SpanView, ...],
    scoring_intervals: tuple[EvaluationInterval, ...],
    collar_ms: int,
) -> tuple[EvaluationInterval, ...]:
    boundaries = {
        (point, span.channel_id)
        for span in reference_spans
        for point in (span.start_ms, span.end_ms)
        if any(
            region.start_ms <= point <= region.end_ms
            and _channels_match(span.channel_id, region.channel_id)
            and min(span.end_ms, region.end_ms) > max(span.start_ms, region.start_ms)
            for region in scoring_intervals
        )
    }
    return _clip_intervals_to_regions(
        tuple(
            EvaluationInterval(max(0, point - collar_ms), point + collar_ms, channel_id)
            for point, channel_id in sorted(boundaries, key=lambda item: (item[0], item[1] or ""))
        ),
        scoring_intervals,
    )


def _is_reference_overlap(active_spans: tuple[_SpanView, ...]) -> bool:
    return len({span.speaker_ref for span in active_spans}) > 1 or any(span.overlap for span in active_spans)


def _turn_duration_intervals(
    reference_spans: tuple[_SpanView, ...],
    scoring_intervals: tuple[EvaluationInterval, ...],
    *,
    short: bool,
) -> tuple[EvaluationInterval, ...]:
    intervals: list[EvaluationInterval] = []
    for span in reference_spans:
        duration_ms = span.end_ms - span.start_ms
        if short and duration_ms > _SHORT_TURN_MAX_MS:
            continue
        if not short and duration_ms < _LONG_TURN_MIN_MS:
            continue
        intervals.extend(_clip_interval_to_regions(span.interval, scoring_intervals))
    return _normalize_intervals(tuple(intervals))


def _speaker_change_boundary_intervals(
    reference_spans: tuple[_SpanView, ...],
    scoring_intervals: tuple[EvaluationInterval, ...],
    collar_ms: int,
) -> tuple[EvaluationInterval, ...]:
    window_ms = max(collar_ms, 250)
    boundaries: set[tuple[int, str | None]] = set()
    channel_ids = tuple(sorted({region.channel_id for region in scoring_intervals}, key=lambda value: value or ""))
    for channel_id in channel_ids:
        points = tuple(
            sorted(
                {
                    point
                    for span in reference_spans
                    if _channels_match(span.channel_id, channel_id)
                    for point in (span.start_ms, span.end_ms)
                }
            )
        )
        for point in points:
            if point <= 0:
                continue
            before = EvaluationInterval(max(0, point - 1), point, channel_id)
            after = EvaluationInterval(point, point + 1, channel_id)
            before_speakers = {span.speaker_ref for span in _active_spans(reference_spans, before)}
            after_speakers = {span.speaker_ref for span in _active_spans(reference_spans, after)}
            if not before_speakers or not after_speakers or before_speakers == after_speakers:
                continue
            in_boundary_region = any(
                region.start_ms < point < region.end_ms and _channels_match(channel_id, region.channel_id)
                for region in scoring_intervals
            )
            if in_boundary_region:
                boundaries.add((point, channel_id))
    intervals = tuple(
        EvaluationInterval(max(0, point - window_ms), point + window_ms, channel_id)
        for point, channel_id in sorted(boundaries, key=lambda item: (item[0], item[1] or ""))
    )
    clipped: list[EvaluationInterval] = []
    for interval in intervals:
        clipped.extend(_clip_interval_to_regions(interval, scoring_intervals))
    return _normalize_intervals(tuple(clipped))


def _clip_interval_to_regions(
    interval: EvaluationInterval,
    scoring_intervals: tuple[EvaluationInterval, ...],
) -> tuple[EvaluationInterval, ...]:
    clipped: list[EvaluationInterval] = []
    for region in scoring_intervals:
        if not _channels_match(interval.channel_id, region.channel_id):
            continue
        start_ms = max(interval.start_ms, region.start_ms)
        end_ms = min(interval.end_ms, region.end_ms)
        if end_ms <= start_ms:
            continue
        channel_id = region.channel_id if region.channel_id is not None else interval.channel_id
        clipped.append(EvaluationInterval(start_ms, end_ms, channel_id))
    return tuple(clipped)


def _subtract_intervals(
    intervals: tuple[EvaluationInterval, ...],
    excluded_intervals: tuple[EvaluationInterval, ...],
) -> tuple[EvaluationInterval, ...]:
    result: list[EvaluationInterval] = []
    for interval in intervals:
        remaining = [(interval.start_ms, interval.end_ms)]
        for excluded in excluded_intervals:
            if not _channels_match(interval.channel_id, excluded.channel_id):
                continue
            next_remaining: list[tuple[int, int]] = []
            for start_ms, end_ms in remaining:
                overlap_start = max(start_ms, excluded.start_ms)
                overlap_end = min(end_ms, excluded.end_ms)
                if overlap_end <= overlap_start:
                    next_remaining.append((start_ms, end_ms))
                    continue
                if start_ms < overlap_start:
                    next_remaining.append((start_ms, overlap_start))
                if overlap_end < end_ms:
                    next_remaining.append((overlap_end, end_ms))
            remaining = next_remaining
        result.extend(
            EvaluationInterval(start_ms, end_ms, interval.channel_id)
            for start_ms, end_ms in remaining
            if end_ms > start_ms
        )
    return _normalize_intervals(tuple(result))


def _clip_intervals_to_regions(
    intervals: tuple[EvaluationInterval, ...],
    regions: tuple[EvaluationInterval, ...],
) -> tuple[EvaluationInterval, ...]:
    clipped: list[EvaluationInterval] = []
    for interval in intervals:
        clipped.extend(_clip_interval_to_regions(interval, regions))
    return _normalize_intervals(tuple(clipped))


def _slice(
    dimension: str,
    value: str,
    intervals: tuple[EvaluationInterval, ...],
    minimum_support_ms: int,
) -> EvaluationSliceDefinition:
    intervals = _normalize_intervals(intervals)
    support_ms = _interval_support_ms(intervals)
    status: EvaluationSliceStatus = "ready" if support_ms >= minimum_support_ms else "insufficient_support"
    return EvaluationSliceDefinition(
        slice_id=f"{dimension}:{value}",
        dimension=dimension,
        value=value,
        status=status,
        support_ms=support_ms,
        minimum_support_ms=minimum_support_ms,
        intervals=intervals,
    )


def _speaker_count_bucket(count: int) -> str:
    if count <= 0:
        return "0"
    if count == 1:
        return "1"
    if count == 2:
        return "2"
    return "3_plus"


def _normalize_intervals(intervals: tuple[EvaluationInterval, ...]) -> tuple[EvaluationInterval, ...]:
    ordered = sorted(intervals, key=lambda item: (item.channel_id or "", item.start_ms, item.end_ms))
    merged: list[EvaluationInterval] = []
    for interval in ordered:
        if not merged or merged[-1].channel_id != interval.channel_id or interval.start_ms > merged[-1].end_ms:
            merged.append(interval)
            continue
        previous = merged[-1]
        merged[-1] = EvaluationInterval(previous.start_ms, max(previous.end_ms, interval.end_ms), previous.channel_id)
    return tuple(merged)


def _interval_support_ms(intervals: tuple[EvaluationInterval, ...]) -> int:
    return sum(interval.duration_ms for interval in _normalize_intervals(intervals))


def _channels_match(left: str | None, right: str | None) -> bool:
    return left is None or right is None or left == right


def _round_metric(value: float) -> float:
    if not math.isfinite(value):
        raise ValidationError("metric values must be finite")
    return round(value, 6)


def _tuple_of(values: object, item_type: type[Any], field_name: str) -> tuple[Any, ...]:
    try:
        items = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValidationError(f"{field_name} must be an iterable") from exc
    for index, item in enumerate(items):
        if not isinstance(item, item_type):
            raise ValidationError(f"{field_name}[{index}] must be a {item_type.__name__}")
    return items


def _validate_string_map(value: object, field_name: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValidationError(f"{field_name} must be an object")
    result: dict[str, str] = {}
    for key, item in value.items():
        result[_require_id(key, f"{field_name}.key")] = _require_id(item, f"{field_name}.{key}")
    return result


def _validate_json_object(value: object, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{field_name} must be an object")
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValidationError(f"{field_name} field names must be strings")
        result[key] = _validate_json_value(item, f"{field_name}.{key}")
    return result


def _validate_json_value(value: object, field_name: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError(f"{field_name} must be finite")
        return value
    if isinstance(value, list):
        return [_validate_json_value(item, f"{field_name}[]") for item in value]
    if isinstance(value, tuple):
        return tuple(_validate_json_value(item, f"{field_name}[]") for item in value)
    if isinstance(value, dict):
        return _validate_json_object(value, field_name)
    raise ValidationError(f"{field_name} must be JSON-compatible")


def _thaw_json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _thaw_json_value(item) for key, item in value.items()}
    return value


def _require_non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{field_name} must be an integer")
    if value < 0:
        raise ValidationError(f"{field_name} must be >= 0")
    return value


def _require_positive_int(value: object, field_name: str) -> int:
    value = _require_non_negative_int(value, field_name)
    if value <= 0:
        raise ValidationError(f"{field_name} must be greater than 0")
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
