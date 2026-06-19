"""Sentinel baselines for central diarization evaluator health checks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

from keyframe.diarization.bundles import ReferenceBundle
from keyframe.diarization.engines import EngineConfigMetadata, NormalizedEngineOutput
from keyframe.diarization.evaluator import (
    _item_is_reference_overlap,
    _item_is_scoreable,
    _scoring_intervals,
    _scoreable_words,
    _spans_from_recording,
    _spans_from_words,
    DiarizationEvaluationResult,
    evaluate_diarization_candidate,
)
from keyframe.diarization.manifests import ScoringPolicyManifest, default_scoring_policy
from keyframe.diarization.models import CanonicalWord, SpeakerSpan, ValidationError
from keyframe.diarization.provenance import NormalizedArtifactProvenance


SentinelBaselineId = Literal[
    "oracle",
    "single_speaker_collapse",
    "channel_only",
    "timestamp_shifted",
    "shuffled_speakers",
    "perfect_text_wrong_speaker",
    "bad_text_perfect_speaker",
    "bad_turn_builder",
]
SentinelBaselineStatus = Literal["passed", "failed"]

SENTINEL_BASELINE_IDS: tuple[SentinelBaselineId, ...] = (
    "oracle",
    "single_speaker_collapse",
    "channel_only",
    "timestamp_shifted",
    "shuffled_speakers",
    "perfect_text_wrong_speaker",
    "bad_text_perfect_speaker",
    "bad_turn_builder",
)
_DIAGNOSTIC_BASELINES = frozenset(
    {
        "oracle",
        "single_speaker_collapse",
        "channel_only",
        "timestamp_shifted",
        "shuffled_speakers",
    }
)
_DIAGNOSTIC_METRICS = (
    "diarization_error_rate",
    "speaker_error_rate",
    "false_alarm_rate",
    "miss_rate",
    "reference_speaker_ms",
    "matched_speaker_ms",
    "speaker_label_accuracy",
    "candidate_speaker_count",
    "reference_speaker_count",
)
_SENTINEL_ADAPTER_ID = "keyframe-sentinel-baseline"


@dataclass(frozen=True)
class SentinelBaselineCheck:
    """One sentinel hypothesis and its evaluator health result."""

    baseline_id: SentinelBaselineId
    status: SentinelBaselineStatus
    output_id: str
    policy_id: str
    metrics: dict[str, Any]
    evaluation: DiarizationEvaluationResult
    failures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        baseline_id = _validate_baseline_id(self.baseline_id)
        object.__setattr__(self, "baseline_id", baseline_id)
        object.__setattr__(self, "status", _validate_status(self.status))
        object.__setattr__(self, "output_id", _require_text(self.output_id, "sentinel_check.output_id"))
        object.__setattr__(self, "policy_id", _require_text(self.policy_id, "sentinel_check.policy_id"))
        object.__setattr__(self, "metrics", _validate_json_object(self.metrics, "sentinel_check.metrics"))
        if not isinstance(self.evaluation, DiarizationEvaluationResult):
            raise ValidationError("sentinel_check.evaluation must be a DiarizationEvaluationResult")
        object.__setattr__(self, "failures", _tuple_of_text(self.failures, "sentinel_check.failures"))
        if self.status == "passed" and self.failures:
            raise ValidationError("passed sentinel checks cannot include failures")
        if self.status == "failed" and not self.failures:
            raise ValidationError("failed sentinel checks must include failures")

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_id": self.baseline_id,
            "evaluation": self.evaluation.to_dict(),
            "failures": list(self.failures),
            "metrics": _thaw_json_value(self.metrics),
            "output_id": self.output_id,
            "policy_id": self.policy_id,
            "status": self.status,
        }


@dataclass(frozen=True)
class SentinelBaselineReport:
    """Evaluator health report that must pass before engine benchmark reporting."""

    status: SentinelBaselineStatus
    checks: tuple[SentinelBaselineCheck, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _validate_status(self.status))
        checks = _tuple_of(self.checks, SentinelBaselineCheck, "sentinel_report.checks")
        object.__setattr__(self, "checks", checks)
        if self.status == "passed" and any(not check.passed for check in checks):
            raise ValidationError("passed sentinel reports cannot include failed checks")
        if self.status == "failed" and all(check.passed for check in checks):
            raise ValidationError("failed sentinel reports must include failed checks")

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    @property
    def failures(self) -> tuple[str, ...]:
        return tuple(
            f"{check.baseline_id}: {failure}"
            for check in self.checks
            for failure in check.failures
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "checks": [check.to_dict() for check in self.checks],
            "failures": list(self.failures),
            "status": self.status,
        }


def build_sentinel_baseline_outputs(reference: ReferenceBundle) -> tuple[NormalizedEngineOutput, ...]:
    """Generate all sentinel hypotheses from a reference bundle, without model output."""

    if not isinstance(reference, ReferenceBundle):
        raise ValidationError("reference must be a ReferenceBundle")
    return tuple(build_sentinel_baseline_output(reference, baseline_id) for baseline_id in SENTINEL_BASELINE_IDS)


def build_sentinel_baseline_output(
    reference: ReferenceBundle,
    baseline_id: SentinelBaselineId,
) -> NormalizedEngineOutput:
    """Generate one labeled sentinel hypothesis from reference evidence."""

    if not isinstance(reference, ReferenceBundle):
        raise ValidationError("reference must be a ReferenceBundle")
    baseline_id = _validate_baseline_id(baseline_id)
    recording = reference.recording
    speaker_map = _sentinel_speaker_map(recording, baseline_id)
    words = _baseline_words(recording.words, speaker_map, baseline_id, recording.duration_ms)
    spans = _baseline_spans(recording.speaker_spans, speaker_map, baseline_id, recording.duration_ms)
    output_id = f"sentinel-baseline:{recording.recording_id}:{baseline_id}"
    return NormalizedEngineOutput(
        output_id=output_id,
        output_kind="word_spans",
        artifact=NormalizedArtifactProvenance.from_recording(
            recording,
            artifact_id=output_id,
            artifact_kind="fixture",
        ),
        config=EngineConfigMetadata(
            adapter_id=_SENTINEL_ADAPTER_ID,
            provider="keyframe",
            model_name=f"sentinel-{baseline_id}",
            model_version="1",
            config_id=baseline_id,
            parameters={
                "baseline_id": baseline_id,
                "sentinel": True,
            },
        ),
        words=words,
        speaker_spans=spans,
    )


def evaluate_sentinel_baselines(
    reference: ReferenceBundle,
    *,
    diagnostic_policy: ScoringPolicyManifest | None = None,
    product_policy: ScoringPolicyManifest | None = None,
) -> SentinelBaselineReport:
    """Evaluate every sentinel baseline and assert expected pass/fail directions."""

    if not isinstance(reference, ReferenceBundle):
        raise ValidationError("reference must be a ReferenceBundle")
    diagnostic_policy = diagnostic_policy or _default_diagnostic_policy()
    product_policy = product_policy or default_scoring_policy("product_transcript")
    baseline_ids = _supported_sentinel_baseline_ids(
        reference,
        diagnostic_policy=diagnostic_policy,
        product_policy=product_policy,
    )
    checks = tuple(
        _evaluate_sentinel_check(
            reference,
            build_sentinel_baseline_output(reference, baseline_id),
            baseline_id,
            diagnostic_policy=diagnostic_policy,
            product_policy=product_policy,
        )
        for baseline_id in baseline_ids
    )
    status: SentinelBaselineStatus = "passed" if all(check.passed for check in checks) else "failed"
    return SentinelBaselineReport(status=status, checks=checks)


def require_passing_sentinel_baselines(report: SentinelBaselineReport) -> None:
    """Block engine benchmark reporting when sentinel evaluator health checks fail."""

    if not isinstance(report, SentinelBaselineReport):
        raise ValidationError("sentinel_report must be a SentinelBaselineReport")
    if report.passed:
        return
    raise ValidationError(
        "sentinel baseline health checks failed; refusing engine benchmark execution: "
        + "; ".join(report.failures)
    )


def _evaluate_sentinel_check(
    reference: ReferenceBundle,
    output: NormalizedEngineOutput,
    baseline_id: SentinelBaselineId,
    *,
    diagnostic_policy: ScoringPolicyManifest,
    product_policy: ScoringPolicyManifest,
) -> SentinelBaselineCheck:
    policy = diagnostic_policy if baseline_id in _DIAGNOSTIC_BASELINES else product_policy
    evaluation = evaluate_diarization_candidate(reference, output, scoring_policy=policy)
    metrics = dict(evaluation.recording_metrics[0].metrics)
    if baseline_id == "bad_text_perfect_speaker":
        metrics["sentinel_text_mismatch_rate"] = _text_mismatch_rate(reference.recording.words, output.words)
    failures = _sentinel_failures(baseline_id, metrics)
    return SentinelBaselineCheck(
        baseline_id=baseline_id,
        status="failed" if failures else "passed",
        output_id=output.output_id,
        policy_id=policy.policy_id,
        metrics=metrics,
        evaluation=evaluation,
        failures=failures,
    )


def _sentinel_failures(baseline_id: SentinelBaselineId, metrics: dict[str, Any]) -> tuple[str, ...]:
    failures: list[str] = []
    if baseline_id == "oracle":
        if metrics.get("diarization_error_rate") != 0.0:
            failures.append("oracle diarization_error_rate must be 0.0")
        if metrics.get("speaker_label_accuracy") != 1.0:
            failures.append("oracle speaker_label_accuracy must be 1.0")
        return tuple(failures)
    if baseline_id in {"single_speaker_collapse", "channel_only", "timestamp_shifted", "shuffled_speakers"}:
        if metrics.get("diarization_error_rate", 0.0) <= 0.0:
            failures.append(f"{baseline_id} must degrade diarization_error_rate")
        return tuple(failures)
    if baseline_id == "perfect_text_wrong_speaker":
        if metrics.get("word_speaker_label_accuracy", 1.0) >= 1.0:
            failures.append("perfect_text_wrong_speaker must degrade word_speaker_label_accuracy")
        return tuple(failures)
    if baseline_id == "bad_text_perfect_speaker":
        if metrics.get("sentinel_text_mismatch_rate", 0.0) <= 0.0:
            failures.append("bad_text_perfect_speaker must degrade sentinel_text_mismatch_rate")
        if metrics.get("word_speaker_label_accuracy") != 1.0:
            failures.append("bad_text_perfect_speaker must preserve speaker attribution")
        return tuple(failures)
    if baseline_id == "bad_turn_builder":
        if metrics.get("word_speaker_label_accuracy") != 1.0:
            failures.append("bad_turn_builder must preserve word speaker attribution")
        if metrics.get("turn_speaker_label_accuracy", 1.0) >= 1.0:
            failures.append("bad_turn_builder must degrade turn_speaker_label_accuracy")
        return tuple(failures)
    raise ValidationError(f"sentinel baseline is not supported: {baseline_id}")


def _default_diagnostic_policy() -> ScoringPolicyManifest:
    return replace(
        default_scoring_policy("diagnostic_diarization"),
        policy_id="sentinel-diagnostic-diarization-v1",
        description="Sentinel diagnostic diarization policy with zero collar and expanded health metrics.",
        collar_ms=0,
        metric_set=_DIAGNOSTIC_METRICS,
    )


def _supported_sentinel_baseline_ids(
    reference: ReferenceBundle,
    *,
    diagnostic_policy: ScoringPolicyManifest,
    product_policy: ScoringPolicyManifest,
) -> tuple[SentinelBaselineId, ...]:
    recording = reference.recording
    diagnostic_intervals = _scoring_intervals(recording, diagnostic_policy)
    diagnostic_spans = _scoreable_reference_spans(
        _spans_from_recording(recording),
        diagnostic_intervals,
        diagnostic_policy,
    )
    diagnostic_speaker_refs = _speaker_refs_from_spans(diagnostic_spans)
    product_intervals = _scoring_intervals(recording, product_policy)
    product_word_spans = _scoreable_reference_spans(
        _spans_from_words(_scoreable_words(recording.words, product_policy)),
        product_intervals,
        product_policy,
    )
    product_word_speaker_refs = _speaker_refs_from_spans(product_word_spans)

    baseline_ids: list[SentinelBaselineId] = []
    if diagnostic_spans:
        baseline_ids.append("oracle")
    if len(diagnostic_speaker_refs) >= 2:
        baseline_ids.append("single_speaker_collapse")
    if _has_channel_only_degradation_support(diagnostic_spans):
        baseline_ids.append("channel_only")
    if _has_timestamp_shift_degradation_support(diagnostic_spans, recording.duration_ms):
        baseline_ids.append("timestamp_shifted")
    if len(diagnostic_speaker_refs) >= 2:
        baseline_ids.append("shuffled_speakers")
    if len(product_word_speaker_refs) >= 2:
        baseline_ids.append("perfect_text_wrong_speaker")
    if product_word_spans:
        baseline_ids.append("bad_text_perfect_speaker")
    if len(product_word_speaker_refs) >= 2:
        baseline_ids.append("bad_turn_builder")
    return tuple(baseline_ids)


def _scoreable_reference_spans(
    spans: tuple[Any, ...],
    intervals: tuple[Any, ...],
    policy: ScoringPolicyManifest,
) -> tuple[Any, ...]:
    return tuple(
        span
        for span in spans
        if _item_is_scoreable(span, intervals, policy)
        and (policy.score_overlap or not _item_is_reference_overlap(span, spans, policy))
    )


def _speaker_refs_from_spans(spans: tuple[Any, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(span.speaker_ref for span in spans))


def _has_channel_only_degradation_support(spans: tuple[Any, ...]) -> bool:
    speaker_refs_by_channel: dict[str | None, set[str]] = {}
    for span in spans:
        speaker_refs_by_channel.setdefault(span.channel_id, set()).add(span.speaker_ref)
    return any(len(speaker_refs) >= 2 for speaker_refs in speaker_refs_by_channel.values())


def _has_timestamp_shift_degradation_support(spans: tuple[Any, ...], duration_ms: int) -> bool:
    return any(
        _shift_interval(span.start_ms, span.end_ms, duration_ms) != (span.start_ms, span.end_ms)
        for span in spans
    )


def _sentinel_speaker_map(recording: Any, baseline_id: SentinelBaselineId) -> dict[str, str]:
    return {
        speaker_ref: f"sentinel-baseline:{baseline_id}:speaker-{index:02d}"
        for index, speaker_ref in enumerate(_speaker_refs(recording), start=1)
    }


def _speaker_refs(recording: Any) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    events = [
        (word.start_ms, word.speaker_ref)
        for word in recording.words
        if word.speaker_ref is not None
    ]
    events.extend((span.start_ms, span.speaker_ref) for span in recording.speaker_spans)
    for _, speaker_ref in sorted(events):
        if speaker_ref not in seen:
            seen.add(speaker_ref)
            ordered.append(speaker_ref)
    for speaker in recording.speakers:
        if speaker.speaker_ref not in seen:
            seen.add(speaker.speaker_ref)
            ordered.append(speaker.speaker_ref)
    return tuple(ordered)


def _baseline_words(
    words: tuple[CanonicalWord, ...],
    speaker_map: dict[str, str],
    baseline_id: SentinelBaselineId,
    duration_ms: int,
) -> tuple[CanonicalWord, ...]:
    if baseline_id == "shuffled_speakers":
        return _shuffled_words(words, speaker_map, baseline_id)
    result = []
    collapsed_ref = f"sentinel-baseline:{baseline_id}:collapsed-speaker"
    channel_refs: dict[str | None, str] = {}
    for index, word in enumerate(words):
        text = word.text
        speaker_ref = _mapped_speaker_ref(word.speaker_ref, speaker_map)
        start_ms = word.start_ms
        end_ms = word.end_ms
        if baseline_id == "single_speaker_collapse":
            speaker_ref = collapsed_ref if word.speaker_ref is not None else None
        elif baseline_id == "channel_only":
            speaker_ref = _channel_speaker_ref(word.channel_id, channel_refs)
        elif baseline_id == "timestamp_shifted":
            start_ms, end_ms = _shift_interval(start_ms, end_ms, duration_ms)
        elif baseline_id == "perfect_text_wrong_speaker":
            speaker_ref = collapsed_ref if word.speaker_ref is not None else None
        elif baseline_id == "bad_turn_builder":
            speaker_ref = _mapped_speaker_ref(word.speaker_ref, speaker_map)
        elif baseline_id == "bad_text_perfect_speaker":
            text = f"sentinel-bad-text-{index + 1}"
        result.append(
            replace(
                word,
                word_id=f"sentinel-baseline:{baseline_id}:word:{index + 1:06d}",
                text=text,
                start_ms=start_ms,
                end_ms=end_ms,
                speaker_ref=speaker_ref,
                display_label=None,
            )
        )
    if baseline_id == "bad_turn_builder" and result:
        existing_refs = tuple(dict.fromkeys(word.speaker_ref for word in result if word.speaker_ref is not None))
        if len(existing_refs) < 2:
            return tuple(result)
        first_start = min(word.start_ms for word in result)
        last_end = max(word.end_ms for word in result)
        template = next(word for word in result if word.speaker_ref is not None)
        confuser_ref = next(speaker_ref for speaker_ref in existing_refs if speaker_ref != template.speaker_ref)
        result.append(
            replace(
                template,
                word_id=f"sentinel-baseline:{baseline_id}:word:{len(result) + 1:06d}",
                text="sentinel bad turn",
                start_ms=first_start,
                end_ms=last_end,
                speaker_ref=confuser_ref,
                display_label=None,
            )
        )
    return tuple(result)


def _baseline_spans(
    spans: tuple[SpeakerSpan, ...],
    speaker_map: dict[str, str],
    baseline_id: SentinelBaselineId,
    duration_ms: int,
) -> tuple[SpeakerSpan, ...]:
    if baseline_id == "shuffled_speakers":
        return _shuffled_spans(spans, speaker_map, baseline_id)
    result = []
    collapsed_ref = f"sentinel-baseline:{baseline_id}:collapsed-speaker"
    channel_refs: dict[str | None, str] = {}
    for index, span in enumerate(spans):
        speaker_ref = _mapped_speaker_ref(span.speaker_ref, speaker_map)
        start_ms = span.start_ms
        end_ms = span.end_ms
        if baseline_id == "single_speaker_collapse":
            speaker_ref = collapsed_ref
        elif baseline_id == "channel_only":
            speaker_ref = _channel_speaker_ref(span.channel_id, channel_refs)
        elif baseline_id == "timestamp_shifted":
            start_ms, end_ms = _shift_interval(start_ms, end_ms, duration_ms)
        elif baseline_id == "perfect_text_wrong_speaker":
            speaker_ref = collapsed_ref
        result.append(
            replace(
                span,
                span_id=f"sentinel-baseline:{baseline_id}:span:{index + 1:06d}",
                start_ms=start_ms,
                end_ms=end_ms,
                speaker_ref=speaker_ref,
            )
        )
    return tuple(result)


def _shuffled_words(
    words: tuple[CanonicalWord, ...],
    speaker_map: dict[str, str],
    baseline_id: SentinelBaselineId,
) -> tuple[CanonicalWord, ...]:
    rotated_refs = _rotated_speaker_map(speaker_map)
    if not rotated_refs:
        return ()
    result = []
    for index, word in enumerate(words):
        base_id = f"sentinel-baseline:{baseline_id}:word:{index + 1:06d}"
        if word.speaker_ref is None:
            result.append(replace(word, word_id=base_id, display_label=None))
            continue
        speaker_ref = speaker_map[word.speaker_ref]
        rotated_ref = rotated_refs[word.speaker_ref]
        if speaker_ref == rotated_ref:
            result.append(
                replace(
                    word,
                    word_id=base_id,
                    speaker_ref=speaker_ref,
                    display_label=None,
                )
            )
            continue
        midpoint = word.start_ms + ((word.end_ms - word.start_ms) // 2)
        if word.start_ms < midpoint < word.end_ms:
            result.append(
                replace(
                    word,
                    word_id=f"{base_id}:a",
                    end_ms=midpoint,
                    speaker_ref=speaker_ref,
                    display_label=None,
                )
            )
            result.append(
                replace(
                    word,
                    word_id=f"{base_id}:b",
                    start_ms=midpoint,
                    speaker_ref=rotated_ref,
                    display_label=None,
                )
            )
            continue
        result.append(
            replace(
                word,
                word_id=f"{base_id}:a",
                speaker_ref=speaker_ref,
                display_label=None,
            )
        )
        result.append(
            replace(
                word,
                word_id=f"{base_id}:b",
                speaker_ref=rotated_ref,
                display_label=None,
            )
        )
    return tuple(result)


def _rotated_speaker_map(speaker_map: dict[str, str]) -> dict[str, str]:
    original_refs = tuple(speaker_map)
    if len(original_refs) < 2:
        return dict(speaker_map)
    return {
        speaker_ref: speaker_map[original_refs[(index + 1) % len(original_refs)]]
        for index, speaker_ref in enumerate(original_refs)
    }


def _shuffled_spans(
    spans: tuple[SpeakerSpan, ...],
    speaker_map: dict[str, str],
    baseline_id: SentinelBaselineId,
) -> tuple[SpeakerSpan, ...]:
    speaker_refs = tuple(speaker_map.values())
    if not speaker_refs:
        return ()
    result = []
    for index, span in enumerate(spans):
        midpoint = span.start_ms + max(1, (span.end_ms - span.start_ms) // 2)
        speaker_ref = speaker_refs[index % len(speaker_refs)]
        next_speaker_ref = speaker_refs[(index + 1) % len(speaker_refs)]
        if midpoint >= span.end_ms or speaker_ref == next_speaker_ref:
            result.append(
                replace(
                    span,
                    span_id=f"sentinel-baseline:{baseline_id}:span:{len(result) + 1:06d}",
                    speaker_ref=next_speaker_ref,
                )
            )
            continue
        result.append(
            replace(
                span,
                span_id=f"sentinel-baseline:{baseline_id}:span:{len(result) + 1:06d}",
                end_ms=midpoint,
                speaker_ref=speaker_ref,
            )
        )
        result.append(
            replace(
                span,
                span_id=f"sentinel-baseline:{baseline_id}:span:{len(result) + 1:06d}",
                start_ms=midpoint,
                speaker_ref=next_speaker_ref,
            )
        )
    return tuple(result)


def _mapped_speaker_ref(speaker_ref: str | None, speaker_map: dict[str, str]) -> str | None:
    if speaker_ref is None:
        return None
    return speaker_map[speaker_ref]


def _channel_speaker_ref(channel_id: str | None, channel_refs: dict[str | None, str]) -> str:
    if channel_id not in channel_refs:
        label = "none" if channel_id is None else channel_id
        channel_refs[channel_id] = f"sentinel-baseline:channel-only:{label}"
    return channel_refs[channel_id]


def _shift_interval(start_ms: int, end_ms: int, duration_ms: int, shift_ms: int = 300) -> tuple[int, int]:
    interval_ms = end_ms - start_ms
    if interval_ms >= duration_ms:
        return start_ms, end_ms
    latest_start = duration_ms - interval_ms
    right_start = min(start_ms + shift_ms, latest_start)
    shifted_start = right_start if right_start != start_ms else max(0, start_ms - shift_ms)
    return shifted_start, shifted_start + interval_ms


def _text_mismatch_rate(
    reference_words: tuple[CanonicalWord, ...],
    candidate_words: tuple[CanonicalWord, ...],
) -> float:
    reference_texts = tuple(word.text for word in reference_words if word.speaker_ref is not None)
    candidate_texts = tuple(word.text for word in candidate_words if word.speaker_ref is not None)
    if not reference_texts:
        return 0.0 if not candidate_texts else 1.0
    mismatch_count = 0
    for index, reference_text in enumerate(reference_texts):
        candidate_text = candidate_texts[index] if index < len(candidate_texts) else None
        if candidate_text != reference_text:
            mismatch_count += 1
    mismatch_count += max(0, len(candidate_texts) - len(reference_texts))
    return round(mismatch_count / max(len(reference_texts), len(candidate_texts), 1), 6)


def _validate_baseline_id(value: object) -> SentinelBaselineId:
    value = _require_text(value, "sentinel_baseline_id")
    if value not in SENTINEL_BASELINE_IDS:
        raise ValidationError(f"sentinel baseline is not supported: {value}")
    return value  # type: ignore[return-value]


def _validate_status(value: object) -> SentinelBaselineStatus:
    value = _require_text(value, "sentinel_status")
    if value not in {"passed", "failed"}:
        raise ValidationError(f"sentinel status is not supported: {value}")
    return value  # type: ignore[return-value]


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field_name} is required")
    return value


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
    try:
        items = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValidationError(f"{field_name} must be an iterable") from exc
    return tuple(_require_text(item, f"{field_name}[{index}]") for index, item in enumerate(items))


def _validate_json_object(value: object, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{field_name} must be an object")
    return {str(key): _freeze_json_value(item, f"{field_name}.{key}") for key, item in value.items()}


def _freeze_json_value(value: object, field_name: str) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return _validate_json_object(value, field_name)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_json_value(item, f"{field_name}[]") for item in value)
    raise ValidationError(f"{field_name} must be JSON-compatible")


def _thaw_json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _thaw_json_value(item) for key, item in value.items()}
    return value
