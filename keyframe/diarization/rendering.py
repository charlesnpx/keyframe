"""Rendered transcript turns and transcript-local correction overlays."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

from keyframe.diarization.attribution import apply_session_local_attribution
from keyframe.diarization.models import (
    CanonicalRecording,
    CanonicalWord,
    DisplayLabel,
    LabelSource,
    SpeakerRecord,
    ValidationError,
)


RenderedTranscriptState = Literal[
    "confident_pipeline",
    "needs_review",
    "diagnostic_only",
    "unsupported",
    "speaker_attribution_unavailable",
]
SpeakerAttributionState = Literal["available", "unreliable", "unavailable"]
ReviewReason = Literal[
    "unsupported",
    "diagnostic_only",
    "speaker_attribution_unavailable",
    "low_speaker_confidence",
    "missing_speaker_confidence",
    "overlap_detected",
    "manual_uncertain",
    "manual_review_required",
]

_ALLOWED_RENDERED_TRANSCRIPT_STATES = frozenset(
    {
        "confident_pipeline",
        "needs_review",
        "diagnostic_only",
        "unsupported",
        "speaker_attribution_unavailable",
    }
)
_ALLOWED_REVIEW_REASONS = frozenset(
    {
        "unsupported",
        "diagnostic_only",
        "speaker_attribution_unavailable",
        "low_speaker_confidence",
        "missing_speaker_confidence",
        "overlap_detected",
        "manual_uncertain",
        "manual_review_required",
    }
)
_REVIEW_REASON_ORDER: tuple[ReviewReason, ...] = (
    "unsupported",
    "diagnostic_only",
    "speaker_attribution_unavailable",
    "low_speaker_confidence",
    "missing_speaker_confidence",
    "overlap_detected",
    "manual_uncertain",
    "manual_review_required",
)
_FORCED_STATE_REASONS: dict[RenderedTranscriptState, tuple[ReviewReason, ...]] = {
    "confident_pipeline": (),
    "needs_review": ("manual_review_required",),
    "diagnostic_only": ("diagnostic_only",),
    "unsupported": ("unsupported",),
    "speaker_attribution_unavailable": ("speaker_attribution_unavailable",),
}
_LABEL_SUPPRESSED_STATES = frozenset(
    {
        "diagnostic_only",
        "unsupported",
        "speaker_attribution_unavailable",
    }
)

OverlayOperationType = Literal[
    "rename_label",
    "merge_speakers",
    "split_speaker",
    "assign_span",
    "mark_uncertain",
    "mark_overlap",
]


@dataclass(frozen=True)
class RenderedWord:
    """Transcript-visible word payload that keeps provenance back to canonical words."""

    word_id: str
    text: str
    start_ms: int
    end_ms: int
    label: str | None
    channel_id: str | None = None
    speaker_confidence: float | None = None
    uncertain: bool = False
    overlap: bool = False
    display_label: DisplayLabel | None = None
    speaker_attribution: SpeakerAttributionState = "available"
    review_reasons: tuple[ReviewReason, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["review_reasons"] = list(self.review_reasons)
        return payload


@dataclass(frozen=True)
class RenderedTurn:
    """A contiguous display turn assembled from canonical words."""

    turn_id: str
    start_ms: int
    end_ms: int
    label: str | None
    word_ids: tuple[str, ...]
    text: str
    channel_id: str | None = None
    uncertain: bool = False
    overlap: bool = False
    display_label: DisplayLabel | None = None
    speaker_attribution: SpeakerAttributionState = "available"
    review_reasons: tuple[ReviewReason, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["word_ids"] = list(self.word_ids)
        payload["review_reasons"] = list(self.review_reasons)
        return payload


@dataclass(frozen=True)
class RenderedTranscript:
    """Rendered transcript JSON plus overlay provenance."""

    recording_id: str
    turns: tuple[RenderedTurn, ...]
    words: tuple[RenderedWord, ...]
    applied_overlay_ids: tuple[str, ...] = ()
    state: RenderedTranscriptState = "confident_pipeline"
    review_reasons: tuple[ReviewReason, ...] = ()
    speaker_attribution: SpeakerAttributionState = "available"

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied_overlay_ids": list(self.applied_overlay_ids),
            "recording_id": self.recording_id,
            "review_reasons": list(self.review_reasons),
            "speaker_attribution": self.speaker_attribution,
            "state": self.state,
            "turns": [turn.to_dict() for turn in self.turns],
            "words": [word.to_dict() for word in self.words],
        }


def rendered_transcript_json_dumps(transcript: RenderedTranscript) -> str:
    """Serialize rendered transcript JSON with byte-stable formatting."""

    if not isinstance(transcript, RenderedTranscript):
        raise ValidationError("transcript must be a RenderedTranscript")
    return json.dumps(transcript.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


@dataclass(frozen=True)
class RenameLabelOverlay:
    operation_id: str
    speaker_ref: str
    label: str
    operation_type: OverlayOperationType = "rename_label"


@dataclass(frozen=True)
class MergeSpeakersOverlay:
    operation_id: str
    source_speaker_refs: tuple[str, ...]
    target_speaker_ref: str
    operation_type: OverlayOperationType = "merge_speakers"


@dataclass(frozen=True)
class SplitSpeakerOverlay:
    operation_id: str
    source_speaker_ref: str
    new_speaker_ref: str
    start_ms: int
    end_ms: int
    label: str | None = None
    operation_type: OverlayOperationType = "split_speaker"


@dataclass(frozen=True)
class AssignSpanOverlay:
    operation_id: str
    speaker_ref: str | None
    start_ms: int
    end_ms: int
    label: str | None = None
    operation_type: OverlayOperationType = "assign_span"


@dataclass(frozen=True)
class MarkUncertainOverlay:
    operation_id: str
    word_ids: tuple[str, ...]
    uncertain: bool = True
    operation_type: OverlayOperationType = "mark_uncertain"


@dataclass(frozen=True)
class MarkOverlapOverlay:
    operation_id: str
    word_ids: tuple[str, ...]
    overlap: bool = True
    operation_type: OverlayOperationType = "mark_overlap"


TranscriptOverlay = (
    RenameLabelOverlay
    | MergeSpeakersOverlay
    | SplitSpeakerOverlay
    | AssignSpanOverlay
    | MarkUncertainOverlay
    | MarkOverlapOverlay
)


def render_transcript(
    recording: CanonicalRecording,
    *,
    overlays: tuple[TranscriptOverlay, ...] = (),
    label_source: LabelSource = "diarization_cluster",
    max_gap_ms: int = 900,
    split_after_punctuation: bool = True,
    degraded_state: RenderedTranscriptState | None = None,
    review_reasons: tuple[ReviewReason, ...] = (),
    min_speaker_confidence: float = 0.5,
) -> RenderedTranscript:
    """Return transcript turns assembled from immutable canonical evidence."""

    if not isinstance(recording, CanonicalRecording):
        raise ValidationError("recording must be a CanonicalRecording")
    if max_gap_ms < 0:
        raise ValidationError("max_gap_ms must be >= 0")
    requested_state = _validate_degraded_state(degraded_state)
    requested_review_reasons = _validate_review_reasons(review_reasons)
    min_speaker_confidence = _validate_min_speaker_confidence(min_speaker_confidence)

    attributed = apply_session_local_attribution(recording, label_source=label_source)
    ordered_overlays = _ordered_overlays(overlays)
    overlay_result = _apply_overlays(attributed, ordered_overlays)
    indexed_words = tuple(enumerate(overlay_result.recording.words))
    words = tuple(word for _, word in sorted(indexed_words, key=lambda item: (item[1].start_ms, item[0])))
    suppress_labels = requested_state in _LABEL_SUPPRESSED_STATES
    rendered_words = tuple(
        _render_word(word, overlay_result.uncertain_word_ids, min_speaker_confidence, suppress_labels)
        for word in words
    )
    turns = tuple(
        _build_turns(
            words,
            overlay_result.uncertain_word_ids,
            max_gap_ms,
            split_after_punctuation,
            min_speaker_confidence,
            suppress_labels,
        )
    )
    transcript_review_reasons = _combine_review_reasons(
        _FORCED_STATE_REASONS[requested_state or "confident_pipeline"],
        requested_review_reasons,
        *(word.review_reasons for word in rendered_words),
    )
    state = _resolve_transcript_state(requested_state, transcript_review_reasons)
    return RenderedTranscript(
        recording_id=recording.recording_id,
        turns=turns,
        words=rendered_words,
        applied_overlay_ids=tuple(_validate_operation_id(overlay.operation_id) for overlay in ordered_overlays),
        state=state,
        review_reasons=transcript_review_reasons,
        speaker_attribution=_transcript_speaker_attribution(state, rendered_words),
    )


@dataclass(frozen=True)
class _OverlayResult:
    recording: CanonicalRecording
    uncertain_word_ids: frozenset[str]


def _ordered_overlays(overlays: tuple[TranscriptOverlay, ...]) -> tuple[TranscriptOverlay, ...]:
    result = tuple(overlays)
    seen: set[str] = set()
    for overlay in result:
        operation_id = _validate_operation_id(overlay.operation_id)
        if operation_id in seen:
            raise ValidationError(f"duplicate overlay.operation_id: {operation_id}")
        seen.add(operation_id)
    return result


def _apply_overlays(recording: CanonicalRecording, overlays: tuple[TranscriptOverlay, ...]) -> _OverlayResult:
    result = recording
    uncertain_word_ids: frozenset[str] = frozenset()
    for overlay in overlays:
        _validate_operation_id(overlay.operation_id)
        if isinstance(overlay, RenameLabelOverlay):
            result = _rename_label(result, overlay)
        elif isinstance(overlay, MergeSpeakersOverlay):
            result = _merge_speakers(result, overlay)
        elif isinstance(overlay, SplitSpeakerOverlay):
            result = _split_speaker(result, overlay)
        elif isinstance(overlay, AssignSpanOverlay):
            result = _assign_span(result, overlay)
        elif isinstance(overlay, MarkUncertainOverlay):
            uncertain_word_ids = _mark_uncertain(result, uncertain_word_ids, overlay)
        elif isinstance(overlay, MarkOverlapOverlay):
            result = _mark_overlap(result, overlay)
        else:
            raise ValidationError(f"unsupported transcript overlay: {type(overlay).__name__}")
    return _OverlayResult(recording=result, uncertain_word_ids=uncertain_word_ids)


def _rename_label(recording: CanonicalRecording, overlay: RenameLabelOverlay) -> CanonicalRecording:
    speaker_ref = _validate_speaker_exists(recording, overlay.speaker_ref).speaker_ref
    label = _display_label(overlay.label, "reviewer_rename")
    speakers = tuple(
        replace(speaker, display_label=label) if speaker.speaker_ref == speaker_ref else speaker
        for speaker in recording.speakers
    )
    words = tuple(
        replace(word, display_label=label) if word.speaker_ref == speaker_ref else word
        for word in recording.words
    )
    return replace(recording, speakers=speakers, words=words)


def _merge_speakers(recording: CanonicalRecording, overlay: MergeSpeakersOverlay) -> CanonicalRecording:
    target = _validate_speaker_exists(recording, overlay.target_speaker_ref)
    target_speaker_ref = target.speaker_ref
    source_refs = tuple(_require_ref(value, "merge_speakers.source_speaker_refs") for value in overlay.source_speaker_refs)
    if not source_refs:
        raise ValidationError("merge_speakers.source_speaker_refs is required")
    for speaker_ref in source_refs:
        _validate_speaker_exists(recording, speaker_ref)
    merged_refs = set(source_refs)
    words = tuple(
        replace(
            word,
            speaker_ref=target_speaker_ref,
            speaker_confidence=(word.speaker_confidence if word.speaker_ref == target_speaker_ref else None),
            display_label=target.display_label,
        )
        if word.speaker_ref in merged_refs
        else word
        for word in recording.words
    )
    return replace(recording, words=words)


def _split_speaker(recording: CanonicalRecording, overlay: SplitSpeakerOverlay) -> CanonicalRecording:
    source_speaker_ref = _validate_speaker_exists(recording, overlay.source_speaker_ref).speaker_ref
    _ensure_valid_interval(overlay.start_ms, overlay.end_ms, "split_speaker")
    new_speaker_ref = _validate_new_speaker(recording, overlay.new_speaker_ref)
    label = _display_label(overlay.label or _next_person_label(recording), "reviewer_rename")
    new_speaker = SpeakerRecord(speaker_ref=new_speaker_ref, display_label=label)
    words = tuple(
        replace(word, speaker_ref=new_speaker_ref, speaker_confidence=None, display_label=label)
        if word.speaker_ref == source_speaker_ref and _word_within(word, overlay.start_ms, overlay.end_ms)
        else word
        for word in recording.words
    )
    return replace(recording, speakers=recording.speakers + (new_speaker,), words=words)


def _assign_span(recording: CanonicalRecording, overlay: AssignSpanOverlay) -> CanonicalRecording:
    _ensure_valid_interval(overlay.start_ms, overlay.end_ms, "assign_span")
    label = None
    assigned_speaker = None
    speaker_ref = None
    if overlay.speaker_ref is not None:
        assigned_speaker = _ensure_speaker(recording, overlay.speaker_ref, overlay.label)
        speaker_ref = assigned_speaker.speaker_ref
        label = assigned_speaker.display_label
    words = []
    for word in recording.words:
        if _word_within(word, overlay.start_ms, overlay.end_ms):
            words.append(
                replace(
                    word,
                    speaker_ref=speaker_ref,
                    speaker_confidence=(word.speaker_confidence if word.speaker_ref == speaker_ref else None),
                    display_label=label,
                )
            )
        elif overlay.label is not None and speaker_ref is not None and word.speaker_ref == speaker_ref:
            words.append(replace(word, display_label=label))
        else:
            words.append(word)
    speakers = recording.speakers
    if assigned_speaker is not None:
        if any(speaker.speaker_ref == assigned_speaker.speaker_ref for speaker in speakers):
            speakers = tuple(
                assigned_speaker if speaker.speaker_ref == assigned_speaker.speaker_ref else speaker
                for speaker in speakers
            )
        else:
            speakers = speakers + (assigned_speaker,)
    return replace(recording, speakers=speakers, words=tuple(words))


def _mark_uncertain(
    recording: CanonicalRecording,
    uncertain_word_ids: frozenset[str],
    overlay: MarkUncertainOverlay,
) -> frozenset[str]:
    word_ids = _validate_word_ids(recording, overlay.word_ids, "mark_uncertain.word_ids")
    if overlay.uncertain:
        return uncertain_word_ids | frozenset(word_ids)
    return frozenset(word_id for word_id in uncertain_word_ids if word_id not in word_ids)


def _mark_overlap(recording: CanonicalRecording, overlay: MarkOverlapOverlay) -> CanonicalRecording:
    word_ids = _validate_word_ids(recording, overlay.word_ids, "mark_overlap.word_ids")
    return replace(
        recording,
        words=tuple(
            replace(word, overlap=overlay.overlap) if word.word_id in word_ids else word for word in recording.words
        ),
    )


def _build_turns(
    words: tuple[CanonicalWord, ...],
    uncertain_word_ids: frozenset[str],
    max_gap_ms: int,
    split_after_punctuation: bool,
    min_speaker_confidence: float,
    suppress_labels: bool,
) -> list[RenderedTurn]:
    turns: list[RenderedTurn] = []
    current: list[CanonicalWord] = []
    current_end_ms = 0
    for word in words:
        if current and _starts_new_turn(current[-1], word, current_end_ms, max_gap_ms, split_after_punctuation):
            turns.append(
                _render_turn(
                    len(turns) + 1,
                    current,
                    uncertain_word_ids,
                    min_speaker_confidence,
                    suppress_labels,
                )
            )
            current = []
            current_end_ms = 0
        current.append(word)
        current_end_ms = max(current_end_ms, word.end_ms)
    if current:
        turns.append(
            _render_turn(
                len(turns) + 1,
                current,
                uncertain_word_ids,
                min_speaker_confidence,
                suppress_labels,
            )
        )
    return turns


def _starts_new_turn(
    previous: CanonicalWord,
    current: CanonicalWord,
    current_turn_end_ms: int,
    max_gap_ms: int,
    split_after_punctuation: bool,
) -> bool:
    if previous.speaker_ref != current.speaker_ref:
        return True
    if previous.channel_id != current.channel_id:
        return True
    if previous.overlap != current.overlap:
        return True
    if current.start_ms - current_turn_end_ms > max_gap_ms:
        return True
    if split_after_punctuation and previous.text.rstrip().endswith((".", "?", "!")):
        return True
    return False


def _render_turn(
    index: int,
    words: list[CanonicalWord],
    uncertain_word_ids: frozenset[str],
    min_speaker_confidence: float,
    suppress_labels: bool,
) -> RenderedTurn:
    review_reasons = _combine_review_reasons(
        *(_word_review_reasons(word, uncertain_word_ids, min_speaker_confidence, suppress_labels) for word in words)
    )
    return RenderedTurn(
        turn_id=f"turn_{index}",
        start_ms=words[0].start_ms,
        end_ms=max(word.end_ms for word in words),
        label=_word_label(words[0], suppress_labels),
        word_ids=tuple(word.word_id for word in words),
        text=_join_words(words),
        display_label=_word_display_label(words[0], suppress_labels),
        channel_id=words[0].channel_id,
        uncertain=any(word.word_id in uncertain_word_ids or word.speaker_confidence is None for word in words),
        overlap=any(word.overlap for word in words),
        speaker_attribution=_speaker_attribution_from_reasons(review_reasons),
        review_reasons=review_reasons,
    )


def _render_word(
    word: CanonicalWord,
    uncertain_word_ids: frozenset[str],
    min_speaker_confidence: float,
    suppress_labels: bool,
) -> RenderedWord:
    review_reasons = _word_review_reasons(word, uncertain_word_ids, min_speaker_confidence, suppress_labels)
    return RenderedWord(
        word_id=word.word_id,
        text=word.text,
        start_ms=word.start_ms,
        end_ms=word.end_ms,
        label=_word_label(word, suppress_labels),
        display_label=_word_display_label(word, suppress_labels),
        channel_id=word.channel_id,
        speaker_confidence=word.speaker_confidence,
        uncertain=word.word_id in uncertain_word_ids or word.speaker_confidence is None,
        overlap=word.overlap,
        speaker_attribution=_speaker_attribution_from_reasons(review_reasons),
        review_reasons=review_reasons,
    )


def _word_review_reasons(
    word: CanonicalWord,
    uncertain_word_ids: frozenset[str],
    min_speaker_confidence: float,
    suppress_labels: bool,
) -> tuple[ReviewReason, ...]:
    reasons: list[ReviewReason] = []
    if suppress_labels or word.speaker_ref is None or word.display_label is None:
        reasons.append("speaker_attribution_unavailable")
    elif word.speaker_confidence is None:
        reasons.append("missing_speaker_confidence")
    elif word.speaker_confidence < min_speaker_confidence:
        reasons.append("low_speaker_confidence")
    if word.overlap:
        reasons.append("overlap_detected")
    if word.word_id in uncertain_word_ids:
        reasons.append("manual_uncertain")
    return _combine_review_reasons(reasons)


def _combine_review_reasons(*reason_groups: tuple[ReviewReason, ...] | list[ReviewReason]) -> tuple[ReviewReason, ...]:
    seen = {reason for reasons in reason_groups for reason in reasons}
    for reason in seen:
        _validate_review_reason(reason)
    return tuple(reason for reason in _REVIEW_REASON_ORDER if reason in seen)


def _speaker_attribution_from_reasons(review_reasons: tuple[ReviewReason, ...]) -> SpeakerAttributionState:
    if "speaker_attribution_unavailable" in review_reasons:
        return "unavailable"
    if review_reasons:
        return "unreliable"
    return "available"


def _transcript_speaker_attribution(
    state: RenderedTranscriptState,
    rendered_words: tuple[RenderedWord, ...],
) -> SpeakerAttributionState:
    if state in _LABEL_SUPPRESSED_STATES:
        return "unavailable"
    if any(word.speaker_attribution == "unavailable" for word in rendered_words):
        return "unavailable"
    if state != "confident_pipeline" or any(word.speaker_attribution == "unreliable" for word in rendered_words):
        return "unreliable"
    return "available"


def _resolve_transcript_state(
    requested_state: RenderedTranscriptState | None,
    review_reasons: tuple[ReviewReason, ...],
) -> RenderedTranscriptState:
    if requested_state is not None and requested_state != "confident_pipeline":
        return requested_state
    if "speaker_attribution_unavailable" in review_reasons:
        return "speaker_attribution_unavailable"
    if review_reasons:
        return "needs_review"
    return "confident_pipeline"


def _join_words(words: list[CanonicalWord]) -> str:
    text = ""
    for word in words:
        if not text or word.text in {".", ",", "?", "!", ":", ";"}:
            text += word.text
        else:
            text += " " + word.text
    return text


def _word_label(word: CanonicalWord, suppress_labels: bool) -> str | None:
    if suppress_labels:
        return None
    return None if word.display_label is None else word.display_label.label


def _word_display_label(word: CanonicalWord, suppress_labels: bool) -> DisplayLabel | None:
    if suppress_labels:
        return None
    return word.display_label


def _display_label(label: str, source: str) -> DisplayLabel:
    return DisplayLabel(label=label, source=source, scope="recording", source_ref=None)


def _ensure_speaker(recording: CanonicalRecording, speaker_ref: str, label: str | None) -> SpeakerRecord:
    speaker_ref = _require_ref(speaker_ref, "speaker_ref")
    for speaker in recording.speakers:
        if speaker.speaker_ref == speaker_ref:
            if label is None:
                return speaker
            return replace(speaker, display_label=_display_label(label, "reviewer_rename"))
    return SpeakerRecord(
        speaker_ref=speaker_ref,
        display_label=_display_label(label or _next_person_label(recording), "reviewer_rename"),
    )


def _next_person_label(recording: CanonicalRecording) -> str:
    highest = 0
    for speaker in recording.speakers:
        if speaker.display_label is None:
            continue
        prefix, _, suffix = speaker.display_label.label.partition("_")
        if prefix == "person" and suffix.isdigit():
            highest = max(highest, int(suffix))
    return f"person_{highest + 1}"


def _validate_speaker_exists(recording: CanonicalRecording, speaker_ref: str) -> SpeakerRecord:
    speaker_ref = _require_ref(speaker_ref, "speaker_ref")
    for speaker in recording.speakers:
        if speaker.speaker_ref == speaker_ref:
            return speaker
    raise ValidationError(f"unknown speaker_ref: {speaker_ref}")


def _validate_new_speaker(recording: CanonicalRecording, speaker_ref: str) -> str:
    speaker_ref = _require_ref(speaker_ref, "speaker_ref")
    if any(speaker.speaker_ref == speaker_ref for speaker in recording.speakers):
        raise ValidationError(f"speaker_ref already exists: {speaker_ref}")
    return speaker_ref


def _validate_word_ids(recording: CanonicalRecording, word_ids: tuple[str, ...], field_name: str) -> set[str]:
    result = {_require_ref(word_id, field_name) for word_id in word_ids}
    if not result:
        raise ValidationError(f"{field_name} is required")
    known = {word.word_id for word in recording.words}
    unknown = sorted(result - known)
    if unknown:
        raise ValidationError(f"{field_name} references unknown word_id: {unknown[0]}")
    return result


def _validate_operation_id(value: str) -> str:
    return _require_ref(value, "overlay.operation_id")


def _validate_degraded_state(value: str | None) -> RenderedTranscriptState | None:
    if value is None:
        return None
    value = _require_ref(value, "degraded_state")
    if value not in _ALLOWED_RENDERED_TRANSCRIPT_STATES:
        raise ValidationError(f"degraded_state is not supported: {value}")
    return value  # type: ignore[return-value]


def _validate_review_reasons(values: tuple[ReviewReason, ...]) -> tuple[ReviewReason, ...]:
    try:
        reasons = tuple(values)
    except TypeError as exc:
        raise ValidationError("review_reasons must be an iterable") from exc
    return _combine_review_reasons(list(reasons))


def _validate_review_reason(value: object) -> ReviewReason:
    value = _require_ref(value, "review_reasons")
    if value not in _ALLOWED_REVIEW_REASONS:
        raise ValidationError(f"review_reasons contains unsupported reason: {value}")
    return value  # type: ignore[return-value]


def _validate_min_speaker_confidence(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError("min_speaker_confidence must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise ValidationError("min_speaker_confidence must be a finite number")
    if not 0.0 <= value <= 1.0:
        raise ValidationError("min_speaker_confidence must be between 0.0 and 1.0")
    return value


def _require_ref(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{field_name} must be a string")
    value = value.strip()
    if not value:
        raise ValidationError(f"{field_name} is required")
    return value


def _ensure_valid_interval(start_ms: int, end_ms: int, context: str) -> None:
    if isinstance(start_ms, bool) or not isinstance(start_ms, int):
        raise ValidationError(f"{context}.start_ms must be an integer")
    if isinstance(end_ms, bool) or not isinstance(end_ms, int):
        raise ValidationError(f"{context}.end_ms must be an integer")
    if start_ms < 0:
        raise ValidationError(f"{context}.start_ms must be >= 0")
    if end_ms <= start_ms:
        raise ValidationError(f"{context}.end_ms must be greater than start_ms")


def _word_within(word: CanonicalWord, start_ms: int, end_ms: int) -> bool:
    return word.start_ms >= start_ms and word.end_ms <= end_ms
