"""Rendered transcript turns and transcript-local correction overlays."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

from keyframe.diarization.attribution import apply_session_local_attribution
from keyframe.diarization.models import CanonicalRecording, CanonicalWord, DisplayLabel, SpeakerRecord, ValidationError


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["word_ids"] = list(self.word_ids)
        return payload


@dataclass(frozen=True)
class RenderedTranscript:
    """Rendered transcript JSON plus overlay provenance."""

    recording_id: str
    turns: tuple[RenderedTurn, ...]
    words: tuple[RenderedWord, ...]
    applied_overlay_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied_overlay_ids": list(self.applied_overlay_ids),
            "recording_id": self.recording_id,
            "turns": [turn.to_dict() for turn in self.turns],
            "words": [word.to_dict() for word in self.words],
        }


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
    max_gap_ms: int = 900,
    split_after_punctuation: bool = True,
) -> RenderedTranscript:
    """Return transcript turns assembled from immutable canonical evidence."""

    if not isinstance(recording, CanonicalRecording):
        raise ValidationError("recording must be a CanonicalRecording")
    if max_gap_ms < 0:
        raise ValidationError("max_gap_ms must be >= 0")

    attributed = apply_session_local_attribution(recording)
    ordered_overlays = _ordered_overlays(overlays)
    overlay_result = _apply_overlays(attributed, ordered_overlays)
    words = tuple(sorted(overlay_result.recording.words, key=lambda word: (word.start_ms, word.end_ms, word.word_id)))
    rendered_words = tuple(_render_word(word, overlay_result.uncertain_word_ids) for word in words)
    turns = tuple(_build_turns(words, overlay_result.uncertain_word_ids, max_gap_ms, split_after_punctuation))
    return RenderedTranscript(
        recording_id=recording.recording_id,
        turns=turns,
        words=rendered_words,
        applied_overlay_ids=tuple(overlay.operation_id for overlay in ordered_overlays),
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
    return tuple(sorted(result, key=lambda item: item.operation_id))


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
    _validate_speaker_exists(recording, overlay.speaker_ref)
    label = _display_label(overlay.label, "reviewer_rename")
    speakers = tuple(
        replace(speaker, display_label=label) if speaker.speaker_ref == overlay.speaker_ref else speaker
        for speaker in recording.speakers
    )
    words = tuple(
        replace(word, display_label=label) if word.speaker_ref == overlay.speaker_ref else word
        for word in recording.words
    )
    return replace(recording, speakers=speakers, words=words)


def _merge_speakers(recording: CanonicalRecording, overlay: MergeSpeakersOverlay) -> CanonicalRecording:
    target = _validate_speaker_exists(recording, overlay.target_speaker_ref)
    source_refs = tuple(_require_ref(value, "merge_speakers.source_speaker_refs") for value in overlay.source_speaker_refs)
    if not source_refs:
        raise ValidationError("merge_speakers.source_speaker_refs is required")
    for speaker_ref in source_refs:
        _validate_speaker_exists(recording, speaker_ref)
    merged_refs = set(source_refs)
    words = tuple(
        replace(
            word,
            speaker_ref=overlay.target_speaker_ref,
            display_label=target.display_label,
        )
        if word.speaker_ref in merged_refs
        else word
        for word in recording.words
    )
    return replace(recording, words=words)


def _split_speaker(recording: CanonicalRecording, overlay: SplitSpeakerOverlay) -> CanonicalRecording:
    _validate_speaker_exists(recording, overlay.source_speaker_ref)
    _ensure_valid_interval(overlay.start_ms, overlay.end_ms, "split_speaker")
    _validate_new_speaker(recording, overlay.new_speaker_ref)
    label = _display_label(overlay.label or _next_person_label(recording), "reviewer_rename")
    new_speaker = SpeakerRecord(speaker_ref=overlay.new_speaker_ref, display_label=label)
    words = tuple(
        replace(word, speaker_ref=overlay.new_speaker_ref, speaker_confidence=None, display_label=label)
        if word.speaker_ref == overlay.source_speaker_ref and _word_within(word, overlay.start_ms, overlay.end_ms)
        else word
        for word in recording.words
    )
    return replace(recording, speakers=recording.speakers + (new_speaker,), words=words)


def _assign_span(recording: CanonicalRecording, overlay: AssignSpanOverlay) -> CanonicalRecording:
    _ensure_valid_interval(overlay.start_ms, overlay.end_ms, "assign_span")
    label = None
    assigned_speaker = None
    if overlay.speaker_ref is not None:
        assigned_speaker = _ensure_speaker(recording, overlay.speaker_ref, overlay.label)
        label = assigned_speaker.display_label
    words = tuple(
        replace(
            word,
            speaker_ref=overlay.speaker_ref,
            speaker_confidence=(word.speaker_confidence if word.speaker_ref == overlay.speaker_ref else None),
            display_label=label,
        )
        if _word_within(word, overlay.start_ms, overlay.end_ms)
        else word
        for word in recording.words
    )
    speakers = recording.speakers
    if assigned_speaker is not None:
        if any(speaker.speaker_ref == assigned_speaker.speaker_ref for speaker in speakers):
            speakers = tuple(
                assigned_speaker if speaker.speaker_ref == assigned_speaker.speaker_ref else speaker
                for speaker in speakers
            )
        else:
            speakers = speakers + (assigned_speaker,)
    return replace(recording, speakers=speakers, words=words)


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
) -> list[RenderedTurn]:
    turns: list[RenderedTurn] = []
    current: list[CanonicalWord] = []
    for word in words:
        if current and _starts_new_turn(current[-1], word, max_gap_ms, split_after_punctuation):
            turns.append(_render_turn(len(turns) + 1, current, uncertain_word_ids))
            current = []
        current.append(word)
    if current:
        turns.append(_render_turn(len(turns) + 1, current, uncertain_word_ids))
    return turns


def _starts_new_turn(
    previous: CanonicalWord,
    current: CanonicalWord,
    max_gap_ms: int,
    split_after_punctuation: bool,
) -> bool:
    if _word_label(previous) != _word_label(current):
        return True
    if previous.channel_id != current.channel_id:
        return True
    if previous.overlap != current.overlap:
        return True
    if current.start_ms - previous.end_ms > max_gap_ms:
        return True
    if split_after_punctuation and previous.text.rstrip().endswith((".", "?", "!")):
        return True
    return False


def _render_turn(index: int, words: list[CanonicalWord], uncertain_word_ids: frozenset[str]) -> RenderedTurn:
    return RenderedTurn(
        turn_id=f"turn_{index}",
        start_ms=words[0].start_ms,
        end_ms=words[-1].end_ms,
        label=_word_label(words[0]),
        word_ids=tuple(word.word_id for word in words),
        text=_join_words(words),
        channel_id=words[0].channel_id,
        uncertain=any(word.word_id in uncertain_word_ids or word.speaker_confidence is None for word in words),
        overlap=any(word.overlap for word in words),
    )


def _render_word(word: CanonicalWord, uncertain_word_ids: frozenset[str]) -> RenderedWord:
    return RenderedWord(
        word_id=word.word_id,
        text=word.text,
        start_ms=word.start_ms,
        end_ms=word.end_ms,
        label=_word_label(word),
        channel_id=word.channel_id,
        speaker_confidence=word.speaker_confidence,
        uncertain=word.word_id in uncertain_word_ids or word.speaker_confidence is None,
        overlap=word.overlap,
    )


def _join_words(words: list[CanonicalWord]) -> str:
    text = ""
    for word in words:
        if not text or word.text in {".", ",", "?", "!", ":", ";"}:
            text += word.text
        else:
            text += " " + word.text
    return text


def _word_label(word: CanonicalWord) -> str | None:
    return None if word.display_label is None else word.display_label.label


def _display_label(label: str, source: str) -> DisplayLabel:
    return DisplayLabel(label=label, source=source, scope="recording", source_ref=None)


def _ensure_speaker(recording: CanonicalRecording, speaker_ref: str, label: str | None) -> SpeakerRecord:
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


def _validate_new_speaker(recording: CanonicalRecording, speaker_ref: str) -> None:
    speaker_ref = _require_ref(speaker_ref, "speaker_ref")
    if any(speaker.speaker_ref == speaker_ref for speaker in recording.speakers):
        raise ValidationError(f"speaker_ref already exists: {speaker_ref}")


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
