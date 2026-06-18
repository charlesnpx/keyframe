"""Session-local speaker label assignment and word attribution."""

from __future__ import annotations

from dataclasses import replace

from keyframe.diarization.models import (
    CanonicalRecording,
    CanonicalWord,
    DisplayLabel,
    LabelSource,
    SpeakerSpan,
    ValidationError,
)


def apply_session_local_attribution(
    recording: CanonicalRecording,
    *,
    label_source: LabelSource = "diarization_cluster",
) -> CanonicalRecording:
    """Return a recording with deterministic session-local display labels."""

    if not isinstance(recording, CanonicalRecording):
        raise ValidationError("recording must be a CanonicalRecording")

    attributed_words = tuple(_attribute_word(word, recording.speaker_spans) for word in recording.words)
    label_by_speaker_ref = _assign_display_labels(recording, attributed_words)
    speakers = tuple(
        replace(
            speaker,
            display_label=_display_label(label_by_speaker_ref[speaker.speaker_ref], label_source),
        )
        for speaker in recording.speakers
    )
    words = tuple(
        replace(
            word,
            display_label=(
                _display_label(label_by_speaker_ref[word.speaker_ref], label_source)
                if word.speaker_ref is not None
                else None
            ),
        )
        for word in attributed_words
    )
    return replace(recording, speakers=speakers, words=words)


def _attribute_word(word: CanonicalWord, spans: tuple[SpeakerSpan, ...]) -> CanonicalWord:
    matching_spans = tuple(span for span in spans if _overlap_ms(word, span) > 0 and _channels_match(word, span))
    selected_span = _select_span_for_word(word, matching_spans)
    speaker_ref = word.speaker_ref
    speaker_confidence = word.speaker_confidence
    if speaker_ref is None and selected_span is not None:
        speaker_ref = selected_span.speaker_ref
    if speaker_confidence is None and selected_span is not None and selected_span.speaker_ref == speaker_ref:
        speaker_confidence = selected_span.confidence

    overlapping_speakers = {span.speaker_ref for span in matching_spans}
    if speaker_ref is not None:
        overlapping_speakers.add(speaker_ref)
    overlap = word.overlap or len(overlapping_speakers) > 1
    if selected_span is not None:
        overlap = overlap or selected_span.overlap

    return replace(
        word,
        speaker_ref=speaker_ref,
        speaker_confidence=speaker_confidence,
        overlap=overlap,
        display_label=None,
    )


def _select_span_for_word(word: CanonicalWord, spans: tuple[SpeakerSpan, ...]) -> SpeakerSpan | None:
    if not spans:
        return None
    if word.speaker_ref is not None:
        same_speaker_spans = tuple(span for span in spans if span.speaker_ref == word.speaker_ref)
        if same_speaker_spans:
            spans = same_speaker_spans
        else:
            return None
    return min(spans, key=lambda span: (-_overlap_ms(word, span), span.start_ms, span.span_id))


def _assign_display_labels(
    recording: CanonicalRecording,
    words: tuple[CanonicalWord, ...],
) -> dict[str, str]:
    seen: set[str] = set()
    ordered_refs: list[str] = []
    events: list[tuple[int, int, str]] = []
    for index, word in enumerate(words):
        if word.speaker_ref is not None:
            events.append((word.start_ms, index, word.speaker_ref))
    span_offset = len(events)
    for index, span in enumerate(recording.speaker_spans):
        events.append((span.start_ms, span_offset + index, span.speaker_ref))

    for _, _, speaker_ref in sorted(events):
        if speaker_ref not in seen:
            seen.add(speaker_ref)
            ordered_refs.append(speaker_ref)
    for speaker in recording.speakers:
        if speaker.speaker_ref not in seen:
            seen.add(speaker.speaker_ref)
            ordered_refs.append(speaker.speaker_ref)

    return {speaker_ref: f"person_{index}" for index, speaker_ref in enumerate(ordered_refs, start=1)}


def _display_label(label: str, source: LabelSource) -> DisplayLabel:
    return DisplayLabel(label=label, source=source, scope="recording", source_ref=None)


def _overlap_ms(word: CanonicalWord, span: SpeakerSpan) -> int:
    return max(0, min(word.end_ms, span.end_ms) - max(word.start_ms, span.start_ms))


def _channels_match(word: CanonicalWord, span: SpeakerSpan) -> bool:
    if word.channel_id is None or span.channel_id is None:
        return True
    return word.channel_id == span.channel_id
