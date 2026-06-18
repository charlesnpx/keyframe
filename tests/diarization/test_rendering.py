from keyframe.diarization import (
    AssignSpanOverlay,
    CanonicalRecording,
    CanonicalWord,
    ChannelRecord,
    MarkOverlapOverlay,
    MarkUncertainOverlay,
    MergeSpeakersOverlay,
    RenameLabelOverlay,
    SpeakerRecord,
    SplitSpeakerOverlay,
    render_transcript,
)


def _recording(**overrides):
    base = {
        "recording_id": "rec-rendering",
        "original_audio_id": "original-local-fixture",
        "canonical_audio_id": "canonical-local-fixture",
        "timeline_id": "timeline-rendering",
        "duration_ms": 4_000,
        "channels": (ChannelRecord("ch-1"), ChannelRecord("ch-2")),
        "speakers": (SpeakerRecord("spk-a"), SpeakerRecord("spk-b")),
        "words": (
            CanonicalWord("w-1", "hello", 0, 100, speaker_ref="spk-a", channel_id="ch-1", speaker_confidence=0.9),
            CanonicalWord("w-2", "there", 150, 250, speaker_ref="spk-a", channel_id="ch-1", speaker_confidence=0.9),
            CanonicalWord("w-3", "yes.", 300, 400, speaker_ref="spk-b", channel_id="ch-1", speaker_confidence=0.8),
            CanonicalWord("w-4", "next", 450, 550, speaker_ref="spk-b", channel_id="ch-1", speaker_confidence=0.8),
        ),
        "speaker_spans": (),
        "scoring_regions": (),
    }
    base.update(overrides)
    return CanonicalRecording(**base)


def test_turn_assembly_groups_by_speaker_punctuation_gap_channel_and_overlap():
    recording = _recording(
        words=(
            CanonicalWord("w-1", "hello", 0, 100, speaker_ref="spk-a", channel_id="ch-1", speaker_confidence=0.9),
            CanonicalWord("w-2", "there", 150, 250, speaker_ref="spk-a", channel_id="ch-1", speaker_confidence=0.9),
            CanonicalWord("w-3", "yes.", 300, 400, speaker_ref="spk-b", channel_id="ch-1", speaker_confidence=0.8),
            CanonicalWord("w-4", "next", 450, 550, speaker_ref="spk-b", channel_id="ch-1", speaker_confidence=0.8),
            CanonicalWord("w-5", "later", 2_000, 2_100, speaker_ref="spk-b", channel_id="ch-1", speaker_confidence=0.8),
            CanonicalWord("w-6", "right", 2_150, 2_250, speaker_ref="spk-b", channel_id="ch-2", speaker_confidence=0.8),
            CanonicalWord(
                "w-7",
                "overlap",
                2_300,
                2_400,
                speaker_ref="spk-b",
                channel_id="ch-2",
                speaker_confidence=0.8,
                overlap=True,
            ),
        ),
    )

    transcript = render_transcript(recording, max_gap_ms=900)

    assert [turn.word_ids for turn in transcript.turns] == [
        ("w-1", "w-2"),
        ("w-3",),
        ("w-4",),
        ("w-5",),
        ("w-6",),
        ("w-7",),
    ]
    assert transcript.turns[0].text == "hello there"
    assert transcript.turns[0].label == "person_1"
    assert transcript.turns[1].label == "person_2"
    assert transcript.turns[4].channel_id == "ch-2"
    assert transcript.turns[5].overlap is True


def test_rename_overlay_changes_rendered_labels_without_mutating_recording():
    recording = _recording()
    original = recording.to_dict()

    transcript = render_transcript(
        recording,
        overlays=(RenameLabelOverlay(operation_id="op-rename", speaker_ref="spk-a", label="Alex"),),
    )

    assert transcript.turns[0].label == "Alex"
    assert transcript.words[0].label == "Alex"
    assert recording.to_dict() == original


def test_merge_overlay_combines_clusters_before_turn_assembly():
    recording = _recording(
        words=(
            CanonicalWord("w-1", "hello", 0, 100, speaker_ref="spk-a", channel_id="ch-1", speaker_confidence=0.9),
            CanonicalWord("w-2", "there", 150, 250, speaker_ref="spk-b", channel_id="ch-1", speaker_confidence=0.8),
        )
    )

    transcript = render_transcript(
        recording,
        overlays=(
            MergeSpeakersOverlay(
                operation_id="op-merge",
                source_speaker_refs=("spk-b",),
                target_speaker_ref="spk-a",
            ),
        ),
    )

    assert len(transcript.turns) == 1
    assert transcript.turns[0].word_ids == ("w-1", "w-2")
    assert transcript.turns[0].label == "person_1"


def test_split_overlay_reassigns_a_time_span_to_a_transcript_local_speaker():
    recording = _recording(
        words=(
            CanonicalWord("w-1", "first", 0, 100, speaker_ref="spk-a", channel_id="ch-1", speaker_confidence=0.9),
            CanonicalWord("w-2", "second", 150, 250, speaker_ref="spk-a", channel_id="ch-1", speaker_confidence=0.9),
        )
    )

    transcript = render_transcript(
        recording,
        overlays=(
            SplitSpeakerOverlay(
                operation_id="op-split",
                source_speaker_ref="spk-a",
                new_speaker_ref="review-speaker-1",
                start_ms=150,
                end_ms=250,
                label="Blair",
            ),
        ),
    )

    assert [turn.label for turn in transcript.turns] == ["person_1", "Blair"]
    assert [turn.word_ids for turn in transcript.turns] == [("w-1",), ("w-2",)]
    assert transcript.words[1].speaker_confidence is None
    assert transcript.turns[1].uncertain is True


def test_assign_span_overlay_assigns_unknown_words():
    recording = _recording(
        words=(
            CanonicalWord("w-1", "known", 0, 100, speaker_ref="spk-a", channel_id="ch-1", speaker_confidence=0.9),
            CanonicalWord("w-2", "unknown", 150, 250, channel_id="ch-1", speaker_confidence=None),
        )
    )

    transcript = render_transcript(
        recording,
        overlays=(
            AssignSpanOverlay(
                operation_id="op-assign",
                speaker_ref="spk-b",
                start_ms=150,
                end_ms=250,
                label="Taylor",
            ),
        ),
    )

    assert transcript.words[1].label == "Taylor"
    assert transcript.words[1].uncertain is True
    assert transcript.turns[1].word_ids == ("w-2",)
    assert transcript.turns[1].label == "Taylor"


def test_assign_span_overlay_clears_stale_confidence_when_reassigning_words():
    recording = _recording(
        words=(
            CanonicalWord("w-1", "wrong", 0, 100, speaker_ref="spk-a", channel_id="ch-1", speaker_confidence=0.9),
        )
    )

    transcript = render_transcript(
        recording,
        overlays=(AssignSpanOverlay(operation_id="op-assign", speaker_ref="spk-b", start_ms=0, end_ms=100),),
    )

    assert transcript.words[0].label == "person_2"
    assert transcript.words[0].speaker_confidence is None
    assert transcript.words[0].uncertain is True


def test_mark_uncertain_and_overlap_overlays_are_transcript_local():
    recording = _recording()
    original = recording.to_dict()

    transcript = render_transcript(
        recording,
        overlays=(
            MarkOverlapOverlay(operation_id="op-overlap", word_ids=("w-2",)),
            MarkUncertainOverlay(operation_id="op-uncertain", word_ids=("w-2",)),
        ),
    )

    word = next(item for item in transcript.words if item.word_id == "w-2")
    assert word.uncertain is True
    assert word.overlap is True
    assert any(turn.word_ids == ("w-2",) and turn.overlap is True for turn in transcript.turns)
    assert recording.to_dict() == original


def test_rendered_transcript_payload_preserves_word_ids_and_overlay_provenance():
    transcript = render_transcript(
        _recording(),
        overlays=(
            MarkUncertainOverlay(operation_id="op-z-uncertain", word_ids=("w-1",)),
            MarkOverlapOverlay(operation_id="op-a-overlap", word_ids=("w-1",)),
        ),
    )

    payload = transcript.to_dict()

    assert payload["recording_id"] == "rec-rendering"
    assert payload["applied_overlay_ids"] == ["op-a-overlap", "op-z-uncertain"]
    assert payload["turns"][0]["word_ids"] == ["w-1"]
    assert payload["words"][0]["word_id"] == "w-1"
