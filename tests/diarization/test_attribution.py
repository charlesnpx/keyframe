from keyframe.diarization import (
    CanonicalRecording,
    CanonicalWord,
    ChannelRecord,
    SpeakerRecord,
    SpeakerSpan,
    apply_session_local_attribution,
)


def _recording(**overrides):
    base = {
        "recording_id": "rec-attribution",
        "original_audio_id": "original-local-fixture",
        "canonical_audio_id": "canonical-local-fixture",
        "timeline_id": "timeline-attribution",
        "duration_ms": 2_000,
        "channels": (ChannelRecord("ch-1"), ChannelRecord("ch-2")),
        "speakers": (SpeakerRecord("backend-b"), SpeakerRecord("backend-a")),
        "words": (
            CanonicalWord("w-1", "first", 100, 200, channel_id="ch-1"),
            CanonicalWord("w-2", "second", 500, 600, channel_id="ch-1"),
        ),
        "speaker_spans": (
            SpeakerSpan("span-a", "backend-a", 90, 250, channel_id="ch-1", confidence=0.91),
            SpeakerSpan("span-b", "backend-b", 450, 700, channel_id="ch-1", confidence=0.83),
        ),
        "scoring_regions": (),
    }
    base.update(overrides)
    return CanonicalRecording(**base)


def test_session_local_labels_follow_first_heard_speaker_not_backend_order():
    attributed = apply_session_local_attribution(_recording())

    speaker_labels = {speaker.speaker_ref: speaker.display_label for speaker in attributed.speakers}
    assert speaker_labels["backend-a"].label == "person_1"
    assert speaker_labels["backend-b"].label == "person_2"
    assert all(label.scope == "recording" for label in speaker_labels.values())
    assert all(label.source == "diarization_cluster" for label in speaker_labels.values())
    assert all(label.source_ref is None for label in speaker_labels.values())

    assert attributed.words[0].speaker_ref == "backend-a"
    assert attributed.words[0].display_label.label == "person_1"
    assert attributed.words[0].display_label.source_ref is None
    assert attributed.words[1].speaker_ref == "backend-b"
    assert attributed.words[1].display_label.label == "person_2"


def test_session_local_attribution_is_deterministic_across_repeated_runs():
    recording = _recording()

    first = apply_session_local_attribution(recording)
    second = apply_session_local_attribution(recording)

    assert first.to_dict() == second.to_dict()


def test_same_time_label_order_never_falls_back_to_backend_speaker_ref_sorting():
    recording = _recording(
        speakers=(SpeakerRecord("backend-a"), SpeakerRecord("backend-z")),
        words=(
            CanonicalWord("w-unknown", "unknown", 0, 50, channel_id="ch-2"),
            CanonicalWord("w-known", "known", 100, 200, speaker_ref="backend-z", channel_id="ch-2"),
        ),
        speaker_spans=(SpeakerSpan("span-a", "backend-a", 100, 200, channel_id="ch-1", confidence=0.91),),
    )

    attributed = apply_session_local_attribution(recording)
    speaker_labels = {speaker.speaker_ref: speaker.display_label.label for speaker in attributed.speakers}

    assert speaker_labels == {"backend-a": "person_2", "backend-z": "person_1"}


def test_unknown_speaker_words_remain_unknown_without_invented_confidence():
    recording = _recording(
        speakers=(),
        words=(CanonicalWord("w-unknown", "unknown", 100, 200, channel_id="ch-1"),),
        speaker_spans=(),
    )

    attributed = apply_session_local_attribution(recording)

    assert attributed.words[0].speaker_ref is None
    assert attributed.words[0].speaker_confidence is None
    assert attributed.words[0].display_label is None
    assert attributed.words[0].overlap is False


def test_overlap_sets_overlap_flag_without_inventing_missing_confidence():
    recording = _recording(
        speakers=(SpeakerRecord("backend-a"), SpeakerRecord("backend-b")),
        words=(CanonicalWord("w-overlap", "together", 200, 500, channel_id="ch-1"),),
        speaker_spans=(
            SpeakerSpan("span-a", "backend-a", 100, 550, channel_id="ch-1", confidence=None, overlap=True),
            SpeakerSpan("span-b", "backend-b", 250, 450, channel_id="ch-1", confidence=0.72, overlap=True),
        ),
    )

    attributed = apply_session_local_attribution(recording)

    assert attributed.words[0].speaker_ref == "backend-a"
    assert attributed.words[0].display_label.label == "person_1"
    assert attributed.words[0].speaker_confidence is None
    assert attributed.words[0].overlap is True


def test_existing_word_speaker_counts_when_detecting_overlap():
    recording = _recording(
        speakers=(SpeakerRecord("backend-a"), SpeakerRecord("backend-b")),
        words=(CanonicalWord("w-existing", "known", 200, 500, speaker_ref="backend-a", channel_id="ch-1"),),
        speaker_spans=(SpeakerSpan("span-b", "backend-b", 250, 450, channel_id="ch-1", confidence=0.72),),
    )

    attributed = apply_session_local_attribution(recording)

    assert attributed.words[0].speaker_ref == "backend-a"
    assert attributed.words[0].display_label.label == "person_1"
    assert attributed.words[0].overlap is True


def test_word_attribution_respects_channel_ids():
    recording = _recording(
        words=(CanonicalWord("w-ch2", "right", 100, 200, channel_id="ch-2"),),
        speaker_spans=(SpeakerSpan("span-a", "backend-a", 90, 250, channel_id="ch-1", confidence=0.91),),
    )

    attributed = apply_session_local_attribution(recording)

    assert attributed.words[0].speaker_ref is None
    assert attributed.words[0].speaker_confidence is None
    assert attributed.words[0].display_label is None


def test_existing_backend_speaker_refs_remain_evidence_only_in_display_labels():
    recording = _recording(
        words=(CanonicalWord("w-existing", "known", 100, 200, speaker_ref="backend-b", channel_id="ch-1"),),
        speaker_spans=(SpeakerSpan("span-b", "backend-b", 90, 250, channel_id="ch-1", confidence=0.64),),
    )

    attributed = apply_session_local_attribution(recording)
    payload = attributed.to_dict()

    assert payload["words"][0]["speaker_ref"] == "backend-b"
    assert payload["words"][0]["display_label"] == {
        "confidence": None,
        "label": "person_1",
        "scope": "recording",
        "source": "diarization_cluster",
        "source_ref": None,
    }
    assert payload["speakers"][0]["display_label"]["label"] == "person_1"
    assert payload["speakers"][0]["display_label"]["source_ref"] is None
