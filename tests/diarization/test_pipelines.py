import pytest

from keyframe.diarization import (
    AudioTimelineProvenance,
    CanonicalRecording,
    CanonicalWord,
    ChannelRecord,
    DatasetCacheConfig,
    DiarizationEvaluationResult,
    EngineConfigMetadata,
    NormalizedArtifactProvenance,
    NormalizedEngineOutput,
    ReferenceBundle,
    SpeakerSpan,
    ValidationError,
    build_candidate_bundle,
    build_pipeline_branch_evaluation_case,
    create_pipeline_branch_run_record,
    read_dataset_manifest_json,
    render_branch_transcript,
    run_authenticated_track_metadata_branch,
    run_separate_track_branch,
    validate_pipeline_branch_payload,
)


FORBIDDEN_SAFE_PAYLOAD_KEYS = {
    "corpus_identity",
    "participant_id",
    "voice_fingerprint",
    "voice_profile",
    "original_audio_id",
    "canonical_audio_id",
    "local_audio_sha256",
}


def _recording(channel_ids=("left", "right"), channel_names=("Alex", "Blair")):
    return CanonicalRecording(
        recording_id="rec-two-track",
        original_audio_id="original-local-rec-two-track",
        canonical_audio_id="canonical-local-rec-two-track",
        timeline_id="timeline-rec-two-track",
        transform_chain_id="identity",
        duration_ms=1_200,
        sample_rate_hz=16_000,
        channels=tuple(
            ChannelRecord(channel_id, name=channel_names[index])
            for index, channel_id in enumerate(channel_ids)
        ),
    )


def _bundle(mode, channel_ids=("left", "right"), channel_names=("Alex", "Blair")):
    return build_candidate_bundle(
        ReferenceBundle.from_recording(
            _recording(channel_ids=channel_ids, channel_names=channel_names),
            artifact_id=f"reference-{mode}",
        ),
        bundle_id=f"candidate-{mode}",
        mode=mode,
    )


def _track_output(channel_id, output_id, words, *, channel_ids=("left", "right")):
    recording = _recording(channel_ids=channel_ids)
    spans = _spans_for_words(channel_id, output_id, words)
    return NormalizedEngineOutput(
        output_id=output_id,
        output_kind="word_spans",
        artifact=NormalizedArtifactProvenance(
            artifact_id=f"{output_id}:artifact",
            artifact_kind="candidate",
            timeline=AudioTimelineProvenance(
                original_audio_id=recording.original_audio_id,
                canonical_audio_id=recording.canonical_audio_id,
                timeline_id=recording.timeline_id,
                transform_chain_id=recording.transform_chain_id,
                sample_rate_hz=recording.sample_rate_hz,
                duration_ms=recording.duration_ms,
                channel_ids=(channel_id,),
            ),
        ),
        config=EngineConfigMetadata(
            adapter_id=f"test-track-{channel_id}",
            provider="test-provider",
            model_name="track-asr-diarizer",
        ),
        words=tuple(
            CanonicalWord(
                word_id=f"{output_id}:source-word:{index}",
                text=text,
                start_ms=start_ms,
                end_ms=end_ms,
                speaker_ref=speaker_ref,
                channel_id=word_channel_id,
                text_confidence=0.99,
                speaker_confidence=0.93,
            )
            for index, (text, start_ms, end_ms, speaker_ref, word_channel_id) in enumerate(words, start=1)
        ),
        speaker_spans=spans,
    )


def _spans_for_words(channel_id, output_id, words):
    result = []
    seen = []
    for _, _, _, speaker_ref, _ in words:
        if speaker_ref not in seen:
            seen.append(speaker_ref)
    for index, speaker_ref in enumerate(seen, start=1):
        speaker_words = [word for word in words if word[3] == speaker_ref]
        result.append(
            SpeakerSpan(
                span_id=f"{output_id}:source-span:{index}",
                speaker_ref=speaker_ref,
                start_ms=min(word[1] for word in speaker_words),
                end_ms=max(word[2] for word in speaker_words),
                channel_id=channel_id,
                confidence=0.93,
            )
        )
    return tuple(result)


def _branch_outputs():
    return (
        _track_output(
            "left",
            "left-track-output",
            (
                ("left-first", 200, 300, "raw-speaker-1", None),
                ("left-second", 500, 600, "raw-speaker-1", None),
            ),
        ),
        _track_output(
            "right",
            "right-track-output",
            (
                ("right-first", 100, 180, "raw-speaker-1", None),
                ("right-second", 500, 580, "raw-speaker-1", None),
            ),
        ),
    )


def _tied_time_outputs():
    return (
        _track_output(
            "left",
            "left-tied-output",
            (("left-tied", 100, 180, "raw-speaker-1", None),),
        ),
        _track_output(
            "right",
            "right-tied-output",
            (("right-tied", 100, 180, "raw-speaker-1", None),),
        ),
    )


def _colliding_channel_outputs():
    channel_ids = ("left:1", "left/1")
    return (
        _track_output(
            "left:1",
            "left-colon-output",
            (("colon-first", 100, 180, "raw-speaker-1", None),),
            channel_ids=channel_ids,
        ),
        _track_output(
            "left/1",
            "left-slash-output",
            (("slash-first", 200, 280, "raw-speaker-1", None),),
            channel_ids=channel_ids,
        ),
    )


def _walk_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def test_separate_track_branch_merges_asr_per_track_outputs_on_timeline():
    result = run_separate_track_branch(_bundle("product_realistic"), _branch_outputs(), output_id="merged-output")

    assert result.branch_id == "separate_tracks"
    assert result.output.output_id == "merged-output"
    assert result.output.artifact.timeline.channel_ids == ("left", "right")
    assert [word.text for word in result.output.words] == [
        "right-first",
        "left-first",
        "left-second",
        "right-second",
    ]
    assert [word.word_id for word in result.output.words] == [
        "merged-output:word:000001",
        "merged-output:word:000002",
        "merged-output:word:000003",
        "merged-output:word:000004",
    ]
    assert [word.channel_id for word in result.output.words] == ["right", "left", "left", "right"]
    assert [span.channel_id for span in result.output.speaker_spans] == ["right", "left"]
    assert [span.span_id for span in result.output.speaker_spans] == [
        "merged-output:span:000001",
        "merged-output:span:000002",
    ]
    assert result.to_dict()["branch_id"] == "separate_tracks"
    assert FORBIDDEN_SAFE_PAYLOAD_KEYS.isdisjoint(set(_walk_keys(result.to_dict())))


def test_track_merge_is_deterministic_for_reordered_engine_outputs():
    outputs = _tied_time_outputs()
    bundle = _bundle("product_realistic")

    first = run_separate_track_branch(bundle, outputs, output_id="tied-output")
    second = run_separate_track_branch(bundle, tuple(reversed(outputs)), output_id="tied-output")

    assert [word.to_dict() for word in first.output.words] == [word.to_dict() for word in second.output.words]
    assert [span.to_dict() for span in first.output.speaker_spans] == [
        span.to_dict() for span in second.output.speaker_spans
    ]
    assert [word.text for word in first.output.words] == ["left-tied", "right-tied"]


def test_same_raw_speaker_id_on_different_tracks_remains_channel_local():
    result = run_separate_track_branch(_bundle("product_realistic"), _branch_outputs())

    speaker_refs_by_channel = {
        word.channel_id: word.speaker_ref
        for word in result.output.words
        if word.text.endswith("first")
    }

    assert speaker_refs_by_channel == {
        "left": "separate_tracks:left:speaker_1",
        "right": "separate_tracks:right:speaker_1",
    }
    assert len(set(speaker_refs_by_channel.values())) == 2
    assert result.output.raw_speaker_evidence == ()


def test_sanitized_channel_ids_cannot_collapse_channel_local_speakers():
    channel_ids = ("left:1", "left/1")
    result = run_separate_track_branch(
        _bundle("product_realistic", channel_ids=channel_ids),
        _colliding_channel_outputs(),
    )

    speaker_refs_by_channel = {word.channel_id: word.speaker_ref for word in result.output.words}

    assert speaker_refs_by_channel == {
        "left:1": "separate_tracks:left%3A1:speaker_1",
        "left/1": "separate_tracks:left%2F1:speaker_1",
    }
    assert len(set(speaker_refs_by_channel.values())) == 2


def test_duplicate_per_track_outputs_are_rejected():
    duplicate_left = (
        *_branch_outputs(),
        _track_output(
            "left",
            "second-left-output",
            (("duplicate-left", 700, 780, "raw-speaker-2", None),),
        ),
    )

    with pytest.raises(ValidationError, match="duplicate per-track output for channel: left"):
        run_separate_track_branch(_bundle("product_realistic"), duplicate_left)


def test_authenticated_track_metadata_branch_renders_permitted_track_labels():
    bundle = _bundle("authenticated_track_metadata")
    result = run_authenticated_track_metadata_branch(bundle, _branch_outputs())

    transcript = render_branch_transcript(result, candidate_bundle=bundle)

    assert [word.label for word in transcript.words] == ["Blair", "Alex", "Alex", "Blair"]
    assert transcript.words[0].display_label.source == "channel_metadata"
    assert transcript.words[0].display_label.source_ref == "right"
    assert transcript.words[1].display_label.source == "channel_metadata"
    assert transcript.words[1].display_label.source_ref == "left"
    assert FORBIDDEN_SAFE_PAYLOAD_KEYS.isdisjoint(set(_walk_keys(transcript.to_dict())))


def test_track_metadata_rendering_is_rejected_outside_authenticated_mode():
    product_bundle = _bundle("product_realistic")
    metadata_bundle = _bundle("authenticated_track_metadata")
    separate_result = run_separate_track_branch(product_bundle, _branch_outputs())

    with pytest.raises(ValidationError, match="requires authenticated_track_metadata"):
        run_authenticated_track_metadata_branch(product_bundle, _branch_outputs())

    with pytest.raises(ValidationError, match="requires product_realistic"):
        run_separate_track_branch(metadata_bundle, _branch_outputs())

    with pytest.raises(ValidationError, match="requires product_realistic"):
        render_branch_transcript(separate_result, candidate_bundle=metadata_bundle)


def test_pipeline_payload_rejects_forbidden_corpus_identity_leakage():
    payload = {
        "branch_id": "separate_tracks",
        "candidate_bundle": _bundle("product_realistic").to_dict(),
        "corpus_identity": "ami-ES2002a",
    }

    with pytest.raises(ValidationError, match="corpus_identity is forbidden"):
        validate_pipeline_branch_payload(payload)

    payload = {
        "branch_id": "authenticated_track_metadata",
        "candidate_bundle": _bundle("authenticated_track_metadata").to_dict(),
    }
    payload["candidate_bundle"]["channels"][0]["participant_id"] = "AMI-P1"

    with pytest.raises(ValidationError, match="participant_id is forbidden"):
        validate_pipeline_branch_payload(payload)


def test_pipeline_branch_ids_flow_into_run_records_and_report_cases(tmp_path):
    result = run_separate_track_branch(_bundle("product_realistic"), _branch_outputs())
    manifest = read_dataset_manifest_json("keyframe/diarization/dataset_manifests/ami.json")

    record = create_pipeline_branch_run_record(
        run_id="run-separate-tracks",
        manifest=manifest,
        split_id="ami-public-dev",
        branch_id=result.branch_id,
        artifact_root=tmp_path / "artifacts",
        cache=DatasetCacheConfig(cache_root=str(tmp_path / "cache")),
        evaluated_split_ids=("ami-public-dev",),
    )

    evaluation = DiarizationEvaluationResult(
        recording_id="rec-two-track",
        output_id=result.output.output_id,
        scoring_policy={"policy_id": "diagnostic-diarization-v1", "version": "1"},
        speaker_mapping={},
        slices=(),
        recording_metrics=(),
        slice_metrics=(),
        reference_artifact={"artifact_id": "reference", "artifact_kind": "reference"},
        candidate_artifact={"artifact_id": "candidate", "artifact_kind": "candidate"},
    )
    case = build_pipeline_branch_evaluation_case(
        corpus_id="ami",
        result=result,
        evaluation=evaluation,
        scored_duration_ms=1_200,
        scored_words=len(result.output.words),
        scored_speaker_turns=len(result.output.speaker_spans),
    )

    assert record.branch == "separate_tracks"
    assert case.branch_id == "separate_tracks"
    assert case.evaluation.output_id == result.output.output_id
