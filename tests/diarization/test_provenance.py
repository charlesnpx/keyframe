import pytest

from keyframe.diarization import (
    AudioTimelineProvenance,
    NormalizedArtifactProvenance,
    OffsetMapSegment,
    TimelineOffsetMap,
    TransformChain,
    TransformStep,
    ValidationError,
    boundary_shift_degrades_scoring,
    read_recording_json,
    validate_timeline_merge,
)


def _timeline(**overrides):
    base = {
        "original_audio_id": "original-audio-local",
        "canonical_audio_id": "canonical-audio-local",
        "timeline_id": "timeline-a",
        "transform_chain_id": "identity",
        "sample_rate_hz": 16_000,
        "duration_ms": 2_000,
        "channel_ids": ("ch-1",),
        "time_basis": "canonical_ms",
    }
    base.update(overrides)
    return AudioTimelineProvenance(**base)


def _artifact(kind, timeline=None):
    return NormalizedArtifactProvenance(
        artifact_id=f"{kind}-artifact",
        artifact_kind=kind,
        timeline=timeline or _timeline(),
    )


def _assert_no_local_audio_identity(payload):
    if isinstance(payload, dict):
        assert "original_audio_id" not in payload
        assert "canonical_audio_id" not in payload
        assert "local_audio_sha256" not in payload
        for value in payload.values():
            _assert_no_local_audio_identity(value)
    elif isinstance(payload, list):
        for value in payload:
            _assert_no_local_audio_identity(value)


def test_recording_provenance_includes_required_normalized_artifact_fields():
    recording = read_recording_json("tests/diarization/fixtures/clean_two_speaker.json")

    provenance = AudioTimelineProvenance.from_recording(recording, local_audio_sha256="fixture-sha")

    assert provenance.original_audio_id == recording.original_audio_id
    assert provenance.canonical_audio_id == recording.canonical_audio_id
    assert provenance.timeline_id == recording.timeline_id
    assert provenance.transform_chain_id == recording.transform_chain_id
    assert provenance.sample_rate_hz == recording.sample_rate_hz
    assert provenance.duration_ms == recording.duration_ms
    assert provenance.channel_ids == ("ch-1",)
    assert provenance.time_basis == recording.time_basis


def test_timeline_channel_ids_must_be_a_nonempty_collection():
    with pytest.raises(ValidationError, match="channel_ids must be an iterable"):
        _timeline(channel_ids="ch-1")

    with pytest.raises(ValidationError, match="channel_ids is required"):
        _timeline(channel_ids=())


def test_audio_identity_and_hash_stay_out_of_rendered_and_monitoring_metadata():
    artifact = _artifact(
        "asr",
        _timeline(local_audio_sha256="fixture-local-cache-hash"),
    )

    integrity = artifact.to_integrity_dict()
    assert integrity["timeline"]["original_audio_id"] == "original-audio-local"
    assert integrity["timeline"]["canonical_audio_id"] == "canonical-audio-local"
    assert integrity["timeline"]["local_audio_sha256"] == "fixture-local-cache-hash"

    _assert_no_local_audio_identity(artifact.to_rendered_transcript_metadata())
    _assert_no_local_audio_identity(artifact.to_monitoring_metadata())
    _assert_no_local_audio_identity(artifact.to_cross_session_linking_metadata())
    assert "timeline_id" not in artifact.to_cross_session_linking_metadata()["timeline"]


def test_compatible_timeline_merge_is_direct():
    result = validate_timeline_merge(_artifact("asr"), _artifact("diarization"))

    assert result.direct_timeline_match is True
    assert result.offset_map_id is None


def test_timeline_mismatch_requires_validated_offset_map():
    source = _artifact("asr", _timeline(timeline_id="timeline-asr", transform_chain_id="asr-chunk"))
    target = _artifact("diarization", _timeline(timeline_id="timeline-diarization", transform_chain_id="diar-chain"))

    with pytest.raises(ValidationError, match="requires a validated offset map"):
        validate_timeline_merge(source, target)

    offset_map = TimelineOffsetMap(
        offset_map_id="offset-map-1",
        source_timeline_id="timeline-asr",
        target_timeline_id="timeline-diarization",
        source_transform_chain_id="asr-chunk",
        target_transform_chain_id="diar-chain",
        source_time_basis="canonical_ms",
        target_time_basis="canonical_ms",
        segments=(OffsetMapSegment(0, 1_700, 300, 2_000),),
    )

    result = validate_timeline_merge(source, target, offset_map=offset_map)

    assert result.direct_timeline_match is False
    assert result.offset_map_id == "offset-map-1"
    assert offset_map.convert_source_ms(250) == 550


@pytest.mark.parametrize(
    "target_overrides, message",
    [
        ({"duration_ms": 2_500}, "duration_ms conflicts"),
        ({"sample_rate_hz": 48_000}, "sample_rate_hz conflicts"),
        ({"channel_ids": ("ch-2",)}, "channel layout conflicts"),
    ],
)
def test_conflicting_audio_metadata_rejects_merge_even_with_offset_map(target_overrides, message):
    source = _timeline(timeline_id="source", transform_chain_id="source-chain")
    target = _timeline(timeline_id="target", transform_chain_id="target-chain", **target_overrides)
    offset_map = TimelineOffsetMap(
        offset_map_id="offset-map-1",
        source_timeline_id="source",
        target_timeline_id="target",
        source_transform_chain_id="source-chain",
        target_transform_chain_id="target-chain",
        source_time_basis="canonical_ms",
        target_time_basis="canonical_ms",
        segments=(OffsetMapSegment(0, 1_000, 100, 1_100),),
    )

    with pytest.raises(ValidationError, match=message):
        validate_timeline_merge(source, target, offset_map=offset_map)


def test_chunk_relative_sample_index_and_frame_index_convert_to_canonical_ms():
    assert _timeline(time_basis="chunk_relative_ms").to_canonical_ms(250, chunk_start_ms=1_000) == 1_250
    assert _timeline(time_basis="sample_index", sample_rate_hz=16_000).to_canonical_ms(8_000) == 500
    assert _timeline(time_basis="frame_index").to_canonical_ms(30, frame_rate_fps=30.0) == 1_000

    with pytest.raises(ValidationError, match="chunk_start_ms is required"):
        _timeline(time_basis="chunk_relative_ms").to_canonical_ms(250)


def test_offset_map_segments_are_validated_and_bound_checked():
    with pytest.raises(ValidationError, match="must not overlap"):
        TimelineOffsetMap(
            offset_map_id="offset-map-1",
            source_timeline_id="source",
            target_timeline_id="target",
            source_transform_chain_id="source-chain",
            target_transform_chain_id="target-chain",
            source_time_basis="canonical_ms",
            target_time_basis="canonical_ms",
            segments=(OffsetMapSegment(0, 500, 0, 500), OffsetMapSegment(400, 900, 400, 900)),
        )

    offset_map = TimelineOffsetMap(
        offset_map_id="offset-map-1",
        source_timeline_id="source",
        target_timeline_id="target",
        source_transform_chain_id="source-chain",
        target_transform_chain_id="target-chain",
        source_time_basis="canonical_ms",
        target_time_basis="canonical_ms",
        segments=(OffsetMapSegment(0, 1_000, 0, 1_000),),
    )

    with pytest.raises(ValidationError, match="not covered"):
        offset_map.convert_source_ms(1_500)


def test_boundary_shift_sentinel_marks_few_hundred_ms_degradation():
    assert boundary_shift_degrades_scoring(1_000, 1_300, tolerance_ms=250) is True
    assert boundary_shift_degrades_scoring(1_000, 1_200, tolerance_ms=250) is False


def test_transform_chain_captures_auditable_steps():
    chain = TransformChain(
        "chunked-normalization",
        steps=(
            TransformStep("step-1", "resample", {"from_hz": 48_000, "to_hz": 16_000}),
            TransformStep("step-2", "chunk", {"chunk_start_ms": 1_000}),
        ),
    )

    assert chain.to_dict()["transform_chain_id"] == "chunked-normalization"
    assert chain.to_dict()["steps"][1]["parameters"]["chunk_start_ms"] == 1_000
