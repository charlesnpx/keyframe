import math

import pytest

from keyframe.diarization import (
    AudioChannelMapping,
    AudioTimelineProvenance,
    AudioTransformConfig,
    AudioTransformManifest,
    NormalizedArtifactProvenance,
    OffsetMapSegment,
    TimelineOffsetMap,
    TransformChain,
    TransformStep,
    ValidationError,
    boundary_shift_degrades_scoring,
    build_audio_transform_manifest,
    build_mono_mix_transform_manifest,
    hash_audio_transform_config,
    normalize_transform_command,
    read_recording_json,
    sha256_bytes,
    sha256_file,
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


def _assert_no_audio_hashes(payload):
    if isinstance(payload, dict):
        for key, value in payload.items():
            assert "sha256" not in key
            assert key != "transform_config_hash"
            _assert_no_audio_hashes(value)
    elif isinstance(payload, list):
        for value in payload:
            _assert_no_audio_hashes(value)


def _transform_config(*, source_channel_ids=("ch-1",), output_channel_id="ch-1", command=None):
    return AudioTransformConfig(
        tool_name="ffmpeg",
        tool_version="6.1",
        normalized_command=command or ["ffmpeg", "-i", "input.wav", "-ar", "16000", "canonical.wav"],
        channel_mapping=(
            AudioChannelMapping(
                output_channel_id=output_channel_id,
                source_channel_ids=tuple(source_channel_ids),
            ),
        ),
        gain_policy="preserve",
        downmix_policy="none",
    )


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


def test_transform_command_normalization_and_config_hash_are_stable():
    spaced = _transform_config(command="ffmpeg   -i 'input file.wav'   -ar 16000 canonical.wav")
    tokenized = _transform_config(command=("ffmpeg", "-i", "input file.wav", "-ar", "16000", "canonical.wav"))
    remapped = _transform_config(source_channel_ids=("ch-2",), command=tokenized.normalized_command)

    assert normalize_transform_command(spaced.normalized_command) == tokenized.normalized_command
    assert hash_audio_transform_config(spaced) == hash_audio_transform_config(tokenized)
    assert hash_audio_transform_config(remapped) != hash_audio_transform_config(tokenized)


def test_audio_transform_manifest_records_integrity_only_transform_details():
    config = _transform_config()
    manifest = build_audio_transform_manifest(
        branch_id="separate_tracks",
        original_audio_id="original-audio-local",
        canonical_audio_id="canonical-audio-local",
        original_audio_sha256=sha256_bytes(b"original"),
        canonical_audio_sha256=sha256_bytes(b"canonical"),
        config=config,
    )
    timeline = _timeline(
        local_audio_sha256=manifest.canonical_audio_sha256,
        transform_manifest=manifest,
    )
    artifact = _artifact("fixture", timeline)

    integrity = artifact.to_integrity_dict()
    assert integrity["timeline"]["transform_manifest"]["transform_config_hash"] == config.config_hash
    assert integrity["timeline"]["transform_manifest"]["config"]["tool_name"] == "ffmpeg"
    assert integrity["timeline"]["transform_manifest"]["config"]["channel_mapping"][0]["source_channel_ids"] == ["ch-1"]

    rendered = artifact.to_rendered_transcript_metadata()
    monitoring = artifact.to_monitoring_metadata()
    assert "transform_manifest" not in rendered["timeline"]
    _assert_no_audio_hashes(rendered)
    _assert_no_audio_hashes(monitoring)


def test_audio_transform_manifest_rejects_forged_transform_id():
    config = _transform_config()

    with pytest.raises(ValidationError, match="transform_id does not match content address"):
        AudioTransformManifest(
            transform_id="audio-transform:separate_tracks:forged",
            branch_id="separate_tracks",
            original_audio_id="original-audio-local",
            canonical_audio_id="canonical-audio-local",
            original_audio_sha256=sha256_bytes(b"original"),
            canonical_audio_sha256=sha256_bytes(b"canonical"),
            transform_config_hash=config.config_hash,
            config=config,
        )


def test_sha256_file_hashes_file_content(tmp_path):
    path = tmp_path / "audio.fake"
    path.write_bytes(b"canonical audio bytes")

    assert sha256_file(path) == sha256_bytes(b"canonical audio bytes")


def test_mono_mix_transform_manifest_is_reproducible_and_branch_specific():
    common = {
        "original_audio_id": "original-audio-local",
        "canonical_audio_id": "canonical-audio-local",
        "original_audio_sha256": sha256_bytes(b"original"),
        "canonical_audio_sha256": sha256_bytes(b"canonical-mono"),
        "source_channel_ids": ("ihm-P1", "ihm-P2"),
        "tool_name": "ffmpeg",
        "tool_version": "6.1",
        "command": ("ffmpeg", "-i", "input.wav", "-ac", "1", "mono.wav"),
    }

    first = build_mono_mix_transform_manifest(branch_id="mono_mix", **common)
    second = build_mono_mix_transform_manifest(branch_id="mono_mix", **common)
    alternate_branch = build_mono_mix_transform_manifest(branch_id="diagnostic_mono_mix", **common)

    assert first.transform_id == second.transform_id
    assert first.transform_id != alternate_branch.transform_id
    assert first.transform_config_hash == alternate_branch.transform_config_hash
    assert first.config.downmix_policy == "mono_mix"
    assert first.config.channel_mapping[0].output_channel_id == "mono-mix"
    assert first.config.channel_mapping[0].source_channel_ids == ("ihm-P1", "ihm-P2")


def test_transform_manifest_must_match_timeline_audio_identity_and_hash():
    manifest = build_audio_transform_manifest(
        branch_id="separate_tracks",
        original_audio_id="original-audio-local",
        canonical_audio_id="canonical-audio-local",
        original_audio_sha256=sha256_bytes(b"original"),
        canonical_audio_sha256=sha256_bytes(b"canonical"),
        config=_transform_config(),
    )

    with pytest.raises(ValidationError, match="canonical_audio_sha256 conflicts"):
        _timeline(local_audio_sha256=sha256_bytes(b"other"), transform_manifest=manifest)


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
        segments=(OffsetMapSegment(0, 2_000, 0, 2_000),),
    )

    result = validate_timeline_merge(source, target, offset_map=offset_map)

    assert result.direct_timeline_match is False
    assert result.offset_map_id == "offset-map-1"
    assert offset_map.convert_source_ms(250) == 250


def test_offset_map_allows_chunk_relative_timeline_duration_to_differ_from_canonical_timeline():
    source = _artifact(
        "asr",
        _timeline(
            timeline_id="chunk-300-2000",
            transform_chain_id="chunk-chain",
            duration_ms=1_700,
            time_basis="chunk_relative_ms",
        ),
    )
    target = _artifact(
        "diarization",
        _timeline(timeline_id="canonical", transform_chain_id="canonical-chain", duration_ms=2_000),
    )
    offset_map = TimelineOffsetMap(
        offset_map_id="chunk-to-canonical",
        source_timeline_id="chunk-300-2000",
        target_timeline_id="canonical",
        source_transform_chain_id="chunk-chain",
        target_transform_chain_id="canonical-chain",
        source_time_basis="chunk_relative_ms",
        target_time_basis="canonical_ms",
        segments=(OffsetMapSegment(0, 1_700, 300, 2_000),),
    )

    result = validate_timeline_merge(source, target, offset_map=offset_map)

    assert result.offset_map_id == "chunk-to-canonical"
    assert offset_map.convert_source_ms(0) == 300
    assert offset_map.convert_source_ms(1_699) == 1_999
    assert offset_map.convert_source_ms(1_700) == 2_000


@pytest.mark.parametrize(
    "target_overrides, message",
    [
        ({"sample_rate_hz": 48_000}, "sample_rate_hz conflicts"),
        ({"channel_ids": ("ch-2",)}, "channel layout conflicts"),
    ],
)
def test_conflicting_non_duration_audio_metadata_rejects_merge_even_with_offset_map(target_overrides, message):
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


def test_duration_mismatch_without_offset_map_rejects_merge():
    source = _timeline(timeline_id="source", transform_chain_id="source-chain", duration_ms=1_700)
    target = _timeline(timeline_id="target", transform_chain_id="target-chain", duration_ms=2_000)

    with pytest.raises(ValidationError, match="duration_ms conflicts"):
        validate_timeline_merge(source, target)


def test_chunk_relative_sample_index_and_frame_index_convert_to_canonical_ms():
    assert _timeline(time_basis="chunk_relative_ms").to_canonical_ms(250, chunk_start_ms=1_000) == 1_250
    assert _timeline(time_basis="sample_index", sample_rate_hz=16_000).to_canonical_ms(8_000) == 500
    assert _timeline(time_basis="frame_index").to_canonical_ms(30, frame_rate_fps=30.0) == 1_000

    with pytest.raises(ValidationError, match="chunk_start_ms is required"):
        _timeline(time_basis="chunk_relative_ms").to_canonical_ms(250)

    with pytest.raises(ValidationError, match="frame_rate_fps must be a number"):
        _timeline(time_basis="frame_index").to_canonical_ms(30, frame_rate_fps="30")

    with pytest.raises(ValidationError, match="frame_rate_fps must be greater than 0"):
        _timeline(time_basis="frame_index").to_canonical_ms(30, frame_rate_fps=0.0)


def test_offset_map_segments_are_validated_and_bound_checked():
    with pytest.raises(ValidationError, match="source segments must not overlap"):
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

    with pytest.raises(ValidationError, match="target segments must not overlap"):
        TimelineOffsetMap(
            offset_map_id="offset-map-1",
            source_timeline_id="source",
            target_timeline_id="target",
            source_transform_chain_id="source-chain",
            target_transform_chain_id="target-chain",
            source_time_basis="canonical_ms",
            target_time_basis="canonical_ms",
            segments=(OffsetMapSegment(0, 500, 0, 500), OffsetMapSegment(500, 900, 250, 650)),
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


@pytest.mark.parametrize(
    "basis_field, basis_value, message",
    [
        ("source_time_basis", "sample_index", "source_time_basis must be millisecond-based"),
        ("target_time_basis", "frame_index", "target_time_basis must be millisecond-based"),
    ],
)
def test_offset_map_rejects_non_millisecond_time_bases(basis_field, basis_value, message):
    kwargs = {
        "offset_map_id": "offset-map-1",
        "source_timeline_id": "source",
        "target_timeline_id": "target",
        "source_transform_chain_id": "source-chain",
        "target_transform_chain_id": "target-chain",
        "source_time_basis": "canonical_ms",
        "target_time_basis": "canonical_ms",
        "segments": (OffsetMapSegment(0, 1_000, 0, 1_000),),
    }
    kwargs[basis_field] = basis_value

    with pytest.raises(ValidationError, match=message):
        TimelineOffsetMap(**kwargs)


@pytest.mark.parametrize(
    "segments",
    [
        (OffsetMapSegment(0, 1_000, 0, 1_000),),
        (OffsetMapSegment(0, 800, 0, 800), OffsetMapSegment(900, 1_700, 900, 1_700)),
    ],
)
def test_validated_offset_map_must_cover_full_source_timeline(segments):
    source = _timeline(timeline_id="source", transform_chain_id="source-chain", duration_ms=1_700)
    target = _timeline(timeline_id="target", transform_chain_id="target-chain", duration_ms=2_000)
    offset_map = TimelineOffsetMap(
        offset_map_id="offset-map-1",
        source_timeline_id="source",
        target_timeline_id="target",
        source_transform_chain_id="source-chain",
        target_transform_chain_id="target-chain",
        source_time_basis="canonical_ms",
        target_time_basis="canonical_ms",
        segments=segments,
    )

    with pytest.raises(ValidationError, match="must cover the full source timeline"):
        validate_timeline_merge(source, target, offset_map=offset_map)


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


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_transform_parameters_reject_non_finite_json_numbers(bad_value):
    with pytest.raises(ValidationError, match="must be a finite JSON number"):
        TransformStep("step-1", "resample", {"from_hz": bad_value})
