import json
from pathlib import Path

import pytest

from keyframe.diarization import (
    AudioTimelineProvenance,
    CannedJsonEngineAdapter,
    DiarizationEngineAdapter,
    EngineConfigMetadata,
    NormalizedArtifactProvenance,
    OffsetMapSegment,
    TimelineOffsetMap,
    ValidationError,
)


FIXTURE_DIR = Path("tests/diarization/fixtures/engine_outputs")


def _payload(name):
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _artifact():
    return NormalizedArtifactProvenance(
        artifact_id="candidate-engine-output",
        artifact_kind="candidate",
        timeline=AudioTimelineProvenance(
            original_audio_id="original-audio-local",
            canonical_audio_id="canonical-audio-local",
            timeline_id="candidate-timeline",
            transform_chain_id="identity",
            sample_rate_hz=16_000,
            duration_ms=1_200,
            channel_ids=("ch-1",),
        ),
    )


def _artifact_with_timeline(**overrides):
    timeline = {
        "original_audio_id": "original-audio-local",
        "canonical_audio_id": "canonical-audio-local",
        "timeline_id": "candidate-timeline",
        "transform_chain_id": "identity",
        "sample_rate_hz": 16_000,
        "duration_ms": 1_200,
        "channel_ids": ("ch-1",),
    }
    timeline.update(overrides)
    return NormalizedArtifactProvenance(
        artifact_id="candidate-engine-output",
        artifact_kind="candidate",
        timeline=AudioTimelineProvenance(**timeline),
    )


def _adapter():
    return CannedJsonEngineAdapter(
        EngineConfigMetadata(
            adapter_id="canned-json",
            provider="fixture-provider",
            model_name="fixture-diarizer",
            model_version="2026-06",
            config_id="fixture-config",
            parameters={"known_speaker_count": 2, "temperature": None},
        )
    )


def _vad_trimmed_source(duration_ms=900):
    return _artifact_with_timeline(
        timeline_id="vad-trimmed",
        transform_chain_id="vad-trim",
        duration_ms=duration_ms,
    )


def _plus_100_offset_map(source_end_ms=900, target_end_ms=1000):
    return TimelineOffsetMap(
        offset_map_id="vad-trim-to-canonical",
        source_timeline_id="vad-trimmed",
        target_timeline_id="candidate-timeline",
        source_transform_chain_id="vad-trim",
        target_transform_chain_id="identity",
        source_time_basis="canonical_ms",
        target_time_basis="canonical_ms",
        segments=(OffsetMapSegment(0, source_end_ms, 100, target_end_ms),),
    )


def test_canned_json_adapter_satisfies_engine_protocol():
    assert isinstance(_adapter(), DiarizationEngineAdapter)


def test_clean_engine_output_normalizes_to_canonical_words_spans_and_provenance():
    output = _adapter().normalize_raw_output(_payload("clean_provider.json"), artifact=_artifact())

    assert output.output_id == "clean-provider"
    assert [word.word_id for word in output.words] == [
        "clean-provider:word:000001",
        "clean-provider:word:000002",
        "clean-provider:word:000003",
    ]
    assert [word.text for word in output.words] == ["hello", "there", "friend"]
    assert [word.speaker_ref for word in output.words] == [
        "engine:ch-1:speaker-a",
        "engine:ch-1:speaker-a",
        "engine:ch-1:speaker-b",
    ]
    assert [span.speaker_ref for span in output.speaker_spans] == [
        "engine:ch-1:speaker-a",
        "engine:ch-1:speaker-b",
    ]
    assert output.config.provider == "fixture-provider"
    assert output.artifact.artifact_id == "candidate-engine-output"
    assert output.to_dict()["config"]["model_name"] == "fixture-diarizer"


def test_word_ids_are_stable_for_same_raw_output():
    adapter = _adapter()

    first = adapter.normalize_raw_output(_payload("clean_provider.json"), artifact=_artifact())
    second = adapter.normalize_raw_output(_payload("clean_provider.json"), artifact=_artifact())

    assert [word.word_id for word in first.words] == [word.word_id for word in second.words]
    assert [span.span_id for span in first.speaker_spans] == [span.span_id for span in second.speaker_spans]


def test_same_raw_speaker_id_on_different_channels_remains_channel_local():
    artifact = NormalizedArtifactProvenance(
        artifact_id="candidate-engine-output",
        artifact_kind="candidate",
        timeline=AudioTimelineProvenance(
            original_audio_id="original-audio-local",
            canonical_audio_id="canonical-audio-local",
            timeline_id="candidate-timeline",
            transform_chain_id="identity",
            sample_rate_hz=16_000,
            duration_ms=1_200,
            channel_ids=("left", "right"),
        ),
    )

    output = _adapter().normalize_raw_output(
        _payload("same_speaker_id_different_channels.json"),
        artifact=artifact,
    )

    assert [word.speaker_ref for word in output.words] == ["engine:left:s1", "engine:right:s1"]
    assert len({word.speaker_ref for word in output.words}) == 2
    assert [item.raw_speaker_id for item in output.raw_speaker_evidence] == ["S1", "S1"]
    assert [item.channel_id for item in output.raw_speaker_evidence] == ["left", "right"]
    assert all(word.display_label is None for word in output.words)


def test_distinct_raw_speaker_ids_do_not_collapse_after_sanitizing():
    output = _adapter().normalize_raw_output(_payload("speaker_id_collision.json"), artifact=_artifact())

    assert [word.speaker_ref for word in output.words] == ["engine:ch-1:s-1", "engine:ch-1:s-1-2"]
    assert [item.raw_speaker_id for item in output.raw_speaker_evidence] == ["S 1", "S-1"]
    assert [item.speaker_ref for item in output.raw_speaker_evidence] == [
        "engine:ch-1:s-1",
        "engine:ch-1:s-1-2",
    ]


def test_missing_confidence_is_preserved_as_null():
    output = _adapter().normalize_raw_output(_payload("missing_confidence.json"), artifact=_artifact())

    assert output.words[0].text_confidence is None
    assert output.words[0].speaker_confidence is None
    assert output.speaker_spans[0].confidence is None


def test_word_without_speaker_preserves_null_speaker_ref_without_display_label():
    output = _adapter().normalize_raw_output(_payload("word_without_speaker.json"), artifact=_artifact())

    assert output.words[0].speaker_ref == "engine:ch-1:speaker-a"
    assert output.words[1].speaker_ref is None
    assert output.words[1].display_label is None
    assert output.speaker_spans[0].speaker_ref == "engine:ch-1:speaker-a"


def test_paragraph_only_output_fails_closed():
    with pytest.raises(ValidationError, match="words is required"):
        _adapter().normalize_raw_output(_payload("paragraph_only.json"), artifact=_artifact())


def test_model_config_metadata_is_attached_to_normalized_output():
    output = _adapter().normalize_raw_output(_payload("clean_provider.json"), artifact=_artifact())
    payload = output.to_dict()

    assert payload["config"] == {
        "adapter_id": "canned-json",
        "config_id": "fixture-config",
        "model_name": "fixture-diarizer",
        "model_version": "2026-06",
        "parameters": {"known_speaker_count": 2, "temperature": None},
        "provider": "fixture-provider",
    }
    assert payload["artifact"]["artifact_kind"] == "candidate"


def test_chunk_relative_time_basis_normalizes_to_canonical_ms_with_chunk_offset():
    output = _adapter().normalize_raw_output(_payload("chunk_relative.json"), artifact=_artifact())

    assert [(word.start_ms, word.end_ms) for word in output.words] == [(1100, 1300)]
    assert [(span.start_ms, span.end_ms) for span in output.speaker_spans] == [(1100, 1300)]


def test_chunk_relative_time_basis_requires_validated_offset():
    payload = _payload("chunk_relative.json")
    payload["segments"][0].pop("chunk_start_ms")

    with pytest.raises(ValidationError, match="chunk_relative_ms requires"):
        _adapter().normalize_raw_output(payload, artifact=_artifact())


def test_transform_offset_map_requires_source_artifact():
    with pytest.raises(ValidationError, match="transform_offset_map requires source_artifact"):
        _adapter().normalize_raw_output(
            _payload("clean_provider.json"),
            artifact=_artifact(),
            transform_offset_map=_plus_100_offset_map(),
        )


def test_direct_timeline_match_does_not_apply_unused_transform_offset_map():
    output = _adapter().normalize_raw_output(
        _payload("clean_provider.json"),
        artifact=_artifact(),
        source_artifact=_artifact(),
        transform_offset_map=_plus_100_offset_map(),
    )

    assert [(word.start_ms, word.end_ms) for word in output.words] == [
        (0, 300),
        (350, 650),
        (800, 1050),
    ]


def test_chunk_relative_time_basis_applies_transform_offset_map_after_chunk_offset():
    payload = {
        "output_id": "chunk-relative-offset-map",
        "segments": [
            {
                "channel_id": "ch-1",
                "chunk_start_ms": 350,
                "speaker_id": "speaker-a",
                "time_basis": "chunk_relative_ms",
                "words": [
                    {
                        "end_ms": 200,
                        "start_ms": 100,
                        "text": "shifted",
                        "text_confidence": 0.98,
                    }
                ],
            }
        ],
    }

    output = _adapter().normalize_raw_output(
        payload,
        artifact=_artifact(),
        source_artifact=_vad_trimmed_source(),
        transform_offset_map=_plus_100_offset_map(),
    )

    assert [(word.start_ms, word.end_ms) for word in output.words] == [(550, 650)]
    assert [(span.start_ms, span.end_ms) for span in output.speaker_spans] == [(550, 650)]


def test_sample_index_time_basis_applies_transform_offset_map_after_conversion():
    payload = {
        "output_id": "sample-index-offset-map",
        "segments": [
            {
                "channel_id": "ch-1",
                "speaker_id": "speaker-a",
                "time_basis": "sample_index",
                "words": [
                    {
                        "end": 3200,
                        "start": 1600,
                        "text": "samples",
                        "text_confidence": 0.98,
                    }
                ],
            }
        ],
    }

    output = _adapter().normalize_raw_output(
        payload,
        artifact=_artifact(),
        source_artifact=_vad_trimmed_source(),
        transform_offset_map=_plus_100_offset_map(),
    )

    assert [(word.start_ms, word.end_ms) for word in output.words] == [(200, 300)]


def test_resampled_frame_index_time_basis_converts_to_canonical_ms():
    output = _adapter().normalize_raw_output(_payload("frame_index.json"), artifact=_artifact())

    assert [(word.start_ms, word.end_ms) for word in output.words] == [(1000, 2000)]


def test_frame_index_time_basis_applies_transform_offset_map_after_conversion():
    output = _adapter().normalize_raw_output(
        _payload("frame_index.json"),
        artifact=_artifact_with_timeline(duration_ms=2200),
        source_artifact=_vad_trimmed_source(duration_ms=2000),
        transform_offset_map=_plus_100_offset_map(source_end_ms=2000, target_end_ms=2100),
    )

    assert [(word.start_ms, word.end_ms) for word in output.words] == [(1100, 2100)]


def test_streaming_partial_final_outputs_use_stable_events_without_duplicate_words():
    output = _adapter().normalize_raw_output(_payload("streaming_partial_final.json"), artifact=_artifact())

    assert [word.word_id for word in output.words] == [
        "streaming-output:word:000001",
        "streaming-output:word:000002",
    ]
    assert [word.text for word in output.words] == ["hello", "there"]
    assert all(word.text != "hel" for word in output.words)


def test_duplicate_chunks_do_not_duplicate_words():
    output = _adapter().normalize_raw_output(_payload("duplicate_chunks.json"), artifact=_artifact())

    assert [word.text for word in output.words] == ["repeat"]
    assert [word.word_id for word in output.words] == ["duplicate-chunks:word:000001"]


def test_timeline_mismatch_rejects_without_valid_offset_map():
    source = _vad_trimmed_source()
    target = _artifact()

    with pytest.raises(ValidationError, match="duration_ms conflicts"):
        _adapter().normalize_raw_output(
            _payload("clean_provider.json"),
            artifact=target,
            source_artifact=source,
        )


def test_timeline_mismatch_accepts_valid_transform_offset_map():
    source = _vad_trimmed_source()
    target = _artifact()

    output = _adapter().normalize_raw_output(
        _payload("vad_trimmed_source.json"),
        artifact=target,
        source_artifact=source,
        transform_offset_map=_plus_100_offset_map(),
    )

    assert output.words[0].text == "hello"
    assert [(word.start_ms, word.end_ms) for word in output.words] == [(100, 400), (450, 750)]
    assert [(span.start_ms, span.end_ms) for span in output.speaker_spans] == [(100, 750)]
