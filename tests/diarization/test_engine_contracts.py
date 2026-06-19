import json
from pathlib import Path

import pytest

from keyframe.diarization import (
    AudioTimelineProvenance,
    CannedJsonEngineAdapter,
    DiarizationEngineAdapter,
    EngineConfigMetadata,
    HostedProviderGovernance,
    HostedProviderJsonAdapter,
    ModelArtifactGovernance,
    NormalizedArtifactProvenance,
    OffsetMapSegment,
    SelfHostedWhisperXPyannoteAdapter,
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


def _whisperx_artifact(duration_ms=1_300):
    return NormalizedArtifactProvenance(
        artifact_id="candidate-whisperx-output",
        artifact_kind="candidate",
        timeline=AudioTimelineProvenance(
            original_audio_id="original-audio-local",
            canonical_audio_id="canonical-audio-local",
            timeline_id="candidate-timeline",
            transform_chain_id="identity-mono-mix",
            sample_rate_hz=16_000,
            duration_ms=duration_ms,
            channel_ids=("mono-mix",),
        ),
    )


def _provider_artifact(channel_ids=("ch-1",), duration_ms=1_200):
    return NormalizedArtifactProvenance(
        artifact_id="candidate-hosted-provider-output",
        artifact_kind="candidate",
        timeline=AudioTimelineProvenance(
            original_audio_id="original-audio-local",
            canonical_audio_id="canonical-audio-local",
            timeline_id="candidate-timeline",
            transform_chain_id="identity",
            sample_rate_hz=16_000,
            duration_ms=duration_ms,
            channel_ids=channel_ids,
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


def _whisperx_adapter():
    return SelfHostedWhisperXPyannoteAdapter(
        ModelArtifactGovernance(
            checkpoint="pyannote/speaker-diarization-3.1",
            package_versions={"pyannote.audio": "3.1.0", "whisperx": "3.2.0"},
            runtime_config={
                "allow_download": False,
                "cache_root": "/models/local",
                "compute_type": "float16",
                "device": "cuda",
                "requires_gpu": True,
            },
            accepted_terms=("pyannote gated model terms accepted locally",),
            registry_source="https://huggingface.co/pyannote/speaker-diarization-3.1",
        ),
        config_id="local-whisperx-pyannote",
        model_version="2026-06",
    )


def _hosted_adapter(provider):
    return HostedProviderJsonAdapter(
        HostedProviderGovernance(
            provider=provider,
            region="us-east-1",
            model_version="2026-06",
            version_pinning="provider-version-pinned-in-run-record",
            retention_policy="raw-json-retained-in-private-benchmark-artifacts",
            raw_json_export=f"raw/provider_json/{provider}.json",
            terms_constraints=("no live provider call in default test suite",),
            parameters={"batch": provider != "google_speech", "live_api_enabled": False},
        ),
        config_id=f"{provider}-canned",
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


def test_self_hosted_whisperx_pyannote_adapter_satisfies_engine_protocol():
    assert isinstance(_whisperx_adapter(), DiarizationEngineAdapter)


def test_hosted_provider_adapter_satisfies_engine_protocol():
    assert isinstance(_hosted_adapter("aws_transcribe"), DiarizationEngineAdapter)


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


def test_chunk_relative_time_basis_rejects_canonical_offset_map_without_chunk_offset():
    payload = _payload("chunk_relative.json")
    payload["segments"][0].pop("chunk_start_ms")

    with pytest.raises(ValidationError, match="chunk_relative_ms requires"):
        _adapter().normalize_raw_output(
            payload,
            artifact=_artifact(),
            source_artifact=_vad_trimmed_source(),
            transform_offset_map=_plus_100_offset_map(),
        )


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


def test_whisperx_pyannote_saved_output_normalizes_to_canonical_candidate_artifact():
    output = _whisperx_adapter().normalize_raw_output(
        _payload("whisperx_pyannote_output.json"),
        artifact=_whisperx_artifact(),
    )

    assert output.output_id == "whisperx-pyannote-saved"
    assert [word.text for word in output.words] == ["Hello", "there", "friend"]
    assert [(word.start_ms, word.end_ms) for word in output.words] == [
        (20, 310),
        (360, 680),
        (820, 1150),
    ]
    assert [word.speaker_ref for word in output.words] == [
        "engine:mono-mix:speaker-00",
        "engine:mono-mix:speaker-00",
        "engine:mono-mix:speaker-01",
    ]
    assert [span.speaker_ref for span in output.speaker_spans] == [
        "engine:mono-mix:speaker-00",
        "engine:mono-mix:speaker-01",
    ]
    assert [(span.start_ms, span.end_ms) for span in output.speaker_spans] == [(0, 740), (740, 1220)]
    assert [item.raw_speaker_id for item in output.raw_speaker_evidence] == ["SPEAKER_00", "SPEAKER_01"]
    assert all(word.display_label is None for word in output.words)


def test_whisperx_governance_is_attached_to_engine_config():
    output = _whisperx_adapter().normalize_raw_output(
        _payload("whisperx_pyannote_output.json"),
        artifact=_whisperx_artifact(),
    )

    governance = output.to_dict()["config"]["parameters"]["model_governance"]
    assert governance["checkpoint"] == "pyannote/speaker-diarization-3.1"
    assert governance["package_versions"] == {"pyannote.audio": "3.1.0", "whisperx": "3.2.0"}
    assert governance["runtime_config"]["cache_root"] == "/models/local"
    assert governance["accepted_terms"] == ["pyannote gated model terms accepted locally"]
    assert governance["registry_source"] == "https://huggingface.co/pyannote/speaker-diarization-3.1"


def test_whisperx_word_segments_fall_back_to_pyannote_intervals_for_speakers():
    output = _whisperx_adapter().normalize_raw_output(
        _payload("whisperx_pyannote_word_segments.json"),
        artifact=_whisperx_artifact(duration_ms=1_000),
    )

    assert [word.text for word in output.words] == ["first", "second"]
    assert [word.speaker_ref for word in output.words] == [
        "engine:mono-mix:speaker-00",
        "engine:mono-mix:speaker-01",
    ]
    assert [item.source_field for item in output.raw_speaker_evidence] == ["speaker", "speaker"]


def test_whisperx_word_derived_spans_preserve_separate_speaker_turns():
    payload = {
        "channel_id": "mono-mix",
        "output_id": "word-speaker-turns",
        "segments": [
            {
                "words": [
                    {"end_ms": 100, "speaker": "A", "start_ms": 0, "word": "alpha"},
                    {"end_ms": 300, "speaker": "B", "start_ms": 200, "word": "bravo"},
                    {"end_ms": 500, "speaker": "A", "start_ms": 400, "word": "again"},
                ]
            }
        ],
    }

    output = _whisperx_adapter().normalize_raw_output(payload, artifact=_whisperx_artifact(duration_ms=600))

    assert [(span.speaker_ref, span.start_ms, span.end_ms) for span in output.speaker_spans] == [
        ("engine:mono-mix:a", 0, 100),
        ("engine:mono-mix:b", 200, 300),
        ("engine:mono-mix:a", 400, 500),
    ]


def test_whisperx_word_derived_spans_do_not_bridge_unknown_speaker_gaps():
    payload = {
        "channel_id": "mono-mix",
        "output_id": "word-speaker-gap",
        "segments": [
            {
                "words": [
                    {"end_ms": 100, "speaker": "A", "start_ms": 0, "word": "known"},
                    {"end_ms": 300, "start_ms": 200, "word": "unknown"},
                    {"end_ms": 500, "speaker": "A", "start_ms": 400, "word": "again"},
                ]
            }
        ],
    }

    output = _whisperx_adapter().normalize_raw_output(payload, artifact=_whisperx_artifact(duration_ms=600))

    assert [word.speaker_ref for word in output.words] == ["engine:mono-mix:a", None, "engine:mono-mix:a"]
    assert [(span.speaker_ref, span.start_ms, span.end_ms) for span in output.speaker_spans] == [
        ("engine:mono-mix:a", 0, 100),
        ("engine:mono-mix:a", 400, 500),
    ]


def test_whisperx_word_derived_spans_do_not_bridge_long_same_speaker_silences():
    payload = {
        "channel_id": "mono-mix",
        "output_id": "word-speaker-silence",
        "segments": [
            {
                "words": [
                    {"end_ms": 100, "speaker": "A", "start_ms": 0, "word": "early"},
                    {"end_ms": 5_100, "speaker": "A", "start_ms": 5_000, "word": "late"},
                ]
            }
        ],
    }

    output = _whisperx_adapter().normalize_raw_output(payload, artifact=_whisperx_artifact(duration_ms=5_200))

    assert [(span.speaker_ref, span.start_ms, span.end_ms) for span in output.speaker_spans] == [
        ("engine:mono-mix:a", 0, 100),
        ("engine:mono-mix:a", 5_000, 5_100),
    ]


def test_whisperx_runtime_preflight_reports_missing_optional_dependencies_without_import_failure():
    status = _whisperx_adapter().runtime_preflight(
        dependency_modules={"keyframe_missing_whisperx_for_test": "whisperx-test-only"}
    )

    assert status.status == "unsupported"
    assert status.available is False
    assert status.missing_packages == ("whisperx-test-only",)
    assert "Install optional runtime packages" in status.reasons[0]
    assert status.requires_model_access is True
    assert status.requires_gpu is True


def test_aws_transcribe_saved_output_normalizes_to_canonical_provider_artifact():
    output = _hosted_adapter("aws_transcribe").normalize_raw_output(
        _payload("aws_transcribe_provider.json"),
        artifact=_provider_artifact(),
    )

    assert output.output_id == "aws-transcribe-canned"
    assert [word.text for word in output.words] == ["hello", "there", "friend"]
    assert [(word.start_ms, word.end_ms) for word in output.words] == [(0, 300), (340, 640), (800, 1080)]
    assert [word.text_confidence for word in output.words] == [0.98, 0.96, 0.94]
    assert [word.speaker_ref for word in output.words] == [
        "engine:ch-1:spk-0",
        "engine:ch-1:spk-0",
        "engine:ch-1:spk-1",
    ]
    assert [(span.speaker_ref, span.start_ms, span.end_ms) for span in output.speaker_spans] == [
        ("engine:ch-1:spk-0", 0, 640),
        ("engine:ch-1:spk-1", 800, 1080),
    ]


def test_google_streaming_saved_output_uses_final_word_timestamps_only():
    payload = _payload("google_speech_provider.json")
    payload.pop("channel_id")

    output = _hosted_adapter("google_speech").normalize_raw_output(
        payload,
        artifact=_provider_artifact(),
    )

    assert output.output_id == "google-speech-canned"
    assert [word.text for word in output.words] == ["hello", "there"]
    assert [word.channel_id for word in output.words] == ["ch-1", "ch-1"]
    assert [word.word_id for word in output.words] == [
        "google-speech-canned:word:000001",
        "google-speech-canned:word:000002",
    ]
    assert [word.text_confidence for word in output.words] == [0.91, 0.91]
    assert [word.speaker_ref for word in output.words] == ["engine:ch-1:1", "engine:ch-1:2"]
    assert all(word.text != "par" for word in output.words)


def test_google_snake_case_streaming_partials_are_not_treated_as_final():
    payload = _payload("google_speech_provider.json")
    payload["results"][0].pop("isFinal")
    payload["results"][0]["is_final"] = False
    payload["results"][1].pop("isFinal")
    payload["results"][1]["is_final"] = True

    output = _hosted_adapter("google_speech").normalize_raw_output(payload, artifact=_provider_artifact())

    assert [word.text for word in output.words] == ["hello", "there"]
    assert all(word.text != "par" for word in output.words)


def test_google_cumulative_streaming_finals_replace_prior_word_metadata_without_duplicates():
    payload = _payload("google_speech_provider.json")
    payload.pop("channel_id")
    payload["results"].append(
        {
            "alternatives": [
                {
                    "confidence": 0.97,
                    "transcript": "hello there",
                    "words": [
                        {
                            "confidence": 0.96,
                            "endTime": "0.30s",
                            "speakerTag": 2,
                            "startTime": "0.00s",
                            "word": "hello",
                        },
                        {
                            "confidence": 0.95,
                            "endTime": "0.66s",
                            "speakerTag": 2,
                            "startTime": "0.35s",
                            "word": "there",
                        },
                    ],
                }
            ],
            "channelTag": 1,
            "isFinal": True,
        }
    )

    output = _hosted_adapter("google_speech").normalize_raw_output(payload, artifact=_provider_artifact())

    assert [word.text for word in output.words] == ["hello", "there"]
    assert [word.word_id for word in output.words] == [
        "google-speech-canned:word:000001",
        "google-speech-canned:word:000002",
    ]
    assert [word.text_confidence for word in output.words] == [0.96, 0.95]
    assert [word.speaker_ref for word in output.words] == ["engine:ch-1:2", "engine:ch-1:2"]


def test_google_channel_tag_overrides_root_channel_fallback_for_multichannel_outputs():
    payload = _payload("google_speech_provider.json")
    payload["channel_id"] = "left"
    for result in payload["results"]:
        result["channelTag"] = 2

    output = _hosted_adapter("google_speech").normalize_raw_output(
        payload,
        artifact=_provider_artifact(channel_ids=("left", "right")),
    )

    assert [word.channel_id for word in output.words] == ["right", "right"]
    assert [word.speaker_ref for word in output.words] == ["engine:right:1", "engine:right:2"]


def test_deepgram_saved_output_keeps_same_speaker_ids_channel_local():
    output = _hosted_adapter("deepgram").normalize_raw_output(
        _payload("deepgram_provider.json"),
        artifact=_provider_artifact(channel_ids=("left", "right")),
    )

    assert [word.text for word in output.words] == ["left", "answer"]
    assert [word.channel_id for word in output.words] == ["left", "right"]
    assert [word.speaker_ref for word in output.words] == ["engine:left:0", "engine:right:0"]
    assert [word.speaker_confidence for word in output.words] == [0.88, 0.82]
    assert len({word.speaker_ref for word in output.words}) == 2


def test_aws_channel_labels_map_to_canonical_artifact_channels():
    payload = _payload("aws_transcribe_provider.json")
    payload.pop("channel_id")
    for item in payload["results"]["items"]:
        item.pop("channel_id", None)
        if item.get("type") == "pronunciation":
            item["channel_label"] = "ch_0"

    output = _hosted_adapter("aws_transcribe").normalize_raw_output(payload, artifact=_provider_artifact())

    assert [word.channel_id for word in output.words] == ["ch-1", "ch-1", "ch-1"]
    assert [word.speaker_ref for word in output.words] == [
        "engine:ch-1:spk-0",
        "engine:ch-1:spk-0",
        "engine:ch-1:spk-1",
    ]


def test_aws_speaker_labels_sidecar_normalizes_speaker_metadata():
    payload = _payload("aws_transcribe_provider.json")
    for item in payload["results"]["items"]:
        item.pop("speaker_label", None)
    payload["results"]["speaker_labels"] = {
        "segments": [
            {
                "items": [
                    {"end_time": "0.30", "speaker_label": "spk_0", "start_time": "0.00"},
                    {"end_time": "0.64", "speaker_label": "spk_0", "start_time": "0.34"},
                    {"end_time": "1.08", "speaker_label": "spk_1", "start_time": "0.80"},
                ],
                "speaker_label": "spk_0",
            }
        ]
    }

    output = _hosted_adapter("aws_transcribe").normalize_raw_output(payload, artifact=_provider_artifact())

    assert [word.speaker_ref for word in output.words] == [
        "engine:ch-1:spk-0",
        "engine:ch-1:spk-0",
        "engine:ch-1:spk-1",
    ]


def test_hosted_provider_governance_is_attached_to_engine_config_without_live_call_state():
    output = _hosted_adapter("aws_transcribe").normalize_raw_output(
        _payload("aws_transcribe_provider.json"),
        artifact=_provider_artifact(),
    )

    governance = output.to_dict()["config"]["parameters"]["hosted_provider_governance"]
    assert governance["provider"] == "aws_transcribe"
    assert governance["region"] == "us-east-1"
    assert governance["model_version"] == "2026-06"
    assert governance["version_pinning"] == "provider-version-pinned-in-run-record"
    assert governance["retention_policy"] == "raw-json-retained-in-private-benchmark-artifacts"
    assert governance["raw_json_export"] == "raw/provider_json/aws_transcribe.json"
    assert governance["terms_constraints"] == ["no live provider call in default test suite"]
    assert governance["parameters"]["live_api_enabled"] is False


def test_hosted_provider_missing_word_timing_fails_closed():
    payload = _payload("aws_transcribe_provider.json")
    payload["results"]["items"][0].pop("start_time")

    with pytest.raises(ValidationError, match="start is required|start_time is required"):
        _hosted_adapter("aws_transcribe").normalize_raw_output(payload, artifact=_provider_artifact())


def test_hosted_provider_missing_speaker_metadata_fails_closed():
    payload = _payload("aws_transcribe_provider.json")
    for item in payload["results"]["items"]:
        item.pop("speaker_label", None)

    with pytest.raises(ValidationError, match="speaker metadata"):
        _hosted_adapter("aws_transcribe").normalize_raw_output(payload, artifact=_provider_artifact())
