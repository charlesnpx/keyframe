import json
from pathlib import Path

from keyframe.diarization import (
    AMIAdapter,
    AMIRecordingSource,
    AMISpeakerSegment,
    AMIWordAnnotation,
    DatasetCacheConfig,
    build_ami_reference_bundle,
    build_artifact_layout,
    build_candidate_bundle,
    normalize_ami_recording,
    read_ami_channels_xml,
    read_ami_segments_xml,
    read_ami_words_xml,
    render_transcript,
    validate_candidate_bundle_payload,
)


FIXTURE_ROOT = Path("tests/diarization/fixtures/ami_minimal")
SOURCE_IDS = ("AMI-P1", "AMI-P2")


def _recording_source(recording_id="ES2002a"):
    recording_root = FIXTURE_ROOT / recording_id
    return AMIRecordingSource(
        recording_id=recording_id,
        word_paths=tuple(sorted((recording_root / "words").glob("*.xml"))),
        segment_paths=tuple(sorted((recording_root / "segments").glob("*.xml"))),
        channels_path=recording_root / "channels.xml",
    )


def _payload_text(payload):
    return json.dumps(payload, sort_keys=True)


def test_ami_nxt_words_segments_and_channels_parse_from_minimal_fixture():
    channels = read_ami_channels_xml(FIXTURE_ROOT / "ES2002a" / "channels.xml")
    words = read_ami_words_xml(FIXTURE_ROOT / "ES2002a" / "words" / "ES2002a.AMI-P1.words.xml")
    segments = read_ami_segments_xml(FIXTURE_ROOT / "ES2002a" / "segments" / "ES2002a.AMI-P1.segments.xml")

    assert [channel.channel_id for channel in channels] == ["ihm-P1", "ihm-P2"]
    assert channels[0].name == "Headset microphone 1"
    assert channels[0].source_speaker_id == "AMI-P1"
    assert words == (
        AMIWordAnnotation(
            source_word_id="ES2002a.P1.words0",
            source_speaker_id="AMI-P1",
            text="good",
            start_ms=100,
            end_ms=420,
        ),
        AMIWordAnnotation(
            source_word_id="ES2002a.P1.words1",
            source_speaker_id="AMI-P1",
            text="morning",
            start_ms=560,
            end_ms=840,
        ),
    )
    assert segments == (
        AMISpeakerSegment(
            source_segment_id="ES2002a.P1.seg0",
            source_speaker_id="AMI-P1",
            start_ms=0,
            end_ms=900,
        ),
    )


def test_ami_normalization_maps_source_speakers_to_session_local_canonical_reference():
    recording = normalize_ami_recording(_recording_source())

    assert recording.recording_id == "ES2002a"
    assert recording.duration_ms == 1300
    assert [channel.to_dict() for channel in recording.channels] == [
        {"channel_id": "ihm-P1", "name": "Headset microphone 1"},
        {"channel_id": "ihm-P2", "name": "Headset microphone 2"},
    ]
    assert [speaker.speaker_ref for speaker in recording.speakers] == ["spk-1", "spk-2"]
    assert [speaker.display_label.source_ref for speaker in recording.speakers] == list(SOURCE_IDS)
    assert [word.text for word in recording.words] == ["good", "morning", "yes", "hello"]
    assert [word.speaker_ref for word in recording.words] == ["spk-1", "spk-1", "spk-2", "spk-2"]
    assert [word.channel_id for word in recording.words] == ["ihm-P1", "ihm-P1", "ihm-P2", "ihm-P2"]
    assert recording.words[1].overlap is True
    assert recording.words[2].overlap is True
    assert [span.overlap for span in recording.speaker_spans] == [True, True]
    assert [region.to_dict() for region in recording.scoring_regions] == [
        {"channel_id": "ihm-P1", "end_ms": 900, "region_id": "uem-1", "start_ms": 0},
        {"channel_id": "ihm-P2", "end_ms": 1300, "region_id": "uem-2", "start_ms": 650},
    ]


def test_ami_adapter_validates_local_cache_and_exports_reference_and_candidate_artifacts(tmp_path):
    adapter = AMIAdapter(source_root=FIXTURE_ROOT)
    cache = DatasetCacheConfig(cache_root=str(FIXTURE_ROOT))

    validation = adapter.validate_source("ami-public-dev", cache)
    recordings = adapter.normalize("ami-public-dev", cache)
    results = adapter.export_reference("ami-public-dev", recordings, build_artifact_layout(tmp_path / "artifacts"))

    assert validation.valid is True
    assert isinstance(adapter, AMIAdapter)
    assert [recording.recording_id for recording in recordings] == ["ES2002a", "ES2002b"]
    assert len(results) == 2
    assert results[0].reference_bundle.evaluator_speaker_map == {"spk-1": "AMI-P1", "spk-2": "AMI-P2"}
    for result in results:
        assert result.dataset_id == "ami"
        assert result.split_id == "ami-public-dev"
        for path in result.artifact_paths.values():
            assert Path(path).is_file()

    product_payload = json.loads(Path(results[0].artifact_paths["candidate_bundle"]).read_text(encoding="utf-8"))
    authenticated_payload = json.loads(
        Path(results[0].artifact_paths["authenticated_track_metadata_candidate_bundle"]).read_text(encoding="utf-8")
    )
    validate_candidate_bundle_payload(product_payload)
    validate_candidate_bundle_payload(authenticated_payload)
    assert product_payload["channels"] == [{"channel_id": "ihm-P1"}, {"channel_id": "ihm-P2"}]
    assert authenticated_payload["channels"] == [
        {"channel_id": "ihm-P1", "track_name": "Headset microphone 1"},
        {"channel_id": "ihm-P2", "track_name": "Headset microphone 2"},
    ]


def test_ami_source_speaker_ids_are_reference_only_not_candidate_or_rendered_product_output():
    recording = normalize_ami_recording(_recording_source())
    reference = build_ami_reference_bundle(recording)

    evaluator_payload = reference.to_evaluator_dict()
    product_payload = build_candidate_bundle(
        reference,
        bundle_id="ami-product",
        mode="product_realistic",
    ).to_dict()
    authenticated_payload = build_candidate_bundle(
        reference,
        bundle_id="ami-authenticated",
        mode="authenticated_track_metadata",
    ).to_dict()
    rendered_payload = render_transcript(recording).to_dict()

    assert evaluator_payload["evaluator_speaker_map"] == {"spk-1": "AMI-P1", "spk-2": "AMI-P2"}
    for source_id in SOURCE_IDS:
        assert source_id in _payload_text(evaluator_payload)
        assert source_id not in _payload_text(product_payload)
        assert source_id not in _payload_text(authenticated_payload)
        assert source_id not in _payload_text(rendered_payload)
    assert "track_name" not in product_payload["channels"][0]
    assert authenticated_payload["channels"][0]["track_name"] == "Headset microphone 1"
    assert rendered_payload["turns"][0]["label"] == "person_1"
