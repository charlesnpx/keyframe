import json
from dataclasses import replace
from pathlib import Path

from keyframe.diarization import (
    EngineConfigMetadata,
    NormalizedArtifactProvenance,
    NormalizedEngineOutput,
    ReferenceBundle,
    SpeakerRecord,
    SpeakerSpan,
    build_candidate_bundle,
    build_evaluation_slices,
    evaluate_diarization_candidate,
    read_recording_json,
    render_transcript,
)


FIXTURE_DIR = Path("tests/diarization/fixtures")


def _recording(name="clean_two_speaker.json"):
    return read_recording_json(FIXTURE_DIR / name)


def _candidate_output(recording, speaker_map):
    spans = tuple(
        SpeakerSpan(
            span_id=f"candidate-{span.span_id}",
            speaker_ref=speaker_map[span.speaker_ref],
            start_ms=span.start_ms,
            end_ms=span.end_ms,
            channel_id=span.channel_id,
            confidence=span.confidence,
            overlap=span.overlap,
        )
        for span in recording.speaker_spans
    )
    words = tuple(
        replace(
            word,
            speaker_ref=speaker_map[word.speaker_ref] if word.speaker_ref is not None else None,
            display_label=None,
        )
        for word in recording.words
    )
    return NormalizedEngineOutput(
        output_id="anonymous-candidate",
        output_kind="word_spans",
        artifact=NormalizedArtifactProvenance.from_recording(
            recording,
            artifact_id="anonymous-candidate-artifact",
            artifact_kind="candidate",
        ),
        config=EngineConfigMetadata(
            adapter_id="test-anonymous-diarizer",
            provider="test-provider",
            model_name="anonymous-speaker-clusterer",
            model_version="fixture",
        ),
        words=words,
        speaker_spans=spans,
    )


def _reference_bundle(recording):
    return ReferenceBundle.from_recording(recording, artifact_id="reference-fixture")


def _candidate_recording(recording, candidate):
    speaker_refs = tuple(sorted({span.speaker_ref for span in candidate.speaker_spans}))
    return replace(
        recording,
        speakers=tuple(SpeakerRecord(speaker_ref) for speaker_ref in speaker_refs),
        words=candidate.words,
        speaker_spans=candidate.speaker_spans,
    )


def test_perfect_anonymous_diarizer_scores_with_permutation_invariant_labels():
    recording = _recording()
    reference = _reference_bundle(recording)
    candidate = _candidate_output(
        recording,
        {
            "spk-a": "engine:local:speaker-2",
            "spk-b": "engine:local:speaker-1",
        },
    )

    result = evaluate_diarization_candidate(reference, candidate)

    assert result.speaker_mapping == {
        "engine:local:speaker-1": "spk-b",
        "engine:local:speaker-2": "spk-a",
    }
    assert result.recording_metrics[0].metrics["speaker_label_accuracy"] == 1.0
    assert result.recording_metrics[0].metrics["speaker_label_error_rate"] == 0.0
    assert result.recording_metrics[0].metrics["diarization_error_rate"] == 0.0
    assert result.recording_metrics[0].metrics["matched_speaker_ms"] == 750


def test_reference_derived_slices_include_required_dimensions_and_sparse_statuses():
    slices = build_evaluation_slices(_recording(), minimum_support_ms=1)
    by_id = {item.slice_id: item for item in slices}

    assert {
        "channel_mode:mono",
        "channel_mode:multichannel",
        "overlap:non_overlap",
        "overlap:overlap",
        "speaker_change_boundary:within_collar",
        "speaker_count:0",
        "speaker_count:1",
        "speaker_count:2",
        "speaker_count:3_plus",
        "turn_duration:long",
        "turn_duration:short",
    }.issubset(by_id)
    assert by_id["overlap:non_overlap"].status == "ready"
    assert by_id["overlap:overlap"].status == "insufficient_support"
    assert by_id["speaker_count:1"].status == "ready"
    assert by_id["speaker_count:2"].status == "insufficient_support"
    assert by_id["turn_duration:short"].status == "ready"
    assert by_id["turn_duration:long"].status == "insufficient_support"
    assert by_id["channel_mode:mono"].status == "ready"
    assert by_id["channel_mode:multichannel"].status == "insufficient_support"


def test_overlap_reference_scores_overlap_slice_when_reference_supports_it():
    recording = _recording("overlap.json")
    reference = _reference_bundle(recording)
    candidate = _candidate_output(
        recording,
        {
            "spk-a": "engine:anonymous:a",
            "spk-b": "engine:anonymous:b",
        },
    )

    result = evaluate_diarization_candidate(reference, candidate)
    overlap_row = {row.slice_id: row for row in result.slice_metrics}["overlap:overlap"]

    assert overlap_row.status == "scored"
    assert overlap_row.support_ms == 500
    assert overlap_row.metrics["speaker_label_accuracy"] == 1.0


def test_speaker_mapping_stays_in_score_artifact_not_candidate_or_rendered_transcript():
    recording = _recording()
    reference = _reference_bundle(recording)
    candidate = _candidate_output(
        recording,
        {
            "spk-a": "engine:local:speaker-2",
            "spk-b": "engine:local:speaker-1",
        },
    )
    candidate_bundle = build_candidate_bundle(reference, bundle_id="candidate-fixture")
    candidate_payload_before = candidate.to_dict()

    result = evaluate_diarization_candidate(reference, candidate)

    assert result.to_dict()["speaker_mapping"] == {
        "engine:local:speaker-1": "spk-b",
        "engine:local:speaker-2": "spk-a",
    }
    assert candidate.to_dict() == candidate_payload_before
    candidate_json = json.dumps(candidate.to_dict(), sort_keys=True)
    bundle_json = json.dumps(candidate_bundle.to_dict(), sort_keys=True)
    rendered_json = json.dumps(
        render_transcript(_candidate_recording(recording, candidate)).to_dict(),
        sort_keys=True,
    )
    for score_only_key in ("speaker_mapping", "evaluator_speaker_map", "spk-a", "spk-b"):
        assert score_only_key not in candidate_json
        assert score_only_key not in bundle_json
        assert score_only_key not in rendered_json
