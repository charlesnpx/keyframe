import json
from dataclasses import replace
from pathlib import Path

import pytest

from keyframe.diarization import (
    ChannelRecord,
    EngineConfigMetadata,
    NormalizedArtifactProvenance,
    NormalizedEngineOutput,
    ReferenceBundle,
    ScoringRegion,
    SpeakerRecord,
    SpeakerSpan,
    build_candidate_bundle,
    build_evaluation_slices,
    default_scoring_policy,
    evaluate_diarization_candidate,
    read_recording_json,
    render_transcript,
    ValidationError,
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


def _many_speaker_recording(speaker_count=17):
    recording = _recording()
    speaker_refs = tuple(f"spk-{index:02d}" for index in range(speaker_count))
    return replace(
        recording,
        duration_ms=2_000,
        speakers=tuple(SpeakerRecord(speaker_ref) for speaker_ref in speaker_refs),
        words=(),
        speaker_spans=tuple(
            SpeakerSpan(
                span_id=f"span-{index:02d}",
                speaker_ref=speaker_ref,
                start_ms=index * 100,
                end_ms=index * 100 + 50,
                channel_id="ch-1",
            )
            for index, speaker_ref in enumerate(speaker_refs)
        ),
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

    result = evaluate_diarization_candidate(
        reference,
        candidate,
        scoring_policy=replace(default_scoring_policy("diagnostic_diarization"), collar_ms=0),
    )

    assert result.speaker_mapping == {
        "engine:local:speaker-1": "spk-b",
        "engine:local:speaker-2": "spk-a",
    }
    assert result.recording_metrics[0].metrics["speaker_label_accuracy"] == 1.0
    assert result.recording_metrics[0].metrics["speaker_label_error_rate"] == 0.0
    assert result.recording_metrics[0].metrics["diarization_error_rate"] == 0.0
    assert result.recording_metrics[0].metrics["matched_speaker_ms"] == 750


def test_candidate_from_different_recording_is_rejected_even_with_same_duration():
    recording = _recording()
    other_recording = replace(
        recording,
        recording_id="different-recording",
        original_audio_id="different-original",
        canonical_audio_id="different-canonical",
        timeline_id="different-timeline",
    )
    candidate = _candidate_output(
        other_recording,
        {
            "spk-a": "engine:local:speaker-2",
            "spk-b": "engine:local:speaker-1",
        },
    )

    with pytest.raises(ValidationError, match="conflicts"):
        evaluate_diarization_candidate(_reference_bundle(recording), candidate)


def test_candidate_channels_must_match_reference_channel_layout():
    recording = _recording()
    reference = _reference_bundle(recording)
    candidate = _candidate_output(
        recording,
        {
            "spk-a": "engine:local:speaker-2",
            "spk-b": "engine:local:speaker-1",
        },
    )
    bad_span_channel = replace(
        candidate,
        speaker_spans=(
            replace(candidate.speaker_spans[0], channel_id="not-a-channel"),
            candidate.speaker_spans[1],
        ),
    )
    bad_word_channel = replace(
        candidate,
        words=(
            replace(candidate.words[0], channel_id="not-a-channel"),
            candidate.words[1],
        ),
    )

    with pytest.raises(ValidationError, match="speaker span channel_id conflicts"):
        evaluate_diarization_candidate(reference, bad_span_channel)
    with pytest.raises(ValidationError, match="word channel_id conflicts"):
        evaluate_diarization_candidate(reference, bad_word_channel)


def test_large_speaker_assignment_fails_closed_instead_of_using_greedy_mapping():
    recording = _many_speaker_recording()
    candidate = _candidate_output(
        recording,
        {speaker.speaker_ref: f"engine:{speaker.speaker_ref}" for speaker in recording.speakers},
    )

    with pytest.raises(ValidationError, match="bounded exact matcher"):
        evaluate_diarization_candidate(_reference_bundle(recording), candidate)


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
    assert by_id["speaker_change_boundary:within_collar"].status == "insufficient_support"
    assert by_id["turn_duration:short"].status == "ready"
    assert by_id["turn_duration:long"].status == "insufficient_support"
    assert by_id["channel_mode:mono"].status == "ready"
    assert by_id["channel_mode:multichannel"].status == "insufficient_support"


def test_speaker_change_boundary_slice_requires_actual_speaker_change():
    recording = _recording()
    contiguous_change = replace(
        recording,
        speaker_spans=(
            replace(recording.speaker_spans[0], end_ms=500, overlap=False),
            replace(recording.speaker_spans[1], start_ms=500, overlap=False),
        ),
    )

    slices = build_evaluation_slices(contiguous_change, minimum_support_ms=1)
    by_id = {item.slice_id: item for item in slices}

    boundary = by_id["speaker_change_boundary:within_collar"]
    assert boundary.status == "ready"
    assert [(item.start_ms, item.end_ms) for item in boundary.intervals] == [(250, 750)]


def test_diagnostic_collar_excludes_boundary_shift_from_metrics():
    recording = _recording()
    reference_recording = replace(
        recording,
        duration_ms=2_000,
        speaker_spans=(
            replace(recording.speaker_spans[0], start_ms=300, end_ms=1_000, overlap=False),
            replace(recording.speaker_spans[1], start_ms=1_000, end_ms=1_700, overlap=False),
        ),
        scoring_regions=(replace(recording.scoring_regions[0], end_ms=2_000),),
    )
    candidate = _candidate_output(
        reference_recording,
        {
            "spk-a": "engine:local:speaker-1",
            "spk-b": "engine:local:speaker-2",
        },
    )
    shifted_candidate = replace(
        candidate,
        speaker_spans=(
            replace(candidate.speaker_spans[0], end_ms=1_100),
            replace(candidate.speaker_spans[1], start_ms=1_100),
        ),
    )

    result = evaluate_diarization_candidate(_reference_bundle(reference_recording), shifted_candidate)
    metrics = result.recording_metrics[0].metrics

    assert metrics["reference_speaker_ms"] == 400
    assert metrics["matched_speaker_ms"] == 400
    assert metrics["diarization_error_rate"] == 0.0


def test_diagnostic_collar_excludes_single_speaker_onset_offset_shift_from_metrics():
    recording = _recording()
    reference_recording = replace(
        recording,
        duration_ms=2_000,
        speakers=(recording.speakers[0],),
        words=(),
        speaker_spans=(
            SpeakerSpan(
                span_id="span-1",
                speaker_ref="spk-a",
                start_ms=300,
                end_ms=1_700,
                channel_id="ch-1",
            ),
        ),
        scoring_regions=(replace(recording.scoring_regions[0], end_ms=2_000),),
    )
    candidate = _candidate_output(reference_recording, {"spk-a": "engine:local:speaker-1"})
    shifted_candidate = replace(
        candidate,
        speaker_spans=(
            replace(candidate.speaker_spans[0], start_ms=400, end_ms=1_600),
        ),
    )

    result = evaluate_diarization_candidate(_reference_bundle(reference_recording), shifted_candidate)
    metrics = result.recording_metrics[0].metrics

    assert metrics["reference_speaker_ms"] == 900
    assert metrics["matched_speaker_ms"] == 900
    assert metrics["diarization_error_rate"] == 0.0


def test_diagnostic_collar_excludes_uem_edge_onset_offset_shift_from_metrics():
    recording = _recording()
    reference_recording = replace(
        recording,
        duration_ms=1_000,
        speakers=(recording.speakers[0],),
        words=(),
        speaker_spans=(
            SpeakerSpan(
                span_id="span-1",
                speaker_ref="spk-a",
                start_ms=0,
                end_ms=1_000,
                channel_id="ch-1",
            ),
        ),
    )
    candidate = _candidate_output(reference_recording, {"spk-a": "engine:local:speaker-1"})
    shifted_candidate = replace(
        candidate,
        speaker_spans=(
            replace(candidate.speaker_spans[0], start_ms=100, end_ms=900),
        ),
    )

    result = evaluate_diarization_candidate(_reference_bundle(reference_recording), shifted_candidate)
    metrics = result.recording_metrics[0].metrics

    assert metrics["reference_speaker_ms"] == 500
    assert metrics["matched_speaker_ms"] == 500
    assert metrics["diarization_error_rate"] == 0.0


def test_rendered_transcript_policy_collapses_physical_channels_for_scoring():
    recording = _recording()
    multichannel = replace(
        recording,
        channels=(
            recording.channels[0],
            ChannelRecord("ch-2", "second"),
        ),
        words=(),
        speaker_spans=(
            SpeakerSpan(
                span_id="span-1",
                speaker_ref="spk-a",
                start_ms=0,
                end_ms=500,
                channel_id="ch-1",
            ),
            SpeakerSpan(
                span_id="span-2",
                speaker_ref="spk-b",
                start_ms=0,
                end_ms=500,
                channel_id="ch-2",
            ),
        ),
        scoring_regions=(
            ScoringRegion("uem-1", 0, 1_000, channel_id="ch-1"),
            ScoringRegion("uem-2", 0, 1_000, channel_id="ch-2"),
        ),
    )
    candidate = _candidate_output(
        multichannel,
        {
            "spk-a": "engine:local:speaker-1",
            "spk-b": "engine:local:speaker-2",
        },
    )
    rendered_candidate = replace(
        candidate,
        words=tuple(replace(word, channel_id=None) for word in candidate.words),
        speaker_spans=tuple(replace(span, channel_id=None) for span in candidate.speaker_spans),
    )

    result = evaluate_diarization_candidate(
        _reference_bundle(multichannel),
        rendered_candidate,
        scoring_policy=replace(
            default_scoring_policy("diagnostic_diarization"),
            channel_mode="rendered_transcript",
            collar_ms=0,
        ),
    )
    metrics = result.recording_metrics[0].metrics
    short_turn_metrics = {row.slice_id: row for row in result.slice_metrics}["turn_duration:short"].metrics

    assert metrics["reference_speaker_ms"] == 1000
    assert metrics["hypothesis_speaker_ms"] == 1000
    assert metrics["false_alarm_speaker_ms"] == 0
    assert metrics["diarization_error_rate"] == 0.0
    assert short_turn_metrics["reference_speaker_ms"] == 1000
    assert short_turn_metrics["hypothesis_speaker_ms"] == 1000
    assert short_turn_metrics["false_alarm_speaker_ms"] == 0


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

    result = evaluate_diarization_candidate(
        reference,
        candidate,
        scoring_policy=replace(default_scoring_policy("diagnostic_diarization"), collar_ms=0),
    )
    overlap_row = {row.slice_id: row for row in result.slice_metrics}["overlap:overlap"]

    assert overlap_row.status == "scored"
    assert overlap_row.support_ms == 500
    assert overlap_row.metrics["speaker_label_accuracy"] == 1.0


def test_product_policy_excludes_single_speaker_regions_flagged_as_overlap():
    recording = _recording()
    overlap_flagged = replace(
        recording,
        speaker_spans=(
            replace(recording.speaker_spans[0], overlap=True),
            recording.speaker_spans[1],
        ),
        words=(
            replace(recording.words[0], overlap=True),
            recording.words[1],
        ),
    )
    reference = _reference_bundle(overlap_flagged)
    candidate = _candidate_output(
        overlap_flagged,
        {
            "spk-a": "engine:local:speaker-2",
            "spk-b": "engine:local:speaker-1",
        },
    )

    result = evaluate_diarization_candidate(
        reference,
        candidate,
        scoring_policy=default_scoring_policy("product_transcript"),
    )
    metrics = result.recording_metrics[0].metrics
    overlap_row = {row.slice_id: row for row in result.slice_metrics}["overlap:overlap"]

    assert result.speaker_mapping == {"engine:local:speaker-1": "spk-b"}
    assert metrics["reference_speaker_ms"] == 400
    assert metrics["matched_speaker_ms"] == 400
    assert metrics["speaker_label_accuracy"] == 1.0
    assert overlap_row.status == "insufficient_support"
    assert overlap_row.support_ms == 0
    assert overlap_row.metrics == {}


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

    result = evaluate_diarization_candidate(
        reference,
        candidate,
        scoring_policy=replace(default_scoring_policy("diagnostic_diarization"), collar_ms=0),
    )

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
