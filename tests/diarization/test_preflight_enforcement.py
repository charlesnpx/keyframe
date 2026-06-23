from keyframe.diarization import (
    CanonicalRecording,
    CanonicalWord,
    ChannelRecord,
    PreflightFeatures,
    PreflightJobRecord,
    PreflightManualOverrideAudit,
    PreflightRouteAssessment,
    PreflightRouteDecision,
    SpeakerRecord,
    build_preflight_route_confusion_report,
    render_transcript_for_preflight,
)


_FROZEN_SHA = "b" * 40


def _features():
    return PreflightFeatures(
        declared_locale="en-US",
        source="zoom",
        capture_mode="separate_tracks",
        channel_count=2,
        duration_ms=900_000,
        sample_rate_hz=16_000,
        codec="pcm_s16le",
        clipping_estimate=0.01,
        speech_ratio=0.62,
        rough_overlap_estimate=0.12,
        speaker_count_hint=3,
    )


def _decision(route, reasons=()):
    return PreflightRouteDecision(
        route=route,
        reasons=reasons,
        policy_id="launch-preflight",
        policy_version="2026-06-23",
        frozen_git_sha=_FROZEN_SHA,
        tuned_on_splits=("public_dev",),
        validated_on_splits=("public_holdout", "private_acceptance"),
        features=_features(),
    )


def _recording():
    return CanonicalRecording(
        recording_id="rec-route",
        original_audio_id="original-route",
        canonical_audio_id="canonical-route",
        timeline_id="timeline-route",
        duration_ms=2_000,
        channels=(ChannelRecord("ch-1"),),
        speakers=(SpeakerRecord("spk-a"), SpeakerRecord("spk-b")),
        words=(
            CanonicalWord("w-1", "hello", 0, 200, speaker_ref="spk-a", channel_id="ch-1", speaker_confidence=0.92),
            CanonicalWord("w-2", "there", 250, 450, speaker_ref="spk-b", channel_id="ch-1", speaker_confidence=0.91),
        ),
        speaker_spans=(),
        scoring_regions=(),
    )


def test_needs_review_preflight_route_renders_degraded_attribution_without_speaker_labels():
    transcript = render_transcript_for_preflight(
        _recording(),
        _decision("needs_review", ("speaker_count_hint_unknown",)),
    )

    assert transcript.state == "needs_review"
    assert transcript.speaker_attribution == "unavailable"
    assert transcript.review_reasons == ("speaker_attribution_unavailable", "manual_review_required")
    assert [word.text for word in transcript.words] == ["hello", "there"]
    assert all(word.label is None and word.display_label is None for word in transcript.words)
    assert all(turn.label is None and turn.display_label is None for turn in transcript.turns)


def test_diagnostic_only_preflight_route_renders_transcript_only_output():
    transcript = render_transcript_for_preflight(
        _recording(),
        _decision("diagnostic_only", ("capture_mode_outside_confident_scope",)),
    )

    assert transcript.state == "diagnostic_only"
    assert transcript.speaker_attribution == "unavailable"
    assert transcript.review_reasons == ("diagnostic_only", "speaker_attribution_unavailable")
    assert [word.text for word in transcript.words] == ["hello", "there"]
    assert all(word.label is None for word in transcript.words)


def test_route_confusion_excludes_manual_overrides_from_benchmark_truth():
    report = build_preflight_route_confusion_report(
        (
            PreflightRouteAssessment(
                corpus_id="private",
                branch_id="mono-mix",
                recording_id="manual-override",
                predicted_route="confident_pipeline",
                reference_route="diagnostic_only",
                manual_override_applied=True,
            ),
            PreflightRouteAssessment(
                corpus_id="private",
                branch_id="mono-mix",
                recording_id="false-confident",
                predicted_route="confident_pipeline",
                reference_route="diagnostic_only",
            ),
        )
    )

    assert report.manual_override_count == 1
    assert report.out_of_scope_false_confident_count == 1
    assert report.serious_failure_count == 1
    assert report.matrix["diagnostic_only"]["confident_pipeline"] == 1
    assert report.to_dict()["assessments"][0]["counted_in_benchmark"] is False


def test_manual_override_audit_is_marked_out_of_benchmark_truth():
    audit = PreflightManualOverrideAudit(
        override_id="override-001",
        actor_id="reviewer-1",
        reason="licensed customer escalation with human review",
        override_route="needs_review",
        created_at="2026-06-23T16:00:00Z",
    )

    assert audit.to_dict()["excluded_from_benchmark_truth"] is True


def test_manual_override_effective_route_suppresses_confident_speaker_labels():
    job = PreflightJobRecord(
        job_id="job-override",
        decision=_decision("confident_pipeline"),
        validated_launch_scope_version="private-coverage-v1@2026-06-23",
        manual_override=PreflightManualOverrideAudit(
            override_id="override-001",
            actor_id="reviewer-1",
            reason="force human review before customer delivery",
            override_route="diagnostic_only",
            created_at="2026-06-23T16:00:00Z",
        ),
    )

    transcript = render_transcript_for_preflight(_recording(), job)

    assert transcript.state == "diagnostic_only"
    assert transcript.speaker_attribution == "unavailable"
    assert all(word.label is None for word in transcript.words)
