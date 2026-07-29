from types import SimpleNamespace
from pathlib import Path


def _fake_result(output_dir):
    from keyframe.pipeline import KeyframeExtractionResult

    output_dir = Path(output_dir)
    return KeyframeExtractionResult(
        final=[{"path": str(output_dir / "frame_000001_1.00s.png"), "caption": "frame"}],
        output_dir=output_dir,
        caption_log_path=output_dir / "captions.json",
        manifest_path=output_dir / "manifest.json",
        manifest_metadata={"rescue": {"pre_rescue_candidate_count": 1, "rescue_budget": 3}},
        sampled_frame_count=4,
        pre_rescue_candidate_count=1,
        post_rescue_candidate_count=1,
        final_frame_count=1,
    )


def _fake_record_result(output_dir):
    import json

    from PIL import Image
    from keyframe.pipeline import KeyframeExtractionResult
    from keyframe.pipeline.contracts import CandidateRecord

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate = CandidateRecord(sample_idx=0, frame_idx=30, timestamp=1.0).with_evidence(
        caption="frame",
        ocr_tokens=("approved",),
    )
    filename = "frame_000030_1.00s.png"
    Image.new("RGB", (4, 4), "white").save(output_dir / filename)
    (output_dir / "captions.json").write_text(
        json.dumps([{"file": filename}]),
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "frames": [{"filename": filename}],
            }
        ),
        encoding="utf-8",
    )
    return KeyframeExtractionResult(
        final=(candidate,),
        output_dir=output_dir,
        caption_log_path=output_dir / "captions.json",
        manifest_path=output_dir / "manifest.json",
        manifest_metadata={"rescue": {"pre_rescue_candidate_count": 1, "rescue_budget": 3}},
        sampled_frame_count=4,
        pre_rescue_candidate_count=1,
        post_rescue_candidate_count=1,
        final_frame_count=1,
    )


def _stub_cli_preflight(monkeypatch, *, audio: bool = False, video: bool = True):
    from keyframe import frame_preflight, media_preflight
    import keyframe.full_pipeline as full_pipeline

    streams = []
    if video:
        streams.append(
            media_preflight.MediaStream(
                codec_type="video",
                codec_name="h264",
                width=16,
                height=16,
            )
        )
    if audio:
        streams.append(
            media_preflight.MediaStream(
                codec_type="audio",
                codec_name="aac",
                channels=1,
            )
        )
    monkeypatch.setattr(
        media_preflight,
        "probe_media",
        lambda _path: media_preflight.MediaProbeResult(tuple(streams)),
    )
    monkeypatch.setattr(
        frame_preflight,
        "preflight_frame_runtime",
        lambda: frame_preflight.FrameRuntimePlatform("Darwin", "arm64"),
    )
    monkeypatch.setattr(
        frame_preflight,
        "resolve_frame_execution_device",
        lambda _runtime: "cpu",
    )
    monkeypatch.setattr(full_pipeline, "resolve_frame_device", lambda _preflight: "cpu")


def test_cli_frames_only_delegates_to_shared_pipeline(tmp_path, monkeypatch):
    from keyframe import cli
    import keyframe.pipeline as pipeline

    video = tmp_path / "input.mp4"
    video.write_bytes(b"not a real video")
    out_dir = tmp_path / "out"
    calls = []

    def fake_extract(video_path, output_dir, config, **kwargs):
        calls.append((video_path, output_dir, config, kwargs))
        return _fake_record_result(output_dir)

    monkeypatch.setattr(pipeline, "extract_keyframes", fake_extract)
    _stub_cli_preflight(monkeypatch)

    cli.cmd_extract(SimpleNamespace(
        video=str(video),
        output=str(out_dir),
        transcript_only=False,
        frames_only=True,
        sample_interval=0.75,
        pass1_clusters=9,
        similarity_threshold=0.85,
        whisper_model="medium",
        transcript_format="txt",
    ))

    assert len(calls) == 1
    video_path, output_dir, config, kwargs = calls[0]
    assert video_path == video
    assert output_dir.name == "frames"
    assert output_dir.parent.name.startswith("keyframe-run-")
    assert kwargs["report_output_dir"] == out_dir / "frames"
    assert config.sample_interval == 0.75
    assert config.pass1_clusters == 9
    assert config.device == "cpu"
    assert (out_dir / "frames" / "frame_000030_1.00s.png").exists()


def test_frames_main_delegates_to_shared_pipeline(tmp_path, monkeypatch):
    import sys
    import keyframe.frames as frames_mod
    import keyframe.pipeline as pipeline

    video = tmp_path / "input.mp4"
    video.write_bytes(b"not a real video")
    out_dir = tmp_path / "frames"
    calls = []

    def fake_extract(video_path, output_dir, config):
        calls.append((video_path, output_dir, config))
        return _fake_record_result(output_dir)

    monkeypatch.setattr(pipeline, "extract_keyframes", fake_extract)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "frames.py",
            str(video),
            "--output-dir",
            str(out_dir),
            "--sample-interval",
            "0.25",
            "--pass1-clusters",
            "7",
        ],
    )

    frames_mod.main()

    assert len(calls) == 1
    video_path, output_dir, config = calls[0]
    assert video_path == str(video)
    assert output_dir == str(out_dir)
    assert config.sample_interval == 0.25
    assert config.pass1_clusters == 7


def test_cli_transcript_manifest_rewrite_materializes_candidate_records(tmp_path, monkeypatch):
    from keyframe import cli
    import keyframe.pipeline as pipeline

    video = tmp_path / "input.mp4"
    video.write_bytes(b"not a real video")
    out_dir = tmp_path / "out"

    monkeypatch.setattr(
        pipeline,
        "extract_keyframes",
        lambda video_path, output_dir, config, **_kwargs: _fake_record_result(
            output_dir
        ),
    )
    monkeypatch.setattr(cli, "_preflight_transcript", lambda _args: object())
    _stub_cli_preflight(monkeypatch, audio=True)

    def fake_full_pipeline(video_path, output, args, _preflight, supervisor):
        generation = cli._run_frame_generation(
            video_path,
            output,
            args,
            supervisor,
            frame_device="cpu",
        )
        generation.enrich_manifest(
            [{"start": 0.0, "end": 2.0, "text": "hello"}]
        )
        return SimpleNamespace(frames=generation.promote())

    monkeypatch.setattr(cli, "_run_full_pipeline", fake_full_pipeline)

    cli.cmd_extract(SimpleNamespace(
        video=str(video),
        output=str(out_dir),
        transcript_only=False,
        frames_only=False,
        sample_interval=0.75,
        pass1_clusters=9,
        similarity_threshold=0.85,
        max_output_frames=None,
        verbose_trace=False,
        debug_qa_targets=None,
        whisper_model="medium",
        transcript_format="txt",
    ))

    import json

    manifest = json.loads((out_dir / "frames" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["frames"][0]["timestamp"] == 1.0
    assert manifest["frames"][0]["filename"] == "frame_000030_1.00s.png"
    assert manifest["frames"][0]["transcript_window"] == "hello"


def test_cli_no_speaker_detection_passed_to_transcript(tmp_path, monkeypatch):
    from keyframe import cli

    video = tmp_path / "input.mp4"
    video.write_bytes(b"not a real video")
    out_dir = tmp_path / "out"
    calls = []

    def fake_run_transcript(_video, _output, preflight):
        calls.append(preflight)
        return SimpleNamespace(
            segments=[{"start": 0.0, "end": 2.0, "text": "hello"}],
            language="en",
        )

    monkeypatch.setattr(cli, "_run_transcript", fake_run_transcript)
    _stub_cli_preflight(monkeypatch, audio=True, video=False)

    cli.cmd_extract(SimpleNamespace(
        video=str(video),
        output=str(out_dir),
        transcript_only=True,
        frames_only=False,
        sample_interval=0.75,
        pass1_clusters=9,
        similarity_threshold=0.85,
        max_output_frames=None,
        verbose_trace=False,
        debug_qa_targets=None,
        whisper_model="medium",
        transcript_format="json",
        no_speaker_detection=True,
    ))

    assert calls[0].config.speaker_detection is False


def test_cli_frames_only_does_not_import_transcript(tmp_path, monkeypatch):
    import sys
    from keyframe import cli
    import keyframe.pipeline as pipeline

    video = tmp_path / "input.mp4"
    video.write_bytes(b"not a real video")
    out_dir = tmp_path / "out"

    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setattr(
        pipeline,
        "extract_keyframes",
        lambda video_path, output_dir, config, **_kwargs: _fake_record_result(
            output_dir
        ),
    )
    monkeypatch.delitem(sys.modules, "keyframe.transcript", raising=False)
    _stub_cli_preflight(monkeypatch)

    cli.cmd_extract(SimpleNamespace(
        video=str(video),
        output=str(out_dir),
        transcript_only=False,
        frames_only=True,
        sample_interval=0.75,
        pass1_clusters=9,
        similarity_threshold=0.85,
        max_output_frames=None,
        verbose_trace=False,
        debug_qa_targets=None,
        whisper_model="medium",
        transcript_format="txt",
        no_speaker_detection=False,
    ))

    assert "keyframe.transcript" not in sys.modules


def test_survival_stage_applies_explicit_output_cap_after_dedupe():
    from PIL import Image
    from keyframe.pipeline.config import KeyframeExtractionConfig
    from keyframe.pipeline.context import make_context
    from keyframe.pipeline.contracts import CandidateRecord, FeatureOutput, FrameStore, SampleTable, SamplingOutput
    from keyframe.pipeline.orchestrator import SurvivalStage
    from keyframe.pipeline.trace import NoOpTraceSink

    candidates = tuple(
        CandidateRecord(sample_idx=i, frame_idx=i, timestamp=float(i * 10)).with_evidence(
            ocr_tokens=(f"token{i}", "common", "field"),
        ).with_selection(candidate_score=float(i))
        for i in range(4)
    )
    sampling = SamplingOutput(
        frame_store=FrameStore([Image.new("RGB", (16, 16), "white") for _ in range(4)]),
        samples=SampleTable(timestamps=[float(i * 10) for i in range(4)], frame_indices=list(range(4))),
    )
    features = FeatureOutput(dhashes=[i for i in range(4)], clip_embeddings=None)
    ctx = make_context(KeyframeExtractionConfig(max_output_frames=2), NoOpTraceSink())

    final = SurvivalStage().run(candidates, sampling, features, ctx)

    assert len(final) == 2
    assert ctx.metadata["survival"]["cap_pressure"] == 2
    assert len(ctx.metadata["survival"]["cap_dropped_frames"]) == 2


def test_survival_output_cap_preserves_coverage_before_score():
    from PIL import Image
    from keyframe.pipeline.config import KeyframeExtractionConfig
    from keyframe.pipeline.context import make_context
    from keyframe.pipeline.contracts import CandidateRecord, FeatureOutput, FrameStore, SampleTable, SamplingOutput
    from keyframe.pipeline.orchestrator import SurvivalStage
    from keyframe.pipeline.trace import NoOpTraceSink

    candidates = (
        CandidateRecord(sample_idx=0, frame_idx=0, timestamp=0.0)
        .with_evidence(ocr_tokens=("coverage", "window", "state"))
        .with_temporal(coverage_window_ids=(0,))
        .with_selection(candidate_score=0.0, selection_role="coverage"),
        CandidateRecord(sample_idx=1, frame_idx=1, timestamp=10.0)
        .with_evidence(ocr_tokens=("high", "score", "state"))
        .with_selection(candidate_score=100.0),
    )
    sampling = SamplingOutput(
        frame_store=FrameStore([Image.new("RGB", (16, 16), "white") for _ in range(2)]),
        samples=SampleTable(timestamps=[0.0, 10.0], frame_indices=[0, 1]),
    )
    features = FeatureOutput(dhashes=[0, 255], clip_embeddings=None)
    ctx = make_context(KeyframeExtractionConfig(max_output_frames=1), NoOpTraceSink())

    final = SurvivalStage().run(candidates, sampling, features, ctx)

    assert [candidate.sample_idx for candidate in final] == [0]


def test_duration_coverage_pool_selects_settled_global_windows():
    import numpy as np
    from keyframe.pipeline.orchestrator import _coverage_candidate_pool

    timestamps = [float(index) for index in range(300)]
    frame_indices = list(range(300))
    dhashes = [0] * 100 + [255] * 100 + [0] * 100
    embeddings = np.zeros((300, 3), dtype=np.float32)
    embeddings[:100, 0] = 1.0
    embeddings[100:200, 1] = 1.0
    embeddings[200:, 2] = 1.0

    pool = _coverage_candidate_pool(
        timestamps=timestamps,
        frame_indices=frame_indices,
        dhashes=dhashes,
        clip_embeddings=embeddings,
        frame_metrics=None,
        coverage_interval_seconds=90.0,
        minimum_settled_dwell_seconds=2.0,
        duration_seconds=300.0,
    )

    assert [candidate.temporal.coverage_window_ids for candidate in pool] == [
        (0,),
        (1,),
        (2,),
        (3,),
    ]
    assert [candidate.selection.selection_role for candidate in pool] == [
        "coverage",
        "coverage",
        "coverage",
        "coverage",
    ]


def test_survival_output_cap_orders_coverage_durable_structured_then_remaining():
    import numpy as np
    from PIL import Image
    from keyframe.pipeline.config import KeyframeExtractionConfig
    from keyframe.pipeline.context import make_context
    from keyframe.pipeline.contracts import CandidateRecord, FeatureOutput, FrameStore, SampleTable, SamplingOutput
    from keyframe.pipeline.orchestrator import SurvivalStage
    from keyframe.pipeline.trace import NoOpTraceSink

    candidates = (
        CandidateRecord(sample_idx=0, frame_idx=0, timestamp=0.0)
        .with_evidence(ocr_tokens=("coverage", "window", "zero"))
        .with_selection(candidate_score=0.0, selection_role="coverage"),
        CandidateRecord(sample_idx=1, frame_idx=1, timestamp=10.0)
        .with_temporal(durable_state_group_id=1)
        .with_evidence(ocr_tokens=("durable", "workflow", "state"))
        .with_selection(candidate_score=1.0, selection_role="durable_state"),
        CandidateRecord(sample_idx=2, frame_idx=2, timestamp=20.0)
        .with_evidence(
            ocr_tokens=("structured", "status", "approved"),
            field_signature=("field-state:status:approved",),
        )
        .with_selection(candidate_score=2.0, retention_reason="differing_evidence"),
        CandidateRecord(sample_idx=3, frame_idx=3, timestamp=30.0)
        .with_evidence(ocr_tokens=("semantic", "high", "score"))
        .with_selection(candidate_score=100.0, selection_role="semantic"),
    )
    sampling = SamplingOutput(
        frame_store=FrameStore([Image.new("RGB", (16, 16), "white") for _ in candidates]),
        samples=SampleTable(
            timestamps=[candidate.timestamp for candidate in candidates],
            frame_indices=[candidate.frame_idx for candidate in candidates],
        ),
    )
    features = FeatureOutput(
        dhashes=[0, 0xFF, 0xFF00, 0xFF0000],
        clip_embeddings=np.eye(4, dtype=np.float32),
    )
    ctx = make_context(KeyframeExtractionConfig(max_output_frames=3), NoOpTraceSink())

    final = SurvivalStage().run(candidates, sampling, features, ctx)

    assert [candidate.sample_idx for candidate in final] == [0, 1, 2]


def test_survival_output_cap_balances_durable_states_across_windows():
    import numpy as np
    from PIL import Image
    from keyframe.pipeline.config import KeyframeExtractionConfig
    from keyframe.pipeline.context import make_context
    from keyframe.pipeline.contracts import CandidateRecord, FeatureOutput, FrameStore, SampleTable, SamplingOutput
    from keyframe.pipeline.orchestrator import SurvivalStage
    from keyframe.pipeline.trace import NoOpTraceSink

    candidates = (
        CandidateRecord(sample_idx=0, frame_idx=0, timestamp=0.0)
        .with_evidence(ocr_tokens=("coverage", "window", "state"))
        .with_selection(selection_role="coverage"),
        CandidateRecord(sample_idx=1, frame_idx=1, timestamp=10.0)
        .with_temporal(temporal_window_id=0, durable_state_group_id=1)
        .with_evidence(ocr_tokens=("durable", "first", "state"))
        .with_selection(selection_role="durable_state", candidate_score=10.0),
        CandidateRecord(sample_idx=2, frame_idx=2, timestamp=20.0)
        .with_temporal(temporal_window_id=0, durable_state_group_id=2)
        .with_evidence(ocr_tokens=("durable", "second", "state"))
        .with_selection(selection_role="durable_state", candidate_score=9.0),
        CandidateRecord(sample_idx=3, frame_idx=3, timestamp=30.0)
        .with_temporal(temporal_window_id=1, durable_state_group_id=3)
        .with_evidence(ocr_tokens=("durable", "later", "state"))
        .with_selection(selection_role="durable_state", candidate_score=1.0),
        CandidateRecord(sample_idx=4, frame_idx=4, timestamp=40.0)
        .with_evidence(ocr_tokens=("semantic", "ordinary", "state"))
        .with_selection(selection_role="semantic", candidate_score=100.0),
    )
    sampling = SamplingOutput(
        frame_store=FrameStore([Image.new("RGB", (16, 16), "white") for _ in candidates]),
        samples=SampleTable(
            timestamps=[candidate.timestamp for candidate in candidates],
            frame_indices=[candidate.frame_idx for candidate in candidates],
        ),
    )
    features = FeatureOutput(
        dhashes=[0, 0xFF, 0xFF00, 0xFF0000, 0xFF000000],
        clip_embeddings=np.eye(5, dtype=np.float32),
    )
    ctx = make_context(KeyframeExtractionConfig(max_output_frames=3), NoOpTraceSink())

    final = SurvivalStage().run(candidates, sampling, features, ctx)

    assert [candidate.sample_idx for candidate in final] == [0, 1, 3]
