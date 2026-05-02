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
    from keyframe.pipeline import KeyframeExtractionResult
    from keyframe.pipeline.contracts import CandidateRecord

    output_dir = Path(output_dir)
    candidate = CandidateRecord(sample_idx=0, frame_idx=30, timestamp=1.0).with_evidence(
        caption="frame",
        ocr_tokens=("approved",),
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


def test_cli_frames_only_delegates_to_shared_pipeline(tmp_path, monkeypatch):
    from keyframe import cli
    import keyframe.pipeline as pipeline

    video = tmp_path / "input.mp4"
    video.write_bytes(b"not a real video")
    out_dir = tmp_path / "out"
    calls = []

    def fake_extract(video_path, output_dir, config):
        calls.append((video_path, output_dir, config))
        return _fake_record_result(output_dir)

    monkeypatch.setattr(pipeline, "extract_keyframes", fake_extract)

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
    video_path, output_dir, config = calls[0]
    assert video_path == video
    assert output_dir == out_dir / "frames"
    assert config.sample_interval == 0.75
    assert config.pass1_clusters == 9


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
    import keyframe.transcript as transcript

    video = tmp_path / "input.mp4"
    video.write_bytes(b"not a real video")
    out_dir = tmp_path / "out"

    monkeypatch.setattr(pipeline, "extract_keyframes", lambda video_path, output_dir, config: _fake_record_result(output_dir))
    monkeypatch.setattr(
        transcript,
        "extract_transcript",
        lambda **kwargs: ([{"start": 0.0, "end": 2.0, "text": "hello"}], "en"),
    )

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


def test_survival_stage_applies_explicit_output_cap_after_dedupe():
    from PIL import Image
    from keyframe.pipeline.config import KeyframeExtractionConfig
    from keyframe.pipeline.context import make_context
    from keyframe.pipeline.contracts import CandidateRecord, FeatureOutput, FrameStore, SampleTable, SamplingOutput
    from keyframe.pipeline.orchestrator import SurvivalStage
    from keyframe.pipeline.trace import NoOpTraceSink

    candidates = tuple(
        CandidateRecord(sample_idx=i, frame_idx=i, timestamp=float(i)).with_evidence(
            ocr_tokens=(f"token{i}", "common", "field"),
        ).with_selection(candidate_score=float(i))
        for i in range(4)
    )
    sampling = SamplingOutput(
        frame_store=FrameStore([Image.new("RGB", (16, 16), "white") for _ in range(4)]),
        samples=SampleTable(timestamps=[float(i) for i in range(4)], frame_indices=list(range(4))),
    )
    features = FeatureOutput(dhashes=[i for i in range(4)], clip_embeddings=None)
    ctx = make_context(KeyframeExtractionConfig(max_output_frames=2), NoOpTraceSink())

    final = SurvivalStage().run(candidates, sampling, features, ctx)

    assert len(final) == 2
    assert ctx.metadata["survival"]["cap_pressure"] == 2
    assert len(ctx.metadata["survival"]["cap_dropped_frames"]) == 2
