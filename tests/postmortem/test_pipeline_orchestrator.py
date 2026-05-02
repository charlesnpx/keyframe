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


def test_cli_frames_only_delegates_to_shared_pipeline(tmp_path, monkeypatch):
    from keyframe import cli
    import keyframe.pipeline as pipeline

    video = tmp_path / "input.mp4"
    video.write_bytes(b"not a real video")
    out_dir = tmp_path / "out"
    calls = []

    def fake_extract(video_path, output_dir, config):
        calls.append((video_path, output_dir, config))
        return _fake_result(output_dir)

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
        return _fake_result(output_dir)

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
