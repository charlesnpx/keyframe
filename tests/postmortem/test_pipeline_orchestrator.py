from pathlib import Path
from types import SimpleNamespace

from tests.preflight_helpers import patch_cli_media, transcript_preflight_stub


def test_model_provenance_uses_direct_loader_metadata(tmp_path):
    from keyframe.frames import _observed_model_provenance

    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"stable model weights")
    loaded = SimpleNamespace(
        config=SimpleNamespace(
            name_or_path="organization/observed-model",
            _commit_hash="abc123",
        ),
        checkpoint_file=weight,
    )

    provenance = _observed_model_provenance(
        "captioning",
        "fallback/model",
        loaded,
    )

    assert provenance == {
        "role": "captioning",
        "model_id": "organization/observed-model",
        "repository_revision": "abc123",
        "stable_weight_files": [
            {
                "loader_attribute": "checkpoint_file",
                "filename": "model.safetensors",
                "size_bytes": 20,
                "sha256": (
                    "72881e60b2f41b03362680924bd2d7d0"
                    "755bea0baf0f55f48dd4d286898afbb7"
                ),
            }
        ],
    }


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


def test_cli_frames_only_delegates_to_shared_pipeline(tmp_path, monkeypatch):
    import keyframe.pipeline as pipeline
    from keyframe import cli

    video = tmp_path / "input.mp4"
    video.write_bytes(b"not a real video")
    out_dir = tmp_path / "out"
    calls = []

    def fake_extract(video_path, output_dir, config, **kwargs):
        calls.append((video_path, output_dir, config, kwargs))
        return _fake_record_result(output_dir)

    monkeypatch.setattr(pipeline, "extract_keyframes", fake_extract)
    patch_cli_media(monkeypatch, video=True, audio=False)

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
    assert output_dir.parent.parent.name == "runs"
    assert output_dir.parent.parent.parent.name == ".keyframe-work"
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
    import keyframe.pipeline as pipeline
    from keyframe import cli

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
    monkeypatch.setattr(
        cli,
        "_preflight_transcript",
        lambda _args: transcript_preflight_stub(),
    )

    def fake_full_pipeline(
        video_path,
        output,
        _preflight,
        frame_config,
        supervisor,
    ):
        generation = cli._run_frame_generation(
            video_path,
            output,
            frame_config,
            supervisor,
        )
        generation.enrich_manifest(
            [{"start": 0.0, "end": 2.0, "text": "hello"}]
        )
        return SimpleNamespace(frames=generation.promote())

    monkeypatch.setattr(cli, "_run_full_pipeline", fake_full_pipeline)
    patch_cli_media(monkeypatch, video=True, audio=True)

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
    patch_cli_media(monkeypatch, video=False, audio=True)

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

    import keyframe.pipeline as pipeline
    from keyframe import cli

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
    patch_cli_media(monkeypatch, video=True, audio=False)

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
    from keyframe.pipeline.contracts import (
        CandidateRecord,
        FeatureOutput,
        FrameStore,
        SampleTable,
        SamplingOutput,
    )
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


def test_survival_output_cap_prioritizes_structured_delta_candidate():
    from PIL import Image

    from keyframe.pipeline.config import KeyframeExtractionConfig
    from keyframe.pipeline.context import make_context
    from keyframe.pipeline.contracts import (
        CandidateRecord,
        FeatureOutput,
        FrameStore,
        SampleTable,
        SamplingOutput,
    )
    from keyframe.pipeline.orchestrator import SurvivalStage
    from keyframe.pipeline.trace import NoOpTraceSink

    ordinary = CandidateRecord(
        sample_idx=0,
        frame_idx=0,
        timestamp=0.0,
    ).with_evidence(
        ocr_tokens=("ordinary", "screen", "content"),
    ).with_selection(candidate_score=100.0)
    structured = CandidateRecord(
        sample_idx=1,
        frame_idx=1,
        timestamp=10.0,
    ).with_evidence(
        ocr_tokens=("structured", "screen", "content"),
    ).with_selection(
        candidate_score=0.01,
        structured_delta_categories=("same_label_value",),
        structured_changed_signature_count=2,
    )
    sampling = SamplingOutput(
        frame_store=FrameStore(
            [
                Image.new("RGB", (16, 16), "white"),
                Image.new("RGB", (16, 16), "black"),
            ]
        ),
        samples=SampleTable(
            timestamps=[0.0, 10.0],
            frame_indices=[0, 1],
        ),
    )
    features = FeatureOutput(
        dhashes=[0, (1 << 16) - 1],
        clip_embeddings=None,
    )
    ctx = make_context(
        KeyframeExtractionConfig(max_output_frames=1),
        NoOpTraceSink(),
    )

    final = SurvivalStage().run(
        (ordinary, structured),
        sampling,
        features,
        ctx,
    )

    assert [candidate.sample_idx for candidate in final] == [1]
