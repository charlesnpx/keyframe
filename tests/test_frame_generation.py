import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from keyframe import cli
from keyframe.frame_generation import (
    FrameGenerationPromotionError,
    FrameGenerationSession,
    FrameGenerationValidationError,
    StagedFrameGeneration,
    validate_frame_generation,
)
from keyframe.pipeline.config import KeyframeExtractionResult
from keyframe.pipeline.contracts import CandidateRecord


def _candidate(frame_idx, timestamp, text="frame"):
    return CandidateRecord(
        sample_idx=frame_idx,
        frame_idx=frame_idx,
        timestamp=timestamp,
    ).with_evidence(caption=text, ocr_tokens=(text,))


def _filename(candidate):
    return f"frame_{candidate.frame_idx:06d}_{candidate.timestamp:.2f}s.png"


def _write_result(output_dir, candidates):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = tuple(candidates)
    names = [_filename(candidate) for candidate in candidates]
    for index, name in enumerate(names):
        Image.new("RGB", (6, 6), (index * 40, 10, 20)).save(output_dir / name)
    (output_dir / "captions.json").write_text(
        json.dumps(
            [
                {
                    "file": name,
                    "timestamp": candidate.timestamp,
                    "caption": candidate.evidence.caption,
                }
                for name, candidate in zip(names, candidates)
            ]
        ),
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "frames": [
                    {
                        "filename": name,
                        "timestamp": candidate.timestamp,
                        "caption": candidate.evidence.caption,
                        "transcript_window": "",
                    }
                    for name, candidate in zip(names, candidates)
                ],
                "metadata": {"generation": "test"},
            }
        ),
        encoding="utf-8",
    )
    return KeyframeExtractionResult(
        final=candidates,
        output_dir=output_dir,
        caption_log_path=output_dir / "captions.json",
        manifest_path=output_dir / "manifest.json",
        manifest_metadata={"generation": "test"},
        sampled_frame_count=max(4, len(candidates)),
        pre_rescue_candidate_count=len(candidates),
        post_rescue_candidate_count=len(candidates),
        final_frame_count=len(candidates),
    )


def _stage(session, candidates):
    assert session.staging is not None
    result = _write_result(session.staging.frames, candidates)
    return StagedFrameGeneration.from_extraction(session, result)


def _tree_snapshot(path):
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def _frames_only_args(video, output):
    return SimpleNamespace(
        video=str(video),
        output=str(output),
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
    )


def test_rerun_with_fewer_frames_replaces_the_complete_generation(tmp_path):
    output = tmp_path / "out"
    first = (
        _candidate(10, 1.0, "first"),
        _candidate(20, 2.0, "obsolete"),
    )
    second = (_candidate(30, 3.0, "replacement"),)

    with FrameGenerationSession(output, run_id="first") as session:
        public_result = _stage(session, first).promote()

    assert public_result.output_dir == output / "frames"
    assert sorted(path.name for path in (output / "frames").glob("*.png")) == [
        _filename(candidate) for candidate in first
    ]

    with FrameGenerationSession(output, run_id="second") as session:
        _stage(session, second).promote()

    assert sorted(path.name for path in (output / "frames").glob("*.png")) == [
        _filename(second[0])
    ]
    captions = json.loads((output / "frames" / "captions.json").read_text())
    manifest = json.loads((output / "frames" / "manifest.json").read_text())
    assert [row["file"] for row in captions] == [_filename(second[0])]
    assert [row["filename"] for row in manifest["frames"]] == [
        _filename(second[0])
    ]


def test_replacement_preserves_public_directory_permissions(tmp_path):
    output = tmp_path / "out"
    with FrameGenerationSession(output, run_id="first") as session:
        _stage(session, (_candidate(10, 1.0),)).promote()
    public = output / "frames"
    public.chmod(0o700)

    with FrameGenerationSession(output, run_id="second") as session:
        _stage(session, (_candidate(20, 2.0),)).promote()

    assert stat.S_IMODE(public.stat().st_mode) == 0o700


def test_replacement_of_read_only_generation_cleans_backup_and_allows_rerun(
    tmp_path,
):
    output = tmp_path / "out"
    with FrameGenerationSession(output, run_id="first") as session:
        _stage(session, (_candidate(10, 1.0),)).promote()
    public = output / "frames"
    public.chmod(0o555)

    with FrameGenerationSession(output, run_id="second") as session:
        _stage(session, (_candidate(20, 2.0),)).promote()

    assert stat.S_IMODE(public.stat().st_mode) == 0o555
    assert not list(output.glob("keyframe-frame-backup-*"))

    with FrameGenerationSession(output, run_id="third") as session:
        assert session.staging is not None
        assert session.staging.root.exists()


def test_failed_read_only_replacement_cleans_staging_directory(
    tmp_path,
    monkeypatch,
):
    import keyframe.frame_generation as frame_generation

    output = tmp_path / "out"
    with FrameGenerationSession(output, run_id="previous") as session:
        _stage(session, (_candidate(5, 0.5, "previous"),)).promote()
    public = output / "frames"
    public.chmod(0o555)
    staging_root = output / "keyframe-run-replacement"

    with pytest.raises(
        FrameGenerationPromotionError,
        match="previous generation was restored",
    ):
        with FrameGenerationSession(output, run_id="replacement") as session:
            generation = _stage(session, (_candidate(10, 1.0, "new"),))
            real_replace = frame_generation.os.replace

            def fail_staged_promotion(source, target):
                if Path(source) == generation.staged_dir:
                    raise OSError("injected staged rename failure")
                return real_replace(source, target)

            monkeypatch.setattr(
                frame_generation.os,
                "replace",
                fail_staged_promotion,
            )
            generation.promote()

    assert stat.S_IMODE(public.stat().st_mode) == 0o555
    assert not staging_root.exists()
    assert not list(output.glob("keyframe-frame-backup-*"))


def test_deferred_enrichment_keeps_public_generation_unchanged_until_promote(
    tmp_path,
):
    output = tmp_path / "out"
    old = (_candidate(10, 1.0, "old"),)
    new = (_candidate(20, 2.0, "new"),)
    with FrameGenerationSession(output, run_id="old") as session:
        _stage(session, old).promote()
    previous = _tree_snapshot(output / "frames")

    with FrameGenerationSession(output, run_id="new") as session:
        generation = _stage(session, new)
        generation.enrich_manifest(
            [{"start": 1.5, "end": 2.5, "text": "current transcript"}]
        )

        assert _tree_snapshot(output / "frames") == previous
        staged_manifest = json.loads(generation.result.manifest_path.read_text())
        assert staged_manifest["frames"][0]["transcript_window"] == (
            "current transcript"
        )
        generation.promote()

    assert not (output / "frames" / _filename(old[0])).exists()
    assert (output / "frames" / _filename(new[0])).exists()
    public_manifest = json.loads((output / "frames" / "manifest.json").read_text())
    assert public_manifest["frames"][0]["transcript_window"] == (
        "current transcript"
    )


@pytest.mark.parametrize("failure_step", ["second_png", "captions", "manifest"])
def test_cli_frame_write_failure_discards_stage_and_preserves_public_generation(
    tmp_path,
    monkeypatch,
    failure_step,
):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"media")
    output = tmp_path / "out"
    with FrameGenerationSession(output, run_id="previous") as session:
        _stage(session, (_candidate(5, 0.5, "previous"),)).promote()
    previous = _tree_snapshot(output / "frames")

    def fail_generation(_video, _output, _args, session):
        assert session.staging is not None
        staged = session.staging.frames
        staged.mkdir()
        Image.new("RGB", (4, 4), "white").save(staged / "frame_000010_1.00s.png")
        if failure_step == "second_png":
            raise OSError("injected second PNG failure")
        (staged / "captions.json").write_text("[]", encoding="utf-8")
        if failure_step == "captions":
            raise OSError("injected captions failure")
        (staged / "manifest.json").write_text("{}", encoding="utf-8")
        raise OSError("injected manifest failure")

    monkeypatch.setattr(cli, "_run_frame_generation", fail_generation)

    with pytest.raises(OSError, match="injected"):
        cli.cmd_extract(_frames_only_args(video, output))

    assert _tree_snapshot(output / "frames") == previous
    assert not list(output.glob("keyframe-run-*"))


@pytest.mark.parametrize("failure_calls", [{2}, {2, 3}])
def test_replacement_failure_restores_previous_generation_even_after_retry(
    tmp_path,
    monkeypatch,
    failure_calls,
):
    import keyframe.frame_generation as frame_generation

    output = tmp_path / "out"
    with FrameGenerationSession(output, run_id="previous") as session:
        _stage(session, (_candidate(5, 0.5, "previous"),)).promote()
    previous = _tree_snapshot(output / "frames")

    with FrameGenerationSession(output, run_id="replacement") as session:
        generation = _stage(session, (_candidate(10, 1.0, "new"),))
        real_replace = frame_generation.os.replace
        calls = 0

        def flaky_replace(source, target):
            nonlocal calls
            calls += 1
            if calls in failure_calls:
                raise OSError(f"injected rename failure {calls}")
            return real_replace(source, target)

        monkeypatch.setattr(frame_generation.os, "replace", flaky_replace)

        with pytest.raises(
            FrameGenerationPromotionError,
            match="previous generation was restored",
        ):
            generation.promote()

        assert _tree_snapshot(output / "frames") == previous
        assert generation.staged_dir.exists()
        assert not generation.backup_dir.exists()


def test_unrecoverable_rollback_keeps_external_backup_for_next_locked_run(
    tmp_path,
    monkeypatch,
):
    import keyframe.frame_generation as frame_generation

    output = tmp_path / "out"
    with FrameGenerationSession(output, run_id="previous") as session:
        _stage(session, (_candidate(5, 0.5, "previous"),)).promote()
    previous = _tree_snapshot(output / "frames")
    backup = output / "keyframe-frame-backup-broken"

    with FrameGenerationSession(output, run_id="broken") as session:
        generation = _stage(session, (_candidate(10, 1.0, "new"),))
        real_replace = frame_generation.os.replace
        calls = 0

        def persistent_failure(source, target):
            nonlocal calls
            calls += 1
            if calls >= 2:
                raise OSError("persistent injected rename failure")
            return real_replace(source, target)

        with monkeypatch.context() as patcher:
            patcher.setattr(frame_generation.os, "replace", persistent_failure)
            with pytest.raises(
                FrameGenerationPromotionError,
                match="recovery backup remains",
            ):
                generation.promote()
        assert not (output / "frames").exists()
        assert backup.exists()
        assert _tree_snapshot(backup) == previous

    assert backup.exists()
    with FrameGenerationSession(output, run_id="recovery"):
        assert _tree_snapshot(output / "frames") == previous
        assert not backup.exists()


def test_locked_session_cleans_abandoned_staging_and_obsolete_backup(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    public = output / "frames"
    public.mkdir()
    (public / "sentinel.txt").write_text("current", encoding="utf-8")
    stale = output / "keyframe-run-stale"
    stale.mkdir()
    (stale / "partial.png").write_bytes(b"partial")
    obsolete_backup = output / "keyframe-frame-backup-stale"
    obsolete_backup.mkdir()
    (obsolete_backup / "old.txt").write_text("old", encoding="utf-8")
    unrelated = output / "user-data"
    unrelated.mkdir()

    with FrameGenerationSession(output, run_id="current") as session:
        assert not stale.exists()
        assert not obsolete_backup.exists()
        assert unrelated.exists()
        assert session.staging is not None
        assert session.staging.root.exists()


@pytest.mark.parametrize("artifact", ["png", "captions", "manifest"])
def test_validation_rejects_corrupt_or_mismatched_generation(tmp_path, artifact):
    staged = tmp_path / "frames"
    candidate = _candidate(10, 1.0)
    _write_result(staged, (candidate,))
    if artifact == "png":
        (staged / _filename(candidate)).write_bytes(b"not png")
    elif artifact == "captions":
        (staged / "captions.json").write_text("[]", encoding="utf-8")
    else:
        (staged / "manifest.json").write_text("not json", encoding="utf-8")

    with pytest.raises(FrameGenerationValidationError):
        validate_frame_generation(staged, (_filename(candidate),))


def test_full_cli_defers_publication_until_transcript_manifest_enrichment(
    tmp_path,
    monkeypatch,
    capsys,
):
    import keyframe.pipeline as pipeline

    video = tmp_path / "input.mp4"
    video.write_bytes(b"media")
    output = tmp_path / "out"
    public = output / "frames"
    public.mkdir(parents=True)
    (public / "old.png").write_bytes(b"old")
    (public / "captions.json").write_text("old", encoding="utf-8")
    (public / "manifest.json").write_text("old", encoding="utf-8")
    candidate = _candidate(30, 3.0, "replacement")
    calls = []

    def fake_extract(_video, staged_dir, _config, **kwargs):
        calls.append((Path(staged_dir), kwargs))
        print("Frame generation staged; awaiting validation and promotion.")
        return _write_result(staged_dir, (candidate,))

    def fake_transcript(_video, _output, _preflight, *, supervisor):
        assert supervisor.staging is not None
        assert supervisor.staging.frames.exists()
        assert (public / "old.png").exists()
        assert not (public / _filename(candidate)).exists()
        return SimpleNamespace(
            segments=[{"start": 2.5, "end": 3.5, "text": "enriched"}],
            language="en",
        )

    monkeypatch.setattr(pipeline, "extract_keyframes", fake_extract)
    monkeypatch.setattr(cli, "_preflight_transcript", lambda _args: object())
    monkeypatch.setattr(cli, "_run_transcript", fake_transcript)
    args = _frames_only_args(video, output)
    args.frames_only = False

    cli.cmd_extract(args)

    assert calls[0][0].name == "frames"
    assert calls[0][0].parent.name.startswith("keyframe-run-")
    assert calls[0][1]["report_output_dir"] == public
    assert not (public / "old.png").exists()
    assert (public / _filename(candidate)).exists()
    manifest = json.loads((public / "manifest.json").read_text())
    assert manifest["frames"][0]["transcript_window"] == "enriched"
    output_lines = capsys.readouterr().out.splitlines()
    staged_index = next(
        index
        for index, line in enumerate(output_lines)
        if "awaiting validation and promotion" in line
    )
    saved_index = next(
        index
        for index, line in enumerate(output_lines)
        if "Saved to:" in line and str(public.resolve()) in line
    )
    assert staged_index < saved_index
