from pathlib import Path

from keyframe import cli


def _video(tmp_path):
    video = tmp_path / "recording.mp4"
    video.write_bytes(b"not real media")
    return video


def test_resolve_out_dir_defaults_next_to_input(tmp_path):
    video = _video(tmp_path)

    out_dir = cli._resolve_out_dir(video, None)

    assert out_dir == tmp_path / "recording_extracted"


def test_resolve_out_dir_honors_explicit_output(tmp_path):
    video = _video(tmp_path)
    explicit = tmp_path / "somewhere" / "else"

    out_dir = cli._resolve_out_dir(video, str(explicit))

    assert out_dir == explicit


def test_resolve_out_dir_falls_back_to_tmp_when_unwritable(tmp_path, monkeypatch):
    video = _video(tmp_path)

    monkeypatch.setattr(cli.os, "access", lambda path, mode: False)

    out_dir = cli._resolve_out_dir(video, None)

    assert out_dir == Path("/tmp") / "recording_extracted"


def test_resolve_out_dir_explicit_output_ignores_writability(tmp_path, monkeypatch):
    video = _video(tmp_path)
    explicit = tmp_path / "chosen"

    # Even if the input folder is unwritable, an explicit -o is honored verbatim.
    monkeypatch.setattr(cli.os, "access", lambda path, mode: False)

    out_dir = cli._resolve_out_dir(video, str(explicit))

    assert out_dir == explicit
