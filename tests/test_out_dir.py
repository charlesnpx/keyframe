import os
import shutil
from pathlib import Path

import pytest

from keyframe import cli


def _video(dir_path):
    video = dir_path / "recording.mp4"
    video.write_bytes(b"not real media")
    return video


def test_resolve_out_dir_defaults_next_to_input(tmp_path):
    video = _video(tmp_path)

    out_dir = cli._resolve_out_dir(video, None)

    assert out_dir == tmp_path / "recording_extracted"
    assert out_dir.is_dir()


def test_resolve_out_dir_honors_explicit_output(tmp_path):
    video = _video(tmp_path)
    explicit = tmp_path / "somewhere" / "else"

    out_dir = cli._resolve_out_dir(video, str(explicit))

    assert out_dir == explicit
    assert out_dir.is_dir()


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory permission semantics")
def test_resolve_out_dir_falls_back_to_tmp_when_parent_unwritable(tmp_path):
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root bypasses directory permissions")

    readonly = tmp_path / "readonly"
    readonly.mkdir()
    video = _video(readonly)
    readonly.chmod(0o500)  # r-x: cannot create the _extracted subdir here

    fallback = Path("/tmp") / "recording_extracted"
    if fallback.exists():
        shutil.rmtree(fallback)

    try:
        out_dir = cli._resolve_out_dir(video, None)
        assert out_dir == fallback
        assert out_dir.is_dir()
    finally:
        readonly.chmod(0o700)  # restore so tmp_path cleanup can remove it
        if fallback.exists():
            shutil.rmtree(fallback)
