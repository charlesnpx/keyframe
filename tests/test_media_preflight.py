import json
import subprocess
from types import SimpleNamespace

import pytest

from keyframe.frame_preflight import (
    FramePreflightError,
    FrameRuntimePlatform,
    preflight_frame_runtime,
)
from keyframe.media_preflight import (
    MediaPreflightError,
    parse_ffprobe_payload,
    probe_media,
    resolve_extraction_mode,
)


def test_attached_art_video_does_not_enable_frames():
    probe = parse_ffprobe_payload(
        {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "mjpeg",
                    "width": 600,
                    "height": 600,
                    "disposition": {"attached_pic": 1},
                },
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "channels": 2,
                },
            ]
        }
    )

    assert not probe.has_usable_video
    assert probe.has_usable_audio
    mode = resolve_extraction_mode(probe, frames_only=False, transcript_only=False)
    assert not mode.do_frames
    assert mode.do_transcript

    with pytest.raises(MediaPreflightError, match="usable non-attached-picture"):
        resolve_extraction_mode(probe, frames_only=True, transcript_only=False)


def test_probe_media_wraps_ffprobe_json():
    def runner(command, **kwargs):
        assert command[:4] == ("ffprobe", "-v", "error", "-show_streams")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "streams": [
                        {
                            "codec_type": "video",
                            "codec_name": "h264",
                            "width": 1280,
                            "height": 720,
                        }
                    ]
                }
            ),
            stderr="",
        )

    probe = probe_media("recording.mp4", runner=runner)

    assert probe.has_usable_video
    assert not probe.has_usable_audio


def test_linux_x86_64_frame_preflight_requires_paddleocr():
    def importer(name):
        if name == "keyframe.frames":
            return object()
        raise ImportError(name)

    with pytest.raises(FramePreflightError, match="Linux frame dependencies"):
        preflight_frame_runtime(
            FrameRuntimePlatform("Linux", "x86_64"),
            importer=importer,
        )


def test_linux_frame_preflight_sets_up_paddle_before_importing_dependencies():
    events = []
    selection = SimpleNamespace(ocr_device="gpu:0")

    def setup(**_kwargs):
        events.append("setup")
        return selection

    def importer(name):
        events.append(name)
        if name == "paddleocr":
            return SimpleNamespace(PaddleOCR=lambda **_kwargs: object())
        return object()

    runtime = preflight_frame_runtime(
        FrameRuntimePlatform("Linux", "x86_64"),
        importer=importer,
        paddle_setup=setup,
    )

    assert events == ["setup", "keyframe.frames", "paddleocr"]
    assert runtime.paddle_runtime is selection
