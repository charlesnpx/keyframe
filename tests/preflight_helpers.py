from __future__ import annotations

from types import SimpleNamespace

import keyframe.frame_preflight as frame_preflight
import keyframe.media_preflight as media_preflight
from keyframe.frame_preflight import FrameRuntimePlatform
from keyframe.media_preflight import MediaProbeResult, MediaStream


def patch_cli_media(
    monkeypatch,
    *,
    video: bool,
    audio: bool,
) -> MediaProbeResult:
    streams = []
    if video:
        streams.append(
            MediaStream(
                codec_type="video",
                codec_name="h264",
                width=1920,
                height=1080,
            )
        )
    if audio:
        streams.append(
            MediaStream(
                codec_type="audio",
                codec_name="aac",
                channels=2,
            )
        )
    result = MediaProbeResult(tuple(streams))
    monkeypatch.setattr(media_preflight, "probe_media", lambda _path: result)
    if video:
        monkeypatch.setattr(
            frame_preflight,
            "preflight_frame_runtime",
            lambda: FrameRuntimePlatform("Darwin", "arm64"),
        )
        monkeypatch.setattr(
            frame_preflight,
            "resolve_frame_execution_device",
            lambda _runtime: "cpu",
        )
    return result


def transcript_preflight_stub(*, supports_mlx: bool = False):
    return SimpleNamespace(
        runtime_platform=SimpleNamespace(supports_mlx_whisper=supports_mlx),
        transcription_device="cpu",
        effective_diarization_device=None,
    )
