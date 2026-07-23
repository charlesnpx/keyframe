"""Side-effect-free input and ffprobe preflight for CLI extraction routing."""

from __future__ import annotations

import json
import stat
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FFPROBE_TIMEOUT_SECONDS = 15.0
_UNKNOWN_CODECS = {"", "unknown", "none", "n/a"}


class MediaPreflightError(ValueError):
    """The input cannot be routed safely to a selected extraction mode."""


@dataclass(frozen=True)
class MediaStream:
    codec_type: str
    codec_name: str | None
    width: int | None = None
    height: int | None = None
    channels: int | None = None
    attached_picture: bool = False

    @property
    def is_usable_video(self) -> bool:
        return (
            self.codec_type == "video"
            and not self.attached_picture
            and self.codec_name is not None
            and self.width is not None
            and self.width > 0
            and self.height is not None
            and self.height > 0
        )

    @property
    def is_usable_audio(self) -> bool:
        return (
            self.codec_type == "audio"
            and self.codec_name is not None
            and self.channels is not None
            and self.channels > 0
        )


@dataclass(frozen=True)
class MediaProbeResult:
    streams: tuple[MediaStream, ...]

    @property
    def has_usable_video(self) -> bool:
        return any(stream.is_usable_video for stream in self.streams)

    @property
    def has_usable_audio(self) -> bool:
        return any(stream.is_usable_audio for stream in self.streams)


@dataclass(frozen=True)
class ExtractionMode:
    do_frames: bool
    do_transcript: bool
    notice: str | None = None


def resolve_readable_media_file(path: str | Path) -> Path:
    """Resolve ``path`` and prove that the target is a readable regular file."""

    candidate = Path(path).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
        mode = resolved.stat().st_mode
    except (OSError, RuntimeError) as exc:
        raise MediaPreflightError(
            f"input must resolve to a readable regular file: {candidate}: {exc}"
        ) from exc
    if not stat.S_ISREG(mode):
        raise MediaPreflightError(
            f"input must resolve to a readable regular file: {candidate}"
        )
    try:
        with resolved.open("rb"):
            pass
    except OSError as exc:
        raise MediaPreflightError(
            f"input is not readable: {resolved}: {exc}"
        ) from exc
    return resolved


def _codec_name(value: Any, *, stream_index: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise MediaPreflightError(
            f"ffprobe stream {stream_index} has a malformed codec_name"
        )
    normalized = value.strip()
    return None if normalized.lower() in _UNKNOWN_CODECS else normalized


def _integer_field(
    stream: Mapping[str, Any],
    name: str,
    *,
    stream_index: int,
) -> int | None:
    value = stream.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise MediaPreflightError(
            f"ffprobe stream {stream_index} has a malformed {name}"
        )
    return value


def _attached_picture(
    stream: Mapping[str, Any],
    *,
    stream_index: int,
) -> bool:
    disposition = stream.get("disposition")
    if disposition is None:
        return False
    if not isinstance(disposition, Mapping):
        raise MediaPreflightError(
            f"ffprobe stream {stream_index} has a malformed disposition"
        )
    value = disposition.get("attached_pic", 0)
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise MediaPreflightError(
        f"ffprobe stream {stream_index} has a malformed attached_pic disposition"
    )


def parse_ffprobe_payload(payload: Any) -> MediaProbeResult:
    if not isinstance(payload, Mapping):
        raise MediaPreflightError("ffprobe output must be a JSON object")
    raw_streams = payload.get("streams")
    if not isinstance(raw_streams, list):
        raise MediaPreflightError("ffprobe output must contain a streams list")

    streams: list[MediaStream] = []
    for index, raw_stream in enumerate(raw_streams):
        if not isinstance(raw_stream, Mapping):
            raise MediaPreflightError(
                f"ffprobe stream {index} must be a JSON object"
            )
        codec_type = raw_stream.get("codec_type")
        if not isinstance(codec_type, str) or not codec_type.strip():
            raise MediaPreflightError(
                f"ffprobe stream {index} has a malformed codec_type"
            )
        normalized_type = codec_type.strip().lower()
        streams.append(
            MediaStream(
                codec_type=normalized_type,
                codec_name=_codec_name(
                    raw_stream.get("codec_name"),
                    stream_index=index,
                ),
                width=(
                    _integer_field(raw_stream, "width", stream_index=index)
                    if normalized_type == "video"
                    else None
                ),
                height=(
                    _integer_field(raw_stream, "height", stream_index=index)
                    if normalized_type == "video"
                    else None
                ),
                channels=(
                    _integer_field(raw_stream, "channels", stream_index=index)
                    if normalized_type == "audio"
                    else None
                ),
                attached_picture=(
                    _attached_picture(raw_stream, stream_index=index)
                    if normalized_type == "video"
                    else False
                ),
            )
        )
    return MediaProbeResult(tuple(streams))


def probe_media(
    path: str | Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> MediaProbeResult:
    """Run a bounded ffprobe JSON probe and return a typed stream inventory."""

    command: Sequence[str] = (
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-of",
        "json",
        str(path),
    )
    try:
        completed = runner(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=FFPROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise MediaPreflightError(
            f"ffprobe timed out after {FFPROBE_TIMEOUT_SECONDS:g} seconds"
        ) from exc
    except FileNotFoundError as exc:
        raise MediaPreflightError(
            "ffprobe is not installed or is not available on PATH"
        ) from exc
    except OSError as exc:
        raise MediaPreflightError(f"ffprobe could not be executed: {exc}") from exc

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        suffix = f": {detail}" if detail else ""
        raise MediaPreflightError(
            f"ffprobe failed with exit status {completed.returncode}{suffix}"
        )
    try:
        payload = json.loads(completed.stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise MediaPreflightError(f"ffprobe returned malformed JSON: {exc}") from exc
    return parse_ffprobe_payload(payload)


def resolve_extraction_mode(
    probe: MediaProbeResult,
    *,
    frames_only: bool,
    transcript_only: bool,
) -> ExtractionMode:
    """Resolve explicit precedence and default stream-based routing."""

    if frames_only and transcript_only:
        raise MediaPreflightError(
            "--frames-only and --transcript-only cannot be used together"
        )
    if frames_only:
        if not probe.has_usable_video:
            raise MediaPreflightError(
                "--frames-only requires a usable non-attached-picture video stream"
            )
        return ExtractionMode(do_frames=True, do_transcript=False)
    if transcript_only:
        if not probe.has_usable_audio:
            raise MediaPreflightError(
                "--transcript-only requires a usable audio stream"
            )
        return ExtractionMode(do_frames=False, do_transcript=True)

    if probe.has_usable_video and probe.has_usable_audio:
        return ExtractionMode(do_frames=True, do_transcript=True)
    if probe.has_usable_video:
        return ExtractionMode(
            do_frames=True,
            do_transcript=False,
            notice="no usable audio stream; running frames-only extraction",
        )
    if probe.has_usable_audio:
        return ExtractionMode(
            do_frames=False,
            do_transcript=True,
            notice="no usable video stream; running transcript-only extraction",
        )
    raise MediaPreflightError(
        "input contains neither a usable video stream nor a usable audio stream"
    )
