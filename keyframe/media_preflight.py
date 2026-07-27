"""Side-effect-free input and ffprobe preflight for extraction routing."""

from __future__ import annotations

import json
import math
import stat
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

FFPROBE_TIMEOUT_SECONDS = 15.0
_UNKNOWN_CODECS = {"", "unknown", "none", "n/a"}


class MediaPreflightError(ValueError):
    """The input cannot be routed safely to a selected extraction mode."""


@dataclass(frozen=True)
class VideoTimingMetadata:
    classification: str
    reason: str
    avg_frame_rate: str | None
    r_frame_rate: str | None
    time_base: str | None
    stream_start_seconds: float | None
    duration_seconds: float | None
    duration_source: str | None
    duration_ts: int | None
    nb_frames: int | None
    nominal_frame_rate: float | None
    expected_duration_seconds: float | None
    duration_delta_seconds: float | None
    duration_tolerance_seconds: float | None

    @property
    def confirmed_cfr_rate(self) -> float | None:
        if self.classification != "cfr":
            return None
        return self.nominal_frame_rate

    def to_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification,
            "reason": self.reason,
            "avg_frame_rate": self.avg_frame_rate,
            "r_frame_rate": self.r_frame_rate,
            "time_base": self.time_base,
            "stream_start_seconds": self.stream_start_seconds,
            "duration_seconds": self.duration_seconds,
            "duration_source": self.duration_source,
            "duration_ts": self.duration_ts,
            "nb_frames": self.nb_frames,
            "nominal_frame_rate": self.nominal_frame_rate,
            "expected_duration_seconds": self.expected_duration_seconds,
            "duration_delta_seconds": self.duration_delta_seconds,
            "duration_tolerance_seconds": self.duration_tolerance_seconds,
        }


def _positive_fraction(value: Any) -> Fraction | None:
    if isinstance(value, bool) or value is None:
        return None
    if not isinstance(value, (str, int)):
        return None
    try:
        parsed = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    return parsed if parsed > 0 else None


def _fraction_text(value: Fraction | None) -> str | None:
    if value is None:
        return None
    return f"{value.numerator}/{value.denominator}"


def _finite_number(value: Any, *, positive: bool = False) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if not isinstance(value, (str, int, float)):
        return None
    try:
        parsed = float(value)
    except (OverflowError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    if positive and parsed <= 0:
        return None
    return parsed


def _integer_text(value: Any, *, positive: bool = False) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError:
            return None
    else:
        return None
    if positive and parsed <= 0:
        return None
    return parsed


def classify_video_timing(stream: Mapping[str, Any]) -> VideoTimingMetadata:
    avg_rate = _positive_fraction(stream.get("avg_frame_rate"))
    real_rate = _positive_fraction(stream.get("r_frame_rate"))
    time_base = _positive_fraction(stream.get("time_base"))
    start_time = _finite_number(stream.get("start_time"))
    stream_duration = _finite_number(stream.get("duration"), positive=True)
    duration_ts = _integer_text(stream.get("duration_ts"), positive=True)
    nb_frames = _integer_text(stream.get("nb_frames"), positive=True)

    duration_seconds = stream_duration
    duration_source = "stream_duration" if stream_duration is not None else None
    if duration_seconds is None and duration_ts is not None and time_base is not None:
        derived = float(duration_ts * time_base)
        if math.isfinite(derived) and derived > 0:
            duration_seconds = derived
            duration_source = "duration_ts"

    nominal_rate = float(avg_rate) if avg_rate is not None else None
    expected_duration = None
    duration_delta = None
    duration_tolerance = None
    if avg_rate is None or real_rate is None:
        classification = "unknown"
        reason = "missing or invalid average/real frame-rate evidence"
    elif avg_rate != real_rate:
        classification = "vfr"
        reason = "average and real frame rates differ after rational reduction"
    elif duration_seconds is None or nb_frames is None:
        classification = "unknown"
        reason = "equal rates lack duration/frame-count consistency evidence"
    else:
        expected_duration = float(Fraction(nb_frames, 1) / avg_rate)
        duration_delta = abs(duration_seconds - expected_duration)
        duration_tolerance = max(float(Fraction(1, 1) / avg_rate), 0.001 * duration_seconds)
        if duration_delta <= duration_tolerance:
            classification = "cfr"
            reason = "equal reduced rates and consistent duration/frame count"
        else:
            classification = "unknown"
            reason = "equal rates have inconsistent duration/frame-count evidence"

    return VideoTimingMetadata(
        classification=classification,
        reason=reason,
        avg_frame_rate=_fraction_text(avg_rate),
        r_frame_rate=_fraction_text(real_rate),
        time_base=_fraction_text(time_base),
        stream_start_seconds=start_time,
        duration_seconds=duration_seconds,
        duration_source=duration_source,
        duration_ts=duration_ts,
        nb_frames=nb_frames,
        nominal_frame_rate=nominal_rate,
        expected_duration_seconds=expected_duration,
        duration_delta_seconds=duration_delta,
        duration_tolerance_seconds=duration_tolerance,
    )


@dataclass(frozen=True)
class MediaStream:
    codec_type: str
    codec_name: str | None
    width: int | None = None
    height: int | None = None
    channels: int | None = None
    attached_picture: bool = False
    video_timing: VideoTimingMetadata | None = None

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

    @property
    def selected_video_stream(self) -> MediaStream | None:
        return next((stream for stream in self.streams if stream.is_usable_video), None)

    @property
    def selected_video_timing(self) -> VideoTimingMetadata | None:
        selected = self.selected_video_stream
        return selected.video_timing if selected is not None else None


@dataclass(frozen=True)
class ExtractionMode:
    do_frames: bool
    do_transcript: bool
    notice: str | None = None


def resolve_readable_media_file(path: str | Path) -> Path:
    try:
        candidate = Path(path).expanduser()
        resolved = candidate.resolve(strict=True)
        mode = resolved.stat().st_mode
    except (OSError, RuntimeError, ValueError) as exc:
        raise MediaPreflightError(
            f"input must resolve to a readable regular file: {path}: {exc}"
        ) from exc
    if not stat.S_ISREG(mode):
        raise MediaPreflightError(
            f"input must resolve to a readable regular file: {candidate}"
        )
    try:
        with resolved.open("rb"):
            pass
    except OSError as exc:
        raise MediaPreflightError(f"input is not readable: {resolved}: {exc}") from exc
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


def _integer_field(stream: Mapping[str, Any], name: str, *, stream_index: int) -> int | None:
    value = stream.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise MediaPreflightError(
            f"ffprobe stream {stream_index} has a malformed {name}"
        )
    return value


def _attached_picture(stream: Mapping[str, Any], *, stream_index: int) -> bool:
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
            raise MediaPreflightError(f"ffprobe stream {index} must be a JSON object")
        codec_type = raw_stream.get("codec_type")
        if not isinstance(codec_type, str) or not codec_type.strip():
            raise MediaPreflightError(
                f"ffprobe stream {index} has a malformed codec_type"
            )
        normalized_type = codec_type.strip().lower()
        streams.append(
            MediaStream(
                codec_type=normalized_type,
                codec_name=_codec_name(raw_stream.get("codec_name"), stream_index=index),
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
                video_timing=(
                    classify_video_timing(raw_stream)
                    if normalized_type == "video"
                    else None
                ),
            )
        )
    return MediaProbeResult(tuple(streams))


def probe_media(
    path: str | Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> MediaProbeResult:
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
        raise MediaPreflightError("ffprobe is not installed or is not available on PATH") from exc
    except UnicodeError as exc:
        raise MediaPreflightError(f"ffprobe returned undecodable output: {exc}") from exc
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
    except (json.JSONDecodeError, TypeError, UnicodeError) as exc:
        raise MediaPreflightError(f"ffprobe returned malformed JSON: {exc}") from exc
    return parse_ffprobe_payload(payload)


def resolve_extraction_mode(
    probe: MediaProbeResult,
    *,
    frames_only: bool,
    transcript_only: bool,
) -> ExtractionMode:
    if frames_only:
        if not probe.has_usable_video:
            raise MediaPreflightError(
                "--frames-only requires a usable non-attached-picture video stream"
            )
        return ExtractionMode(do_frames=True, do_transcript=False)
    if transcript_only:
        if not probe.has_usable_audio:
            raise MediaPreflightError("--transcript-only requires a usable audio stream")
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
