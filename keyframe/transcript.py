#!/usr/bin/env python3
"""
extract_transcript.py

Extracts a timestamped transcript from a video file using OpenAI's Whisper model
(runs locally, no API key needed).

Usage:
    python extract_transcript.py input.mp4 [--model medium] [--output transcript.txt] [--format txt]

Dependencies:
    pip install whisperx openai-whisper

Whisper will download the model on first run (~1.4GB for 'medium').
Models in order of speed -> accuracy:
    tiny, base, small, medium, large
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import platform
import re
import sys
import time
import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable

from keyframe.artifacts import atomic_write_json, atomic_write_text, reject_path_aliases


PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"
TRANSCRIPTION_BACKENDS = ("auto", "mlx", "whisper")
DIARIZATION_DEVICES = ("auto", "cpu", "mps", "cuda")
MLX_MINIMUM_MACOS_MAJOR = 14
MLX_MINIMUM_DARWIN_MAJOR = 23
DIARIZATION_SOURCE_IGNORED_FIELDS = frozenset({"label", "segment"})
SPEAKER_DETECTION_SETUP_WARNING = """Warning: no HF_TOKEN found; falling back to transcript without speaker detection.
To enable speaker detection, accept the pyannote model terms at:
https://huggingface.co/pyannote/speaker-diarization-community-1
Then create a Hugging Face access token at:
https://huggingface.co/settings/tokens
and export it as HF_TOKEN."""


@dataclass(frozen=True)
class _PyannoteWarningFingerprint:
    message_prefix: str
    origin_suffix: str
    concise_message: str


_PYANNOTE_WARNING_FINGERPRINTS = (
    _PyannoteWarningFingerprint(
        message_prefix=(
            "torchcodec is not installed correctly so built-in audio decoding "
            "will fail. Solutions are:"
        ),
        origin_suffix="pyannote/audio/core/io.py",
        concise_message=(
            "[diarization] pyannote TorchCodec decoding is unavailable; "
            "Keyframe supplies preloaded audio instead."
        ),
    ),
    _PyannoteWarningFingerprint(
        message_prefix=(
            "std(): degrees of freedom is <= 0. Correction should be strictly "
            "less than the reduction factor"
        ),
        origin_suffix="pyannote/audio/models/blocks/pooling.py",
        concise_message=(
            "[diarization] pyannote encountered a too-short pooling window; "
            "continuing diarization."
        ),
    ),
)


def _normalized_warning_message(message: Warning | str) -> str:
    return " ".join(str(message).split())


def _warning_origin_matches(filename: str, suffix: str) -> bool:
    normalized = str(filename).replace("\\", "/")
    return normalized == suffix or normalized.endswith(f"/{suffix}")


@contextmanager
def _condense_expected_pyannote_warnings() -> Iterator[None]:
    """Condense two exact pyannote warnings and delegate every other warning."""

    emitted: set[_PyannoteWarningFingerprint] = set()
    with warnings.catch_warnings():
        active_showwarning = warnings.showwarning

        def route_warning(
            message: Warning | str,
            category: type[Warning],
            filename: str,
            lineno: int,
            file: Any = None,
            line: str | None = None,
        ) -> None:
            normalized_message = _normalized_warning_message(message)
            for fingerprint in _PYANNOTE_WARNING_FINGERPRINTS:
                if (
                    category is UserWarning
                    and normalized_message.startswith(fingerprint.message_prefix)
                    and _warning_origin_matches(
                        filename,
                        fingerprint.origin_suffix,
                    )
                ):
                    if fingerprint not in emitted:
                        emitted.add(fingerprint)
                        print(
                            fingerprint.concise_message,
                            file=sys.stderr,
                            flush=True,
                        )
                    return
            active_showwarning(
                message,
                category,
                filename,
                lineno,
                file,
                line,
            )

        warnings.showwarning = route_warning
        yield


@dataclass(frozen=True)
class MLXModelSpec:
    repository: str
    revision: str


MLX_MODEL_SPECS = {
    "tiny": MLXModelSpec(
        "mlx-community/whisper-tiny-mlx",
        "6caf9c55601caafbe6508a8b0d216bdf4783c4e8",
    ),
    "base": MLXModelSpec(
        "mlx-community/whisper-base-mlx",
        "1e3e249fb8d01c655324bd6841b1deadffd6d04c",
    ),
    "small": MLXModelSpec(
        "mlx-community/whisper-small-mlx",
        "45f3915923c7a79a5a5b5a7d909d39aeb0e5630e",
    ),
    "medium": MLXModelSpec(
        "mlx-community/whisper-medium-mlx",
        "7fc08c4eac4c316526498f147dfdee6f6303f975",
    ),
    "large": MLXModelSpec(
        "mlx-community/whisper-large-mlx",
        "9310354911111f2406ead1478e0139d9c6ea3acc",
    ),
}


@dataclass(frozen=True)
class RuntimePlatform:
    system: str
    machine: str
    macos_major: int | None = None
    darwin_major: int | None = None

    @property
    def supports_mlx_whisper(self) -> bool:
        if self.system != "Darwin" or self.machine.lower() != "arm64":
            return False
        if self.macos_major is not None:
            return self.macos_major >= MLX_MINIMUM_MACOS_MAJOR
        return (
            self.darwin_major is not None
            and self.darwin_major >= MLX_MINIMUM_DARWIN_MAJOR
        )


class TranscriptionError(RuntimeError):
    """Base class for stage errors that callers may classify without string parsing."""


class UnsupportedTranscriptionBackendError(TranscriptionError):
    """The requested backend cannot run on the current platform."""


class UnsupportedDiarizationDeviceError(TranscriptionError):
    """The requested diarization device cannot run in the current environment."""


class MPSDiarizationError(TranscriptionError):
    """An MPS compute failure for which automatic diarization may retry on CPU."""


class MPSDiarizationModelLoadError(MPSDiarizationError):
    """The pyannote model could not be initialized on MPS."""


class MPSDiarizationInferenceError(MPSDiarizationError):
    """The initialized pyannote model failed during MPS inference."""


class TranscriptionCancelled(TranscriptionError):
    """The parent explicitly cancelled transcription."""


class TranscriptOutputError(TranscriptionError):
    """Writing an output failed after inference completed."""


class MLXBackendError(TranscriptionError):
    """An MLX failure for which auto mode may start a fresh Whisper process."""


class MLXImportError(MLXBackendError):
    """The pinned MLX runtime could not be imported."""


class MLXModelAcquisitionError(MLXBackendError):
    """The pinned model snapshot could not be acquired."""


class MLXModelLoadError(MLXBackendError):
    """The acquired model could not be loaded into MLX."""


class MLXInferenceError(MLXBackendError):
    """MLX inference failed or returned a malformed result."""


@dataclass(frozen=True)
class MLXRuntime:
    snapshot_download: Callable[..., str]
    load_model: Callable[[str, Any], Any]
    transcribe: Callable[..., Mapping[str, Any]]
    float16: Any
    local_entry_not_found_error: type[Exception]
    reset_peak_memory: Callable[[], None]
    get_peak_memory: Callable[[], int]


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str
    speaker: str | None = None

    def __post_init__(self) -> None:
        start = float(self.start)
        end = max(float(self.end), start)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "text", str(self.text).strip())
        object.__setattr__(self, "speaker", _normalize_speaker_label(self.speaker))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }
        if self.speaker:
            payload["speaker"] = self.speaker
        return payload

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        if not hasattr(self, key):
            raise KeyError(key)
        return getattr(self, key)


@dataclass(frozen=True)
class TranscriptionResult:
    """Internal inference result with reliable, extensible execution evidence."""

    segments: tuple[TranscriptSegment, ...]
    language: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "segments", tuple(self.segments))
        object.__setattr__(self, "language", str(self.language))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class DiarizationRow:
    start: float
    end: float
    speaker: str


class CheckpointValidationError(ValueError):
    """A transcript-stage checkpoint violates its strict public schema."""


RAW_TRANSCRIPT_FIELDS = frozenset({"start", "end", "text"})
DIARIZATION_FIELDS = frozenset({"start", "end", "speaker"})


def format_time(seconds):
    """Format seconds into HH:MM:SS.sss"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def format_srt_time(seconds):
    """SRT uses comma as decimal separator."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def transcript_segments(segments: Iterable[Mapping[str, Any] | TranscriptSegment]) -> tuple[TranscriptSegment, ...]:
    return tuple(_as_transcript_segment(seg) for seg in segments)


def _as_transcript_segment(seg: Mapping[str, Any] | TranscriptSegment) -> TranscriptSegment:
    if isinstance(seg, TranscriptSegment):
        return seg
    return TranscriptSegment(
        start=_coerce_seconds(seg.get("start"), 0.0),
        end=_coerce_seconds(seg.get("end"), _coerce_seconds(seg.get("start"), 0.0)),
        text=str(seg.get("text", "")),
        speaker=seg.get("speaker"),
    )


def write_txt(segments, out_path):
    """Plain text with timestamps."""
    payload = io.StringIO()
    for seg in transcript_segments(segments):
        start = format_time(seg.start)
        end = format_time(seg.end)
        text = seg.text
        speaker = seg.speaker
        prefix = f"{speaker}  " if speaker else ""
        payload.write(f"[{start} --> {end}]  {prefix}{text}\n")
    atomic_write_text(out_path, payload.getvalue())
    print(f"Saved: {out_path}")


def write_srt(segments, out_path):
    """SubRip subtitle format."""
    payload = io.StringIO()
    for i, seg in enumerate(transcript_segments(segments), 1):
        start = format_srt_time(seg.start)
        end = format_srt_time(seg.end)
        text = _caption_text(seg)
        payload.write(f"{i}\n{start} --> {end}\n{text}\n\n")
    atomic_write_text(out_path, payload.getvalue())
    print(f"Saved: {out_path}")


def write_vtt(segments, out_path):
    """WebVTT subtitle format."""
    payload = io.StringIO()
    payload.write("WEBVTT\n\n")
    for seg in transcript_segments(segments):
        start = format_time(seg.start)
        end = format_time(seg.end)
        text = _caption_text(seg)
        payload.write(f"{start} --> {end}\n{text}\n\n")
    atomic_write_text(out_path, payload.getvalue())
    print(f"Saved: {out_path}")


def write_json(segments, out_path):
    """JSON with full segment data."""
    data = [
        {
            **seg.to_dict(),
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
        }
        for seg in transcript_segments(segments)
    ]
    atomic_write_json(out_path, data, allow_nan=True)
    print(f"Saved: {out_path}")


WRITERS = {
    "txt": write_txt,
    "srt": write_srt,
    "vtt": write_vtt,
    "json": write_json,
}


def _caption_text(seg: Mapping[str, Any] | TranscriptSegment) -> str:
    seg = _as_transcript_segment(seg)
    text = seg.text
    speaker = seg.speaker
    return f"{speaker}: {text}" if speaker else text


def _coerce_seconds(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


NULL_SPEAKER_LABELS = {"<na>", "nat", "none", "null", "nan"}


def _normalize_speaker_label(label: Any) -> str | None:
    if label is None:
        return None
    normalized = str(label).strip()
    if not normalized or normalized.lower() in NULL_SPEAKER_LABELS:
        return None
    return normalized


def _finite_seconds(value: Any) -> float | None:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(seconds):
        return None
    return seconds


def _strict_checkpoint_seconds(
    value: Any,
    *,
    row_index: int,
    field: str,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CheckpointValidationError(
            f"checkpoint row {row_index} field {field!r} must be a number"
        )
    try:
        seconds = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CheckpointValidationError(
            f"checkpoint row {row_index} field {field!r} must be a finite number"
        ) from exc
    if not math.isfinite(seconds):
        raise CheckpointValidationError(
            f"checkpoint row {row_index} field {field!r} must be finite"
        )
    if seconds < 0:
        raise CheckpointValidationError(
            f"checkpoint row {row_index} field {field!r} must be non-negative"
        )
    return seconds


def _raw_transcript_checkpoint_row(
    row: Mapping[str, Any] | TranscriptSegment,
    row_index: int,
) -> dict[str, Any]:
    if isinstance(row, TranscriptSegment):
        if row.speaker is not None:
            raise CheckpointValidationError(
                f"raw transcript row {row_index} must not contain a speaker"
            )
        values = {"start": row.start, "end": row.end, "text": row.text}
    elif isinstance(row, Mapping):
        if set(row) != RAW_TRANSCRIPT_FIELDS:
            raise CheckpointValidationError(
                f"raw transcript row {row_index} must contain exactly "
                f"{sorted(RAW_TRANSCRIPT_FIELDS)}"
            )
        values = row
    else:
        raise CheckpointValidationError(
            f"raw transcript row {row_index} must be an object"
        )

    start = _strict_checkpoint_seconds(values.get("start"), row_index=row_index, field="start")
    end = _strict_checkpoint_seconds(values.get("end"), row_index=row_index, field="end")
    if end < start:
        raise CheckpointValidationError(
            f"raw transcript row {row_index} ends before it starts"
        )
    text = values.get("text")
    if not isinstance(text, str):
        raise CheckpointValidationError(
            f"raw transcript row {row_index} field 'text' must be a string"
        )
    return {"start": start, "end": end, "text": text.strip()}


def _diarization_checkpoint_row(
    row: Mapping[str, Any] | DiarizationRow,
    row_index: int,
) -> dict[str, Any]:
    if isinstance(row, DiarizationRow):
        values = {"start": row.start, "end": row.end, "speaker": row.speaker}
    elif isinstance(row, Mapping):
        if set(row) != DIARIZATION_FIELDS:
            raise CheckpointValidationError(
                f"diarization row {row_index} must contain exactly "
                f"{sorted(DIARIZATION_FIELDS)}"
            )
        values = row
    else:
        raise CheckpointValidationError(
            f"diarization row {row_index} must be an object"
        )

    start = _strict_checkpoint_seconds(values.get("start"), row_index=row_index, field="start")
    end = _strict_checkpoint_seconds(values.get("end"), row_index=row_index, field="end")
    if end <= start:
        raise CheckpointValidationError(
            f"diarization row {row_index} must have positive duration"
        )
    speaker_value = values.get("speaker")
    if not isinstance(speaker_value, str):
        raise CheckpointValidationError(
            f"diarization row {row_index} field 'speaker' must be a string"
        )
    speaker = _normalize_speaker_label(speaker_value)
    if speaker is None:
        raise CheckpointValidationError(
            f"diarization row {row_index} has an empty speaker label"
        )
    return {"start": start, "end": end, "speaker": speaker}


def _checkpoint_rows(value: Any, *, checkpoint_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise CheckpointValidationError(f"{checkpoint_name} checkpoint must be a JSON array")
    return value


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value = {}
    for key, item in pairs:
        if key in value:
            raise CheckpointValidationError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise CheckpointValidationError(f"non-finite JSON number: {value}")


def _read_checkpoint_json(path: str | Path) -> Any:
    try:
        return json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CheckpointValidationError(f"invalid checkpoint JSON: {exc}") from exc


def write_raw_transcript_checkpoint(
    segments: Iterable[Mapping[str, Any] | TranscriptSegment],
    path: str | Path,
    *,
    final_output_paths: Iterable[str | Path] = (),
) -> Path:
    reject_path_aliases(path, final_output_paths)
    payload = [
        _raw_transcript_checkpoint_row(segment, index)
        for index, segment in enumerate(segments)
    ]
    return atomic_write_json(path, payload, allow_nan=False)


def read_raw_transcript_checkpoint(
    path: str | Path,
    *,
    final_output_paths: Iterable[str | Path] = (),
) -> tuple[TranscriptSegment, ...]:
    reject_path_aliases(path, final_output_paths)
    rows = _checkpoint_rows(
        _read_checkpoint_json(path),
        checkpoint_name="raw transcript",
    )
    normalized = [
        _raw_transcript_checkpoint_row(row, index)
        for index, row in enumerate(rows)
    ]
    return tuple(TranscriptSegment(**row) for row in normalized)


def write_diarization_checkpoint(
    rows: Iterable[Mapping[str, Any] | DiarizationRow],
    path: str | Path,
    *,
    final_output_paths: Iterable[str | Path] = (),
) -> Path:
    reject_path_aliases(path, final_output_paths)
    payload = [
        _diarization_checkpoint_row(row, index)
        for index, row in enumerate(rows)
    ]
    return atomic_write_json(path, payload, allow_nan=False)


def read_diarization_checkpoint(
    path: str | Path,
    *,
    final_output_paths: Iterable[str | Path] = (),
) -> tuple[DiarizationRow, ...]:
    reject_path_aliases(path, final_output_paths)
    rows = _checkpoint_rows(
        _read_checkpoint_json(path),
        checkpoint_name="diarization",
    )
    normalized = [
        _diarization_checkpoint_row(row, index)
        for index, row in enumerate(rows)
    ]
    return tuple(DiarizationRow(**row) for row in normalized)


def _diarization_records(diarization: Any) -> tuple[Mapping[str, Any], ...]:
    if diarization is None:
        return ()
    if (
        isinstance(diarization, tuple)
        and len(diarization) == 2
        and not isinstance(diarization[0], Mapping)
        and (
            hasattr(diarization[0], "to_dict")
            or isinstance(diarization[0], Sequence)
        )
        and (diarization[1] is None or isinstance(diarization[1], Mapping))
    ):
        diarization = diarization[0]
    if hasattr(diarization, "to_dict"):
        try:
            records = diarization.to_dict("records")
        except TypeError:
            records = ()
        return tuple(record for record in records if isinstance(record, Mapping))
    if isinstance(diarization, Mapping):
        return (diarization,)
    if isinstance(diarization, Sequence) and not isinstance(diarization, (str, bytes)):
        return tuple(record for record in diarization if isinstance(record, Mapping))
    return ()


def _valid_diarization_rows(diarization: Any) -> tuple[DiarizationRow, ...]:
    rows = []
    if isinstance(diarization, DiarizationRow):
        rows.append(diarization)
    elif isinstance(diarization, Sequence) and not isinstance(diarization, (str, bytes)):
        rows.extend(row for row in diarization if isinstance(row, DiarizationRow))

    for record in _diarization_records(diarization):
        speaker = _normalize_speaker_label(record.get("speaker"))
        start = _finite_seconds(record.get("start"))
        end = _finite_seconds(record.get("end"))
        if speaker is None or start is None or end is None or end <= start:
            continue
        rows.append(DiarizationRow(start=start, end=end, speaker=speaker))
    normalized_rows = []
    for row in rows:
        speaker = _normalize_speaker_label(row.speaker)
        start = _finite_seconds(row.start)
        end = _finite_seconds(row.end)
        if speaker is None or start is None or end is None or end <= start:
            continue
        normalized_rows.append(DiarizationRow(start=start, end=end, speaker=speaker))
    return tuple(normalized_rows)


def _strict_diarization_rows(diarization: Any) -> tuple[DiarizationRow, ...]:
    """Convert every pyannote row or reject the complete diarization result."""

    if (
        isinstance(diarization, tuple)
        and len(diarization) == 2
        and not isinstance(diarization[0], Mapping)
        and (
            hasattr(diarization[0], "to_dict")
            or isinstance(diarization[0], Sequence)
        )
        and (diarization[1] is None or isinstance(diarization[1], Mapping))
    ):
        diarization = diarization[0]

    if diarization is None:
        records: Sequence[Any] = ()
    elif isinstance(diarization, DiarizationRow):
        records = (diarization,)
    elif hasattr(diarization, "to_dict"):
        try:
            converted = diarization.to_dict("records")
        except Exception as exc:
            raise CheckpointValidationError(
                f"could not convert diarization rows: {exc}"
            ) from exc
        if not isinstance(converted, Sequence) or isinstance(
            converted,
            (str, bytes),
        ):
            raise CheckpointValidationError(
                "diarization conversion must return a row sequence"
            )
        records = converted
    elif isinstance(diarization, Mapping):
        records = (diarization,)
    elif isinstance(diarization, Sequence) and not isinstance(
        diarization,
        (str, bytes),
    ):
        records = diarization
    else:
        raise CheckpointValidationError(
            "diarization result must contain a row sequence"
        )

    normalized_rows = []
    accepted_fields = DIARIZATION_FIELDS | DIARIZATION_SOURCE_IGNORED_FIELDS
    for index, record in enumerate(records):
        if isinstance(record, DiarizationRow):
            normalized = _diarization_checkpoint_row(record, index)
        elif isinstance(record, Mapping):
            fields = set(record)
            missing = DIARIZATION_FIELDS - fields
            unexpected = fields - accepted_fields
            if missing:
                raise CheckpointValidationError(
                    f"diarization row {index} is missing fields: "
                    + ", ".join(sorted(str(field) for field in missing))
                )
            if unexpected:
                raise CheckpointValidationError(
                    f"diarization row {index} has unknown fields: "
                    + ", ".join(sorted(str(field) for field in unexpected))
                )
            normalized = _diarization_checkpoint_row(
                {field: record[field] for field in DIARIZATION_FIELDS},
                index,
            )
        else:
            raise CheckpointValidationError(
                f"diarization row {index} must be an object"
            )
        normalized_rows.append(DiarizationRow(**normalized))
    return tuple(normalized_rows)


def _assign_speakers(
    segments: Iterable[Mapping[str, Any] | TranscriptSegment],
    diarization: Any,
) -> tuple[TranscriptSegment, ...]:
    source_segments = transcript_segments(segments)
    diarization_rows = _valid_diarization_rows(diarization)
    labeled_segments = []

    for segment in source_segments:
        overlaps: dict[str, tuple[float, float]] = {}
        for row in diarization_rows:
            overlap_start = max(segment.start, row.start)
            overlap_end = min(segment.end, row.end)
            overlap = overlap_end - overlap_start
            if overlap <= 0:
                continue
            total, earliest_start = overlaps.get(row.speaker, (0.0, overlap_start))
            overlaps[row.speaker] = (total + overlap, min(earliest_start, overlap_start))

        speaker = None
        if overlaps:
            speaker = min(
                overlaps.items(),
                key=lambda item: (-item[1][0], item[1][1], item[0]),
            )[0]
        labeled_segments.append(
            TranscriptSegment(segment.start, segment.end, segment.text, speaker)
        )

    return tuple(labeled_segments)


def cuda_is_available() -> bool:
    try:
        import torch
    except (ImportError, OSError):
        return False
    try:
        return bool(torch.cuda.is_available())
    except (AttributeError, RuntimeError):
        return False


def mps_is_available() -> bool:
    """Return whether this Torch build can execute on the local MPS backend."""

    try:
        import torch
    except (ImportError, OSError):
        return False
    try:
        backend = torch.backends.mps
        return bool(backend.is_built() and backend.is_available())
    except (AttributeError, RuntimeError):
        return False


def resolve_diarization_device(
    requested: str,
    *,
    cuda_available: bool | None = None,
    mps_available: bool | None = None,
    runtime_platform: RuntimePlatform | None = None,
) -> str:
    if requested not in DIARIZATION_DEVICES:
        choices = ", ".join(DIARIZATION_DEVICES)
        raise ValueError(f"unknown diarization device {requested!r}; choose from: {choices}")
    if requested == "cpu":
        return "cpu"

    runtime_platform = runtime_platform or current_runtime_platform()
    supports_mps = (
        runtime_platform.system == "Darwin"
        and runtime_platform.machine.lower() == "arm64"
    )

    def has_mps() -> bool:
        if not supports_mps:
            return False
        return (
            mps_is_available()
            if mps_available is None
            else bool(mps_available)
        )

    def has_cuda() -> bool:
        return (
            cuda_is_available()
            if cuda_available is None
            else bool(cuda_available)
        )

    if requested == "mps":
        if not has_mps():
            raise UnsupportedDiarizationDeviceError(
                "MPS diarization was requested, but Torch reports no available "
                "MPS device on Darwin ARM64"
            )
        return "mps"
    if requested == "cuda" and not has_cuda():
        raise UnsupportedDiarizationDeviceError(
            "CUDA diarization was requested, but Torch reports no available CUDA device"
        )
    if requested == "cuda":
        return "cuda"
    if supports_mps:
        return "mps" if has_mps() else "cpu"
    return "cuda" if has_cuda() else "cpu"


def _select_whisperx_device(
    requested: str = "auto",
    *,
    cuda_available: bool | None = None,
    mps_available: bool | None = None,
    runtime_platform: RuntimePlatform | None = None,
) -> tuple[str, str]:
    device = resolve_diarization_device(
        requested,
        cuda_available=cuda_available,
        mps_available=mps_available,
        runtime_platform=runtime_platform,
    )
    compute_type = (
        "float16"
        if device == "cuda"
        else "float32"
        if device == "mps"
        else "int8"
    )
    return (device, compute_type)


def is_auto_diarization_fallback_eligible(exc: BaseException) -> bool:
    """Return whether an automatic MPS attempt may be retried once on CPU."""

    return isinstance(exc, MPSDiarizationError)


_MPS_NON_COMPUTE_FAILURE_MARKERS = (
    "access token",
    "acquisition",
    "authenticat",
    "authoriz",
    "checkpoint",
    "codec",
    "credential",
    "data shape",
    "decod",
    "download",
    "ffmpeg",
    "fetch",
    "forbidden",
    "gated",
    "hf token",
    "hf_token",
    "hugging face",
    "invalid shape",
    "malformed shape",
    "network",
    "protocol",
    "repository",
    "revision",
    "shape mismatch",
    "size mismatch",
    "tensor size",
    "tensor shape",
    "timestamp",
    "unauthorized",
)
_MPS_COMPUTE_FAILURE_PATTERNS = (
    re.compile(
        r"\bmps(?:graph)?\s+(?:allocator|backend|command buffer|device|"
        r"execution|graph|kernel|operation|operator)\b"
    ),
    re.compile(
        r"\b(?:allocator|backend|command buffer|device|execution|graph|kernel|"
        r"operation|operator)\b[^\n]*\bmps(?:graph)?\b"
    ),
    re.compile(
        r"\bmetal\s+(?:allocator|backend|command buffer|device|kernel|library|"
        r"performance shaders|pipeline|resource)\b"
    ),
    re.compile(r"\bmps\b[^\n]*\b(?:oom|out of memory)\b"),
    re.compile(
        r"\bmps\b[^\n]*\b(?:does not support|not implemented|not supported|"
        r"unsupported)\b"
    ),
    re.compile(r"\bmpsndarray\b"),
)


def _is_mps_compute_failure(exc: BaseException) -> bool:
    """Exclude authentication, acquisition, decoding, and data-shape failures."""

    if not isinstance(exc, (RuntimeError, NotImplementedError, MemoryError)):
        return False
    message = " ".join(str(exc).lower().split())
    if any(marker in message for marker in _MPS_NON_COMPUTE_FAILURE_MARKERS):
        return False
    if isinstance(exc, MemoryError):
        return True
    return (
        any(pattern.search(message) for pattern in _MPS_COMPUTE_FAILURE_PATTERNS)
        or "placeholder storage" in message
        or "invalid buffer size" in message
    )


def _print_missing_hf_token_warning() -> None:
    print(SPEAKER_DETECTION_SETUP_WARNING, file=sys.stderr)


def _print_speaker_detection_failure(exc: Exception) -> None:
    print(
        "Warning: speaker detection failed; falling back to transcript without speaker detection.",
        file=sys.stderr,
    )
    print(f"Reason: {exc}", file=sys.stderr)
    print(
        "To enable speaker detection, accept the pyannote model terms at:\n"
        f"https://huggingface.co/{PYANNOTE_MODEL}\n"
        "Then create a Hugging Face access token at:\n"
        "https://huggingface.co/settings/tokens\n"
        "and export it as HF_TOKEN.",
        file=sys.stderr,
    )


def _major_version(version: str) -> int | None:
    try:
        return int(version.split(".", 1)[0])
    except (AttributeError, TypeError, ValueError):
        return None


def current_runtime_platform() -> RuntimePlatform:
    macos_version = platform.mac_ver()[0]
    return RuntimePlatform(
        system=platform.system(),
        machine=platform.machine(),
        macos_major=_major_version(macos_version) if macos_version else None,
        darwin_major=_major_version(platform.release()),
    )


def resolve_transcription_backend(
    requested: str,
    runtime_platform: RuntimePlatform | None = None,
) -> str:
    if requested not in TRANSCRIPTION_BACKENDS:
        choices = ", ".join(TRANSCRIPTION_BACKENDS)
        raise ValueError(f"unknown transcription backend {requested!r}; choose from: {choices}")

    runtime_platform = runtime_platform or current_runtime_platform()
    if requested == "auto":
        return "mlx" if runtime_platform.supports_mlx_whisper else "whisper"
    if requested == "mlx" and not runtime_platform.supports_mlx_whisper:
        raise UnsupportedTranscriptionBackendError(
            "MLX transcription requires Apple Silicon running macOS 14 or newer "
            f"(detected system={runtime_platform.system!r}, "
            f"machine={runtime_platform.machine!r}, "
            f"macos_major={runtime_platform.macos_major!r}, "
            f"darwin_major={runtime_platform.darwin_major!r})"
        )
    return requested


def is_auto_fallback_eligible(exc: BaseException) -> bool:
    return isinstance(exc, MLXBackendError)


def _load_mlx_runtime() -> MLXRuntime:
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
        import mlx.core as mx
        import mlx_whisper
        from mlx_whisper.transcribe import ModelHolder
        reset_peak_memory = mx.reset_peak_memory
        get_peak_memory = mx.get_peak_memory
    except (AttributeError, ImportError, OSError) as exc:
        raise MLXImportError(
            "the pinned MLX runtime could not be imported; reinstall keyframe on a "
            "supported Apple Silicon Mac"
        ) from exc

    return MLXRuntime(
        snapshot_download=snapshot_download,
        load_model=ModelHolder.get_model,
        transcribe=mlx_whisper.transcribe,
        float16=mx.float16,
        local_entry_not_found_error=LocalEntryNotFoundError,
        reset_peak_memory=reset_peak_memory,
        get_peak_memory=get_peak_memory,
    )


def _resolve_mlx_model_snapshot(
    runtime: MLXRuntime,
    model_spec: MLXModelSpec,
) -> tuple[Path, Mapping[str, Any]]:
    """Resolve an immutable MLX snapshot locally before permitting network I/O."""

    started = time.monotonic()
    source = "local-hit"
    print("Resolving cached MLX model...")
    try:
        try:
            resolved = runtime.snapshot_download(
                repo_id=model_spec.repository,
                revision=model_spec.revision,
                local_files_only=True,
            )
        except runtime.local_entry_not_found_error:
            source = "downloaded"
            print("Cached MLX model not found; downloading pinned snapshot...")
            resolved = runtime.snapshot_download(
                repo_id=model_spec.repository,
                revision=model_spec.revision,
            )

        if not isinstance(resolved, (str, os.PathLike)):
            raise TypeError("snapshot resolver returned a non-path result")
        model_path = Path(resolved)
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"resolved snapshot is not an existing directory: {model_path}"
            )
    except TranscriptionCancelled:
        raise
    except Exception as exc:
        raise MLXModelAcquisitionError(
            f"failed to acquire {model_spec.repository}@{model_spec.revision}"
        ) from exc

    elapsed = time.monotonic() - started
    if not math.isfinite(elapsed) or elapsed < 0:
        elapsed = 0.0
    return model_path, MappingProxyType(
        {
            "model_repository": model_spec.repository,
            "model_revision": model_spec.revision,
            "model_resolution_source": source,
            "model_resolution_seconds": elapsed,
        }
    )


def _normalize_mlx_result(
    result: Mapping[str, Any],
) -> tuple[tuple[TranscriptSegment, ...], str]:
    if not isinstance(result, Mapping):
        raise TypeError("MLX transcription result is not a mapping")
    raw_segments = result.get("segments")
    if not isinstance(raw_segments, Sequence) or isinstance(raw_segments, (str, bytes)):
        raise TypeError("MLX transcription result has no segment sequence")

    normalized = []
    for index, raw_segment in enumerate(raw_segments):
        if not isinstance(raw_segment, Mapping):
            raise TypeError(f"MLX segment {index} is not a mapping")
        start = _finite_seconds(raw_segment.get("start"))
        end = _finite_seconds(raw_segment.get("end"))
        if start is None or end is None or start < 0 or end < start:
            raise ValueError(f"MLX segment {index} has invalid timestamps")
        normalized.append(
            TranscriptSegment(
                start=start,
                end=end,
                text=str(raw_segment.get("text", "")),
            )
        )

    language = result.get("language")
    return tuple(normalized), str(language) if language else "unknown"


def _extract_with_mlx(
    video: Path,
    model_name: str,
    runtime_platform: RuntimePlatform | None = None,
) -> TranscriptionResult:
    runtime_platform = runtime_platform or current_runtime_platform()
    resolve_transcription_backend("mlx", runtime_platform)
    try:
        model_spec = MLX_MODEL_SPECS[model_name]
    except KeyError as exc:
        raise ValueError(f"unsupported MLX Whisper model size: {model_name!r}") from exc

    print("Importing MLX runtime...")
    runtime = _load_mlx_runtime()
    model_path, execution_metadata = _resolve_mlx_model_snapshot(runtime, model_spec)

    print("Loading MLX model...")
    try:
        runtime.reset_peak_memory()
        runtime.load_model(str(model_path), runtime.float16)
    except TranscriptionCancelled:
        raise
    except Exception as exc:
        raise MLXModelLoadError(
            f"failed to load {model_spec.repository}@{model_spec.revision}"
        ) from exc

    print("Transcribing with MLX (this may take a while on long videos)...")
    try:
        result = runtime.transcribe(
            str(video),
            path_or_hf_repo=str(model_path),
            verbose=False,
            word_timestamps=False,
        )
        segments, language = _normalize_mlx_result(result)
        peak_memory_bytes = runtime.get_peak_memory()
        if isinstance(peak_memory_bytes, bool) or not isinstance(
            peak_memory_bytes,
            int,
        ):
            raise TypeError("MLX peak memory counter did not return an integer")
        if peak_memory_bytes < 0:
            raise ValueError("MLX peak memory counter returned a negative value")
        metadata = dict(execution_metadata)
        metadata["mlx_peak_memory_bytes"] = peak_memory_bytes
        return TranscriptionResult(segments, language, metadata)
    except TranscriptionCancelled:
        raise
    except Exception as exc:
        raise MLXInferenceError(
            f"MLX inference failed for {model_spec.repository}@{model_spec.revision}"
        ) from exc


def _extract_with_whisper(video: Path, model_name: str) -> TranscriptionResult:
    try:
        import whisper
    except ImportError:
        print("Error: whisper not installed. Run:\n"
              "  pip install openai-whisper\n"
              "(Requires ffmpeg installed on your system too.)",
              file=sys.stderr)
        sys.exit(1)

    print("Loading model...")
    model = whisper.load_model(model_name)

    print("Transcribing (this may take a while on long videos)...")
    result = model.transcribe(
        str(video),
        verbose=False,
        word_timestamps=False,
    )

    return TranscriptionResult(
        transcript_segments(result["segments"]),
        result.get("language", "unknown"),
        {},
    )


def _extract_with_transcription_backend(
    video: Path,
    model_name: str,
    requested_backend: str,
    runtime_platform: RuntimePlatform | None = None,
) -> TranscriptionResult:
    runtime_platform = runtime_platform or current_runtime_platform()
    effective_backend = resolve_transcription_backend(requested_backend, runtime_platform)
    print(
        "Transcription backend: "
        f"requested={requested_backend}, effective={effective_backend}"
    )
    if effective_backend == "mlx":
        model_spec = MLX_MODEL_SPECS[model_name]
        print(
            "MLX model: "
            f"{model_spec.repository}@{model_spec.revision}"
        )
        return _extract_with_mlx(video, model_name, runtime_platform)
    return _extract_with_whisper(video, model_name)


def _detect_speakers(
    video: Path,
    hf_token: str,
    *,
    device: str = "cpu",
) -> tuple[DiarizationRow, ...]:
    """Detect speakers; supervised CLI workers pass their resolved device."""

    with _condense_expected_pyannote_warnings():
        import whisperx
        from whisperx.diarize import DiarizationPipeline
        from tqdm import tqdm

        if device == "auto":
            device, _compute_type = _select_whisperx_device()
        else:
            device, _compute_type = _select_whisperx_device(device)
        audio = whisperx.load_audio(str(video))

        print("Detecting speakers with pyannote...")
        try:
            diarize_model = DiarizationPipeline(
                PYANNOTE_MODEL,
                token=hf_token,
                device=device,
            )
        except Exception as exc:
            if device == "mps" and _is_mps_compute_failure(exc):
                raise MPSDiarizationModelLoadError(
                    "pyannote model initialization failed on MPS"
                ) from exc
            raise
        progress = tqdm(
            total=100,
            desc="Detecting speakers",
            unit="%",
            leave=False,
        )
        last_progress = 0.0

        def update_progress(percent: float) -> None:
            nonlocal last_progress
            try:
                bounded = min(100.0, max(last_progress, float(percent)))
            except (TypeError, ValueError):
                return
            if bounded > last_progress:
                progress.update(bounded - last_progress)
            last_progress = bounded

        try:
            try:
                detected = diarize_model(
                    audio,
                    progress_callback=update_progress,
                )
            except Exception as exc:
                if device == "mps" and _is_mps_compute_failure(exc):
                    raise MPSDiarizationInferenceError(
                        "pyannote inference failed on MPS"
                    ) from exc
                raise
            return _strict_diarization_rows(detected)
        finally:
            progress.close()


def extract_transcript(
    video_path,
    model_name="medium",
    output=None,
    fmt="txt",
    speaker_detection=True,
    transcription_backend="whisper",
):
    """
    Run Whisper on a video file and save the timestamped transcript.

    Args:
        video_path:  Path to input video
        model_name:  Whisper model size (tiny/base/small/medium/large)
        output:      Output file path (auto-generated if None)
        fmt:         Output format: txt, srt, vtt, json
        speaker_detection:
                     Attempt pyannote speaker detection when HF_TOKEN is set
        transcription_backend:
                     Transcription backend: auto, mlx, or whisper

    Returns:
        (segments, language) tuple
    """
    video = Path(video_path)
    if not video.exists():
        print(f"Error: file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    if fmt not in WRITERS:
        print(f"Error: unknown format '{fmt}'. Choose from: {', '.join(WRITERS)}",
              file=sys.stderr)
        sys.exit(1)

    if output is None:
        output = video.with_suffix(f".{fmt}")

    print(f"Video: {video_path}")
    print(f"Model: {model_name}")

    result = _extract_with_transcription_backend(
        video,
        model_name,
        transcription_backend,
    )
    segments, language = result.segments, result.language

    if speaker_detection and segments:
        hf_token = (os.environ.get("HF_TOKEN") or "").strip()
        if not hf_token:
            _print_missing_hf_token_warning()
        else:
            try:
                diarization_rows = _detect_speakers(video, hf_token)
                speaker_segments = _assign_speakers(segments, diarization_rows)
                if not any(seg.speaker for seg in speaker_segments):
                    raise RuntimeError("speaker diarization produced no usable speaker overlaps")
                segments = speaker_segments
            except Exception as exc:
                _print_speaker_detection_failure(exc)

    print(f"Detected language: {language}")
    print(f"Segments: {len(segments)}")

    if segments:
        duration = segments[-1].end
        print(f"Duration covered: {format_time(duration)}")

    WRITERS[fmt](segments, output)

    print(f"\n--- Preview (first 10 segments) ---")
    for seg in segments[:10]:
        start = format_time(seg.start)
        text = seg.text
        speaker = seg.speaker
        speaker_prefix = f"{speaker} " if speaker else ""
        print(f"  [{start}] {speaker_prefix}{text}")
    if len(segments) > 10:
        print(f"  ... ({len(segments) - 10} more segments)")

    return segments, language


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract a timestamped transcript from a video using Whisper."
    )
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--model", "-m", default="medium",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: medium)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output file path (default: same name as video)")
    parser.add_argument("--format", "-f", default="txt",
                        choices=["txt", "srt", "vtt", "json"],
                        help="Output format (default: txt)")
    parser.add_argument("--no-speaker-detection", action="store_true",
                        help="Use Whisper-only transcription even when HF_TOKEN is set")

    args = parser.parse_args()

    extract_transcript(
        video_path=args.video,
        model_name=args.model,
        output=args.output,
        fmt=args.format,
        speaker_detection=not args.no_speaker_detection,
    )
