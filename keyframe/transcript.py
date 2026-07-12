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
import json
import math
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"
SPEAKER_DETECTION_SETUP_WARNING = """Warning: no HF_TOKEN found; falling back to transcript without speaker detection.
To enable speaker detection, accept the pyannote model terms at:
https://huggingface.co/pyannote/speaker-diarization-community-1
Then create a Hugging Face access token at:
https://huggingface.co/settings/tokens
and export it as HF_TOKEN."""


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
class DiarizationRow:
    start: float
    end: float
    speaker: str


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
    with open(out_path, "w", encoding="utf-8") as f:
        for seg in transcript_segments(segments):
            start = format_time(seg.start)
            end = format_time(seg.end)
            text = seg.text
            speaker = seg.speaker
            prefix = f"{speaker}  " if speaker else ""
            f.write(f"[{start} --> {end}]  {prefix}{text}\n")
    print(f"Saved: {out_path}")


def write_srt(segments, out_path):
    """SubRip subtitle format."""
    with open(out_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(transcript_segments(segments), 1):
            start = format_srt_time(seg.start)
            end = format_srt_time(seg.end)
            text = _caption_text(seg)
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
    print(f"Saved: {out_path}")


def write_vtt(segments, out_path):
    """WebVTT subtitle format."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in transcript_segments(segments):
            start = format_time(seg.start)
            end = format_time(seg.end)
            text = _caption_text(seg)
            f.write(f"{start} --> {end}\n{text}\n\n")
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
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
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


def _select_whisperx_device() -> tuple[str, str]:
    import torch

    if torch.cuda.is_available():
        return "cuda", "float16"
    return "cpu", "int8"


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


def _extract_with_whisper(video: Path, model_name: str) -> tuple[tuple[TranscriptSegment, ...], str]:
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

    return transcript_segments(result["segments"]), result.get("language", "unknown")


def _detect_speakers(video: Path, hf_token: str) -> tuple[DiarizationRow, ...]:
    import whisperx
    from whisperx.diarize import DiarizationPipeline
    from tqdm import tqdm

    device, _compute_type = _select_whisperx_device()
    audio = whisperx.load_audio(str(video))

    print("Detecting speakers with pyannote...")
    diarize_model = DiarizationPipeline(PYANNOTE_MODEL, token=hf_token, device=device)
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
        return _valid_diarization_rows(diarize_model(audio, progress_callback=update_progress))
    finally:
        progress.close()


def extract_transcript(
    video_path,
    model_name="medium",
    output=None,
    fmt="txt",
    speaker_detection=True,
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

    segments, language = _extract_with_whisper(video, model_name)

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
