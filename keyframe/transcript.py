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
import os
import re
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
        if self.speaker is not None:
            object.__setattr__(self, "speaker", str(self.speaker))

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


def _clean_word_text(words: Sequence[Mapping[str, Any]]) -> str:
    text = " ".join(
        str(word.get("word") or word.get("text") or "").strip()
        for word in words
        if str(word.get("word") or word.get("text") or "").strip()
    )
    text = re.sub(r"\s+([,.;:!?%])", r"\1", text)
    text = re.sub(r"([({\[])\s+", r"\1", text)
    text = re.sub(r"\s+([)}\]])", r"\1", text)
    return text.strip()


def _word_run_segment(
    words: Sequence[Mapping[str, Any]],
    speaker: str | None,
    fallback_text: str,
    fallback_start: float,
    fallback_end: float,
) -> TranscriptSegment | None:
    text = _clean_word_text(words) or fallback_text.strip()
    if not text:
        return None
    starts = [_coerce_seconds(word.get("start"), fallback_start) for word in words if word.get("start") is not None]
    ends = [_coerce_seconds(word.get("end"), fallback_end) for word in words if word.get("end") is not None]
    start = starts[0] if starts else fallback_start
    end = ends[-1] if ends else fallback_end
    return TranscriptSegment(start=start, end=end, text=text, speaker=speaker)


def _split_whisperx_segment(raw_segment: Mapping[str, Any]) -> tuple[TranscriptSegment, ...]:
    raw_start = _coerce_seconds(raw_segment.get("start"), 0.0)
    raw_end = _coerce_seconds(raw_segment.get("end"), raw_start)
    raw_text = str(raw_segment.get("text", "")).strip()
    segment_speaker = raw_segment.get("speaker")
    words = tuple(word for word in raw_segment.get("words", ()) if isinstance(word, Mapping))

    if not words or not any(word.get("speaker") or segment_speaker for word in words):
        return (TranscriptSegment(raw_start, raw_end, raw_text, segment_speaker),)

    runs: list[TranscriptSegment] = []
    run_words: list[Mapping[str, Any]] = []
    run_speaker: str | None = None
    for word in words:
        word_speaker = word.get("speaker") or segment_speaker
        if run_words and word_speaker != run_speaker:
            segment = _word_run_segment(run_words, run_speaker, raw_text, raw_start, raw_end)
            if segment is not None:
                runs.append(segment)
            run_words = []
        run_words.append(word)
        run_speaker = word_speaker

    if run_words:
        segment = _word_run_segment(run_words, run_speaker, raw_text, raw_start, raw_end)
        if segment is not None:
            runs.append(segment)
    return tuple(runs)


def whisperx_segments_to_transcript_segments(result: Mapping[str, Any]) -> tuple[TranscriptSegment, ...]:
    """Convert WhisperX word-speaker output into product transcript segments."""
    return tuple(
        segment
        for raw_segment in result.get("segments", ())
        if isinstance(raw_segment, Mapping)
        for segment in _split_whisperx_segment(raw_segment)
    )


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


def _extract_with_whisperx(video: Path, model_name: str, hf_token: str) -> tuple[tuple[TranscriptSegment, ...], str]:
    import whisperx
    from whisperx.diarize import DiarizationPipeline

    device, compute_type = _select_whisperx_device()
    print(f"Loading WhisperX model on {device} ({compute_type})...")
    model = whisperx.load_model(model_name, device, compute_type=compute_type)
    audio = whisperx.load_audio(str(video))

    print("Transcribing and aligning words with WhisperX...")
    result = model.transcribe(audio, batch_size=16)
    language = result.get("language", "unknown")
    align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
    result = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    print("Detecting speakers with pyannote...")
    diarize_model = DiarizationPipeline(PYANNOTE_MODEL, token=hf_token, device=device)
    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    return whisperx_segments_to_transcript_segments(result), language


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
                     Attempt WhisperX + pyannote speaker detection when HF_TOKEN is set

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

    hf_token = os.environ.get("HF_TOKEN")
    if speaker_detection and hf_token:
        try:
            segments, language = _extract_with_whisperx(video, model_name, hf_token)
        except Exception as exc:
            _print_speaker_detection_failure(exc)
            segments, language = _extract_with_whisper(video, model_name)
    else:
        if speaker_detection:
            _print_missing_hf_token_warning()
        segments, language = _extract_with_whisper(video, model_name)

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
