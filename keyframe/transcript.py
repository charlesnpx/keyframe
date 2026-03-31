#!/usr/bin/env python3
"""
extract_transcript.py

Extracts a timestamped transcript from a video file using OpenAI's Whisper model
(runs locally, no API key needed).

Usage:
    python extract_transcript.py input.mp4 [--model base] [--output transcript.txt] [--format txt]

Dependencies:
    pip install openai-whisper

Whisper will download the model on first run (~140MB for 'base').
Models in order of speed -> accuracy:
    tiny, base, small, medium, large
"""

import argparse
import sys
import json
from pathlib import Path


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


def write_txt(segments, out_path):
    """Plain text with timestamps."""
    with open(out_path, "w", encoding="utf-8") as f:
        for seg in segments:
            start = format_time(seg["start"])
            end = format_time(seg["end"])
            text = seg["text"].strip()
            f.write(f"[{start} --> {end}]  {text}\n")
    print(f"Saved: {out_path}")


def write_srt(segments, out_path):
    """SubRip subtitle format."""
    with open(out_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start = format_srt_time(seg["start"])
            end = format_srt_time(seg["end"])
            text = seg["text"].strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
    print(f"Saved: {out_path}")


def write_vtt(segments, out_path):
    """WebVTT subtitle format."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            start = format_time(seg["start"])
            end = format_time(seg["end"])
            text = seg["text"].strip()
            f.write(f"{start} --> {end}\n{text}\n\n")
    print(f"Saved: {out_path}")


def write_json(segments, out_path):
    """JSON with full segment data."""
    data = [
        {
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
            "text": seg["text"].strip(),
        }
        for seg in segments
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


def extract_transcript(video_path, model_name="base", output=None, fmt="txt"):
    """
    Run Whisper on a video file and save the timestamped transcript.

    Args:
        video_path:  Path to input video
        model_name:  Whisper model size (tiny/base/small/medium/large)
        output:      Output file path (auto-generated if None)
        fmt:         Output format: txt, srt, vtt, json

    Returns:
        (segments, language) tuple
    """
    try:
        import whisper
    except ImportError:
        print("Error: whisper not installed. Run:\n"
              "  pip install openai-whisper\n"
              "(Requires ffmpeg installed on your system too.)",
              file=sys.stderr)
        sys.exit(1)

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
    print(f"Loading model...")

    model = whisper.load_model(model_name)

    print("Transcribing (this may take a while on long videos)...")
    result = model.transcribe(
        str(video),
        verbose=False,
        word_timestamps=False,
    )

    segments = result["segments"]
    language = result.get("language", "unknown")

    print(f"Detected language: {language}")
    print(f"Segments: {len(segments)}")

    if segments:
        duration = segments[-1]["end"]
        print(f"Duration covered: {format_time(duration)}")

    WRITERS[fmt](segments, output)

    print(f"\n--- Preview (first 10 segments) ---")
    for seg in segments[:10]:
        start = format_time(seg["start"])
        text = seg["text"].strip()
        print(f"  [{start}] {text}")
    if len(segments) > 10:
        print(f"  ... ({len(segments) - 10} more segments)")

    return segments, language


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract a timestamped transcript from a video using Whisper."
    )
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--model", "-m", default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: base)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output file path (default: same name as video)")
    parser.add_argument("--format", "-f", default="txt",
                        choices=["txt", "srt", "vtt", "json"],
                        help="Output format (default: txt)")

    args = parser.parse_args()

    extract_transcript(
        video_path=args.video,
        model_name=args.model,
        output=args.output,
        fmt=args.format,
    )
