#!/usr/bin/env python3
"""Run a small public synthetic media smoke check."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from keyframe.media_preflight import MediaPreflightError, probe_media, resolve_extraction_mode
from keyframe.pipeline.streaming import stream_video_features


class SmokeError(RuntimeError):
    """Synthetic smoke check failed."""


def _run(command: list[str]) -> None:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise SmokeError(f"{command[0]} failed with status {completed.returncode}: {detail}")


def _write_synthetic_video(path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        raise SmokeError("ffmpeg is not installed or is not available on PATH")
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x90:rate=12:duration=3",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "mpeg4",
            str(path),
        ]
    )


def _zero_embeddings(images: Any) -> np.ndarray:
    return np.zeros((len(images), 4), dtype=np.float32)


def run_smoke(work_dir: Path, *, sample_interval: float = 1.0) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    video = work_dir / "keyframe_synthetic_smoke.mp4"
    _write_synthetic_video(video)

    try:
        probe = probe_media(video)
        mode = resolve_extraction_mode(probe, frames_only=True, transcript_only=False)
    except MediaPreflightError as exc:
        raise SmokeError(str(exc)) from exc

    features = stream_video_features(
        video,
        sample_interval,
        embed_images=_zero_embeddings,
        video_timing=probe.selected_video_timing,
    )
    if len(features.timestamps) < 2:
        raise SmokeError(f"expected at least two sampled frames, got {len(features.timestamps)}")
    if any(
        right < left
        for left, right in zip(features.timestamps, features.timestamps[1:])
    ):
        raise SmokeError("sample timestamps are not chronological")
    timing_source = features.sampling_timing.get("source")
    if timing_source not in {"decoder_presentation_time", "nominal_source_index_cfr"}:
        raise SmokeError(f"unexpected sampling timing source: {timing_source!r}")

    return {
        "passed": True,
        "video": str(video),
        "sample_interval_seconds": sample_interval,
        "mode": {
            "do_frames": mode.do_frames,
            "do_transcript": mode.do_transcript,
            "notice": mode.notice,
        },
        "stream_count": len(probe.streams),
        "video_timing": (
            probe.selected_video_timing.to_dict()
            if probe.selected_video_timing is not None
            else None
        ),
        "sampling_timing_source": timing_source,
        "sample_count": len(features.timestamps),
        "timestamps": [round(float(value), 3) for value in features.timestamps],
        "frame_indices": [int(value) for value in features.frame_indices],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a model-free synthetic Keyframe smoke check.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Directory for generated synthetic media; default uses a temporary directory.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for the diagnostic JSON report.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.work_dir is None:
        with tempfile.TemporaryDirectory(prefix="keyframe-smoke-") as tmp:
            report = run_smoke(Path(tmp))
    else:
        report = run_smoke(args.work_dir)

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print("PASS synthetic smoke")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SmokeError as exc:
        print(f"FAIL synthetic smoke: {exc}", file=sys.stderr)
        raise SystemExit(1)
