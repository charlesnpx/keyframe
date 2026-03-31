#!/usr/bin/env python3
"""
keyframe CLI — Extract key frames and transcripts from video files.

Usage:
    keyframe video.mp4
    keyframe video.mp4 -o ./output
    keyframe video.mp4 --frames-only
    keyframe video.mp4 --transcript-only
    keyframe install-skills
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path


def _skill_bundle_dir() -> Path:
    """Resolve the bundled skill directory (works both in dev and installed)."""
    return Path(__file__).resolve().parent.parent / "skill"


def install_skills() -> list[str]:
    """Install bundled skills for Claude Code and Codex CLI."""
    skill_dir = _skill_bundle_dir()
    installed: list[str] = []

    if shutil.which("claude"):
        dest = Path.home() / ".claude" / "skills" / "keyframe" / "SKILL.md"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(skill_dir / "SKILL.md", dest)
        installed.append(f"Claude Code skill → {dest}")

    if shutil.which("codex"):
        codex_home = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
        dest = codex_home / "skills" / "keyframe" / "SKILL.md"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(skill_dir / "codex" / "SKILL.md", dest)
        installed.append(f"Codex skill → {dest}")

    return installed


def cmd_install_skills(_args):
    installed = install_skills()
    if installed:
        for msg in installed:
            print(f"  ✓ {msg}")
    else:
        print("  No supported CLIs found (claude, codex).")
        print("  Install Claude Code or Codex CLI first.")


def cmd_extract(args):
    video = Path(args.video)
    if not video.exists():
        print(f"Error: file not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        out_dir = Path(args.output)
    else:
        out_dir = Path("/tmp") / f"{video.stem}_extracted"

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir.resolve()}\n")

    t0 = time.time()
    do_frames = not args.transcript_only
    do_transcript = not args.frames_only

    # ── Key frames ──────────────────────────────────────────────────────
    if do_frames:
        print("=" * 60)
        print("KEY FRAME EXTRACTION")
        print("=" * 60)

        from keyframe.frames import (
            sample_frames, CLIPEncoder, clip_oversegment,
            caption_candidates, merge_by_caption, save_results,
        )
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        frames_dir = out_dir / "frames"

        frames, timestamps, frame_indices = sample_frames(
            str(video), args.sample_interval
        )

        clip = CLIPEncoder(device=device)
        clip_emb = clip.embed_images(frames)
        print(f"  Embedded {len(frames)} frames -> {clip_emb.shape}")

        n_clusters = min(args.pass1_clusters, len(frames) // 2)
        candidates = clip_oversegment(clip_emb, timestamps, frame_indices, n_clusters)

        captions = caption_candidates(candidates, frames, device=device)
        final = merge_by_caption(candidates, captions, clip, args.similarity_threshold)
        clip.cleanup()

        save_results(final, frames, str(frames_dir))

        print(f"\n  {len(frames)} sampled -> {len(candidates)} candidates -> "
              f"{len(final)} key frames")
        print(f"  Saved to: {frames_dir.resolve()}")

        del frames, clip_emb, candidates, final
        if device == "mps":
            torch.mps.empty_cache()

    # ── Transcript ──────────────────────────────────────────────────────
    if do_transcript:
        print(f"\n{'=' * 60}")
        print("TRANSCRIPT EXTRACTION")
        print("=" * 60)

        from keyframe.transcript import extract_transcript, write_json

        transcript_path = out_dir / f"transcript.{args.transcript_format}"
        segments, language = extract_transcript(
            video_path=str(video),
            model_name=args.whisper_model,
            output=str(transcript_path),
            fmt=args.transcript_format,
        )

        if args.transcript_format != "json":
            json_path = out_dir / "transcript.json"
            write_json(segments, str(json_path))

    # ── Summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Done in {elapsed:.1f}s")
    print(f"Output: {out_dir.resolve()}")

    files = sorted(out_dir.rglob("*"))
    files = [f for f in files if f.is_file()]
    print(f"\nFiles ({len(files)}):")
    for f in files:
        rel = f.relative_to(out_dir)
        size_kb = f.stat().st_size / 1024
        print(f"  {rel}  ({size_kb:.0f} KB)")


def main():
    # Check for `install-skills` subcommand first
    if len(sys.argv) > 1 and sys.argv[1] == "install-skills":
        cmd_install_skills(None)
        return

    # Everything else is extract mode
    parser = argparse.ArgumentParser(
        prog="keyframe",
        description="Extract key frames and transcripts from video files.\n\n"
                    "Usage:\n"
                    "  keyframe video.mp4\n"
                    "  keyframe video.mp4 -o ./output\n"
                    "  keyframe install-skills",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_extract_args(parser)
    args = parser.parse_args()

    if not args.video:
        parser.print_help()
        return

    cmd_extract(args)


def _add_extract_args(parser):
    parser.add_argument("video", nargs="?", help="Path to input video/audio file")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory (default: /tmp/<video>_extracted/)")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--frames-only", action="store_true",
                      help="Only extract key frames, skip transcript")
    mode.add_argument("--transcript-only", action="store_true",
                      help="Only extract transcript, skip key frames")

    parser.add_argument("--sample-interval", "-i", type=float, default=0.5,
                        help="Sample one frame every N seconds (default: 0.5)")
    parser.add_argument("--pass1-clusters", "-c", type=int, default=15,
                        help="Number of CLIP clusters in pass 1 (default: 15)")
    parser.add_argument("--similarity-threshold", "-t", type=float, default=0.85,
                        help="Caption similarity threshold for merging (default: 0.85)")
    parser.add_argument("--whisper-model", "-w", default="large",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: large)")
    parser.add_argument("--transcript-format", default="txt",
                        choices=["txt", "srt", "vtt", "json"],
                        help="Transcript format (default: txt)")


if __name__ == "__main__":
    main()
