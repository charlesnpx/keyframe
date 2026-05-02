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
import hashlib
import json
import os
import shutil
import sys
import time
import tomllib
from importlib import metadata as importlib_metadata
from pathlib import Path


def _skill_bundle_dir() -> Path:
    """Resolve the bundled skill directory (works both in dev and installed)."""
    return Path(__file__).resolve().parent.parent / "skill"


def _version() -> str:
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if pyproject.exists():
        try:
            return tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"]
        except Exception:
            pass
    try:
        return importlib_metadata.version("keyframe")
    except importlib_metadata.PackageNotFoundError:
        return "0.0.0"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _home_for_install(install_root: str | None) -> Path:
    return Path(install_root).expanduser().resolve() if install_root else Path.home()


def _target_specs(target: str = "all", install_root: str | None = None) -> dict[str, tuple[Path, Path]]:
    skill_dir = _skill_bundle_dir()
    home = _home_for_install(install_root)
    codex_home = home / ".codex" if install_root else Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
    specs = {
        "claude": (skill_dir / "SKILL.md", home / ".claude" / "skills" / "keyframe" / "SKILL.md"),
        "codex": (skill_dir / "codex" / "SKILL.md", codex_home / "skills" / "keyframe" / "SKILL.md"),
    }
    if target == "all":
        return specs
    return {target: specs[target]}


def delegated_result(
    operation: str,
    target: str = "all",
    *,
    perform: bool = False,
    install_root: str | None = None,
) -> dict:
    result = {
        "schema": 1,
        "name": "keyframe",
        "version": _version(),
        "operation": operation,
        "kind": "delegated",
        "targets": {},
        "warnings": [],
    }
    for target_name, (src, dst) in _target_specs(target, install_root).items():
        if operation == "install" and perform:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        elif operation == "uninstall" and perform:
            dst.unlink(missing_ok=True)
        rec = {"path": str(dst.resolve())}
        if operation == "install" and dst.exists():
            rec["sha256"] = _sha256(dst)
        result["targets"][target_name] = {"files": [rec]}
    return result


def install_skills(target: str = "all", *, json_mode: bool = False, install_root: str | None = None) -> list[str]:
    """Install bundled skills for Claude Code and Codex CLI."""
    result = delegated_result("install", target, perform=True, install_root=install_root)
    if json_mode:
        print(json.dumps(result, indent=2))
        return []
    installed = []
    for target_name, info in result["targets"].items():
        for f in info["files"]:
            installed.append(f"{target_name.title()} skill → {f['path']}")
    return installed


def cmd_install_skills(args):
    target = getattr(args, "target", "all") if args is not None else "all"
    operation = "install"
    if getattr(args, "plan", False):
        operation = "plan"
    elif getattr(args, "uninstall", False):
        operation = "uninstall"
    json_mode = bool(getattr(args, "json", False))
    if operation != "install" or json_mode:
        result = delegated_result(
            operation,
            target,
            perform=operation != "plan",
            install_root=getattr(args, "install_root", None),
        )
        if json_mode:
            print(json.dumps(result, indent=2))
        else:
            for target_name, info in result["targets"].items():
                print(f"{operation} {target_name}:")
                for f in info["files"]:
                    print(f"  {f['path']}")
        return

    installed = install_skills(target, install_root=getattr(args, "install_root", None))
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
    manifest_frames = None
    manifest_dir = None
    manifest_run_metadata = None

    # ── Key frames ──────────────────────────────────────────────────────
    if do_frames:
        print("=" * 60)
        print("KEY FRAME EXTRACTION")
        print("=" * 60)

        frames_dir = out_dir / "frames"
        from keyframe.pipeline import KeyframeExtractionConfig, extract_keyframes

        result = extract_keyframes(
            video,
            frames_dir,
            KeyframeExtractionConfig(
                sample_interval=args.sample_interval,
                pass1_clusters=args.pass1_clusters,
                similarity_threshold=args.similarity_threshold,
                max_output_frames=getattr(args, "max_output_frames", None),
                verbose_trace=bool(getattr(args, "verbose_trace", False)),
                debug_qa_targets_path=(
                    Path(args.debug_qa_targets)
                    if getattr(args, "debug_qa_targets", None)
                    else None
                ),
            ),
        )
        manifest_frames = result.final
        manifest_dir = frames_dir
        manifest_run_metadata = result.manifest_metadata

        print(f"\n  {result.final_frame_count} key frames")
        print(f"  Saved to: {frames_dir.resolve()}")

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

        if manifest_frames is not None and manifest_dir is not None:
            from keyframe.manifest import write_manifest
            from keyframe.pipeline.contracts import CandidateRecord, candidate_to_manifest_row

            manifest_rows = [
                candidate_to_manifest_row(
                    frame,
                    filename=f"frame_{frame.frame_idx:06d}_{frame.timestamp:.2f}s.png",
                )
                if isinstance(frame, CandidateRecord)
                else dict(frame)
                for frame in manifest_frames
            ]
            write_manifest(manifest_rows, manifest_dir, segments, metadata=manifest_run_metadata)

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
        parser = argparse.ArgumentParser(prog="keyframe install-skills")
        parser.add_argument("--target", choices=["claude", "codex", "all"], default="all")
        op = parser.add_mutually_exclusive_group()
        op.add_argument("--plan", action="store_true", help="Print intended files without writing")
        op.add_argument("--install", action="store_true", help="Install skill files (default)")
        op.add_argument("--uninstall", action="store_true", help="Remove skill files")
        parser.add_argument("--json", action="store_true", help="Emit mise-en-place delegated-installer JSON on stdout")
        parser.add_argument("--install-root", help="Stage install under this absolute directory as if it were HOME")
        cmd_install_skills(parser.parse_args(sys.argv[2:]))
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
                        help="Deprecated no-op; deterministic merge vetoes are used")
    parser.add_argument("--max-output-frames", type=int, default=None,
                        help="Optional final frame cap applied after scoring and dedupe")
    parser.add_argument("--verbose-trace", action="store_true",
                        help="Write structured pipeline trace snapshots for debugging")
    parser.add_argument("--debug-qa-targets", default=None,
                        help="Internal QA debug: JSON target windows to trace through extraction stages")
    parser.add_argument("--whisper-model", "-w", default="medium",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: medium)")
    parser.add_argument("--transcript-format", default="txt",
                        choices=["txt", "srt", "vtt", "json"],
                        help="Transcript format (default: txt)")


if __name__ == "__main__":
    main()
