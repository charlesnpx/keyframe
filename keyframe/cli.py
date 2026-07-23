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
import tempfile
import time
import tomllib
from contextlib import nullcontext
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any


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
    if target == "tools":
        result["targets"]["tools"] = {"files": []}
        return result
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
    if target == "tools":
        return ["Tools target is managed by mise-en-place via pipx"]
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


def _resolve_out_dir(video: Path, output: str | None) -> Path:
    """Resolve and create the output directory for an extraction run.

    When ``--output`` is not given, default to a folder next to the input file
    (``<input-file-folder>/<stem>_extracted``). If that folder isn't writable,
    fall back to ``/tmp``. An explicit ``--output`` is always honored verbatim.

    The directory is created here so the fallback triggers on the actual failure
    (EAFP) rather than a separate, advisory ``os.access`` pre-check.
    """
    if output:
        out_dir = Path(output)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    preferred = video.parent / f"{video.stem}_extracted"
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        return preferred
    except OSError:
        fallback = Path("/tmp") / f"{video.stem}_extracted"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


class ExtractionPreflightError(ValueError):
    """CLI extraction inputs cannot safely proceed to side effects."""


@dataclass(frozen=True)
class ExtractionPreflight:
    input_path: Path
    output_dir: Path
    do_frames: bool
    do_transcript: bool
    notice: str | None
    transcript: Any | None
    frame_config: Any | None


def _nearest_existing_directory(path: Path, *, label: str) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    ancestor = candidate
    while not os.path.lexists(ancestor):
        parent = ancestor.parent
        if parent == ancestor:
            raise ExtractionPreflightError(
                f"{label} has no existing directory ancestor: {path}"
            )
        ancestor = parent
    try:
        resolved = ancestor.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ExtractionPreflightError(
            f"{label} ancestor cannot be resolved: {ancestor}: {exc}"
        ) from exc
    if not resolved.is_dir():
        raise ExtractionPreflightError(
            f"{label} nearest existing ancestor is not a directory: {ancestor}"
        )
    if not os.access(resolved, os.W_OK | os.X_OK):
        raise ExtractionPreflightError(
            f"{label} nearest existing directory is not writable: {resolved}"
        )
    return resolved


def _validate_directory_destination(path: Path, *, label: str) -> Path:
    _nearest_existing_directory(path, label=label)
    return path


def _plan_out_dir(video: Path, output: str | None) -> Path:
    if output:
        return _validate_directory_destination(
            Path(output),
            label="output directory",
        )
    preferred = video.parent / f"{video.stem}_extracted"
    try:
        return _validate_directory_destination(
            preferred,
            label="default output directory",
        )
    except ExtractionPreflightError:
        fallback = Path("/tmp") / f"{video.stem}_extracted"
        return _validate_directory_destination(
            fallback,
            label="fallback output directory",
        )


def _transcript_config(args):
    from keyframe.transcript_cli import TranscriptRunConfig

    return TranscriptRunConfig(
        model_name=args.whisper_model,
        fmt=args.transcript_format,
        transcription_backend=getattr(args, "transcription_backend", "auto"),
        diarization_device=getattr(args, "diarization_device", "auto"),
        stage_concurrency=getattr(args, "stage_concurrency", "auto"),
        speaker_detection=not bool(getattr(args, "no_speaker_detection", False)),
    )


def _preflight_transcript(args):
    from keyframe.transcript_cli import preflight_transcript_run

    return preflight_transcript_run(_transcript_config(args))


def _run_transcript(video: Path, out_dir: Path, preflight, *, supervisor=None):
    from keyframe.transcript_cli import run_supervised_transcript

    if supervisor is None:
        return run_supervised_transcript(video, out_dir, preflight)
    return run_supervised_transcript(
        video,
        out_dir,
        preflight,
        supervisor=supervisor,
    )


def _frame_config(
    args,
    *,
    device: str | None = None,
    include_paths: bool = True,
):
    from keyframe.pipeline import KeyframeExtractionConfig

    frame_cache_arg = (
        getattr(args, "frame_cache_dir", None) if include_paths else None
    )
    qa_targets_arg = (
        getattr(args, "debug_qa_targets", None) if include_paths else None
    )
    return KeyframeExtractionConfig(
        sample_interval=getattr(args, "sample_interval", 0.5),
        pass1_clusters=getattr(args, "pass1_clusters", 15),
        similarity_threshold=getattr(args, "similarity_threshold", 0.85),
        device=device,
        max_output_frames=getattr(args, "max_output_frames", None),
        max_clustering_memory_mb=getattr(args, "max_clustering_memory_mb", 2048),
        max_frame_cache_mb=getattr(args, "max_frame_cache_mb", 8192),
        frame_cache_dir=(
            Path(frame_cache_arg) if frame_cache_arg else None
        ),
        verbose_trace=bool(getattr(args, "verbose_trace", False)),
        debug_qa_targets_path=(
            Path(qa_targets_arg) if qa_targets_arg else None
        ),
    )


def _preflight_extract(args) -> ExtractionPreflight:
    from keyframe.frame_preflight import (
        FramePreflightError,
        preflight_frame_runtime,
        resolve_frame_execution_device,
    )
    from keyframe.media_preflight import (
        MediaPreflightError,
        probe_media,
        resolve_extraction_mode,
        resolve_readable_media_file,
    )

    try:
        video = resolve_readable_media_file(args.video)
        media = probe_media(video)
        mode = resolve_extraction_mode(
            media,
            frames_only=bool(getattr(args, "frames_only", False)),
            transcript_only=bool(getattr(args, "transcript_only", False)),
        )
        output_dir = _plan_out_dir(video, getattr(args, "output", None))
    except (MediaPreflightError, ExtractionPreflightError) as exc:
        raise ExtractionPreflightError(str(exc)) from exc

    transcript_preflight = None
    if mode.do_transcript:
        from keyframe.transcript import TranscriptionError

        try:
            transcript_preflight = _preflight_transcript(args)
        except (ValueError, TranscriptionError) as exc:
            raise ExtractionPreflightError(str(exc)) from exc

    frame_config = None
    if mode.do_frames:
        try:
            frame_runtime = preflight_frame_runtime()
            if transcript_preflight is not None:
                from keyframe.full_pipeline import resolve_frame_device

                frame_device = resolve_frame_device(transcript_preflight)
            else:
                frame_device = resolve_frame_execution_device(frame_runtime)
            frame_config = _frame_config(args, device=frame_device)
            cache_root = (
                frame_config.frame_cache_dir
                if frame_config.frame_cache_dir is not None
                else Path(tempfile.gettempdir())
            )
            _validate_directory_destination(
                cache_root,
                label="frame cache directory",
            )
        except (FramePreflightError, ValueError) as exc:
            raise ExtractionPreflightError(str(exc)) from exc
    else:
        try:
            _frame_config(args, include_paths=False)
        except ValueError as exc:
            raise ExtractionPreflightError(str(exc)) from exc

    return ExtractionPreflight(
        input_path=video,
        output_dir=output_dir,
        do_frames=mode.do_frames,
        do_transcript=mode.do_transcript,
        notice=mode.notice,
        transcript=transcript_preflight,
        frame_config=frame_config,
    )


def _run_frame_generation(
    video: Path,
    out_dir: Path,
    frame_config,
    session,
):
    from keyframe.frame_generation import StagedFrameGeneration
    from keyframe.pipeline import extract_keyframes

    if session.staging is None:
        raise RuntimeError("frame generation session did not initialize staging paths")
    result = extract_keyframes(
        video,
        session.staging.frames,
        frame_config,
        report_output_dir=out_dir / "frames",
    )
    return StagedFrameGeneration.from_extraction(session, result)


def _run_full_pipeline(
    video: Path,
    out_dir: Path,
    transcript_preflight,
    frame_config,
    supervisor,
):
    from keyframe.full_pipeline import (
        resolve_frame_device,
        run_supervised_full_pipeline,
    )

    frame_device = resolve_frame_device(transcript_preflight)
    if frame_config.device != frame_device:
        raise RuntimeError(
            "validated frame configuration does not match the scheduled device"
        )
    return run_supervised_full_pipeline(
        video,
        out_dir,
        transcript_preflight,
        supervisor=supervisor,
        frame_device=frame_device,
        frame_runner=lambda: _run_frame_generation(
            video,
            out_dir,
            frame_config,
            supervisor,
        ),
    )


def _frame_session(out_dir: Path, *, with_transcript: bool):
    if with_transcript:
        from keyframe.stage_supervisor import StageSupervisor
        from keyframe.transcript_cli import print_stage_progress

        return StageSupervisor(
            out_dir,
            progress_callback=print_stage_progress,
        )
    from keyframe.frame_generation import FrameGenerationSession

    return FrameGenerationSession(out_dir)


def _print_frame_result(result) -> None:
    print(f"\n  {result.final_frame_count} key frames")
    print(f"  Saved to: {result.output_dir.resolve()}")


def cmd_extract(args):
    try:
        preflight = _preflight_extract(args)
    except ExtractionPreflightError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    video = preflight.input_path
    do_frames = preflight.do_frames
    do_transcript = preflight.do_transcript
    transcript_preflight = preflight.transcript
    frame_config = preflight.frame_config
    if preflight.notice is not None:
        print(f"Notice: {preflight.notice}.")

    if frame_config is not None and frame_config.frame_cache_dir is not None:
        try:
            frame_config.frame_cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            print(
                f"Error: could not create frame cache directory: {exc}",
                file=sys.stderr,
            )
            raise SystemExit(1) from None
    try:
        out_dir = _resolve_out_dir(video, getattr(args, "output", None))
    except OSError as exc:
        print(f"Error: could not create output directory: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    if out_dir.resolve() != preflight.output_dir.resolve():
        print(
            "Error: output directory changed after preflight",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print(f"Output: {out_dir.resolve()}\n")

    t0 = time.time()
    session_context = (
        _frame_session(out_dir, with_transcript=do_transcript)
        if do_frames
        else nullcontext(None)
    )
    from keyframe.output_session import OutputSessionError

    try:
        with session_context as session:
            if do_frames and do_transcript:
                print("=" * 60)
                print("FULL EXTRACTION")
                print("=" * 60)
                if session is None:
                    raise RuntimeError("full extraction session was not initialized")
                if transcript_preflight is None:
                    raise RuntimeError("transcript preflight was not initialized")
                if frame_config is None:
                    raise RuntimeError("frame preflight was not initialized")
                full_result = _run_full_pipeline(
                    video,
                    out_dir,
                    transcript_preflight,
                    frame_config,
                    session,
                )
                _print_frame_result(full_result.frames)
            elif do_frames:
                print("=" * 60)
                print("KEY FRAME EXTRACTION")
                print("=" * 60)
                if session is None:
                    raise RuntimeError("frame generation session was not initialized")
                if frame_config is None:
                    raise RuntimeError("frame preflight was not initialized")
                frame_generation = _run_frame_generation(
                    video,
                    out_dir,
                    frame_config,
                    session,
                )
                _print_frame_result(frame_generation.promote())
            elif do_transcript:
                print(f"\n{'=' * 60}")
                print("TRANSCRIPT EXTRACTION")
                print("=" * 60)

                if transcript_preflight is None:
                    raise RuntimeError("transcript preflight was not initialized")
                _run_transcript(
                    video,
                    out_dir,
                    transcript_preflight,
                )
    except OutputSessionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        print("Error: extraction interrupted", file=sys.stderr)
        raise SystemExit(130) from None
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None

    # ── Summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Done in {elapsed:.1f}s")
    print(f"Output: {out_dir.resolve()}")

    from keyframe.managed_workspace import known_public_artifact_paths

    files = known_public_artifact_paths(out_dir)
    print(f"\nFiles ({len(files)}):")
    for f in files:
        rel = f.relative_to(out_dir)
        size_kb = f.stat().st_size / 1024
        print(f"  {rel}  ({size_kb:.0f} KB)")


def _build_extract_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="keyframe",
        description="Probe media and extract supported key frames and/or "
                    "transcripts.\n"
                    "Frames: Darwin ARM64 or Linux x86-64. "
                    "Transcript-only remains portable.\n\n"
                    "Usage:\n"
                    "  keyframe video.mp4\n"
                    "  keyframe extract video.mp4\n"
                    "  keyframe video.mp4 -o ./output\n"
                    "  keyframe install-skills",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_extract_args(parser)
    return parser


def _parse_extract_args(argv):
    argv = list(argv)
    if argv and argv[0] == "extract":
        argv = argv[1:]
    return _build_extract_parser().parse_args(argv)


def main():
    # Check for `install-skills` subcommand first
    if len(sys.argv) > 1 and sys.argv[1] == "install-skills":
        parser = argparse.ArgumentParser(prog="keyframe install-skills")
        parser.add_argument("--target", choices=["claude", "codex", "tools", "all"], default="all")
        op = parser.add_mutually_exclusive_group()
        op.add_argument("--plan", action="store_true", help="Print intended files without writing")
        op.add_argument("--install", action="store_true", help="Install skill files (default)")
        op.add_argument("--uninstall", action="store_true", help="Remove skill files")
        parser.add_argument("--json", action="store_true", help="Emit mise-en-place delegated-installer JSON on stdout")
        parser.add_argument("--install-root", help="Stage install under this absolute directory as if it were HOME")
        cmd_install_skills(parser.parse_args(sys.argv[2:]))
        return

    # Direct extraction and the explicit `extract` alias share one parser.
    parser = _build_extract_parser()
    argv = sys.argv[1:]
    if argv and argv[0] == "extract":
        argv = argv[1:]
    args = parser.parse_args(argv)

    if not args.video:
        parser.print_help()
        return

    cmd_extract(args)


def _add_extract_args(parser):
    parser.add_argument("video", nargs="?", help="Path to input video/audio file")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory (default: <input-file-folder>/<video>_extracted/, "
                             "falls back to /tmp if that folder isn't writable)")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--frames-only", action="store_true",
                      help="Require usable video and extract frames only "
                           "(Darwin ARM64 or Linux x86-64)")
    mode.add_argument("--transcript-only", action="store_true",
                      help="Require usable audio and extract transcript only")

    parser.add_argument("--sample-interval", "-i", type=float, default=0.5,
                        help="Sample one frame every N seconds (default: 0.5)")
    parser.add_argument("--pass1-clusters", "-c", type=int, default=15,
                        help="Number of CLIP clusters in pass 1 (1-64, default: 15)")
    parser.add_argument("--max-clustering-memory-mb", type=int, default=2048,
                        help="Maximum memory admitted for an isolated clustering worker (default: 2048)")
    parser.add_argument("--max-frame-cache-mb", type=int, default=8192,
                        help="Maximum lossless candidate cache size in MiB (default: 8192)")
    parser.add_argument("--frame-cache-dir", default=None,
                        help="Directory for temporary candidate frames (default: the OS temp directory)")
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
    parser.add_argument("--transcription-backend", default="auto",
                        choices=["auto", "mlx", "whisper"],
                        help="Transcription backend (default: auto)")
    parser.add_argument("--diarization-device", default="auto",
                        choices=["auto", "cpu", "mps", "cuda"],
                        help="Speaker-detection device (default: auto)")
    parser.add_argument("--stage-concurrency", default="auto",
                        choices=["auto", "serial", "parallel"],
                        help="Transcript-stage concurrency policy (default: auto)")
    parser.add_argument("--no-speaker-detection", action="store_true",
                        help="Skip pyannote speaker detection even when HF_TOKEN is set")


if __name__ == "__main__":
    main()
