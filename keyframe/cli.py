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
    try:
        return importlib_metadata.version("keyframe")
    except importlib_metadata.PackageNotFoundError:
        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        try:
            return tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"]
        except Exception:
            return "0.0.0"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _target_specs(target: str = "all") -> dict[str, tuple[Path, Path]]:
    skill_dir = _skill_bundle_dir()
    codex_home = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
    specs = {
        "claude": (skill_dir / "SKILL.md", Path.home() / ".claude" / "skills" / "keyframe" / "SKILL.md"),
        "codex": (skill_dir / "codex" / "SKILL.md", codex_home / "skills" / "keyframe" / "SKILL.md"),
    }
    if target == "all":
        return specs
    return {target: specs[target]}


def delegated_result(operation: str, target: str = "all", *, perform: bool = False) -> dict:
    result = {
        "schema": 1,
        "name": "keyframe",
        "version": _version(),
        "operation": operation,
        "kind": "delegated",
        "targets": {},
        "warnings": [],
    }
    for target_name, (src, dst) in _target_specs(target).items():
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


def install_skills(target: str = "all", *, json_mode: bool = False) -> list[str]:
    """Install bundled skills for Claude Code and Codex CLI."""
    result = delegated_result("install", target, perform=True)
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
        result = delegated_result(operation, target, perform=operation != "plan")
        if json_mode:
            print(json.dumps(result, indent=2))
        else:
            for target_name, info in result["targets"].items():
                print(f"{operation} {target_name}:")
                for f in info["files"]:
                    print(f"  {f['path']}")
        return

    installed = install_skills(target)
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

    # ── Key frames ──────────────────────────────────────────────────────
    if do_frames:
        print("=" * 60)
        print("KEY FRAME EXTRACTION")
        print("=" * 60)

        from keyframe.frames import (
            sample_frames, CLIPEncoder, ModelPreloader, clip_oversegment,
            caption_candidates, ocr_candidates, _filter_ocr_tokens,
            _build_hybrid_captions, _build_ocr_token_sets,
            save_results, detect_scenes, _laplacian_sharpness,
        )
        from keyframe.dedupe import (
            adjacent_same_screen_dedupe,
            clean_ocr_token_sets,
            compute_dhash,
            filter_low_information_candidates,
            global_candidate_dedupe,
            near_time_dedupe,
        )
        from keyframe.manifest import write_manifest
        from keyframe.merge import union_find_merge
        from keyframe.scoring import (
            allocate_clusters_by_novelty,
            candidate_budget_for_scenes,
            coalesce_tiny_scenes,
        )
        import torch

        if args.similarity_threshold != 0.85:
            print(
                "Warning: --similarity-threshold is deprecated and currently ignored; "
                "deterministic merge vetoes are used instead.",
                file=sys.stderr,
            )

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        frames_dir = out_dir / "frames"

        # Start loading all models in parallel while we sample frames
        preloader = ModelPreloader(device=device, need_florence=True, need_ocr=True)

        frames, timestamps, frame_indices = sample_frames(
            str(video), args.sample_interval
        )
        dhashes = [compute_dhash(frame) for frame in frames]

        clip = CLIPEncoder(device=device, preloaded=preloader.get_clip())
        clip_emb = clip.embed_images(frames)
        print(f"  Embedded {len(frames)} frames -> {clip_emb.shape}")
        # CLIP image encoder is no longer needed after over-segmentation.
        clip.cleanup()
        del clip
        preloader.release_clip()

        # Scene detection → per-scene clustering
        n_clusters = min(args.pass1_clusters, len(frames) // 2)
        scenes = detect_scenes(str(video), timestamps)
        original_scene_count = len(scenes)
        scenes = coalesce_tiny_scenes(scenes, timestamps, dhashes)
        if len(scenes) != original_scene_count:
            print(f"  Coalesced scenes: {original_scene_count} -> {len(scenes)}")
        cluster_budget = candidate_budget_for_scenes(n_clusters, len(scenes))
        cluster_allocs = allocate_clusters_by_novelty(scenes, cluster_budget, dhashes, floor=1)
        print(f"  Cluster allocation budget: {sum(cluster_allocs)} candidates")

        all_candidates = []
        cluster_offset = 0
        for (s_start, s_end), scene_clusters in zip(scenes, cluster_allocs):
            if scene_clusters <= 0:
                continue
            scene_emb = clip_emb[s_start:s_end + 1]
            scene_ts = timestamps[s_start:s_end + 1]
            scene_fi = frame_indices[s_start:s_end + 1]
            scene_frames = frames[s_start:s_end + 1]
            scene_clusters = min(scene_clusters, len(scene_emb))

            if scene_clusters < 2 or len(scene_emb) < 2:
                best = max(range(len(scene_frames)),
                           key=lambda i: _laplacian_sharpness(scene_frames[i]))
                sharpness = _laplacian_sharpness(scene_frames[best])
                all_candidates.append({
                    "sample_idx": s_start + best,
                    "frame_idx": scene_fi[best],
                    "timestamp": scene_ts[best],
                    "clip_cluster": cluster_offset,
                    "clip_cluster_size": len(scene_emb),
                    "sharpness": float(sharpness),
                })
                cluster_offset += 1
                continue

            scene_cands = clip_oversegment(
                scene_emb, scene_ts, scene_fi, scene_clusters, scene_frames,
                dhashes=dhashes[s_start:s_end + 1],
            )
            for c in scene_cands:
                c["sample_idx"] = s_start + c["sample_idx"]
                c["clip_cluster"] += cluster_offset
            cluster_offset += scene_clusters
            all_candidates.extend(scene_cands)

        candidates = sorted(all_candidates, key=lambda c: c["timestamp"])
        for cand in candidates:
            cand["dhash"] = dhashes[cand["sample_idx"]]
            cand["dhash_hex"] = f"{dhashes[cand['sample_idx']]:016x}"

        florence_captions = caption_candidates(
            candidates, frames, device=device, preloaded=preloader.get_florence()
        )
        # Florence is not needed past pass 2; free it before OCR loads.
        preloader.release_florence()

        ocr_texts = ocr_candidates(
            candidates, frames, preloaded_engine=preloader.get_ocr_engine()
        )
        preloader.release_ocr_engine()

        filtered_ocr = _filter_ocr_tokens(ocr_texts)
        florence_caps, ocr_caps, has_ocr = _build_hybrid_captions(
            filtered_ocr, florence_captions, candidates
        )
        ocr_token_sets = _build_ocr_token_sets(filtered_ocr)
        ocr_token_sets = clean_ocr_token_sets(ocr_token_sets)
        for cand, tokens in zip(candidates, ocr_token_sets):
            cand["ocr_tokens"] = sorted(tokens)

        deduped = near_time_dedupe(candidates, ocr_token_sets, dhashes)
        print(f"  Near-time dedupe: {len(candidates)} -> {len(deduped)} candidates")
        deduped_token_sets = [set(c.get("ocr_tokens", [])) for c in deduped]
        globally_deduped = global_candidate_dedupe(deduped, deduped_token_sets, dhashes)
        print(f"  Global conservative dedupe: {len(deduped)} -> {len(globally_deduped)} candidates")
        filtered = filter_low_information_candidates(globally_deduped, frames)
        print(f"  Low-information filter: {len(globally_deduped)} -> {len(filtered)} candidates")
        adjacent_deduped = adjacent_same_screen_dedupe(filtered)
        print(f"  Adjacent same-screen dedupe: {len(filtered)} -> {len(adjacent_deduped)} candidates")
        deduped_token_sets = [set(c.get("ocr_tokens", [])) for c in adjacent_deduped]
        deduped_has_ocr = [len(tokens) >= 3 for tokens in deduped_token_sets]
        final = union_find_merge(adjacent_deduped, deduped_token_sets, deduped_has_ocr, frames)

        # Keep only the PIL images we still need to write to disk, then drop
        # the full sampled-frames list before Whisper loads.
        selected_imgs = {c["sample_idx"]: frames[c["sample_idx"]] for c in final}
        del frames, clip_emb, candidates, deduped, globally_deduped, filtered, adjacent_deduped

        save_results(final, selected_imgs, str(frames_dir))
        write_manifest(final, frames_dir)
        manifest_frames = final
        manifest_dir = frames_dir

        print(f"\n  {len(final)} key frames")
        print(f"  Saved to: {frames_dir.resolve()}")

        del selected_imgs
        preloader.shutdown()
        del preloader
        import gc
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

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

            write_manifest(manifest_frames, manifest_dir, segments)

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
    parser.add_argument("--whisper-model", "-w", default="medium",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: medium)")
    parser.add_argument("--transcript-format", default="txt",
                        choices=["txt", "srt", "vtt", "json"],
                        help="Transcript format (default: txt)")


if __name__ == "__main__":
    main()
