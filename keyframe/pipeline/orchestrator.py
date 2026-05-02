from __future__ import annotations

from pathlib import Path
from typing import Any
import gc
import sys
import time

from keyframe.pipeline.config import KeyframeExtractionConfig, KeyframeExtractionResult
from keyframe.pipeline.context import make_context, RunContext
from keyframe.pipeline.contracts import (
    CandidateBatch,
    CandidateRecord,
    FeatureOutput,
    FrameStore,
    OutputArtifacts,
    ProposalOutput,
    SampleTable,
    SamplingOutput,
    TemporalOutput,
    as_candidate_record,
    candidate_records,
    candidate_to_manifest_row,
)
from keyframe.pipeline.qa_targets import write_debug_qa_trace
from keyframe.pipeline.trace import NoOpTraceSink, SnapshotTraceSink, TraceSink


def _default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _empty_cache(device: str) -> None:
    import torch

    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def _candidate_batch(stage: str, candidates, metadata: dict[str, Any] | None = None) -> CandidateBatch:
    return CandidateBatch(stage=stage, candidates=candidate_records(candidates), metadata=metadata or {})


def _select_pass1_candidates(
    *,
    clip_emb,
    frames,
    timestamps,
    frame_indices,
    scenes,
    cluster_allocs,
    dhashes,
) -> tuple[tuple[CandidateRecord, ...], dict[int, int], dict[int, int]]:
    from keyframe.frames import clip_oversegment
    from keyframe.visual import laplacian_sharpness

    all_candidates: list[dict[str, Any]] = []
    sample_clusters: dict[int, int] = {}
    sample_scenes: dict[int, int] = {}
    cluster_offset = 0

    for scene_id, ((s_start, s_end), scene_clusters) in enumerate(zip(scenes, cluster_allocs)):
        for sample_idx in range(s_start, s_end + 1):
            sample_scenes[sample_idx] = scene_id
        if scene_clusters <= 0:
            continue

        scene_emb = clip_emb[s_start:s_end + 1]
        scene_ts = timestamps[s_start:s_end + 1]
        scene_fi = frame_indices[s_start:s_end + 1]
        scene_frames = frames[s_start:s_end + 1]
        scene_clusters = min(scene_clusters, len(scene_emb))

        if scene_clusters < 2 or len(scene_emb) < 2:
            best = max(range(len(scene_frames)), key=lambda i: laplacian_sharpness(scene_frames[i]))
            sharpness = laplacian_sharpness(scene_frames[best])
            all_candidates.append({
                "sample_idx": s_start + best,
                "frame_idx": scene_fi[best],
                "timestamp": scene_ts[best],
                "clip_cluster": cluster_offset,
                "clip_cluster_size": len(scene_emb),
                "sharpness": float(sharpness),
                "cluster_role": "single",
                "scene_id": scene_id,
            })
            for sample_idx in range(s_start, s_end + 1):
                sample_clusters[sample_idx] = cluster_offset
            cluster_offset += 1
            continue

        scene_cands, scene_labels = clip_oversegment(
            scene_emb,
            scene_ts,
            scene_fi,
            scene_clusters,
            scene_frames,
            dhashes=dhashes[s_start:s_end + 1],
            max_reps_per_cluster=2,
            return_labels=True,
        )
        for local_idx, label in enumerate(scene_labels):
            sample_clusters[s_start + local_idx] = cluster_offset + int(label)
        scene_cands = [
            {
                **row,
                "sample_idx": s_start + row["sample_idx"],
                "clip_cluster": row["clip_cluster"] + cluster_offset,
                "scene_id": scene_id,
            }
            for row in scene_cands
        ]
        cluster_offset += scene_clusters
        all_candidates.extend(scene_cands)

    candidates = tuple(
        as_candidate_record(row).with_visual(
            dhash=dhashes[int(row["sample_idx"])],
            dhash_hex=f"{dhashes[int(row['sample_idx'])]:016x}",
        )
        for row in sorted(all_candidates, key=lambda c: c["timestamp"])
    )
    return candidates, sample_clusters, sample_scenes


class SamplingStage:
    name = "sampling"

    def run(self, video_path: Path, ctx: RunContext) -> SamplingOutput:
        from keyframe.frames import sample_frames

        frames, timestamps, frame_indices = sample_frames(str(video_path), ctx.config.sample_interval)
        if len(frames) < 4:
            print("Too few frames.", file=sys.stderr)
            sys.exit(1)
        out = SamplingOutput(
            frame_store=FrameStore(frames=list(frames)),
            samples=SampleTable(
                timestamps=[float(ts) for ts in timestamps],
                frame_indices=[int(idx) for idx in frame_indices],
            ),
        )
        ctx.trace.exit(self.name, out)
        return out


class FeatureStage:
    name = "features"

    def __init__(self, preloader: Any, device: str):
        self.preloader = preloader
        self.device = device

    def run(self, sampling: SamplingOutput, ctx: RunContext) -> FeatureOutput:
        from keyframe.dedupe import compute_dhash
        from keyframe.frames import CLIPEncoder

        frames = sampling.frame_store.frames
        dhashes = [compute_dhash(frame) for frame in frames]

        print("\n── Pass 1: CLIP visual embedding ──")
        clip = CLIPEncoder(device=self.device, preloaded=self.preloader.get_clip())
        clip_emb = clip.embed_images(frames)
        print(f"  Embedded {len(frames)} frames -> {clip_emb.shape}")
        clip.cleanup()
        del clip
        self.preloader.release_clip()

        out = FeatureOutput(dhashes=dhashes, clip_embeddings=clip_emb)
        ctx.trace.exit("features.dhash", {"dhash_count": len(dhashes)})
        ctx.trace.exit("features.clip", out)
        return out


class TemporalStage:
    name = "temporal"

    def run(self, video_path: Path, sampling: SamplingOutput, features: FeatureOutput, ctx: RunContext) -> TemporalOutput:
        from keyframe.frames import detect_scenes
        from keyframe.scoring import (
            allocate_clusters_by_novelty,
            candidate_budget_for_scenes,
            coalesce_tiny_scenes,
        )

        timestamps = sampling.samples.timestamps
        dhashes = features.dhashes
        n_clusters = min(ctx.config.pass1_clusters, len(timestamps) // 2)
        scenes = detect_scenes(str(video_path), timestamps)
        scenes, scene_coalescence = coalesce_tiny_scenes(
            scenes, timestamps, dhashes, return_trace=True
        )
        if scene_coalescence["coalesced_scene_count"] != scene_coalescence["original_scene_count"]:
            print(
                "  Coalesced scenes: "
                f"{scene_coalescence['original_scene_count']} -> "
                f"{scene_coalescence['coalesced_scene_count']}"
            )
        cluster_budget = candidate_budget_for_scenes(n_clusters, len(scenes))
        cluster_allocs = allocate_clusters_by_novelty(scenes, cluster_budget, dhashes, floor=1)
        print(f"  Cluster allocation budget: {sum(cluster_allocs)} candidates")
        out = TemporalOutput(
            scenes=list(scenes),
            scene_coalescence=scene_coalescence,
            cluster_allocs=list(cluster_allocs),
            sample_clusters={},
            sample_scenes={},
            sample_temporal_windows={},
        )
        ctx.trace.exit("temporal.scenes", out)
        ctx.trace.exit("temporal.cluster_allocation", out)
        return out


class ProposalStage:
    name = "proposal"

    def run(
        self,
        sampling: SamplingOutput,
        features: FeatureOutput,
        temporal: TemporalOutput,
        ctx: RunContext,
    ) -> ProposalOutput:
        from keyframe.scoring import (
            assign_temporal_window_ids,
            build_rescue_shortlist,
            rescue_window_seconds,
        )

        frames = sampling.frame_store.frames
        timestamps = sampling.samples.timestamps
        frame_indices = sampling.samples.frame_indices
        n_clusters = min(ctx.config.pass1_clusters, len(frames) // 2)

        candidates, sample_clusters, sample_scenes = _select_pass1_candidates(
            clip_emb=features.clip_embeddings,
            frames=frames,
            timestamps=timestamps,
            frame_indices=frame_indices,
            scenes=temporal.scenes,
            cluster_allocs=temporal.cluster_allocs,
            dhashes=features.dhashes,
        )
        temporal.sample_clusters = sample_clusters
        temporal.sample_scenes = sample_scenes

        window_seconds = rescue_window_seconds(timestamps)
        temporal_window_ids = assign_temporal_window_ids(
            timestamps,
            sample_scenes,
            window_seconds=window_seconds,
        )
        temporal.sample_temporal_windows = {
            sample_idx: int(window_id)
            for sample_idx, window_id in enumerate(temporal_window_ids)
        }
        candidates = tuple(
            cand.with_selection(
                proxy_content_score=float(cand.selection.proxy_content_score or 0.0),
            ).with_temporal(
                scene_id=sample_scenes.get(int(cand.sample_idx)),
                temporal_window_id=(
                    temporal_window_ids[int(cand.sample_idx)]
                    if int(cand.sample_idx) < len(temporal_window_ids)
                    else None
                ),
                temporal_window_seconds=(
                    window_seconds
                    if int(cand.sample_idx) < len(temporal_window_ids)
                    else None
                ),
            )
            for cand in candidates
        )
        ctx.trace.exit("temporal.sample_context", temporal)

        pass1_primary = [c for c in candidates if c.visual.cluster_role != "alt"]
        cluster_alt = [c for c in candidates if c.visual.cluster_role == "alt"]
        ctx.trace.exit("proposal.pass1_primary", _candidate_batch("proposal.pass1_primary", pass1_primary))
        ctx.trace.exit("proposal.cluster_alt", _candidate_batch("proposal.cluster_alt", cluster_alt))

        (
            shortlist,
            proxy_rows,
            rescue_budget,
            rescue_ocr_cap,
            temporal_window_count,
            scene_count,
            legacy_proxy_dropped_count,
        ) = build_rescue_shortlist(
            frames,
            timestamps,
            frame_indices,
            candidates,
            n_clusters,
            sample_clusters=sample_clusters,
            sample_scenes=sample_scenes,
        )
        candidates = tuple(
            cand.with_selection(proxy_content_score=float(proxy_rows[int(cand.sample_idx)]["proxy_content_score"]))
            if 0 <= int(cand.sample_idx) < len(proxy_rows)
            else cand
            for cand in candidates
        )
        shortlist = tuple(
            row.with_visual(
                dhash=features.dhashes[int(row.sample_idx)],
                dhash_hex=f"{features.dhashes[int(row.sample_idx)]:016x}",
            )
            for row in shortlist
        )
        rescue_metadata = {
            "rescue_budget": int(rescue_budget),
            "rescue_ocr_cap": int(rescue_ocr_cap),
            "temporal_window_count": int(temporal_window_count),
            "scene_count": int(scene_count),
            "legacy_proxy_dropped_count": int(legacy_proxy_dropped_count),
        }
        ctx.trace.exit(
            "proposal.rescue_shortlist",
            _candidate_batch("proposal.rescue_shortlist", shortlist, rescue_metadata),
        )
        return ProposalOutput(
            candidates=candidates,
            rescue_shortlist=shortlist,
            proxy_rows=proxy_rows,
            rescue_budget=rescue_budget,
            rescue_ocr_cap=rescue_ocr_cap,
            temporal_window_count=temporal_window_count,
            scene_count=scene_count,
            legacy_proxy_dropped_count=legacy_proxy_dropped_count,
        )


class RescueEvidenceStage:
    name = "evidence.rescue_ocr_batch"

    def __init__(self, preloader: Any):
        self.preloader = preloader

    def run(self, proposal: ProposalOutput, sampling: SamplingOutput, ctx: RunContext) -> None:
        from keyframe.frames import (
            _comparison_primary_sample_idxs,
            attach_rescue_ocr_metadata,
            ocr_candidates,
        )
        from keyframe.visual import laplacian_sharpness

        shortlist = proposal.rescue_shortlist
        frames = sampling.frame_store.frames
        if not shortlist:
            print("  Rescue shortlist: 0 frames")
            return
        shortlist = tuple(
            row.with_visual(sharpness=float(laplacian_sharpness(frames[int(row.sample_idx)])))
            for row in shortlist
        )
        proposal.rescue_shortlist = shortlist
        print(
            f"  Rescue shortlist: {len(shortlist)} frames, "
            f"budget {proposal.rescue_budget}, OCR cap {proposal.rescue_ocr_cap}"
        )
        comparison_idxs = _comparison_primary_sample_idxs(proposal.candidates, shortlist)
        candidate_by_idx = {int(cand.sample_idx): cand for cand in proposal.candidates}
        comparison_primaries = [
            candidate_by_idx[sample_idx]
            for sample_idx in sorted(comparison_idxs)
            if sample_idx in candidate_by_idx
        ]
        ocr_batch = shortlist + tuple(comparison_primaries)
        rescue_ocr_texts, ocr_batch = ocr_candidates(ocr_batch, frames, preloaded_engine=self.preloader.get_ocr_engine())
        ocr_batch = attach_rescue_ocr_metadata(ocr_batch, rescue_ocr_texts)
        updated_by_idx = {int(cand.sample_idx): cand for cand in ocr_batch}
        proposal.rescue_shortlist = tuple(updated_by_idx.get(int(row.sample_idx), row) for row in shortlist)
        proposal.candidates = tuple(updated_by_idx.get(int(cand.sample_idx), cand) for cand in proposal.candidates)
        print(
            "  Rescue OCR comparison primaries: "
            f"{len(comparison_primaries)} cached selected candidates"
        )
        ctx.trace.exit(self.name, _candidate_batch(self.name, ocr_batch))


class RescueSelectionStage:
    name = "selection.after_rescue"

    def run(self, proposal: ProposalOutput, sampling: SamplingOutput, features: FeatureOutput, ctx: RunContext) -> tuple[CandidateRecord, ...]:
        from keyframe.scoring import assign_dwell_ids, promote_rescue_candidates, rescue_promotion_preflight_report

        candidates = proposal.candidates
        shortlist = proposal.rescue_shortlist
        if not shortlist:
            promoted = tuple(sorted(candidates, key=lambda c: c.timestamp))
            ctx.trace.exit("selection.promoted_rescue_only", _candidate_batch("selection.promoted_rescue_only", []))
            ctx.trace.exit(self.name, _candidate_batch(self.name, promoted))
            return promoted
        dwell_ids = assign_dwell_ids(features.dhashes)
        candidates = tuple(
            cand.with_temporal(dwell_id=dwell_ids[int(cand.sample_idx)])
            if 0 <= int(cand.sample_idx) < len(dwell_ids)
            else cand
            for cand in candidates
        )
        shortlist = tuple(
            row.with_temporal(dwell_id=dwell_ids[int(row.sample_idx)])
            if 0 <= int(row.sample_idx) < len(dwell_ids)
            else row
            for row in shortlist
        )
        promoted = promote_rescue_candidates(
            candidates,
            shortlist,
            dwell_ids,
            rescue_budget=proposal.rescue_budget,
            clip_embeddings=features.clip_embeddings,
        )
        if ctx.config.verbose_trace or ctx.config.debug_qa_targets_path is not None:
            preflight = rescue_promotion_preflight_report(
                candidates,
                shortlist,
                promoted,
                dwell_ids,
                proposal.rescue_budget,
                features.clip_embeddings,
            )
            ctx.trace.decision(
                "selection.rescue_promotion_preflight",
                "promotion_preflight",
                preflight,
            )
        rescue_count = sum(1 for cand in promoted if cand.selection.rescue_origin)
        print(f"  Rescue promotion: {len(candidates)} -> {len(promoted)} candidates ({rescue_count} promoted)")
        promoted_rescue_only = [cand for cand in promoted if cand.selection.rescue_origin]
        ctx.trace.exit(
            "selection.promoted_rescue_only",
            _candidate_batch("selection.promoted_rescue_only", promoted_rescue_only),
        )
        ctx.trace.exit(self.name, _candidate_batch(self.name, promoted))
        return promoted


class FinalEvidenceStage:
    name = "evidence.final_ocr"

    def __init__(self, preloader: Any, device: str):
        self.preloader = preloader
        self.device = device

    def run(self, candidates: tuple[CandidateRecord, ...], sampling: SamplingOutput, ctx: RunContext) -> tuple[CandidateRecord, ...]:
        from keyframe.dedupe import clean_ocr_token_sets
        from keyframe.frames import (
            _build_hybrid_captions,
            _build_ocr_token_sets,
            _filter_ocr_tokens,
            attach_ocr_token_attribution,
            caption_candidates,
            ocr_candidates,
        )

        frames = sampling.frame_store.frames
        florence_captions, candidates = caption_candidates(
            candidates, frames, device=self.device, preloaded=self.preloader.get_florence()
        )
        self.preloader.release_florence()

        ocr_texts, candidates = ocr_candidates(
            candidates, frames, preloaded_engine=self.preloader.get_ocr_engine()
        )
        self.preloader.release_ocr_engine()

        filtered_ocr = _filter_ocr_tokens(ocr_texts)
        _florence_caps, _ocr_caps, _has_ocr, candidates = _build_hybrid_captions(
            filtered_ocr, florence_captions, candidates
        )
        ocr_token_sets = _build_ocr_token_sets(filtered_ocr)
        ocr_token_sets = clean_ocr_token_sets(ocr_token_sets)
        candidates = attach_ocr_token_attribution(candidates, ocr_texts, filtered_ocr, ocr_token_sets)
        ctx.trace.exit(self.name, _candidate_batch(self.name, candidates))
        return candidates


class SelectionStage:
    name = "selection.retained_after_alt"

    def run(self, candidates: tuple[CandidateRecord, ...], ctx: RunContext) -> tuple[CandidateRecord, ...]:
        from keyframe.dedupe import retain_cluster_alternates

        retained = retain_cluster_alternates(candidates)
        print(f"  Cluster alternate retention: {len(candidates)} -> {len(retained)} candidates")
        ctx.trace.exit(self.name, _candidate_batch(self.name, retained))
        return retained


class SurvivalStage:
    name = "survival"

    def run(
        self,
        candidates: list[dict[str, Any]],
        sampling: SamplingOutput,
        features: FeatureOutput,
        ctx: RunContext,
    ) -> tuple[CandidateRecord, ...]:
        from keyframe.dedupe import (
            adjacent_same_screen_dedupe,
            content_area_duplicate_veto,
            filter_low_information_candidates,
            global_candidate_dedupe,
            near_time_dedupe,
        )
        from keyframe.merge import union_find_merge

        frames = sampling.frame_store.frames
        dhashes = features.dhashes
        deduped = near_time_dedupe(candidates, [set(c.evidence.ocr_tokens) for c in candidates], dhashes)
        print(f"  Near-time dedupe: {len(candidates)} -> {len(deduped)} candidates")
        ctx.trace.exit("survival.after_near_dedupe", _candidate_batch("survival.after_near_dedupe", deduped))

        deduped_token_sets = [set(c.evidence.ocr_tokens) for c in deduped]
        globally_deduped = global_candidate_dedupe(deduped, deduped_token_sets, dhashes)
        print(f"  Global conservative dedupe: {len(deduped)} -> {len(globally_deduped)} candidates")
        ctx.trace.exit("survival.after_global_dedupe", _candidate_batch("survival.after_global_dedupe", globally_deduped))

        filtered = filter_low_information_candidates(globally_deduped, frames)
        print(f"  Low-information filter: {len(globally_deduped)} -> {len(filtered)} candidates")
        ctx.trace.exit("survival.after_low_info_filter", _candidate_batch("survival.after_low_info_filter", filtered))

        adjacent_deduped = adjacent_same_screen_dedupe(filtered)
        print(f"  Adjacent same-screen dedupe: {len(filtered)} -> {len(adjacent_deduped)} candidates")
        ctx.trace.exit("survival.after_adjacent_dedupe", _candidate_batch("survival.after_adjacent_dedupe", adjacent_deduped))

        deduped_token_sets = [set(c.evidence.ocr_tokens) for c in adjacent_deduped]
        deduped_has_ocr = [len(tokens) >= 3 for tokens in deduped_token_sets]
        final = union_find_merge(adjacent_deduped, deduped_token_sets, deduped_has_ocr, frames)
        post_veto, dropped_duplicates = content_area_duplicate_veto(final, frames)
        print(f"  Content-area duplicate veto: {len(final)} -> {len(post_veto)} candidates")
        ctx.trace.exit(
            "survival.after_content_area_veto",
            _candidate_batch(
                "survival.after_content_area_veto",
                post_veto,
                {"dropped_duplicates": dropped_duplicates},
            ),
        )
        ctx.trace.exit("survival.final_pre_cap", _candidate_batch("survival.final_pre_cap", post_veto))

        capped = post_veto
        cap_drops: list[dict[str, Any]] = []
        max_output_frames = ctx.config.max_output_frames
        if max_output_frames is not None and max_output_frames >= 0 and len(post_veto) > max_output_frames:
            ranked = sorted(
                post_veto,
                key=lambda cand: (
                    len(cand.evidence.ocr_tokens),
                    float(cand.selection.candidate_score or cand.selection.score or cand.visual.sharpness or 0.0),
                    -float(cand.timestamp),
                    -int(cand.sample_idx),
                ),
                reverse=True,
            )
            kept_idxs = {int(cand.sample_idx) for cand in ranked[:max_output_frames]}
            capped = tuple(cand for cand in post_veto if int(cand.sample_idx) in kept_idxs)
            cap_drops = [
                {
                    "sample_idx": int(cand.sample_idx),
                    "timestamp": float(cand.timestamp),
                    "reason": "max_output_frames",
                }
                for cand in post_veto
                if int(cand.sample_idx) not in kept_idxs
            ]
            print(f"  Output cap: {len(post_veto)} -> {len(capped)} candidates")
        cap_pressure = (
            max(0, len(post_veto) - int(max_output_frames))
            if max_output_frames is not None and max_output_frames >= 0
            else 0
        )
        ctx.metadata["survival"] = {
            "max_output_frames": max_output_frames,
            "cap_pressure": cap_pressure,
            "cap_dropped_frames": cap_drops,
            "content_area_duplicate_drops": dropped_duplicates,
        }
        ctx.trace.exit(
            "survival.final_post_cap",
            _candidate_batch(
                "survival.final_post_cap",
                capped,
                {
                    "max_output_frames": max_output_frames,
                    "cap_pressure": cap_pressure,
                    "cap_dropped_frames": cap_drops,
                    "content_area_duplicate_drops": dropped_duplicates,
                },
            ),
        )
        return capped


class OutputStage:
    name = "output"

    def run(
        self,
        final: tuple[CandidateRecord, ...],
        sampling: SamplingOutput,
        temporal: TemporalOutput,
        output_dir: Path,
        rescue_budget: int,
        rescue_ocr_cap: int,
        temporal_window_count: int,
        scene_count: int,
        legacy_proxy_dropped_count: int,
        pre_rescue_candidate_count: int,
        ctx: RunContext,
    ) -> OutputArtifacts:
        from keyframe.frames import save_results
        from keyframe.manifest import write_manifest

        frames = sampling.frame_store.frames
        selected_imgs = {cand.sample_idx: frames[cand.sample_idx] for cand in final}
        caption_log_path = save_results(final, selected_imgs, output_dir)
        manifest_metadata = {
            "scene_coalescence": temporal.scene_coalescence,
            "output_cap": {
                "max_output_frames": ctx.config.max_output_frames,
                "applied": ctx.config.max_output_frames is not None,
                "cap_pressure": (ctx.metadata.get("survival") or {}).get("cap_pressure")
                if isinstance(ctx.metadata.get("survival"), dict)
                else 0,
                "dropped_frames": (ctx.metadata.get("survival") or {}).get("cap_dropped_frames", [])
                if isinstance(ctx.metadata.get("survival"), dict)
                else [],
                "content_area_duplicate_drops": (ctx.metadata.get("survival") or {}).get("content_area_duplicate_drops", [])
                if isinstance(ctx.metadata.get("survival"), dict)
                else [],
            },
            "rescue": {
                "pre_rescue_candidate_count": pre_rescue_candidate_count,
                "rescue_budget": rescue_budget,
                "rescue_ocr_cap": rescue_ocr_cap,
                "temporal_window_count": temporal_window_count,
                "scene_count": scene_count,
                "legacy_proxy_dropped_count": legacy_proxy_dropped_count,
            },
        }
        manifest_rows = [
            candidate_to_manifest_row(
                cand,
                filename=f"frame_{cand.frame_idx:06d}_{cand.timestamp:.2f}s.png",
            )
            for cand in final
        ]
        manifest_path = write_manifest(manifest_rows, output_dir, metadata=manifest_metadata)
        ctx.trace.exit("output.write_manifest", {
            "caption_log_path": str(caption_log_path),
            "manifest_path": str(manifest_path),
            "final_frame_count": len(final),
        })
        return OutputArtifacts(
            caption_log_path=Path(caption_log_path),
            manifest_path=Path(manifest_path),
            manifest_metadata=manifest_metadata,
        )


def _build_trace_sink(config: KeyframeExtractionConfig, trace_sink: TraceSink | None) -> TraceSink:
    if trace_sink is not None:
        return trace_sink
    if config.verbose_trace or config.debug_qa_targets_path is not None:
        return SnapshotTraceSink()
    return NoOpTraceSink()


def extract_keyframes(
    video_path: str | Path,
    output_dir: str | Path,
    config: KeyframeExtractionConfig | None = None,
    *,
    trace_sink: TraceSink | None = None,
) -> KeyframeExtractionResult:
    """Run the shared key-frame extraction pipeline and write frame artifacts."""
    from keyframe.frames import ModelPreloader
    from keyframe.visual import laplacian_sharpness

    cfg = config or KeyframeExtractionConfig()
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    device = cfg.device or _default_device()
    trace = _build_trace_sink(cfg, trace_sink)
    ctx = make_context(cfg, trace)

    if cfg.similarity_threshold != 0.85:
        print(
            "Warning: --similarity-threshold is deprecated and currently ignored; "
            "deterministic merge vetoes are used instead.",
            file=sys.stderr,
        )

    preloader = ModelPreloader(device=device, need_florence=True, need_ocr=True)
    t0 = time.time()
    pipeline_trace_path: Path | None = None
    debug_qa_trace_path: Path | None = None

    try:
        sampling = SamplingStage().run(video_path, ctx)
        features = FeatureStage(preloader, device).run(sampling, ctx)
        temporal = TemporalStage().run(video_path, sampling, features, ctx)
        proposal = ProposalStage().run(sampling, features, temporal, ctx)
        pre_rescue_candidate_count = len(proposal.candidates)

        RescueEvidenceStage(preloader).run(proposal, sampling, ctx)
        candidates = RescueSelectionStage().run(proposal, sampling, features, ctx)
        post_rescue_candidate_count = len(candidates)
        candidates = tuple(
            cand.with_visual(
                dhash=(
                    cand.visual.dhash
                    if cand.visual.dhash is not None
                    else features.dhashes[cand.sample_idx]
                ),
                dhash_hex=(
                    cand.visual.dhash_hex
                    if cand.visual.dhash_hex is not None
                    else f"{features.dhashes[cand.sample_idx]:016x}"
                ),
                sharpness=(
                    cand.visual.sharpness
                    if cand.visual.sharpness is not None
                    else float(laplacian_sharpness(sampling.frame_store.frames[cand.sample_idx]))
                ),
            )
            for cand in candidates
        )

        candidates = FinalEvidenceStage(preloader, device).run(candidates, sampling, ctx)
        retained = SelectionStage().run(candidates, ctx)
        final = SurvivalStage().run(retained, sampling, features, ctx)

        artifacts = OutputStage().run(
            final,
            sampling,
            temporal,
            output_dir,
            proposal.rescue_budget,
            proposal.rescue_ocr_cap,
            proposal.temporal_window_count,
            proposal.scene_count,
            proposal.legacy_proxy_dropped_count,
            pre_rescue_candidate_count,
            ctx,
        )

        if isinstance(trace, SnapshotTraceSink):
            if cfg.verbose_trace:
                pipeline_trace_path = output_dir / "pipeline_trace.json"
                trace.write(pipeline_trace_path)
            if cfg.debug_qa_targets_path is not None:
                debug_qa_trace_path = write_debug_qa_trace(
                    trace_records=trace.records,
                    targets_path=cfg.debug_qa_targets_path,
                    video=str(video_path),
                    output_path=output_dir / "debug_qa_trace.json",
                )

        elapsed = time.time() - t0
        print(f"\n{'='*60}")
        print(f"Frame extraction done in {elapsed:.1f}s")
        print(f"Pass 1: {sampling.samples.sample_count} sampled -> {post_rescue_candidate_count} CLIP candidates")
        print(f"Pass 2: {post_rescue_candidate_count} captioned -> {len(final)} final frames")
        print(f"Saved to: {Path(output_dir).resolve()}")
        print(f"Caption log: {artifacts.caption_log_path}")
        print(f"Manifest: {artifacts.manifest_path}")
        if pipeline_trace_path:
            print(f"Pipeline trace: {pipeline_trace_path}")
        if debug_qa_trace_path:
            print(f"Debug QA trace: {debug_qa_trace_path}")

        return KeyframeExtractionResult(
            final=final,
            output_dir=Path(output_dir),
            caption_log_path=artifacts.caption_log_path,
            manifest_path=artifacts.manifest_path,
            manifest_metadata=artifacts.manifest_metadata,
            sampled_frame_count=sampling.samples.sample_count,
            pre_rescue_candidate_count=pre_rescue_candidate_count,
            post_rescue_candidate_count=post_rescue_candidate_count,
            final_frame_count=len(final),
            pipeline_trace_path=pipeline_trace_path,
            debug_qa_trace_path=debug_qa_trace_path,
        )
    finally:
        preloader.shutdown()
        gc.collect()
        _empty_cache(device)
