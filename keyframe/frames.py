#!/usr/bin/env python3
"""
extract_florence.py

Two-pass key frame extraction using unified CLIP embedding space:
  Pass 1 (CLIP): Sample all frames, embed with CLIP, over-segment into ~15-20
          clusters, pick one candidate per cluster.
  Pass 2 (Florence-2 + CLIP text): Caption candidates with Florence-2, embed
          captions with CLIP's text encoder (same space as image embeddings),
          merge candidates whose captions are too similar.

Usage:
    python extract_florence.py input.mp4 [--output-dir frames] [--sample-interval 0.5]

Dependencies:
    pip install opencv-python torch transformers open-clip-torch scikit-learn numpy pillow
"""

import cv2
import gc
import numpy as np
import os
import re
import torch
from collections import Counter
from PIL import Image
from pathlib import Path
from transformers import Florence2ForConditionalGeneration, AutoProcessor
import open_clip
from sklearn.cluster import AgglomerativeClustering
import argparse
import sys
import time
import json

from keyframe.dedupe import (
    adjacent_same_screen_dedupe,
    clean_ocr_token_sets,
    compute_dhash,
    filter_low_information_candidates,
    global_candidate_dedupe,
    hamming,
    near_time_dedupe,
    retain_cluster_alternates,
)
from keyframe.manifest import write_manifest
from keyframe.merge import (
    build_ocr_token_sets,
    jaccard_sim_matrix,
    jaccard_similarity,
    union_find_merge,
)
from keyframe.scoring import (
    allocate_clusters_by_novelty,
    assign_dwell_ids,
    assign_temporal_window_ids,
    build_rescue_shortlist,
    candidate_budget_for_scenes,
    coalesce_tiny_scenes,
    promote_rescue_candidates,
    rescue_window_seconds,
    score_candidate_for_rep,
)


# ── Video sampling ──────────────────────────────────────────────────────────

def sample_frames(video_path, interval_seconds=0.5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: could not open {video_path}", file=sys.stderr)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"Video: {video_path}")
    print(f"  {width}x{height}, {fps:.1f} fps, {total_frames} frames, {duration:.1f}s")

    interval_frames = max(1, int(interval_seconds * fps))
    frames, timestamps, frame_indices = [], [], []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval_frames == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
            timestamps.append(frame_idx / fps)
            frame_indices.append(frame_idx)
        frame_idx += 1

    cap.release()
    print(f"  Sampled {len(frames)} frames at {interval_seconds}s intervals")
    return frames, timestamps, frame_indices


# ── Scene detection ──────────────────────────────────────────────────────

def detect_scenes(video_path, timestamps, threshold=27.0):
    """Run pySceneDetect ContentDetector and return scene boundaries as
    (start_idx, end_idx) tuples indexed into the timestamps/frames arrays."""
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector
    import bisect

    print(f"\n── Scene detection (ContentDetector, threshold={threshold}) ──")
    video = open_video(video_path)
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold))
    sm.detect_scenes(video)
    scene_list = sm.get_scene_list()

    if not scene_list:
        print(f"  No scene cuts detected — treating entire video as one scene")
        return [(0, len(timestamps) - 1)]

    scenes = []
    for start_tc, end_tc in scene_list:
        start_sec = start_tc.get_seconds()
        end_sec = end_tc.get_seconds()
        s_idx = bisect.bisect_left(timestamps, start_sec)
        e_idx = bisect.bisect_right(timestamps, end_sec) - 1
        s_idx = max(0, min(s_idx, len(timestamps) - 1))
        e_idx = max(s_idx, min(e_idx, len(timestamps) - 1))
        scenes.append((s_idx, e_idx))

    print(f"  Detected {len(scenes)} scenes:")
    for i, (s, e) in enumerate(scenes):
        print(f"    Scene {i}: frames {s}-{e} "
              f"({timestamps[s]:.1f}s - {timestamps[e]:.1f}s, "
              f"{e - s + 1} frames)")
    return scenes


def _allocate_clusters(scenes, total_clusters, min_per_scene=2):
    """Deprecated duration allocator retained for compatibility."""
    total_frames = sum(e - s + 1 for s, e in scenes)
    if total_frames == 0:
        return [min_per_scene] * len(scenes)

    allocs = []
    for s, e in scenes:
        scene_len = e - s + 1
        share = max(min_per_scene, round(total_clusters * scene_len / total_frames))
        allocs.append(share)
    return allocs


# ── Parallel model preloading ─────────────────────────────────────────────

def _load_clip(device, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"):
    """Load CLIP model (called in background thread)."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, preprocess, tokenizer


def _florence_dtype(device):
    """Pick Florence-2 dtype based on device.

    fp16 on CUDA/MPS halves the model footprint vs fp32. CPU stays fp32 because
    fp16 CPU kernels are slow and not always supported.
    """
    dev = str(device)
    if dev.startswith("cuda") or dev == "mps":
        return torch.float16
    return torch.float32


def _load_florence(device):
    """Load Florence-2 model + processor (called in background thread)."""
    model_name = "florence-community/Florence-2-base"
    processor = AutoProcessor.from_pretrained(model_name)
    model = Florence2ForConditionalGeneration.from_pretrained(
        model_name, dtype=_florence_dtype(device),
    ).to(device)
    model.eval()
    return model, processor


def _load_ocr_engine():
    """Load PaddleOCR engine for non-macOS (called in background thread)."""
    os.environ["FLAGS_use_mkldnn"] = "0"
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    from paddleocr import PaddleOCR
    return PaddleOCR(
        lang='en',
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def _empty_device_cache(device):
    dev = str(device)
    if dev == "mps":
        torch.mps.empty_cache()
    elif dev.startswith("cuda"):
        torch.cuda.empty_cache()


class ModelPreloader:
    """Lazily loads each model on first access and releases it on demand.

    Each `get_*` loads its model synchronously the first time and caches it.
    Each `release_*` drops the cached model, runs a GC pass, and frees device
    memory. This keeps peak RAM low because callers can free CLIP before
    Florence loads, and Florence before OCR loads.
    """

    def __init__(self, device="mps", need_florence=True, need_ocr=True):
        self._device = device
        self._need_florence = need_florence
        self._need_ocr = need_ocr and not _is_macos()

        self._clip = None
        self._florence = None
        self._ocr = None

    def get_clip(self):
        if self._clip is None:
            print("  Loading CLIP...")
            self._clip = _load_clip(self._device)
        return self._clip

    def get_florence(self):
        if not self._need_florence:
            raise RuntimeError("Florence-2 was not requested")
        if self._florence is None:
            print("  Loading Florence-2...")
            self._florence = _load_florence(self._device)
        return self._florence

    def get_ocr_engine(self):
        if not self._need_ocr:
            return None
        if self._ocr is None:
            print("  Loading PaddleOCR...")
            self._ocr = _load_ocr_engine()
        return self._ocr

    def release_clip(self):
        if self._clip is not None:
            self._clip = None
            gc.collect()
            _empty_device_cache(self._device)

    def release_florence(self):
        if self._florence is not None:
            self._florence = None
            gc.collect()
            _empty_device_cache(self._device)

    def release_ocr_engine(self):
        if self._ocr is not None:
            self._ocr = None
            gc.collect()

    def shutdown(self):
        self.release_clip()
        self.release_florence()
        self.release_ocr_engine()


# ── CLIP model (shared across passes) ──────────────────────────────────────

class CLIPEncoder:
    """Wraps CLIP model for both image and text embedding in a single space."""

    def __init__(self, device="mps", model_name="ViT-B-32", pretrained="laion2b_s34b_b79k",
                 preloaded=None):
        self.device = device
        if preloaded:
            print(f"  Using preloaded CLIP ({model_name})...")
            self.model, self.preprocess, self.tokenizer = preloaded
        else:
            print(f"  Loading CLIP ({model_name})...")
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=device
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)
            self.model.eval()

    def embed_images(self, frames, batch_size=32):
        all_emb = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            images = torch.stack([self.preprocess(img) for img in batch]).to(self.device)
            with torch.no_grad(), torch.amp.autocast(device_type=str(self.device)):
                features = self.model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)
            all_emb.append(features.float().cpu().numpy())
        return np.vstack(all_emb).astype(np.float32)

    def embed_texts(self, texts):
        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad(), torch.amp.autocast(device_type=str(self.device)):
            features = self.model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.float().cpu().numpy().astype(np.float32)

    def cleanup(self):
        del self.model, self.tokenizer
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()


# ── Pass 1: CLIP image embedding + over-segmentation ───────────────────────

def _laplacian_sharpness(pil_img):
    """Score frame sharpness via Laplacian variance (higher = sharper)."""
    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def clip_oversegment(
    embeddings,
    timestamps,
    frame_indices,
    n_clusters,
    frames,
    transcript_density=None,
    dhashes=None,
    max_reps_per_cluster=1,
    alt_dhash_threshold=12,
    alt_clip_distance_threshold=0.08,
    alt_sharpness_ratio_floor=0.5,
    return_labels=False,
):
    print(f"  Over-segmenting into {n_clusters} clusters...")
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters, metric="cosine", linkage="average",
    )
    labels = clustering.fit_predict(embeddings)

    candidates = []
    for cid in range(n_clusters):
        mask = labels == cid
        idxs = np.where(mask)[0]

        scored = []
        for idx in idxs:
            sharpness = _laplacian_sharpness(frames[idx])
            if dhashes is not None and idx < len(dhashes):
                left = dhashes[idx - 1] if idx > 0 else dhashes[idx]
                right = dhashes[idx + 1] if idx + 1 < len(dhashes) else dhashes[idx]
                dwell_bonus = 1.0 if max(hamming(dhashes[idx], left), hamming(dhashes[idx], right)) <= 6 else 0.0
            else:
                dwell_bonus = 0.0
            cand = {
                "sample_idx": int(idx),
                "timestamp": timestamps[idx],
                "sharpness": sharpness,
                "end_of_dwell_bonus": dwell_bonus,
            }
            density = 0.0 if transcript_density is None else transcript_density.get(idx, 0.0)
            scored.append((score_candidate_for_rep(cand, frames[idx], density, dwell_bonus), idx, sharpness))
        best_score, best, best_sharpness = max(scored)

        selected = [{
            "sample_idx": int(best),
            "frame_idx": frame_indices[best],
            "timestamp": timestamps[best],
            "clip_cluster": cid,
            "clip_cluster_size": int(mask.sum()),
            "sharpness": float(best_sharpness),
            "candidate_score": float(best_score),
        }]

        if max_reps_per_cluster >= 2 and len(scored) > 1:
            alt_choices = []
            best_embedding = embeddings[best]
            best_hash = dhashes[best] if dhashes is not None and best < len(dhashes) else None
            for score, idx, sharpness in scored:
                if idx == best:
                    continue
                sharpness_ratio = float(sharpness) / max(float(best_sharpness), 1e-9)
                if sharpness_ratio < alt_sharpness_ratio_floor:
                    continue
                clip_distance = float(1.0 - np.dot(best_embedding, embeddings[idx]))
                dhash_distance = None
                if best_hash is not None and dhashes is not None and idx < len(dhashes):
                    dhash_distance = hamming(best_hash, dhashes[idx])
                diverse_by_hash = dhash_distance is not None and dhash_distance >= alt_dhash_threshold
                diverse_by_clip = clip_distance > alt_clip_distance_threshold
                if not (diverse_by_hash or diverse_by_clip):
                    continue
                if diverse_by_hash:
                    reason = "dhash"
                    diversity_distance = float(dhash_distance)
                else:
                    reason = "clip"
                    diversity_distance = clip_distance
                alt_choices.append((score, idx, sharpness, reason, diversity_distance, sharpness_ratio))

            if alt_choices:
                alt_score, alt, alt_sharpness, reason, distance, ratio = max(alt_choices)
                selected[0]["cluster_role"] = "primary"
                selected.append({
                    "sample_idx": int(alt),
                    "frame_idx": frame_indices[alt],
                    "timestamp": timestamps[alt],
                    "clip_cluster": cid,
                    "clip_cluster_size": int(mask.sum()),
                    "sharpness": float(alt_sharpness),
                    "candidate_score": float(alt_score),
                    "cluster_role": "alt",
                    "cluster_alt_reason": reason,
                    "cluster_diversity_distance": float(distance),
                    "cluster_sharpness_ratio": float(ratio),
                })
            else:
                selected[0]["cluster_role"] = "single"
        else:
            selected[0]["cluster_role"] = "single"

        candidates.extend(selected)

    candidates.sort(key=lambda c: c["timestamp"])
    print(f"  → {len(candidates)} candidates")
    for c in candidates:
        print(f"    {c['timestamp']:5.1f}s  (cluster {c['clip_cluster']}, "
              f"{c['clip_cluster_size']} frames)")
    if return_labels:
        return candidates, labels
    return candidates


# ── Pass 2: Florence-2 captioning ──────────────────────────────────────────

def caption_candidates(candidates, frames, device="mps", preloaded=None):
    print(f"\n── Pass 2: Florence-2 captioning ({len(candidates)} frames) ──")

    if preloaded:
        print("  Using preloaded Florence-2...")
        model, processor = preloaded
    else:
        model_name = "florence-community/Florence-2-base"
        processor = AutoProcessor.from_pretrained(model_name)
        model = Florence2ForConditionalGeneration.from_pretrained(
            model_name, dtype=_florence_dtype(device),
        ).to(device)
        model.eval()

    prompt = "<MORE_DETAILED_CAPTION>"

    candidate_images = [frames[c["sample_idx"]] for c in candidates]
    batch_inputs = processor(
        text=[prompt] * len(candidates),
        images=candidate_images,
        return_tensors="pt",
        padding=True,
    ).to(device)

    # Cast pixel inputs to the model's compute dtype (fp16 on CUDA/MPS).
    model_dtype = next(model.parameters()).dtype
    if "pixel_values" in batch_inputs and batch_inputs["pixel_values"].dtype != model_dtype:
        batch_inputs["pixel_values"] = batch_inputs["pixel_values"].to(model_dtype)

    with torch.no_grad():
        generated_ids = model.generate(
            **batch_inputs,
            max_new_tokens=80,
            do_sample=False,
        )

    captions = []
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=False)
    for i, (cand, text) in enumerate(zip(candidates, generated_texts)):
        img = candidate_images[i]
        parsed = processor.post_process_generation(
            text, task=prompt, image_size=(img.width, img.height)
        )
        caption = parsed.get(prompt, text)
        captions.append(caption)
        cand["caption"] = caption
        print(f"  [{i+1}/{len(candidates)}] {cand['timestamp']:5.1f}s → \"{caption[:120]}\"")

    del model, processor
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
    return captions


# ── Pass 2b: PaddleOCR text extraction ───────────────────────────────────



def _ocr_apple_vision(img):
    """Use macOS Vision framework for hardware-accelerated OCR."""
    import Vision
    from Foundation import NSURL
    from Quartz import CIImage
    import tempfile

    # Vision needs a file path or CIImage — write temp PNG
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img.save(f, format="PNG")
        tmp_path = f.name

    try:
        img_url = NSURL.fileURLWithPath_(tmp_path)
        ci_image = CIImage.imageWithContentsOfURL_(img_url)

        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(False)

        handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(
            ci_image, None
        )
        success, error = handler.performRequests_error_([request], None)
        if not success:
            return []

        lines = []
        for obs in request.results():
            text = obs.topCandidates_(1)[0].string()
            conf = obs.confidence()
            if conf > 0.5:
                lines.append(text)
        return lines
    finally:
        os.unlink(tmp_path)


def _ocr_paddle(img, ocr_engine):
    """Use PaddleOCR (CPU fallback for non-macOS)."""
    img_array = np.array(img)
    result = ocr_engine.predict(img_array)

    lines = []
    for item in result:
        rec_texts = item.get('rec_texts', [])
        rec_scores = item.get('rec_scores', [])
        for text, score in zip(rec_texts, rec_scores):
            if score > 0.7:
                lines.append(text)
    return lines


def _is_macos():
    import platform
    return platform.system() == "Darwin"


def ocr_candidates(candidates, frames, preloaded_engine=None):
    """Run OCR on each candidate frame. Uses Apple Vision on macOS, PaddleOCR elsewhere."""
    use_apple = _is_macos()
    backend = "Apple Vision" if use_apple else "PaddleOCR"
    print(f"\n── Pass 2b: OCR text extraction ({len(candidates)} frames, {backend}) ──")

    need_ocr = any("ocr_text" not in cand or cand["ocr_text"] is None for cand in candidates)
    paddle_engine = None
    if not use_apple and need_ocr:
        if preloaded_engine:
            paddle_engine = preloaded_engine
        else:
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            from paddleocr import PaddleOCR
            paddle_engine = PaddleOCR(
                lang='en',
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )

    ocr_texts = []
    for i, cand in enumerate(candidates):
        if "ocr_text" in cand and cand["ocr_text"] is not None:
            raw_text = str(cand["ocr_text"])
            ocr_texts.append(raw_text)
            preview = raw_text[:120] if raw_text else "(no text)"
            print(f"  [{i+1}/{len(candidates)}] {cand['timestamp']:5.1f}s → "
                  f"cached, {len(raw_text)} chars: \"{preview}\"")
            continue

        img = frames[cand["sample_idx"]]
        lines = _ocr_apple_vision(img) if use_apple else _ocr_paddle(img, paddle_engine)

        raw_text = " ".join(lines)
        ocr_texts.append(raw_text)
        cand["ocr_text"] = raw_text
        preview = raw_text[:120] if raw_text else "(no text)"
        print(f"  [{i+1}/{len(candidates)}] {cand['timestamp']:5.1f}s → "
              f"{len(lines)} lines, {len(raw_text)} chars: \"{preview}\"")

    return ocr_texts


def _normalize_token(word):
    """Lowercase and strip punctuation for DF comparison."""
    return re.sub(r'[^\w\s]', '', word.lower())


def _filter_ocr_tokens(all_ocr_texts, max_tokens=70, df_cutoff=0.5):
    """Drop words that appear in >df_cutoff fraction of frames (UI chrome), keep the rest."""
    n = len(all_ocr_texts)
    if n == 0:
        return all_ocr_texts

    # Build normalized token sets per frame for DF calculation
    tokenized = []
    for text in all_ocr_texts:
        normed = {_normalize_token(w) for w in text.split() if _normalize_token(w)}
        tokenized.append(normed)

    df = Counter()
    for tokens in tokenized:
        df.update(tokens)

    filtered = []
    for text in all_ocr_texts:
        words = text.split()
        kept = [w for w in words if df[_normalize_token(w)] / n <= df_cutoff]
        filtered.append(" ".join(kept[:max_tokens]))
    return filtered


def _build_hybrid_captions(filtered_ocr, florence_captions, candidates,
                           min_ocr_chars=30):
    """Tag each candidate with its caption source and log the hybrid decision.

    Returns (florence_captions, ocr_captions, has_ocr) — the merge step
    embeds Florence-2 and OCR separately to avoid CLIP's 77-token limit.
    """
    print(f"\n── Hybrid caption analysis ──")
    has_ocr = []
    for i, (ocr, flor) in enumerate(zip(filtered_ocr, florence_captions)):
        has = len(ocr) >= min_ocr_chars
        has_ocr.append(has)
        tag = "both" if has else "florence"
        candidates[i]["caption_source"] = tag
        display = f"{flor[:60]} + OCR({len(ocr)}ch)" if has else flor[:100]
        print(f"  [{i+1}/{len(filtered_ocr)}] {candidates[i]['timestamp']:5.1f}s → "
              f"[{tag}] \"{display}\"")
    n_both = sum(has_ocr)
    print(f"  → {n_both} with OCR signal, {len(has_ocr) - n_both} Florence-2 only")
    return florence_captions, filtered_ocr, has_ocr


# ── Merge: CLIP image similarity + Jaccard OCR overlap ───────────────────

def _jaccard_similarity(tokens_a, tokens_b):
    """Jaccard similarity between two sets of normalized tokens."""
    return jaccard_similarity(tokens_a, tokens_b)


def _jaccard_sim_matrix(token_sets, has_ocr_mask):
    """Build a full Jaccard similarity matrix, vectorized over the OCR mask."""
    return jaccard_sim_matrix(token_sets, has_ocr_mask)


def _build_combined_sim(clip_sim, jaccard_sim, has_ocr_mask, ocr_weight):
    """Build combined similarity: weighted blend where both have OCR, else CLIP only."""
    both_ocr = np.outer(has_ocr_mask, has_ocr_mask)
    fw = 1.0 - ocr_weight
    combined = np.where(both_ocr,
                        fw * clip_sim + ocr_weight * jaccard_sim,
                        clip_sim)
    return combined, both_ocr


def _build_ocr_token_sets(filtered_ocr_texts):
    """Build normalized token sets from filtered OCR texts for Jaccard comparison."""
    return build_ocr_token_sets(filtered_ocr_texts, _normalize_token)


def _normalize_rescue_token(word):
    token = re.sub(r"^[^\w.-]+|[^\w.-]+$", "", word.lower())
    token = re.sub(r"[^\w.-]", "", token)
    return token


def _build_rescue_token_sets(raw_ocr_texts, max_tokens=120):
    """Build discovery-oriented tokens that preserve source/content references."""
    rescue_sets = []
    for text in raw_ocr_texts:
        tokens = []
        for word in text.split():
            token = _normalize_rescue_token(word)
            if not token:
                continue
            if len(token) > 48:
                continue
            if token.startswith(("http", "www")) or "users" in token or "downloads" in token:
                continue
            tokens.append(token)
        rescue_sets.append(set(tokens[:max_tokens]))
    return rescue_sets


def attach_ocr_token_attribution(candidates, raw_ocr_texts, filtered_ocr_texts, cleaned_token_sets):
    raw_token_sets = _build_ocr_token_sets(raw_ocr_texts)
    filtered_token_sets = _build_ocr_token_sets(filtered_ocr_texts)
    rescue_token_sets = _build_rescue_token_sets(raw_ocr_texts)
    for cand, raw_tokens, filtered_tokens, cleaned_tokens, rescue_tokens in zip(
        candidates, raw_token_sets, filtered_token_sets, cleaned_token_sets, rescue_token_sets
    ):
        raw_count = len(raw_tokens)
        filtered_count = len(filtered_tokens)
        cleaned_count = len(cleaned_tokens)
        cand["ocr_tokens"] = sorted(cleaned_tokens)
        cand["dedupe_tokens"] = sorted(cleaned_tokens)
        cand["rescue_tokens"] = sorted(rescue_tokens)
        cand["raw_token_count"] = raw_count
        cand["filtered_token_count"] = filtered_count
        cand["cleaned_token_count"] = cleaned_count
        cand["cleaning_attrition_ratio"] = (
            0.0 if raw_count == 0 else round((raw_count - cleaned_count) / raw_count, 4)
        )
        cand.setdefault("retention_reason", "none")
        cand.setdefault("retention_reasons_seen", [cand["retention_reason"]])
        if cand.get("cluster_role"):
            cand.setdefault("lineage_roles", [cand["cluster_role"]])


def attach_rescue_ocr_metadata(candidates, raw_ocr_texts):
    """Attach temporary split OCR tokens while preserving raw OCR as cache."""
    filtered_ocr = _filter_ocr_tokens(raw_ocr_texts)
    dedupe_token_sets = clean_ocr_token_sets(_build_ocr_token_sets(filtered_ocr))
    rescue_token_sets = _build_rescue_token_sets(raw_ocr_texts)
    for cand, raw_text, filtered_text, dedupe_tokens, rescue_tokens in zip(
        candidates, raw_ocr_texts, filtered_ocr, dedupe_token_sets, rescue_token_sets
    ):
        raw_tokens = _build_ocr_token_sets([raw_text])[0]
        filtered_tokens = _build_ocr_token_sets([filtered_text])[0]
        cand["ocr_text"] = raw_text
        cand.setdefault("ocr_cache_source", "rescue")
        cand["ocr_tokens"] = sorted(dedupe_tokens)
        cand["dedupe_tokens"] = sorted(dedupe_tokens)
        cand["rescue_tokens"] = sorted(rescue_tokens)
        cand["raw_token_count"] = len(raw_tokens)
        cand["filtered_token_count"] = len(filtered_tokens)
        cand["cleaned_token_count"] = len(dedupe_tokens)


def _comparison_primary_sample_idxs(candidates, shortlist):
    selected: set[int] = set()
    for rescue in shortlist:
        rescue_ts = float(rescue.get("timestamp", 0.0))
        rescue_cluster = rescue.get("clip_cluster")
        rescue_scene = rescue.get("scene_id")

        same_cluster = [
            cand for cand in candidates
            if rescue_cluster is not None and cand.get("clip_cluster") == rescue_cluster
        ]
        same_scene = [
            cand for cand in candidates
            if rescue_scene is not None and cand.get("scene_id") == rescue_scene
        ]
        for pool in (same_cluster, same_scene):
            if not pool:
                continue
            primary = min(
                pool,
                key=lambda cand: abs(float(cand.get("timestamp", 0.0)) - rescue_ts),
            )
            selected.add(int(primary["sample_idx"]))
    return selected


def rescue_from_sampled_frames(
    candidates,
    frames,
    timestamps,
    frame_indices,
    dhashes,
    clip_emb,
    pass1_clusters,
    *,
    sample_clusters=None,
    sample_scenes=None,
    preloaded_engine=None,
):
    window_seconds = rescue_window_seconds(timestamps)
    temporal_window_ids = assign_temporal_window_ids(
        timestamps,
        sample_scenes,
        window_seconds=window_seconds,
    )
    for cand in candidates:
        sample_idx = int(cand["sample_idx"])
        cand["proxy_content_score"] = float(cand.get("proxy_content_score", 0.0) or 0.0)
        if sample_scenes:
            cand["scene_id"] = sample_scenes.get(sample_idx)
        if sample_idx < len(temporal_window_ids):
            cand["temporal_window_id"] = temporal_window_ids[sample_idx]
            cand["temporal_window_seconds"] = window_seconds

    shortlist, proxy_rows, rescue_budget = build_rescue_shortlist(
        frames,
        timestamps,
        frame_indices,
        candidates,
        pass1_clusters,
        sample_clusters=sample_clusters,
        sample_scenes=sample_scenes,
    )
    for cand in candidates:
        sample_idx = int(cand["sample_idx"])
        if 0 <= sample_idx < len(proxy_rows):
            cand["proxy_content_score"] = float(proxy_rows[sample_idx]["proxy_content_score"])
    if not shortlist:
        print("  Rescue shortlist: 0 frames")
        return sorted(candidates, key=lambda c: c["timestamp"])

    for row in shortlist:
        sample_idx = int(row["sample_idx"])
        row["dhash"] = dhashes[sample_idx]
        row["dhash_hex"] = f"{dhashes[sample_idx]:016x}"
        row["sharpness"] = float(_laplacian_sharpness(frames[sample_idx]))

    print(f"  Rescue shortlist: {len(shortlist)} frames, budget {rescue_budget}")
    comparison_idxs = _comparison_primary_sample_idxs(candidates, shortlist)
    candidate_by_idx = {int(cand["sample_idx"]): cand for cand in candidates}
    comparison_primaries = [
        candidate_by_idx[sample_idx]
        for sample_idx in sorted(comparison_idxs)
        if sample_idx in candidate_by_idx
    ]
    ocr_batch = shortlist + comparison_primaries
    rescue_ocr_texts = ocr_candidates(ocr_batch, frames, preloaded_engine=preloaded_engine)
    attach_rescue_ocr_metadata(ocr_batch, rescue_ocr_texts)
    print(
        "  Rescue OCR comparison primaries: "
        f"{len(comparison_primaries)} cached selected candidates"
    )
    dwell_ids = assign_dwell_ids(dhashes)
    for cand in candidates:
        sample_idx = int(cand["sample_idx"])
        if 0 <= sample_idx < len(dwell_ids):
            cand["dwell_id"] = dwell_ids[sample_idx]
    for row in shortlist:
        sample_idx = int(row["sample_idx"])
        if 0 <= sample_idx < len(dwell_ids):
            row["dwell_id"] = dwell_ids[sample_idx]
    promoted = promote_rescue_candidates(
        candidates,
        shortlist,
        dwell_ids,
        rescue_budget=rescue_budget,
        clip_embeddings=clip_emb,
    )
    rescue_count = sum(1 for cand in promoted if cand.get("rescue_origin"))
    print(f"  Rescue promotion: {len(candidates)} -> {len(promoted)} candidates ({rescue_count} promoted)")
    return promoted


def merge_by_caption(candidates, clip_emb, ocr_token_sets, has_ocr, frames,
                     similarity_threshold=0.85, ocr_weight=0.5):
    """
    Merge candidates using CLIP image embeddings (from Pass 1) for semantic
    similarity and Jaccard token overlap for OCR deduplication.

    Args:
        clip_emb: Full CLIP image embedding array from Pass 1 (all sampled frames).
        ocr_token_sets: List of normalized token sets (one per candidate).
        has_ocr: List of bools — whether each candidate has substantial OCR.
        frames: List of PIL images (all sampled frames) for sharpness scoring.
        ocr_weight: Weight for Jaccard OCR similarity (0-1). CLIP image gets 1-ocr_weight.
    """
    print(f"\n── Merging via CLIP image + Jaccard OCR (threshold={similarity_threshold}) ──")

    # CLIP image similarity for candidates only
    candidate_indices = np.array([c["sample_idx"] for c in candidates])
    cand_emb = clip_emb[candidate_indices]
    clip_sim = cand_emb @ cand_emb.T
    print(f"  CLIP image similarity: {cand_emb.shape[0]} candidates")

    # Jaccard OCR similarity + combined matrix (vectorized)
    has_ocr_mask = np.array(has_ocr, dtype=bool)
    jaccard_sim = _jaccard_sim_matrix(ocr_token_sets, has_ocr_mask)
    combined_sim, both_ocr = _build_combined_sim(clip_sim, jaccard_sim, has_ocr_mask, ocr_weight)

    # Log merge decisions
    merge_i, merge_j = np.where(np.triu(combined_sim > similarity_threshold, k=1))
    for i, j in zip(merge_i, merge_j):
        detail = ""
        if both_ocr[i, j]:
            detail = (f" [CLIP:{clip_sim[i, j]:.3f} "
                      f"Jaccard:{jaccard_sim[i, j]:.3f}]")
        print(f"    {candidates[i]['timestamp']:5.1f}s ↔ "
              f"{candidates[j]['timestamp']:5.1f}s: "
              f"{combined_sim[i, j]:.3f}{detail} (will merge)")

    # Convert combined similarity to distance for clustering
    combined_dist = 1.0 - combined_sim
    np.fill_diagonal(combined_dist, 0)
    combined_dist = np.clip(combined_dist, 0, None)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=1 - similarity_threshold,
    )
    cap_labels = clustering.fit_predict(combined_dist)
    n_clusters = len(set(cap_labels))
    print(f"  Combined clusters: {n_clusters}")

    # ── Intra-group OCR refinement ──
    # For groups where members share OCR text, the global DF filter doesn't
    # remove within-group common tokens. Re-filter OCR within each group to
    # surface what actually differs, then re-cluster using Jaccard.
    n_refined = 0
    refined_labels = np.array(cap_labels, copy=True)
    next_label = max(cap_labels) + 1

    for cid in sorted(set(cap_labels)):
        group_idxs = [i for i, l in enumerate(cap_labels) if l == cid]
        if len(group_idxs) < 4:
            continue
        group_has_ocr = [has_ocr[i] for i in group_idxs]
        if sum(group_has_ocr) < len(group_idxs) * 0.5:
            continue

        # Reconstruct text from token sets for intra-group DF filtering
        group_ocr = [
            " ".join(ocr_token_sets[i]) if has_ocr[i] else ""
            for i in group_idxs
        ]
        local_filtered = _filter_ocr_tokens(group_ocr, max_tokens=70, df_cutoff=0.5)
        local_token_sets = _build_ocr_token_sets(local_filtered)

        non_empty = [s for s in local_token_sets if len(s) >= 3]
        if len(non_empty) < 2:
            continue

        # Recompute combined similarity with locally-filtered Jaccard
        gn = len(group_idxs)
        group_arr = np.array(group_idxs)
        local_clip = clip_sim[np.ix_(group_arr, group_arr)]
        local_has_ocr = np.array([len(s) >= 3 for s in local_token_sets])
        local_jaccard = _jaccard_sim_matrix(local_token_sets, local_has_ocr)
        local_combined, _ = _build_combined_sim(
            local_clip, local_jaccard, local_has_ocr, ocr_weight
        )

        local_dist = 1.0 - local_combined
        np.fill_diagonal(local_dist, 0)
        local_dist = np.clip(local_dist, 0, None)

        sub_clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",
            distance_threshold=1 - similarity_threshold,
        )
        sub_labels = sub_clustering.fit_predict(local_dist)
        n_sub = len(set(sub_labels))

        if n_sub > 1:
            n_refined += 1
            print(f"  Group {cid} refined: {gn} candidates → {n_sub} sub-groups "
                  f"(intra-group OCR filtering)")
            for gi, idx in enumerate(group_idxs):
                refined_labels[idx] = next_label + sub_labels[gi]
                filt_preview = " ".join(list(local_token_sets[gi])[:10]) if local_token_sets[gi] else "(no OCR)"
                print(f"    {candidates[idx]['timestamp']:5.1f}s → sub-group "
                      f"{sub_labels[gi]}: \"{filt_preview}\"")
            next_label += n_sub

    if n_refined:
        cap_labels = refined_labels
        n_clusters = len(set(cap_labels))
        print(f"  After refinement: {n_clusters} clusters")

    # ── Build final frame list ──
    final = []
    for cid in sorted(set(cap_labels)):
        group_idxs = [i for i, l in enumerate(cap_labels) if l == cid]
        group_cands = [candidates[i] for i in group_idxs]

        # Pick sharpest frame in group
        sharpness = [_laplacian_sharpness(frames[c["sample_idx"]]) for c in group_cands]
        best_local = np.argmax(sharpness)

        winner = group_cands[best_local].copy()
        winner["caption_cluster"] = int(cid)
        winner["merged_from"] = len(group_idxs)
        winner["merged_captions"] = [candidates[i].get("caption", "") for i in group_idxs]
        winner["merged_timestamps"] = [candidates[i]["timestamp"] for i in group_idxs]
        final.append(winner)

        merged_tag = "" if len(group_idxs) == 1 else f" (merged {len(group_idxs)} candidates)"
        print(f"  Group {cid}: kept {winner['timestamp']:.1f}s{merged_tag}")
        caption_preview = candidates[group_idxs[best_local]].get("caption", "")
        print(f"    \"{caption_preview[:120]}\"")
        if len(group_idxs) > 1:
            for i in group_idxs:
                if i != group_idxs[best_local]:
                    cap = candidates[i].get("caption", "")
                    print(f"    dropped {candidates[i]['timestamp']:.1f}s: "
                          f"\"{cap[:100]}\"")

    final.sort(key=lambda c: c["timestamp"])
    return final


# ── Output ──────────────────────────────────────────────────────────────────

def save_results(selected, frames, output_dir):
    """Save selected frames to disk.

    `frames` may be a list (indexed by sample_idx) or a mapping
    {sample_idx: PIL.Image} containing only the selected frames. The mapping
    form lets callers free the full frames list before saving.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for s in selected:
        ts = s["timestamp"]
        fidx = s["frame_idx"]
        path = out / f"frame_{fidx:06d}_{ts:.2f}s.png"
        frames[s["sample_idx"]].save(str(path))
        s["path"] = str(path)

    log = [{
        "file": Path(s["path"]).name,
        "timestamp": s["timestamp"],
        "caption": s["caption"],
        "caption_cluster": s["caption_cluster"],
        "merged_from": s["merged_from"],
        "merged_captions": s.get("merged_captions", []),
        "merged_timestamps": s.get("merged_timestamps", []),
        "merged_from_sample_idxs": s.get("merged_from_sample_idxs", [s["sample_idx"]]),
        "clip_cluster": s["clip_cluster"],
        "clip_cluster_size": s["clip_cluster_size"],
        "split_from_generic": s.get("split_from_generic", False),
        "ocr_text": s.get("ocr_text", ""),
        "ocr_cache_source": s.get("ocr_cache_source"),
        "ocr_tokens": s.get("ocr_tokens", []),
        "dhash": s.get("dhash_hex") or (f"{int(s['dhash']):016x}" if "dhash" in s else None),
        "caption_source": s.get("caption_source", "florence"),
        "cluster_role": s.get("cluster_role"),
        "retention_reason": s.get("retention_reason", "none"),
        "retention_reasons_seen": s.get("retention_reasons_seen", [s.get("retention_reason", "none")]),
        "retention_candidate_reason": s.get("retention_candidate_reason"),
        "retention_rejected_reason": s.get("retention_rejected_reason"),
        "rescue_origin": s.get("rescue_origin"),
        "rescue_reason": s.get("rescue_reason"),
        "rescue_origins_seen": s.get("rescue_origins_seen", [s.get("rescue_origin")] if s.get("rescue_origin") else []),
        "rescue_priorities_seen": s.get("rescue_priorities_seen", [s.get("rescue_priority")] if s.get("rescue_priority") is not None else []),
        "proxy_content_score": s.get("proxy_content_score"),
        "rescue_priority": s.get("rescue_priority"),
        "dwell_id": s.get("dwell_id"),
        "temporal_window_id": s.get("temporal_window_id"),
        "lineage_roles": s.get("lineage_roles", [s.get("cluster_role")] if s.get("cluster_role") else []),
        "raw_token_count": s.get("raw_token_count", 0),
        "filtered_token_count": s.get("filtered_token_count", 0),
        "cleaned_token_count": s.get("cleaned_token_count", len(s.get("ocr_tokens", []))),
        "cleaning_attrition_ratio": s.get("cleaning_attrition_ratio", 0.0),
        "low_information_filter_reason": s.get("low_information_filter_reason"),
        "dedupe_stage": s.get("dedupe_stage"),
        "merge_reason": s.get("merge_reason"),
    } for s in selected]

    log_path = out / "captions.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    return log_path


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Two-pass key frame extraction: CLIP over-segment → Florence-2 caption merge."
    )
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--output-dir", "-o", default="frames",
                        help="Output directory (default: frames)")
    parser.add_argument("--sample-interval", "-i", type=float, default=0.5,
                        help="Sample one frame every N seconds (default: 0.5)")
    parser.add_argument("--pass1-clusters", "-c", type=int, default=15,
                        help="Number of CLIP clusters in pass 1 (default: 15)")
    parser.add_argument("--similarity-threshold", "-t", type=float, default=0.85,
                        help="Deprecated no-op; deterministic merge vetoes are used")

    args = parser.parse_args()
    t0 = time.time()
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Start loading all models in parallel while we sample frames
    preloader = ModelPreloader(device=device, need_florence=True, need_ocr=True)

    # Sample
    frames, timestamps, frame_indices = sample_frames(args.video, args.sample_interval)
    if len(frames) < 4:
        print("Too few frames.", file=sys.stderr)
        sys.exit(1)

    # CLIP is ready (or we wait for it here)
    print("\n── Pass 1: CLIP visual embedding ──")
    clip = CLIPEncoder(device=device, preloaded=preloader.get_clip())
    clip_emb = clip.embed_images(frames)
    print(f"  Embedded {len(frames)} frames → {clip_emb.shape}")

    # Scene detection → per-scene clustering
    n_clusters = min(args.pass1_clusters, len(frames) // 2)
    scenes = detect_scenes(args.video, timestamps)
    dhashes = [compute_dhash(frame) for frame in frames]
    scenes, scene_coalescence = coalesce_tiny_scenes(
        scenes, timestamps, dhashes, return_trace=True
    )
    if scene_coalescence["coalesced_scene_count"] != scene_coalescence["original_scene_count"]:
        print(
            "  Coalesced scenes: "
            f"{scene_coalescence['original_scene_count']} -> {scene_coalescence['coalesced_scene_count']}"
        )
    cluster_budget = candidate_budget_for_scenes(n_clusters, len(scenes))
    cluster_allocs = allocate_clusters_by_novelty(scenes, cluster_budget, dhashes, floor=1)
    print(f"  Cluster allocation budget: {sum(cluster_allocs)} candidates")

    all_candidates = []
    sample_clusters = {}
    sample_scenes = {}
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
                "cluster_role": "single",
                "scene_id": scene_id,
            })
            for sample_idx in range(s_start, s_end + 1):
                sample_clusters[sample_idx] = cluster_offset
            cluster_offset += 1
            continue

        scene_cands, scene_labels = clip_oversegment(
            scene_emb, scene_ts, scene_fi, scene_clusters, scene_frames,
            dhashes=dhashes[s_start:s_end + 1],
            max_reps_per_cluster=2,
            return_labels=True,
        )
        for local_idx, label in enumerate(scene_labels):
            sample_clusters[s_start + local_idx] = cluster_offset + int(label)
        for c in scene_cands:
            c["sample_idx"] = s_start + c["sample_idx"]
            c["clip_cluster"] += cluster_offset
            c["scene_id"] = scene_id
        cluster_offset += scene_clusters
        all_candidates.extend(scene_cands)

    candidates = sorted(all_candidates, key=lambda c: c["timestamp"])
    for cand in candidates:
        cand["dhash"] = dhashes[cand["sample_idx"]]
        cand["dhash_hex"] = f"{dhashes[cand['sample_idx']]:016x}"

    pre_rescue_candidate_count = len(candidates)
    rescue_budget = max(3, round(n_clusters * 0.35))
    clip.cleanup()

    candidates = rescue_from_sampled_frames(
        candidates,
        frames,
        timestamps,
        frame_indices,
        dhashes,
        clip_emb,
        n_clusters,
        sample_clusters=sample_clusters,
        sample_scenes=sample_scenes,
        preloaded_engine=preloader.get_ocr_engine(),
    )
    for cand in candidates:
        if "dhash" not in cand:
            cand["dhash"] = dhashes[cand["sample_idx"]]
            cand["dhash_hex"] = f"{dhashes[cand['sample_idx']]:016x}"
        if "sharpness" not in cand:
            cand["sharpness"] = float(_laplacian_sharpness(frames[cand["sample_idx"]]))

    # Florence-2 captioning (batched) — model already loaded in background
    florence_captions = caption_candidates(
        candidates, frames, device=device, preloaded=preloader.get_florence()
    )

    # OCR text extraction + hybrid caption combination
    ocr_texts = ocr_candidates(
        candidates, frames, preloaded_engine=preloader.get_ocr_engine()
    )
    filtered_ocr = _filter_ocr_tokens(ocr_texts)
    florence_caps, ocr_caps, has_ocr = _build_hybrid_captions(
        filtered_ocr, florence_captions, candidates
    )

    # Merge using CLIP image similarity + Jaccard OCR overlap
    ocr_token_sets = _build_ocr_token_sets(filtered_ocr)
    ocr_token_sets = clean_ocr_token_sets(ocr_token_sets)
    attach_ocr_token_attribution(candidates, ocr_texts, filtered_ocr, ocr_token_sets)
    retained = retain_cluster_alternates(candidates)
    print(f"  Cluster alternate retention: {len(candidates)} -> {len(retained)} candidates")
    deduped = near_time_dedupe(retained, [set(c.get("ocr_tokens", [])) for c in retained], dhashes)
    print(f"  Near-time dedupe: {len(retained)} -> {len(deduped)} candidates")
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
    preloader.shutdown()

    # Save
    log_path = save_results(final, frames, args.output_dir)
    manifest_path = write_manifest(
        final,
        args.output_dir,
        metadata={
            "scene_coalescence": scene_coalescence,
            "rescue": {
                "pre_rescue_candidate_count": pre_rescue_candidate_count,
                "rescue_budget": rescue_budget,
            },
        },
    )

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done in {elapsed:.1f}s")
    print(f"Pass 1: {len(frames)} sampled → {len(candidates)} CLIP candidates")
    print(f"Pass 2: {len(candidates)} captioned → {len(final)} final frames")
    print(f"Saved to: {Path(args.output_dir).resolve()}")
    print(f"Caption log: {log_path}")
    print(f"Manifest: {manifest_path}")
    print(f"\nFinal key frames:")
    for s in final:
        print(f"  {Path(s['path']).name}  \"{s['caption'][:100]}\"")


if __name__ == "__main__":
    main()
