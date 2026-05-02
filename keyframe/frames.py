#!/usr/bin/env python3
"""
Video, model, OCR, and output helpers used by the shared keyframe pipeline.

The executable path delegates to keyframe.pipeline.extract_keyframes; this
module keeps the model-bound helpers that are still shared by the orchestrator
and focused postmortem tests.

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
import json

from keyframe.dedupe import (
    clean_ocr_token_sets,
    hamming,
)
from keyframe.merge import build_ocr_token_sets
from keyframe.scoring import score_candidate_for_rep
from keyframe.pipeline.contracts import CandidateRecord, candidate_records, candidate_to_caption_log_row


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
            cand = CandidateRecord(
                sample_idx=int(idx),
                frame_idx=frame_indices[idx],
                timestamp=float(timestamps[idx]),
            ).with_visual(sharpness=float(sharpness)).with_selection(end_of_dwell_bonus=float(dwell_bonus))
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
    candidates = candidate_records(candidates)
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

    candidate_images = [frames[c.sample_idx] for c in candidates]
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
    updated_candidates = []
    for i, (cand, text) in enumerate(zip(candidates, generated_texts)):
        img = candidate_images[i]
        parsed = processor.post_process_generation(
            text, task=prompt, image_size=(img.width, img.height)
        )
        caption = parsed.get(prompt, text)
        captions.append(caption)
        updated_candidates.append(cand.with_evidence(caption=caption))
        print(f"  [{i+1}/{len(candidates)}] {cand.timestamp:5.1f}s -> \"{caption[:120]}\"")

    del model, processor
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
    return captions, tuple(updated_candidates)


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
    candidates = candidate_records(candidates)
    use_apple = _is_macos()
    backend = "Apple Vision" if use_apple else "PaddleOCR"
    print(f"\n── Pass 2b: OCR text extraction ({len(candidates)} frames, {backend}) ──")

    need_ocr = any(cand.evidence.ocr_text is None for cand in candidates)
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
    updated_candidates = []
    for i, cand in enumerate(candidates):
        if cand.evidence.ocr_text is not None:
            raw_text = str(cand.evidence.ocr_text)
            ocr_texts.append(raw_text)
            updated_candidates.append(cand)
            preview = raw_text[:120] if raw_text else "(no text)"
            print(f"  [{i+1}/{len(candidates)}] {cand.timestamp:5.1f}s -> "
                  f"cached, {len(raw_text)} chars: \"{preview}\"")
            continue

        img = frames[cand.sample_idx]
        lines = _ocr_apple_vision(img) if use_apple else _ocr_paddle(img, paddle_engine)

        raw_text = " ".join(lines)
        ocr_texts.append(raw_text)
        updated_candidates.append(cand.with_evidence(ocr_text=raw_text))
        preview = raw_text[:120] if raw_text else "(no text)"
        print(f"  [{i+1}/{len(candidates)}] {cand.timestamp:5.1f}s -> "
              f"{len(lines)} lines, {len(raw_text)} chars: \"{preview}\"")

    return ocr_texts, tuple(updated_candidates)


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
    candidates = candidate_records(candidates)
    has_ocr = []
    updated_candidates = []
    for i, (ocr, flor) in enumerate(zip(filtered_ocr, florence_captions)):
        has = len(ocr) >= min_ocr_chars
        has_ocr.append(has)
        tag = "both" if has else "florence"
        cand = candidates[i]
        updated_candidates.append(cand.with_evidence(caption_source=tag))
        display = f"{flor[:60]} + OCR({len(ocr)}ch)" if has else flor[:100]
        print(f"  [{i+1}/{len(filtered_ocr)}] {candidates[i].timestamp:5.1f}s -> "
              f"[{tag}] \"{display}\"")
    n_both = sum(has_ocr)
    print(f"  → {n_both} with OCR signal, {len(has_ocr) - n_both} Florence-2 only")
    return florence_captions, filtered_ocr, has_ocr, tuple(updated_candidates)


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
    candidates = candidate_records(candidates)
    raw_token_sets = _build_ocr_token_sets(raw_ocr_texts)
    filtered_token_sets = _build_ocr_token_sets(filtered_ocr_texts)
    rescue_token_sets = _build_rescue_token_sets(raw_ocr_texts)
    updated_candidates = []
    for cand, raw_tokens, filtered_tokens, cleaned_tokens, rescue_tokens in zip(
        candidates, raw_token_sets, filtered_token_sets, cleaned_token_sets, rescue_token_sets
    ):
        raw_count = len(raw_tokens)
        filtered_count = len(filtered_tokens)
        cleaned_count = len(cleaned_tokens)
        updated = cand.with_evidence(
            ocr_tokens=tuple(sorted(cleaned_tokens)),
            dedupe_tokens=tuple(sorted(cleaned_tokens)),
            rescue_tokens=tuple(sorted(rescue_tokens)),
            raw_token_count=raw_count,
            filtered_token_count=filtered_count,
            cleaned_token_count=cleaned_count,
            cleaning_attrition_ratio=0.0 if raw_count == 0 else round((raw_count - cleaned_count) / raw_count, 4),
        ).with_selection(
            retention_reason=cand.selection.retention_reason or "none",
        )
        reasons_seen = set(cand.lineage.retention_reasons_seen) | {updated.selection.retention_reason or "none"}
        roles = set(cand.lineage.lineage_roles)
        if cand.visual.cluster_role:
            roles.add(str(cand.visual.cluster_role))
        updated_candidates.append(
            updated.with_lineage(
                retention_reasons_seen=tuple(sorted(reasons_seen)),
                lineage_roles=tuple(sorted(roles)),
            )
        )
    return tuple(updated_candidates)


def attach_rescue_ocr_metadata(candidates, raw_ocr_texts):
    """Attach temporary split OCR tokens while preserving raw OCR as cache."""
    candidates = candidate_records(candidates)
    filtered_ocr = _filter_ocr_tokens(raw_ocr_texts)
    dedupe_token_sets = clean_ocr_token_sets(_build_ocr_token_sets(filtered_ocr))
    rescue_token_sets = _build_rescue_token_sets(raw_ocr_texts)
    updated_candidates = []
    for cand, raw_text, filtered_text, dedupe_tokens, rescue_tokens in zip(
        candidates, raw_ocr_texts, filtered_ocr, dedupe_token_sets, rescue_token_sets
    ):
        raw_tokens = _build_ocr_token_sets([raw_text])[0]
        filtered_tokens = _build_ocr_token_sets([filtered_text])[0]
        updated_candidates.append(
            cand.with_evidence(
                ocr_text=raw_text,
                ocr_cache_source=cand.evidence.ocr_cache_source or "rescue",
                ocr_tokens=tuple(sorted(dedupe_tokens)),
                dedupe_tokens=tuple(sorted(dedupe_tokens)),
                rescue_tokens=tuple(sorted(rescue_tokens)),
                raw_token_count=len(raw_tokens),
                filtered_token_count=len(filtered_tokens),
                cleaned_token_count=len(dedupe_tokens),
            )
        )
    return tuple(updated_candidates)


def _comparison_primary_sample_idxs(candidates, shortlist):
    candidates = candidate_records(candidates)
    shortlist = candidate_records(shortlist)
    selected: set[int] = set()
    for rescue in shortlist:
        rescue_ts = float(rescue.timestamp)
        rescue_cluster = rescue.visual.clip_cluster
        rescue_scene = rescue.temporal.scene_id

        same_cluster = [
            cand for cand in candidates
            if rescue_cluster is not None and cand.visual.clip_cluster == rescue_cluster
        ]
        same_scene = [
            cand for cand in candidates
            if rescue_scene is not None and cand.temporal.scene_id == rescue_scene
        ]
        for pool in (same_cluster, same_scene):
            if not pool:
                continue
            primary = min(
                pool,
                key=lambda cand: abs(float(cand.timestamp) - rescue_ts),
            )
            selected.add(int(primary.sample_idx))
    return selected


# ── Output ──────────────────────────────────────────────────────────────────

def save_results(selected, frames, output_dir):
    """Save selected frames to disk.

    `frames` may be a list (indexed by sample_idx) or a mapping
    {sample_idx: PIL.Image} containing only the selected frames. The mapping
    form lets callers free the full frames list before saving.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for s in selected:
        if isinstance(s, CandidateRecord):
            ts = s.timestamp
            fidx = s.frame_idx
            sample_idx = s.sample_idx
        else:
            ts = s["timestamp"]
            fidx = s["frame_idx"]
            sample_idx = s["sample_idx"]
        path = out / f"frame_{fidx:06d}_{ts:.2f}s.png"
        frames[sample_idx].save(str(path))
        if isinstance(s, CandidateRecord):
            rows.append(candidate_to_caption_log_row(s, path=path))
        else:
            row = dict(s)
            row["path"] = str(path)
            rows.append(row)

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
    } for s in rows]

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
    from keyframe.pipeline import KeyframeExtractionConfig, extract_keyframes

    result = extract_keyframes(
        args.video,
        args.output_dir,
        KeyframeExtractionConfig(
            sample_interval=args.sample_interval,
            pass1_clusters=args.pass1_clusters,
            similarity_threshold=args.similarity_threshold,
        ),
    )

    print(f"\nFinal key frames:")
    for s in result.final:
        print(f"  {Path(s['path']).name}  \"{s['caption'][:100]}\"")


if __name__ == "__main__":
    main()
