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
import numpy as np
import os
import re
import torch
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, Future
from PIL import Image
from pathlib import Path
from transformers import Florence2ForConditionalGeneration, AutoProcessor
import open_clip
from sklearn.cluster import AgglomerativeClustering
import argparse
import sys
import time
import json


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


# ── Parallel model preloading ─────────────────────────────────────────────

def _load_clip(device, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"):
    """Load CLIP model (called in background thread)."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, preprocess, tokenizer


def _load_florence(device):
    """Load Florence-2 model + processor (called in background thread)."""
    model_name = "florence-community/Florence-2-base"
    processor = AutoProcessor.from_pretrained(model_name)
    model = Florence2ForConditionalGeneration.from_pretrained(
        model_name, dtype=torch.float32,
    ).to(device)
    model.eval()
    return model, processor


def _load_ocr_engine():
    """Load PaddleOCR engine for non-macOS (called in background thread)."""
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    from paddleocr import PaddleOCR
    return PaddleOCR(
        lang='en',
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


class ModelPreloader:
    """Starts loading all models in background threads at pipeline start."""

    def __init__(self, device="mps", need_florence=True, need_ocr=True):
        self._pool = ThreadPoolExecutor(max_workers=3)
        self._device = device

        print("  Preloading models in background...")
        self._clip_future = self._pool.submit(_load_clip, device)

        self._florence_future = None
        if need_florence:
            self._florence_future = self._pool.submit(_load_florence, device)

        self._ocr_future = None
        if need_ocr and not _is_macos():
            self._ocr_future = self._pool.submit(_load_ocr_engine)

    def get_clip(self):
        model, preprocess, tokenizer = self._clip_future.result()
        return model, preprocess, tokenizer

    def get_florence(self):
        if self._florence_future is None:
            raise RuntimeError("Florence-2 was not preloaded")
        return self._florence_future.result()

    def get_ocr_engine(self):
        if self._ocr_future is None:
            return None
        return self._ocr_future.result()

    def shutdown(self):
        self._pool.shutdown(wait=False)


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
            all_emb.append(features.cpu().numpy())
        return np.vstack(all_emb).astype(np.float32)

    def embed_texts(self, texts):
        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad(), torch.amp.autocast(device_type=str(self.device)):
            features = self.model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32)

    def cleanup(self):
        del self.model, self.tokenizer
        if self.device == "mps":
            torch.mps.empty_cache()


# ── Pass 1: CLIP image embedding + over-segmentation ───────────────────────

def clip_oversegment(embeddings, timestamps, frame_indices, n_clusters):
    print(f"  Over-segmenting into {n_clusters} clusters...")
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters, metric="cosine", linkage="average",
    )
    labels = clustering.fit_predict(embeddings)

    candidates = []
    for cid in range(n_clusters):
        mask = labels == cid
        idxs = np.where(mask)[0]
        cluster_emb = embeddings[idxs]

        centroid = cluster_emb.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-8
        sims = cluster_emb @ centroid
        best = idxs[np.argmax(sims)]

        candidates.append({
            "sample_idx": int(best),
            "frame_idx": frame_indices[best],
            "timestamp": timestamps[best],
            "clip_cluster": cid,
            "clip_cluster_size": int(mask.sum()),
        })

    candidates.sort(key=lambda c: c["timestamp"])
    print(f"  → {len(candidates)} candidates")
    for c in candidates:
        print(f"    {c['timestamp']:5.1f}s  (cluster {c['clip_cluster']}, "
              f"{c['clip_cluster_size']} frames)")
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
            model_name, dtype=torch.float32,
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

    paddle_engine = None
    if not use_apple:
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


# ── Merge: CLIP text embeddings of captions ────────────────────────────────

# Captions matching any of these prefixes are considered low-information —
# Florence-2 produces them for almost any screen recording regardless of content.
_GENERIC_CAPTION_PREFIXES = (
    "The image is a screenshot of a computer screen",
    "The image shows a screenshot of a computer screen",
    "The image is a screenshot of a screen",
    "A screenshot of a computer screen",
)


def _is_generic(caption):
    """Return True if a caption matches a known low-information pattern."""
    return any(caption.startswith(p) for p in _GENERIC_CAPTION_PREFIXES)


def merge_by_caption(candidates, florence_captions, ocr_captions, has_ocr,
                     clip_encoder, similarity_threshold=0.85, ocr_weight=0.6):
    """
    Dual-embedding merge: embed Florence-2 captions and OCR text separately
    with CLIP's text encoder, then combine similarity matrices to decide merges.

    This avoids CLIP's 77-token truncation — each signal gets its own full
    encoding. When OCR is available, frames need to be similar on BOTH scene
    description AND on-screen text to merge.

    Args:
        florence_captions: List of Florence-2 captions (scene descriptions).
        ocr_captions: List of filtered OCR texts (on-screen content).
        has_ocr: List of bools — whether each frame has substantial OCR.
        ocr_weight: Weight for OCR similarity (0-1). Florence gets 1-ocr_weight.
    """
    print(f"\n── Merging via dual CLIP text embeddings (threshold={similarity_threshold}) ──")

    # Embed Florence-2 captions
    florence_emb = clip_encoder.embed_texts(florence_captions)
    florence_sim = florence_emb @ florence_emb.T
    print(f"  Florence embeddings: {florence_emb.shape}")

    # Embed OCR texts (use empty string for frames without OCR)
    ocr_emb = clip_encoder.embed_texts(
        [ocr if h else "" for ocr, h in zip(ocr_captions, has_ocr)]
    )
    ocr_sim = ocr_emb @ ocr_emb.T
    print(f"  OCR embeddings: {ocr_emb.shape}")

    # Build combined similarity matrix
    # When both frames have OCR: weighted blend of florence + OCR similarity
    # When either lacks OCR: use florence similarity only
    n = len(candidates)
    combined_sim = np.zeros((n, n))
    fw = 1.0 - ocr_weight
    for i in range(n):
        for j in range(n):
            if has_ocr[i] and has_ocr[j]:
                combined_sim[i][j] = fw * florence_sim[i][j] + ocr_weight * ocr_sim[i][j]
            else:
                combined_sim[i][j] = florence_sim[i][j]

    for i in range(n):
        for j in range(i + 1, n):
            if combined_sim[i][j] > similarity_threshold:
                detail = ""
                if has_ocr[i] and has_ocr[j]:
                    detail = (f" [F:{florence_sim[i][j]:.3f} "
                              f"OCR:{ocr_sim[i][j]:.3f}]")
                print(f"    {candidates[i]['timestamp']:5.1f}s ↔ "
                      f"{candidates[j]['timestamp']:5.1f}s: "
                      f"{combined_sim[i][j]:.3f}{detail} (will merge)")

    # Convert combined similarity to distance for clustering
    combined_dist = 1.0 - combined_sim
    np.fill_diagonal(combined_dist, 0)
    combined_dist = np.clip(combined_dist, 0, None)  # numerical safety

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
    # For groups where members share OCR text (same app/branch), the global DF
    # filter doesn't remove within-group common tokens. Re-filter OCR within
    # each group to surface what actually differs, then re-cluster.
    n_refined = 0
    refined_labels = np.array(cap_labels, copy=True)
    next_label = max(cap_labels) + 1

    for cid in sorted(set(cap_labels)):
        group_idxs = [i for i, l in enumerate(cap_labels) if l == cid]
        if len(group_idxs) < 4:
            continue
        # Only refine if most members have OCR
        group_has_ocr = [has_ocr[i] for i in group_idxs]
        if sum(group_has_ocr) < len(group_idxs) * 0.5:
            continue

        # Intra-group DF filter: remove tokens common within this group
        group_ocr = [ocr_captions[i] for i in group_idxs]
        local_filtered = _filter_ocr_tokens(group_ocr, max_tokens=70, df_cutoff=0.5)

        # Check if local filtering produced any differentiation
        non_empty = [t for t in local_filtered if len(t.strip()) >= 30]
        if len(non_empty) < 2:
            continue

        # Re-embed locally-filtered OCR and recompute combined similarity
        local_ocr_emb = clip_encoder.embed_texts(
            [t if len(t.strip()) >= 30 else "" for t in local_filtered]
        )
        local_ocr_sim = local_ocr_emb @ local_ocr_emb.T
        gn = len(group_idxs)
        local_combined = np.zeros((gn, gn))
        for gi in range(gn):
            for gj in range(gn):
                ii, jj = group_idxs[gi], group_idxs[gj]
                has_both = (len(local_filtered[gi].strip()) >= 30 and
                            len(local_filtered[gj].strip()) >= 30)
                if has_both:
                    local_combined[gi][gj] = (fw * florence_sim[ii][jj] +
                                              ocr_weight * local_ocr_sim[gi][gj])
                else:
                    local_combined[gi][gj] = florence_sim[ii][jj]

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
                filt_preview = local_filtered[gi][:80] if local_filtered[gi] else "(no OCR)"
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

        # Pick the candidate closest to the group centroid (using florence embeddings)
        group_emb = florence_emb[group_idxs]
        centroid = group_emb.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-8
        sims = group_emb @ centroid
        best_local = np.argmax(sims)

        winner = group_cands[best_local].copy()
        winner["caption_cluster"] = int(cid)
        winner["merged_from"] = len(group_idxs)
        winner["merged_captions"] = [florence_captions[i] for i in group_idxs]
        winner["merged_timestamps"] = [candidates[i]["timestamp"] for i in group_idxs]
        final.append(winner)

        merged_tag = "" if len(group_idxs) == 1 else f" (merged {len(group_idxs)} candidates)"
        print(f"  Group {cid}: kept {winner['timestamp']:.1f}s{merged_tag}")
        print(f"    \"{florence_captions[group_idxs[best_local]][:120]}\"")
        if len(group_idxs) > 1:
            for i in group_idxs:
                if i != group_idxs[best_local]:
                    print(f"    dropped {candidates[i]['timestamp']:.1f}s: "
                          f"\"{florence_captions[i][:100]}\"")

    final.sort(key=lambda c: c["timestamp"])
    return final


# ── Output ──────────────────────────────────────────────────────────────────

def save_results(selected, frames, output_dir):
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
        "clip_cluster": s["clip_cluster"],
        "clip_cluster_size": s["clip_cluster_size"],
        "split_from_generic": s.get("split_from_generic", False),
        "ocr_text": s.get("ocr_text", ""),
        "caption_source": s.get("caption_source", "florence"),
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
                        help="Caption similarity threshold for merging (default: 0.85)")

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

    n_clusters = min(args.pass1_clusters, len(frames) // 2)
    candidates = clip_oversegment(clip_emb, timestamps, frame_indices, n_clusters)

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

    # Merge using dual CLIP text embeddings (florence + OCR separately)
    final = merge_by_caption(
        candidates, florence_caps, ocr_caps, has_ocr,
        clip, args.similarity_threshold,
    )
    clip.cleanup()
    preloader.shutdown()

    # Save
    log_path = save_results(final, frames, args.output_dir)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done in {elapsed:.1f}s")
    print(f"Pass 1: {len(frames)} sampled → {len(candidates)} CLIP candidates")
    print(f"Pass 2: {len(candidates)} captioned → {len(final)} final frames")
    print(f"Saved to: {Path(args.output_dir).resolve()}")
    print(f"Caption log: {log_path}")
    print(f"\nFinal key frames:")
    for s in final:
        print(f"  {Path(s['path']).name}  \"{s['caption'][:100]}\"")


if __name__ == "__main__":
    main()
