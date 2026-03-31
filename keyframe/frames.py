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
import torch
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


# ── CLIP model (shared across passes) ──────────────────────────────────────

class CLIPEncoder:
    """Wraps CLIP model for both image and text embedding in a single space."""

    def __init__(self, device="mps", model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"):
        print(f"  Loading CLIP ({model_name})...")
        self.device = device
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

def caption_candidates(candidates, frames, device="mps"):
    print(f"\n── Pass 2: Florence-2 captioning ({len(candidates)} frames) ──")
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


# ── Merge: CLIP text embeddings of captions ────────────────────────────────

def merge_by_caption(candidates, captions, clip_encoder, similarity_threshold=0.85):
    """
    Embed Florence-2 captions using CLIP's text encoder (same space as the
    image embeddings), then merge candidates whose caption embeddings are
    above the similarity threshold.
    """
    print(f"\n── Merging via CLIP text embeddings (threshold={similarity_threshold}) ──")
    cap_embeddings = clip_encoder.embed_texts(captions)
    print(f"  Caption embeddings: {cap_embeddings.shape}")

    sim_matrix = cap_embeddings @ cap_embeddings.T

    for i in range(len(captions)):
        for j in range(i + 1, len(captions)):
            sim = sim_matrix[i][j]
            if sim > similarity_threshold:
                print(f"    {candidates[i]['timestamp']:5.1f}s ↔ "
                      f"{candidates[j]['timestamp']:5.1f}s: {sim:.3f} (will merge)")

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=1 - similarity_threshold,
    )
    cap_labels = clustering.fit_predict(cap_embeddings)
    print(f"  Caption clusters: {len(set(cap_labels))}")

    final = []
    for cid in sorted(set(cap_labels)):
        group_idxs = [i for i, l in enumerate(cap_labels) if l == cid]
        group_emb = cap_embeddings[group_idxs]
        group_cands = [candidates[i] for i in group_idxs]

        centroid = group_emb.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-8
        sims = group_emb @ centroid
        best_local = np.argmax(sims)

        winner = group_cands[best_local].copy()
        winner["caption_cluster"] = int(cid)
        winner["merged_from"] = len(group_idxs)
        winner["merged_captions"] = [candidates[i]["caption"] for i in group_idxs]
        winner["merged_timestamps"] = [candidates[i]["timestamp"] for i in group_idxs]
        final.append(winner)

        merged_tag = "" if len(group_idxs) == 1 else f" (merged {len(group_idxs)} candidates)"
        print(f"  Group {cid}: kept {winner['timestamp']:.1f}s{merged_tag}")
        print(f"    \"{winner['caption'][:120]}\"")
        if len(group_idxs) > 1:
            for i in group_idxs:
                if i != group_idxs[best_local]:
                    print(f"    dropped {candidates[i]['timestamp']:.1f}s: "
                          f"\"{candidates[i]['caption'][:100]}\"")

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

    # Sample
    frames, timestamps, frame_indices = sample_frames(args.video, args.sample_interval)
    if len(frames) < 4:
        print("Too few frames.", file=sys.stderr)
        sys.exit(1)

    # Load CLIP once, use for both image and text embedding
    print("\n── Pass 1: CLIP visual embedding ──")
    clip = CLIPEncoder(device=device)
    clip_emb = clip.embed_images(frames)
    print(f"  Embedded {len(frames)} frames → {clip_emb.shape}")

    n_clusters = min(args.pass1_clusters, len(frames) // 2)
    candidates = clip_oversegment(clip_emb, timestamps, frame_indices, n_clusters)

    # Florence-2 captioning (batched)
    captions = caption_candidates(candidates, frames, device=device)

    # Merge using CLIP text encoder (same embedding space as images)
    final = merge_by_caption(candidates, captions, clip, args.similarity_threshold)
    clip.cleanup()

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
