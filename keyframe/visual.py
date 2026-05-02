"""Dependency-light visual metrics used by proposal, scoring, and dedupe."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Sequence
import math
from typing import Any

import numpy as np
from PIL import Image, ImageStat


def laplacian_sharpness(pil_img: Image.Image) -> float:
    """Score frame sharpness via a small Laplacian variance implementation."""
    gray = np.asarray(pil_img.convert("L"), dtype=np.float32)
    padded = np.pad(gray, 1, mode="edge")
    lap = (
        -4.0 * padded[1:-1, 1:-1]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )
    return float(lap.var())


def sobel_edges(gray_array: np.ndarray) -> np.ndarray:
    padded = np.pad(gray_array.astype(np.float32), 1, mode="edge")
    gx = (
        -padded[:-2, :-2] + padded[:-2, 2:]
        - 2 * padded[1:-1, :-2] + 2 * padded[1:-1, 2:]
        - padded[2:, :-2] + padded[2:, 2:]
    )
    gy = (
        -padded[:-2, :-2] - 2 * padded[:-2, 1:-1] - padded[:-2, 2:]
        + padded[2:, :-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:]
    )
    return np.hypot(gx, gy)


def proxy_frame_components(image: Image.Image) -> dict[str, float]:
    """Return raw cheap content metrics for all-sampled-frame rescue ranking."""
    gray = image.convert("L").resize((160, 90), Image.Resampling.LANCZOS)
    arr = np.asarray(gray, dtype=np.float32)
    edges = sobel_edges(arr)
    total = max(arr.size, 1)
    dark_ratio = float(np.count_nonzero(arr <= 24) / total)
    bright_ratio = float(np.count_nonzero(arr >= 232) / total)

    if edges.size:
        threshold = max(float(np.percentile(edges, 75)), 12.0)
        edge_mask = edges >= threshold
        band_density = []
        for start in range(0, edge_mask.shape[0], 3):
            band = edge_mask[start:start + 3, :]
            if band.size:
                band_density.append(float(np.count_nonzero(band) / band.size))
        textline_score = sum(1 for density in band_density if density >= 0.08) / max(len(band_density), 1)
        edge_score = float(edges.mean())
    else:
        textline_score = 0.0
        edge_score = 0.0

    histogram = gray.histogram()
    entropy = 0.0
    for count in histogram:
        if not count:
            continue
        p = count / total
        entropy -= p * math.log2(p)

    return {
        "textline_score": float(textline_score),
        "edge_score": float(edge_score),
        "entropy": float(entropy),
        "dark_ratio": dark_ratio,
        "bright_ratio": bright_ratio,
    }


def _normalize_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi == lo:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def _resample_gray_stack(
    frames: Sequence[Image.Image],
    *,
    size: tuple[int, int],
    content_only: bool = False,
    resample: Image.Resampling = Image.Resampling.LANCZOS,
    chunk_size: int = 128,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, len(frames), chunk_size):
        chunk = []
        for frame in frames[start:start + chunk_size]:
            image = content_crop(frame) if content_only else frame
            gray = image.convert("L").resize(size, resample)
            chunk.append(np.asarray(gray, dtype=np.float32))
        if chunk:
            chunks.append(np.stack(chunk, axis=0))
    if not chunks:
        return np.empty((0, size[1], size[0]), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def _sobel_edges_stack(gray_stack: np.ndarray) -> np.ndarray:
    if gray_stack.size == 0:
        return np.empty_like(gray_stack, dtype=np.float32)
    padded = np.pad(gray_stack.astype(np.float32), ((0, 0), (1, 1), (1, 1)), mode="edge")
    gx = (
        -padded[:, :-2, :-2] + padded[:, :-2, 2:]
        - 2 * padded[:, 1:-1, :-2] + 2 * padded[:, 1:-1, 2:]
        - padded[:, 2:, :-2] + padded[:, 2:, 2:]
    )
    gy = (
        -padded[:, :-2, :-2] - 2 * padded[:, :-2, 1:-1] - padded[:, :-2, 2:]
        + padded[:, 2:, :-2] + 2 * padded[:, 2:, 1:-1] + padded[:, 2:, 2:]
    )
    return np.hypot(gx, gy).astype(np.float32)


def _laplacian_variance_stack(gray_stack: np.ndarray) -> np.ndarray:
    if gray_stack.size == 0:
        return np.empty((0,), dtype=np.float32)
    padded = np.pad(gray_stack.astype(np.float32), ((0, 0), (1, 1), (1, 1)), mode="edge")
    lap = (
        -4.0 * padded[:, 1:-1, 1:-1]
        + padded[:, :-2, 1:-1]
        + padded[:, 2:, 1:-1]
        + padded[:, 1:-1, :-2]
        + padded[:, 1:-1, 2:]
    )
    return np.var(lap, axis=(1, 2)).astype(np.float32)


def _entropy_256_stack(gray_stack: np.ndarray) -> np.ndarray:
    if gray_stack.size == 0:
        return np.empty((0,), dtype=np.float32)
    quantized = np.clip(np.rint(gray_stack), 0, 255).astype(np.uint8)
    total = max(int(gray_stack.shape[1] * gray_stack.shape[2]), 1)
    entropies = np.zeros((gray_stack.shape[0],), dtype=np.float32)
    for idx, frame in enumerate(quantized):
        counts = np.bincount(frame.ravel(), minlength=256).astype(np.float32)
        probs = counts[counts > 0] / float(total)
        entropies[idx] = float(-np.sum(probs * np.log2(probs)))
    return entropies


def _visual_entropy_and_buckets(gray_stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if gray_stack.size == 0:
        empty = np.empty((0,), dtype=np.float32)
        return empty, empty
    quantized = np.clip(np.rint(gray_stack), 0, 255).astype(np.uint8)
    total = max(int(gray_stack.shape[1] * gray_stack.shape[2]), 1)
    entropies = np.zeros((gray_stack.shape[0],), dtype=np.float32)
    unique_buckets = np.zeros((gray_stack.shape[0],), dtype=np.float32)
    for idx, frame in enumerate(quantized):
        bucket_ids = (frame.ravel() // 8).astype(np.int16)
        counts = np.bincount(bucket_ids, minlength=32).astype(np.float32)
        nonzero = counts[counts > 0]
        unique_buckets[idx] = float(nonzero.size)
        probs = nonzero / float(total)
        entropies[idx] = float(-np.sum(probs * np.log2(probs)))
    return entropies, unique_buckets


@dataclass
class FrameMetricTable:
    """Reusable NumPy metrics for sampled frames; arrays are indexed by sample_idx."""

    sample_idx: np.ndarray
    frame_idx: np.ndarray
    timestamp: np.ndarray
    textline_score: np.ndarray
    edge_score: np.ndarray
    entropy: np.ndarray
    dark_ratio: np.ndarray
    bright_ratio: np.ndarray
    normalized_textline_score: np.ndarray
    normalized_edge_score: np.ndarray
    normalized_entropy: np.ndarray
    blank_penalty: np.ndarray
    proxy_content_score: np.ndarray
    content_prev_delta: np.ndarray
    content_next_delta: np.ndarray
    content_area_delta_score: np.ndarray
    visual_stddev: np.ndarray
    visual_edge_score: np.ndarray
    visual_dark_ratio: np.ndarray
    visual_bright_ratio: np.ndarray
    visual_entropy: np.ndarray
    visual_unique_buckets: np.ndarray
    sharpness: np.ndarray
    full_gray_stack: np.ndarray = field(repr=False)
    content_gray_stack: np.ndarray = field(repr=False)

    @property
    def sample_count(self) -> int:
        return int(self.sample_idx.size)

    def has_sample(self, sample_idx: int) -> bool:
        return 0 <= int(sample_idx) < self.sample_count

    def proxy_row(self, sample_idx: int) -> dict[str, float]:
        idx = int(sample_idx)
        return {
            "textline_score": float(self.textline_score[idx]),
            "edge_score": float(self.edge_score[idx]),
            "entropy": float(self.entropy[idx]),
            "dark_ratio": float(self.dark_ratio[idx]),
            "bright_ratio": float(self.bright_ratio[idx]),
            "proxy_content_score": float(self.proxy_content_score[idx]),
            "normalized_textline_score": float(self.normalized_textline_score[idx]),
            "normalized_edge_score": float(self.normalized_edge_score[idx]),
            "normalized_entropy": float(self.normalized_entropy[idx]),
            "blank_penalty": float(self.blank_penalty[idx]),
            "content_area_delta_score": float(self.content_area_delta_score[idx]),
            "content_area_previous_delta": float(self.content_prev_delta[idx]),
            "content_area_next_delta": float(self.content_next_delta[idx]),
        }

    def to_proxy_rows(self) -> list[dict[str, float]]:
        return [self.proxy_row(idx) for idx in range(self.sample_count)]

    def visual_information_for(self, sample_idx: int) -> dict[str, float] | None:
        if not self.has_sample(sample_idx):
            return None
        idx = int(sample_idx)
        return {
            "stddev": float(self.visual_stddev[idx]),
            "edge_score": float(self.visual_edge_score[idx]),
            "dark_ratio": float(self.visual_dark_ratio[idx]),
            "bright_ratio": float(self.visual_bright_ratio[idx]),
            "entropy": float(self.visual_entropy[idx]),
            "unique_buckets": float(self.visual_unique_buckets[idx]),
        }

    def sharpness_for(self, sample_idx: int) -> float | None:
        if not self.has_sample(sample_idx):
            return None
        return float(self.sharpness[int(sample_idx)])

    def content_delta_between(self, sample_idx_a: int, sample_idx_b: int) -> float | None:
        left = int(sample_idx_a)
        right = int(sample_idx_b)
        if not self.has_sample(left) or not self.has_sample(right):
            return None
        if left == right:
            return 0.0
        if abs(left - right) == 1:
            return float(self.content_next_delta[min(left, right)])
        return float(np.mean(np.abs(self.content_gray_stack[left] - self.content_gray_stack[right])))

    def summary(self) -> dict[str, Any]:
        proxy_max = float(np.max(self.proxy_content_score)) if self.sample_count else 0.0
        delta_max = float(np.max(self.content_area_delta_score)) if self.sample_count else 0.0
        return {
            "sample_count": self.sample_count,
            "proxy_row_count": self.sample_count,
            "content_gray_shape": list(self.content_gray_stack.shape),
            "full_gray_shape": list(self.full_gray_stack.shape),
            "proxy_content_score_max": proxy_max,
            "content_area_delta_score_max": delta_max,
        }


def build_frame_metric_table(
    frames: Sequence[Image.Image],
    timestamps: Sequence[float],
    frame_indices: Sequence[int],
) -> FrameMetricTable:
    """Build reusable low-res visual metrics for all sampled frames."""
    sample_count = len(frames)
    full_stack = _resample_gray_stack(frames, size=(160, 90), content_only=False, resample=Image.Resampling.LANCZOS)
    content_stack = _resample_gray_stack(frames, size=(160, 90), content_only=True, resample=Image.Resampling.BILINEAR)

    total = max(160 * 90, 1)
    edges = _sobel_edges_stack(full_stack)
    dark_ratio = np.mean(full_stack <= 24, axis=(1, 2)).astype(np.float32) if sample_count else np.empty((0,), dtype=np.float32)
    bright_ratio = np.mean(full_stack >= 232, axis=(1, 2)).astype(np.float32) if sample_count else np.empty((0,), dtype=np.float32)
    edge_score = np.mean(edges, axis=(1, 2)).astype(np.float32) if sample_count else np.empty((0,), dtype=np.float32)
    entropy = _entropy_256_stack(full_stack)

    if sample_count:
        thresholds = np.maximum(np.percentile(edges, 75, axis=(1, 2)), 12.0).astype(np.float32)
        edge_mask = edges >= thresholds[:, None, None]
        band_density = edge_mask.reshape(sample_count, 30, 3, 160).mean(axis=(2, 3))
        textline_score = (band_density >= 0.08).mean(axis=1).astype(np.float32)
    else:
        textline_score = np.empty((0,), dtype=np.float32)

    normalized_textline = _normalize_array(textline_score)
    normalized_edge = _normalize_array(edge_score)
    normalized_entropy = _normalize_array(entropy)
    blank_penalty = (
        0.5 * np.maximum(0.0, dark_ratio - 0.7)
        + 0.5 * np.maximum(0.0, bright_ratio - 0.7)
    ).astype(np.float32)
    proxy_content_score = np.clip(
        0.45 * normalized_textline + 0.30 * normalized_edge + 0.25 * normalized_entropy - blank_penalty,
        0.0,
        1.0,
    ).astype(np.float32)

    content_prev_delta = np.zeros((sample_count,), dtype=np.float32)
    if sample_count > 1:
        adjacent = np.mean(np.abs(content_stack[1:] - content_stack[:-1]), axis=(1, 2)).astype(np.float32)
        content_prev_delta[1:] = adjacent
    content_next_delta = np.zeros((sample_count,), dtype=np.float32)
    if sample_count > 1:
        content_next_delta[:-1] = content_prev_delta[1:]
    content_delta_score = (np.maximum(content_prev_delta, content_next_delta) / 255.0).astype(np.float32)

    visual_stddev = np.std(full_stack, axis=(1, 2)).astype(np.float32) if sample_count else np.empty((0,), dtype=np.float32)
    if sample_count:
        horizontal = np.abs(full_stack[:, :, :-1] - full_stack[:, :, 1:]).sum(axis=(1, 2))
        vertical = np.abs(full_stack[:, :-1, :] - full_stack[:, 1:, :]).sum(axis=(1, 2))
        visual_edge_score = ((horizontal + vertical) / max((90 * 159) + (89 * 160), 1)).astype(np.float32)
    else:
        visual_edge_score = np.empty((0,), dtype=np.float32)
    visual_entropy, visual_unique_buckets = _visual_entropy_and_buckets(full_stack)
    sharpness = _laplacian_variance_stack(full_stack)

    return FrameMetricTable(
        sample_idx=np.arange(sample_count, dtype=np.int64),
        frame_idx=np.asarray(list(frame_indices), dtype=np.int64),
        timestamp=np.asarray(list(timestamps), dtype=np.float64),
        textline_score=textline_score,
        edge_score=edge_score,
        entropy=entropy,
        dark_ratio=dark_ratio,
        bright_ratio=bright_ratio,
        normalized_textline_score=normalized_textline,
        normalized_edge_score=normalized_edge,
        normalized_entropy=normalized_entropy,
        blank_penalty=blank_penalty,
        proxy_content_score=proxy_content_score,
        content_prev_delta=content_prev_delta,
        content_next_delta=content_next_delta,
        content_area_delta_score=content_delta_score,
        visual_stddev=visual_stddev,
        visual_edge_score=visual_edge_score,
        visual_dark_ratio=dark_ratio.copy(),
        visual_bright_ratio=bright_ratio.copy(),
        visual_entropy=visual_entropy,
        visual_unique_buckets=visual_unique_buckets,
        sharpness=sharpness,
        full_gray_stack=full_stack,
        content_gray_stack=content_stack,
    )


def visual_information_score(image: Image.Image) -> dict[str, float]:
    """Return cheap grayscale information metrics for a selected frame."""
    gray = image.convert("L").resize((160, 90), Image.Resampling.LANCZOS)
    if hasattr(gray, "get_flattened_data"):
        pixels = list(gray.get_flattened_data())
    else:
        pixels = list(gray.getdata())
    stat = ImageStat.Stat(gray)
    stddev = float(stat.stddev[0])
    total = len(pixels) or 1
    dark_ratio = sum(1 for p in pixels if p <= 24) / total
    bright_ratio = sum(1 for p in pixels if p >= 232) / total

    if pixels:
        width, height = gray.size
        horizontal = [
            abs(pixels[y * width + x] - pixels[y * width + x + 1])
            for y in range(height)
            for x in range(width - 1)
        ]
        vertical = [
            abs(pixels[y * width + x] - pixels[(y + 1) * width + x])
            for y in range(height - 1)
            for x in range(width)
        ]
        edge_score = (sum(horizontal) + sum(vertical)) / max(len(horizontal) + len(vertical), 1)
    else:
        edge_score = 0.0

    histogram = gray.histogram()
    entropy = 0.0
    unique_buckets = 0
    bucket_size = 8
    for i in range(0, len(histogram), bucket_size):
        bucket_count = sum(histogram[i:i + bucket_size])
        if bucket_count:
            unique_buckets += 1
            p = bucket_count / total
            entropy -= p * math.log2(p)

    return {
        "stddev": stddev,
        "edge_score": float(edge_score),
        "dark_ratio": float(dark_ratio),
        "bright_ratio": float(bright_ratio),
        "entropy": float(entropy),
        "unique_buckets": float(unique_buckets),
    }


def content_crop(image: Image.Image, margin_x: float = 0.12, margin_y: float = 0.10) -> Image.Image:
    width, height = image.size
    left = int(width * margin_x)
    top = int(height * margin_y)
    right = max(left + 1, int(width * (1.0 - margin_x)))
    bottom = max(top + 1, int(height * (1.0 - margin_y)))
    return image.crop((left, top, right, bottom))


def mean_abs_content_delta(
    image_a: Image.Image,
    image_b: Image.Image,
    *,
    size: tuple[int, int] = (160, 90),
) -> float:
    """Mean absolute grayscale delta over the central content area."""
    a = content_crop(image_a).convert("L").resize(size, Image.Resampling.BILINEAR)
    b = content_crop(image_b).convert("L").resize(size, Image.Resampling.BILINEAR)
    arr_a = np.asarray(a, dtype=np.float32)
    arr_b = np.asarray(b, dtype=np.float32)
    return float(np.mean(np.abs(arr_a - arr_b)))


def frame_for_index(frames: Sequence[Any] | dict[int, Any], sample_idx: int) -> Any | None:
    if isinstance(frames, dict):
        return frames.get(sample_idx)
    return frames[sample_idx] if 0 <= sample_idx < len(frames) else None
