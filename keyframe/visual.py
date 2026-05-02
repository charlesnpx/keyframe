"""Dependency-light visual metrics used by proposal, scoring, and dedupe."""

from __future__ import annotations

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
