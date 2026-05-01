"""Deterministic frame dedupe helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import re
from typing import Any

from PIL import Image, ImageStat


def compute_dhash(image: Image.Image, hash_size: int = 8) -> int:
    """Compute a 64-bit difference hash from a center crop of an image."""
    if hash_size <= 0:
        raise ValueError("hash_size must be positive")

    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    cropped = image.crop((left, top, left + side, top + side))
    gray = cropped.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())

    value = 0
    for row in range(hash_size):
        offset = row * (hash_size + 1)
        for col in range(hash_size):
            value = (value << 1) | int(pixels[offset + col] > pixels[offset + col + 1])
    return value


def hamming(a: int, b: int) -> int:
    """Return bitwise Hamming distance between two integer hashes."""
    return (int(a) ^ int(b)).bit_count()


def _jaccard(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


CHROME_TOKENS = {
    "backlog",
    "brainstorm",
    "browser",
    "chrome",
    "dashboard",
    "echo",
    "echoteam",
    "figm",
    "figma",
    "file",
    "home",
    "microsoft",
    "teams",
    "tab",
    "tabs",
    "window",
}
STATUS_TOKENS = {
    "approved",
    "approve",
    "complete",
    "completed",
    "confirmed",
    "draft",
    "pending",
    "rejected",
    "submitted",
}
MARKER_RE = re.compile(r"^(page|option|section|step|status|revision)?\d+[a-z]?$")
UUIDISH_RE = re.compile(r"^[0-9a-f]{6,}([0-9a-f-]{4,})?$")


def clean_ocr_token_sets(
    token_sets: Sequence[set[str]],
    df_cutoff: float = 0.65,
) -> list[set[str]]:
    """Drop OCR chrome/noise while preserving page/status/content markers."""
    if not token_sets:
        return []

    doc_freq: dict[str, int] = {}
    for tokens in token_sets:
        for token in tokens:
            doc_freq[token] = doc_freq.get(token, 0) + 1

    n = len(token_sets)
    cleaned: list[set[str]] = []
    for tokens in token_sets:
        kept = set()
        for token in tokens:
            if not token:
                continue
            if token in STATUS_TOKENS or MARKER_RE.match(token):
                kept.add(token)
                continue
            if token in CHROME_TOKENS:
                continue
            if UUIDISH_RE.match(token) and any(ch.isdigit() for ch in token):
                continue
            if "users" in token or "downloads" in token or "http" in token or "www" in token:
                continue
            if len(token) > 36:
                continue
            if doc_freq[token] / n > df_cutoff:
                continue
            kept.add(token)
        cleaned.append(kept)
    return cleaned


def evidence_markers(tokens: set[str]) -> dict[str, set[str]]:
    """Extract page/status/option markers used as conservative merge vetoes."""
    page = {t for t in tokens if t == "page" or t.startswith("page")}
    option = {t for t in tokens if t == "option" or t.startswith("option")}
    section = {t for t in tokens if t == "section" or t.startswith("section")}
    status = tokens & STATUS_TOKENS
    numeric_pages = set()
    if "page" in tokens:
        numeric_pages = {t for t in tokens if t.isdigit() or "/" in t}
    return {
        "page": page | numeric_pages,
        "option": option,
        "section": section,
        "status": status,
    }


def has_differing_evidence(tokens_a: set[str], tokens_b: set[str]) -> bool:
    markers_a = evidence_markers(tokens_a)
    markers_b = evidence_markers(tokens_b)
    for key in ("page", "option", "section", "status"):
        a = markers_a[key]
        b = markers_b[key]
        if a and b and a != b:
            return True
    return False


def _hash_for(candidate: Mapping[str, Any], dhashes: Mapping[int, int] | Sequence[int] | None) -> int | None:
    if "dhash" in candidate:
        return int(candidate["dhash"])
    if dhashes is None:
        return None
    idx = int(candidate["sample_idx"])
    if isinstance(dhashes, Mapping):
        value = dhashes.get(idx)
    else:
        value = dhashes[idx] if 0 <= idx < len(dhashes) else None
    return int(value) if value is not None else None


def _candidate_score(candidate: Mapping[str, Any]) -> tuple[float, float]:
    score = candidate.get("candidate_score", candidate.get("score"))
    if score is None:
        score = candidate.get("sharpness", 0.0)
    return float(score or 0.0), -float(candidate.get("timestamp", 0.0))


def _candidate_information_score(candidate: Mapping[str, Any]) -> tuple[int, float, float]:
    tokens = set(candidate.get("ocr_tokens", []))
    score = candidate.get("candidate_score", candidate.get("score"))
    if score is None:
        score = candidate.get("sharpness", 0.0)
    return len(tokens), float(score or 0.0), -float(candidate.get("timestamp", 0.0))


def _merge_metadata(winner: dict[str, Any], loser: Mapping[str, Any]) -> None:
    winner.setdefault("merged_from_sample_idxs", [winner["sample_idx"]])
    winner.setdefault("merged_timestamps", [winner["timestamp"]])

    loser_idxs = loser.get("merged_from_sample_idxs", [loser["sample_idx"]])
    loser_ts = loser.get("merged_timestamps", [loser["timestamp"]])
    winner["merged_from_sample_idxs"] = sorted(
        {int(idx) for idx in winner["merged_from_sample_idxs"] + list(loser_idxs)}
    )
    winner["merged_timestamps"] = sorted(
        {float(ts) for ts in winner["merged_timestamps"] + list(loser_ts)}
    )


PROTECTIVE_CAPTION_SUBSTRINGS = {
    "approval",
    "dashboard",
    "dialog",
    "document",
    "echo testing",
    "form",
    "modal",
    "page",
    "pdf",
    "screenshot",
    "software interface",
    "status",
    "table",
    "title",
    "webpage",
}

STRONG_PROTECTIVE_CAPTION_SUBSTRINGS = PROTECTIVE_CAPTION_SUBSTRINGS - {"screenshot"}


def _frame_for_candidate(
    candidate: Mapping[str, Any],
    frames: Sequence[Any] | Mapping[int, Any],
) -> Any | None:
    sample_idx = int(candidate["sample_idx"])
    if isinstance(frames, Mapping):
        return frames.get(sample_idx)
    if 0 <= sample_idx < len(frames):
        return frames[sample_idx]
    return None


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


def _has_protective_caption(candidate: Mapping[str, Any]) -> bool:
    caption = str(candidate.get("caption", "")).casefold()
    return any(marker in caption for marker in PROTECTIVE_CAPTION_SUBSTRINGS)


def _has_strong_protective_caption(candidate: Mapping[str, Any]) -> bool:
    caption = str(candidate.get("caption", "")).casefold()
    return any(marker in caption for marker in STRONG_PROTECTIVE_CAPTION_SUBSTRINGS)


def _is_generic_screen_transition(candidate: Mapping[str, Any]) -> bool:
    caption = str(candidate.get("caption", "")).casefold()
    if _has_strong_protective_caption(candidate):
        return False
    return (
        "screenshot of a computer screen" in caption
        or "computer screen with a white background" in caption
        or "computer screen with a black background" in caption
    )


def _has_evidence_markers(tokens: set[str]) -> bool:
    return any(evidence_markers(tokens).values())


def filter_low_information_candidates(
    candidates: Sequence[Mapping[str, Any]],
    frames: Sequence[Any] | Mapping[int, Any],
    min_clean_tokens: int = 3,
) -> list[dict[str, Any]]:
    """Drop blank/avatar-like selected frames only when text and pixels are weak."""
    survivors: list[dict[str, Any]] = []
    rows = sorted(
        (dict(candidate) for candidate in candidates),
        key=lambda c: (float(c.get("timestamp", 0.0)), int(c.get("sample_idx", 0))),
    )

    for row in rows:
        tokens = set(row.get("ocr_tokens", []))
        has_evidence = _has_evidence_markers(tokens)
        has_protective_caption = _has_protective_caption(row)
        has_strong_protective_caption = _has_strong_protective_caption(row)

        image = _frame_for_candidate(row, frames)
        if image is None:
            survivors.append(row)
            continue

        metrics = visual_information_score(image)
        generic_sparse_transition = (
            not has_evidence
            and _is_generic_screen_transition(row)
            and metrics["edge_score"] < 6.0
            and metrics["entropy"] < 2.2
            and (metrics["bright_ratio"] > 0.60 or metrics["dark_ratio"] > 0.60)
        )
        avatar_only = (
            len(tokens) < min_clean_tokens
            and not has_evidence
            and not has_strong_protective_caption
            and metrics["dark_ratio"] > 0.95
            and metrics["edge_score"] < 2.0
            and metrics["entropy"] < 1.0
        )
        if generic_sparse_transition or avatar_only:
            continue

        if len(tokens) >= min_clean_tokens or has_evidence or has_protective_caption:
            survivors.append(row)
            continue

        low_variance = metrics["stddev"] < 12.0
        low_signal = (
            metrics["dark_ratio"] > 0.85
            or metrics["bright_ratio"] > 0.92
            or metrics["edge_score"] < 20.0
        )
        if low_variance and low_signal:
            continue
        survivors.append(row)

    return survivors


def adjacent_same_screen_dedupe(
    candidates: Sequence[Mapping[str, Any]],
    max_dt_seconds: float = 90.0,
    ocr_jaccard_threshold: float = 0.82,
) -> list[dict[str, Any]]:
    """Collapse neighboring candidates with nearly identical cleaned OCR."""
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        row = dict(candidate)
        row["ocr_tokens"] = sorted(set(row.get("ocr_tokens", [])))
        row.setdefault("merged_from_sample_idxs", [row["sample_idx"]])
        row.setdefault("merged_timestamps", [row["timestamp"]])
        rows.append(row)

    rows.sort(key=lambda c: (float(c.get("timestamp", 0.0)), int(c.get("sample_idx", 0))))
    survivors: list[dict[str, Any]] = []

    for row in rows:
        if not survivors:
            survivors.append(row)
            continue

        previous = survivors[-1]
        dt = abs(float(row["timestamp"]) - float(previous["timestamp"]))
        row_tokens = set(row.get("ocr_tokens", []))
        previous_tokens = set(previous.get("ocr_tokens", []))
        should_merge = (
            dt <= max_dt_seconds
            and bool(row_tokens)
            and bool(previous_tokens)
            and _jaccard(row_tokens, previous_tokens) >= ocr_jaccard_threshold
            and not has_differing_evidence(row_tokens, previous_tokens)
            and not _density_asymmetry_veto(row_tokens, previous_tokens)
        )

        if not should_merge:
            survivors.append(row)
            continue

        if _candidate_information_score(row) > _candidate_information_score(previous):
            replacement = row
            _merge_metadata(replacement, previous)
            survivors[-1] = replacement
        else:
            _merge_metadata(previous, row)

    return sorted(survivors, key=lambda c: float(c.get("timestamp", 0.0)))


def near_time_dedupe(
    candidates: Sequence[Mapping[str, Any]],
    ocr_token_sets: Sequence[set[str]] | None = None,
    dhashes: Mapping[int, int] | Sequence[int] | None = None,
    max_dt_seconds: float = 2.0,
    ocr_jaccard_threshold: float = 0.9,
    dhash_hamming_threshold: int = 6,
) -> list[dict[str, Any]]:
    """Collapse near-time duplicate candidates with identical OCR or weak/no-OCR dHash matches."""
    rows: list[dict[str, Any]] = []
    for i, cand in enumerate(candidates):
        row = dict(cand)
        tokens = set(ocr_token_sets[i]) if ocr_token_sets is not None else set(row.get("ocr_tokens", []))
        row["ocr_tokens"] = sorted(tokens)
        row.setdefault("merged_from_sample_idxs", [row["sample_idx"]])
        row.setdefault("merged_timestamps", [row["timestamp"]])
        rows.append(row)

    rows.sort(key=lambda c: (float(c.get("timestamp", 0.0)), int(c.get("sample_idx", 0))))
    survivors: list[dict[str, Any]] = []

    for row in rows:
        duplicate_idx: int | None = None
        row_tokens = set(row.get("ocr_tokens", []))
        row_hash = _hash_for(row, dhashes)

        for i in range(len(survivors) - 1, -1, -1):
            survivor = survivors[i]
            dt = abs(float(row["timestamp"]) - float(survivor["timestamp"]))
            if dt > max_dt_seconds:
                break

            survivor_tokens = set(survivor.get("ocr_tokens", []))
            if row_tokens and survivor_tokens:
                if _jaccard(row_tokens, survivor_tokens) >= ocr_jaccard_threshold:
                    duplicate_idx = i
                    break
                continue

            survivor_hash = _hash_for(survivor, dhashes)
            if row_hash is not None and survivor_hash is not None:
                if hamming(row_hash, survivor_hash) <= dhash_hamming_threshold:
                    duplicate_idx = i
                    break

        if duplicate_idx is None:
            survivors.append(row)
            continue

        survivor = survivors[duplicate_idx]
        if _candidate_score(row) > _candidate_score(survivor):
            replacement = row
            _merge_metadata(replacement, survivor)
            survivors[duplicate_idx] = replacement
        else:
            _merge_metadata(survivor, row)

    return sorted(survivors, key=lambda c: float(c.get("timestamp", 0.0)))


def global_candidate_dedupe(
    candidates: Sequence[Mapping[str, Any]],
    ocr_token_sets: Sequence[set[str]],
    dhashes: Mapping[int, int] | Sequence[int] | None = None,
    ocr_jaccard_threshold: float = 0.85,
    dhash_hamming_threshold: int = 2,
) -> list[dict[str, Any]]:
    """Conservatively collapse duplicate candidates across the whole video."""
    rows: list[dict[str, Any]] = []
    for i, cand in enumerate(candidates):
        row = dict(cand)
        row["ocr_tokens"] = sorted(set(ocr_token_sets[i]))
        row.setdefault("merged_from_sample_idxs", [row["sample_idx"]])
        row.setdefault("merged_timestamps", [row["timestamp"]])
        rows.append(row)

    rows.sort(key=lambda c: (float(c.get("timestamp", 0.0)), int(c.get("sample_idx", 0))))
    survivors: list[dict[str, Any]] = []

    for row in rows:
        row_tokens = set(row.get("ocr_tokens", []))
        row_hash = _hash_for(row, dhashes)
        duplicate_idx: int | None = None

        for i, survivor in enumerate(survivors):
            survivor_tokens = set(survivor.get("ocr_tokens", []))
            if row_tokens and survivor_tokens:
                if has_differing_evidence(row_tokens, survivor_tokens):
                    continue
                if _density_asymmetry_veto(row_tokens, survivor_tokens):
                    continue
                if _jaccard(row_tokens, survivor_tokens) >= ocr_jaccard_threshold:
                    duplicate_idx = i
                    break
                continue

            survivor_hash = _hash_for(survivor, dhashes)
            if row_hash is not None and survivor_hash is not None:
                if hamming(row_hash, survivor_hash) <= dhash_hamming_threshold:
                    duplicate_idx = i
                    break

        if duplicate_idx is None:
            survivors.append(row)
            continue

        survivor = survivors[duplicate_idx]
        if _candidate_score(row) > _candidate_score(survivor):
            replacement = row
            _merge_metadata(replacement, survivor)
            survivors[duplicate_idx] = replacement
        else:
            _merge_metadata(survivor, row)

    return sorted(survivors, key=lambda c: float(c.get("timestamp", 0.0)))


def _density_asymmetry_veto(tokens_a: set[str], tokens_b: set[str]) -> bool:
    if not tokens_a or not tokens_b:
        return False
    large, small = (tokens_a, tokens_b) if len(tokens_a) >= len(tokens_b) else (tokens_b, tokens_a)
    if len(large) < 2 * len(small):
        return False
    return bool(large - small)
