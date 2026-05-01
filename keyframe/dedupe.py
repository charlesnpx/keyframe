"""Deterministic frame dedupe helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any

from PIL import Image


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
