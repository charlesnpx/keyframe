"""Deterministic frame dedupe helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import re
from typing import Any

from PIL import Image, ImageStat

from keyframe.pipeline.contracts import candidate_records


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
PAGE_RE = re.compile(r"^page(\d+[a-z]?)?$")
OPTION_RE = re.compile(r"^option(\d+[a-z]?)?$")
SECTION_RE = re.compile(r"^section(\d+[a-z]?)?$")
UUIDISH_RE = re.compile(r"^[0-9a-f]{6,}([0-9a-f-]{4,})?$")


def clean_ocr_token_sets(
    token_sets: Sequence[set[str]],
    df_cutoff: float = 0.65,
) -> list[set[str]]:
    """Drop OCR chrome/noise while preserving page/status/content markers."""
    if not token_sets:
        return []

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


def canonical_markers(tokens: set[str]) -> dict[str, set[str]]:
    """Return normalized evidence markers for retention/rescue ranking only."""
    page: set[str] = set()
    option: set[str] = set()
    section: set[str] = set()
    status = set(tokens) & STATUS_TOKENS

    has_page_word = "page" in tokens
    for token in tokens:
        if token != "page" and PAGE_RE.match(token):
            page.add(token)
        elif has_page_word and token.isdigit():
            page.add(f"page{token}")

        if token != "option" and OPTION_RE.match(token):
            option.add(token)
        if token != "section" and SECTION_RE.match(token):
            section.add(token)

    return {
        "page": page,
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


def has_evidence_markers(tokens: set[str]) -> bool:
    return any(evidence_markers(tokens).values())


def has_evidence_asymmetry(tokens_a: set[str], tokens_b: set[str]) -> bool:
    return has_evidence_markers(tokens_a) != has_evidence_markers(tokens_b)


def _canonical_marker_category_delta(markers_a: dict[str, set[str]], markers_b: dict[str, set[str]], key: str) -> bool:
    return markers_a[key] != markers_b[key]


def has_meaningful_evidence_for_retention(
    tokens_a: set[str],
    tokens_b: set[str],
    *,
    visual_delta: float | None = None,
    text_density_delta: float | None = None,
) -> bool:
    """Precision-oriented evidence comparison for retention/rescue ranking.

    This intentionally does not replace ``has_differing_evidence``. Merge vetoes
    stay raw and conservative; this helper only decides whether extra frames are
    worth retaining or rescuing.
    """
    markers_a = canonical_markers(tokens_a)
    markers_b = canonical_markers(tokens_b)

    if _canonical_marker_category_delta(markers_a, markers_b, "status"):
        return True

    category_presence_flips = [
        key
        for key in ("page", "option", "section", "status")
        if bool(markers_a[key]) != bool(markers_b[key])
    ]
    if category_presence_flips:
        return True

    changed_categories = [
        key
        for key in ("page", "option", "section", "status")
        if markers_a[key] and markers_b[key] and markers_a[key] != markers_b[key]
    ]
    if len(changed_categories) >= 2:
        return True

    visual_delta = 0.0 if visual_delta is None else abs(float(visual_delta))
    text_density_delta = 0.0 if text_density_delta is None else abs(float(text_density_delta))
    has_supplied_delta = visual_delta >= 0.15 or text_density_delta >= 0.25

    for key in ("page", "option", "section"):
        if key not in changed_categories:
            continue
        token_delta = len(set(tokens_a) ^ set(tokens_b))
        if token_delta >= 3 or has_supplied_delta:
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


RETENTION_REASON_ORDER = {
    "none": 0,
    "protective_caption_asymmetry": 1,
    "evidence_asymmetry": 2,
    "differing_evidence": 3,
}


def _strictest_retention_reason(*reasons: str | None) -> str:
    normalized = [str(reason or "none") for reason in reasons]
    return max(normalized, key=lambda reason: RETENTION_REASON_ORDER.get(reason, 0), default="none")


def _as_sorted_strings(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    try:
        return sorted({str(value) for value in values if value is not None})
    except TypeError:
        return [str(values)]


def is_protected_candidate(candidate: Mapping[str, Any]) -> bool:
    """Return whether dedupe/merge stages should handle a candidate conservatively."""
    return (
        str(candidate.get("retention_reason", "none") or "none") != "none"
        or bool(candidate.get("rescue_origin"))
    )


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

    winner_reason = str(winner.get("retention_reason", "none") or "none")
    loser_reason = str(loser.get("retention_reason", "none") or "none")
    winner["retention_reason"] = _strictest_retention_reason(winner_reason, loser_reason)

    rescue_origins_seen = set(_as_sorted_strings(winner.get("rescue_origins_seen")))
    rescue_origins_seen.update(_as_sorted_strings(loser.get("rescue_origins_seen")))
    if winner.get("rescue_origin"):
        rescue_origins_seen.add(str(winner["rescue_origin"]))
    if loser.get("rescue_origin"):
        rescue_origins_seen.add(str(loser["rescue_origin"]))
    if not winner.get("rescue_origin") and loser.get("rescue_origin"):
        winner["rescue_origin"] = loser.get("rescue_origin")
        winner["rescue_reason"] = loser.get("rescue_reason")
    if rescue_origins_seen:
        winner["rescue_origins_seen"] = sorted(rescue_origins_seen)

    rescue_priorities_seen = set(_as_sorted_strings(winner.get("rescue_priorities_seen")))
    rescue_priorities_seen.update(_as_sorted_strings(loser.get("rescue_priorities_seen")))
    if winner.get("rescue_priority") is not None:
        rescue_priorities_seen.add(str(winner["rescue_priority"]))
    if loser.get("rescue_priority") is not None:
        rescue_priorities_seen.add(str(loser["rescue_priority"]))
    if rescue_priorities_seen:
        winner["rescue_priorities_seen"] = sorted(rescue_priorities_seen)

    reasons_seen = set(_as_sorted_strings(winner.get("retention_reasons_seen")))
    reasons_seen.update(_as_sorted_strings(loser.get("retention_reasons_seen")))
    reasons_seen.add(winner_reason)
    reasons_seen.add(loser_reason)
    winner["retention_reasons_seen"] = sorted(reason for reason in reasons_seen if reason)

    roles = set(_as_sorted_strings(winner.get("lineage_roles")))
    roles.update(_as_sorted_strings(loser.get("lineage_roles")))
    if winner.get("cluster_role"):
        roles.add(str(winner["cluster_role"]))
    if loser.get("cluster_role"):
        roles.add(str(loser["cluster_role"]))
    winner["lineage_roles"] = sorted(role for role in roles if role)


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


def has_protective_caption(candidate: Mapping[str, Any]) -> bool:
    caption = str(candidate.get("caption", "")).casefold()
    return any(marker in caption for marker in PROTECTIVE_CAPTION_SUBSTRINGS)


def has_strong_protective_caption(candidate: Mapping[str, Any]) -> bool:
    caption = str(candidate.get("caption", "")).casefold()
    return any(marker in caption for marker in STRONG_PROTECTIVE_CAPTION_SUBSTRINGS)


def has_protective_caption_asymmetry(
    candidate_a: Mapping[str, Any],
    candidate_b: Mapping[str, Any],
) -> bool:
    return has_strong_protective_caption(candidate_a) != has_strong_protective_caption(candidate_b)


def _has_protective_caption(candidate: Mapping[str, Any]) -> bool:
    return has_protective_caption(candidate)


def _has_strong_protective_caption(candidate: Mapping[str, Any]) -> bool:
    return has_strong_protective_caption(candidate)


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
    return has_evidence_markers(tokens)


def _is_retained_evidence_candidate(candidate: Mapping[str, Any]) -> bool:
    return is_protected_candidate(candidate)


def _ocr_merge_threshold(
    candidate_a: Mapping[str, Any],
    candidate_b: Mapping[str, Any],
    default_threshold: float,
) -> float:
    if _is_retained_evidence_candidate(candidate_a) or _is_retained_evidence_candidate(candidate_b):
        return max(default_threshold, 0.9)
    return default_threshold


def _ocr_policy_allows_merge(
    candidate_a: Mapping[str, Any],
    candidate_b: Mapping[str, Any],
    tokens_a: set[str],
    tokens_b: set[str],
    default_threshold: float,
) -> tuple[bool, str]:
    if has_differing_evidence(tokens_a, tokens_b):
        return False, "differing_evidence"
    if _density_asymmetry_veto(tokens_a, tokens_b):
        return False, "density_asymmetry"

    protected = _is_retained_evidence_candidate(candidate_a) or _is_retained_evidence_candidate(candidate_b)
    if protected:
        if has_evidence_asymmetry(tokens_a, tokens_b):
            return False, "evidence_asymmetry"
        if has_protective_caption_asymmetry(candidate_a, candidate_b):
            return False, "protective_caption_asymmetry"

    threshold = _ocr_merge_threshold(candidate_a, candidate_b, default_threshold)
    if _jaccard(tokens_a, tokens_b) < threshold:
        return False, "ocr_jaccard"
    return True, "ocr_jaccard"


def retain_cluster_alternates(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep dual CLIP representatives only when post-OCR/caption evidence differs."""
    rows = [dict(candidate) for candidate in candidates]
    by_cluster: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        row.setdefault("retention_reason", "none")
        row.setdefault("retention_reasons_seen", [row["retention_reason"]])
        if row.get("cluster_role"):
            row.setdefault("lineage_roles", [row["cluster_role"]])
        by_cluster.setdefault(row.get("clip_cluster", row.get("sample_idx")), []).append(row)

    retained: list[dict[str, Any]] = []
    for group in by_cluster.values():
        group.sort(key=lambda c: (float(c.get("timestamp", 0.0)), int(c.get("sample_idx", 0))))
        primary = next((c for c in group if c.get("cluster_role") == "primary"), None)
        if primary is None:
            primary = next((c for c in group if c.get("cluster_role") in {"single", None}), group[0])

        primary_tokens = set(primary.get("ocr_tokens", []))
        primary["retention_reason"] = str(primary.get("retention_reason", "none") or "none")
        primary.setdefault("retention_candidate_reason", "primary")
        primary.setdefault("retention_rejected_reason", None)
        primary["retention_reasons_seen"] = sorted(set(_as_sorted_strings(primary.get("retention_reasons_seen"))) | {primary["retention_reason"]})
        if primary.get("cluster_role"):
            primary["lineage_roles"] = sorted(set(_as_sorted_strings(primary.get("lineage_roles"))) | {str(primary["cluster_role"])})

        for row in group:
            role = row.get("cluster_role")
            if row is primary or role == "single" or role not in {"alt"}:
                if row is not primary:
                    row["retention_reason"] = str(row.get("retention_reason", "none") or "none")
                    row.setdefault("retention_candidate_reason", "single_or_non_alt")
                    row.setdefault("retention_rejected_reason", None)
                    row["retention_reasons_seen"] = sorted(set(_as_sorted_strings(row.get("retention_reasons_seen"))) | {row["retention_reason"]})
                    if row.get("cluster_role"):
                        row["lineage_roles"] = sorted(set(_as_sorted_strings(row.get("lineage_roles"))) | {str(row["cluster_role"])})
                retained.append(row)
                continue

            alt_tokens = set(row.get("ocr_tokens", []))
            if has_differing_evidence(primary_tokens, alt_tokens):
                reason = "differing_evidence"
                candidate_reason = "raw_differing_evidence"
            elif has_meaningful_evidence_for_retention(primary_tokens, alt_tokens):
                reason = "evidence_asymmetry"
                candidate_reason = "meaningful_evidence_delta"
            elif has_evidence_markers(alt_tokens) and not has_evidence_markers(primary_tokens):
                reason = "evidence_asymmetry"
                candidate_reason = "raw_evidence_asymmetry"
            elif has_strong_protective_caption(row) and not has_strong_protective_caption(primary):
                reason = "protective_caption_asymmetry"
                candidate_reason = "protective_caption_asymmetry"
            else:
                row["retention_reason"] = "none"
                row["retention_candidate_reason"] = "no_meaningful_evidence_delta"
                row["retention_rejected_reason"] = "dropped_no_asymmetry"
                row["retention_reasons_seen"] = sorted(set(_as_sorted_strings(row.get("retention_reasons_seen"))) | {"none"})
                row["dedupe_stage"] = "retain_cluster_alternates"
                row["merge_reason"] = "dropped_no_asymmetry"
                continue

            row["retention_reason"] = reason
            row["retention_candidate_reason"] = candidate_reason
            row["retention_rejected_reason"] = None
            row["retention_reasons_seen"] = sorted(set(_as_sorted_strings(row.get("retention_reasons_seen"))) | {reason})
            row["lineage_roles"] = sorted(set(_as_sorted_strings(row.get("lineage_roles"))) | {"alt"})
            retained.append(row)

    return candidate_records(sorted(retained, key=lambda c: (float(c.get("timestamp", 0.0)), int(c.get("sample_idx", 0)))))


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
            row["low_information_filter_reason"] = "no_frame"
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
        generic_dark_viewer_transition = (
            not has_evidence
            and not has_strong_protective_caption
            and "computer screen with a black background" in str(row.get("caption", "")).casefold()
            and metrics["edge_score"] < 8.0
            and metrics["entropy"] < 2.7
            and metrics["dark_ratio"] > 0.20
        )
        avatar_only = (
            len(tokens) < min_clean_tokens
            and not has_evidence
            and not has_strong_protective_caption
            and metrics["dark_ratio"] > 0.95
            and metrics["edge_score"] < 2.0
            and metrics["entropy"] < 1.0
        )
        if _is_retained_evidence_candidate(row) and (has_evidence or has_strong_protective_caption):
            row["low_information_filter_reason"] = "protected_retained_evidence"
            survivors.append(row)
            continue
        if generic_sparse_transition or generic_dark_viewer_transition or avatar_only:
            row["low_information_filter_reason"] = (
                "generic_sparse_transition"
                if generic_sparse_transition
                else "generic_dark_viewer_transition"
                if generic_dark_viewer_transition
                else "avatar_only"
            )
            continue

        if len(tokens) >= min_clean_tokens or has_evidence or has_protective_caption:
            row["low_information_filter_reason"] = "text_or_protective_signal"
            survivors.append(row)
            continue

        low_variance = metrics["stddev"] < 12.0
        low_signal = (
            metrics["dark_ratio"] > 0.85
            or metrics["bright_ratio"] > 0.92
            or metrics["edge_score"] < 20.0
        )
        if low_variance and low_signal:
            row["low_information_filter_reason"] = "low_variance_low_signal"
            continue
        row["low_information_filter_reason"] = "visual_signal"
        survivors.append(row)

    return candidate_records(survivors)


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
            and _ocr_policy_allows_merge(
                row, previous, row_tokens, previous_tokens, ocr_jaccard_threshold
            )[0]
        )

        if not should_merge:
            survivors.append(row)
            continue

        if _candidate_information_score(row) > _candidate_information_score(previous):
            replacement = row
            replacement["dedupe_stage"] = "adjacent_same_screen_dedupe"
            replacement["merge_reason"] = "ocr_jaccard"
            _merge_metadata(replacement, previous)
            survivors[-1] = replacement
        else:
            previous["dedupe_stage"] = "adjacent_same_screen_dedupe"
            previous["merge_reason"] = "ocr_jaccard"
            _merge_metadata(previous, row)

    return candidate_records(sorted(survivors, key=lambda c: float(c.get("timestamp", 0.0))))


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
                ok, _reason = _ocr_policy_allows_merge(
                    row, survivor, row_tokens, survivor_tokens, ocr_jaccard_threshold
                )
                if ok:
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
            replacement["dedupe_stage"] = "near_time_dedupe"
            replacement["merge_reason"] = "ocr_or_dhash"
            _merge_metadata(replacement, survivor)
            survivors[duplicate_idx] = replacement
        else:
            survivor["dedupe_stage"] = "near_time_dedupe"
            survivor["merge_reason"] = "ocr_or_dhash"
            _merge_metadata(survivor, row)

    return candidate_records(sorted(survivors, key=lambda c: float(c.get("timestamp", 0.0))))


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
                ok, _reason = _ocr_policy_allows_merge(
                    row, survivor, row_tokens, survivor_tokens, ocr_jaccard_threshold
                )
                if ok:
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
            replacement["dedupe_stage"] = "global_candidate_dedupe"
            replacement["merge_reason"] = "ocr_or_dhash"
            _merge_metadata(replacement, survivor)
            survivors[duplicate_idx] = replacement
        else:
            survivor["dedupe_stage"] = "global_candidate_dedupe"
            survivor["merge_reason"] = "ocr_or_dhash"
            _merge_metadata(survivor, row)

    return candidate_records(sorted(survivors, key=lambda c: float(c.get("timestamp", 0.0))))


def _density_asymmetry_veto(tokens_a: set[str], tokens_b: set[str]) -> bool:
    if not tokens_a or not tokens_b:
        return False
    large, small = (tokens_a, tokens_b) if len(tokens_a) >= len(tokens_b) else (tokens_b, tokens_a)
    if len(large) < 2 * len(small):
        return False
    return bool(large - small)
