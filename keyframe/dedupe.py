"""Deterministic frame dedupe helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import re
from typing import Any

from PIL import Image, ImageStat

from keyframe.pipeline.contracts import CandidateRecord, candidate_records


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


def _records(candidates: Sequence[Mapping[str, Any] | CandidateRecord]) -> tuple[CandidateRecord, ...]:
    return candidate_records(candidates)


def _tokens(candidate: CandidateRecord) -> set[str]:
    return set(candidate.evidence.ocr_tokens)


def _caption(candidate: CandidateRecord) -> str:
    return candidate.evidence.caption or ""


def _hash_for(candidate: CandidateRecord, dhashes: Mapping[int, int] | Sequence[int] | None) -> int | None:
    if candidate.visual.dhash is not None:
        return int(candidate.visual.dhash)
    if dhashes is None:
        return None
    idx = int(candidate.sample_idx)
    if isinstance(dhashes, Mapping):
        value = dhashes.get(idx)
    else:
        value = dhashes[idx] if 0 <= idx < len(dhashes) else None
    return int(value) if value is not None else None


def _score_value(candidate: CandidateRecord) -> float:
    score = candidate.selection.candidate_score
    if score is None:
        score = candidate.selection.score
    if score is None:
        score = candidate.visual.sharpness
    return float(score or 0.0)


def _candidate_score(candidate: CandidateRecord) -> tuple[float, float]:
    return _score_value(candidate), -float(candidate.timestamp)


def _candidate_information_score(candidate: CandidateRecord) -> tuple[int, float, float]:
    return len(_tokens(candidate)), _score_value(candidate), -float(candidate.timestamp)


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


def is_protected_candidate(candidate: Mapping[str, Any] | CandidateRecord) -> bool:
    """Return whether dedupe/merge stages should handle a candidate conservatively."""
    candidate = candidate_records((candidate,))[0]
    return (
        str(candidate.selection.retention_reason or "none") != "none"
        or bool(candidate.selection.rescue_origin)
    )


def merge_candidate_lineage(
    winner: CandidateRecord,
    loser: CandidateRecord,
    *,
    stage: str,
    reason: str,
) -> CandidateRecord:
    merged_from_sample_idxs = tuple(sorted({
        *(int(idx) for idx in winner.lineage.merged_from_sample_idxs),
        *(int(idx) for idx in loser.lineage.merged_from_sample_idxs),
    }))
    merged_timestamps = tuple(sorted({
        *(float(ts) for ts in winner.lineage.merged_timestamps),
        *(float(ts) for ts in loser.lineage.merged_timestamps),
    }))

    winner_reason = str(winner.selection.retention_reason or "none")
    loser_reason = str(loser.selection.retention_reason or "none")
    retention_reason = _strictest_retention_reason(winner_reason, loser_reason)

    rescue_origins_seen = set(winner.lineage.rescue_origins_seen) | set(loser.lineage.rescue_origins_seen)
    if winner.selection.rescue_origin:
        rescue_origins_seen.add(str(winner.selection.rescue_origin))
    if loser.selection.rescue_origin:
        rescue_origins_seen.add(str(loser.selection.rescue_origin))

    rescue_priorities_seen = {int(v) for v in winner.lineage.rescue_priorities_seen}
    rescue_priorities_seen.update(int(v) for v in loser.lineage.rescue_priorities_seen)
    if winner.selection.rescue_priority is not None:
        rescue_priorities_seen.add(int(winner.selection.rescue_priority))
    if loser.selection.rescue_priority is not None:
        rescue_priorities_seen.add(int(loser.selection.rescue_priority))

    reasons_seen = set(winner.lineage.retention_reasons_seen) | set(loser.lineage.retention_reasons_seen)
    reasons_seen.update({winner_reason, loser_reason})

    roles = set(winner.lineage.lineage_roles) | set(loser.lineage.lineage_roles)
    if winner.visual.cluster_role:
        roles.add(str(winner.visual.cluster_role))
    if loser.visual.cluster_role:
        roles.add(str(loser.visual.cluster_role))

    selection_updates: dict[str, Any] = {"retention_reason": retention_reason}
    if not winner.selection.rescue_origin and loser.selection.rescue_origin:
        selection_updates["rescue_origin"] = loser.selection.rescue_origin
        selection_updates["rescue_reason"] = loser.selection.rescue_reason

    return winner.with_selection(**selection_updates).with_lineage(
        merged_from_sample_idxs=merged_from_sample_idxs,
        merged_timestamps=merged_timestamps,
        retention_reasons_seen=tuple(sorted(reason for reason in reasons_seen if reason)),
        rescue_origins_seen=tuple(sorted(rescue_origins_seen)),
        rescue_priorities_seen=tuple(sorted(rescue_priorities_seen)),
        lineage_roles=tuple(sorted(role for role in roles if role)),
        dedupe_stage=stage,
        merge_reason=reason,
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
    candidate: CandidateRecord,
    frames: Sequence[Any] | Mapping[int, Any],
) -> Any | None:
    sample_idx = int(candidate.sample_idx)
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


def has_protective_caption(candidate: Mapping[str, Any] | CandidateRecord) -> bool:
    candidate = candidate_records((candidate,))[0]
    caption = str(_caption(candidate)).casefold()
    return any(marker in caption for marker in PROTECTIVE_CAPTION_SUBSTRINGS)


def has_strong_protective_caption(candidate: Mapping[str, Any] | CandidateRecord) -> bool:
    candidate = candidate_records((candidate,))[0]
    caption = str(_caption(candidate)).casefold()
    return any(marker in caption for marker in STRONG_PROTECTIVE_CAPTION_SUBSTRINGS)


def has_protective_caption_asymmetry(
    candidate_a: Mapping[str, Any] | CandidateRecord,
    candidate_b: Mapping[str, Any] | CandidateRecord,
) -> bool:
    return has_strong_protective_caption(candidate_a) != has_strong_protective_caption(candidate_b)


def _has_protective_caption(candidate: CandidateRecord) -> bool:
    return has_protective_caption(candidate)


def _has_strong_protective_caption(candidate: CandidateRecord) -> bool:
    return has_strong_protective_caption(candidate)


def _is_generic_screen_transition(candidate: CandidateRecord) -> bool:
    caption = str(_caption(candidate)).casefold()
    if _has_strong_protective_caption(candidate):
        return False
    return (
        "screenshot of a computer screen" in caption
        or "computer screen with a white background" in caption
        or "computer screen with a black background" in caption
    )


def _has_evidence_markers(tokens: set[str]) -> bool:
    return has_evidence_markers(tokens)


def _is_retained_evidence_candidate(candidate: CandidateRecord) -> bool:
    return is_protected_candidate(candidate)


def _ocr_merge_threshold(
    candidate_a: CandidateRecord,
    candidate_b: CandidateRecord,
    default_threshold: float,
) -> float:
    if _is_retained_evidence_candidate(candidate_a) or _is_retained_evidence_candidate(candidate_b):
        return max(default_threshold, 0.9)
    return default_threshold


def _ocr_policy_allows_merge(
    candidate_a: CandidateRecord,
    candidate_b: CandidateRecord,
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


def retain_cluster_alternates(candidates: Sequence[Mapping[str, Any] | CandidateRecord]) -> tuple[CandidateRecord, ...]:
    """Keep dual CLIP representatives only when post-OCR/caption evidence differs."""
    rows = list(_records(candidates))
    by_cluster: dict[Any, list[CandidateRecord]] = {}
    for row in rows:
        key = row.visual.clip_cluster if row.visual.clip_cluster is not None else row.sample_idx
        by_cluster.setdefault(key, []).append(row)

    retained: list[CandidateRecord] = []
    for group in by_cluster.values():
        group.sort(key=lambda c: (float(c.timestamp), int(c.sample_idx)))
        primary = next((c for c in group if c.visual.cluster_role == "primary"), None)
        if primary is None:
            primary = next((c for c in group if c.visual.cluster_role in {"single", None}), group[0])
        primary_sample_idx = primary.sample_idx

        primary_tokens = _tokens(primary)
        primary = primary.with_selection(
            retention_reason=str(primary.selection.retention_reason or "none"),
            retention_candidate_reason=primary.selection.retention_candidate_reason or "primary",
            retention_rejected_reason=primary.selection.retention_rejected_reason,
        )
        primary_reasons = set(primary.lineage.retention_reasons_seen) | {primary.selection.retention_reason or "none"}
        primary_roles = set(primary.lineage.lineage_roles)
        if primary.visual.cluster_role:
            primary_roles.add(str(primary.visual.cluster_role))
        primary = primary.with_lineage(
            retention_reasons_seen=tuple(sorted(primary_reasons)),
            lineage_roles=tuple(sorted(primary_roles)),
        )

        for row in group:
            if row.sample_idx == primary_sample_idx:
                retained.append(primary)
                continue
            role = row.visual.cluster_role
            if role == "single" or role not in {"alt"}:
                reason = str(row.selection.retention_reason or "none")
                reasons_seen = set(row.lineage.retention_reasons_seen) | {reason}
                roles = set(row.lineage.lineage_roles)
                if row.visual.cluster_role:
                    roles.add(str(row.visual.cluster_role))
                retained.append(
                    row.with_selection(
                        retention_reason=reason,
                        retention_candidate_reason=row.selection.retention_candidate_reason or "single_or_non_alt",
                        retention_rejected_reason=row.selection.retention_rejected_reason,
                    ).with_lineage(
                        retention_reasons_seen=tuple(sorted(reasons_seen)),
                        lineage_roles=tuple(sorted(roles)),
                    )
                )
                continue

            alt_tokens = _tokens(row)
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
                continue

            retained.append(
                row.with_selection(
                    retention_reason=reason,
                    retention_candidate_reason=candidate_reason,
                    retention_rejected_reason=None,
                ).with_lineage(
                    retention_reasons_seen=tuple(sorted(set(row.lineage.retention_reasons_seen) | {reason})),
                    lineage_roles=tuple(sorted(set(row.lineage.lineage_roles) | {"alt"})),
                )
            )

    return tuple(sorted(retained, key=lambda c: (float(c.timestamp), int(c.sample_idx))))


def filter_low_information_candidates(
    candidates: Sequence[Mapping[str, Any] | CandidateRecord],
    frames: Sequence[Any] | Mapping[int, Any],
    min_clean_tokens: int = 3,
) -> tuple[CandidateRecord, ...]:
    """Drop blank/avatar-like selected frames only when text and pixels are weak."""
    survivors: list[CandidateRecord] = []
    rows = sorted(
        _records(candidates),
        key=lambda c: (float(c.timestamp), int(c.sample_idx)),
    )

    for row in rows:
        tokens = _tokens(row)
        has_evidence = _has_evidence_markers(tokens)
        has_protective_caption = _has_protective_caption(row)
        has_strong_protective_caption = _has_strong_protective_caption(row)

        image = _frame_for_candidate(row, frames)
        if image is None:
            survivors.append(row.with_selection(low_information_filter_reason="no_frame"))
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
            and "computer screen with a black background" in str(_caption(row)).casefold()
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
            survivors.append(row.with_selection(low_information_filter_reason="protected_retained_evidence"))
            continue
        if generic_sparse_transition or generic_dark_viewer_transition or avatar_only:
            continue

        if len(tokens) >= min_clean_tokens or has_evidence or has_protective_caption:
            survivors.append(row.with_selection(low_information_filter_reason="text_or_protective_signal"))
            continue

        low_variance = metrics["stddev"] < 12.0
        low_signal = (
            metrics["dark_ratio"] > 0.85
            or metrics["bright_ratio"] > 0.92
            or metrics["edge_score"] < 20.0
        )
        if low_variance and low_signal:
            continue
        survivors.append(row.with_selection(low_information_filter_reason="visual_signal"))

    return tuple(survivors)


def adjacent_same_screen_dedupe(
    candidates: Sequence[Mapping[str, Any] | CandidateRecord],
    max_dt_seconds: float = 90.0,
    ocr_jaccard_threshold: float = 0.82,
) -> tuple[CandidateRecord, ...]:
    """Collapse neighboring candidates with nearly identical cleaned OCR."""
    rows = tuple(c.with_evidence(ocr_tokens=tuple(sorted(_tokens(c)))) for c in _records(candidates))

    rows = tuple(sorted(rows, key=lambda c: (float(c.timestamp), int(c.sample_idx))))
    survivors: list[CandidateRecord] = []

    for row in rows:
        if not survivors:
            survivors.append(row)
            continue

        previous = survivors[-1]
        dt = abs(float(row.timestamp) - float(previous.timestamp))
        row_tokens = _tokens(row)
        previous_tokens = _tokens(previous)
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
            survivors[-1] = merge_candidate_lineage(row, previous, stage="adjacent_same_screen_dedupe", reason="ocr_jaccard")
        else:
            survivors[-1] = merge_candidate_lineage(previous, row, stage="adjacent_same_screen_dedupe", reason="ocr_jaccard")

    return tuple(sorted(survivors, key=lambda c: float(c.timestamp)))


def near_time_dedupe(
    candidates: Sequence[Mapping[str, Any] | CandidateRecord],
    ocr_token_sets: Sequence[set[str]] | None = None,
    dhashes: Mapping[int, int] | Sequence[int] | None = None,
    max_dt_seconds: float = 2.0,
    ocr_jaccard_threshold: float = 0.9,
    dhash_hamming_threshold: int = 6,
) -> tuple[CandidateRecord, ...]:
    """Collapse near-time duplicate candidates with identical OCR or weak/no-OCR dHash matches."""
    rows: list[CandidateRecord] = []
    for i, cand in enumerate(_records(candidates)):
        tokens = set(ocr_token_sets[i]) if ocr_token_sets is not None else _tokens(cand)
        rows.append(cand.with_evidence(ocr_tokens=tuple(sorted(tokens))))

    rows.sort(key=lambda c: (float(c.timestamp), int(c.sample_idx)))
    survivors: list[CandidateRecord] = []

    for row in rows:
        duplicate_idx: int | None = None
        row_tokens = _tokens(row)
        row_hash = _hash_for(row, dhashes)

        for i in range(len(survivors) - 1, -1, -1):
            survivor = survivors[i]
            dt = abs(float(row.timestamp) - float(survivor.timestamp))
            if dt > max_dt_seconds:
                break

            survivor_tokens = _tokens(survivor)
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
            survivors[duplicate_idx] = merge_candidate_lineage(row, survivor, stage="near_time_dedupe", reason="ocr_or_dhash")
        else:
            survivors[duplicate_idx] = merge_candidate_lineage(survivor, row, stage="near_time_dedupe", reason="ocr_or_dhash")

    return tuple(sorted(survivors, key=lambda c: float(c.timestamp)))


def global_candidate_dedupe(
    candidates: Sequence[Mapping[str, Any] | CandidateRecord],
    ocr_token_sets: Sequence[set[str]],
    dhashes: Mapping[int, int] | Sequence[int] | None = None,
    ocr_jaccard_threshold: float = 0.85,
    dhash_hamming_threshold: int = 2,
) -> tuple[CandidateRecord, ...]:
    """Conservatively collapse duplicate candidates across the whole video."""
    rows = [
        cand.with_evidence(ocr_tokens=tuple(sorted(set(ocr_token_sets[i]))))
        for i, cand in enumerate(_records(candidates))
    ]

    rows.sort(key=lambda c: (float(c.timestamp), int(c.sample_idx)))
    survivors: list[CandidateRecord] = []

    for row in rows:
        row_tokens = _tokens(row)
        row_hash = _hash_for(row, dhashes)
        duplicate_idx: int | None = None

        for i, survivor in enumerate(survivors):
            survivor_tokens = _tokens(survivor)
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
            survivors[duplicate_idx] = merge_candidate_lineage(row, survivor, stage="global_candidate_dedupe", reason="ocr_or_dhash")
        else:
            survivors[duplicate_idx] = merge_candidate_lineage(survivor, row, stage="global_candidate_dedupe", reason="ocr_or_dhash")

    return tuple(sorted(survivors, key=lambda c: float(c.timestamp)))


def _density_asymmetry_veto(tokens_a: set[str], tokens_b: set[str]) -> bool:
    if not tokens_a or not tokens_b:
        return False
    large, small = (tokens_a, tokens_b) if len(tokens_a) >= len(tokens_b) else (tokens_b, tokens_a)
    if len(large) < 2 * len(small):
        return False
    return bool(large - small)
