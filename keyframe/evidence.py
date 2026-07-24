"""Lightweight OCR evidence signatures for retention and dedupe policy."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from hashlib import blake2b
from typing import Any

from keyframe.pipeline.contracts import (
    CandidateRecord,
    as_candidate_record,
    candidate_records,
)


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9._/-]*", re.IGNORECASE)
DATE_VALUE_RE = re.compile(
    r"\b(?:\d{1,2}[a-z]{3}\d{4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b",
    re.IGNORECASE,
)
PAGE_LINE_RE = re.compile(
    r"^\s*page\s*#?\s*(\d+[a-z]?)(?:\s*/\s*\d+)?\s*$",
    re.IGNORECASE,
)
SECTION_LINE_RE = re.compile(
    r"^\s*section\s*#?\s*(\d+[a-z]?)\s*$",
    re.IGNORECASE,
)
EXPLICIT_LABEL_RE = re.compile(
    r"^\s*(?P<label>[^:=]{1,120}?)\s*(?P<delimiter>[:=])\s*(?P<value>.*?)\s*$"
)

STATUS_WORDS = {"approved", "approve", "complete", "completed", "draft", "pending", "rejected", "submitted"}
VALUE_WORDS = {"false", "na", "n/a", "no", "none", "true", "yes"} | STATUS_WORDS
PROSE_LABEL_HEADS = {"given", "note", "then", "when"}
BROWSER_CHROME_LABEL_TOKENS = {"bookmarks", "favourites", "favorites"}


def normalize_ocr_tokens(text: str) -> tuple[str, ...]:
    tokens = []
    for match in TOKEN_RE.finditer(text.casefold()):
        token = match.group(0).strip("._/-")
        if token:
            tokens.append(token)
    return tuple(tokens)


def normalized_ocr_line_signatures(text: str, *, max_lines: int = 80) -> tuple[str, ...]:
    signatures: list[str] = []
    for line in str(text or "").splitlines():
        tokens = normalize_ocr_tokens(line)
        if not tokens:
            continue
        signatures.append(" ".join(tokens[:12]))
        if len(signatures) >= max_lines:
            break
    return tuple(dict.fromkeys(signatures))


def _stable_signature(prefix: str, tokens: Iterable[str]) -> str:
    digest_input = " ".join(tokens).encode("utf-8", errors="ignore")
    digest = blake2b(digest_input, digest_size=5).hexdigest()
    return f"{prefix}:{digest}"


def _looks_like_value_token(token: str) -> bool:
    return (
        token in VALUE_WORDS
        or any(ch.isdigit() for ch in token)
        or DATE_VALUE_RE.fullmatch(token) is not None
    )


def _explicit_label(raw_line: str) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    rendered = str(raw_line or "")
    match = EXPLICIT_LABEL_RE.fullmatch(rendered)
    if match is None:
        return None
    label_tokens = normalize_ocr_tokens(match.group("label"))
    if not 1 <= len(label_tokens) <= 8:
        return None
    # Drive letters, clocks, and similar one-character OCR fragments are not
    # stable field labels.
    if sum(len(token) for token in label_tokens) < 2:
        return None
    folded = rendered.casefold().strip()
    if label_tokens[0] in PROSE_LABEL_HEADS:
        return None
    if set(label_tokens) & BROWSER_CHROME_LABEL_TOKENS:
        return None
    if folded.startswith(("©", "®")):
        return None
    if "/users/" in folded or "\\users\\" in folded or "file://" in folded:
        return None
    if any("/" in token or "\\" in token for token in label_tokens):
        return None
    value_tokens = normalize_ocr_tokens(match.group("value"))
    return label_tokens, value_tokens


def _value_only_line(raw_line: str) -> tuple[str, ...]:
    stripped = str(raw_line or "").strip()
    if not stripped or EXPLICIT_LABEL_RE.fullmatch(stripped):
        return ()
    tokens = normalize_ocr_tokens(stripped)
    if not 1 <= len(tokens) <= 8 or len(stripped) > 120:
        return ()
    return tokens


def _material_value_signature(value_tokens: tuple[str, ...]) -> str | None:
    if not value_tokens or len(value_tokens) > 8:
        return None
    # A single one-character OCR difference is too weak to classify as a
    # structured field-state change.
    if sum(len(token) for token in value_tokens) < 2:
        return None
    # Preserve normalized value text inside the already-published OCR
    # evidence so comparison can distinguish a real value change from a
    # one-character recognition error. TOKEN_RE never emits "~".
    return "~".join(value_tokens)


def _add_material_value_categories(
    signatures: set[str],
    label_tokens: tuple[str, ...],
    value_tokens: tuple[str, ...],
) -> None:
    label_set = set(label_tokens)
    for token in value_tokens:
        if "status" in label_set and token in STATUS_WORDS:
            signatures.add(f"status:{token}")
        if "date" in label_set and DATE_VALUE_RE.fullmatch(token):
            signatures.add(f"date:{token}")
        if "page" in label_set and token.isdigit():
            signatures.add(f"page:{token}")
        if "section" in label_set and token.isdigit():
            signatures.add(f"section:{token}")


def _add_contextual_line_categories(
    signatures: set[str],
    raw_line: str,
) -> None:
    page_match = PAGE_LINE_RE.fullmatch(raw_line)
    if page_match is not None:
        signatures.add(f"page:{page_match.group(1).casefold()}")
    section_match = SECTION_LINE_RE.fullmatch(raw_line)
    if section_match is not None:
        signatures.add(
            f"section:{section_match.group(1).casefold()}"
        )

    tokens = normalize_ocr_tokens(raw_line)
    if not 2 <= len(tokens) <= 8:
        return
    token_set = set(tokens)
    if "status" in token_set:
        for status in sorted(token_set & STATUS_WORDS):
            signatures.add(f"status:{status}")
    if "date" in token_set:
        for token in tokens:
            if DATE_VALUE_RE.fullmatch(token):
                signatures.add(f"date:{token}")


def field_section_signatures(text: str, tokens: Iterable[str] = ()) -> tuple[str, ...]:
    del tokens  # Signatures must come from line-structured OCR, not token bags.
    rendered = str(text or "")
    signatures: set[str] = set()

    lines = rendered.splitlines()
    for line_index, raw_line in enumerate(lines):
        _add_contextual_line_categories(signatures, raw_line)
        parsed = _explicit_label(raw_line)
        if parsed is None:
            continue
        label_tokens, inline_value_tokens = parsed
        label_signature = _stable_signature("label", label_tokens)
        label_id = label_signature.removeprefix("label:")
        signatures.add(label_signature)

        value_tokens = inline_value_tokens
        if (
            not value_tokens
            and line_index + 1 < len(lines)
        ):
            value_tokens = _value_only_line(lines[line_index + 1])

        value_signature = _material_value_signature(value_tokens)
        state = "populated" if value_signature is not None else "blank"
        signatures.add(f"field-state:{label_id}:{state}")
        if value_signature is not None:
            signatures.add(
                "label-value:"
                f"{label_id}:{value_signature}"
            )
            _add_material_value_categories(
                signatures,
                label_tokens,
                value_tokens,
            )

    return tuple(sorted(signatures))


def _prefixed_values(
    signatures: Iterable[str],
    prefix: str,
) -> set[str]:
    marker = f"{prefix}:"
    return {
        signature.removeprefix(marker)
        for signature in signatures
        if str(signature).startswith(marker)
    }


def _label_mapping(
    signatures: Iterable[str],
    prefix: str,
) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    marker = f"{prefix}:"
    for signature in signatures:
        rendered = str(signature)
        if not rendered.startswith(marker):
            continue
        remainder = rendered.removeprefix(marker)
        label_id, separator, value = remainder.partition(":")
        if not separator or not label_id or not value:
            continue
        mapping.setdefault(label_id, set()).add(value)
    return mapping


def _within_one_character_edit(left: str, right: str) -> bool:
    if left == right:
        return True
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) > len(right):
        left, right = right, left
    left_index = 0
    right_index = 0
    edits = 0
    while left_index < len(left) and right_index < len(right):
        if left[left_index] == right[right_index]:
            left_index += 1
            right_index += 1
            continue
        edits += 1
        if edits > 1:
            return False
        if len(left) == len(right):
            left_index += 1
        right_index += 1
    if left_index < len(left) or right_index < len(right):
        edits += 1
    return edits <= 1


def _value_sets_materially_different(
    left: set[str],
    right: set[str],
) -> bool:
    if not left or not right:
        return False
    return not (
        all(
            any(
                _within_one_character_edit(left_value, right_value)
                for right_value in right
            )
            for left_value in left
        )
        and all(
            any(
                _within_one_character_edit(left_value, right_value)
                for left_value in left
            )
            for right_value in right
        )
    )


def structured_delta_categories(
    field_sig_a: Iterable[str],
    field_sig_b: Iterable[str],
) -> tuple[str, ...]:
    """Return only material, explicitly supported structured-state deltas."""
    fields_a = tuple(str(value) for value in field_sig_a)
    fields_b = tuple(str(value) for value in field_sig_b)
    categories: list[str] = []

    states_a = _label_mapping(fields_a, "field-state")
    states_b = _label_mapping(fields_b, "field-state")
    values_a = _label_mapping(fields_a, "label-value")
    values_b = _label_mapping(fields_b, "label-value")
    shared_labels = set(states_a) & set(states_b)
    blank_populated = any(
        states_a[label] != states_b[label]
        for label in shared_labels
    )

    for category in ("status", "date", "page", "section"):
        category_a = _prefixed_values(fields_a, category)
        category_b = _prefixed_values(fields_b, category)
        if category_a == category_b:
            continue
        if category_a and category_b:
            categories.append(category)
        elif category == "date" and blank_populated:
            # A date value appearing in an explicitly shared field is both a
            # date change and a blank/populated state change. Mere OCR
            # absence is not enough for status/page/section categories.
            categories.append(category)

    if blank_populated:
        categories.append("blank_populated")
    if any(
        values_a.get(label)
        and values_b.get(label)
        and _value_sets_materially_different(
            values_a[label],
            values_b[label],
        )
        for label in shared_labels
    ):
        categories.append("same_label_value")
    return tuple(categories)


def structured_signature_change_count(
    field_sig_a: Iterable[str],
    field_sig_b: Iterable[str],
) -> int:
    fields_a = tuple(str(value) for value in field_sig_a)
    fields_b = tuple(str(value) for value in field_sig_b)
    categories = set(
        structured_delta_categories(fields_a, fields_b)
    )
    count = sum(
        len(
            _prefixed_values(fields_a, category)
            ^ _prefixed_values(fields_b, category)
        )
        for category in ("status", "date", "page", "section")
        if category in categories
    )
    states_a = _label_mapping(fields_a, "field-state")
    states_b = _label_mapping(fields_b, "field-state")
    values_a = _label_mapping(fields_a, "label-value")
    values_b = _label_mapping(fields_b, "label-value")
    shared_labels = set(states_a) & set(states_b)
    count += sum(
        states_a[label] != states_b[label]
        for label in shared_labels
    )
    count += sum(
        _value_sets_materially_different(
            values_a.get(label, set()),
            values_b.get(label, set()),
        )
        for label in shared_labels
    )
    return int(count)


def select_structured_comparator(
    rescue: Mapping[str, Any] | CandidateRecord,
    candidates: Sequence[Mapping[str, Any] | CandidateRecord],
) -> CandidateRecord | None:
    """Choose the deterministic selected candidate used for field comparison."""
    rescue_record = as_candidate_record(rescue)
    rescue_scene = rescue_record.temporal.scene_id
    if rescue_scene is None:
        return None

    pool = [
        candidate
        for candidate in candidate_records(candidates)
        if candidate.sample_idx != rescue_record.sample_idx
        and candidate.temporal.scene_id == rescue_scene
    ]
    if not pool:
        return None

    rescue_dwell = rescue_record.temporal.dwell_id
    rescue_window = rescue_record.temporal.temporal_window_id
    return min(
        pool,
        key=lambda candidate: (
            0
            if rescue_dwell is not None
            and candidate.temporal.dwell_id == rescue_dwell
            else 1,
            0
            if rescue_window is not None
            and candidate.temporal.temporal_window_id == rescue_window
            else 1,
            abs(float(candidate.timestamp) - float(rescue_record.timestamp)),
            int(candidate.frame_idx),
            int(candidate.sample_idx),
        ),
    )


def has_signature_delta(
    line_sig_a: Iterable[str],
    field_sig_a: Iterable[str],
    line_sig_b: Iterable[str],
    field_sig_b: Iterable[str],
) -> bool:
    del line_sig_a, line_sig_b
    return bool(structured_delta_categories(field_sig_a, field_sig_b))
