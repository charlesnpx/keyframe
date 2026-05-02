"""Lightweight OCR evidence signatures for retention and dedupe policy."""

from __future__ import annotations

import re
from collections.abc import Iterable
from hashlib import blake2b


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9._/-]*", re.IGNORECASE)
DATE_VALUE_RE = re.compile(
    r"\b(?:\d{1,2}[a-z]{3}\d{4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b",
    re.IGNORECASE,
)
PAGE_LINE_RE = re.compile(r"\bpage\s*(\d+[a-z]?)\b", re.IGNORECASE)
LABEL_VALUE_RE = re.compile(r"\s*[:=]\s*")

STATUS_WORDS = {"approved", "approve", "complete", "completed", "draft", "pending", "rejected", "submitted"}
VALUE_WORDS = {"false", "na", "n/a", "no", "none", "true", "yes"} | STATUS_WORDS
REFERENCE_WORDS = {
    "attachment",
    "artifact",
    "design",
    "doc",
    "document",
    "file",
    "form",
    "image",
    "link",
    "mockup",
    "pdf",
    "record",
    "reference",
    "report",
    "screenshot",
    "source",
    "spreadsheet",
    "url",
}


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


def _value_signature(token: str) -> str:
    if token in STATUS_WORDS:
        return f"status:{token}"
    if DATE_VALUE_RE.fullmatch(token):
        return f"date:{token}"
    if token.startswith("page") and len(token) > 4:
        return f"page:{token[4:]}"
    return _stable_signature("value", (token,))


def _add_label_value_signatures(signatures: set[str], label_tokens: tuple[str, ...], value_tokens: tuple[str, ...]) -> None:
    if not label_tokens or not value_tokens:
        return
    label_signature = _stable_signature("label", label_tokens[:8])
    value_signatures = tuple(_value_signature(token) for token in value_tokens if _looks_like_value_token(token))
    if not value_signatures:
        return
    signatures.add(label_signature)
    for value in value_signatures[:4]:
        signatures.add(f"label-value:{label_signature.removeprefix('label:')}:{value}")


def _is_heading_like_line(raw_line: str, tokens: tuple[str, ...]) -> bool:
    if not 1 <= len(tokens) <= 8:
        return False
    if any(_looks_like_value_token(token) for token in tokens):
        return False
    stripped = raw_line.strip()
    if len(stripped) > 80 or LABEL_VALUE_RE.search(stripped):
        return False
    letters = [ch for ch in stripped if ch.isalpha()]
    if not letters:
        return False
    uppercase_ratio = sum(1 for ch in letters if ch.isupper()) / len(letters)
    title_tokens = sum(1 for part in stripped.split() if part[:1].isupper())
    return uppercase_ratio >= 0.6 or title_tokens >= max(1, len(tokens) - 1)


def field_section_signatures(text: str, tokens: Iterable[str] = ()) -> tuple[str, ...]:
    haystack = " ".join([str(text or "").casefold(), " ".join(str(t).casefold() for t in tokens)])
    token_set = set(normalize_ocr_tokens(haystack))
    signatures: set[str] = set()

    for status in sorted(token_set & STATUS_WORDS):
        signatures.add(f"status:{status}")
    for value in DATE_VALUE_RE.findall(haystack):
        signatures.add(f"date:{value.casefold()}")
    for page in PAGE_LINE_RE.findall(haystack):
        signatures.add(f"page:{page.casefold()}")
    for token in token_set:
        if token.startswith("page") and len(token) > 4:
            signatures.add(f"page:{token[4:]}")
        if token.startswith("section") and len(token) > 7:
            signatures.add(f"section:{token[7:]}")
    for word in sorted(token_set & REFERENCE_WORDS):
        signatures.add(f"reference:{word}")

    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        line_tokens = normalize_ocr_tokens(line)
        if not line_tokens:
            continue

        parts = LABEL_VALUE_RE.split(line, maxsplit=1)
        if len(parts) == 2:
            _add_label_value_signatures(
                signatures,
                normalize_ocr_tokens(parts[0]),
                normalize_ocr_tokens(parts[1]),
            )
            continue

        if len(line_tokens) >= 2 and _looks_like_value_token(line_tokens[-1]):
            _add_label_value_signatures(signatures, line_tokens[:-1], line_tokens[-1:])
            continue

        if _is_heading_like_line(line, line_tokens):
            signatures.add(_stable_signature("heading", line_tokens[:8]))

    return tuple(sorted(signatures))


def has_signature_delta(
    line_sig_a: Iterable[str],
    field_sig_a: Iterable[str],
    line_sig_b: Iterable[str],
    field_sig_b: Iterable[str],
) -> bool:
    fields_a = set(field_sig_a)
    fields_b = set(field_sig_b)
    if fields_a != fields_b:
        return True
    lines_a = set(line_sig_a)
    lines_b = set(line_sig_b)
    if not lines_a or not lines_b:
        return False
    shared = len(lines_a & lines_b)
    union = len(lines_a | lines_b) or 1
    return (shared / union) < 0.72
