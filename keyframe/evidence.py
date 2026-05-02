"""Lightweight OCR evidence signatures for retention and dedupe policy."""

from __future__ import annotations

import re
from collections.abc import Iterable


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9._/-]*", re.IGNORECASE)
DATE_VALUE_RE = re.compile(r"\b\d{1,2}[a-z]{3}\d{4}\b", re.IGNORECASE)
PAGE_LINE_RE = re.compile(r"\bpage\s*(\d+[a-z]?)\b", re.IGNORECASE)

STATUS_WORDS = {"approved", "approve", "complete", "completed", "draft", "pending", "rejected", "submitted"}
SECTION_WORDS = {
    "consequence",
    "dependencies",
    "comments",
    "description",
    "priority",
    "justification",
    "considerations",
    "summary",
    "changes",
}
FIELD_PHRASES = {
    "amot revision": ("field:amot_revision", ("amot", "revision")),
    "amot revision date": ("field:amot_revision_date", ("amot", "revision", "date")),
    "approved date": ("field:approved_date", ("approved", "date")),
    "cover page revision": ("field:cover_page_revision", ("cover", "page", "revision")),
    "override justification": ("section:override_justification", ("override", "justification")),
    "priority form": ("section:priority_form", ("priority", "form")),
    "risk justification": ("section:risk_justification", ("risk", "justification")),
    "signed on behalf": ("field:signed_on_behalf", ("signed", "behalf")),
    "source": ("reference:source", ("source",)),
    "status date": ("field:status_date", ("status", "date")),
}
REFERENCE_WORDS = {"design", "figma", "mockup", "pdf", "reference", "source"}


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
    for name, (_label, phrase_tokens) in FIELD_PHRASES.items():
        if all(part in token_set for part in phrase_tokens):
            signatures.add(FIELD_PHRASES[name][0])
    for word in sorted(token_set & SECTION_WORDS):
        signatures.add(f"section-token:{word}")
    for word in sorted(token_set & REFERENCE_WORDS):
        signatures.add(f"reference:{word}")
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
