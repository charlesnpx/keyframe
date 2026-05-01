"""Shared postmortem fixture checker for pytest and acceptance replay."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


SCHEMA_VERSION = 1
TOKEN_NORMALIZATION_VERSION = 1
TRANSCRIPT_NORMALIZATION_VERSION = 1


class Status(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNRESOLVED_UPSTREAM = "UNRESOLVED_UPSTREAM"


@dataclass(frozen=True)
class CheckResult:
    status: Status
    stage: str
    assertion: str
    rationale: str
    detail: str = ""


def _check_versions(doc: dict[str, Any], label: str) -> list[CheckResult]:
    expected = {
        "schema_version": SCHEMA_VERSION,
        "token_normalization_version": TOKEN_NORMALIZATION_VERSION,
        "transcript_normalization_version": TRANSCRIPT_NORMALIZATION_VERSION,
    }
    results = []
    for key, value in expected.items():
        got = doc.get(key)
        status = Status.PASS if got == value else Status.FAIL
        results.append(CheckResult(status, "schema", f"{label}.{key}", "fixture compatibility", f"expected {value}, got {got}"))
    return results


def _assertion_meta(assertion: dict[str, Any]) -> tuple[str, str]:
    return str(assertion.get("stage", "")), str(assertion.get("rationale", ""))


def check_fixture(pre_p1: dict[str, Any], post_p1: dict[str, Any], expected: dict[str, Any]) -> list[CheckResult]:
    results: list[CheckResult] = []
    results.extend(_check_versions(pre_p1, "pre_p1"))
    results.extend(_check_versions(post_p1, "post_p1"))
    results.extend(_check_versions(expected, "expected"))

    survivors = list(post_p1.get("survivor_sample_idxs", []))
    survivor_set = set(survivors)
    merged_groups = [set(group) for group in post_p1.get("merged_from_sample_idxs", [])]
    sample_to_group = {
        sample_idx: group
        for group in merged_groups
        for sample_idx in group
    }

    for assertion in expected.get("must_appear_sample_idxs", []):
        stage, rationale = _assertion_meta(assertion)
        idx = assertion.get("sample_idx")
        status = Status.PASS if idx in survivor_set else Status.FAIL
        results.append(CheckResult(status, stage, f"must_appear:{idx}", rationale))

    for assertion in expected.get("must_collapse_pairs", []):
        stage, rationale = _assertion_meta(assertion)
        a, b = assertion.get("sample_idxs", [None, None])
        group = sample_to_group.get(a)
        if group is None or b not in sample_to_group:
            status = Status.UNRESOLVED_UPSTREAM
        else:
            status = Status.PASS if b in group else Status.FAIL
        results.append(CheckResult(status, stage, f"must_collapse:{a},{b}", rationale))

    for assertion in expected.get("must_not_merge_pairs", []):
        stage, rationale = _assertion_meta(assertion)
        a, b = assertion.get("sample_idxs", [None, None])
        group_a = sample_to_group.get(a)
        group_b = sample_to_group.get(b)
        if group_a is None or group_b is None:
            status = Status.UNRESOLVED_UPSTREAM
        else:
            status = Status.PASS if group_a is not group_b else Status.FAIL
        results.append(CheckResult(status, stage, f"must_not_merge:{a},{b}", rationale))

    for assertion in expected.get("must_select_representative_from", []):
        stage, rationale = _assertion_meta(assertion)
        choices = set(assertion.get("sample_idxs", []))
        if not choices:
            status = Status.FAIL
        elif not any(choice in sample_to_group for choice in choices):
            status = Status.UNRESOLVED_UPSTREAM
        else:
            status = Status.PASS if survivor_set & choices else Status.FAIL
        results.append(CheckResult(status, stage, f"must_select_from:{sorted(choices)}", rationale))

    min_total = expected.get("min_total_frames")
    if min_total is not None:
        status = Status.PASS if len(survivors) >= int(min_total) else Status.FAIL
        results.append(CheckResult(status, "pipeline", "min_total_frames", "frame count lower bound"))

    max_total = expected.get("max_total_frames")
    if max_total is not None:
        status = Status.PASS if len(survivors) <= int(max_total) else Status.FAIL
        results.append(CheckResult(status, "pipeline", "max_total_frames", "frame count upper bound"))

    return results


def assert_no_failures(results: list[CheckResult]) -> None:
    failures = [r for r in results if r.status == Status.FAIL]
    assert not failures, "\n".join(f"{r.status} {r.assertion}: {r.detail or r.rationale}" for r in failures)
