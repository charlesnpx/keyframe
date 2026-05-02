"""Deterministic candidate merge logic."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from keyframe.dedupe import (
    _merge_metadata,
    has_differing_evidence,
    has_evidence_asymmetry,
    has_protective_caption_asymmetry,
    is_protected_candidate,
)
from keyframe.scoring import score_candidate_for_rep
from keyframe.pipeline.contracts import candidate_records


def jaccard_similarity(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def build_ocr_token_sets(filtered_ocr_texts: Sequence[str], normalizer) -> list[set[str]]:
    return [
        {normalizer(w) for w in text.split() if normalizer(w)}
        for text in filtered_ocr_texts
    ]


def jaccard_sim_matrix(token_sets: Sequence[set[str]], has_ocr_mask: Sequence[bool]):
    n = len(token_sets)
    sim = np.zeros((n, n))
    ocr_idxs = np.where(np.array(has_ocr_mask, dtype=bool))[0]
    for i_pos, i in enumerate(ocr_idxs):
        ti = token_sets[i]
        for j in ocr_idxs[i_pos:]:
            tj = token_sets[j]
            sim[i, j] = jaccard_similarity(ti, tj)
            sim[j, i] = sim[i, j]
    return sim


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _density_asymmetry_veto(tokens_a: set[str], tokens_b: set[str]) -> bool:
    if not tokens_a or not tokens_b:
        return False
    large, small = (tokens_a, tokens_b) if len(tokens_a) >= len(tokens_b) else (tokens_b, tokens_a)
    if len(large) < 2 * len(small):
        return False
    return bool(large - small)


def _transcript_overlap(a: set[str], b: set[str]) -> int:
    if not a or not b:
        return 0
    return len(a & b)


def _should_merge(
    cand_a: Mapping[str, Any],
    cand_b: Mapping[str, Any],
    tokens_a: set[str],
    tokens_b: set[str],
    transcript_a: set[str],
    transcript_b: set[str],
    has_ocr_a: bool,
    has_ocr_b: bool,
) -> tuple[bool, str]:
    if has_ocr_a and has_ocr_b:
        jac = jaccard_similarity(tokens_a, tokens_b)
        if has_differing_evidence(tokens_a, tokens_b):
            return False, "ocr-evidence"
        protected = is_protected_candidate(cand_a) or is_protected_candidate(cand_b)
        threshold = 0.9 if protected else 0.75
        if protected and has_evidence_asymmetry(tokens_a, tokens_b):
            return False, "ocr-evidence-asymmetry"
        if protected and has_protective_caption_asymmetry(cand_a, cand_b):
            return False, "ocr-caption-asymmetry"
        if jac < threshold:
            return False, "ocr-jaccard"
        if _density_asymmetry_veto(tokens_a, tokens_b):
            return False, "ocr-density"
    elif has_ocr_a != has_ocr_b:
        return False, "ocr-presence"

    dt = abs(float(cand_a["timestamp"]) - float(cand_b["timestamp"]))
    overlap = _transcript_overlap(transcript_a, transcript_b)
    if dt > 30.0 and overlap < 2:
        return False, "time-transcript"

    if not has_ocr_a and not has_ocr_b and overlap < 2:
        return False, "weak-signal"

    return True, "merge"


def _component_evidence_compatible(
    left: Sequence[int],
    right: Sequence[int],
    candidates: Sequence[Mapping[str, Any]],
    ocr_token_sets: Sequence[set[str]],
) -> bool:
    for i in left:
        for j in right:
            tokens_i = set(ocr_token_sets[i])
            tokens_j = set(ocr_token_sets[j])
            if has_differing_evidence(tokens_i, tokens_j):
                return False
            protected = is_protected_candidate(candidates[i]) or is_protected_candidate(candidates[j])
            if protected and has_evidence_asymmetry(tokens_i, tokens_j):
                return False
            if protected and has_protective_caption_asymmetry(candidates[i], candidates[j]):
                return False
    return True


def union_find_merge(
    candidates: Sequence[Mapping[str, Any]],
    ocr_token_sets: Sequence[set[str]],
    has_ocr: Sequence[bool],
    frames: Sequence[Any] | Mapping[int, Any],
    transcript_token_sets: Sequence[set[str]] | None = None,
) -> tuple[Any, ...]:
    """Merge candidate pairs with explicit vetoes and union-find components."""
    print("\n── Merging via deterministic union-find veto graph ──")
    n = len(candidates)
    uf = _UnionFind(n)
    transcript_token_sets = transcript_token_sets or [set() for _ in candidates]

    for i in range(n):
        for j in range(i + 1, n):
            ok, reason = _should_merge(
                candidates[i], candidates[j],
                set(ocr_token_sets[i]), set(ocr_token_sets[j]),
                set(transcript_token_sets[i]), set(transcript_token_sets[j]),
                bool(has_ocr[i]), bool(has_ocr[j]),
            )
            if ok:
                comp_i = [idx for idx in range(n) if uf.find(idx) == uf.find(i)]
                comp_j = [idx for idx in range(n) if uf.find(idx) == uf.find(j)]
                if not _component_evidence_compatible(comp_i, comp_j, candidates, ocr_token_sets):
                    continue
                print(f"    {candidates[i]['timestamp']:5.1f}s ↔ {candidates[j]['timestamp']:5.1f}s (will merge)")
                uf.union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)

    final: list[dict[str, Any]] = []
    for cid, group_idxs in enumerate(groups.values()):
        def score_idx(idx: int) -> float:
            cand = candidates[idx]
            if "candidate_score" in cand:
                return float(cand["candidate_score"])
            sample_idx = int(cand["sample_idx"])
            image = frames[sample_idx]
            return float(score_candidate_for_rep(cand, image))

        best_idx = max(group_idxs, key=score_idx)
        winner = dict(candidates[best_idx])
        winner["caption_cluster"] = int(cid)
        winner.setdefault("merged_from_sample_idxs", [winner["sample_idx"]])
        winner.setdefault("merged_timestamps", [winner["timestamp"]])
        winner.setdefault("retention_reason", "none")
        winner.setdefault("retention_reasons_seen", [winner["retention_reason"]])
        if winner.get("cluster_role"):
            winner.setdefault("lineage_roles", [winner["cluster_role"]])
        for group_idx in group_idxs:
            if group_idx == best_idx:
                continue
            winner["dedupe_stage"] = "union_find_merge"
            winner["merge_reason"] = "component"
            _merge_metadata(winner, candidates[group_idx])
        winner["merged_from"] = len(winner["merged_from_sample_idxs"])
        winner["merged_captions"] = [candidates[i].get("caption", "") for i in group_idxs]
        final.append(winner)

        tag = "" if len(group_idxs) == 1 else f" (merged {len(group_idxs)} candidates)"
        print(f"  Group {cid}: kept {winner['timestamp']:.1f}s{tag}")

    return candidate_records(sorted(final, key=lambda c: float(c.get("timestamp", 0.0))))
