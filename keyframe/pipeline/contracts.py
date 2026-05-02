from __future__ import annotations

from dataclasses import dataclass, field, replace
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, TypeVar


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class Stage(Protocol[InputT, OutputT]):
    name: str

    def run(self, inp: InputT, ctx: Any) -> OutputT:
        ...


@dataclass
class FrameStore:
    frames: list[Any]


@dataclass
class SampleTable:
    timestamps: list[float]
    frame_indices: list[int]

    @property
    def sample_count(self) -> int:
        return len(self.timestamps)


@dataclass
class SamplingOutput:
    frame_store: FrameStore
    samples: SampleTable


@dataclass
class FeatureOutput:
    dhashes: list[int]
    clip_embeddings: Any


@dataclass
class TemporalOutput:
    scenes: list[tuple[int, int]]
    scene_coalescence: dict[str, Any]
    cluster_allocs: list[int]
    sample_clusters: dict[int, int]
    sample_scenes: dict[int, int]
    sample_temporal_windows: dict[int, int]


_HEAVY_METADATA_KEYS = {
    "frame",
    "frames",
    "image",
    "images",
    "pil_image",
    "array",
    "ndarray",
    "embedding",
    "embeddings",
    "model",
    "processor",
    "ocr_engine",
}


def _thaw_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _thaw_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw_value(v) for v in value]
    if isinstance(value, frozenset):
        return sorted(_thaw_value(v) for v in value)
    return value


@dataclass(frozen=True)
class CandidateVisualMetadata:
    clip_cluster: int | None = None
    clip_cluster_size: int | None = None
    cluster_role: str | None = None
    cluster_alt_reason: str | None = None
    cluster_diversity_distance: float | None = None
    cluster_sharpness_ratio: float | None = None
    sharpness: float | None = None
    dhash: int | None = None
    dhash_hex: str | None = None


@dataclass(frozen=True)
class CandidateTemporalMetadata:
    scene_id: int | None = None
    dwell_id: int | None = None
    temporal_window_id: int | None = None
    temporal_window_seconds: float | None = None


@dataclass(frozen=True)
class CandidateEvidenceMetadata:
    caption: str | None = None
    caption_source: str | None = None
    ocr_text: str | None = None
    ocr_cache_source: str | None = None
    ocr_tokens: tuple[str, ...] = ()
    dedupe_tokens: tuple[str, ...] = ()
    rescue_tokens: tuple[str, ...] = ()
    raw_token_count: int | None = None
    filtered_token_count: int | None = None
    cleaned_token_count: int | None = None
    cleaning_attrition_ratio: float | None = None


@dataclass(frozen=True)
class CandidateSelectionMetadata:
    candidate_score: float | None = None
    score: float | None = None
    proxy_content_score: float | None = None
    proposal_lane: str | None = None
    end_of_dwell_bonus: float | None = None
    rescue_origin: str | None = None
    rescue_reason: str | None = None
    rescue_priority: int | None = None
    retention_reason: str | None = None
    retention_candidate_reason: str | None = None
    retention_rejected_reason: str | None = None
    low_information_filter_reason: str | None = None
    split_from_generic: bool | None = None


@dataclass(frozen=True)
class CandidateLineageMetadata:
    merged_from_sample_idxs: tuple[int, ...] = ()
    merged_timestamps: tuple[float, ...] = ()
    retention_reasons_seen: tuple[str, ...] = ()
    rescue_origins_seen: tuple[str, ...] = ()
    rescue_priorities_seen: tuple[int, ...] = ()
    lineage_roles: tuple[str, ...] = ()
    dedupe_stage: str | None = None
    merge_reason: str | None = None
    merged_from: int | None = None
    merged_captions: tuple[str, ...] = ()
    caption_cluster: int | None = None


_GROUP_FIELDS = {
    **{field_name: ("visual", field_name) for field_name in CandidateVisualMetadata.__dataclass_fields__},
    **{field_name: ("temporal", field_name) for field_name in CandidateTemporalMetadata.__dataclass_fields__},
    **{field_name: ("evidence", field_name) for field_name in CandidateEvidenceMetadata.__dataclass_fields__},
    **{field_name: ("selection", field_name) for field_name in CandidateSelectionMetadata.__dataclass_fields__},
    **{field_name: ("lineage", field_name) for field_name in CandidateLineageMetadata.__dataclass_fields__},
}


@dataclass(frozen=True)
class CandidateRecord:
    sample_idx: int
    frame_idx: int
    timestamp: float
    origin: str = "proposal"
    visual: CandidateVisualMetadata = field(default_factory=CandidateVisualMetadata)
    temporal: CandidateTemporalMetadata = field(default_factory=CandidateTemporalMetadata)
    evidence: CandidateEvidenceMetadata = field(default_factory=CandidateEvidenceMetadata)
    selection: CandidateSelectionMetadata = field(default_factory=CandidateSelectionMetadata)
    lineage: CandidateLineageMetadata = field(default_factory=CandidateLineageMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_idx", int(self.sample_idx))
        object.__setattr__(self, "frame_idx", int(self.frame_idx))
        object.__setattr__(self, "timestamp", float(self.timestamp))
        if not self.lineage.merged_from_sample_idxs or not self.lineage.merged_timestamps:
            object.__setattr__(
                self,
                "lineage",
                replace(
                    self.lineage,
                    merged_from_sample_idxs=self.lineage.merged_from_sample_idxs or (int(self.sample_idx),),
                    merged_timestamps=self.lineage.merged_timestamps or (float(self.timestamp),),
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        return candidate_to_trace_row(self)

    def with_core(
        self,
        *,
        sample_idx: int | None = None,
        frame_idx: int | None = None,
        timestamp: float | None = None,
        origin: str | None = None,
    ) -> "CandidateRecord":
        return replace(
            self,
            sample_idx=self.sample_idx if sample_idx is None else int(sample_idx),
            frame_idx=self.frame_idx if frame_idx is None else int(frame_idx),
            timestamp=self.timestamp if timestamp is None else float(timestamp),
            origin=self.origin if origin is None else str(origin),
        )

    def with_visual(self, **updates: Any) -> "CandidateRecord":
        return replace(self, visual=_replace_group(self.visual, updates))

    def with_temporal(self, **updates: Any) -> "CandidateRecord":
        return replace(self, temporal=_replace_group(self.temporal, updates))

    def with_evidence(self, **updates: Any) -> "CandidateRecord":
        return replace(self, evidence=_replace_group(self.evidence, updates))

    def with_selection(self, **updates: Any) -> "CandidateRecord":
        return replace(self, selection=_replace_group(self.selection, updates))

    def with_lineage(self, **updates: Any) -> "CandidateRecord":
        return replace(self, lineage=_replace_group(self.lineage, updates))


def _normalize_update_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, set):
        return tuple(sorted(value))
    return value


def _replace_group(group: Any, updates: Mapping[str, Any]) -> Any:
    bad = _HEAVY_METADATA_KEYS & set(updates)
    if bad:
        raise ValueError(f"CandidateRecord cannot store heavy artifact field(s): {sorted(bad)}")
    unknown = sorted(set(updates) - set(group.__dataclass_fields__))
    if unknown:
        raise TypeError(f"Unknown {type(group).__name__} field(s): {unknown}")
    return replace(group, **{key: _normalize_update_value(value) for key, value in updates.items()})


def _apply_flat_update(candidate: CandidateRecord, key: str, value: Any) -> CandidateRecord:
    if key in _HEAVY_METADATA_KEYS:
        raise ValueError(f"CandidateRecord cannot store heavy artifact field {key!r}")
    if key == "sample_idx":
        return candidate.with_core(sample_idx=value)
    if key == "frame_idx":
        return candidate.with_core(frame_idx=value)
    if key == "timestamp":
        return candidate.with_core(timestamp=value)
    if key == "origin":
        return candidate.with_core(origin=value)
    if key not in _GROUP_FIELDS:
        raise TypeError(f"Unknown CandidateRecord field {key!r}")
    group_name, field_name = _GROUP_FIELDS[key]
    helper = getattr(candidate, f"with_{group_name}")
    return helper(**{field_name: value})


def as_candidate_record(candidate: Mapping[str, Any] | CandidateRecord, *, origin: str | None = None) -> CandidateRecord:
    if isinstance(candidate, CandidateRecord):
        return candidate if origin is None else candidate.with_core(origin=origin)
    row = dict(candidate)
    sample_idx = int(row.pop("sample_idx"))
    record = CandidateRecord(
        sample_idx=sample_idx,
        frame_idx=int(row.pop("frame_idx", sample_idx)),
        timestamp=float(row.pop("timestamp")),
        origin=str(row.pop("origin", origin or "proposal")),
    )
    for key, value in row.items():
        record = _apply_flat_update(record, key, value)
    return record


def candidate_records(candidates: Sequence[Mapping[str, Any] | CandidateRecord]) -> tuple[CandidateRecord, ...]:
    return tuple(as_candidate_record(candidate) for candidate in candidates)


def _candidate_projection(candidate: CandidateRecord) -> dict[str, Any]:
    row = {
        "sample_idx": candidate.sample_idx,
        "frame_idx": candidate.frame_idx,
        "timestamp": candidate.timestamp,
        "origin": candidate.origin,
    }
    for group in (candidate.visual, candidate.temporal, candidate.evidence, candidate.selection, candidate.lineage):
        for key, value in group.__dict__.items():
            if value is not None and value != ():
                row[key] = _thaw_value(value)
    return row


def candidate_to_trace_row(candidate: CandidateRecord) -> dict[str, Any]:
    return _candidate_projection(candidate)


def candidate_to_manifest_row(candidate: CandidateRecord, *, filename: str | None = None) -> dict[str, Any]:
    row = _candidate_projection(candidate)
    if filename is not None:
        row["path"] = filename
    return row


def candidate_to_caption_log_row(candidate: CandidateRecord, *, path: Path) -> dict[str, Any]:
    row = _candidate_projection(candidate)
    row["path"] = str(path)
    return row


@dataclass(frozen=True)
class CandidateBatch:
    stage: str
    candidates: tuple[CandidateRecord, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProposalOutput:
    candidates: tuple[CandidateRecord, ...]
    rescue_shortlist: tuple[CandidateRecord, ...]
    proxy_rows: list[dict[str, Any]]
    rescue_budget: int
    rescue_ocr_cap: int
    temporal_window_count: int
    scene_count: int
    legacy_proxy_dropped_count: int = 0


@dataclass
class OutputArtifacts:
    caption_log_path: Path
    manifest_path: Path
    manifest_metadata: dict[str, Any]
