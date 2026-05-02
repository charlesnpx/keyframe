from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from collections.abc import Iterator, Mapping, Sequence
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


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze_value(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_value(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_freeze_value(v) for v in value)
    if isinstance(value, set):
        return frozenset(_freeze_value(v) for v in value)
    return value


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
    proxy_content_score: float | None = None
    rescue_origin: str | None = None
    rescue_reason: str | None = None
    rescue_priority: int | None = None
    retention_reason: str | None = None
    retention_candidate_reason: str | None = None
    retention_rejected_reason: str | None = None
    low_information_filter_reason: str | None = None


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
class CandidateRecord(Mapping[str, Any]):
    sample_idx: int
    frame_idx: int
    timestamp: float
    origin: str = "proposal"
    visual: CandidateVisualMetadata = field(default_factory=CandidateVisualMetadata)
    temporal: CandidateTemporalMetadata = field(default_factory=CandidateTemporalMetadata)
    evidence: CandidateEvidenceMetadata = field(default_factory=CandidateEvidenceMetadata)
    selection: CandidateSelectionMetadata = field(default_factory=CandidateSelectionMetadata)
    lineage: CandidateLineageMetadata = field(default_factory=CandidateLineageMetadata)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        bad = _HEAVY_METADATA_KEYS & set(self.metadata)
        if bad:
            raise ValueError(f"CandidateRecord metadata cannot contain heavy artifacts: {sorted(bad)}")
        object.__setattr__(self, "sample_idx", int(self.sample_idx))
        object.__setattr__(self, "frame_idx", int(self.frame_idx))
        object.__setattr__(self, "timestamp", float(self.timestamp))
        object.__setattr__(self, "metadata", _freeze_value(dict(self.metadata)))
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

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    def __getitem__(self, key: str) -> Any:
        data = self.to_dict()
        if key not in data:
            raise KeyError(key)
        return data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def to_dict(self) -> dict[str, Any]:
        row = {
            "sample_idx": self.sample_idx,
            "frame_idx": self.frame_idx,
            "timestamp": self.timestamp,
            "origin": self.origin,
        }
        for group in (self.visual, self.temporal, self.evidence, self.selection, self.lineage):
            for key, value in group.__dict__.items():
                if value is not None and value != ():
                    row[key] = _thaw_value(value)
        row.update(_thaw_value(self.metadata))
        return row

    def with_updates(self, **updates: Any) -> "CandidateRecord":
        core = {
            "sample_idx": self.sample_idx,
            "frame_idx": self.frame_idx,
            "timestamp": self.timestamp,
            "origin": self.origin,
        }
        groups = {
            "visual": self.visual,
            "temporal": self.temporal,
            "evidence": self.evidence,
            "selection": self.selection,
            "lineage": self.lineage,
        }
        metadata = dict(self.metadata)
        grouped_updates: dict[str, dict[str, Any]] = {key: {} for key in groups}
        for key, value in updates.items():
            if key in _HEAVY_METADATA_KEYS:
                raise ValueError(f"CandidateRecord cannot store heavy artifact field {key!r}")
            if key in core:
                core[key] = value
            elif key in _GROUP_FIELDS:
                group_name, field_name = _GROUP_FIELDS[key]
                if isinstance(value, list):
                    value = tuple(value)
                grouped_updates[group_name][field_name] = value
            else:
                metadata[key] = value
        for group_name, group_updates in grouped_updates.items():
            if group_updates:
                groups[group_name] = replace(groups[group_name], **group_updates)
        return CandidateRecord(**core, **groups, metadata=metadata)


def as_candidate_record(candidate: Mapping[str, Any] | CandidateRecord, *, origin: str | None = None) -> CandidateRecord:
    if isinstance(candidate, CandidateRecord):
        return candidate if origin is None else candidate.with_updates(origin=origin)
    row = dict(candidate)
    return CandidateRecord(
        sample_idx=int(row.pop("sample_idx")),
        frame_idx=int(row.pop("frame_idx", row.get("sample_idx", 0))),
        timestamp=float(row.pop("timestamp")),
        origin=str(row.pop("origin", origin or "proposal")),
    ).with_updates(**row)


def candidate_records(candidates: Sequence[Mapping[str, Any] | CandidateRecord]) -> tuple[CandidateRecord, ...]:
    return tuple(as_candidate_record(candidate) for candidate in candidates)


@dataclass(frozen=True)
class CandidateBatch:
    stage: str
    candidates: tuple[CandidateRecord, ...]


@dataclass
class ProposalOutput:
    candidates: tuple[CandidateRecord, ...]
    rescue_shortlist: tuple[CandidateRecord, ...]
    proxy_rows: list[dict[str, Any]]
    rescue_budget: int


@dataclass
class OutputArtifacts:
    caption_log_path: Path
    manifest_path: Path
    manifest_metadata: dict[str, Any]
