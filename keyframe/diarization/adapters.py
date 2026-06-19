"""Dataset adapter lifecycle and benchmark run-record contracts."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from keyframe.diarization.bundles import ReferenceBundle
from keyframe.diarization.manifests import DatasetManifest, DatasetSplitManifest, dataset_manifest_from_dict
from keyframe.diarization.models import CanonicalRecording, ValidationError


BENCHMARK_RUN_RECORD_SCHEMA_VERSION = 1
BenchmarkExecutionMode = Literal["default_no_network", "dry_run", "full_benchmark"]
ArtifactKind = Literal[
    "canonical_reference",
    "candidate_bundle",
    "report",
    "rttm",
    "uem",
    "raw_provider_json",
    "run_record",
]

_ALLOWED_EXECUTION_MODES = frozenset({"default_no_network", "dry_run", "full_benchmark"})
_ARTIFACT_SUBDIRS: dict[ArtifactKind, str] = {
    "canonical_reference": "references/canonical",
    "candidate_bundle": "candidates",
    "report": "reports",
    "rttm": "exports/rttm",
    "uem": "exports/uem",
    "raw_provider_json": "raw/provider_json",
    "run_record": "run_records",
}


@dataclass(frozen=True)
class DatasetCacheConfig:
    """Local dataset cache and explicit download policy."""

    cache_root: str | None = None
    allow_download: bool = False
    allow_network: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "cache_root", _optional_local_path(self.cache_root, "dataset_cache.cache_root"))
        object.__setattr__(self, "allow_download", _require_bool(self.allow_download, "dataset_cache.allow_download"))
        object.__setattr__(self, "allow_network", _require_bool(self.allow_network, "dataset_cache.allow_network"))
        if self.allow_download and not self.allow_network:
            raise ValidationError("dataset_cache.allow_network is required when allow_download is true")

    @property
    def cache_path(self) -> Path | None:
        return None if self.cache_root is None else Path(self.cache_root)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetPreparationPlan:
    """Dry-run plan for adapter preparation before any external access."""

    dataset_id: str
    split_ids: tuple[str, ...]
    cache_root: str | None
    network_required: bool
    download_required: bool
    actions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_id", _require_id(self.dataset_id, "preparation_plan.dataset_id"))
        object.__setattr__(self, "split_ids", _tuple_of_ids(self.split_ids, "preparation_plan.split_ids"))
        object.__setattr__(self, "cache_root", _optional_local_path(self.cache_root, "preparation_plan.cache_root"))
        object.__setattr__(
            self,
            "network_required",
            _require_bool(self.network_required, "preparation_plan.network_required"),
        )
        object.__setattr__(
            self,
            "download_required",
            _require_bool(self.download_required, "preparation_plan.download_required"),
        )
        object.__setattr__(self, "actions", _tuple_of_text(self.actions, "preparation_plan.actions"))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["actions"] = list(self.actions)
        payload["split_ids"] = list(self.split_ids)
        return payload


@dataclass(frozen=True)
class DatasetValidationResult:
    """Result of validating locally available dataset source files."""

    dataset_id: str
    split_id: str
    valid: bool
    checked_files: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_id", _require_id(self.dataset_id, "validation_result.dataset_id"))
        object.__setattr__(self, "split_id", _require_id(self.split_id, "validation_result.split_id"))
        object.__setattr__(self, "valid", _require_bool(self.valid, "validation_result.valid"))
        object.__setattr__(self, "checked_files", _tuple_of_text(self.checked_files, "validation_result.checked_files"))
        object.__setattr__(self, "errors", _tuple_of_text(self.errors, "validation_result.errors"))
        if self.valid and self.errors:
            raise ValidationError("validation_result.errors must be empty when valid is true")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["checked_files"] = list(self.checked_files)
        payload["errors"] = list(self.errors)
        return payload


@dataclass(frozen=True)
class DatasetExportResult:
    """Reference artifacts exported for central scoring."""

    dataset_id: str
    split_id: str
    reference_bundle: ReferenceBundle
    artifact_paths: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_id", _require_id(self.dataset_id, "export_result.dataset_id"))
        object.__setattr__(self, "split_id", _require_id(self.split_id, "export_result.split_id"))
        if not isinstance(self.reference_bundle, ReferenceBundle):
            raise ValidationError("export_result.reference_bundle must be a ReferenceBundle")
        object.__setattr__(
            self,
            "artifact_paths",
            _validate_string_map(self.artifact_paths, "export_result.artifact_paths"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_paths": dict(self.artifact_paths),
            "dataset_id": self.dataset_id,
            "reference_bundle": self.reference_bundle.to_evaluator_dict(),
            "split_id": self.split_id,
        }


@dataclass(frozen=True)
class BenchmarkArtifactLayout:
    """Deterministic directory layout for one benchmark run."""

    root: str
    canonical_references_dir: str
    candidate_bundles_dir: str
    reports_dir: str
    rttm_dir: str
    uem_dir: str
    raw_provider_json_dir: str
    run_records_dir: str

    def __post_init__(self) -> None:
        for field_name in (
            "root",
            "canonical_references_dir",
            "candidate_bundles_dir",
            "reports_dir",
            "rttm_dir",
            "uem_dir",
            "raw_provider_json_dir",
            "run_records_dir",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_id(getattr(self, field_name), f"artifact_layout.{field_name}"),
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkRunRecord:
    """Audit record for a benchmark run, independent of scoring implementation."""

    run_id: str
    dataset_id: str
    dataset_snapshot: dict[str, Any]
    split_id: str
    branch: str
    artifact_layout: BenchmarkArtifactLayout
    cache_root: str | None
    tuned_split_ids: tuple[str, ...]
    evaluated_split_ids: tuple[str, ...]
    execution_mode: BenchmarkExecutionMode
    no_network: bool
    derived_artifacts: dict[str, str] = field(default_factory=dict)
    schema_version: int = BENCHMARK_RUN_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _validate_run_record_schema_version(self.schema_version))
        object.__setattr__(self, "run_id", _require_id(self.run_id, "run_record.run_id"))
        object.__setattr__(self, "dataset_id", _require_id(self.dataset_id, "run_record.dataset_id"))
        dataset_snapshot = dataset_manifest_from_dict(
            _validate_metadata(self.dataset_snapshot, "run_record.dataset_snapshot")
        )
        object.__setattr__(self, "dataset_snapshot", dataset_snapshot.to_dict())
        if self.dataset_id != dataset_snapshot.dataset_id:
            raise ValidationError("run_record.dataset_id must match dataset_snapshot.dataset_id")
        object.__setattr__(self, "split_id", _require_id(self.split_id, "run_record.split_id"))
        object.__setattr__(self, "branch", _require_id(self.branch, "run_record.branch"))
        if not isinstance(self.artifact_layout, BenchmarkArtifactLayout):
            raise ValidationError("run_record.artifact_layout must be a BenchmarkArtifactLayout")
        object.__setattr__(self, "cache_root", _optional_local_path(self.cache_root, "run_record.cache_root"))
        object.__setattr__(self, "tuned_split_ids", _tuple_of_ids(self.tuned_split_ids, "run_record.tuned_split_ids"))
        object.__setattr__(
            self,
            "evaluated_split_ids",
            _tuple_of_ids(self.evaluated_split_ids, "run_record.evaluated_split_ids"),
        )
        object.__setattr__(self, "execution_mode", _validate_execution_mode(self.execution_mode))
        object.__setattr__(self, "no_network", _require_bool(self.no_network, "run_record.no_network"))
        object.__setattr__(
            self,
            "derived_artifacts",
            _validate_string_map(self.derived_artifacts, "run_record.derived_artifacts"),
        )
        manifest_split_ids = frozenset(split.split_id for split in dataset_snapshot.splits)
        if self.split_id not in manifest_split_ids:
            raise ValidationError(f"run_record.split_id is unknown: {self.split_id}")
        _validate_manifest_split_ids(self.tuned_split_ids, manifest_split_ids, "run_record.tuned_split_ids")
        _validate_manifest_split_ids(self.evaluated_split_ids, manifest_split_ids, "run_record.evaluated_split_ids")
        if self.split_id not in self.evaluated_split_ids:
            raise ValidationError("run_record.split_id must be included in evaluated_split_ids")
        if self.execution_mode in {"default_no_network", "dry_run"} and not self.no_network:
            raise ValidationError("no-network run records must set no_network")
        ensure_adapter_cache_policy(
            dataset_snapshot,
            DatasetCacheConfig(cache_root=self.cache_root),
            execution_mode=self.execution_mode,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_layout": self.artifact_layout.to_dict(),
            "branch": self.branch,
            "cache_root": self.cache_root,
            "dataset_id": self.dataset_id,
            "dataset_snapshot": _thaw_metadata(self.dataset_snapshot),
            "derived_artifacts": dict(self.derived_artifacts),
            "evaluated_split_ids": list(self.evaluated_split_ids),
            "execution_mode": self.execution_mode,
            "no_network": self.no_network,
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "split_id": self.split_id,
            "tuned_split_ids": list(self.tuned_split_ids),
        }


@runtime_checkable
class DatasetAdapter(Protocol):
    """Lifecycle hooks owned by dataset adapters before central scoring."""

    adapter_id: str
    manifest: DatasetManifest

    def describe_splits(self) -> tuple[DatasetSplitManifest, ...]:
        """Return manifest splits available through the adapter."""

    def prepare(self, cache: DatasetCacheConfig, *, download: bool = False) -> DatasetPreparationPlan:
        """Prepare local source material or return a dry-run plan."""

    def validate_source(self, split_id: str, cache: DatasetCacheConfig) -> DatasetValidationResult:
        """Validate source files and checksums for a split."""

    def normalize(self, split_id: str, cache: DatasetCacheConfig) -> tuple[CanonicalRecording, ...]:
        """Normalize source files into canonical recordings."""

    def export_reference(
        self,
        split_id: str,
        recordings: tuple[CanonicalRecording, ...],
        artifact_layout: BenchmarkArtifactLayout,
    ) -> DatasetExportResult:
        """Export reference bundle artifacts for central scoring."""


def build_artifact_layout(root: str | Path) -> BenchmarkArtifactLayout:
    root_path = Path(root)
    root_text = _path_text(root_path)
    return BenchmarkArtifactLayout(
        root=root_text,
        canonical_references_dir=_path_text(root_path / _ARTIFACT_SUBDIRS["canonical_reference"]),
        candidate_bundles_dir=_path_text(root_path / _ARTIFACT_SUBDIRS["candidate_bundle"]),
        reports_dir=_path_text(root_path / _ARTIFACT_SUBDIRS["report"]),
        rttm_dir=_path_text(root_path / _ARTIFACT_SUBDIRS["rttm"]),
        uem_dir=_path_text(root_path / _ARTIFACT_SUBDIRS["uem"]),
        raw_provider_json_dir=_path_text(root_path / _ARTIFACT_SUBDIRS["raw_provider_json"]),
        run_records_dir=_path_text(root_path / _ARTIFACT_SUBDIRS["run_record"]),
    )


def ensure_adapter_cache_policy(
    manifest: DatasetManifest,
    cache: DatasetCacheConfig,
    *,
    execution_mode: BenchmarkExecutionMode = "default_no_network",
) -> None:
    if not isinstance(manifest, DatasetManifest):
        raise ValidationError("manifest must be a DatasetManifest")
    if not isinstance(cache, DatasetCacheConfig):
        raise ValidationError("cache must be a DatasetCacheConfig")
    execution_mode = _validate_execution_mode(execution_mode)
    if execution_mode == "default_no_network" and (cache.allow_download or cache.allow_network):
        raise ValidationError("default benchmark preparation must not allow network or downloads")
    if manifest.access.redistribution == "forbidden" and cache.cache_root is None:
        raise ValidationError("non-redistributable datasets require an explicit local cache path")


def plan_dataset_preparation(
    manifest: DatasetManifest,
    cache: DatasetCacheConfig,
    *,
    download: bool = False,
) -> DatasetPreparationPlan:
    execution_mode: BenchmarkExecutionMode = "full_benchmark" if download else "default_no_network"
    ensure_adapter_cache_policy(manifest, cache, execution_mode=execution_mode)
    if download and not cache.allow_download:
        raise ValidationError("dataset downloads require an explicit allow_download cache policy")
    if download and manifest.access.mode != "public_direct":
        raise ValidationError("dataset downloads require public_direct access")
    split_ids = tuple(split.split_id for split in manifest.splits)
    actions = ["validate_manifest"]
    if cache.cache_root is not None:
        actions.append("validate_cache")
    if download:
        actions.append("download")
    return DatasetPreparationPlan(
        dataset_id=manifest.dataset_id,
        split_ids=split_ids,
        cache_root=cache.cache_root,
        network_required=download,
        download_required=download,
        actions=tuple(actions),
    )


def create_benchmark_run_record(
    *,
    run_id: str,
    manifest: DatasetManifest,
    split_id: str,
    branch: str,
    artifact_root: str | Path,
    cache: DatasetCacheConfig,
    tuned_split_ids: tuple[str, ...] = (),
    evaluated_split_ids: tuple[str, ...] = (),
    execution_mode: BenchmarkExecutionMode = "default_no_network",
    derived_artifacts: dict[str, str] | None = None,
) -> BenchmarkRunRecord:
    ensure_adapter_cache_policy(manifest, cache, execution_mode=execution_mode)
    manifest_split_ids = frozenset(split.split_id for split in manifest.splits)
    if split_id not in manifest_split_ids:
        raise ValidationError(f"run_record.split_id is unknown: {split_id}")
    if not evaluated_split_ids:
        evaluated_split_ids = (split_id,)
    _validate_manifest_split_ids(tuned_split_ids, manifest_split_ids, "run_record.tuned_split_ids")
    _validate_manifest_split_ids(evaluated_split_ids, manifest_split_ids, "run_record.evaluated_split_ids")
    return BenchmarkRunRecord(
        run_id=run_id,
        dataset_id=manifest.dataset_id,
        dataset_snapshot=manifest.to_dict(),
        split_id=split_id,
        branch=branch,
        artifact_layout=build_artifact_layout(artifact_root),
        cache_root=cache.cache_root,
        tuned_split_ids=tuned_split_ids,
        evaluated_split_ids=evaluated_split_ids,
        execution_mode=execution_mode,
        no_network=execution_mode in {"default_no_network", "dry_run"},
        derived_artifacts={} if derived_artifacts is None else derived_artifacts,
    )


def benchmark_run_record_json_dumps(record: BenchmarkRunRecord) -> str:
    if not isinstance(record, BenchmarkRunRecord):
        raise ValidationError("record must be a BenchmarkRunRecord")
    return json.dumps(record.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def benchmark_run_record_json_loads(text: str) -> BenchmarkRunRecord:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"benchmark run record JSON is invalid: {exc.msg}") from exc
    return benchmark_run_record_from_dict(payload)


def benchmark_run_record_from_dict(payload: dict[str, Any]) -> BenchmarkRunRecord:
    data = _require_mapping(payload, "run_record")
    _reject_unknown_fields(
        data,
        {
            "artifact_layout",
            "branch",
            "cache_root",
            "dataset_id",
            "dataset_snapshot",
            "derived_artifacts",
            "evaluated_split_ids",
            "execution_mode",
            "no_network",
            "run_id",
            "schema_version",
            "split_id",
            "tuned_split_ids",
        },
        "run_record",
    )
    dataset_snapshot = dataset_manifest_from_dict(_required(data, "dataset_snapshot", "run_record"))
    dataset_id = _required(data, "dataset_id", "run_record")
    split_id = _required(data, "split_id", "run_record")
    tuned_split_ids = tuple(_sequence(data.get("tuned_split_ids", ())))
    evaluated_split_ids = tuple(_sequence(_required(data, "evaluated_split_ids", "run_record")))
    return BenchmarkRunRecord(
        schema_version=_required(data, "schema_version", "run_record"),
        run_id=_required(data, "run_id", "run_record"),
        dataset_id=dataset_id,
        dataset_snapshot=dataset_snapshot.to_dict(),
        split_id=split_id,
        branch=_required(data, "branch", "run_record"),
        artifact_layout=_artifact_layout_from_dict(_required(data, "artifact_layout", "run_record")),
        cache_root=data.get("cache_root"),
        tuned_split_ids=tuned_split_ids,
        evaluated_split_ids=evaluated_split_ids,
        execution_mode=_required(data, "execution_mode", "run_record"),
        no_network=_required(data, "no_network", "run_record"),
        derived_artifacts=data.get("derived_artifacts", {}),
    )


def read_benchmark_run_record_json(path: str | Path) -> BenchmarkRunRecord:
    return benchmark_run_record_json_loads(Path(path).read_text(encoding="utf-8"))


def write_benchmark_run_record_json(path: str | Path, record: BenchmarkRunRecord) -> None:
    Path(path).write_text(benchmark_run_record_json_dumps(record), encoding="utf-8", newline="\n")


def _artifact_layout_from_dict(payload: object) -> BenchmarkArtifactLayout:
    data = _require_mapping(payload, "artifact_layout")
    _reject_unknown_fields(
        data,
        {
            "canonical_references_dir",
            "candidate_bundles_dir",
            "raw_provider_json_dir",
            "reports_dir",
            "root",
            "rttm_dir",
            "run_records_dir",
            "uem_dir",
        },
        "artifact_layout",
    )
    return BenchmarkArtifactLayout(
        root=_required(data, "root", "artifact_layout"),
        canonical_references_dir=_required(data, "canonical_references_dir", "artifact_layout"),
        candidate_bundles_dir=_required(data, "candidate_bundles_dir", "artifact_layout"),
        reports_dir=_required(data, "reports_dir", "artifact_layout"),
        rttm_dir=_required(data, "rttm_dir", "artifact_layout"),
        uem_dir=_required(data, "uem_dir", "artifact_layout"),
        raw_provider_json_dir=_required(data, "raw_provider_json_dir", "artifact_layout"),
        run_records_dir=_required(data, "run_records_dir", "artifact_layout"),
    )


def _validate_run_record_schema_version(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError("run_record.schema_version must be an integer")
    if value != BENCHMARK_RUN_RECORD_SCHEMA_VERSION:
        raise ValidationError(f"run_record.schema_version is not supported: {value}")
    return value


def _validate_execution_mode(value: object) -> BenchmarkExecutionMode:
    value = _require_id(value, "execution_mode")
    if value not in _ALLOWED_EXECUTION_MODES:
        raise ValidationError(f"execution_mode is not supported: {value}")
    return value  # type: ignore[return-value]


def _validate_manifest_split_ids(values: tuple[str, ...], manifest_split_ids: frozenset[str], field_name: str) -> None:
    for value in values:
        value = _require_id(value, field_name)
        if value not in manifest_split_ids:
            raise ValidationError(f"{field_name} contains unknown split: {value}")


def _optional_local_path(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    value = _require_id(value, field_name)
    if "\n" in value or "\r" in value:
        raise ValidationError(f"{field_name} must be a single-line path")
    return value


def _path_text(path: Path) -> str:
    return path.as_posix()


def _tuple_of_ids(values: object, field_name: str) -> tuple[str, ...]:
    return tuple(_require_id(value, field_name) for value in _sequence(values))


def _tuple_of_text(values: object, field_name: str) -> tuple[str, ...]:
    return tuple(_require_text(value, field_name) for value in _sequence(values))


def _validate_string_map(value: object, field_name: str) -> dict[str, str]:
    data = _require_mapping(value, field_name)
    result: dict[str, str] = {}
    for key, item in data.items():
        result[_require_id(key, f"{field_name}.key")] = _require_id(item, f"{field_name}.{key}")
    return result


def _validate_metadata(value: object, field_name: str) -> dict[str, Any]:
    data = _require_mapping(value, field_name)
    _validate_json_value(data, field_name)
    return data


def _validate_json_value(value: object, field_name: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _require_id(key, f"{field_name}.key")
            _validate_json_value(item, f"{field_name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{field_name}[{index}]")
    elif isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError(f"{field_name} must be finite")
    else:
        raise ValidationError(f"{field_name} must be JSON-serializable")


def _thaw_metadata(value: object) -> object:
    if isinstance(value, dict):
        return {key: _thaw_metadata(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_metadata(item) for item in value]
    if isinstance(value, list):
        return [_thaw_metadata(item) for item in value]
    return value


def _require_id(value: object, field_name: str) -> str:
    if value is None:
        raise ValidationError(f"{field_name} is required")
    if not isinstance(value, str):
        raise ValidationError(f"{field_name} must be a string")
    value = value.strip()
    if not value:
        raise ValidationError(f"{field_name} is required")
    return value


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{field_name} must be a string")
    if not value.strip():
        raise ValidationError(f"{field_name} is required")
    return value


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValidationError(f"{field_name} must be a boolean")
    return value


def _require_mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{context} must be an object")
    return dict(value)


def _required(data: dict[str, Any], field_name: str, context: str) -> object:
    if field_name not in data:
        raise ValidationError(f"{context}.{field_name} is required")
    return data[field_name]


def _sequence(value: object) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValidationError("adapter sequence fields must be arrays")
    return tuple(value)


def _reject_unknown_fields(data: dict[str, Any], allowed_fields: set[str], context: str) -> None:
    unknown = sorted(set(data) - allowed_fields)
    if unknown:
        raise ValidationError(f"{context} has unsupported fields: {', '.join(unknown)}")
