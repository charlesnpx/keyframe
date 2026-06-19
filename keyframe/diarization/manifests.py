"""Dataset manifest contracts for diarization benchmark inputs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from keyframe.diarization.models import ValidationError


DATASET_MANIFEST_SCHEMA_VERSION = 1
DatasetRole = Literal[
    "smoke_ci",
    "public_dev",
    "public_holdout",
    "gated_manual",
    "adversarial",
    "private_in_domain_acceptance",
]
DatasetAccessMode = Literal["public_direct", "public_manual", "auth_required", "local_only", "forbidden"]
RedistributionMode = Literal["allowed", "restricted", "forbidden"]
ExpectedFileRole = Literal["audio", "annotation", "metadata", "manifest", "split", "other"]

ALLOWED_DATASET_ROLES = frozenset(
    {
        "smoke_ci",
        "public_dev",
        "public_holdout",
        "gated_manual",
        "adversarial",
        "private_in_domain_acceptance",
    }
)
ALLOWED_DATASET_ACCESS_MODES = frozenset(
    {
        "public_direct",
        "public_manual",
        "auth_required",
        "local_only",
        "forbidden",
    }
)
ALLOWED_REDISTRIBUTION_MODES = frozenset({"allowed", "restricted", "forbidden"})
ALLOWED_EXPECTED_FILE_ROLES = frozenset({"audio", "annotation", "metadata", "manifest", "split", "other"})
_BENCHMARK_DEFAULT_FULL_ROLES = frozenset({"smoke_ci", "public_dev", "public_holdout", "adversarial"})
_HEX_DIGITS = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class DatasetAccess:
    """How benchmark setup may obtain or use one dataset."""

    mode: DatasetAccessMode
    redistribution: RedistributionMode
    url: str | None = None
    instructions: str | None = None

    def __post_init__(self) -> None:
        mode = _validate_choice(self.mode, ALLOWED_DATASET_ACCESS_MODES, "dataset_access.mode")
        redistribution = _validate_choice(
            self.redistribution,
            ALLOWED_REDISTRIBUTION_MODES,
            "dataset_access.redistribution",
        )
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "redistribution", redistribution)
        object.__setattr__(self, "url", _optional_url(self.url, "dataset_access.url"))
        object.__setattr__(self, "instructions", _optional_text(self.instructions, "dataset_access.instructions"))
        if mode == "public_direct" and self.url is None:
            raise ValidationError("dataset_access.url is required for public_direct access")
        if mode == "forbidden" and redistribution != "forbidden":
            raise ValidationError("forbidden access must also forbid redistribution")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExpectedDatasetFile:
    """A required file plus immutable integrity metadata."""

    path: str
    checksum_sha256: str
    file_role: ExpectedFileRole
    size_bytes: int | None = None
    source_url: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _validate_relative_path(self.path, "expected_file.path"))
        object.__setattr__(self, "checksum_sha256", _validate_sha256(self.checksum_sha256, "expected_file.checksum"))
        object.__setattr__(
            self,
            "file_role",
            _validate_choice(self.file_role, ALLOWED_EXPECTED_FILE_ROLES, "expected_file.file_role"),
        )
        object.__setattr__(self, "size_bytes", _optional_non_negative_int(self.size_bytes, "expected_file.size_bytes"))
        object.__setattr__(self, "source_url", _optional_url(self.source_url, "expected_file.source_url"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScoringPolicyManifest:
    """Named scoring policy referenced by manifest splits."""

    policy_id: str
    version: str
    description: str
    collar_ms: int = 250
    ignore_overlap: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _require_id(self.policy_id, "scoring_policy.policy_id"))
        object.__setattr__(self, "version", _require_id(self.version, "scoring_policy.version"))
        object.__setattr__(self, "description", _require_text(self.description, "scoring_policy.description"))
        object.__setattr__(self, "collar_ms", _non_negative_int(self.collar_ms, "scoring_policy.collar_ms"))
        object.__setattr__(self, "ignore_overlap", _require_bool(self.ignore_overlap, "scoring_policy.ignore_overlap"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetSplitManifest:
    """One benchmark split and the files/policy needed to score it."""

    split_id: str
    role: DatasetRole
    expected_file_paths: tuple[str, ...]
    scoring_policy_id: str
    recording_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "split_id", _require_id(self.split_id, "dataset_split.split_id"))
        object.__setattr__(self, "role", _validate_role(self.role, "dataset_split.role"))
        paths = tuple(
            _validate_relative_path(path, "dataset_split.expected_file_paths")
            for path in _sequence(self.expected_file_paths)
        )
        if not paths:
            raise ValidationError("dataset_split.expected_file_paths is required")
        object.__setattr__(self, "expected_file_paths", paths)
        object.__setattr__(
            self,
            "scoring_policy_id",
            _require_id(self.scoring_policy_id, "dataset_split.scoring_policy_id"),
        )
        object.__setattr__(
            self,
            "recording_ids",
            tuple(
                _require_id(recording_id, "dataset_split.recording_ids")
                for recording_id in _sequence(self.recording_ids)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["expected_file_paths"] = list(self.expected_file_paths)
        payload["recording_ids"] = list(self.recording_ids)
        return payload


@dataclass(frozen=True)
class DatasetManifest:
    """Strict JSON manifest for a benchmark or planned dataset."""

    dataset_id: str
    name: str
    role: DatasetRole
    access: DatasetAccess
    license_url: str | None
    attribution: str | None
    expected_files: tuple[ExpectedDatasetFile, ...]
    splits: tuple[DatasetSplitManifest, ...]
    scoring_policies: tuple[ScoringPolicyManifest, ...]
    benchmarked: bool = True
    schema_version: int = DATASET_MANIFEST_SCHEMA_VERSION
    source_url: str | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _validate_schema_version(self.schema_version))
        object.__setattr__(self, "dataset_id", _require_id(self.dataset_id, "dataset_manifest.dataset_id"))
        object.__setattr__(self, "name", _require_text(self.name, "dataset_manifest.name"))
        object.__setattr__(self, "role", _validate_role(self.role, "dataset_manifest.role"))
        if not isinstance(self.access, DatasetAccess):
            raise ValidationError("dataset_manifest.access must be a DatasetAccess")
        object.__setattr__(self, "license_url", _optional_url(self.license_url, "dataset_manifest.license_url"))
        object.__setattr__(self, "attribution", _optional_text(self.attribution, "dataset_manifest.attribution"))
        object.__setattr__(self, "benchmarked", _require_bool(self.benchmarked, "dataset_manifest.benchmarked"))
        object.__setattr__(self, "source_url", _optional_url(self.source_url, "dataset_manifest.source_url"))
        object.__setattr__(
            self,
            "notes",
            tuple(_require_text(note, "dataset_manifest.notes") for note in _sequence(self.notes)),
        )
        object.__setattr__(self, "expected_files", _as_tuple_of_files(self.expected_files))
        object.__setattr__(self, "splits", _as_tuple_of_splits(self.splits))
        object.__setattr__(self, "scoring_policies", _as_tuple_of_policies(self.scoring_policies))
        self.validate()

    @property
    def default_ci_downloadable(self) -> bool:
        return (
            self.benchmarked
            and self.role == "smoke_ci"
            and self.access.mode == "public_direct"
            and self.access.redistribution == "allowed"
        )

    @property
    def default_full_downloadable(self) -> bool:
        return (
            self.benchmarked
            and self.role in _BENCHMARK_DEFAULT_FULL_ROLES
            and self.access.mode == "public_direct"
            and self.access.redistribution != "forbidden"
        )

    def validate(self) -> None:
        if self.benchmarked:
            if self.license_url is None:
                raise ValidationError("benchmarked dataset_manifest.license_url is required")
            if self.attribution is None:
                raise ValidationError("benchmarked dataset_manifest.attribution is required")
            if not self.expected_files:
                raise ValidationError("benchmarked dataset_manifest.expected_files is required")
            if not self.splits:
                raise ValidationError("benchmarked dataset_manifest.splits is required")
            if not self.scoring_policies:
                raise ValidationError("benchmarked dataset_manifest.scoring_policies is required")

        file_paths = _ensure_unique(tuple(file.path for file in self.expected_files), "expected_file.path")
        policy_ids = _ensure_unique(
            tuple(policy.policy_id for policy in self.scoring_policies),
            "scoring_policy.policy_id",
        )
        _ensure_unique(tuple(split.split_id for split in self.splits), "dataset_split.split_id")
        for split in self.splits:
            if split.scoring_policy_id not in policy_ids:
                raise ValidationError(f"dataset_split.scoring_policy_id is unknown: {split.scoring_policy_id}")
            for path in split.expected_file_paths:
                if path not in file_paths:
                    raise ValidationError(f"dataset_split.expected_file_paths references unknown file: {path}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "access": self.access.to_dict(),
            "attribution": self.attribution,
            "benchmarked": self.benchmarked,
            "dataset_id": self.dataset_id,
            "expected_files": [file.to_dict() for file in self.expected_files],
            "license_url": self.license_url,
            "name": self.name,
            "notes": list(self.notes),
            "role": self.role,
            "schema_version": self.schema_version,
            "scoring_policies": [policy.to_dict() for policy in self.scoring_policies],
            "source_url": self.source_url,
            "splits": [split.to_dict() for split in self.splits],
        }


def dataset_manifest_to_dict(manifest: DatasetManifest) -> dict[str, Any]:
    if not isinstance(manifest, DatasetManifest):
        raise ValidationError("manifest must be a DatasetManifest")
    return manifest.to_dict()


def dataset_manifest_from_dict(payload: dict[str, Any]) -> DatasetManifest:
    data = _require_mapping(payload, "dataset_manifest")
    _reject_unknown_fields(
        data,
        {
            "access",
            "attribution",
            "benchmarked",
            "dataset_id",
            "expected_files",
            "license_url",
            "name",
            "notes",
            "role",
            "schema_version",
            "scoring_policies",
            "source_url",
            "splits",
        },
        "dataset_manifest",
    )
    return DatasetManifest(
        schema_version=_required(data, "schema_version", "dataset_manifest"),
        dataset_id=_required(data, "dataset_id", "dataset_manifest"),
        name=_required(data, "name", "dataset_manifest"),
        role=_required(data, "role", "dataset_manifest"),
        access=_access_from_dict(_required(data, "access", "dataset_manifest")),
        license_url=data.get("license_url"),
        attribution=data.get("attribution"),
        expected_files=tuple(_expected_file_from_dict(item) for item in _sequence(data.get("expected_files", ()))),
        splits=tuple(_split_from_dict(item) for item in _sequence(data.get("splits", ()))),
        scoring_policies=tuple(_scoring_policy_from_dict(item) for item in _sequence(data.get("scoring_policies", ()))),
        benchmarked=data.get("benchmarked", True),
        source_url=data.get("source_url"),
        notes=tuple(_sequence(data.get("notes", ()))),
    )


def dataset_manifest_json_dumps(manifest: DatasetManifest) -> str:
    return json.dumps(dataset_manifest_to_dict(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def dataset_manifest_json_loads(text: str) -> DatasetManifest:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"dataset manifest JSON is invalid: {exc.msg}") from exc
    return dataset_manifest_from_dict(payload)


def read_dataset_manifest_json(path: str | Path) -> DatasetManifest:
    return dataset_manifest_json_loads(Path(path).read_text(encoding="utf-8"))


def write_dataset_manifest_json(path: str | Path, manifest: DatasetManifest) -> None:
    Path(path).write_text(dataset_manifest_json_dumps(manifest), encoding="utf-8", newline="\n")


def manifest_allows_default_ci_download(manifest: DatasetManifest) -> bool:
    if not isinstance(manifest, DatasetManifest):
        raise ValidationError("manifest must be a DatasetManifest")
    return manifest.default_ci_downloadable


def manifest_allows_default_full_download(manifest: DatasetManifest) -> bool:
    if not isinstance(manifest, DatasetManifest):
        raise ValidationError("manifest must be a DatasetManifest")
    return manifest.default_full_downloadable


def _access_from_dict(payload: object) -> DatasetAccess:
    data = _require_mapping(payload, "dataset_access")
    _reject_unknown_fields(data, {"instructions", "mode", "redistribution", "url"}, "dataset_access")
    return DatasetAccess(
        mode=_required(data, "mode", "dataset_access"),
        redistribution=_required(data, "redistribution", "dataset_access"),
        url=data.get("url"),
        instructions=data.get("instructions"),
    )


def _expected_file_from_dict(payload: object) -> ExpectedDatasetFile:
    data = _require_mapping(payload, "expected_file")
    _reject_unknown_fields(
        data,
        {"checksum_sha256", "file_role", "path", "size_bytes", "source_url"},
        "expected_file",
    )
    return ExpectedDatasetFile(
        path=_required(data, "path", "expected_file"),
        checksum_sha256=_required(data, "checksum_sha256", "expected_file"),
        file_role=_required(data, "file_role", "expected_file"),
        size_bytes=data.get("size_bytes"),
        source_url=data.get("source_url"),
    )


def _split_from_dict(payload: object) -> DatasetSplitManifest:
    data = _require_mapping(payload, "dataset_split")
    _reject_unknown_fields(
        data,
        {"expected_file_paths", "recording_ids", "role", "scoring_policy_id", "split_id"},
        "dataset_split",
    )
    return DatasetSplitManifest(
        split_id=_required(data, "split_id", "dataset_split"),
        role=_required(data, "role", "dataset_split"),
        expected_file_paths=tuple(_sequence(_required(data, "expected_file_paths", "dataset_split"))),
        scoring_policy_id=_required(data, "scoring_policy_id", "dataset_split"),
        recording_ids=tuple(_sequence(data.get("recording_ids", ()))),
    )


def _scoring_policy_from_dict(payload: object) -> ScoringPolicyManifest:
    data = _require_mapping(payload, "scoring_policy")
    _reject_unknown_fields(
        data,
        {"collar_ms", "description", "ignore_overlap", "policy_id", "version"},
        "scoring_policy",
    )
    return ScoringPolicyManifest(
        policy_id=_required(data, "policy_id", "scoring_policy"),
        version=_required(data, "version", "scoring_policy"),
        description=_required(data, "description", "scoring_policy"),
        collar_ms=data.get("collar_ms", 250),
        ignore_overlap=data.get("ignore_overlap", False),
    )


def _validate_schema_version(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError("dataset_manifest.schema_version must be an integer")
    if value != DATASET_MANIFEST_SCHEMA_VERSION:
        raise ValidationError(f"dataset_manifest.schema_version is not supported: {value}")
    return value


def _validate_role(value: object, field_name: str) -> DatasetRole:
    return _validate_choice(value, ALLOWED_DATASET_ROLES, field_name)  # type: ignore[return-value]


def _validate_choice(value: object, choices: frozenset[str], field_name: str) -> str:
    value = _require_id(value, field_name)
    if value not in choices:
        raise ValidationError(f"{field_name} is not supported: {value}")
    return value


def _validate_sha256(value: object, field_name: str) -> str:
    value = _require_id(value, field_name)
    if len(value) != 64 or any(character not in _HEX_DIGITS for character in value):
        raise ValidationError(f"{field_name} must be a lowercase sha256 hex digest")
    return value


def _validate_relative_path(value: object, field_name: str) -> str:
    value = _require_id(value, field_name)
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValidationError(f"{field_name} must be a relative path inside the dataset root")
    return value


def _optional_url(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    value = _require_id(value, field_name)
    if not (value.startswith("https://") or value.startswith("http://")):
        raise ValidationError(f"{field_name} must be an http(s) URL")
    return value


def _optional_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_text(value, field_name)


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


def _non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{field_name} must be an integer")
    if value < 0:
        raise ValidationError(f"{field_name} must be >= 0")
    return value


def _optional_non_negative_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    return _non_negative_int(value, field_name)


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
        raise ValidationError("dataset manifest sequence fields must be arrays")
    return tuple(value)


def _reject_unknown_fields(data: dict[str, Any], allowed_fields: set[str], context: str) -> None:
    unknown = sorted(set(data) - allowed_fields)
    if unknown:
        raise ValidationError(f"{context} has unsupported fields: {', '.join(unknown)}")


def _as_tuple_of_files(values: object) -> tuple[ExpectedDatasetFile, ...]:
    try:
        items = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValidationError("dataset_manifest.expected_files must be an iterable") from exc
    for index, item in enumerate(items):
        if not isinstance(item, ExpectedDatasetFile):
            raise ValidationError(f"dataset_manifest.expected_files[{index}] must be an ExpectedDatasetFile")
    return items


def _as_tuple_of_splits(values: object) -> tuple[DatasetSplitManifest, ...]:
    try:
        items = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValidationError("dataset_manifest.splits must be an iterable") from exc
    for index, item in enumerate(items):
        if not isinstance(item, DatasetSplitManifest):
            raise ValidationError(f"dataset_manifest.splits[{index}] must be a DatasetSplitManifest")
    return items


def _as_tuple_of_policies(values: object) -> tuple[ScoringPolicyManifest, ...]:
    try:
        items = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValidationError("dataset_manifest.scoring_policies must be an iterable") from exc
    for index, item in enumerate(items):
        if not isinstance(item, ScoringPolicyManifest):
            raise ValidationError(f"dataset_manifest.scoring_policies[{index}] must be a ScoringPolicyManifest")
    return items


def _ensure_unique(values: tuple[str, ...], field_name: str) -> frozenset[str]:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValidationError(f"duplicate {field_name}: {value}")
        seen.add(value)
    return frozenset(seen)
