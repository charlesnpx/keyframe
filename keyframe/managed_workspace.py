"""Validated application-owned output workspace and frame recovery state machine.

Keyframe only treats paths beneath ``.keyframe-work`` as disposable.  The
workspace is deliberately fail-closed: every managed entry and frame recovery
generation is classified while the output lock is held before any cleanup,
restoration, or publication mutation occurs.

The threat guarantee covers filesystem state observed while Keyframe owns the
output lock.  Mount points are reported diagnostically; hostile races or mounts
introduced inside a validated tree after classification are outside that
guarantee.
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import shutil
import stat
import uuid
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from keyframe.artifacts import RunStagingPaths, atomic_write_json

LOGGER = logging.getLogger(__name__)

WORKSPACE_DIRECTORY_NAME = ".keyframe-work"
OWNERSHIP_FILENAME = "ownership.json"
RUNS_DIRECTORY_NAME = "runs"
RECOVERY_DIRECTORY_NAME = "recovery"
PUBLIC_FRAMES_DIRECTORY_NAME = "frames"
OPTIONAL_FRAME_TRACE_FILENAMES = frozenset(
    {
        "pipeline_trace.json",
        "debug_qa_trace.json",
    }
)
KNOWN_PUBLIC_FILENAMES = (
    "transcript.raw.json",
    "diarization.json",
    "transcript.txt",
    "transcript.json",
    "transcript.srt",
    "transcript.vtt",
)

_ROOT_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_CANONICAL_UUID4_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_WORKSPACE_ENTRY_NAMES = frozenset(
    {
        OWNERSHIP_FILENAME,
        RUNS_DIRECTORY_NAME,
        RECOVERY_DIRECTORY_NAME,
    }
)


class OutputSessionError(RuntimeError):
    """A controlled failure while owning a Keyframe output directory."""


class ManagedWorkspaceError(OutputSessionError):
    """The application-owned workspace could not be safely classified."""


class FrameGenerationError(OutputSessionError):
    """Base class for controlled frame-generation failures."""


class FrameGenerationValidationError(FrameGenerationError):
    """A frame generation is incomplete, unsafe, or internally inconsistent."""


class FrameGenerationPromotionError(FrameGenerationError):
    """A validated frame generation could not replace the public generation."""


class _HeldOutputLock(Protocol):
    @property
    def is_held(self) -> bool: ...


@dataclass(frozen=True)
class FrameGenerationSnapshot:
    path: Path
    frame_names: tuple[str, ...]
    artifact_paths: tuple[Path, ...]


@dataclass(frozen=True)
class _RecoveryEntry:
    entry_id: uuid.UUID
    state: Literal["empty", "valid"]
    generation: FrameGenerationSnapshot | None


@dataclass(frozen=True)
class _WorkspaceInspection:
    root_id: str
    runs_present: bool
    recovery_present: bool
    run_ids: tuple[uuid.UUID, ...]
    recoveries: tuple[_RecoveryEntry, ...]
    public_generation: FrameGenerationSnapshot | None


def manual_workspace_recovery_instruction(
    output_dir: str | Path,
    workspace_root: str | Path,
) -> str:
    """Return the exact operator procedure for an incomplete managed root."""

    output = Path(output_dir)
    root = Path(workspace_root)
    return (
        f"Manual recovery: rename '{root}' to a review location outside "
        f"'{output}', inspect the renamed directory, then rerun Keyframe. "
        "Keyframe will not modify the incomplete managed workspace."
    )


def parse_canonical_uuid4(value: str | uuid.UUID) -> uuid.UUID:
    """Parse one canonical lowercase UUIDv4 without accepting aliases."""

    if isinstance(value, uuid.UUID):
        parsed = value
        rendered = str(value)
    elif isinstance(value, str):
        rendered = value
        if not _CANONICAL_UUID4_PATTERN.fullmatch(rendered):
            raise ValueError("workspace entry id must be a canonical lowercase UUIDv4")
        try:
            parsed = uuid.UUID(rendered)
        except ValueError as exc:
            raise ValueError(
                "workspace entry id must be a canonical lowercase UUIDv4"
            ) from exc
    else:
        raise TypeError("workspace entry id must be a UUID or canonical UUID string")
    if (
        parsed.version != 4
        or parsed.variant != uuid.RFC_4122
        or rendered != str(parsed)
        or not _CANONICAL_UUID4_PATTERN.fullmatch(rendered)
    ):
        raise ValueError("workspace entry id must be a canonical lowercase UUIDv4")
    return parsed


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FrameGenerationValidationError(
            f"{label} is not readable JSON: {path}: {exc}"
        ) from exc


def _require_regular_file(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise FrameGenerationValidationError(
            f"{label} is missing or unreadable: {path}: {exc}"
        ) from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise FrameGenerationValidationError(
            f"{label} is not a regular non-symlinked file: {path}"
        )


def _indexed_artifact_names(
    rows: Any,
    *,
    field_name: str,
    label: str,
) -> tuple[str, ...]:
    if not isinstance(rows, list):
        raise FrameGenerationValidationError(f"{label} must contain a list")
    names: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise FrameGenerationValidationError(
                f"{label} row {index} must be an object"
            )
        name = row.get(field_name)
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or not name.endswith(".png")
        ):
            raise FrameGenerationValidationError(
                f"{label} row {index} has an invalid {field_name}"
            )
        names.append(name)
    if len(set(names)) != len(names):
        raise FrameGenerationValidationError(f"{label} contains duplicate PNG indexes")
    return tuple(names)


def _relative_artifact_paths(root: Path) -> tuple[Path, ...]:
    """List every observed descendant without following symlinks."""

    found: list[Path] = []

    def visit(directory: Path) -> None:
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError as exc:
            raise FrameGenerationValidationError(
                f"frame generation directory is unreadable: {directory}: {exc}"
            ) from exc
        for entry in entries:
            path = Path(entry.path)
            relative = path.relative_to(root)
            found.append(relative)
            try:
                mode = entry.stat(follow_symlinks=False).st_mode
            except OSError as exc:
                raise FrameGenerationValidationError(
                    f"frame artifact is unreadable: {path}: {exc}"
                ) from exc
            if stat.S_ISDIR(mode) and not stat.S_ISLNK(mode):
                visit(path)

    visit(root)
    return tuple(found)


def inspect_frame_generation(
    generation_dir: str | Path,
    *,
    label: str,
    expected_frame_names: Iterable[str] | None = None,
) -> FrameGenerationSnapshot:
    """Validate one exact, flat frame generation and return its artifacts."""

    generation = Path(generation_dir)
    try:
        generation_mode = generation.lstat().st_mode
    except OSError as exc:
        raise FrameGenerationValidationError(
            f"{label} is missing or unreadable: {generation}: {exc}"
        ) from exc
    if stat.S_ISLNK(generation_mode) or not stat.S_ISDIR(generation_mode):
        raise FrameGenerationValidationError(
            f"{label} is not a regular non-symlinked directory: {generation}"
        )

    captions_path = generation / "captions.json"
    manifest_path = generation / "manifest.json"
    _require_regular_file(captions_path, f"{label} captions.json")
    _require_regular_file(manifest_path, f"{label} manifest.json")

    caption_names = _indexed_artifact_names(
        _read_json(captions_path, f"{label} captions.json"),
        field_name="file",
        label=f"{label} captions.json",
    )
    manifest = _read_json(manifest_path, f"{label} manifest.json")
    if (
        not isinstance(manifest, Mapping)
        or isinstance(manifest.get("schema_version"), bool)
        or manifest.get("schema_version") != 1
    ):
        raise FrameGenerationValidationError(
            f"{label} manifest.json must be a schema_version 1 object"
        )
    manifest_names = _indexed_artifact_names(
        manifest.get("frames"),
        field_name="filename",
        label=f"{label} manifest.json frames",
    )
    if Counter(caption_names) != Counter(manifest_names):
        raise FrameGenerationValidationError(
            f"{label} captions.json and manifest.json index different PNG sets: "
            f"captions={sorted(caption_names)}, manifest={sorted(manifest_names)}"
        )

    indexed_names = tuple(manifest_names)
    if expected_frame_names is not None:
        expected = tuple(expected_frame_names)
        if len(set(expected)) != len(expected):
            raise FrameGenerationValidationError(
                f"{label} contains duplicate expected PNG filenames"
            )
        for name in expected:
            if (
                not isinstance(name, str)
                or Path(name).name != name
                or not name.endswith(".png")
            ):
                raise FrameGenerationValidationError(
                    f"{label} has an invalid expected PNG filename: {name!r}"
                )
        if Counter(indexed_names) != Counter(expected):
            raise FrameGenerationValidationError(
                f"{label} does not index the selected frame generation: "
                f"expected={sorted(expected)}, actual={sorted(indexed_names)}"
            )

    relative_paths = _relative_artifact_paths(generation)
    allowed_names = {
        "captions.json",
        "manifest.json",
        *OPTIONAL_FRAME_TRACE_FILENAMES,
        *indexed_names,
    }
    unknown_paths = tuple(
        sorted(
            relative.as_posix()
            for relative in relative_paths
            if len(relative.parts) != 1 or relative.name not in allowed_names
        )
    )
    if unknown_paths:
        raise FrameGenerationValidationError(
            f"{label} contains unknown artifact paths: {list(unknown_paths)}"
        )

    direct_names = tuple(
        sorted(relative.name for relative in relative_paths if len(relative.parts) == 1)
    )
    actual_png_names = tuple(name for name in direct_names if name.endswith(".png"))
    if Counter(actual_png_names) != Counter(indexed_names):
        raise FrameGenerationValidationError(
            f"{label} PNG set does not match its indexes: "
            f"expected={sorted(indexed_names)}, actual={sorted(actual_png_names)}"
        )

    from PIL import Image

    for name in indexed_names:
        path = generation / name
        _require_regular_file(path, f"{label} frame {name}")
        try:
            with Image.open(path) as image:
                if image.format != "PNG":
                    raise ValueError(f"detected format {image.format!r}")
                image.verify()
        except Exception as exc:
            raise FrameGenerationValidationError(
                f"{label} frame is not a valid PNG: {path}: {exc}"
            ) from exc

    for trace_name in OPTIONAL_FRAME_TRACE_FILENAMES:
        trace_path = generation / trace_name
        if os.path.lexists(trace_path):
            _require_regular_file(trace_path, f"{label} {trace_name}")

    artifacts = tuple(generation / name for name in direct_names)
    return FrameGenerationSnapshot(
        path=generation,
        frame_names=indexed_names,
        artifact_paths=artifacts,
    )


def _require_directory(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise ManagedWorkspaceError(f"{label} is unreadable: {path}: {exc}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise ManagedWorkspaceError(
            f"{label} is not a regular non-symlinked directory: {path}"
        )


def _validate_disposable_tree(root: Path, label: str) -> None:
    """Require a directory tree containing only real directories and files."""

    _require_directory(root, label)

    def visit(directory: Path) -> None:
        if os.path.ismount(directory):
            LOGGER.warning("Managed workspace tree contains mount point: %s", directory)
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError as exc:
            raise ManagedWorkspaceError(
                f"{label} is unreadable: {directory}: {exc}"
            ) from exc
        for entry in entries:
            path = Path(entry.path)
            try:
                mode = entry.stat(follow_symlinks=False).st_mode
            except OSError as exc:
                raise ManagedWorkspaceError(
                    f"{label} entry is unreadable: {path}: {exc}"
                ) from exc
            if stat.S_ISLNK(mode):
                raise ManagedWorkspaceError(f"{label} contains a symlink: {path}")
            if stat.S_ISDIR(mode):
                visit(path)
            elif not stat.S_ISREG(mode):
                raise ManagedWorkspaceError(
                    f"{label} contains a non-regular entry: {path}"
                )

    visit(root)


def _ownership_root_id(sentinel: Path) -> str:
    try:
        _require_regular_file(sentinel, "managed workspace ownership sentinel")
    except FrameGenerationValidationError as exc:
        raise ManagedWorkspaceError(str(exc)) from exc
    try:
        payload = json.loads(sentinel.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManagedWorkspaceError(
            f"managed workspace ownership sentinel is invalid: {sentinel}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ManagedWorkspaceError(
            f"managed workspace ownership sentinel must be an object: {sentinel}"
        )
    expected_keys = {"schema_version", "application", "purpose", "root_id"}
    root_id = payload.get("root_id")
    if (
        set(payload) != expected_keys
        or isinstance(payload.get("schema_version"), bool)
        or payload.get("schema_version") != 1
        or payload.get("application") != "keyframe"
        or payload.get("purpose") != "managed-output-workspace"
        or not isinstance(root_id, str)
        or not _ROOT_ID_PATTERN.fullmatch(root_id)
    ):
        raise ManagedWorkspaceError(
            "managed workspace ownership sentinel must contain exactly "
            "schema_version=1, application='keyframe', "
            "purpose='managed-output-workspace', and a 32-character lowercase "
            f"hexadecimal root_id: {sentinel}"
        )
    return root_id


def _entry_uuid(path: Path, label: str) -> uuid.UUID:
    try:
        return parse_canonical_uuid4(path.name)
    except (TypeError, ValueError) as exc:
        raise ManagedWorkspaceError(
            f"{label} has a non-canonical UUIDv4 name: {path.name!r}"
        ) from exc


def _inspect_recovery_entry(path: Path) -> _RecoveryEntry:
    entry_id = _entry_uuid(path, "managed recovery entry")
    _require_directory(path, "managed recovery entry")
    try:
        entries = tuple(sorted(path.iterdir(), key=lambda item: item.name))
    except OSError as exc:
        raise ManagedWorkspaceError(
            f"managed recovery entry is unreadable: {path}: {exc}"
        ) from exc
    if not entries:
        return _RecoveryEntry(entry_id=entry_id, state="empty", generation=None)
    frames = path / PUBLIC_FRAMES_DIRECTORY_NAME
    if len(entries) != 1 or entries[0].name != PUBLIC_FRAMES_DIRECTORY_NAME:
        relative = [entry.relative_to(path).as_posix() for entry in entries]
        raise ManagedWorkspaceError(
            f"managed recovery entry is malformed or partial: {path}: {relative}"
        )
    try:
        generation = inspect_frame_generation(
            frames,
            label=f"recovery generation {entry_id}",
        )
    except FrameGenerationValidationError as exc:
        raise ManagedWorkspaceError(
            f"managed recovery entry is malformed or partial: {path}: {exc}"
        ) from exc
    return _RecoveryEntry(
        entry_id=entry_id,
        state="valid",
        generation=generation,
    )


def _inspect_public_generation(
    output_dir: Path,
) -> FrameGenerationSnapshot | None:
    public = output_dir / PUBLIC_FRAMES_DIRECTORY_NAME
    if not os.path.lexists(public):
        return None
    return inspect_frame_generation(public, label="public frame generation")


def _inspect_workspace(
    output_dir: Path,
    workspace_root: Path,
) -> _WorkspaceInspection:
    _require_directory(workspace_root, "managed workspace root")
    sentinel = workspace_root / OWNERSHIP_FILENAME
    if not os.path.lexists(sentinel):
        instruction = manual_workspace_recovery_instruction(
            output_dir,
            workspace_root,
        )
        raise ManagedWorkspaceError(
            f"managed workspace initialization is incomplete at {workspace_root}. "
            f"{instruction}"
        )

    try:
        root_entries = tuple(
            sorted(workspace_root.iterdir(), key=lambda item: item.name)
        )
    except OSError as exc:
        raise ManagedWorkspaceError(
            f"managed workspace root is unreadable: {workspace_root}: {exc}"
        ) from exc
    unknown_root_entries = tuple(
        entry.name for entry in root_entries if entry.name not in _WORKSPACE_ENTRY_NAMES
    )
    if unknown_root_entries:
        raise ManagedWorkspaceError(
            "managed workspace root contains unknown entries: "
            f"{list(unknown_root_entries)}"
        )

    root_id = _ownership_root_id(sentinel)
    runs_dir = workspace_root / RUNS_DIRECTORY_NAME
    recovery_dir = workspace_root / RECOVERY_DIRECTORY_NAME
    runs_present = os.path.lexists(runs_dir)
    recovery_present = os.path.lexists(recovery_dir)

    run_ids: list[uuid.UUID] = []
    if runs_present:
        _require_directory(runs_dir, "managed runs directory")
        try:
            run_entries = tuple(sorted(runs_dir.iterdir(), key=lambda item: item.name))
        except OSError as exc:
            raise ManagedWorkspaceError(
                f"managed runs directory is unreadable: {runs_dir}: {exc}"
            ) from exc
        for entry in run_entries:
            entry_id = _entry_uuid(entry, "managed run entry")
            _validate_disposable_tree(entry, f"managed run {entry_id}")
            run_ids.append(entry_id)

    recoveries: list[_RecoveryEntry] = []
    if recovery_present:
        _require_directory(recovery_dir, "managed recovery directory")
        try:
            recovery_entries = tuple(
                sorted(recovery_dir.iterdir(), key=lambda item: item.name)
            )
        except OSError as exc:
            raise ManagedWorkspaceError(
                f"managed recovery directory is unreadable: {recovery_dir}: {exc}"
            ) from exc
        for entry in recovery_entries:
            recoveries.append(_inspect_recovery_entry(entry))

    public_generation = _inspect_public_generation(output_dir)
    return _WorkspaceInspection(
        root_id=root_id,
        runs_present=runs_present,
        recovery_present=recovery_present,
        run_ids=tuple(run_ids),
        recoveries=tuple(recoveries),
        public_generation=public_generation,
    )


def _require_unambiguous_recovery_state(
    inspection: _WorkspaceInspection,
) -> None:
    if inspection.public_generation is not None:
        return
    empty_recoveries = tuple(
        recovery for recovery in inspection.recoveries if recovery.state == "empty"
    )
    if empty_recoveries:
        raise ManagedWorkspaceError(
            "empty prepared frame recoveries exist while the public generation "
            "is missing; preserving all workspace state for review: "
            f"{[str(item.entry_id) for item in empty_recoveries]}"
        )
    valid_recoveries = tuple(
        recovery for recovery in inspection.recoveries if recovery.state == "valid"
    )
    if len(valid_recoveries) > 1:
        raise ManagedWorkspaceError(
            "multiple valid frame recoveries exist while the public generation "
            "is missing; refusing to choose among: "
            f"{[str(item.entry_id) for item in valid_recoveries]}"
        )


def _make_tree_owner_writable(root: Path) -> None:
    for current_root, directories, files in os.walk(root, topdown=True):
        current = Path(current_root)
        current_mode = stat.S_IMODE(current.lstat().st_mode)
        current.chmod(current_mode | stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        for name in directories:
            child = current / name
            child_mode = stat.S_IMODE(child.lstat().st_mode)
            child.chmod(child_mode | stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        for name in files:
            child = current / name
            child_mode = stat.S_IMODE(child.lstat().st_mode)
            child.chmod(child_mode | stat.S_IRUSR | stat.S_IWUSR)


class ManagedWorkspace:
    """Validated managed root whose mutations require the held output lock."""

    def __init__(
        self,
        *,
        output_dir: Path,
        root: Path,
        root_id: str,
        lock: _HeldOutputLock,
    ) -> None:
        self.output_dir = output_dir
        self.root = root
        self.root_id = root_id
        self.runs_dir = root / RUNS_DIRECTORY_NAME
        self.recovery_dir = root / RECOVERY_DIRECTORY_NAME
        self.public_frames_dir = output_dir / PUBLIC_FRAMES_DIRECTORY_NAME
        self._lock = lock

    @classmethod
    def open(
        cls,
        output_dir: str | Path,
        lock: _HeldOutputLock,
    ) -> ManagedWorkspace:
        output = Path(output_dir).resolve()
        if not lock.is_held:
            raise ManagedWorkspaceError(
                "managed workspace requires the held output lock"
            )
        root = output / WORKSPACE_DIRECTORY_NAME

        # Invalid public artifacts block even first-time workspace creation.
        _inspect_public_generation(output)

        if not os.path.lexists(root):
            try:
                root.mkdir()
                atomic_write_json(
                    root / OWNERSHIP_FILENAME,
                    {
                        "schema_version": 1,
                        "application": "keyframe",
                        "purpose": "managed-output-workspace",
                        "root_id": secrets.token_hex(16),
                    },
                )
            except BaseException as exc:
                instruction = manual_workspace_recovery_instruction(output, root)
                if isinstance(exc, Exception):
                    raise ManagedWorkspaceError(
                        f"managed workspace initialization failed at {root}. "
                        f"{instruction}"
                    ) from exc
                exc.add_note(instruction)
                raise

        inspection = _inspect_workspace(output, root)
        _require_unambiguous_recovery_state(inspection)

        # Missing structural directories are created only after the ownership
        # sentinel and every existing root, run, recovery, and public entry are
        # fully classified.
        if not inspection.runs_present:
            (root / RUNS_DIRECTORY_NAME).mkdir()
        if not inspection.recovery_present:
            (root / RECOVERY_DIRECTORY_NAME).mkdir()
        if not inspection.runs_present or not inspection.recovery_present:
            inspection = _inspect_workspace(output, root)
            _require_unambiguous_recovery_state(inspection)

        workspace = cls(
            output_dir=output,
            root=root,
            root_id=inspection.root_id,
            lock=lock,
        )
        workspace._reconcile_startup(inspection)
        return workspace

    def _require_lock(self) -> None:
        if not self._lock.is_held:
            raise ManagedWorkspaceError(
                "managed workspace mutation requires the held output lock"
            )

    def staging_paths(self, entry_id: uuid.UUID) -> RunStagingPaths:
        parsed = parse_canonical_uuid4(entry_id)
        run_root = self.runs_dir / str(parsed)
        return RunStagingPaths(
            output_dir=self.output_dir,
            run_id=str(parsed),
            root=run_root,
            transcript_raw=run_root / "transcript.raw.json",
            diarization=run_root / "diarization.json",
            frames=run_root / PUBLIC_FRAMES_DIRECTORY_NAME,
            frame_backup=(
                self.recovery_dir / str(parsed) / PUBLIC_FRAMES_DIRECTORY_NAME
            ),
        )

    def create_run(self, entry_id: uuid.UUID) -> RunStagingPaths:
        self._require_lock()
        paths = self.staging_paths(entry_id)
        if os.path.lexists(paths.root):
            raise ManagedWorkspaceError(
                f"managed run entry already exists: {paths.root}"
            )
        try:
            paths.root.mkdir()
        except OSError as exc:
            raise ManagedWorkspaceError(
                f"failed to create managed run entry {paths.root}: {exc}"
            ) from exc
        return paths

    def _entry_path(
        self,
        kind: Literal["run", "recovery"],
        entry_id: uuid.UUID,
    ) -> Path:
        if not isinstance(entry_id, uuid.UUID):
            raise TypeError("managed deletion requires a parsed UUID")
        parsed = parse_canonical_uuid4(entry_id)
        if kind == "run":
            parent = self.runs_dir
        elif kind == "recovery":
            parent = self.recovery_dir
        else:
            raise ValueError("managed entry kind must be 'run' or 'recovery'")
        target = parent / str(parsed)
        if target.parent != parent or target.name != str(parsed):
            raise ManagedWorkspaceError(
                "managed entry did not resolve to a direct child"
            )
        return target

    def delete_entry(
        self,
        kind: Literal["run", "recovery"],
        entry_id: uuid.UUID,
    ) -> None:
        """Delete one fully validated immediate managed child."""

        self._require_lock()
        target = self._entry_path(kind, entry_id)
        if not os.path.lexists(target):
            return
        if kind == "run":
            _validate_disposable_tree(target, f"managed run {entry_id}")
        else:
            _inspect_recovery_entry(target)
        try:
            _make_tree_owner_writable(target)
            shutil.rmtree(target)
        except OSError as exc:
            raise ManagedWorkspaceError(
                f"failed to delete managed {kind} entry {target}: {exc}"
            ) from exc

    def _restore_recovery(self, recovery: _RecoveryEntry) -> None:
        self._require_lock()
        if recovery.generation is None:
            raise ManagedWorkspaceError("cannot restore an empty recovery entry")
        source = recovery.generation.path
        try:
            os.replace(source, self.public_frames_dir)
            inspect_frame_generation(
                self.public_frames_dir,
                label="restored public frame generation",
            )
        except BaseException as exc:
            if os.path.lexists(self.public_frames_dir) and not os.path.lexists(source):
                try:
                    os.replace(self.public_frames_dir, source)
                except BaseException as rollback_exc:
                    exc.add_note(
                        "failed to roll back recovery restoration: "
                        f"{type(rollback_exc).__name__}: {rollback_exc}"
                    )
            if isinstance(exc, Exception):
                raise ManagedWorkspaceError(
                    f"failed to restore frame recovery {recovery.entry_id}: {exc}"
                ) from exc
            raise

    def _reconcile_startup(self, inspection: _WorkspaceInspection) -> None:
        self._require_lock()
        valid_recoveries = tuple(
            recovery for recovery in inspection.recoveries if recovery.state == "valid"
        )
        public_is_valid = inspection.public_generation is not None
        recoveries_to_delete: tuple[_RecoveryEntry, ...] = ()

        _require_unambiguous_recovery_state(inspection)
        if not public_is_valid and len(valid_recoveries) == 1:
            self._restore_recovery(valid_recoveries[0])
            public_is_valid = True

        if public_is_valid:
            recoveries_to_delete = inspection.recoveries

        # Every entry was classified before the first deletion or restoration.
        for recovery in recoveries_to_delete:
            self.delete_entry("recovery", recovery.entry_id)
        for run_id in inspection.run_ids:
            self.delete_entry("run", run_id)

    def _prepare_recovery(self, entry_id: uuid.UUID) -> Path:
        self._require_lock()
        container = self._entry_path("recovery", entry_id)
        if os.path.lexists(container):
            raise FrameGenerationPromotionError(
                f"frame recovery entry already exists: {container}"
            )
        try:
            container.mkdir()
        except OSError as exc:
            raise FrameGenerationPromotionError(
                f"failed to prepare frame recovery entry {container}: {exc}"
            ) from exc
        return container / PUBLIC_FRAMES_DIRECTORY_NAME

    def promote_frame_generation(
        self,
        staged_dir: str | Path,
        *,
        expected_frame_names: Iterable[str],
        entry_id: uuid.UUID,
    ) -> Path:
        """Publish a validated staged generation through recoverable states."""

        self._require_lock()
        parsed = parse_canonical_uuid4(entry_id)
        expected_staged = self.staging_paths(parsed).frames
        staged = Path(staged_dir)
        if staged != expected_staged:
            raise FrameGenerationPromotionError(
                "staged frame generation must be the current managed run's "
                f"frames directory: expected={expected_staged}, actual={staged}"
            )
        expected_names = tuple(expected_frame_names)
        inspect_frame_generation(
            staged,
            label="staged frame generation",
            expected_frame_names=expected_names,
        )

        # Reclassify all workspace state immediately before the first mutation.
        inspection = _inspect_workspace(self.output_dir, self.root)
        if parsed not in inspection.run_ids:
            raise FrameGenerationPromotionError(
                f"current managed run is no longer present: {parsed}"
            )
        valid_recoveries = tuple(
            recovery for recovery in inspection.recoveries if recovery.state == "valid"
        )
        if inspection.public_generation is None and valid_recoveries:
            raise FrameGenerationPromotionError(
                "frame recovery state changed after session startup; refusing "
                "publication until a new locked session reconciles it"
            )
        if any(recovery.entry_id == parsed for recovery in inspection.recoveries):
            raise FrameGenerationPromotionError(
                f"current run recovery id is already present: {parsed}"
            )

        public_existed = inspection.public_generation is not None
        public_mode: int | None = None
        if public_existed:
            public_mode = stat.S_IMODE(self.public_frames_dir.lstat().st_mode)
            try:
                staged_mode = stat.S_IMODE(staged.lstat().st_mode)
                staged.chmod(staged_mode | stat.S_IRWXU)
                self.public_frames_dir.chmod(public_mode | stat.S_IRWXU)
            except OSError as exc:
                raise FrameGenerationPromotionError(
                    f"failed to prepare frame generation permissions: {exc}"
                ) from exc
        if staged.lstat().st_dev != self.output_dir.lstat().st_dev:
            raise FrameGenerationPromotionError(
                "staged and public frame generations are on different filesystems"
            )

        recovery_frames: Path | None = None
        old_moved = False
        new_moved = False
        try:
            recovery_frames = self._prepare_recovery(parsed)
            if public_existed:
                os.replace(self.public_frames_dir, recovery_frames)
                old_moved = True
            os.replace(staged, self.public_frames_dir)
            new_moved = True
            inspect_frame_generation(
                self.public_frames_dir,
                label="published frame generation",
                expected_frame_names=expected_names,
            )
        except BaseException as exc:
            rollback_errors: list[BaseException] = []
            if new_moved and os.path.lexists(self.public_frames_dir):
                try:
                    os.replace(self.public_frames_dir, staged)
                    new_moved = False
                except BaseException as rollback_exc:
                    rollback_errors.append(rollback_exc)
            if (
                old_moved
                and recovery_frames is not None
                and os.path.lexists(recovery_frames)
                and not os.path.lexists(self.public_frames_dir)
            ):
                try:
                    os.replace(recovery_frames, self.public_frames_dir)
                    old_moved = False
                except BaseException as rollback_exc:
                    rollback_errors.append(rollback_exc)
            if public_mode is not None and os.path.lexists(self.public_frames_dir):
                try:
                    self.public_frames_dir.chmod(public_mode)
                except BaseException as rollback_exc:
                    rollback_errors.append(rollback_exc)
            try:
                recovery_container = self._entry_path("recovery", parsed)
                if os.path.lexists(recovery_container):
                    inspected = _inspect_recovery_entry(recovery_container)
                    if inspected.state == "empty":
                        self.delete_entry("recovery", parsed)
            except BaseException as rollback_exc:
                rollback_errors.append(rollback_exc)
            for rollback_error in rollback_errors:
                exc.add_note(
                    "frame publication rollback failure: "
                    f"{type(rollback_error).__name__}: {rollback_error}"
                )

            previous_restored = (
                public_existed
                and os.path.lexists(self.public_frames_dir)
                and not old_moved
            ) or (not public_existed and not os.path.lexists(self.public_frames_dir))
            if isinstance(exc, Exception):
                if previous_restored:
                    raise FrameGenerationPromotionError(
                        "failed to publish frame generation; previous generation "
                        f"was restored: {exc}"
                    ) from exc
                recovery_location = (
                    recovery_frames
                    if recovery_frames is not None and os.path.lexists(recovery_frames)
                    else self.public_frames_dir
                )
                raise FrameGenerationPromotionError(
                    "failed to publish frame generation and could not restore the "
                    f"previous generation; recovery state remains at "
                    f"{recovery_location}: {exc}"
                ) from exc
            raise

        if public_mode is not None:
            try:
                self.public_frames_dir.chmod(public_mode)
            except OSError as exc:
                LOGGER.warning(
                    "Published frame generation but could not restore directory "
                    "mode %o at %s: %s",
                    public_mode,
                    self.public_frames_dir,
                    exc,
                )

        # Recovery entries are removed only after the replacement has passed
        # post-publication validation.  Cleanup failure leaves a valid recovery
        # for the next locked session.
        cleanup_ids = tuple(
            dict.fromkeys(
                [
                    *(recovery.entry_id for recovery in inspection.recoveries),
                    parsed,
                ]
            )
        )
        for recovery_id in cleanup_ids:
            try:
                self.delete_entry("recovery", recovery_id)
            except OutputSessionError as exc:
                LOGGER.warning(
                    "Published frame generation but could not remove recovery %s: %s",
                    recovery_id,
                    exc,
                )
        return self.public_frames_dir


def known_public_artifact_paths(output_dir: str | Path) -> tuple[Path, ...]:
    """Enumerate only known published artifacts without recursive output scans."""

    output = Path(output_dir)
    artifacts: list[Path] = []
    for name in KNOWN_PUBLIC_FILENAMES:
        path = output / name
        if not os.path.lexists(path):
            continue
        try:
            mode = path.lstat().st_mode
        except OSError:
            continue
        if stat.S_ISREG(mode) and not stat.S_ISLNK(mode):
            artifacts.append(path)
    frames = output / PUBLIC_FRAMES_DIRECTORY_NAME
    if os.path.lexists(frames):
        snapshot = inspect_frame_generation(
            frames,
            label="public frame generation",
        )
        artifacts.extend(snapshot.artifact_paths)
    return tuple(sorted(artifacts))
