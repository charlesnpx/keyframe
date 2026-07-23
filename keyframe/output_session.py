"""Shared output-directory locking and managed run lifecycle."""

from __future__ import annotations

import os
import stat
import uuid
from pathlib import Path
from typing import Any

from keyframe.artifacts import RunStagingPaths
from keyframe.managed_workspace import (
    ManagedWorkspace,
    OutputSessionError,
    parse_canonical_uuid4,
)

LOCK_FILENAME = "keyframe-output.lock"

# These names identify legacy top-level artifacts only.  Keyframe deliberately
# never discovers, cleans, restores, or otherwise mutates them.
RUN_DIRECTORY_PREFIX = "keyframe-run-"
FRAME_BACKUP_PREFIX = "keyframe-frame-backup-"


class OutputDirectoryLockedError(OutputSessionError):
    """Another Keyframe CLI process owns the output directory."""


class OutputDirectoryLock:
    """Non-blocking advisory lock keyed by the resolved output directory."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.path = self.output_dir / LOCK_FILENAME
        self._descriptor: int | None = None

    @property
    def is_held(self) -> bool:
        return self._descriptor is not None

    def _open_regular_lock(self) -> int:
        if os.path.lexists(self.path):
            try:
                before = self.path.lstat()
            except OSError as exc:
                raise OutputSessionError(
                    f"output lock path is unreadable: {self.path}: {exc}"
                ) from exc
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
                raise OutputSessionError(
                    "output lock path must be a regular non-symlinked file: "
                    f"{self.path}"
                )

        flags = os.O_RDWR | os.O_CREAT
        flags |= getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOINHERIT", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except OSError as exc:
            raise OutputSessionError(
                "output lock path could not be opened as a regular "
                f"non-symlinked file: {self.path}: {exc}"
            ) from exc

        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise OutputSessionError(
                    f"opened output lock is not a regular file: {self.path}"
                )
            observed = self.path.lstat()
            if (
                stat.S_ISLNK(observed.st_mode)
                or not stat.S_ISREG(observed.st_mode)
                or (opened.st_dev, opened.st_ino)
                != (observed.st_dev, observed.st_ino)
            ):
                raise OutputSessionError(
                    "output lock path changed while it was being opened: "
                    f"{self.path}"
                )
        except BaseException:
            os.close(descriptor)
            raise
        return descriptor

    def acquire(self) -> None:
        if self._descriptor is not None:
            raise RuntimeError("output directory lock is already held")
        descriptor: int | None = None
        try:
            descriptor = self._open_regular_lock()
            self._descriptor = descriptor
            self._acquire_descriptor(descriptor)
        except BaseException:
            self._descriptor = None
            if descriptor is not None and descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            raise

    def _acquire_descriptor(self, descriptor: int) -> None:
        if os.name == "nt":
            import msvcrt

            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"\0")
            os.lseek(descriptor, 0, os.SEEK_SET)
            try:
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise OutputDirectoryLockedError(
                    f"output directory is already in use: {self.output_dir}"
                ) from exc
            return

        import fcntl

        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise OutputDirectoryLockedError(
                f"output directory is already in use: {self.output_dir}"
            ) from exc

    def release(self) -> None:
        descriptor = self._descriptor
        if descriptor is None:
            return
        self._descriptor = None
        try:
            if os.name == "nt":
                import msvcrt

                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def __enter__(self) -> OutputDirectoryLock:
        self.acquire()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.release()


def workspace_entry_id(run_id: str | uuid.UUID | None) -> uuid.UUID:
    if run_id is None:
        return uuid.uuid4()
    try:
        return parse_canonical_uuid4(run_id)
    except (TypeError, ValueError):
        # Legacy diagnostic run labels remain accepted, but never influence a
        # filesystem path.  Every managed entry is still a fresh UUIDv4.
        if not isinstance(run_id, str):
            raise TypeError("run_id must be a string, UUID, or None") from None
        return uuid.uuid4()


class OutputRunSession:
    """Own one output lock and one UUID-scoped managed run directory."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        run_id: str | uuid.UUID | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.entry_id = workspace_entry_id(run_id)
        self.run_id = str(self.entry_id)
        self.lock: OutputDirectoryLock | None = None
        self.workspace: ManagedWorkspace | None = None
        self.staging: RunStagingPaths | None = None
        self._entered = False

    def __enter__(self) -> OutputRunSession:
        if self._entered:
            raise RuntimeError("output run session cannot be entered twice")
        run_created = False
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.output_dir = self.output_dir.resolve()
            self.lock = OutputDirectoryLock(self.output_dir)
            self.lock.acquire()
            self.workspace = ManagedWorkspace.open(self.output_dir, self.lock)
            self.staging = self.workspace.create_run(self.entry_id)
            run_created = True
            self._entered = True
            return self
        except BaseException as exc:
            try:
                if run_created and self.workspace is not None:
                    try:
                        self.workspace.delete_entry("run", self.entry_id)
                    except BaseException as cleanup_exc:
                        exc.add_note(
                            f"failed to remove managed run entry: {cleanup_exc}"
                        )
            finally:
                if self.lock is not None:
                    self.lock.release()
            if isinstance(exc, OSError):
                raise OutputSessionError(
                    f"failed to initialize output directory {self.output_dir}: {exc}"
                ) from exc
            raise

    def close(self) -> None:
        if not self._entered:
            return
        self._entered = False
        first_error: BaseException | None = None
        try:
            if self.workspace is not None:
                try:
                    self.workspace.delete_entry("run", self.entry_id)
                except BaseException as exc:
                    first_error = exc
        finally:
            if self.lock is not None:
                try:
                    self.lock.release()
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
                    else:
                        first_error.add_note(
                            f"failed to release output lock: "
                            f"{type(exc).__name__}: {exc}"
                        )
        if first_error is not None:
            raise first_error

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()
