"""Shared output-directory locking and run-scoped staging lifecycle."""

from __future__ import annotations

import os
import shutil
import stat
import uuid
from pathlib import Path
from typing import Any

from keyframe.artifacts import RunStagingPaths, run_staging_paths


LOCK_FILENAME = "keyframe-output.lock"
RUN_DIRECTORY_PREFIX = "keyframe-run-"
FRAME_BACKUP_PREFIX = "keyframe-frame-backup-"


class OutputSessionError(RuntimeError):
    """A controlled failure while owning a Keyframe output directory."""


class OutputDirectoryLockedError(OutputSessionError):
    """Another Keyframe CLI process owns the output directory."""


class OutputDirectoryLock:
    """Non-blocking advisory lock keyed by the resolved output directory."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.path = self.output_dir / LOCK_FILENAME
        self._descriptor: int | None = None

    def acquire(self) -> None:
        if self._descriptor is not None:
            raise RuntimeError("output directory lock is already held")
        descriptor: int | None = None
        try:
            descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
            self._descriptor = descriptor
            self._acquire_descriptor(descriptor)
        except BaseException:
            self._descriptor = None
            if descriptor is not None:
                os.close(descriptor)
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


def remove_keyframe_owned_directory(path: str | Path) -> None:
    """Make one validated Keyframe-owned tree removable, then delete it."""

    root = Path(path)
    if not os.path.lexists(root):
        return
    if root.is_symlink() or not root.is_dir():
        raise OutputSessionError(
            f"refusing to recursively remove a non-directory artifact: {root}"
        )

    def make_writable(candidate: Path, *, directory: bool) -> None:
        if candidate.is_symlink():
            return
        mode = stat.S_IMODE(candidate.stat().st_mode)
        owner_bits = stat.S_IRUSR | stat.S_IWUSR
        if directory:
            owner_bits |= stat.S_IXUSR
        candidate.chmod(mode | owner_bits)

    make_writable(root, directory=True)
    for current_root, directories, files in os.walk(root):
        current = Path(current_root)
        make_writable(current, directory=True)
        for name in directories:
            make_writable(current / name, directory=True)
        for name in files:
            make_writable(current / name, directory=False)
    shutil.rmtree(root)


def cleanup_stale_run_directories(output_dir: Path) -> None:
    public_frames = output_dir / "frames"
    backups = sorted(
        candidate
        for candidate in output_dir.iterdir()
        if candidate.name.startswith(FRAME_BACKUP_PREFIX)
        and candidate.is_dir()
        and not candidate.is_symlink()
    )
    if backups:
        if os.path.lexists(public_frames):
            if public_frames.is_symlink() or not public_frames.is_dir():
                raise OutputSessionError(
                    f"public frame path is not a recoverable directory: {public_frames}"
                )
            for backup in backups:
                remove_keyframe_owned_directory(backup)
        elif len(backups) == 1:
            os.replace(backups[0], public_frames)
        else:
            raise OutputSessionError(
                "multiple frame-generation recovery backups exist while the public "
                f"frames directory is missing: {[str(path) for path in backups]}"
            )

    for candidate in output_dir.iterdir():
        if (
            candidate.name.startswith(RUN_DIRECTORY_PREFIX)
            and candidate.is_dir()
            and not candidate.is_symlink()
        ):
            remove_keyframe_owned_directory(candidate)


class OutputRunSession:
    """Own one output lock and disposable run directory without model imports."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        run_id: str | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.run_id = run_id or uuid.uuid4().hex
        self.lock: OutputDirectoryLock | None = None
        self.staging: RunStagingPaths | None = None
        self._entered = False

    def __enter__(self) -> OutputRunSession:
        if self._entered:
            raise RuntimeError("output run session cannot be entered twice")
        staging_root_was_absent = False
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.output_dir = self.output_dir.resolve()
            self.lock = OutputDirectoryLock(self.output_dir)
            self.lock.acquire()
            cleanup_stale_run_directories(self.output_dir)
            self.staging = run_staging_paths(self.output_dir, self.run_id)
            staging_root_was_absent = not os.path.lexists(self.staging.root)
            self.staging.root.mkdir()
            self._entered = True
            return self
        except BaseException as exc:
            try:
                if (
                    staging_root_was_absent
                    and self.staging is not None
                    and self.staging.root.is_dir()
                    and not self.staging.root.is_symlink()
                ):
                    try:
                        remove_keyframe_owned_directory(self.staging.root)
                    except BaseException as cleanup_exc:
                        exc.add_note(
                            f"failed to remove run staging directory: {cleanup_exc}"
                        )
            finally:
                if self.lock is not None:
                    self.lock.release()
            if isinstance(exc, OSError) and not isinstance(exc, FileExistsError):
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
            if (
                self.staging is not None
                and self.staging.root.is_dir()
                and not self.staging.root.is_symlink()
            ):
                try:
                    remove_keyframe_owned_directory(self.staging.root)
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
                            f"failed to release output lock: {type(exc).__name__}: {exc}"
                        )
        if first_error is not None:
            raise first_error

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()
