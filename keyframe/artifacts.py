"""Atomic artifact primitives and run-scoped staging paths."""

from __future__ import annotations

import json
import os
import re
import secrets
import stat
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


class ArtifactPathCollisionError(ValueError):
    """Two logical artifacts resolve to the same filesystem target."""


@dataclass(frozen=True)
class RunStagingPaths:
    output_dir: Path
    run_id: str
    root: Path
    transcript_raw: Path
    diarization: Path
    frames: Path
    frame_backup: Path


@dataclass(frozen=True)
class TranscriptCheckpointPaths:
    output_dir: Path
    transcript_raw: Path
    diarization: Path


def transcript_checkpoint_paths(
    output_dir: str | Path,
) -> TranscriptCheckpointPaths:
    output_dir = Path(output_dir)
    return TranscriptCheckpointPaths(
        output_dir=output_dir,
        transcript_raw=output_dir / "transcript.raw.json",
        diarization=output_dir / "diarization.json",
    )


def run_staging_paths(output_dir: str | Path, run_id: str) -> RunStagingPaths:
    """Return non-hidden, output-filesystem staging paths for one CLI run."""
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run_id must start with an alphanumeric character and contain only "
            "letters, numbers, underscores, or hyphens"
        )
    output_dir = Path(output_dir)
    root = output_dir / f"keyframe-run-{run_id}"
    return RunStagingPaths(
        output_dir=output_dir,
        run_id=run_id,
        root=root,
        transcript_raw=root / "transcript.raw.json",
        diarization=root / "diarization.json",
        frames=root / "frames",
        frame_backup=output_dir / f"keyframe-frame-backup-{run_id}",
    )


def paths_alias(left: str | Path, right: str | Path) -> bool:
    """Detect lexical, symlink, and existing hard-link aliases."""
    left_path = Path(left)
    right_path = Path(right)
    if left_path.resolve(strict=False) == right_path.resolve(strict=False):
        return True
    try:
        return left_path.exists() and right_path.exists() and os.path.samefile(
            left_path,
            right_path,
        )
    except OSError:
        return False


def reject_path_aliases(
    artifact_path: str | Path,
    other_paths: Iterable[str | Path],
) -> None:
    artifact_path = Path(artifact_path)
    for other_path in other_paths:
        if paths_alias(artifact_path, other_path):
            raise ArtifactPathCollisionError(
                f"artifact path {artifact_path} aliases {Path(other_path)}"
            )


def _open_unique_sibling(target: Path) -> tuple[int, Path]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_BINARY", 0)
    for _attempt in range(100):
        temporary_path = target.parent / f"{target.name}.tmp-{secrets.token_hex(8)}"
        try:
            descriptor = os.open(temporary_path, flags, 0o666)
        except FileExistsError:
            continue
        try:
            try:
                existing_mode = stat.S_IMODE(target.stat().st_mode)
            except FileNotFoundError:
                pass
            else:
                os.fchmod(descriptor, existing_mode)
        except BaseException:
            os.close(descriptor)
            temporary_path.unlink(missing_ok=True)
            raise
        return descriptor, temporary_path
    raise FileExistsError(f"could not create a unique sibling for {target}")


def atomic_write_text(
    path: str | Path,
    payload: str,
    *,
    encoding: str = "utf-8",
) -> Path:
    """Flush text to a unique sibling file, then atomically replace the target."""
    target = Path(path)
    temporary_path: Path | None = None
    descriptor: int | None = None
    try:
        descriptor, temporary_path = _open_unique_sibling(target)
        with os.fdopen(descriptor, "w", encoding=encoding) as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
        return target
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def atomic_write_json(
    path: str | Path,
    payload: Any,
    *,
    indent: int | None = 2,
    ensure_ascii: bool = False,
    allow_nan: bool = False,
) -> Path:
    rendered = json.dumps(
        payload,
        indent=indent,
        ensure_ascii=ensure_ascii,
        allow_nan=allow_nan,
    )
    return atomic_write_text(path, rendered)


def atomic_promote_file(staged_path: str | Path, public_path: str | Path) -> Path:
    """Atomically promote a validated staged file on the same filesystem."""
    staged = Path(staged_path)
    public = Path(public_path)
    if staged.stat().st_dev != public.parent.stat().st_dev:
        raise OSError(
            f"cannot atomically promote {staged} to a different filesystem at {public}"
        )
    try:
        existing_mode = stat.S_IMODE(public.stat().st_mode)
    except FileNotFoundError:
        pass
    else:
        staged.chmod(existing_mode)
    os.replace(staged, public)
    return public
