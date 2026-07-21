"""Validated, replace-as-a-generation publication for frame artifacts."""

from __future__ import annotations

import json
import logging
import os
import stat
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from keyframe.output_session import (
    OutputRunSession,
    OutputSessionError,
    remove_keyframe_owned_directory,
)
from keyframe.pipeline.config import KeyframeExtractionResult
from keyframe.pipeline.contracts import CandidateRecord, candidate_to_manifest_row


LOGGER = logging.getLogger(__name__)


class FrameGenerationError(OutputSessionError):
    """Base class for controlled staged-frame publication failures."""


class FrameGenerationValidationError(FrameGenerationError):
    """A staged frame generation is incomplete or internally inconsistent."""


class FrameGenerationPromotionError(FrameGenerationError):
    """A validated frame generation could not replace the public generation."""


class FrameGenerationSession(OutputRunSession):
    """Lightweight frame-only output session with no transcription imports."""


def _frame_filename(candidate: Mapping[str, Any] | CandidateRecord) -> str:
    if isinstance(candidate, CandidateRecord):
        frame_idx = candidate.frame_idx
        timestamp = candidate.timestamp
    elif isinstance(candidate, Mapping):
        frame_idx = candidate.get("frame_idx")
        timestamp = candidate.get("timestamp")
    else:
        frame_idx = getattr(candidate, "frame_idx", None)
        timestamp = getattr(candidate, "timestamp", None)
    try:
        return f"frame_{int(frame_idx):06d}_{float(timestamp):.2f}s.png"
    except (TypeError, ValueError) as exc:
        raise FrameGenerationValidationError(
            "frame record is missing a valid frame index or timestamp"
        ) from exc


def _read_json(path: Path, artifact: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FrameGenerationValidationError(
            f"staged {artifact} is not readable JSON: {exc}"
        ) from exc


def _require_regular_file(path: Path, artifact: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise FrameGenerationValidationError(
            f"staged {artifact} is missing or is not a regular file: {path}"
        )


def _artifact_names(
    rows: Any,
    *,
    field_name: str,
    artifact: str,
) -> tuple[str, ...]:
    if not isinstance(rows, list):
        raise FrameGenerationValidationError(f"staged {artifact} must contain a list")
    names = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise FrameGenerationValidationError(
                f"staged {artifact} row {index} must be an object"
            )
        name = row.get(field_name)
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise FrameGenerationValidationError(
                f"staged {artifact} row {index} has an invalid {field_name}"
            )
        names.append(name)
    return tuple(names)


def validate_frame_generation(
    staged_dir: str | Path,
    expected_frame_names: Iterable[str],
) -> None:
    """Validate the exact PNG set and its captions/manifest indexes."""

    staged = Path(staged_dir)
    if staged.is_symlink() or not staged.is_dir():
        raise FrameGenerationValidationError(
            f"staged frame generation is missing or is not a directory: {staged}"
        )
    expected = tuple(expected_frame_names)
    if len(set(expected)) != len(expected):
        raise FrameGenerationValidationError(
            "staged frame generation contains duplicate expected filenames"
        )
    for name in expected:
        if Path(name).name != name or not name.endswith(".png"):
            raise FrameGenerationValidationError(
                f"invalid expected frame filename: {name!r}"
            )

    actual_pngs = tuple(sorted(path.name for path in staged.glob("*.png")))
    if Counter(actual_pngs) != Counter(expected):
        raise FrameGenerationValidationError(
            "staged PNG set does not match the selected frame generation: "
            f"expected={sorted(expected)}, actual={sorted(actual_pngs)}"
        )

    from PIL import Image

    for name in expected:
        path = staged / name
        _require_regular_file(path, f"frame {name}")
        try:
            with Image.open(path) as image:
                if image.format != "PNG":
                    raise ValueError(f"detected format {image.format!r}")
                image.verify()
        except Exception as exc:
            raise FrameGenerationValidationError(
                f"staged frame is not a valid PNG: {path}: {exc}"
            ) from exc

    captions_path = staged / "captions.json"
    manifest_path = staged / "manifest.json"
    _require_regular_file(captions_path, "captions.json")
    _require_regular_file(manifest_path, "manifest.json")
    caption_names = _artifact_names(
        _read_json(captions_path, "captions.json"),
        field_name="file",
        artifact="captions.json",
    )
    manifest = _read_json(manifest_path, "manifest.json")
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != 1:
        raise FrameGenerationValidationError(
            "staged manifest.json must be a schema_version 1 object"
        )
    manifest_names = _artifact_names(
        manifest.get("frames"),
        field_name="filename",
        artifact="manifest.json frames",
    )
    for artifact, names in (
        ("captions.json", caption_names),
        ("manifest.json", manifest_names),
    ):
        if Counter(names) != Counter(expected):
            raise FrameGenerationValidationError(
                f"staged {artifact} does not index the exact PNG generation: "
                f"expected={sorted(expected)}, actual={sorted(names)}"
            )


def _rollback_previous_generation(
    backup_dir: Path,
    public_dir: Path,
) -> BaseException | None:
    first_error: BaseException | None = None
    for _attempt in range(2):
        try:
            os.replace(backup_dir, public_dir)
            return first_error
        except BaseException as exc:
            if first_error is None:
                first_error = exc
            else:
                first_error.add_note(
                    f"second rollback attempt failed: {type(exc).__name__}: {exc}"
                )
    return first_error


def promote_frame_generation(
    staged_dir: str | Path,
    public_dir: str | Path,
    backup_dir: str | Path,
) -> Path:
    """Replace the public directory, restoring its backup on rename failure."""

    staged = Path(staged_dir)
    public = Path(public_dir)
    backup = Path(backup_dir)
    resolved_paths = {
        staged.resolve(strict=False),
        public.resolve(strict=False),
        backup.resolve(strict=False),
    }
    if len(resolved_paths) != 3:
        raise FrameGenerationPromotionError(
            "staged, public, and backup frame paths must be distinct"
        )
    if not staged.is_dir() or staged.is_symlink():
        raise FrameGenerationPromotionError(
            f"staged frame directory cannot be promoted: {staged}"
        )
    if staged.stat().st_dev != public.parent.stat().st_dev:
        raise FrameGenerationPromotionError(
            f"staged and public frame directories are on different filesystems: "
            f"{staged}, {public}"
        )
    if os.path.lexists(backup):
        raise FrameGenerationPromotionError(
            f"frame backup path already exists: {backup}"
        )
    public_exists = os.path.lexists(public)
    if public_exists and (public.is_symlink() or not public.is_dir()):
        raise FrameGenerationPromotionError(
            f"public frame path is not a replaceable directory: {public}"
        )
    public_mode: int | None = None
    if public_exists:
        try:
            public_mode = stat.S_IMODE(public.stat().st_mode)
            staged.chmod(public_mode | stat.S_IRWXU)
        except OSError as exc:
            raise FrameGenerationPromotionError(
                f"failed to preserve public frame directory permissions: {exc}"
            ) from exc

    if not public_exists:
        try:
            os.replace(staged, public)
        except OSError as exc:
            raise FrameGenerationPromotionError(
                f"failed to publish frame generation at {public}: {exc}"
            ) from exc
        return public

    try:
        os.replace(public, backup)
    except OSError as exc:
        raise FrameGenerationPromotionError(
            f"failed to back up previous frame generation at {public}: {exc}"
        ) from exc

    try:
        os.replace(staged, public)
    except BaseException as promotion_error:
        rollback_error = _rollback_previous_generation(backup, public)
        if os.path.lexists(public) and not os.path.lexists(backup):
            if not isinstance(promotion_error, Exception):
                if rollback_error is not None:
                    promotion_error.add_note(
                        "the first rollback attempt failed before a retry restored "
                        f"the previous generation: {rollback_error}"
                    )
                raise
            error = FrameGenerationPromotionError(
                f"failed to publish frame generation; previous generation was restored: "
                f"{promotion_error}"
            )
            if rollback_error is not None:
                error.add_note(
                    "the first rollback attempt failed before a retry restored the "
                    f"previous generation: {rollback_error}"
                )
            raise error from promotion_error
        if not isinstance(promotion_error, Exception):
            if rollback_error is not None:
                promotion_error.add_note(f"rollback failure: {rollback_error}")
            promotion_error.add_note(
                f"previous frame generation remains at recovery backup {backup}"
            )
            raise
        error = FrameGenerationPromotionError(
            "failed to publish frame generation and could not restore the previous "
            f"generation; recovery backup remains at {backup}: {promotion_error}"
        )
        if rollback_error is not None:
            error.add_note(f"rollback failure: {rollback_error}")
        raise error from promotion_error

    try:
        assert public_mode is not None
        public.chmod(public_mode)
    except OSError as exc:
        LOGGER.warning(
            "Published frame generation but could not restore directory mode %o "
            "at %s: %s",
            public_mode,
            public,
            exc,
        )

    try:
        remove_keyframe_owned_directory(backup)
    except OSError as exc:
        LOGGER.warning(
            "Published frame generation but could not remove recovery backup %s: %s",
            backup,
            exc,
        )
    return public


@dataclass
class StagedFrameGeneration:
    """Current-run frame records plus deferred enrichment and publication."""

    output_dir: Path
    staging_root: Path
    frame_backup_dir: Path
    result: KeyframeExtractionResult
    expected_frame_names: tuple[str, ...]
    _public_result: KeyframeExtractionResult | None = field(
        default=None,
        init=False,
        repr=False,
    )

    @classmethod
    def from_extraction(
        cls,
        session: Any,
        result: KeyframeExtractionResult,
    ) -> StagedFrameGeneration:
        staging = getattr(session, "staging", None)
        if staging is None:
            raise FrameGenerationValidationError(
                "frame generation session is not entered"
            )
        staged_dir = Path(result.output_dir)
        if staged_dir.resolve(strict=False) != staging.frames.resolve(strict=False):
            raise FrameGenerationValidationError(
                "frame extraction did not target the current run staging directory"
            )
        try:
            records = tuple(result.final)
        except TypeError as exc:
            raise FrameGenerationValidationError(
                "frame extraction result final records are not iterable"
            ) from exc
        result.final = records
        expected = tuple(_frame_filename(candidate) for candidate in records)
        if int(result.final_frame_count) != len(expected):
            raise FrameGenerationValidationError(
                "frame extraction result count does not match its final records"
            )
        for artifact_path, expected_path, artifact_name in (
            (result.caption_log_path, staged_dir / "captions.json", "captions.json"),
            (result.manifest_path, staged_dir / "manifest.json", "manifest.json"),
        ):
            if Path(artifact_path).resolve(strict=False) != expected_path.resolve(
                strict=False
            ):
                raise FrameGenerationValidationError(
                    f"frame extraction reported {artifact_name} outside the staged generation"
                )
        generation = cls(
            output_dir=Path(session.output_dir),
            staging_root=Path(staging.root),
            frame_backup_dir=Path(staging.frame_backup),
            result=result,
            expected_frame_names=expected,
        )
        generation.validate()
        return generation

    @property
    def staged_dir(self) -> Path:
        return Path(self.result.output_dir)

    @property
    def public_dir(self) -> Path:
        return self.output_dir / "frames"

    @property
    def backup_dir(self) -> Path:
        return self.frame_backup_dir

    @property
    def records(self) -> Any:
        return self.result.final

    def validate(self) -> None:
        validate_frame_generation(self.staged_dir, self.expected_frame_names)
        for path, artifact in (
            (self.result.pipeline_trace_path, "pipeline trace"),
            (self.result.debug_qa_trace_path, "debug QA trace"),
        ):
            if path is not None:
                self._staged_artifact_relative_path(path, artifact)

    def enrich_manifest(self, transcript_segments: Iterable[Any]) -> Path:
        if self._public_result is not None:
            raise FrameGenerationError(
                "cannot enrich a frame generation after it has been promoted"
            )
        from keyframe.manifest import write_manifest

        manifest_rows = [
            candidate_to_manifest_row(
                candidate,
                filename=_frame_filename(candidate),
            )
            if isinstance(candidate, CandidateRecord)
            else dict(candidate)
            for candidate in self.records
        ]
        self.result.manifest_path = write_manifest(
            manifest_rows,
            self.staged_dir,
            list(transcript_segments),
            metadata=self.result.manifest_metadata,
        )
        self.validate()
        return self.result.manifest_path

    def promote(self) -> KeyframeExtractionResult:
        if self._public_result is not None:
            return self._public_result
        self.validate()
        public_pipeline_trace = self._public_artifact_path(
            self.result.pipeline_trace_path,
            "pipeline trace",
        )
        public_debug_trace = self._public_artifact_path(
            self.result.debug_qa_trace_path,
            "debug QA trace",
        )
        promote_frame_generation(
            self.staged_dir,
            self.public_dir,
            self.backup_dir,
        )
        self._public_result = replace(
            self.result,
            output_dir=self.public_dir,
            caption_log_path=self.public_dir / "captions.json",
            manifest_path=self.public_dir / "manifest.json",
            pipeline_trace_path=public_pipeline_trace,
            debug_qa_trace_path=public_debug_trace,
        )
        return self._public_result

    def _staged_artifact_relative_path(self, path: Path, artifact: str) -> Path:
        if path is None:
            raise FrameGenerationValidationError(f"missing staged {artifact} path")
        try:
            relative = Path(path).relative_to(self.staged_dir)
        except ValueError as exc:
            raise FrameGenerationValidationError(
                f"{artifact} escaped the staged frame generation: {path}"
            ) from exc
        artifact_path = self.staged_dir / relative
        _require_regular_file(artifact_path, artifact)
        if not artifact_path.resolve().is_relative_to(self.staged_dir.resolve()):
            raise FrameGenerationValidationError(
                f"{artifact} resolves outside the staged frame generation: {path}"
            )
        return relative

    def _public_artifact_path(
        self,
        path: Path | None,
        artifact: str,
    ) -> Path | None:
        if path is None:
            return None
        relative = self._staged_artifact_relative_path(path, artifact)
        return self.public_dir / relative
