"""Validated, replace-as-a-generation publication for frame artifacts."""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from keyframe.managed_workspace import (
    FrameGenerationError,
    FrameGenerationPromotionError,
    FrameGenerationValidationError,
    ManagedWorkspace,
    inspect_frame_generation,
)
from keyframe.output_session import OutputRunSession
from keyframe.pipeline.config import KeyframeExtractionResult
from keyframe.pipeline.contracts import CandidateRecord, candidate_to_manifest_row

__all__ = [
    "FrameGenerationError",
    "FrameGenerationPromotionError",
    "FrameGenerationSession",
    "FrameGenerationValidationError",
    "StagedFrameGeneration",
    "validate_frame_generation",
]


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


def validate_frame_generation(
    staged_dir: str | Path,
    expected_frame_names: Iterable[str],
) -> None:
    """Validate the exact PNG set and its captions/manifest indexes."""

    inspect_frame_generation(
        staged_dir,
        label="staged frame generation",
        expected_frame_names=expected_frame_names,
    )


@dataclass
class StagedFrameGeneration:
    """Current-run frame records plus deferred enrichment and publication."""

    output_dir: Path
    workspace: ManagedWorkspace
    entry_id: uuid.UUID
    staging_root: Path
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
        workspace = getattr(session, "workspace", None)
        entry_id = getattr(session, "entry_id", None)
        if not isinstance(workspace, ManagedWorkspace) or not isinstance(
            entry_id,
            uuid.UUID,
        ):
            raise FrameGenerationValidationError(
                "frame generation session has no validated managed workspace"
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
            workspace=workspace,
            entry_id=entry_id,
            staging_root=Path(staging.root),
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
        return (
            self.workspace.recovery_dir
            / str(self.entry_id)
            / "frames"
        )

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
        self.workspace.promote_frame_generation(
            self.staged_dir,
            expected_frame_names=self.expected_frame_names,
            entry_id=self.entry_id,
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
        if artifact_path.is_symlink() or not artifact_path.is_file():
            raise FrameGenerationValidationError(
                f"staged {artifact} is not a regular non-symlinked file: "
                f"{artifact_path}"
            )
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
