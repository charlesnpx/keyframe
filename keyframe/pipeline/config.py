from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from keyframe.pipeline.qa_targets import load_targets, normalize_targets


def _finite_float(
    name: str,
    value: Any,
    *,
    positive: bool = False,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or type(value) not in {int, float}:
        raise ValueError(f"{name} must be a finite number")
    try:
        rendered = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(rendered):
        raise ValueError(f"{name} must be a finite number")
    if positive and rendered <= 0:
        raise ValueError(f"{name} must be positive")
    if minimum is not None and rendered < minimum:
        raise ValueError(f"{name} must be within [{minimum:g}, {maximum:g}]")
    if maximum is not None and rendered > maximum:
        raise ValueError(f"{name} must be within [{minimum:g}, {maximum:g}]")
    return rendered


def _positive_integer(name: str, value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class KeyframeExtractionConfig:
    sample_interval: float = 0.5
    pass1_clusters: int = 15
    similarity_threshold: float = 0.85
    max_output_frames: int | None = None
    device: str | None = None
    max_clustering_memory_mb: int = 2048
    max_frame_cache_mb: int = 8192
    frame_cache_dir: Path | None = None
    verbose_trace: bool = False
    debug_qa_targets_path: Path | None = None
    debug_qa_targets: tuple[Mapping[str, Any], ...] | None = None

    def __post_init__(self) -> None:
        sample_interval = _finite_float(
            "sample_interval",
            self.sample_interval,
            positive=True,
        )
        if (
            type(self.pass1_clusters) is not int
            or not 1 <= self.pass1_clusters <= 64
        ):
            raise ValueError("pass1_clusters must be a true integer from 1 through 64")
        if (
            self.max_output_frames is not None
            and type(self.max_output_frames) is not int
        ):
            raise ValueError("max_output_frames must be None or a positive integer")
        if self.max_output_frames is not None:
            _positive_integer("max_output_frames", self.max_output_frames)
        _positive_integer(
            "max_clustering_memory_mb",
            self.max_clustering_memory_mb,
        )
        _positive_integer("max_frame_cache_mb", self.max_frame_cache_mb)
        similarity_threshold = _finite_float(
            "similarity_threshold",
            self.similarity_threshold,
            minimum=0,
            maximum=1,
        )

        object.__setattr__(self, "sample_interval", sample_interval)
        object.__setattr__(
            self,
            "similarity_threshold",
            similarity_threshold,
        )
        if self.frame_cache_dir is not None:
            object.__setattr__(self, "frame_cache_dir", Path(self.frame_cache_dir))
        if self.debug_qa_targets_path is not None:
            target_path = Path(self.debug_qa_targets_path)
            object.__setattr__(self, "debug_qa_targets_path", target_path)
            if self.debug_qa_targets is None:
                object.__setattr__(
                    self,
                    "debug_qa_targets",
                    load_targets(target_path),
                )
        if self.debug_qa_targets is not None:
            object.__setattr__(
                self,
                "debug_qa_targets",
                normalize_targets(list(self.debug_qa_targets)),
            )


@dataclass
class KeyframeExtractionResult:
    final: Any
    output_dir: Path
    caption_log_path: Path
    manifest_path: Path
    manifest_metadata: dict[str, Any]
    sampled_frame_count: int
    pre_rescue_candidate_count: int
    post_rescue_candidate_count: int
    final_frame_count: int
    pipeline_trace_path: Path | None = None
    debug_qa_trace_path: Path | None = None
