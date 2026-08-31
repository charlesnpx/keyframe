from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
        raise ValueError(f"{name} must be at least {minimum:g}")
    if maximum is not None and rendered > maximum:
        raise ValueError(f"{name} must be at most {maximum:g}")
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
    max_primary_candidates: int = 96
    coverage_interval_seconds: float = 90.0
    minimum_settled_dwell_seconds: float = 2.0
    device: str | None = None
    ocr_device: str | None = None
    paddle_runtime: Any | None = None
    max_clustering_memory_mb: int = 2048
    max_frame_cache_mb: int = 8192
    frame_cache_dir: Path | None = None
    verbose_trace: bool = False
    debug_qa_targets_path: Path | None = None
    video_timing: Any | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sample_interval",
            _finite_float("sample_interval", self.sample_interval, positive=True),
        )
        if type(self.pass1_clusters) is not int or not 1 <= self.pass1_clusters <= 64:
            raise ValueError("pass1_clusters must be a true integer from 1 through 64")
        if self.max_output_frames is not None:
            _positive_integer("max_output_frames", self.max_output_frames)
        _positive_integer("max_primary_candidates", self.max_primary_candidates)
        if self.max_primary_candidates < self.pass1_clusters:
            raise ValueError("max_primary_candidates must be at least pass1_clusters")
        object.__setattr__(
            self,
            "coverage_interval_seconds",
            _finite_float(
                "coverage_interval_seconds",
                self.coverage_interval_seconds,
                positive=True,
            ),
        )
        object.__setattr__(
            self,
            "minimum_settled_dwell_seconds",
            _finite_float(
                "minimum_settled_dwell_seconds",
                self.minimum_settled_dwell_seconds,
                positive=True,
            ),
        )
        object.__setattr__(
            self,
            "similarity_threshold",
            _finite_float(
                "similarity_threshold",
                self.similarity_threshold,
                minimum=0,
                maximum=1,
            ),
        )
        _positive_integer("max_clustering_memory_mb", self.max_clustering_memory_mb)
        _positive_integer("max_frame_cache_mb", self.max_frame_cache_mb)
        if self.frame_cache_dir is not None:
            object.__setattr__(self, "frame_cache_dir", Path(self.frame_cache_dir))


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
    frame_device: str = "cpu"
    ocr_device: str = "cpu"
