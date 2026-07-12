from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


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

    def __post_init__(self) -> None:
        if not 1 <= int(self.pass1_clusters) <= 64:
            raise ValueError("pass1_clusters must be between 1 and 64")
        if int(self.max_clustering_memory_mb) <= 0:
            raise ValueError("max_clustering_memory_mb must be positive")
        if int(self.max_frame_cache_mb) <= 0:
            raise ValueError("max_frame_cache_mb must be positive")


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
