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
    verbose_trace: bool = False
    debug_qa_targets_path: Path | None = None


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
