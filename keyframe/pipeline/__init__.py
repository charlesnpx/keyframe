"""Shared key-frame extraction pipeline orchestration."""

from keyframe.pipeline.config import KeyframeExtractionConfig, KeyframeExtractionResult
from keyframe.pipeline.orchestrator import extract_keyframes
from keyframe.pipeline.trace import NoOpTraceSink, SnapshotTraceSink, TraceSink

__all__ = [
    "KeyframeExtractionConfig",
    "KeyframeExtractionResult",
    "NoOpTraceSink",
    "SnapshotTraceSink",
    "TraceSink",
    "extract_keyframes",
]
