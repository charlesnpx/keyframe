from __future__ import annotations

from dataclasses import dataclass, field
import time

from keyframe.pipeline.config import KeyframeExtractionConfig
from keyframe.pipeline.trace import TraceSink


@dataclass
class RunContext:
    config: KeyframeExtractionConfig
    trace: TraceSink
    run_id: str
    started_at: float
    metadata: dict[str, object] = field(default_factory=dict)


def make_context(config: KeyframeExtractionConfig, trace: TraceSink) -> RunContext:
    return RunContext(
        config=config,
        trace=trace,
        run_id=f"keyframe-{int(time.time() * 1000)}",
        started_at=time.time(),
    )
