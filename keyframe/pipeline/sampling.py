"""Presentation-time sampling primitives shared by video decode passes."""

from __future__ import annotations

import math
from dataclasses import dataclass

PRESENTATION_TIME_EPSILON_SECONDS = 1e-6


class FrameTimingError(RuntimeError):
    """Decoded timing cannot support deterministic frame sampling."""


class DecoderTimingUnavailable(FrameTimingError):
    """The decoder did not expose a usable presentation-time sequence."""


class DecoderTimingRegression(FrameTimingError):
    """Decoder presentation time moved materially backward."""


@dataclass(frozen=True)
class SamplingDecision:
    timestamp: float
    consumed_target: float
    next_target: float


class TargetTimeSampler:
    """Select at most one decoded frame for each reached sampling boundary."""

    def __init__(self, interval_seconds: float):
        interval = float(interval_seconds)
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("sampling interval must be finite and positive")
        self.interval_seconds = interval
        self._next_target_index = 0

    @property
    def next_target(self) -> float:
        return self._next_target_index * self.interval_seconds

    def consider(self, timestamp: float) -> SamplingDecision | None:
        rendered = float(timestamp)
        if not math.isfinite(rendered) or rendered < 0:
            raise FrameTimingError(
                "normalized decoder timestamp must be finite and non-negative"
            )
        effective = rendered + PRESENTATION_TIME_EPSILON_SECONDS
        consumed = self.next_target
        if effective < consumed:
            return None
        next_index = max(
            self._next_target_index + 1,
            math.floor(effective / self.interval_seconds) + 1,
        )
        while next_index * self.interval_seconds <= effective:
            next_index += 1
        self._next_target_index = next_index
        return SamplingDecision(
            timestamp=rendered,
            consumed_target=consumed,
            next_target=self.next_target,
        )


class DecoderTimestampNormalizer:
    """Normalize decoder time and reject materially non-monotonic timing."""

    def __init__(
        self,
        *,
        epsilon_seconds: float = PRESENTATION_TIME_EPSILON_SECONDS,
    ):
        epsilon = float(epsilon_seconds)
        if not math.isfinite(epsilon) or epsilon < 0:
            raise ValueError("presentation-time epsilon must be finite and non-negative")
        self.epsilon_seconds = epsilon
        self.origin_seconds: float | None = None
        self.previous_seconds: float | None = None
        self.decoded_frame_count = 0
        self._saw_positive_normalized_time = False

    def observe(self, raw_seconds: float) -> float | None:
        self.decoded_frame_count += 1
        try:
            rendered = float(raw_seconds)
        except (TypeError, ValueError, OverflowError) as exc:
            raise DecoderTimingUnavailable(
                "decoder returned a non-numeric presentation timestamp"
            ) from exc
        if not math.isfinite(rendered):
            if self.origin_seconds is None:
                return None
            raise DecoderTimingUnavailable(
                "decoder presentation timestamp became unavailable after timing began"
            )

        if self.origin_seconds is None:
            self.origin_seconds = rendered
            self.previous_seconds = 0.0
            return 0.0

        normalized = rendered - self.origin_seconds
        previous = float(self.previous_seconds)
        if normalized < previous:
            regression = previous - normalized
            if regression <= self.epsilon_seconds:
                normalized = previous
            else:
                raise DecoderTimingRegression(
                    "decoder presentation time moved backward by "
                    f"{regression:.9f} seconds"
                )
        if normalized > self.epsilon_seconds:
            self._saw_positive_normalized_time = True
        self.previous_seconds = normalized
        return normalized

    def finalize(self) -> None:
        if self.origin_seconds is None:
            raise DecoderTimingUnavailable(
                "decoder exposed no finite presentation timestamps"
            )
        if self.decoded_frame_count > 1 and not self._saw_positive_normalized_time:
            raise DecoderTimingUnavailable(
                "decoder exposed all-zero presentation time for a multi-frame source"
            )
