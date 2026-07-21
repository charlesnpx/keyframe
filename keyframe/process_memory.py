"""Kernel-backed process high-water memory measurements."""

from __future__ import annotations

import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass


def _ru_maxrss_bytes(value: int, *, platform_name: str = sys.platform) -> int:
    """Normalize ``resource.ru_maxrss`` to bytes on supported platforms."""

    peak = int(value)
    if peak < 0:
        raise ValueError("ru_maxrss must be non-negative")
    if platform_name == "darwin":
        return peak
    if platform_name.startswith("linux"):
        return peak * 1024
    raise OSError(f"ru_maxrss units are unsupported on {platform_name!r}")


def resource_peak_rss_bytes(who: str) -> int:
    """Return the kernel-recorded self or reaped-children RSS high-water mark."""

    if who not in {"self", "children"}:
        raise ValueError("resource peak selector must be 'self' or 'children'")
    try:
        import resource
    except ImportError as exc:  # pragma: no cover - Windows release is unsupported
        raise OSError("resource usage is unavailable") from exc
    selector = resource.RUSAGE_SELF if who == "self" else resource.RUSAGE_CHILDREN
    try:
        peak = resource.getrusage(selector).ru_maxrss
    except (AttributeError, OSError, ValueError) as exc:
        raise OSError(f"could not read {who} RSS high-water mark") from exc
    return _ru_maxrss_bytes(peak)


def process_tree_high_water_rss_bytes() -> int:
    """Conservatively bound one worker and any children it already reaped."""

    return resource_peak_rss_bytes("self") + resource_peak_rss_bytes("children")


@dataclass(frozen=True)
class ProcessTreeHighWaterEvidence:
    """Components of a conservative process-tree RSS high-water bound."""

    case_process_bytes: int
    max_reaped_child_bytes: int
    concurrent_stage_sum_bytes: int
    descendant_bound_bytes: int
    tree_upper_bound_bytes: int
    phase_upper_bound_bytes: Mapping[str, int]

    def to_dict(self) -> dict[str, int | dict[str, int]]:
        return {
            "case_process_bytes": self.case_process_bytes,
            "max_reaped_child_bytes": self.max_reaped_child_bytes,
            "concurrent_stage_sum_bytes": self.concurrent_stage_sum_bytes,
            "descendant_bound_bytes": self.descendant_bound_bytes,
            "tree_upper_bound_bytes": self.tree_upper_bound_bytes,
            "phase_upper_bound_bytes": dict(self.phase_upper_bound_bytes),
        }


def conservative_process_tree_high_water(
    *,
    case_process_bytes: int,
    max_reaped_child_bytes: int,
    concurrent_stage_peaks: Mapping[str, int] | Iterable[int] = (),
    case_phase_peaks: Mapping[str, int] | None = None,
    phase_stage_peaks: Mapping[str, Iterable[int]] | None = None,
) -> ProcessTreeHighWaterEvidence:
    """Bound the tree using persistent per-process kernel high-water marks.

    ``RUSAGE_CHILDREN`` preserves a short-lived child's high-water mark after it
    exits. Concurrent long-lived model stages report their own tree high-water
    marks, which are summed so their overlap cannot be hidden by a per-child
    maximum.
    """

    if isinstance(concurrent_stage_peaks, Mapping):
        values = tuple(concurrent_stage_peaks.values())
    else:
        values = tuple(concurrent_stage_peaks)
    phases = dict(case_phase_peaks or {"complete": case_process_bytes})
    stage_phases = dict(
        {name: values for name in phases}
        if phase_stage_peaks is None
        else phase_stage_peaks
    )
    if set(phases) != set(stage_phases):
        raise ValueError("case and stage high-water phases must match")
    phase_stage_values = {
        name: tuple(stage_phases[name]) for name in phases
    }
    measurements = (
        case_process_bytes,
        max_reaped_child_bytes,
        *values,
        *phases.values(),
        *(value for group in phase_stage_values.values() for value in group),
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in measurements
    ):
        raise TypeError("process high-water measurements must be integers")
    if any(value < 0 for value in measurements):
        raise ValueError("process high-water measurements must be non-negative")
    if max(phases.values(), default=0) != case_process_bytes:
        raise ValueError("case process high-water does not match its phase maximum")
    phase_stage_sums = {
        name: sum(phase_stage_values[name]) for name in phases
    }
    phase_descendant_bounds = {
        name: max(max_reaped_child_bytes, phase_stage_sums[name])
        for name in phases
    }
    phase_upper_bounds = {
        name: phases[name] + phase_descendant_bounds[name]
        for name in phases
    }
    stage_sum = max(phase_stage_sums.values(), default=0)
    descendant_bound = max(phase_descendant_bounds.values(), default=0)
    tree_upper_bound = max(phase_upper_bounds.values(), default=case_process_bytes)
    return ProcessTreeHighWaterEvidence(
        case_process_bytes=case_process_bytes,
        max_reaped_child_bytes=max_reaped_child_bytes,
        concurrent_stage_sum_bytes=stage_sum,
        descendant_bound_bytes=descendant_bound,
        tree_upper_bound_bytes=tree_upper_bound,
        phase_upper_bound_bytes=phase_upper_bounds,
    )
