"""Resource admission, worker budgets, and MLX-to-Whisper fallback policy."""

from __future__ import annotations

import importlib
import logging
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)

GIB = 1024**3
MEMORY_HEADROOM_PERCENT = 10
CONCURRENCY_POLICIES = ("auto", "serial", "parallel")
RESOURCE_SOURCES = frozenset(
    {
        "macos-memory-pressure",
        "macos-vm-stat",
        "linux-proc-meminfo",
        "windows-global-memory-status",
        "posix-sysconf",
        "unavailable",
        "injected",
    }
)
TRANSCRIPTION_MEMORY_GIB = {
    "tiny": 3,
    "base": 3,
    "small": 4,
    "medium": 6,
    "large": 11,
}
DIARIZATION_MEMORY_GIB = 4
FRAME_MEMORY_GIB = 6

# Accelerate uses the historical vecLib environment variable for its native
# thread-pool limit. These are set inside each spawned worker, before importing
# Torch or a backend that loads one of the native numerical runtimes.
NATIVE_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@dataclass(frozen=True)
class RuntimeResources:
    """A point-in-time resource probe used for one scheduling decision."""

    cpu_count: int
    available_memory_bytes: int | None
    source: str = "injected"

    def __post_init__(self) -> None:
        if self.cpu_count < 1:
            raise ValueError("cpu_count must be at least one")
        if self.available_memory_bytes is not None and self.available_memory_bytes < 0:
            raise ValueError("available_memory_bytes cannot be negative")
        if self.source not in RESOURCE_SOURCES:
            raise ValueError(f"unknown runtime resource source: {self.source!r}")


@dataclass(frozen=True)
class StageDemand:
    """The resources conservatively attributed to a disposable stage worker."""

    stage: str
    device: str
    memory_bytes: int

    def __post_init__(self) -> None:
        if not self.stage.strip():
            raise ValueError("stage must not be empty")
        if not self.device.strip():
            raise ValueError("device must not be empty")
        if self.memory_bytes <= 0:
            raise ValueError("memory_bytes must be positive")

    @property
    def accelerator(self) -> str | None:
        """Return the exclusive physical accelerator owned by this stage."""

        device = self.device.strip().lower()
        if device in {"mlx", "mps"} or device.startswith("mps:"):
            return "apple:0"
        if device == "cuda":
            return "cuda:0"
        if device.startswith("cuda:"):
            return device
        return None

    @property
    def is_cpu(self) -> bool:
        return self.device.strip().lower() == "cpu"


@dataclass(frozen=True)
class WorkerBudget:
    stage: str
    cpu_threads: int


@dataclass(frozen=True)
class ScheduleDecision:
    """An auditable serial/parallel decision and its per-worker CPU budgets."""

    policy: str
    mode: str
    stages: tuple[StageDemand, ...]
    resources: RuntimeResources
    required_memory_bytes: int
    budgets: tuple[WorkerBudget, ...]
    reason: str
    warnings: tuple[str, ...] = ()

    @property
    def parallel(self) -> bool:
        return self.mode == "parallel"

    def cpu_threads_for(self, stage: str) -> int:
        matches = [budget.cpu_threads for budget in self.budgets if budget.stage == stage]
        if len(matches) != 1:
            raise KeyError(f"schedule has no unique CPU budget for stage {stage!r}")
        return matches[0]


@dataclass(frozen=True)
class ActiveStage:
    """A running companion stage that an MLX fallback must account for."""

    demand: StageDemand
    handle: Any


@dataclass(frozen=True)
class TranscriptionExecution:
    """The completed attempt selected by auto mode, including fallback evidence."""

    completion: Any
    handle: Any
    fallback_used: bool
    fallback_schedule: ScheduleDecision | None = None
    waited_for_active_stages: bool = False
    settled_active_stages: tuple[tuple[str, float, str], ...] = ()


def transcription_demand(
    model_name: str,
    *,
    backend: str,
    device: str | None = None,
) -> StageDemand:
    """Build the conservative demand for one effective transcription backend."""

    try:
        memory_gib = TRANSCRIPTION_MEMORY_GIB[model_name]
    except KeyError as exc:
        choices = ", ".join(TRANSCRIPTION_MEMORY_GIB)
        raise ValueError(f"unknown Whisper model {model_name!r}; choose from: {choices}") from exc
    if backend == "mlx":
        if device not in (None, "mlx"):
            raise ValueError("MLX transcription must use the mlx device")
        effective_device = "mlx"
    elif backend == "whisper":
        effective_device = (device or "cpu").strip().lower()
        if effective_device != "cpu" and not (
            effective_device == "cuda" or effective_device.startswith("cuda:")
        ):
            raise ValueError("OpenAI Whisper device must be cpu or cuda")
    else:
        raise ValueError("effective transcription backend must be mlx or whisper")
    return StageDemand("transcription", effective_device, memory_gib * GIB)


def diarization_demand(device: str) -> StageDemand:
    effective_device = device.strip().lower()
    if effective_device != "cpu" and not (
        effective_device == "cuda" or effective_device.startswith("cuda:")
    ):
        raise ValueError("diarization device must be cpu or cuda")
    return StageDemand("diarization", effective_device, DIARIZATION_MEMORY_GIB * GIB)


def frame_demand(device: str) -> StageDemand:
    effective_device = device.strip().lower()
    if (
        effective_device != "cpu"
        and effective_device != "mps"
        and not effective_device.startswith("mps:")
        and effective_device != "cuda"
        and not effective_device.startswith("cuda:")
    ):
        raise ValueError("frame device must be cpu, mps, or cuda")
    return StageDemand("frames", effective_device, FRAME_MEMORY_GIB * GIB)


def _linux_available_memory() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _macos_vm_stat_available_memory() -> int | None:
    try:
        result = subprocess.run(
            ["vm_stat"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    page_match = re.search(r"page size of\s+(\d+) bytes", result.stdout)
    if page_match is None:
        return None
    page_size = int(page_match.group(1))
    if page_size <= 0:
        return None
    available_pages = 0
    matched_labels = 0
    labels = {
        "Pages free",
        "Pages inactive",
        "Pages speculative",
    }
    for line in result.stdout.splitlines():
        label, separator, value = line.partition(":")
        if separator and label in labels:
            try:
                available_pages += int(value.strip().rstrip("."))
                matched_labels += 1
            except ValueError:
                return None
    return available_pages * page_size if matched_labels else None


def _macos_memory_pressure_available_memory() -> int | None:
    pressure_result = None
    total_result = None
    try:
        pressure_result = subprocess.run(
            ["memory_pressure", "-Q"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        total_result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        pass
    if pressure_result is None or total_result is None:
        return None

    percentage_match = re.search(
        r"System-wide\s+memory\s+free\s+percentage\s*:\s*([+-]?\d+)\s*%",
        pressure_result.stdout,
        flags=re.IGNORECASE,
    )
    if percentage_match is None:
        return None
    try:
        percentage = int(percentage_match.group(1))
        total_physical_bytes = int(total_result.stdout.strip())
    except (TypeError, ValueError):
        return None
    if not 0 <= percentage <= 100 or total_physical_bytes <= 0:
        return None
    return total_physical_bytes * percentage // 100


def _windows_available_memory() -> int | None:
    try:
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(status.available_physical)
    except (AttributeError, OSError, ValueError):
        return None


def _sysconf_available_memory() -> int | None:
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return int(pages) * int(page_size)
    except (AttributeError, OSError, ValueError):
        pass
    return None


def _probe_available_memory() -> tuple[int | None, str]:
    if sys.platform.startswith("linux"):
        available = _linux_available_memory()
        if available is not None:
            return available, "linux-proc-meminfo"
        available = _sysconf_available_memory()
        return (
            (available, "posix-sysconf")
            if available is not None
            else (None, "unavailable")
        )
    if sys.platform == "darwin":
        available = _macos_memory_pressure_available_memory()
        if available is not None:
            return available, "macos-memory-pressure"
        available = _macos_vm_stat_available_memory()
        return (
            (available, "macos-vm-stat")
            if available is not None
            else (None, "unavailable")
        )
    if os.name == "nt":
        available = _windows_available_memory()
        return (
            (available, "windows-global-memory-status")
            if available is not None
            else (None, "unavailable")
        )
    available = _sysconf_available_memory()
    return (
        (available, "posix-sysconf")
        if available is not None
        else (None, "unavailable")
    )


def probe_available_memory_bytes() -> int | None:
    """Return currently available physical memory without importing model stacks."""

    available, _source = _probe_available_memory()
    return available


def probe_runtime_resources() -> RuntimeResources:
    available_memory_bytes, source = _probe_available_memory()
    return RuntimeResources(
        cpu_count=max(1, os.cpu_count() or 1),
        available_memory_bytes=available_memory_bytes,
        source=source,
    )


def required_memory_with_headroom(stages: Iterable[StageDemand]) -> int:
    estimated = sum(stage.memory_bytes for stage in stages)
    return (
        estimated * (100 + MEMORY_HEADROOM_PERCENT) + 99
    ) // 100


def _shared_accelerator(stages: Sequence[StageDemand]) -> str | None:
    owners: set[str] = set()
    for stage in stages:
        accelerator = stage.accelerator
        if accelerator is not None and accelerator in owners:
            return accelerator
        if accelerator is not None:
            owners.add(accelerator)
    return None


class StageScheduler:
    """Make conservative, freshly-probed scheduling decisions."""

    def __init__(
        self,
        policy: str = "auto",
        *,
        resource_probe: Callable[[], RuntimeResources] = probe_runtime_resources,
        logger: logging.Logger | None = None,
    ) -> None:
        if policy not in CONCURRENCY_POLICIES:
            choices = ", ".join(CONCURRENCY_POLICIES)
            raise ValueError(f"unknown stage concurrency {policy!r}; choose from: {choices}")
        self.policy = policy
        self.resource_probe = resource_probe
        self.logger = logger or LOGGER

    def decide(self, stages: Iterable[StageDemand]) -> ScheduleDecision:
        stage_tuple = tuple(stages)
        if not stage_tuple:
            raise ValueError("at least one stage is required")
        names = [stage.stage for stage in stage_tuple]
        if len(set(names)) != len(names):
            raise ValueError("stage names must be unique within one scheduling decision")

        probe_warning: str | None = None
        try:
            resources = self.resource_probe()
            if not isinstance(resources, RuntimeResources):
                raise TypeError("resource probe did not return RuntimeResources")
        except Exception as exc:
            resources = RuntimeResources(
                max(1, os.cpu_count() or 1),
                None,
                source="unavailable",
            )
            probe_warning = (
                "resource admission probe failed; memory admission is unavailable: "
                f"{type(exc).__name__}: {exc}"
            )
        required_memory = required_memory_with_headroom(stage_tuple)
        shared_accelerator = _shared_accelerator(stage_tuple)
        cpu_stages = tuple(stage for stage in stage_tuple if stage.is_cpu)
        cpu_overlap_supported = len(cpu_stages) < 2 or (
            len(stage_tuple) == 2
            and {stage.stage for stage in cpu_stages}
            == {"transcription", "diarization"}
        )
        warnings: list[str] = [probe_warning] if probe_warning else []

        if len(stage_tuple) == 1:
            mode = "serial"
            reason = "only one stage is ready"
        elif shared_accelerator is not None:
            mode = "serial"
            reason = f"stages share exclusive accelerator {shared_accelerator}"
            if self.policy == "parallel":
                warnings.append(
                    "stage-concurrency=parallel cannot override shared-accelerator exclusion"
                )
        elif len(cpu_stages) >= 2 and not cpu_overlap_supported:
            mode = "serial"
            reason = "CPU frame work cannot overlap another CPU stage"
            if self.policy == "parallel":
                warnings.append(
                    "stage-concurrency=parallel cannot override CPU frame-stage exclusion"
                )
        elif self.policy == "serial":
            mode = "serial"
            reason = "stage-concurrency=serial was requested"
        else:
            available = resources.available_memory_bytes
            memory_admitted = available is not None and available >= required_memory
            cpu_pair_admitted = (
                cpu_overlap_supported
                and (len(cpu_stages) < 2 or resources.cpu_count >= 4)
            )

            if self.policy == "parallel":
                mode = "parallel"
                bypassed: list[str] = []
                if not cpu_pair_admitted:
                    bypassed.append("automatic CPU-stage admission")
                if available is None:
                    bypassed.append("unavailable memory admission")
                elif not memory_admitted:
                    bypassed.append("the memory headroom check")
                details = ", ".join(bypassed) if bypassed else "automatic admission"
                warnings.append(
                    "stage-concurrency=parallel forces overlap past "
                    f"{details}; shared accelerators remain exclusive"
                )
                reason = "explicit parallel override"
            elif len(cpu_stages) >= 2 and not cpu_pair_admitted:
                mode = "serial"
                reason = (
                    "CPU stage overlap requires at least four CPUs "
                    f"(detected {resources.cpu_count})"
                )
            elif available is None:
                mode = "serial"
                reason = "available memory could not be determined"
            elif not memory_admitted:
                mode = "serial"
                reason = (
                    "memory admission failed: "
                    f"required {required_memory} bytes with headroom, "
                    f"available {available} bytes"
                )
            else:
                mode = "parallel"
                reason = "CPU, memory, and accelerator admission succeeded"

        thread_slots = len(stage_tuple) if mode == "parallel" else 1
        thread_budget = max(1, resources.cpu_count // thread_slots)
        budgets = tuple(WorkerBudget(stage.stage, thread_budget) for stage in stage_tuple)
        decision = ScheduleDecision(
            policy=self.policy,
            mode=mode,
            stages=stage_tuple,
            resources=resources,
            required_memory_bytes=required_memory,
            budgets=budgets,
            reason=reason,
            warnings=tuple(warnings),
        )
        stage_summary = ", ".join(
            f"{stage.stage}={stage.device}/{decision.cpu_threads_for(stage.stage)}t"
            for stage in stage_tuple
        )
        self.logger.info(
            "Stage schedule: policy=%s mode=%s stages=[%s] reason=%s",
            self.policy,
            mode,
            stage_summary,
            reason,
        )
        for warning in warnings:
            self.logger.warning("Stage schedule warning: %s", warning)
        return decision


def configure_worker_thread_budget(cpu_threads: int | None, *, torch_threads: bool) -> None:
    """Best-effort native and Torch limits for one freshly spawned worker."""

    if cpu_threads is None:
        return
    try:
        budget = int(cpu_threads)
    except (TypeError, ValueError) as exc:
        raise ValueError("worker CPU thread budget must be an integer") from exc
    if budget < 1:
        raise ValueError("worker CPU thread budget must be at least one")

    value = str(budget)
    for variable in NATIVE_THREAD_ENVIRONMENT:
        os.environ[variable] = value
    if not torch_threads:
        return
    try:
        torch = importlib.import_module("torch")
    except Exception:
        return
    try:
        torch.set_num_threads(budget)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(budget)
    except Exception:
        pass


def _handle_is_running(handle: Any) -> bool:
    process = handle.process
    return process.pid is not None and process.is_alive()


def complete_transcription_with_auto_fallback(
    supervisor: Any,
    transcription_handle: Any,
    *,
    scheduler: StageScheduler,
    video_path: str | Path,
    model_name: str,
    requested_backend: str,
    effective_backend: str,
    active_stages: Iterable[ActiveStage] = (),
    final_output_paths: Iterable[str | Path] = (),
    clock: Callable[[], float] = time.monotonic,
) -> TranscriptionExecution:
    """Complete MLX auto mode, relaunching eligible failures in a fresh worker.

    The initial handle is always fully joined by ``StageSupervisor.complete``
    before this function considers fallback. A fresh resource probe then decides
    whether CPU Whisper may overlap any still-running companion stages.
    """

    from keyframe.stage_supervisor import (
        StageProtocolError,
        StageSupervisorError,
        StageWorkerError,
    )

    try:
        completion = supervisor.complete(transcription_handle)
    except StageWorkerError as exc:
        if (
            requested_backend != "auto"
            or effective_backend != "mlx"
            or not exc.fallback_eligible
        ):
            raise
        process = transcription_handle.process
        if process.is_alive() or process.exitcode is None:
            raise StageProtocolError(
                "MLX worker must exit before starting its Whisper fallback"
            ) from exc

        still_running = tuple(
            active for active in active_stages if _handle_is_running(active.handle)
        )
        fallback = transcription_demand(
            model_name,
            backend="whisper",
            device="cpu",
        )
        fallback_schedule = scheduler.decide(
            (fallback, *(active.demand for active in still_running))
        )
        waited_for_active_stages = bool(still_running) and not fallback_schedule.parallel
        settled_active_stages: list[tuple[str, float, str]] = []
        if waited_for_active_stages:
            for active in still_running:
                outcome = "completed"
                try:
                    active.handle.wait()
                except StageSupervisorError:
                    outcome = "failed"
                    # Preserve the companion's cached failure for its owner. A
                    # diarization failure must not suppress a valid transcript.
                    pass
                finally:
                    settled_active_stages.append(
                        (active.demand.stage, clock(), outcome)
                    )

        scheduler.logger.warning(
            "Eligible MLX failure (%s); starting fresh CPU Whisper worker with %s threads",
            exc.error_type or type(exc).__name__,
            fallback_schedule.cpu_threads_for("transcription"),
        )
        try:
            fallback_handle = supervisor.start_transcription(
                video_path,
                model_name=model_name,
                requested_backend="whisper",
                final_output_paths=final_output_paths,
                thread_budget=fallback_schedule.cpu_threads_for("transcription"),
            )
            fallback_completion = supervisor.complete(fallback_handle)
        except BaseException as fallback_exc:
            fallback_exc.settled_active_stages = tuple(settled_active_stages)
            raise
        return TranscriptionExecution(
            completion=fallback_completion,
            handle=fallback_handle,
            fallback_used=True,
            fallback_schedule=fallback_schedule,
            waited_for_active_stages=waited_for_active_stages,
            settled_active_stages=tuple(settled_active_stages),
        )
    return TranscriptionExecution(
        completion=completion,
        handle=transcription_handle,
        fallback_used=False,
    )
