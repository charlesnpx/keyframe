"""Supported-platform and import-only frame runtime preflight."""

from __future__ import annotations

import importlib
import platform
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType


class FramePreflightError(ValueError):
    """Frame extraction cannot start on the current installation."""


@dataclass(frozen=True)
class FrameRuntimePlatform:
    system: str
    machine: str

    @property
    def normalized(self) -> tuple[str, str]:
        return self.system, self.machine.lower()

    @property
    def supports_frames(self) -> bool:
        return self.normalized in {
            ("Darwin", "arm64"),
            ("Linux", "x86_64"),
        }

    @property
    def requires_paddle(self) -> bool:
        return self.normalized == ("Linux", "x86_64")


def current_frame_runtime_platform() -> FrameRuntimePlatform:
    return FrameRuntimePlatform(platform.system(), platform.machine())


def preflight_frame_runtime(
    runtime_platform: FrameRuntimePlatform | None = None,
    *,
    importer: Callable[[str], ModuleType] = importlib.import_module,
) -> FrameRuntimePlatform:
    runtime = runtime_platform or current_frame_runtime_platform()
    if not runtime.supports_frames:
        raise FramePreflightError(
            "frame extraction is supported only on Darwin ARM64 and Linux "
            f"x86-64 (detected {runtime.system} {runtime.machine}); use "
            "--transcript-only when the input has usable audio"
        )
    try:
        importer("keyframe.frames")
        if runtime.requires_paddle:
            paddleocr = importer("paddleocr")
            constructor = getattr(paddleocr, "PaddleOCR")
            if not callable(constructor):
                raise TypeError("paddleocr.PaddleOCR is not callable")
    except Exception as exc:
        platform_name = "Linux" if runtime.requires_paddle else "Darwin ARM64"
        raise FramePreflightError(
            f"{platform_name} frame dependencies are incomplete: "
            f"{type(exc).__name__}: {exc}. Reinstall Keyframe, or rerun with "
            "--transcript-only"
        ) from exc
    return runtime


def resolve_frame_execution_device(
    runtime_platform: FrameRuntimePlatform,
    *,
    importer: Callable[[str], ModuleType] = importlib.import_module,
) -> str:
    try:
        torch = importer("torch")
        if runtime_platform.normalized == ("Darwin", "arm64"):
            return "mps" if bool(torch.backends.mps.is_available()) else "cpu"
        if runtime_platform.normalized == ("Linux", "x86_64"):
            return "cuda" if bool(torch.cuda.is_available()) else "cpu"
    except Exception as exc:
        raise FramePreflightError(
            "could not resolve the frame execution device during preflight: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    raise FramePreflightError(
        "cannot resolve a frame device for unsupported platform "
        f"{runtime_platform.system} {runtime_platform.machine}"
    )
