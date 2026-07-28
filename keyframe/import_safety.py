from __future__ import annotations

import platform
import sys
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from typing import Any


@contextmanager
def defer_optional_pyav_import(
    *,
    system: str | None = None,
    modules: MutableMapping[str, Any] | None = None,
) -> Iterator[None]:
    """Keep TorchVision's unused PyAV video backend out of the macOS frame runtime."""

    runtime_system = platform.system() if system is None else system
    loaded_modules = sys.modules if modules is None else modules
    if runtime_system != "Darwin" or "av" in loaded_modules:
        yield
        return

    # OpenCV and PyAV wheels bundle conflicting AVFoundation classes on macOS.
    # A None entry makes TorchVision treat its optional PyAV import as absent.
    loaded_modules["av"] = None
    try:
        yield
    finally:
        if loaded_modules.get("av") is None:
            loaded_modules.pop("av", None)
