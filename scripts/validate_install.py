#!/usr/bin/env python3
"""Validate a clean Keyframe installation without loading or downloading models."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import re
import sys
from importlib import metadata as importlib_metadata
from typing import Any


EXPECTED_KEYFRAME_VERSION = "0.6.5"
EXPECTED_MLX_VERSIONS = {
    "mlx": "0.32.0",
    "mlx-whisper": "0.4.3",
}
EXPECTED_LINUX_X86_64_PADDLE_OCR = "paddleocr"
PADDLE_ENGINES = ("paddlepaddle", "paddlepaddle-gpu")
IMPORT_SMOKE_MODULES = (
    "keyframe",
    "keyframe.cli",
    "keyframe.transcript",
    "keyframe.stage_supervisor",
    "keyframe.full_pipeline",
)


class InstallValidationError(RuntimeError):
    """The installed distribution does not satisfy the release contract."""


def _distribution_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _requirement_name(requirement: str) -> str:
    raw_name = re.split(r"[\s\[\]();<>=!~]", requirement, maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", raw_name).lower()


def _mlx_requirements() -> dict[str, str]:
    try:
        requirements = importlib_metadata.requires("keyframe") or []
    except importlib_metadata.PackageNotFoundError as exc:
        raise InstallValidationError("the keyframe distribution is not installed") from exc
    return {
        _requirement_name(requirement): requirement
        for requirement in requirements
        if _requirement_name(requirement) in EXPECTED_MLX_VERSIONS
    }


def _requirements_by_name() -> dict[str, list[str]]:
    try:
        requirements = importlib_metadata.requires("keyframe") or []
    except importlib_metadata.PackageNotFoundError as exc:
        raise InstallValidationError("the keyframe distribution is not installed") from exc
    grouped: dict[str, list[str]] = {}
    for requirement in requirements:
        grouped.setdefault(_requirement_name(requirement), []).append(requirement)
    return grouped


def _is_supported_mlx_runtime() -> bool:
    if platform.system() != "Darwin" or platform.machine().lower() != "arm64":
        return False
    release = platform.release().split(".", 1)[0]
    try:
        return int(release) >= 23
    except ValueError:
        return False


def _validate_expected_platform(expected: str) -> None:
    actual = (platform.system(), platform.machine().lower())
    if expected == "auto":
        return
    if expected == "darwin-arm64" and actual != ("Darwin", "arm64"):
        raise InstallValidationError(
            f"expected Darwin ARM64, found {actual[0]} {actual[1]}"
        )
    if expected == "linux-x86_64" and actual != ("Linux", "x86_64"):
        raise InstallValidationError(
            f"expected Linux x86_64, found {actual[0]} {actual[1]}"
        )


def validate_install(
    expected_platform: str = "auto",
    *,
    setup_paddle: bool = False,
) -> dict[str, Any]:
    """Return a machine-readable clean-install report or raise on mismatch."""

    if not ((3, 11) <= sys.version_info[:2] < (3, 14)):
        raise InstallValidationError(
            f"unsupported Python {platform.python_version()}; expected 3.11 through 3.13"
        )
    _validate_expected_platform(expected_platform)

    keyframe_version = _distribution_version("keyframe")
    if keyframe_version != EXPECTED_KEYFRAME_VERSION:
        raise InstallValidationError(
            f"expected keyframe {EXPECTED_KEYFRAME_VERSION}, found {keyframe_version!r}"
        )

    requirements_by_name = _requirements_by_name()
    mlx_requirements = _mlx_requirements()
    if set(mlx_requirements) != set(EXPECTED_MLX_VERSIONS):
        raise InstallValidationError(
            "installed keyframe metadata does not contain both gated MLX requirements"
        )
    for name, version in EXPECTED_MLX_VERSIONS.items():
        requirement = mlx_requirements[name]
        if f"{name}=={version}" not in requirement.replace(" ", "").lower():
            raise InstallValidationError(
                f"installed keyframe metadata does not pin {name}=={version}"
            )
        marker = requirement.partition(";")[2].lower()
        if not all(
            token in marker
            for token in ("sys_platform", "platform_machine", "platform_release")
        ):
            raise InstallValidationError(
                f"installed {name} requirement is missing the platform gate"
            )

    supports_mlx = _is_supported_mlx_runtime()
    installed_mlx = {
        name: _distribution_version(name) for name in EXPECTED_MLX_VERSIONS
    }
    if supports_mlx:
        mismatches = {
            name: installed_mlx[name]
            for name, expected_version in EXPECTED_MLX_VERSIONS.items()
            if installed_mlx[name] != expected_version
        }
        if mismatches:
            raise InstallValidationError(
                f"supported Darwin ARM64 install has incorrect MLX packages: {mismatches}"
            )
    else:
        unexpected = {
            name: version for name, version in installed_mlx.items() if version is not None
        }
        if unexpected:
            raise InstallValidationError(
                f"unsupported platform installed MLX distributions: {unexpected}"
            )

    if EXPECTED_LINUX_X86_64_PADDLE_OCR not in requirements_by_name:
        raise InstallValidationError(
            "installed keyframe metadata is missing the PaddleOCR dependency"
        )
    paddleocr_markers = [
        requirement.partition(";")[2].lower()
        for requirement in requirements_by_name[EXPECTED_LINUX_X86_64_PADDLE_OCR]
    ]
    if not any(
        "sys_platform" in marker
        and "linux" in marker
        and "platform_machine" in marker
        and "x86_64" in marker
        for marker in paddleocr_markers
    ):
        raise InstallValidationError(
            "installed paddleocr requirement is missing the Linux x86_64 platform gate"
        )

    paddle_engine_requirements = requirements_by_name.get("paddlepaddle", [])
    if not any(
        "paddlepaddle==3.3.1" in requirement.replace(" ", "").lower()
        and "extra" in requirement.partition(";")[2].lower()
        and "linux" in requirement.partition(";")[2].lower()
        and "platform_machine" in requirement.partition(";")[2].lower()
        for requirement in paddle_engine_requirements
    ):
        raise InstallValidationError(
            "the Linux x86-64 paddle extra must pin CPU paddlepaddle==3.3.1"
        )
    if any(
        "linux" in requirement.partition(";")[2].lower()
        and "extra" not in requirement.partition(";")[2].lower()
        for requirement in paddle_engine_requirements
    ):
        raise InstallValidationError(
            "Linux metadata must not force-install a Paddle engine before setup"
        )

    initial_paddle_engines = {
        name: _distribution_version(name) for name in PADDLE_ENGINES
    }
    installed_paddleocr = _distribution_version("paddleocr")
    paddle_setup = None
    actual_platform = (platform.system(), platform.machine().lower())
    if actual_platform == ("Darwin", "arm64"):
        unexpected_engines = {
            name: version
            for name, version in {
                **initial_paddle_engines,
                "paddleocr": installed_paddleocr,
            }.items()
            if version is not None
        }
        if unexpected_engines:
            raise InstallValidationError(
                f"macOS clean install contains unexpected Paddle engines: {unexpected_engines}"
            )
    if setup_paddle and actual_platform == ("Linux", "x86_64"):
        if installed_paddleocr is None:
            raise InstallValidationError(
                "clean Linux install is missing the gated PaddleOCR distribution"
            )
        unexpected = {
            name: version
            for name, version in initial_paddle_engines.items()
            if version is not None
        }
        if unexpected:
            raise InstallValidationError(
                f"clean Linux install already contains a Paddle engine: {unexpected}"
            )
        from keyframe.paddle_runtime import PaddleSetupError, ensure_paddle_runtime

        try:
            selection = ensure_paddle_runtime()
        except PaddleSetupError as exc:
            raise InstallValidationError(f"automatic Paddle setup failed: {exc}") from exc
        paddle_setup = selection.to_dict()
        installed_after_setup = {
            name: _distribution_version(name) for name in PADDLE_ENGINES
        }
        expected_distribution = selection.distribution
        if (
            expected_distribution is None
            or installed_after_setup.get(expected_distribution) != "3.3.1"
            or sum(version is not None for version in installed_after_setup.values()) != 1
        ):
            raise InstallValidationError(
                "Paddle setup did not leave exactly one selected 3.3.1 engine: "
                f"{installed_after_setup}"
            )

    for module_name in IMPORT_SMOKE_MODULES:
        importlib.import_module(module_name)

    return {
        "passed": True,
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "platform_release": platform.release(),
        "keyframe": keyframe_version,
        "supports_mlx": supports_mlx,
        "installed_mlx": installed_mlx,
        "linux_x86_64_paddleocr_requirement": EXPECTED_LINUX_X86_64_PADDLE_OCR,
        "initial_paddle_engines": initial_paddle_engines,
        "installed_paddleocr": installed_paddleocr,
        "paddle_setup": paddle_setup,
        "imports": list(IMPORT_SMOKE_MODULES),
        "model_acquisition_attempted": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate an installed Keyframe release without loading models.",
    )
    parser.add_argument(
        "--expect-platform",
        choices=("auto", "darwin-arm64", "linux-x86_64"),
        default="auto",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_install(args.expect_platform, setup_paddle=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InstallValidationError as exc:
        print(f"Install validation error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
