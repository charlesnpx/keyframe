#!/usr/bin/env python3
"""Validate a clean Keyframe installation without loading or downloading models."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any


EXPECTED_KEYFRAME_VERSION = "0.6.3"
EXPECTED_MLX_VERSIONS = {
    "mlx": "0.32.0",
    "mlx-whisper": "0.4.3",
}
EXPECTED_PADDLE_RANGES = {
    "paddlepaddle": ((3, 3, 1), (4, 0, 0)),
    "paddleocr": ((3, 7, 0), (4, 0, 0)),
}
EXPECTED_PADDLE_SPECIFIERS = {
    "paddlepaddle": ">=3.3.1,<4",
    "paddleocr": ">=3.7,<4",
}
IMPORT_SMOKE_MODULES = (
    "keyframe",
    "keyframe.cli",
    "keyframe.frame_preflight",
    "keyframe.media_preflight",
    "keyframe.pipeline.config",
    "keyframe.transcript",
    "keyframe.stage_supervisor",
    "keyframe.full_pipeline",
)
_IMPORT_REPORT_PREFIX = "KEYFRAME_IMPORT_REPORT="
_ISOLATED_FRAME_IMPORT = r"""
import importlib
import json
import os
from pathlib import Path
import socket

attempts = []
constructor_calls = []

class NetworkAttemptError(RuntimeError):
    pass

_real_socket = socket.socket

class GuardedSocket(_real_socket):
    def connect(self, address):
        attempts.append(repr(address))
        raise NetworkAttemptError(f"network disabled during import validation: {address!r}")

    def connect_ex(self, address):
        attempts.append(repr(address))
        raise NetworkAttemptError(f"network disabled during import validation: {address!r}")

def blocked_create_connection(address, *args, **kwargs):
    attempts.append(repr(address))
    raise NetworkAttemptError(f"network disabled during import validation: {address!r}")

def blocked_getaddrinfo(host, port, *args, **kwargs):
    attempts.append(repr((host, port)))
    raise NetworkAttemptError(
        f"network disabled during import validation: {(host, port)!r}"
    )

socket.socket = GuardedSocket
socket.create_connection = blocked_create_connection
socket.getaddrinfo = blocked_getaddrinfo

validation_root = Path(os.environ["KEYFRAME_IMPORT_ROOT"])
cache_root = Path(os.environ["KEYFRAME_IMPORT_CACHE_ROOT"])
paddleocr = importlib.import_module("paddleocr")
original_constructor = getattr(paddleocr, "PaddleOCR")
if not callable(original_constructor):
    raise TypeError("paddleocr.PaddleOCR is not callable")

def forbidden_constructor(*args, **kwargs):
    constructor_calls.append({"args": len(args), "kwargs": sorted(kwargs)})
    raise RuntimeError("PaddleOCR construction is forbidden during import validation")

paddleocr.PaddleOCR = forbidden_constructor
frames = importlib.import_module("keyframe.frames")

checkpoint_names = {
    "model.safetensors",
    "pytorch_model.bin",
    "model_state.pdparams",
    "inference.pdmodel",
    "inference.pdiparams",
}
checkpoint_suffixes = {
    ".pdparams",
    ".pdmodel",
    ".pdiparams",
    ".safetensors",
    ".onnx",
}
checkpoints = []
for candidate in validation_root.rglob("*"):
    if not candidate.is_file():
        continue
    lowered = candidate.name.lower()
    model_cache_path = any(
        part.lower() == "snapshots" or part.lower().startswith("models--")
        for part in candidate.parts
    )
    if (
        lowered in checkpoint_names
        or candidate.suffix.lower() in checkpoint_suffixes
        or model_cache_path
    ):
        checkpoints.append(str(candidate.relative_to(validation_root)))

report = {
    "imports": [frames.__name__, paddleocr.__name__],
    "network_attempts": attempts,
    "paddleocr_constructor_calls": constructor_calls,
    "recognizable_checkpoints": checkpoints,
}
print("KEYFRAME_IMPORT_REPORT=" + json.dumps(report, sort_keys=True))
"""


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


def _paddle_requirements() -> dict[str, str]:
    try:
        requirements = importlib_metadata.requires("keyframe") or []
    except importlib_metadata.PackageNotFoundError as exc:
        raise InstallValidationError("the keyframe distribution is not installed") from exc
    return {
        _requirement_name(requirement): requirement
        for requirement in requirements
        if _requirement_name(requirement) in EXPECTED_PADDLE_RANGES
    }


def _is_supported_mlx_runtime() -> bool:
    if platform.system() != "Darwin" or platform.machine().lower() != "arm64":
        return False
    release = platform.release().split(".", 1)[0]
    try:
        return int(release) >= 23
    except ValueError:
        return False


def _is_linux_x86_64() -> bool:
    return (
        platform.system() == "Linux"
        and platform.machine().lower() == "x86_64"
    )


def _version_tuple(version: str, *, distribution: str) -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", version)
    if match is None:
        raise InstallValidationError(
            f"installed {distribution} has an unrecognized version: {version!r}"
        )
    return tuple(int(value or 0) for value in match.groups())


def _validate_paddle_install() -> dict[str, str | None]:
    requirements = _paddle_requirements()
    if set(requirements) != set(EXPECTED_PADDLE_RANGES):
        raise InstallValidationError(
            "installed keyframe metadata does not contain both gated Paddle "
            "requirements"
        )
    for name in EXPECTED_PADDLE_RANGES:
        compact = requirements[name].replace(" ", "").lower()
        required_specifiers = EXPECTED_PADDLE_SPECIFIERS[name].split(",")
        requirement_head = compact.partition(";")[0]
        if not requirement_head.startswith(name) or not all(
            specifier in requirement_head for specifier in required_specifiers
        ):
            raise InstallValidationError(
                f"installed keyframe metadata has an incorrect {name} range"
            )
        marker = requirements[name].partition(";")[2].lower()
        if not all(
            token in marker
            for token in ("sys_platform", "platform_machine", "linux", "x86_64")
        ):
            raise InstallValidationError(
                f"installed {name} requirement is missing the Linux x86-64 gate"
            )

    installed = {
        name: _distribution_version(name) for name in EXPECTED_PADDLE_RANGES
    }
    if _is_linux_x86_64():
        missing = [name for name, version in installed.items() if version is None]
        if missing:
            raise InstallValidationError(
                f"Linux x86-64 install is missing default Paddle packages: {missing}"
            )
        for name, version in installed.items():
            assert version is not None
            parsed = _version_tuple(version, distribution=name)
            minimum, maximum = EXPECTED_PADDLE_RANGES[name]
            if not minimum <= parsed < maximum:
                raise InstallValidationError(
                    f"installed {name} {version} is outside the supported range"
                )
    else:
        unexpected = {
            name: version for name, version in installed.items() if version is not None
        }
        if unexpected:
            raise InstallValidationError(
                f"non-Linux frame install contains Paddle distributions: {unexpected}"
            )
    return installed


def _run_isolated_frame_import_validation() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="keyframe-import-validation-") as raw_root:
        root = Path(raw_root)
        cache_root = root / "cache"
        home_root = root / "home"
        temp_root = root / "tmp"
        cache_root.mkdir()
        home_root.mkdir()
        temp_root.mkdir()
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        environment.update(
            {
                "KEYFRAME_IMPORT_ROOT": str(root),
                "KEYFRAME_IMPORT_CACHE_ROOT": str(cache_root),
                "HOME": str(home_root),
                "TMPDIR": str(temp_root),
                "TEMP": str(temp_root),
                "TMP": str(temp_root),
                "XDG_CACHE_HOME": str(cache_root / "xdg"),
                "HF_HOME": str(cache_root / "huggingface"),
                "HUGGINGFACE_HUB_CACHE": str(cache_root / "huggingface" / "hub"),
                "TRANSFORMERS_CACHE": str(cache_root / "transformers"),
                "TORCH_HOME": str(cache_root / "torch"),
                "PADDLE_HOME": str(cache_root / "paddle"),
                "PADDLEX_HOME": str(cache_root / "paddlex"),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
                "HTTP_PROXY": "http://127.0.0.1:9",
                "HTTPS_PROXY": "http://127.0.0.1:9",
                "ALL_PROXY": "http://127.0.0.1:9",
                "NO_PROXY": "",
                "no_proxy": "",
            }
        )
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", _ISOLATED_FRAME_IMPORT],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise InstallValidationError(
                f"isolated Linux frame imports could not complete: {exc}"
            ) from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise InstallValidationError(
            "isolated Linux frame imports failed"
            + (f": {detail}" if detail else "")
        )
    report_line = next(
        (
            line.removeprefix(_IMPORT_REPORT_PREFIX)
            for line in reversed(completed.stdout.splitlines())
            if line.startswith(_IMPORT_REPORT_PREFIX)
        ),
        None,
    )
    if report_line is None:
        raise InstallValidationError(
            "isolated Linux frame imports did not emit a validation report"
        )
    try:
        report = json.loads(report_line)
    except json.JSONDecodeError as exc:
        raise InstallValidationError(
            f"isolated Linux frame import report is malformed: {exc}"
        ) from exc
    if report.get("network_attempts"):
        raise InstallValidationError(
            "isolated Linux frame imports attempted network access: "
            f"{report['network_attempts']}"
        )
    if report.get("paddleocr_constructor_calls"):
        raise InstallValidationError(
            "isolated Linux frame imports constructed PaddleOCR"
        )
    if report.get("recognizable_checkpoints"):
        raise InstallValidationError(
            "isolated Linux frame imports created recognizable model checkpoints: "
            f"{report['recognizable_checkpoints']}"
        )
    expected_imports = {"keyframe.frames", "paddleocr"}
    if set(report.get("imports", ())) != expected_imports:
        raise InstallValidationError(
            f"isolated Linux frame imports are incomplete: {report.get('imports')!r}"
        )
    return report


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


def validate_install(expected_platform: str = "auto") -> dict[str, Any]:
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

    installed_paddle = _validate_paddle_install()
    for module_name in IMPORT_SMOKE_MODULES:
        importlib.import_module(module_name)
    frame_import_validation = (
        _run_isolated_frame_import_validation()
        if _is_linux_x86_64()
        else None
    )

    return {
        "passed": True,
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "platform_release": platform.release(),
        "keyframe": keyframe_version,
        "supports_mlx": supports_mlx,
        "installed_mlx": installed_mlx,
        "installed_paddle": installed_paddle,
        "imports": list(IMPORT_SMOKE_MODULES),
        "frame_import_validation": frame_import_validation,
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
    report = validate_install(args.expect_platform)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InstallValidationError as exc:
        print(f"Install validation error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
