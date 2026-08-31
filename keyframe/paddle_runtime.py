"""Automatic, transactional Paddle runtime selection for Linux frame OCR."""

from __future__ import annotations

import csv
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any


PADDLE_VERSION = "3.3.1"
STATE_SCHEMA_VERSION = 1
MINIMUM_COMPUTE_CAPABILITY = (7, 5)
PADDLE_INDEXES = {
    "cpu": "https://www.paddlepaddle.org.cn/packages/stable/cpu/",
    "cu118": "https://www.paddlepaddle.org.cn/packages/stable/cu118/",
    "cu126": "https://www.paddlepaddle.org.cn/packages/stable/cu126/",
    "cu129": "https://www.paddlepaddle.org.cn/packages/stable/cu129/",
    "cu130": "https://www.paddlepaddle.org.cn/packages/stable/cu130/",
}
CUDA_FLAVORS = (
    ((13, 0), "cu130"),
    ((12, 9), "cu129"),
    ((12, 6), "cu126"),
    ((11, 8), "cu118"),
)


class PaddleSetupError(RuntimeError):
    """No verified Paddle runtime could be established."""

    def __init__(self, message: str, result: PaddleRuntimeSelection | None = None):
        super().__init__(message)
        self.result = result


@dataclass(frozen=True)
class NvidiaDevice:
    logical_index: int
    physical_id: str
    name: str
    capability: tuple[int, int]


@dataclass(frozen=True)
class NvidiaProbe:
    devices: tuple[NvidiaDevice, ...] = ()
    cuda_version: tuple[int, int] | None = None
    reason: str = "no compatible NVIDIA GPU was detected"
    rocm: bool = False

    @property
    def selected_device(self) -> NvidiaDevice | None:
        compatible = tuple(
            device
            for device in self.devices
            if device.capability >= MINIMUM_COMPUTE_CAPABILITY
        )
        if not compatible:
            return None
        return max(
            compatible,
            key=lambda device: (device.capability, -device.logical_index),
        )


@dataclass(frozen=True)
class PaddleRuntimeSelection:
    schema_version: int
    status: str
    distribution: str | None
    version: str | None
    ocr_device: str | None
    cuda_flavor: str | None
    reason: str
    changed: bool = False
    gpu_failure: str | None = None
    pending_cpu_install: bool = False
    state_path: Path | None = field(default=None, compare=False, repr=False)

    @property
    def usable(self) -> bool:
        return self.status in {"gpu", "cpu", "not-applicable"}

    def to_dict(self, *, include_internal: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "status": self.status,
            "distribution": self.distribution,
            "version": self.version,
            "ocr_device": self.ocr_device,
            "cuda_flavor": self.cuda_flavor,
            "changed": self.changed,
            "reason": self.reason,
        }
        if include_internal:
            payload["gpu_failure"] = self.gpu_failure
            payload["pending_cpu_install"] = self.pending_cpu_install
        return payload

    def stored_dict(self) -> dict[str, Any]:
        payload = self.to_dict(include_internal=True)
        payload.pop("changed", None)
        return payload


def state_file_path(environ: Mapping[str, str] | None = None) -> Path:
    environ = os.environ if environ is None else environ
    state_home = environ.get("XDG_STATE_HOME")
    root = Path(state_home).expanduser() if state_home else Path.home() / ".local" / "state"
    return root / "keyframe" / "paddle-runtime.json"


def _parse_version(value: Any) -> tuple[int, int] | None:
    match = re.search(r"(?<!\d)(\d+)\.(\d+)", str(value or ""))
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _parse_capability(value: Any) -> tuple[int, int] | None:
    parsed = _parse_version(value)
    if parsed is None or parsed[0] < 1 or parsed[1] < 0:
        return None
    return parsed


def cuda_flavor_for_version(version: tuple[int, int] | str | None) -> str | None:
    parsed = _parse_version(version) if not isinstance(version, tuple) else version
    if parsed is None:
        return None
    for minimum, flavor in CUDA_FLAVORS:
        if parsed >= minimum:
            return flavor
    return None


def _visible_tokens(environ: Mapping[str, str]) -> tuple[str, ...] | None:
    if "CUDA_VISIBLE_DEVICES" not in environ:
        return None
    raw = environ.get("CUDA_VISIBLE_DEVICES", "")
    tokens = tuple(token.strip() for token in raw.split(",") if token.strip())
    if not tokens or tokens == ("-1",):
        return ()
    return tokens


def _match_visible_device(
    token: str,
    rows: Sequence[tuple[str, str, str, tuple[int, int]]],
) -> tuple[str, str, str, tuple[int, int]] | None:
    if token.isdigit():
        return next((row for row in rows if row[0] == token), None)
    token_lower = token.lower()
    matches = tuple(
        row
        for row in rows
        if row[1].lower() == token_lower or row[1].lower().startswith(token_lower)
    )
    return matches[0] if len(matches) == 1 else None


def parse_nvidia_smi_devices(
    output: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> tuple[NvidiaDevice, ...]:
    """Parse queried devices and apply CUDA visibility/remapping semantics."""

    rows: list[tuple[str, str, str, tuple[int, int]]] = []
    for raw_row in csv.reader(output.splitlines()):
        if len(raw_row) != 4:
            continue
        physical_index, uuid, name, raw_capability = (
            item.strip() for item in raw_row
        )
        capability = _parse_capability(raw_capability)
        if not physical_index.isdigit() or not uuid or not name or capability is None:
            continue
        rows.append((physical_index, uuid, name, capability))

    tokens = _visible_tokens(os.environ if environ is None else environ)
    if tokens is None:
        visible_rows = rows
    else:
        visible_rows = []
        for token in tokens:
            matched = _match_visible_device(token, rows)
            if matched is not None and matched not in visible_rows:
                visible_rows.append(matched)
    return tuple(
        NvidiaDevice(
            logical_index=logical_index,
            physical_id=row[0],
            name=row[2],
            capability=row[3],
        )
        for logical_index, row in enumerate(visible_rows)
    )


def _default_runner(command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess:
    return subprocess.run(command, **kwargs)


_TORCH_PROBE_SCRIPT = r"""
import json
payload = {"cuda_available": False, "cuda_version": None, "hip": None, "devices": []}
try:
    import torch
    payload["hip"] = getattr(torch.version, "hip", None)
    payload["cuda_version"] = getattr(torch.version, "cuda", None)
    payload["cuda_available"] = bool(torch.cuda.is_available())
    if payload["cuda_available"]:
        for index in range(torch.cuda.device_count()):
            payload["devices"].append({
                "logical_index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
            })
except Exception as exc:
    payload["error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(payload))
"""


def _run_probe(
    runner: Callable[..., subprocess.CompletedProcess],
    command: Sequence[str],
    *,
    timeout: float,
) -> subprocess.CompletedProcess | None:
    try:
        return runner(
            tuple(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError, TimeoutError):
        return None


def probe_nvidia(
    *,
    runner: Callable[..., subprocess.CompletedProcess] = _default_runner,
    environ: Mapping[str, str] | None = None,
    python_executable: str = sys.executable,
    timeout: float = 5.0,
) -> NvidiaProbe:
    """Probe NVIDIA visibility without importing Torch or Paddle in this process."""

    environ = os.environ if environ is None else environ
    if _visible_tokens(environ) == ():
        return NvidiaProbe(reason="CUDA_VISIBLE_DEVICES hides all NVIDIA devices")

    query = _run_probe(
        runner,
        (
            "nvidia-smi",
            "--query-gpu=index,uuid,name,compute_cap",
            "--format=csv,noheader,nounits",
        ),
        timeout=timeout,
    )
    smi_devices = (
        parse_nvidia_smi_devices(query.stdout, environ=environ)
        if query is not None and query.returncode == 0
        else ()
    )
    summary = _run_probe(runner, ("nvidia-smi",), timeout=timeout)
    smi_cuda = (
        _parse_version(
            re.search(r"CUDA\s+Version\s*:\s*([0-9.]+)", summary.stdout).group(1)
        )
        if summary is not None
        and summary.returncode == 0
        and re.search(r"CUDA\s+Version\s*:\s*([0-9.]+)", summary.stdout)
        else None
    )

    torch_result = _run_probe(
        runner,
        (python_executable, "-c", _TORCH_PROBE_SCRIPT),
        timeout=timeout,
    )
    torch_payload: Mapping[str, Any] = {}
    if torch_result is not None and torch_result.returncode == 0:
        try:
            loaded = json.loads(torch_result.stdout)
            if isinstance(loaded, Mapping):
                torch_payload = loaded
        except (TypeError, json.JSONDecodeError):
            pass
    if torch_payload.get("hip"):
        return NvidiaProbe(
            reason="Torch reports a ROCm runtime; Paddle GPU selection supports NVIDIA CUDA only",
            rocm=True,
        )

    torch_devices: list[NvidiaDevice] = []
    if torch_payload.get("cuda_available"):
        raw_devices = torch_payload.get("devices")
        if isinstance(raw_devices, list):
            for item in raw_devices:
                if not isinstance(item, Mapping):
                    continue
                index = item.get("logical_index")
                raw_capability = item.get("capability")
                if (
                    type(index) is not int
                    or not isinstance(raw_capability, (list, tuple))
                    or len(raw_capability) != 2
                    or not all(type(part) is int for part in raw_capability)
                ):
                    continue
                torch_devices.append(
                    NvidiaDevice(
                        logical_index=index,
                        physical_id=str(index),
                        name=str(item.get("name") or f"NVIDIA GPU {index}"),
                        capability=(raw_capability[0], raw_capability[1]),
                    )
                )

    devices = tuple(torch_devices) if torch_devices else smi_devices
    torch_cuda = _parse_version(torch_payload.get("cuda_version"))
    versions = tuple(version for version in (smi_cuda, torch_cuda) if version is not None)
    cuda_version = min(versions) if versions else None
    if not devices:
        return NvidiaProbe(
            cuda_version=cuda_version,
            reason="NVIDIA probes found no visible CUDA device",
        )
    compatible = tuple(
        device for device in devices if device.capability >= MINIMUM_COMPUTE_CAPABILITY
    )
    if not compatible:
        capabilities = ", ".join(
            f"{device.name}={device.capability[0]}.{device.capability[1]}"
            for device in devices
        )
        return NvidiaProbe(
            devices=devices,
            cuda_version=cuda_version,
            reason=(
                "visible NVIDIA devices do not meet the minimum compute capability "
                f"7.5 ({capabilities})"
            ),
        )
    if cuda_flavor_for_version(cuda_version) is None:
        rendered = (
            f"{cuda_version[0]}.{cuda_version[1]}"
            if cuda_version is not None
            else "unknown"
        )
        return NvidiaProbe(
            devices=devices,
            cuda_version=cuda_version,
            reason=f"CUDA {rendered} does not support an official pinned Paddle wheel",
        )
    return NvidiaProbe(
        devices=devices,
        cuda_version=cuda_version,
        reason="compatible NVIDIA CUDA device detected",
    )


def _distribution_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _load_state(path: Path) -> PaddleRuntimeSelection | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping) or payload.get("schema_version") != STATE_SCHEMA_VERSION:
        return None
    status = payload.get("status")
    if status not in {"gpu", "cpu", "not-applicable", "error"}:
        return None
    return PaddleRuntimeSelection(
        schema_version=STATE_SCHEMA_VERSION,
        status=status,
        distribution=(
            str(payload["distribution"])
            if payload.get("distribution") is not None
            else None
        ),
        version=(str(payload["version"]) if payload.get("version") is not None else None),
        ocr_device=(
            str(payload["ocr_device"])
            if payload.get("ocr_device") is not None
            else None
        ),
        cuda_flavor=(
            str(payload["cuda_flavor"])
            if payload.get("cuda_flavor") is not None
            else None
        ),
        reason=str(payload.get("reason") or "recorded Paddle runtime selection"),
        gpu_failure=(
            str(payload["gpu_failure"])
            if payload.get("gpu_failure") is not None
            else None
        ),
        pending_cpu_install=bool(payload.get("pending_cpu_install", False)),
        state_path=path,
    )


def load_paddle_runtime_state(path: Path | None = None) -> PaddleRuntimeSelection | None:
    return _load_state(path or state_file_path())


def _write_state(path: Path, selection: PaddleRuntimeSelection) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="paddle-runtime-",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(selection.stored_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class _InterprocessLock:
    def __init__(self, path: Path, *, timeout: float = 600.0):
        self.path = path
        self.timeout = timeout
        self._handle: Any = None

    def __enter__(self) -> _InterprocessLock:
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+")
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    self._handle.close()
                    self._handle = None
                    raise PaddleSetupError(
                        "timed out waiting for another Paddle setup process"
                    )
                time.sleep(0.25)

    def __exit__(self, *_exc: object) -> None:
        if self._handle is None:
            return
        import fcntl

        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None


_VERIFY_SCRIPT = r"""# KEYFRAME_PADDLE_VERIFY
import json
import sys
import paddle
mode = sys.argv[1]
device = sys.argv[2]
if mode == "gpu":
    if not paddle.is_compiled_with_cuda():
        raise RuntimeError("Paddle is not compiled with CUDA")
    index = int(device.split(":", 1)[1])
    if paddle.device.cuda.device_count() <= index:
        raise RuntimeError("selected CUDA device is not visible to Paddle")
paddle.set_device(device)
value = paddle.ones([2, 2], dtype="float32")
if mode == "gpu":
    paddle.device.cuda.synchronize()
if value.numpy().sum() != 4:
    raise RuntimeError("Paddle tensor verification returned the wrong value")
print(json.dumps({"compiled_with_cuda": bool(paddle.is_compiled_with_cuda()), "device": device}))
"""


class PaddleRuntimeManager:
    """Select, install, verify, and persist exactly one Paddle distribution."""

    def __init__(
        self,
        *,
        state_path: Path | None = None,
        system: str | None = None,
        machine: str | None = None,
        runner: Callable[..., subprocess.CompletedProcess] = _default_runner,
        distribution_version: Callable[[str], str | None] = _distribution_version,
        gpu_probe: Callable[[], NvidiaProbe] | None = None,
        progress: Callable[[str], None] | None = None,
        python_executable: str = sys.executable,
    ) -> None:
        self.state_path = state_path or state_file_path()
        self.system = system or platform.system()
        self.machine = (machine or platform.machine()).lower()
        self.runner = runner
        self.distribution_version = distribution_version
        self.gpu_probe = gpu_probe or (
            lambda: probe_nvidia(runner=runner, python_executable=python_executable)
        )
        self.progress = progress or (lambda _message: None)
        self.python_executable = python_executable

    def _installed(self) -> dict[str, str]:
        return {
            name: version
            for name in ("paddlepaddle", "paddlepaddle-gpu")
            if (version := self.distribution_version(name)) is not None
        }

    def _run(self, command: Sequence[str], *, timeout: float = 600.0) -> subprocess.CompletedProcess:
        try:
            result = self.runner(
                tuple(command),
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (OSError, subprocess.SubprocessError, TimeoutError) as exc:
            raise PaddleSetupError(f"command failed: {type(exc).__name__}: {exc}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "no command output").strip()
            raise PaddleSetupError(f"command exited {result.returncode}: {detail}")
        return result

    def _download(self, distribution: str, flavor: str, directory: Path) -> Path:
        self.progress(
            f"Downloading {distribution}=={PADDLE_VERSION} ({flavor}) from Paddle stable packages..."
        )
        before = set(directory.glob("*.whl"))
        self._run(
            (
                self.python_executable,
                "-m",
                "pip",
                "--isolated",
                "download",
                "--disable-pip-version-check",
                "--only-binary=:all:",
                "--dest",
                str(directory),
                "--index-url",
                PADDLE_INDEXES[flavor],
                f"{distribution}=={PADDLE_VERSION}",
            )
        )
        candidates = tuple(path for path in directory.glob("*.whl") if path not in before)
        if len(candidates) != 1:
            normalized = distribution.replace("-", "_").lower()
            candidates = tuple(
                path
                for path in directory.glob("*.whl")
                if path.name.lower().startswith(normalized + "-")
            )
        if len(candidates) != 1:
            raise PaddleSetupError(
                f"Paddle download did not produce one unambiguous {distribution} wheel"
            )
        return candidates[0]

    def _uninstall_all(self) -> None:
        self._run(
            (
                self.python_executable,
                "-m",
                "pip",
                "uninstall",
                "-y",
                "paddlepaddle",
                "paddlepaddle-gpu",
            )
        )

    def _install_wheel(self, wheel: Path) -> None:
        self._run(
            (
                self.python_executable,
                "-m",
                "pip",
                "--isolated",
                "install",
                "--disable-pip-version-check",
                "--no-index",
                "--find-links",
                str(wheel.parent),
                str(wheel),
            )
        )

    def _require_single_distribution(self, distribution: str) -> None:
        installed = self._installed()
        expected = {distribution: PADDLE_VERSION}
        if installed != expected:
            raise PaddleSetupError(
                "Paddle transition did not leave exactly the selected distribution: "
                f"expected {expected}, found {installed}"
            )

    def _verify(self, status: str, device: str) -> None:
        self.progress(f"Verifying Paddle {status.upper()} runtime on {device}...")
        self._run(
            (
                self.python_executable,
                "-c",
                _VERIFY_SCRIPT,
                status,
                device,
            ),
            timeout=60.0,
        )

    def _persist(
        self,
        selection: PaddleRuntimeSelection,
        previous: PaddleRuntimeSelection | None,
        *,
        package_changed: bool = False,
    ) -> PaddleRuntimeSelection:
        selection = replace(selection, state_path=self.state_path)
        state_changed = previous is None or previous.stored_dict() != selection.stored_dict()
        result = replace(selection, changed=package_changed or state_changed)
        if state_changed:
            _write_state(self.state_path, selection)
        return result

    def _cpu_selection(
        self,
        reason: str,
        *,
        gpu_failure: str | None = None,
    ) -> PaddleRuntimeSelection:
        return PaddleRuntimeSelection(
            schema_version=STATE_SCHEMA_VERSION,
            status="cpu",
            distribution="paddlepaddle",
            version=PADDLE_VERSION,
            ocr_device="cpu",
            cuda_flavor=None,
            reason=reason,
            gpu_failure=gpu_failure,
            state_path=self.state_path,
        )

    def _ensure_cpu(
        self,
        reason: str,
        previous: PaddleRuntimeSelection | None,
        *,
        gpu_failure: str | None = None,
        prepared_wheel: Path | None = None,
    ) -> PaddleRuntimeSelection:
        installed = self._installed()
        package_changed = False
        cpu_valid = installed == {"paddlepaddle": PADDLE_VERSION}
        try:
            if not cpu_valid:
                with tempfile.TemporaryDirectory(prefix="keyframe-paddle-cpu-") as raw_dir:
                    directory = Path(raw_dir)
                    wheel = prepared_wheel or self._download("paddlepaddle", "cpu", directory)
                    self.progress("Installing pinned Paddle CPU fallback...")
                    package_changed = True
                    self._uninstall_all()
                    self._install_wheel(wheel)
                    self._require_single_distribution("paddlepaddle")
            self._verify("cpu", "cpu")
        except PaddleSetupError as exc:
            result = PaddleRuntimeSelection(
                schema_version=STATE_SCHEMA_VERSION,
                status="error",
                distribution=None,
                version=None,
                ocr_device=None,
                cuda_flavor=None,
                reason=f"no usable Paddle runtime: {exc}",
                gpu_failure=gpu_failure,
                state_path=self.state_path,
            )
            result = self._persist(result, previous, package_changed=package_changed)
            raise PaddleSetupError(result.reason, result) from exc
        self.progress("Paddle CPU runtime is ready.")
        return self._persist(
            self._cpu_selection(reason, gpu_failure=gpu_failure),
            previous,
            package_changed=package_changed,
        )

    def _ensure_gpu(
        self,
        probe: NvidiaProbe,
        previous: PaddleRuntimeSelection | None,
    ) -> PaddleRuntimeSelection:
        device = probe.selected_device
        flavor = cuda_flavor_for_version(probe.cuda_version)
        if device is None or flavor is None:
            return self._ensure_cpu(probe.reason, previous)
        ocr_device = f"gpu:{device.logical_index}"
        desired = PaddleRuntimeSelection(
            schema_version=STATE_SCHEMA_VERSION,
            status="gpu",
            distribution="paddlepaddle-gpu",
            version=PADDLE_VERSION,
            ocr_device=ocr_device,
            cuda_flavor=flavor,
            reason=(
                f"selected {device.name} (compute capability "
                f"{device.capability[0]}.{device.capability[1]}) with {flavor}"
            ),
            state_path=self.state_path,
        )
        installed = self._installed()
        recorded_match = (
            previous is not None
            and previous.status == "gpu"
            and previous.version == PADDLE_VERSION
            and previous.cuda_flavor == flavor
            and previous.ocr_device == ocr_device
            and not previous.gpu_failure
        )
        if installed == {"paddlepaddle-gpu": PADDLE_VERSION} and recorded_match:
            try:
                self._verify("gpu", ocr_device)
                self.progress(f"Paddle GPU runtime is ready on {ocr_device} ({flavor}).")
                return self._persist(desired, previous)
            except PaddleSetupError as exc:
                failure = f"recorded GPU runtime verification failed: {exc}"
                self.progress(f"Warning: {failure}; falling back to CPU.")
                return self._ensure_cpu(
                    "GPU verification failed; using verified CPU Paddle",
                    previous,
                    gpu_failure=failure,
                )

        package_changed = False
        try:
            with tempfile.TemporaryDirectory(prefix="keyframe-paddle-gpu-") as raw_dir:
                directory = Path(raw_dir)
                gpu_wheel = self._download("paddlepaddle-gpu", flavor, directory)
                cpu_rollback = None
                if installed == {"paddlepaddle": PADDLE_VERSION}:
                    cpu_rollback = self._download("paddlepaddle", "cpu", directory)
                try:
                    self.progress(f"Installing Paddle GPU {flavor} for {ocr_device}...")
                    self._uninstall_all()
                    self._install_wheel(gpu_wheel)
                    self._require_single_distribution("paddlepaddle-gpu")
                    package_changed = True
                    self._verify("gpu", ocr_device)
                except PaddleSetupError as exc:
                    failure = f"GPU installation or verification failed: {exc}"
                    self.progress(f"Warning: {failure}; restoring CPU Paddle.")
                    return self._ensure_cpu(
                        "GPU setup failed; using verified CPU Paddle",
                        previous,
                        gpu_failure=failure,
                        prepared_wheel=cpu_rollback,
                    )
        except PaddleSetupError as exc:
            failure = f"GPU setup failed: {exc}"
            self.progress(f"Warning: {failure}; using CPU Paddle.")
            return self._ensure_cpu(
                "GPU setup failed; using verified CPU Paddle",
                previous,
                gpu_failure=failure,
            )
        self.progress(f"Paddle GPU runtime is ready on {ocr_device} ({flavor}).")
        return self._persist(desired, previous, package_changed=package_changed)

    def ensure(self, *, force: bool = False) -> PaddleRuntimeSelection:
        if (self.system, self.machine) == ("Darwin", "arm64"):
            return PaddleRuntimeSelection(
                schema_version=STATE_SCHEMA_VERSION,
                status="not-applicable",
                distribution=None,
                version=None,
                ocr_device=None,
                cuda_flavor=None,
                changed=False,
                reason="macOS frame OCR uses Apple Vision; Paddle setup is not applicable",
            )
        if (self.system, self.machine) != ("Linux", "x86_64"):
            result = PaddleRuntimeSelection(
                schema_version=STATE_SCHEMA_VERSION,
                status="error",
                distribution=None,
                version=None,
                ocr_device=None,
                cuda_flavor=None,
                changed=False,
                reason=(
                    "automatic Paddle setup supports Linux x86-64 only "
                    f"(detected {self.system} {self.machine})"
                ),
            )
            raise PaddleSetupError(result.reason, result)

        lock_path = self.state_path.with_name("paddle-runtime.lock")
        with _InterprocessLock(lock_path):
            previous = _load_state(self.state_path)
            if previous is not None and previous.gpu_failure and not force:
                return self._ensure_cpu(
                    "a recorded GPU failure suppresses automatic retry; use keyframe setup-paddle --force",
                    previous,
                    gpu_failure=previous.gpu_failure,
                )
            if previous is not None and previous.status == "cpu" and not force:
                return self._ensure_cpu(previous.reason, previous)
            probe = self.gpu_probe()
            if probe.rocm or probe.selected_device is None or cuda_flavor_for_version(probe.cuda_version) is None:
                return self._ensure_cpu(probe.reason, previous)
            return self._ensure_gpu(probe, previous)


def ensure_paddle_runtime(
    *,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
    state_path: Path | None = None,
) -> PaddleRuntimeSelection:
    return PaddleRuntimeManager(state_path=state_path, progress=progress).ensure(force=force)


def record_gpu_runtime_failure(
    selection: PaddleRuntimeSelection | None,
    failure: str,
    *,
    state_path: Path | None = None,
) -> PaddleRuntimeSelection | None:
    """Record a successful in-process GPU-to-CPU OCR fallback for the next run."""

    if selection is None or selection.status != "gpu":
        return selection
    path = state_path or selection.state_path or state_file_path()
    fallback = PaddleRuntimeSelection(
        schema_version=STATE_SCHEMA_VERSION,
        status="cpu",
        distribution=selection.distribution,
        version=selection.version,
        ocr_device="cpu",
        cuda_flavor=selection.cuda_flavor,
        reason="PaddleOCR failed on GPU and succeeded on CPU; CPU repair is pending",
        changed=True,
        gpu_failure=failure,
        pending_cpu_install=True,
        state_path=path,
    )
    with _InterprocessLock(path.with_name("paddle-runtime.lock")):
        _write_state(path, fallback)
    return fallback
