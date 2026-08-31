from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from keyframe import cli
from keyframe.frames import ManagedPaddleOCR, PaddleOCRRuntimeError
from keyframe.paddle_runtime import (
    PADDLE_VERSION,
    NvidiaDevice,
    NvidiaProbe,
    PaddleRuntimeManager,
    PaddleRuntimeSelection,
    PaddleSetupError,
    cuda_flavor_for_version,
    load_paddle_runtime_state,
    parse_nvidia_smi_devices,
    probe_nvidia,
    state_file_path,
)
from keyframe.stage_scheduler import GIB, RuntimeResources, StageScheduler, diarization_demand, frame_demand


def _gpu_probe(*, cuda=(13, 0), capability=(8, 9), logical_index=0):
    return NvidiaProbe(
        devices=(
            NvidiaDevice(
                logical_index=logical_index,
                physical_id=str(logical_index),
                name="NVIDIA Test GPU",
                capability=capability,
            ),
        ),
        cuda_version=cuda,
        reason="compatible NVIDIA CUDA device detected",
    )


class FakePaddleEnvironment:
    def __init__(
        self,
        installed=None,
        *,
        fail_download=None,
        fail_install=None,
        fail_verify=None,
    ):
        self.installed = dict(installed or {})
        self.fail_download = fail_download
        self.fail_install = fail_install
        self.fail_verify = fail_verify
        self.commands = []

    def distribution_version(self, name):
        return self.installed.get(name)

    def __call__(self, command, **_kwargs):
        command = tuple(command)
        self.commands.append(command)
        if "pip" in command and "download" in command:
            spec = command[-1]
            distribution = spec.split("==", 1)[0]
            if distribution == self.fail_download:
                return subprocess.CompletedProcess(command, 1, "", "download failed")
            destination = Path(command[command.index("--dest") + 1])
            wheel_name = distribution.replace("-", "_") + "-3.3.1-py3-none-any.whl"
            (destination / wheel_name).write_bytes(b"wheel")
            return subprocess.CompletedProcess(command, 0, "downloaded", "")
        if "pip" in command and "uninstall" in command:
            self.installed.clear()
            return subprocess.CompletedProcess(command, 0, "uninstalled", "")
        if "pip" in command and "install" in command:
            wheel = Path(command[-1])
            distribution = (
                "paddlepaddle-gpu"
                if wheel.name.startswith("paddlepaddle_gpu-")
                else "paddlepaddle"
            )
            if distribution == self.fail_install:
                return subprocess.CompletedProcess(command, 1, "", "install failed")
            self.installed[distribution] = PADDLE_VERSION
            return subprocess.CompletedProcess(command, 0, "installed", "")
        if len(command) > 2 and "KEYFRAME_PADDLE_VERIFY" in command[2]:
            status = command[-2]
            if status == self.fail_verify:
                return subprocess.CompletedProcess(command, 1, "", f"{status} verify failed")
            return subprocess.CompletedProcess(command, 0, "verified", "")
        raise AssertionError(f"unexpected command: {command}")


def _manager(tmp_path, environment, *, probe=None, system="Linux", machine="x86_64"):
    return PaddleRuntimeManager(
        state_path=tmp_path / "state" / "paddle-runtime.json",
        system=system,
        machine=machine,
        runner=environment,
        distribution_version=environment.distribution_version,
        gpu_probe=lambda: probe or NvidiaProbe(reason="no visible NVIDIA GPU"),
        python_executable="python-test",
    )


def test_state_path_uses_xdg_state_home():
    assert state_file_path({"XDG_STATE_HOME": "/state-root"}) == Path(
        "/state-root/keyframe/paddle-runtime.json"
    )


@pytest.mark.parametrize(
    ("cuda", "flavor"),
    [
        ("13.1", "cu130"),
        ("13.0", "cu130"),
        ("12.9", "cu129"),
        ("12.8", "cu126"),
        ("12.6", "cu126"),
        ("12.5", "cu118"),
        ("11.8", "cu118"),
        ("11.7", None),
        ("malformed", None),
    ],
)
def test_cuda_version_maps_to_highest_supported_official_wheel(cuda, flavor):
    assert cuda_flavor_for_version(cuda) == flavor


def test_nvidia_visibility_mask_remaps_logical_devices_and_filters_capability():
    output = "\n".join(
        (
            "0, GPU-aaa, Old GPU, 7.0",
            "1, GPU-bbb, New GPU, 8.9",
            "2, GPU-ccc, Mid GPU, 7.5",
            "malformed",
        )
    )

    devices = parse_nvidia_smi_devices(
        output,
        environ={"CUDA_VISIBLE_DEVICES": "2,GPU-bbb"},
    )

    assert [(item.logical_index, item.physical_id, item.capability) for item in devices] == [
        (0, "2", (7, 5)),
        (1, "1", (8, 9)),
    ]
    assert NvidiaProbe(devices=devices, cuda_version=(12, 9)).selected_device.logical_index == 1


def test_empty_visibility_mask_skips_all_probes():
    def runner(*_args, **_kwargs):
        pytest.fail("CUDA probes must not run when all devices are masked")

    probe = probe_nvidia(runner=runner, environ={"CUDA_VISIBLE_DEVICES": "-1"})

    assert probe.selected_device is None
    assert "hides all" in probe.reason


def test_probe_rejects_rocm_even_when_nvidia_smi_succeeds():
    def runner(command, **_kwargs):
        if "--query-gpu" in command:
            return subprocess.CompletedProcess(command, 0, "0, GPU-a, GPU, 8.9\n", "")
        if command == ("nvidia-smi",):
            return subprocess.CompletedProcess(command, 0, "CUDA Version: 13.0", "")
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps({"hip": "6.2", "cuda_available": False}),
            "",
        )

    probe = probe_nvidia(runner=runner, environ={})

    assert probe.rocm
    assert probe.selected_device is None
    assert "ROCm" in probe.reason


def test_malformed_and_timed_out_probes_return_a_cpu_reason():
    def runner(command, **_kwargs):
        if command == ("nvidia-smi",):
            raise subprocess.TimeoutExpired(command, 5)
        return subprocess.CompletedProcess(command, 0, "not json", "")

    probe = probe_nvidia(runner=runner, environ={})

    assert probe.selected_device is None
    assert "no visible" in probe.reason


def test_macos_setup_is_a_noop_without_state_or_commands(tmp_path):
    environment = FakePaddleEnvironment()
    result = _manager(
        tmp_path,
        environment,
        system="Darwin",
        machine="arm64",
    ).ensure()

    assert result.status == "not-applicable"
    assert result.changed is False
    assert environment.commands == []
    assert not (tmp_path / "state" / "paddle-runtime.json").exists()


def test_unsupported_platform_returns_a_nonusable_error(tmp_path):
    environment = FakePaddleEnvironment()
    with pytest.raises(PaddleSetupError) as raised:
        _manager(tmp_path, environment, system="Linux", machine="aarch64").ensure()

    assert raised.value.result.status == "error"
    assert environment.commands == []


def test_fresh_gpu_install_downloads_before_uninstall_and_persists_selection(tmp_path):
    environment = FakePaddleEnvironment()
    result = _manager(tmp_path, environment, probe=_gpu_probe()).ensure()

    assert result.status == "gpu"
    assert result.distribution == "paddlepaddle-gpu"
    assert result.cuda_flavor == "cu130"
    assert result.ocr_device == "gpu:0"
    assert result.changed
    assert environment.installed == {"paddlepaddle-gpu": PADDLE_VERSION}
    verbs = [
        "download" if "download" in command else "uninstall" if "uninstall" in command else "other"
        for command in environment.commands
    ]
    assert verbs.index("download") < verbs.index("uninstall")
    assert load_paddle_runtime_state(result.state_path).status == "gpu"


def test_existing_cpu_receives_one_migration_with_a_predownloaded_rollback(tmp_path):
    environment = FakePaddleEnvironment({"paddlepaddle": PADDLE_VERSION})
    result = _manager(tmp_path, environment, probe=_gpu_probe(cuda=(12, 9))).ensure()

    assert result.status == "gpu"
    download_specs = [command[-1] for command in environment.commands if "download" in command]
    assert download_specs == [
        "paddlepaddle-gpu==3.3.1",
        "paddlepaddle==3.3.1",
    ]


def test_recorded_valid_gpu_is_verified_without_package_changes(tmp_path):
    environment = FakePaddleEnvironment({"paddlepaddle-gpu": PADDLE_VERSION})
    path = tmp_path / "state" / "paddle-runtime.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "gpu",
                "distribution": "paddlepaddle-gpu",
                "version": "3.3.1",
                "ocr_device": "gpu:0",
                "cuda_flavor": "cu130",
                "reason": "selected NVIDIA Test GPU (compute capability 8.9) with cu130",
                "gpu_failure": None,
                "pending_cpu_install": False,
            }
        )
    )

    result = _manager(tmp_path, environment, probe=_gpu_probe()).ensure()

    assert result.status == "gpu"
    assert not result.changed
    assert not any("pip" in command for command in environment.commands)


def test_conflicting_distributions_are_replaced_by_exactly_one_gpu_engine(tmp_path):
    environment = FakePaddleEnvironment(
        {"paddlepaddle": PADDLE_VERSION, "paddlepaddle-gpu": PADDLE_VERSION}
    )

    result = _manager(tmp_path, environment, probe=_gpu_probe()).ensure()

    assert result.status == "gpu"
    assert environment.installed == {"paddlepaddle-gpu": PADDLE_VERSION}


@pytest.mark.parametrize("failure", ["download", "install", "verify"])
def test_gpu_setup_failures_restore_and_verify_cpu(tmp_path, failure):
    environment = FakePaddleEnvironment(
        {"paddlepaddle": PADDLE_VERSION},
        fail_download="paddlepaddle-gpu" if failure == "download" else None,
        fail_install="paddlepaddle-gpu" if failure == "install" else None,
        fail_verify="gpu" if failure == "verify" else None,
    )

    result = _manager(tmp_path, environment, probe=_gpu_probe()).ensure()

    assert result.status == "cpu"
    assert result.gpu_failure
    assert environment.installed == {"paddlepaddle": PADDLE_VERSION}
    if failure in {"install", "verify"}:
        assert [
            command[-1]
            for command in environment.commands
            if "download" in command and command[-1].startswith("paddlepaddle==")
        ] == ["paddlepaddle==3.3.1"]
    assert any(
        len(command) > 2
        and "KEYFRAME_PADDLE_VERIFY" in command[2]
        and command[-2] == "cpu"
        for command in environment.commands
    )


def test_no_installable_cpu_runtime_returns_a_persisted_error(tmp_path):
    environment = FakePaddleEnvironment(fail_download="paddlepaddle")

    with pytest.raises(PaddleSetupError) as raised:
        _manager(tmp_path, environment).ensure()

    assert raised.value.result.status == "error"
    assert "no usable Paddle runtime" in raised.value.result.reason
    recorded = load_paddle_runtime_state(tmp_path / "state" / "paddle-runtime.json")
    assert recorded.status == "error"


def test_cpu_verification_failure_returns_a_nonusable_error(tmp_path):
    environment = FakePaddleEnvironment(
        {"paddlepaddle": PADDLE_VERSION},
        fail_verify="cpu",
    )

    with pytest.raises(PaddleSetupError) as raised:
        _manager(tmp_path, environment).ensure()

    assert raised.value.result.status == "error"
    assert "cpu verify failed" in raised.value.result.reason


def test_failure_record_suppresses_probe_until_force(tmp_path):
    environment = FakePaddleEnvironment({"paddlepaddle": PADDLE_VERSION})
    first = _manager(tmp_path, environment, probe=_gpu_probe())
    first_result = first.ensure()
    assert first_result.status == "gpu"

    # Simulate the persisted result of an in-run GPU-to-CPU OCR fallback.
    environment.installed = {"paddlepaddle": PADDLE_VERSION}
    path = first_result.state_path
    payload = json.loads(path.read_text())
    payload.update(
        status="cpu",
        distribution="paddlepaddle",
        ocr_device="cpu",
        cuda_flavor=None,
        gpu_failure="prediction failed",
    )
    path.write_text(json.dumps(payload))
    calls = []
    suppressed = PaddleRuntimeManager(
        state_path=path,
        system="Linux",
        machine="x86_64",
        runner=environment,
        distribution_version=environment.distribution_version,
        gpu_probe=lambda: calls.append("probe") or _gpu_probe(),
        python_executable="python-test",
    ).ensure()

    assert suppressed.status == "cpu"
    assert calls == []

    forced = PaddleRuntimeManager(
        state_path=path,
        system="Linux",
        machine="x86_64",
        runner=environment,
        distribution_version=environment.distribution_version,
        gpu_probe=lambda: calls.append("probe") or _gpu_probe(),
        python_executable="python-test",
    ).ensure(force=True)
    assert forced.status == "gpu"
    assert calls == ["probe"]


def _runtime_selection(tmp_path):
    return PaddleRuntimeSelection(
        schema_version=1,
        status="gpu",
        distribution="paddlepaddle-gpu",
        version="3.3.1",
        ocr_device="gpu:0",
        cuda_flavor="cu130",
        reason="test",
        state_path=tmp_path / "paddle-runtime.json",
    )


class _Engine:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)

    def predict(self, _image):
        outcome = next(self.outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def test_first_gpu_prediction_retries_once_on_explicit_cpu_and_records_failure(tmp_path):
    devices = []

    def constructor(device):
        devices.append(device)
        return (
            _Engine([RuntimeError("gpu kernel")])
            if device == "gpu:0"
            else _Engine([[{"rec_texts": [], "rec_scores": []}]])
        )

    managed = ManagedPaddleOCR(
        "gpu:0",
        _runtime_selection(tmp_path),
        constructor=constructor,
    )

    assert managed.predict(object()) == [{"rec_texts": [], "rec_scores": []}]
    assert managed.device == "cpu"
    assert devices == ["gpu:0", "cpu"]
    recorded = load_paddle_runtime_state(tmp_path / "paddle-runtime.json")
    assert recorded.status == "cpu"
    assert recorded.pending_cpu_install
    assert "first prediction" in recorded.gpu_failure


def test_gpu_initialization_falls_back_to_explicit_cpu(tmp_path):
    devices = []

    def constructor(device):
        devices.append(device)
        if device == "gpu:0":
            raise RuntimeError("gpu init")
        return _Engine([[]])

    managed = ManagedPaddleOCR(
        "gpu:0",
        _runtime_selection(tmp_path),
        constructor=constructor,
    )

    assert managed.device == "cpu"
    assert devices == ["gpu:0", "cpu"]


def test_combined_gpu_and_cpu_prediction_failure_preserves_both_errors(tmp_path):
    def constructor(device):
        return (
            _Engine([RuntimeError("gpu predict")])
            if device == "gpu:0"
            else _Engine([ValueError("cpu predict")])
        )

    managed = ManagedPaddleOCR(
        "gpu:0",
        _runtime_selection(tmp_path),
        constructor=constructor,
    )

    with pytest.raises(PaddleOCRRuntimeError, match="gpu predict.*cpu predict"):
        managed.predict(object())
    assert not (tmp_path / "paddle-runtime.json").exists()


def test_gpu_backed_ocr_claims_cuda_when_torch_frames_are_cpu():
    scheduler = StageScheduler(
        resource_probe=lambda: RuntimeResources(8, 64 * GIB),
    )
    frames = frame_demand("cpu", ocr_device="gpu:0")

    assert frames.owned_accelerators == {"cuda:0"}
    decision = scheduler.decide((frames, diarization_demand("cuda")))
    assert not decision.parallel
    assert "cuda:0" in decision.reason


def test_setup_paddle_json_has_stable_public_schema(monkeypatch, capsys):
    result = PaddleRuntimeSelection(
        schema_version=1,
        status="cpu",
        distribution="paddlepaddle",
        version="3.3.1",
        ocr_device="cpu",
        cuda_flavor=None,
        reason="no GPU",
        changed=True,
    )
    monkeypatch.setattr(
        "keyframe.paddle_runtime.PaddleRuntimeManager.ensure",
        lambda self, force=False: result,
    )

    assert cli.cmd_setup_paddle(SimpleNamespace(json=True, force=False)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {
        "schema_version",
        "status",
        "distribution",
        "version",
        "ocr_device",
        "cuda_flavor",
        "changed",
        "reason",
    }
    assert payload["status"] == "cpu"


def test_setup_paddle_returns_nonzero_only_for_nonusable_runtime(monkeypatch, capsys):
    result = PaddleRuntimeSelection(
        schema_version=1,
        status="error",
        distribution=None,
        version=None,
        ocr_device=None,
        cuda_flavor=None,
        reason="offline",
    )

    def fail(self, force=False):
        raise PaddleSetupError("offline", result)

    monkeypatch.setattr("keyframe.paddle_runtime.PaddleRuntimeManager.ensure", fail)

    assert cli.cmd_setup_paddle(SimpleNamespace(json=True, force=False)) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "error"
