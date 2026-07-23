from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


def _load_validator():
    path = Path(__file__).resolve().parents[1] / "scripts" / "validate_install.py"
    spec = importlib.util.spec_from_file_location(
        "keyframe_validate_install_under_test",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _requirements():
    marker = "sys_platform == 'linux' and platform_machine == 'x86_64'"
    return {
        "paddlepaddle": f"paddlepaddle>=3.3.1,<4; {marker}",
        "paddleocr": f"paddleocr>=3.7,<4; {marker}",
    }


def _completed_report(module, report, *, returncode=0, stderr=""):
    return subprocess.CompletedProcess(
        ["python"],
        returncode,
        stdout=(
            f"import noise\n{module._IMPORT_REPORT_PREFIX}"
            f"{json.dumps(report)}\n"
        ),
        stderr=stderr,
    )


def test_project_metadata_makes_paddle_a_gated_default_and_removes_extra():
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")

    assert (
        '"paddlepaddle>=3.3.1,<4; sys_platform == \'linux\' and '
        'platform_machine == \'x86_64\'"'
    ) in pyproject
    assert (
        '"paddleocr>=3.7,<4; sys_platform == \'linux\' and '
        'platform_machine == \'x86_64\'"'
    ) in pyproject
    assert "[project.optional-dependencies]" not in pyproject
    assert "paddle = [" not in pyproject


def test_install_matrix_builds_wheel_and_sdist_for_every_supported_python():
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "workflows"
        / "install-matrix.yml"
    ).read_text(encoding="utf-8")

    assert workflow.count('python: "3.11"') == 2
    assert workflow.count('python: "3.12"') == 2
    assert workflow.count('python: "3.13"') == 2
    assert workflow.count("platform: darwin-arm64") == 3
    assert workflow.count("platform: linux-x86_64") == 3
    assert "python -m build" in workflow
    assert "python -m pip install dist/keyframe-*.whl" in workflow


def test_linux_paddle_install_validation_accepts_exact_supported_ranges(
    monkeypatch,
):
    module = _load_validator()
    monkeypatch.setattr(module, "_paddle_requirements", _requirements)
    monkeypatch.setattr(module, "_is_linux_x86_64", lambda: True)
    versions = {
        "paddlepaddle": "3.3.1",
        "paddleocr": "3.7.0",
    }
    monkeypatch.setattr(
        module,
        "_distribution_version",
        versions.get,
    )

    assert module._validate_paddle_install() == versions


@pytest.mark.parametrize(
    ("requirements", "versions", "linux", "message"),
    [
        (
            {"paddlepaddle": _requirements()["paddlepaddle"]},
            {"paddlepaddle": "3.3.1", "paddleocr": "3.7.0"},
            True,
            "both gated",
        ),
        (
            {
                **_requirements(),
                "paddleocr": "paddleocr>=3.6,<4; "
                "sys_platform == 'linux' and platform_machine == 'x86_64'",
            },
            {"paddlepaddle": "3.3.1", "paddleocr": "3.7.0"},
            True,
            "incorrect paddleocr range",
        ),
        (
                {
                    **_requirements(),
                    "paddleocr": "paddleocr>=3.7,<4",
                },
            {"paddlepaddle": "3.3.1", "paddleocr": "3.7.0"},
            True,
            "missing the Linux",
        ),
        (
            _requirements(),
            {"paddlepaddle": None, "paddleocr": "3.7.0"},
            True,
            "missing default",
        ),
        (
            _requirements(),
            {"paddlepaddle": "3.3.0", "paddleocr": "3.7.0"},
            True,
            "outside the supported",
        ),
        (
            _requirements(),
            {"paddlepaddle": "3.3.1", "paddleocr": "4.0.0"},
            True,
            "outside the supported",
        ),
        (
            _requirements(),
            {"paddlepaddle": "3.3.1", "paddleocr": None},
            False,
            "non-Linux",
        ),
    ],
)
def test_paddle_install_validation_rejects_incomplete_or_wrong_installs(
    monkeypatch,
    requirements,
    versions,
    linux,
    message,
):
    module = _load_validator()
    monkeypatch.setattr(module, "_paddle_requirements", lambda: requirements)
    monkeypatch.setattr(module, "_is_linux_x86_64", lambda: linux)
    monkeypatch.setattr(module, "_distribution_version", versions.get)

    with pytest.raises(module.InstallValidationError, match=message):
        module._validate_paddle_install()


def test_isolated_import_validation_disables_network_and_uses_clean_child(
    monkeypatch,
):
    module = _load_validator()
    captured = {}
    report = {
        "imports": ["keyframe.frames", "paddleocr"],
        "network_attempts": [],
        "paddleocr_constructor_calls": [],
        "recognizable_checkpoints": [],
    }

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return _completed_report(module, report)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module._run_isolated_frame_import_validation() == report
    assert captured["command"][1:3] == ["-I", "-c"]
    assert captured["kwargs"]["timeout"] == 120
    assert captured["kwargs"]["env"]["HF_HUB_OFFLINE"] == "1"
    assert captured["kwargs"]["env"]["TRANSFORMERS_OFFLINE"] == "1"
    assert captured["kwargs"]["env"]["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] == (
        "True"
    )
    validation_root = Path(captured["kwargs"]["env"]["KEYFRAME_IMPORT_ROOT"])
    assert Path(captured["kwargs"]["env"]["HOME"]).parent == validation_root
    assert Path(captured["kwargs"]["env"]["TMPDIR"]).parent == validation_root
    assert Path(captured["kwargs"]["env"]["KEYFRAME_IMPORT_CACHE_ROOT"]).parent == (
        validation_root
    )
    assert "PYTHONPATH" not in captured["kwargs"]["env"]
    assert Path(captured["kwargs"]["cwd"]).name.startswith(
        "keyframe-import-validation-"
    )
    assert "socket.create_connection = blocked_create_connection" in (
        module._ISOLATED_FRAME_IMPORT
    )
    assert "paddleocr.PaddleOCR = forbidden_constructor" in (
        module._ISOLATED_FRAME_IMPORT
    )
    assert "for candidate in validation_root.rglob" in (
        module._ISOLATED_FRAME_IMPORT
    )


@pytest.mark.parametrize(
    ("report", "message"),
    [
        (
            {
                "imports": ["keyframe.frames", "paddleocr"],
                "network_attempts": ["example.com:443"],
                "paddleocr_constructor_calls": [],
                "recognizable_checkpoints": [],
            },
            "network access",
        ),
        (
            {
                "imports": ["keyframe.frames", "paddleocr"],
                "network_attempts": [],
                "paddleocr_constructor_calls": [{"args": 0}],
                "recognizable_checkpoints": [],
            },
            "constructed PaddleOCR",
        ),
        (
            {
                "imports": ["keyframe.frames", "paddleocr"],
                "network_attempts": [],
                "paddleocr_constructor_calls": [],
                "recognizable_checkpoints": ["paddle/model_state.pdparams"],
            },
            "model checkpoints",
        ),
        (
            {
                "imports": ["keyframe.frames"],
                "network_attempts": [],
                "paddleocr_constructor_calls": [],
                "recognizable_checkpoints": [],
            },
            "imports are incomplete",
        ),
    ],
)
def test_isolated_import_validation_rejects_forbidden_side_effects(
    monkeypatch,
    report,
    message,
):
    module = _load_validator()
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: _completed_report(module, report),
    )

    with pytest.raises(module.InstallValidationError, match=message):
        module._run_isolated_frame_import_validation()


@pytest.mark.parametrize(
    ("completed", "message"),
    [
        (
            subprocess.CompletedProcess(
                ["python"],
                1,
                stdout="",
                stderr="import exploded",
            ),
            "imports failed",
        ),
        (
            subprocess.CompletedProcess(
                ["python"],
                0,
                stdout="no report",
                stderr="",
            ),
            "did not emit",
        ),
        (
            subprocess.CompletedProcess(
                ["python"],
                0,
                stdout="KEYFRAME_IMPORT_REPORT={",
                stderr="",
            ),
            "malformed",
        ),
    ],
)
def test_isolated_import_validation_rejects_failed_or_malformed_child(
    monkeypatch,
    completed,
    message,
):
    module = _load_validator()
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: completed,
    )

    with pytest.raises(module.InstallValidationError, match=message):
        module._run_isolated_frame_import_validation()


def test_isolated_import_timeout_is_controlled(monkeypatch):
    module = _load_validator()
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired("python", 120)
        ),
    )

    with pytest.raises(module.InstallValidationError, match="could not complete"):
        module._run_isolated_frame_import_validation()
