import tomllib
from pathlib import Path

import pytest

from scripts import validate_install


ROOT = Path(__file__).resolve().parents[1]


def test_project_metadata_defers_engine_selection_but_retains_paddleocr():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    dependencies = project["dependencies"]
    paddle_extra = project["optional-dependencies"]["paddle"]

    assert not any(
        requirement.split(";", 1)[0].strip().startswith("paddlepaddle")
        for requirement in dependencies
    )
    assert any(
        requirement.startswith("paddleocr;")
        and "sys_platform == 'linux'" in requirement
        and "platform_machine == 'x86_64'" in requirement
        for requirement in dependencies
    )
    assert any(
        requirement.startswith("paddlepaddle==3.3.1;")
        for requirement in paddle_extra
    )


def _stub_clean_install_metadata(monkeypatch, requirements):
    monkeypatch.setattr(
        validate_install.importlib_metadata,
        "requires",
        lambda name: requirements if name == "keyframe" else [],
    )
    monkeypatch.setattr(
        validate_install,
        "_distribution_version",
        lambda name: "0.6.5" if name == "keyframe" else None,
    )
    monkeypatch.setattr(validate_install, "_is_supported_mlx_runtime", lambda: False)
    monkeypatch.setattr(validate_install.importlib, "import_module", lambda name: object())
    monkeypatch.setattr(validate_install.sys, "version_info", (3, 13, 0))


def test_validate_install_accepts_gated_paddleocr_and_pinned_cpu_extra(monkeypatch):
    _stub_clean_install_metadata(
        monkeypatch,
        [
            "mlx==0.32.0; sys_platform == 'darwin' and platform_machine == 'arm64' and platform_release >= '23.0.0'",
            "mlx-whisper==0.4.3; sys_platform == 'darwin' and platform_machine == 'arm64' and platform_release >= '23.0.0'",
            "paddlepaddle==3.3.1; sys_platform == 'linux' and platform_machine == 'x86_64' and extra == 'paddle'",
            "paddleocr; extra == 'paddle'",
            "paddleocr; sys_platform == 'linux' and platform_machine == 'x86_64'",
        ],
    )

    report = validate_install.validate_install()

    assert report["passed"] is True


def test_validate_install_rejects_paddle_extras_without_default_linux_marker(monkeypatch):
    _stub_clean_install_metadata(
        monkeypatch,
        [
            "mlx==0.32.0; sys_platform == 'darwin' and platform_machine == 'arm64' and platform_release >= '23.0.0'",
            "mlx-whisper==0.4.3; sys_platform == 'darwin' and platform_machine == 'arm64' and platform_release >= '23.0.0'",
            "paddlepaddle==3.3.1; sys_platform == 'linux' and platform_machine == 'x86_64' and extra == 'paddle'",
            "paddleocr; extra == 'paddle'",
        ],
    )

    with pytest.raises(validate_install.InstallValidationError, match="Linux x86_64"):
        validate_install.validate_install()
