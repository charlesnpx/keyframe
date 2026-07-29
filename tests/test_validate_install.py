import pytest

from scripts import validate_install


def _stub_clean_install_metadata(monkeypatch, requirements):
    monkeypatch.setattr(
        validate_install.importlib_metadata,
        "requires",
        lambda name: requirements if name == "keyframe" else [],
    )
    monkeypatch.setattr(
        validate_install,
        "_distribution_version",
        lambda name: "0.6.3" if name == "keyframe" else None,
    )
    monkeypatch.setattr(validate_install, "_is_supported_mlx_runtime", lambda: False)
    monkeypatch.setattr(validate_install.importlib, "import_module", lambda name: object())
    monkeypatch.setattr(validate_install.sys, "version_info", (3, 13, 0))


def test_validate_install_accepts_duplicate_paddle_requirement_names(monkeypatch):
    _stub_clean_install_metadata(
        monkeypatch,
        [
            "mlx==0.32.0; sys_platform == 'darwin' and platform_machine == 'arm64' and platform_release >= '23.0.0'",
            "mlx-whisper==0.4.3; sys_platform == 'darwin' and platform_machine == 'arm64' and platform_release >= '23.0.0'",
            "paddlepaddle; sys_platform == 'linux' and platform_machine == 'x86_64'",
            "paddlepaddle; extra == 'paddle'",
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
            "paddlepaddle; extra == 'paddle'",
            "paddleocr; extra == 'paddle'",
        ],
    )

    with pytest.raises(validate_install.InstallValidationError, match="Linux x86_64"):
        validate_install.validate_install()
