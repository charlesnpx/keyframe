import os
import subprocess
from pathlib import Path

from keyframe.cli import delegated_result


def test_delegated_result_accepts_tools_target_without_files(tmp_path):
    result = delegated_result("plan", "tools", install_root=str(tmp_path / "stage"))

    assert result["targets"] == {"tools": {"files": []}}


def test_install_skill_script_rejects_unsupported_python_without_writing_targets(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for name in ("python3", "python3.11", "python3.12", "python3.13"):
        path = fake_bin / name
        path.write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
        path.chmod(0o755)

    install_root = tmp_path / "stage"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:/usr/bin:/bin",
    }
    env.pop("PYTHON", None)

    result = subprocess.run(
        ["/bin/bash", "install-skill.sh", "--install-root", str(install_root)],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "requires Python >=3.11,<3.14" in result.stderr
    assert not (install_root / ".claude").exists()
    assert not (install_root / ".codex").exists()
