from keyframe.cli import delegated_result


def test_delegated_result_accepts_tools_target_without_files(tmp_path):
    result = delegated_result("plan", "tools", install_root=str(tmp_path / "stage"))

    assert result["targets"] == {"tools": {"files": []}}

