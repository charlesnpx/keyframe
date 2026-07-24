from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from keyframe import release_evidence
from tests.release_evidence_helpers import (
    FIXTURE_ROOT,
    write_frame_evidence_bundle,
)


def _rewrite_report(bundle: Path, report: dict) -> None:
    release_evidence._atomic_write_json(bundle / "evidence.json", report)


def test_public_fixture_metadata_and_media_identity_are_self_consistent():
    contract = release_evidence.load_fixture_contract(
        FIXTURE_ROOT / "metadata.json"
    )

    assert contract.metadata["identifier"] == "keyframe.release-frame-fixture"
    assert contract.metadata["schema_version"] == 1
    assert contract.metadata["license"]["spdx"] == "MIT"
    assert contract.metadata["license"]["redistribution_permitted"] is True
    assert contract.metadata["media"] == {
        "container": "mp4",
        "codec": "h264",
        "pixel_format": "yuv420p",
        "width": 1280,
        "height": 720,
        "duration_seconds": 36.0,
        "frame_rate": "30/1",
        "frame_count": 1080,
        "video_streams": 1,
        "audio_streams": 0,
    }
    assert [
        (target["id"], target["time_seconds"], target["tolerance_seconds"])
        for target in contract.metadata["targets"]
    ] == [
        ("source-fields", 3.0, 2.25),
        ("priority-sections", 9.0, 2.25),
        ("consequence-comments", 15.0, 2.25),
        ("page-transition", 21.0, 2.25),
        ("signed-blank", 27.0, 2.25),
        ("signed-populated", 33.0, 2.25),
    ]


def test_token_normalization_accepts_bounded_cross_backend_aliases():
    aliases = {"imc": ("lmc",), "approved": ("approvecl",)}
    apple_tokens = release_evidence.normalize_tokens(
        "Signed on behalf of the IMC by — APPROVED"
    )
    paddle_tokens = release_evidence.normalize_tokens(
        "Signed-on-behalf-of-the-lMC-by / Approvecl"
    )

    for tokens in (apple_tokens, paddle_tokens):
        assert release_evidence.token_matches(tokens, "imc", aliases)
        assert release_evidence.token_matches(tokens, "approved", aliases)
    assert not release_evidence.token_matches(
        release_evidence.normalize_tokens("Approval pending"),
        "approved",
        aliases,
    )


def test_target_timing_accepts_validated_same_state_lineage():
    contract = release_evidence.load_fixture_contract(
        FIXTURE_ROOT / "metadata.json"
    )
    rows = []
    captions = {}
    for target, text in zip(
        contract.metadata["targets"],
        (
            "SOURCE FIELDS Linked Risks Description Dependencies",
            (
                "PRIORITY FORM Risk Justification Other Considerations "
                "Override"
            ),
            (
                "CONSEQUENCE FAILURE Dependencies Comments Summary Changes"
            ),
            "PAGE 2 Current State Description Prepared",
            (
                "Signed on Behalf of the IMC By Approved Date Status Draft"
            ),
            (
                "Signed on Behalf of the IMC By Alyssa Leon 29APR2026 "
                "Approved"
            ),
        ),
        strict=True,
    ):
        timestamp = float(target["time_seconds"])
        merged = [timestamp]
        if target["id"] == "signed-blank":
            timestamp = 29.5
            merged = [27.5, 29.5]
        elif target["id"] == "signed-populated":
            timestamp = 30.0
            merged = [30.0, 32.5]
        name = f"{target['id']}.png"
        rows.append(
            {
                "filename": name,
                "timestamp": timestamp,
                "merged_timestamps": merged,
            }
        )
        captions[name] = {
            "file": name,
            "timestamp": timestamp,
            "ocr_text": text,
        }

    results, failures = release_evidence._evaluate_targets(
        contract.metadata,
        rows,
        captions,
    )

    assert failures == []
    assert results[-2]["selected_timestamp"] == 29.5
    assert results[-2]["matched_timestamp"] == 27.5
    assert results[-2]["matched_via"] == "merged_timestamps"
    assert results[-1]["selected_timestamp"] == 30.0
    assert results[-1]["matched_timestamp"] == 32.5
    assert results[-1]["matched_via"] == "merged_timestamps"


def test_replay_recomputes_and_ignores_stored_pass_fields(tmp_path):
    bundle = tmp_path / "bundle"
    report = write_frame_evidence_bundle(
        bundle,
        system="Darwin",
        machine="arm64",
    )
    report["validation"] = {
        "passed": False,
        "status": "manually-overridden",
        "failures": ["stored values are not authoritative"],
    }
    for target in report["targets"]:
        target["passed"] = False
    report["budgets"]["passed"] = False
    report["redundancy"]["passed"] = False
    _rewrite_report(bundle, report)

    replay = release_evidence.replay_bundle(bundle)

    assert replay.validation() == {"passed": True, "failures": []}


@pytest.mark.parametrize(
    "artifact_name",
    [
        "recording",
        "metadata",
        "manifest",
        "captions",
        "png",
        "png-aggregate",
        "trace",
        "source-artifact",
    ],
)
def test_replay_rejects_tampered_bundle_artifacts(tmp_path, artifact_name):
    bundle = tmp_path / "bundle"
    report = write_frame_evidence_bundle(
        bundle,
        system="Linux",
        machine="x86_64",
        source_kind="artifact",
        source_value="exact-wheel-bytes",
    )

    if artifact_name == "recording":
        path = release_evidence.resolve_bundle_file(
            bundle,
            report["fixture"]["recording"]["path"],
        )
        path.write_bytes(path.read_bytes() + b"tamper")
    elif artifact_name == "metadata":
        path = release_evidence.resolve_bundle_file(
            bundle,
            report["fixture"]["metadata"]["path"],
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["creator"] = "tampered"
        path.write_text(json.dumps(payload), encoding="utf-8")
    elif artifact_name in {"manifest", "captions"}:
        path = release_evidence.resolve_bundle_file(
            bundle,
            report["artifacts"][artifact_name]["path"],
        )
        path.write_bytes(path.read_bytes() + b"\n ")
    elif artifact_name == "png":
        path = release_evidence.resolve_bundle_file(
            bundle,
            report["artifacts"]["pngs"][0]["path"],
        )
        path.write_bytes(path.read_bytes() + b"tamper")
    elif artifact_name == "png-aggregate":
        report["artifacts"]["canonical_png_aggregate_sha256"] = "0" * 64
        _rewrite_report(bundle, report)
    elif artifact_name == "trace":
        path = release_evidence.resolve_bundle_file(
            bundle,
            report["artifacts"]["traces"][0]["path"],
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["records"] = []
        path.write_text(json.dumps(payload), encoding="utf-8")
    elif artifact_name == "source-artifact":
        path = release_evidence.resolve_bundle_file(
            bundle,
            report["source_identity"]["path"],
        )
        path.write_bytes(path.read_bytes() + b"tamper")
    else:
        raise AssertionError(artifact_name)

    replay = release_evidence.replay_bundle(bundle)

    assert not replay.passed
    assert replay.failures


def test_replay_rejects_path_traversal(tmp_path):
    bundle = tmp_path / "bundle"
    report = write_frame_evidence_bundle(
        bundle,
        system="Darwin",
        machine="arm64",
    )
    report["artifacts"]["manifest"]["path"] = "../outside.json"
    _rewrite_report(bundle, report)

    replay = release_evidence.replay_bundle(bundle)

    assert not replay.passed
    assert any("beneath its bundle" in failure for failure in replay.failures)


def test_replay_rejects_symlinked_artifacts(tmp_path):
    bundle = tmp_path / "bundle"
    report = write_frame_evidence_bundle(
        bundle,
        system="Darwin",
        machine="arm64",
    )
    first = release_evidence.resolve_bundle_file(
        bundle,
        report["artifacts"]["pngs"][0]["path"],
    )
    second = release_evidence.resolve_bundle_file(
        bundle,
        report["artifacts"]["pngs"][1]["path"],
    )
    first.unlink()
    first.symlink_to(second)

    replay = release_evidence.replay_bundle(bundle)

    assert not replay.passed
    assert any("symlink" in failure for failure in replay.failures)


def test_replay_rejects_platform_and_ocr_mismatch(tmp_path):
    bundle = tmp_path / "bundle"
    report = write_frame_evidence_bundle(
        bundle,
        system="Darwin",
        machine="arm64",
    )
    report["platform"]["machine"] = "x86_64"
    _rewrite_report(bundle, report)

    replay = release_evidence.replay_bundle(bundle)

    assert not replay.passed
    assert "evidence platform must be Darwin ARM64 or Linux x86-64" in replay.failures


def test_git_identity_comes_from_loaded_package_tree_not_cwd(
    monkeypatch,
    tmp_path,
):
    repository = tmp_path / "source-tree"
    package = repository / "keyframe"
    package.mkdir(parents=True)
    (repository / "pyproject.toml").write_text(
        '[project]\nname = "keyframe"\nversion = "0.6.3"\n',
        encoding="utf-8",
    )
    unrelated_cwd = tmp_path / "unrelated-repository"
    unrelated_cwd.mkdir()
    calls = []

    def fake_git(candidate, *arguments):
        calls.append((Path(candidate), arguments))
        if arguments == ("rev-parse", "--show-toplevel"):
            return str(repository)
        if arguments == (
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ):
            return ""
        if arguments == ("rev-parse", "HEAD"):
            return "a" * 40
        raise AssertionError(arguments)

    monkeypatch.setattr(release_evidence, "_git", fake_git)
    identity, locations = release_evidence._source_identity_from_probe(
        {
            "keyframe_package_root": str(package),
            "interpreter": "/python",
            "environment_root": "/environment",
            "base_prefix": "/base",
            "keyframe_module_path": str(package / "__init__.py"),
        },
        source_tree=repository,
        working_directory=unrelated_cwd,
    )

    assert identity == {
        "kind": "git",
        "commit_sha": "a" * 40,
        "version": "0.6.3",
    }
    assert locations["working_directory"] == str(unrelated_cwd.resolve())
    assert calls[0][0] == package.resolve()


def test_dirty_git_source_cannot_produce_release_evidence(
    monkeypatch,
    tmp_path,
):
    repository = tmp_path / "source-tree"
    package = repository / "keyframe"
    package.mkdir(parents=True)
    (repository / "pyproject.toml").write_text(
        '[project]\nname = "keyframe"\nversion = "0.6.3"\n',
        encoding="utf-8",
    )

    def fake_git(_candidate, *arguments):
        if arguments == ("rev-parse", "--show-toplevel"):
            return str(repository)
        if arguments == (
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ):
            return " M keyframe/cli.py"
        raise AssertionError(arguments)

    monkeypatch.setattr(release_evidence, "_git", fake_git)

    with pytest.raises(
        release_evidence.ReleaseEvidenceError,
        match="dirty Git source",
    ):
        release_evidence._source_identity_from_probe(
            {"keyframe_package_root": str(package)},
            source_tree=repository,
            working_directory=repository,
        )


def _artifact_probe(
    environment: Path,
    package_root: Path,
) -> dict:
    return {
        "system": "Darwin",
        "machine": "arm64",
        "environment_root": str(environment),
        "base_prefix": str(environment.parent / "base"),
        "interpreter": str(environment / "bin" / "python"),
        "keyframe_module_path": str(package_root / "__init__.py"),
        "keyframe_package_root": str(package_root),
        "distribution_root": str(package_root.parent),
        "distribution_version": "0.6.3",
        "packages": {"keyframe": "0.6.3"},
        "direct_url": {
            "archive_info": {"hashes": {"sha256": "a" * 64}},
            "url": "file:///tmp/keyframe-0.6.3-py3-none-any.whl",
        },
    }


def test_artifact_runtime_rejects_checkout_shadowing(tmp_path):
    environment = tmp_path / "environment"
    package = environment / "lib" / "site-packages" / "keyframe"
    (environment / "bin").mkdir(parents=True)
    package.mkdir(parents=True)
    (environment / "pyvenv.cfg").write_text("home = /base\n", encoding="utf-8")
    (environment / "bin" / "python").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")

    with pytest.raises(
        release_evidence.ReleaseEvidenceError,
        match="shadowed",
    ):
        release_evidence._validate_artifact_runtime(
            _artifact_probe(environment, package),
            artifact_name="keyframe",
            artifact_version="0.6.3",
            artifact_filename="keyframe-0.6.3-py3-none-any.whl",
            artifact_sha256="a" * 64,
            repository_root=environment,
            working_directory=tmp_path,
        )


def test_artifact_runtime_rejects_package_outside_environment(tmp_path):
    environment = tmp_path / "environment"
    package = tmp_path / "checkout" / "keyframe"
    (environment / "bin").mkdir(parents=True)
    package.mkdir(parents=True)
    (environment / "pyvenv.cfg").write_text("home = /base\n", encoding="utf-8")
    (environment / "bin" / "python").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")

    with pytest.raises(
        release_evidence.ReleaseEvidenceError,
        match="outside the clean environment",
    ):
        release_evidence._validate_artifact_runtime(
            _artifact_probe(environment, package),
            artifact_name="keyframe",
            artifact_version="0.6.3",
            artifact_filename="keyframe-0.6.3-py3-none-any.whl",
            artifact_sha256="a" * 64,
            repository_root=tmp_path / "checkout",
            working_directory=tmp_path,
        )


def test_default_artifact_route_is_isolated_and_sanitized(
    monkeypatch,
    tmp_path,
):
    captured = {}
    runtime = tmp_path / "environment" / "bin" / "python"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("", encoding="utf-8")
    fixture = tmp_path / "fixture.mp4"
    fixture.write_bytes(b"fixture")
    targets = tmp_path / "targets.json"
    targets.write_text("{}", encoding="utf-8")
    working = tmp_path / "external"
    working.mkdir()
    monkeypatch.setenv("PYTHONPATH", "/repository")

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(release_evidence.subprocess, "run", fake_run)

    release_evidence._run_default_cli(
        runtime_python=runtime,
        fixture=fixture,
        output_dir=tmp_path / "output",
        qa_targets=targets,
        working_directory=working,
        isolated=True,
    )

    assert captured["command"][:4] == [
        str(runtime),
        "-I",
        "-m",
        "keyframe.cli",
    ]
    assert captured["kwargs"]["cwd"] == working
    assert "PYTHONPATH" not in captured["kwargs"]["env"]
    assert captured["kwargs"]["env"]["PYTHONSAFEPATH"] == "1"


def test_runtime_probe_preserves_virtual_environment_launcher_symlink(
    monkeypatch,
    tmp_path,
):
    base_python = tmp_path / "base" / "python"
    base_python.parent.mkdir()
    base_python.write_text("", encoding="utf-8")
    launcher = tmp_path / "environment" / "bin" / "python"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(base_python)
    working = tmp_path / "working"
    working.mkdir()
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"packages": {}}),
            stderr="",
        )

    monkeypatch.setattr(release_evidence.subprocess, "run", fake_run)

    release_evidence.probe_runtime(
        launcher,
        working_directory=working,
        isolated=True,
    )

    assert captured["command"][0] == str(launcher)
    assert captured["command"][0] != str(base_python)


def test_artifact_filename_identity_is_explicit():
    assert release_evidence.artifact_name_version(
        "keyframe-0.6.3-py3-none-any.whl"
    ) == ("keyframe", "0.6.3")
    assert release_evidence.artifact_name_version(
        "keyframe-0.6.3.tar.gz"
    ) == ("keyframe", "0.6.3")
    with pytest.raises(
        release_evidence.ReleaseEvidenceError,
        match="project name",
    ):
        release_evidence.artifact_name_version(
            "other-0.6.3-py3-none-any.whl"
        )
