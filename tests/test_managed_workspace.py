from __future__ import annotations

import json
import os
import re
import stat
import uuid
from pathlib import Path

import pytest
from PIL import Image

import keyframe.managed_workspace as workspace_module
from keyframe.managed_workspace import (
    FrameGenerationPromotionError,
    FrameGenerationValidationError,
    ManagedWorkspace,
    ManagedWorkspaceError,
    OutputSessionError,
    inspect_frame_generation,
    known_public_artifact_paths,
    parse_canonical_uuid4,
)
from keyframe.output_session import (
    LOCK_FILENAME,
    OutputDirectoryLock,
    OutputRunSession,
)


def _write_generation(
    generation: Path,
    *,
    marker: str = "current",
    frame_index: int = 1,
) -> tuple[str, ...]:
    generation.mkdir(parents=True, exist_ok=True)
    frame_name = f"frame_{frame_index:06d}_{float(frame_index):.2f}s.png"
    color_seed = sum(marker.encode("utf-8"))
    Image.new(
        "RGB",
        (8, 8),
        (
            color_seed % 256,
            (color_seed * 3) % 256,
            (color_seed * 7) % 256,
        ),
    ).save(generation / frame_name)
    (generation / "captions.json").write_text(
        json.dumps(
            [
                {
                    "file": frame_name,
                    "timestamp": float(frame_index),
                    "caption": marker,
                }
            ]
        ),
        encoding="utf-8",
    )
    (generation / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "frames": [
                    {
                        "filename": frame_name,
                        "timestamp": float(frame_index),
                        "caption": marker,
                        "transcript_window": "",
                    }
                ],
                "metadata": {"marker": marker},
            }
        ),
        encoding="utf-8",
    )
    return (frame_name,)


def _initialize_workspace(output: Path) -> None:
    with OutputRunSession(output):
        pass


def _snapshot(root: Path) -> tuple[tuple[str, str, int, bytes | str], ...]:
    if not os.path.lexists(root):
        return ()
    rows: list[tuple[str, str, int, bytes | str]] = []

    def visit(directory: Path) -> None:
        with os.scandir(directory) as iterator:
            entries = sorted(iterator, key=lambda entry: entry.name)
        for entry in entries:
            path = Path(entry.path)
            relative = path.relative_to(root).as_posix()
            info = entry.stat(follow_symlinks=False)
            mode = stat.S_IMODE(info.st_mode)
            if stat.S_ISLNK(info.st_mode):
                rows.append((relative, "symlink", mode, os.readlink(path)))
            elif stat.S_ISDIR(info.st_mode):
                rows.append((relative, "directory", mode, b""))
                visit(path)
            elif stat.S_ISREG(info.st_mode):
                rows.append((relative, "file", mode, path.read_bytes()))
            else:
                rows.append((relative, "special", mode, b""))

    if root.is_dir() and not root.is_symlink():
        visit(root)
    return tuple(rows)


def _workspace_paths(output: Path) -> tuple[Path, Path, Path]:
    root = output / ".keyframe-work"
    return root, root / "runs", root / "recovery"


def test_managed_layout_uses_exact_ownership_and_preserves_legacy_paths(tmp_path):
    output = tmp_path / "out"
    legacy_run = output / "keyframe-run-unrelated-project"
    legacy_backup = output / "keyframe-frame-backup-precious"
    legacy_run.mkdir(parents=True)
    legacy_backup.mkdir()
    (legacy_run / "precious.txt").write_text("keep run", encoding="utf-8")
    (legacy_backup / "precious.txt").write_text("keep backup", encoding="utf-8")
    entry_id = uuid.uuid4()

    with OutputRunSession(output, run_id=entry_id) as session:
        assert session.staging is not None
        assert session.staging.root == (
            output / ".keyframe-work" / "runs" / str(entry_id)
        )
        assert session.staging.frames == session.staging.root / "frames"
        assert session.staging.frame_backup == (
            output / ".keyframe-work" / "recovery" / str(entry_id) / "frames"
        )
        assert session.staging.root.is_dir()
        (session.staging.root / "temporary.txt").write_text(
            "disposable",
            encoding="utf-8",
        )

        ownership = json.loads(
            (output / ".keyframe-work" / "ownership.json").read_text(encoding="utf-8")
        )
        assert set(ownership) == {
            "schema_version",
            "application",
            "purpose",
            "root_id",
        }
        assert ownership["schema_version"] == 1
        assert type(ownership["schema_version"]) is int
        assert ownership["application"] == "keyframe"
        assert ownership["purpose"] == "managed-output-workspace"
        assert re.fullmatch(r"[0-9a-f]{32}", ownership["root_id"])

    assert not session.staging.root.exists()
    assert (legacy_run / "precious.txt").read_text(encoding="utf-8") == "keep run"
    assert (legacy_backup / "precious.txt").read_text(encoding="utf-8") == "keep backup"


@pytest.mark.parametrize(
    "payload",
    [
        {
            "schema_version": True,
            "application": "keyframe",
            "purpose": "managed-output-workspace",
            "root_id": "a" * 32,
        },
        {
            "schema_version": 1,
            "application": "other",
            "purpose": "managed-output-workspace",
            "root_id": "a" * 32,
        },
        {
            "schema_version": 1,
            "application": "keyframe",
            "purpose": "managed-output-workspace",
            "root_id": "A" * 32,
        },
        {
            "schema_version": 1,
            "application": "keyframe",
            "purpose": "managed-output-workspace",
            "root_id": "a" * 32,
            "extra": "not allowed",
        },
    ],
)
def test_invalid_ownership_sentinel_preserves_every_managed_entry(
    tmp_path,
    payload,
):
    output = tmp_path / "out"
    _initialize_workspace(output)
    root, runs, _recovery = _workspace_paths(output)
    stale = runs / str(uuid.uuid4())
    stale.mkdir()
    (stale / "precious.txt").write_text("keep", encoding="utf-8")
    (root / "ownership.json").write_text(json.dumps(payload), encoding="utf-8")
    before = _snapshot(root)

    with pytest.raises(OutputSessionError, match="ownership sentinel"):
        with OutputRunSession(output):
            pytest.fail("invalid ownership must fail closed")

    assert _snapshot(root) == before


def test_interrupted_initialization_before_sentinel_reports_exact_manual_recovery(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "out"

    def fail_write(*_args, **_kwargs):
        raise OSError("injected sentinel write failure")

    monkeypatch.setattr(workspace_module, "atomic_write_json", fail_write)
    with pytest.raises(ManagedWorkspaceError) as raised:
        with OutputRunSession(output):
            pytest.fail("interrupted initialization must fail")

    root, runs, recovery = _workspace_paths(output)
    assert root.is_dir()
    assert not (root / "ownership.json").exists()
    assert not runs.exists()
    assert not recovery.exists()
    assert str(raised.value).endswith(
        "Keyframe will not modify the incomplete managed workspace."
    )
    before = _snapshot(root)

    with pytest.raises(ManagedWorkspaceError, match="Manual recovery"):
        with OutputRunSession(output):
            pytest.fail("incomplete root must remain untouched")
    assert _snapshot(root) == before


def test_interruption_after_sentinel_publication_leaves_reviewable_exact_root(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "out"
    real_write = workspace_module.atomic_write_json

    def publish_then_fail(path, payload):
        real_write(path, payload)
        raise OSError("injected interruption after sentinel publication")

    monkeypatch.setattr(
        workspace_module,
        "atomic_write_json",
        publish_then_fail,
    )
    with pytest.raises(ManagedWorkspaceError, match="Manual recovery"):
        with OutputRunSession(output):
            pytest.fail("interrupted initialization must fail")

    root, runs, recovery = _workspace_paths(output)
    assert set(path.name for path in root.iterdir()) == {"ownership.json"}
    assert not runs.exists()
    assert not recovery.exists()

    monkeypatch.setattr(workspace_module, "atomic_write_json", real_write)
    with OutputRunSession(output):
        assert runs.is_dir()
        assert recovery.is_dir()


def test_missing_structure_is_not_created_before_existing_root_is_validated(
    tmp_path,
):
    output = tmp_path / "out"
    output.mkdir()
    with OutputDirectoryLock(output):
        pass
    root, runs, recovery = _workspace_paths(output)
    root.mkdir()
    (root / "ownership.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "application": "keyframe",
                "purpose": "managed-output-workspace",
                "root_id": "a" * 32,
            }
        ),
        encoding="utf-8",
    )
    (root / "user-note.txt").write_text("keep", encoding="utf-8")
    before = _snapshot(root)

    with pytest.raises(ManagedWorkspaceError, match="unknown entries"):
        with OutputRunSession(output):
            pytest.fail("unknown root entry must fail closed")

    assert not runs.exists()
    assert not recovery.exists()
    assert _snapshot(root) == before


def test_complete_workspace_classification_precedes_stale_cleanup(tmp_path):
    output = tmp_path / "out"
    _initialize_workspace(output)
    root, runs, recovery = _workspace_paths(output)
    stale_id = uuid.uuid4()
    stale = runs / str(stale_id)
    stale.mkdir()
    (stale / "partial.json").write_text("stale", encoding="utf-8")
    _write_generation(output / "frames", marker="public", frame_index=1)
    obsolete_id = uuid.uuid4()
    _write_generation(
        recovery / str(obsolete_id) / "frames",
        marker="obsolete",
        frame_index=2,
    )
    empty_id = uuid.uuid4()
    (recovery / str(empty_id)).mkdir()
    legacy = output / "keyframe-run-unrelated-project"
    legacy.mkdir()
    (legacy / "precious.txt").write_text("keep", encoding="utf-8")

    with OutputRunSession(output) as session:
        assert session.lock is not None and session.lock.is_held
        assert not stale.exists()
        assert not (recovery / str(obsolete_id)).exists()
        assert not (recovery / str(empty_id)).exists()
        assert (legacy / "precious.txt").read_text(encoding="utf-8") == "keep"
        assert session.staging is not None and session.staging.root.exists()

    assert not any(runs.iterdir())


@pytest.mark.parametrize(
    "malformation",
    ["bad-name", "file", "entry-symlink", "tree-symlink"],
)
def test_malformed_run_preserves_the_entire_workspace(
    tmp_path,
    malformation,
):
    output = tmp_path / "out"
    _initialize_workspace(output)
    root, runs, recovery = _workspace_paths(output)
    valid_stale = runs / str(uuid.uuid4())
    valid_stale.mkdir()
    (valid_stale / "precious.txt").write_text("keep", encoding="utf-8")
    malformed_id = uuid.uuid4()
    external = tmp_path / "external"
    external.mkdir()
    if malformation == "bad-name":
        (runs / "not-a-uuid").mkdir()
    elif malformation == "file":
        (runs / str(malformed_id)).write_text("not a directory", encoding="utf-8")
    elif malformation == "entry-symlink":
        (runs / str(malformed_id)).symlink_to(external, target_is_directory=True)
    else:
        malformed = runs / str(malformed_id)
        malformed.mkdir()
        (malformed / "escape").symlink_to(external, target_is_directory=True)
    empty_recovery = recovery / str(uuid.uuid4())
    empty_recovery.mkdir()
    before = _snapshot(root)

    with pytest.raises(ManagedWorkspaceError):
        with OutputRunSession(output):
            pytest.fail("malformed run state must fail closed")

    assert _snapshot(root) == before
    assert external.is_dir()


def test_missing_public_restores_exactly_one_valid_recovery(tmp_path):
    output = tmp_path / "out"
    _initialize_workspace(output)
    _root, _runs, recovery = _workspace_paths(output)
    recovery_id = uuid.uuid4()
    expected_names = _write_generation(
        recovery / str(recovery_id) / "frames",
        marker="restored",
        frame_index=9,
    )

    with OutputRunSession(output):
        assert not (recovery / str(recovery_id)).exists()
        restored = inspect_frame_generation(
            output / "frames",
            label="test restored generation",
        )
        assert restored.frame_names == expected_names


@pytest.mark.parametrize("with_valid_recovery", [False, True])
def test_missing_public_with_empty_prepared_recovery_blocks_all_mutation(
    tmp_path,
    with_valid_recovery,
):
    output = tmp_path / "out"
    _initialize_workspace(output)
    root, runs, recovery = _workspace_paths(output)
    stale = runs / str(uuid.uuid4())
    stale.mkdir()
    (stale / "precious.txt").write_text("keep", encoding="utf-8")
    empty_id = uuid.uuid4()
    (recovery / str(empty_id)).mkdir()
    if with_valid_recovery:
        _write_generation(
            recovery / str(uuid.uuid4()) / "frames",
            marker="recoverable",
        )
    before = _snapshot(root)

    with pytest.raises(ManagedWorkspaceError, match="empty prepared"):
        with OutputRunSession(output):
            pytest.fail("missing public generation must preserve empty recovery state")

    assert _snapshot(root) == before
    assert stale.exists()
    assert not (output / "frames").exists()


def test_multiple_valid_recoveries_fail_without_any_mutation(tmp_path):
    output = tmp_path / "out"
    _initialize_workspace(output)
    root, _runs, recovery = _workspace_paths(output)
    for index in (1, 2):
        _write_generation(
            recovery / str(uuid.uuid4()) / "frames",
            marker=f"recovery-{index}",
            frame_index=index,
        )
    before = _snapshot(root)

    with pytest.raises(ManagedWorkspaceError, match="refusing to choose"):
        with OutputRunSession(output):
            pytest.fail("multiple recoveries must not be selected")

    assert _snapshot(root) == before
    assert not (output / "frames").exists()


@pytest.mark.parametrize(
    "malformation",
    ["partial", "frames-symlink", "invalid-generation", "entry-symlink"],
)
def test_malformed_recovery_blocks_all_mutation(tmp_path, malformation):
    output = tmp_path / "out"
    _initialize_workspace(output)
    root, runs, recovery = _workspace_paths(output)
    _write_generation(output / "frames", marker="public", frame_index=1)
    stale = runs / str(uuid.uuid4())
    stale.mkdir()
    (stale / "precious.txt").write_text("keep", encoding="utf-8")
    recovery_id = uuid.uuid4()
    container = recovery / str(recovery_id)
    external = tmp_path / "external"
    external.mkdir()
    if malformation == "partial":
        container.mkdir()
        (container / "partial.txt").write_text("keep", encoding="utf-8")
    elif malformation == "frames-symlink":
        container.mkdir()
        (container / "frames").symlink_to(external, target_is_directory=True)
    elif malformation == "invalid-generation":
        (container / "frames").mkdir(parents=True)
        (container / "frames" / "partial.png").write_bytes(b"not complete")
    else:
        container.symlink_to(external, target_is_directory=True)
    root_before = _snapshot(root)
    public_before = _snapshot(output / "frames")

    with pytest.raises(ManagedWorkspaceError, match="recovery"):
        with OutputRunSession(output):
            pytest.fail("malformed recovery must fail closed")

    assert _snapshot(root) == root_before
    assert _snapshot(output / "frames") == public_before
    assert stale.exists()
    assert external.is_dir()


def test_invalid_public_lists_all_unknown_paths_and_preserves_everything(tmp_path):
    output = tmp_path / "out"
    _initialize_workspace(output)
    root, runs, _recovery = _workspace_paths(output)
    _write_generation(output / "frames", marker="public")
    (output / "frames" / ".DS_Store").write_bytes(b"metadata")
    (output / "frames" / "user-note.txt").write_text("keep", encoding="utf-8")
    stale = runs / str(uuid.uuid4())
    stale.mkdir()
    (stale / "precious.txt").write_text("keep", encoding="utf-8")
    root_before = _snapshot(root)
    public_before = _snapshot(output / "frames")

    with pytest.raises(FrameGenerationValidationError) as raised:
        with OutputRunSession(output):
            pytest.fail("invalid public generation must block mutation")

    message = str(raised.value)
    assert ".DS_Store" in message
    assert "user-note.txt" in message
    assert _snapshot(root) == root_before
    assert _snapshot(output / "frames") == public_before


def test_optional_traces_must_be_regular_and_no_unknown_artifacts_are_allowed(
    tmp_path,
):
    generation = tmp_path / "frames"
    _write_generation(generation)
    (generation / "pipeline_trace.json").write_text("[]", encoding="utf-8")
    (generation / "debug_qa_trace.json").write_text("[]", encoding="utf-8")
    inspect_frame_generation(generation, label="test generation")

    (generation / "debug_qa_trace.json").unlink()
    (generation / "debug_qa_trace.json").mkdir()
    with pytest.raises(FrameGenerationValidationError, match="regular"):
        inspect_frame_generation(generation, label="test generation")


def test_symlinked_known_generation_artifact_is_rejected(tmp_path):
    generation = tmp_path / "frames"
    _write_generation(generation)
    external = tmp_path / "manifest.json"
    external.write_text(
        (generation / "manifest.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (generation / "manifest.json").unlink()
    (generation / "manifest.json").symlink_to(external)

    with pytest.raises(FrameGenerationValidationError, match="non-symlinked"):
        inspect_frame_generation(generation, label="test generation")
    assert external.is_file()


def test_failed_publish_restores_previous_generation_and_staging(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "out"
    _write_generation(output / "frames", marker="previous", frame_index=1)
    previous = _snapshot(output / "frames")

    with OutputRunSession(output) as session:
        assert session.staging is not None
        assert session.workspace is not None
        expected = _write_generation(
            session.staging.frames,
            marker="replacement",
            frame_index=2,
        )
        real_replace = workspace_module.os.replace

        def fail_staged_publish(source, target):
            if Path(source) == session.staging.frames:
                raise OSError("injected publish failure")
            return real_replace(source, target)

        monkeypatch.setattr(workspace_module.os, "replace", fail_staged_publish)
        with pytest.raises(
            FrameGenerationPromotionError,
            match="previous generation was restored",
        ):
            session.workspace.promote_frame_generation(
                session.staging.frames,
                expected_frame_names=expected,
                entry_id=session.entry_id,
            )

        assert _snapshot(output / "frames") == previous
        assert session.staging.frames.exists()
        assert not session.staging.frame_backup.parent.exists()


def test_failed_post_publish_validation_rolls_back_last_valid_generation(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "out"
    _write_generation(output / "frames", marker="previous", frame_index=1)
    previous = _snapshot(output / "frames")

    with OutputRunSession(output) as session:
        assert session.staging is not None
        assert session.workspace is not None
        expected = _write_generation(
            session.staging.frames,
            marker="replacement",
            frame_index=2,
        )
        real_inspect = workspace_module.inspect_frame_generation

        def fail_published_validation(path, *, label, expected_frame_names=None):
            if label == "published frame generation":
                raise FrameGenerationValidationError(
                    "injected post-publish validation failure"
                )
            return real_inspect(
                path,
                label=label,
                expected_frame_names=expected_frame_names,
            )

        monkeypatch.setattr(
            workspace_module,
            "inspect_frame_generation",
            fail_published_validation,
        )
        with pytest.raises(
            FrameGenerationPromotionError,
            match="previous generation was restored",
        ):
            session.workspace.promote_frame_generation(
                session.staging.frames,
                expected_frame_names=expected,
                entry_id=session.entry_id,
            )

        assert _snapshot(output / "frames") == previous
        assert session.staging.frames.exists()


def test_cleanup_failure_leaves_valid_recovery_for_next_locked_session(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "out"
    _write_generation(output / "frames", marker="previous", frame_index=1)

    with OutputRunSession(output) as session:
        assert session.staging is not None
        assert session.workspace is not None
        expected = _write_generation(
            session.staging.frames,
            marker="replacement",
            frame_index=2,
        )
        real_delete = ManagedWorkspace.delete_entry

        def fail_recovery_delete(self, kind, entry_id):
            if kind == "recovery":
                raise ManagedWorkspaceError("injected cleanup permission failure")
            return real_delete(self, kind, entry_id)

        monkeypatch.setattr(
            ManagedWorkspace,
            "delete_entry",
            fail_recovery_delete,
        )
        session.workspace.promote_frame_generation(
            session.staging.frames,
            expected_frame_names=expected,
            entry_id=session.entry_id,
        )
        recovery_container = session.staging.frame_backup.parent
        assert recovery_container.exists()
        inspect_frame_generation(
            recovery_container / "frames",
            label="preserved previous generation",
        )

    monkeypatch.setattr(ManagedWorkspace, "delete_entry", real_delete)
    with OutputRunSession(output):
        assert not recovery_container.exists()


def test_failed_recovery_restore_preserves_recovery_and_missing_public(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "out"
    _initialize_workspace(output)
    _root, _runs, recovery = _workspace_paths(output)
    recovery_id = uuid.uuid4()
    recovery_frames = recovery / str(recovery_id) / "frames"
    _write_generation(recovery_frames, marker="recoverable")
    before = _snapshot(recovery / str(recovery_id))
    real_replace = workspace_module.os.replace

    def fail_restore(source, target):
        if Path(source) == recovery_frames:
            raise OSError("injected restore failure")
        return real_replace(source, target)

    monkeypatch.setattr(workspace_module.os, "replace", fail_restore)
    with pytest.raises(ManagedWorkspaceError, match="failed to restore"):
        with OutputRunSession(output):
            pytest.fail("failed restore must not admit a run")

    assert not (output / "frames").exists()
    assert _snapshot(recovery / str(recovery_id)) == before


def test_deletion_api_rejects_arbitrary_identifiers_and_symlink_trees(tmp_path):
    output = tmp_path / "out"
    with OutputRunSession(output) as session:
        assert session.workspace is not None
        with pytest.raises(TypeError, match="parsed UUID"):
            session.workspace.delete_entry("run", "../escape")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="UUIDv4"):
            session.workspace.delete_entry("run", uuid.uuid1())
        with pytest.raises(ValueError, match="entry kind"):
            session.workspace.delete_entry("other", uuid.uuid4())  # type: ignore[arg-type]

        extra_id = uuid.uuid4()
        extra = session.workspace.create_run(extra_id).root
        external = tmp_path / "external"
        external.mkdir()
        (extra / "escape").symlink_to(external, target_is_directory=True)
        with pytest.raises(ManagedWorkspaceError, match="symlink"):
            session.workspace.delete_entry("run", extra_id)
        assert extra.is_dir()
        assert external.is_dir()
        (extra / "escape").unlink()
        session.workspace.delete_entry("run", extra_id)

        retained_workspace = session.workspace

    with pytest.raises(ManagedWorkspaceError, match="held output lock"):
        retained_workspace.create_run(uuid.uuid4())


@pytest.mark.parametrize("lock_kind", ["symlink", "directory"])
def test_symlinked_or_non_regular_lock_path_is_rejected_without_following(
    tmp_path,
    lock_kind,
):
    output = tmp_path / "out"
    output.mkdir()
    lock_path = output / LOCK_FILENAME
    external = tmp_path / "external-lock"
    if lock_kind == "symlink":
        external.write_text("do not touch", encoding="utf-8")
        lock_path.symlink_to(external)
    else:
        lock_path.mkdir()

    with pytest.raises(OutputSessionError, match="regular non-symlinked"):
        with OutputRunSession(output):
            pytest.fail("unsafe lock path must be rejected")

    assert not (output / ".keyframe-work").exists()
    if lock_kind == "symlink":
        assert external.read_text(encoding="utf-8") == "do not touch"
        assert lock_path.is_symlink()
    else:
        assert lock_path.is_dir()


@pytest.mark.skipif(os.name == "nt", reason="FIFO creation requires POSIX")
def test_fifo_lock_path_is_rejected_without_blocking(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    lock_path = output / LOCK_FILENAME
    os.mkfifo(lock_path)

    with pytest.raises(OutputSessionError, match="regular non-symlinked"):
        with OutputRunSession(output):
            pytest.fail("FIFO lock path must be rejected")

    assert stat.S_ISFIFO(lock_path.lstat().st_mode)
    assert not (output / ".keyframe-work").exists()


def test_public_summary_enumerates_known_artifacts_without_workspace_traversal(
    tmp_path,
):
    output = tmp_path / "out"
    _initialize_workspace(output)
    frame_names = _write_generation(output / "frames", marker="public")
    (output / "transcript.txt").write_text("transcript", encoding="utf-8")
    (output / "unrelated.txt").write_text("user data", encoding="utf-8")
    root, _runs, _recovery = _workspace_paths(output)
    (root / "ownership.json").read_text(encoding="utf-8")

    artifacts = known_public_artifact_paths(output)
    relatives = {path.relative_to(output).as_posix() for path in artifacts}

    assert "transcript.txt" in relatives
    assert "frames/manifest.json" in relatives
    assert "frames/captions.json" in relatives
    assert f"frames/{frame_names[0]}" in relatives
    assert "unrelated.txt" not in relatives
    assert all(not relative.startswith(".keyframe-work/") for relative in relatives)


@pytest.mark.parametrize(
    "value",
    [
        "",
        ".hidden",
        "../escape",
        "with/slash",
        "A" * 36,
        str(uuid.uuid1()),
        uuid.uuid1(),
    ],
)
def test_canonical_uuid_parser_rejects_aliases_and_non_v4_values(value):
    with pytest.raises((TypeError, ValueError)):
        parse_canonical_uuid4(value)


def test_canonical_uuid_parser_accepts_uuid4_object_and_exact_string():
    value = uuid.uuid4()
    assert parse_canonical_uuid4(value) == value
    assert parse_canonical_uuid4(str(value)) == value
