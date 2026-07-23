from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from keyframe import stage_supervisor as supervisor_module
from keyframe import transcript
from keyframe.output_session import OutputRunSession, OutputSessionError
from keyframe.stage_supervisor import (
    StageCheckpointError,
    StageSupervisor,
    StageWorkerError,
)
from tests.native_process_harness import (
    crashing_transcription_worker,
    disk_full_transcription_worker,
    successful_transcription_worker,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _subprocess_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _wait_for_path(path: Path, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out waiting for {path}")
        time.sleep(0.01)


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_pid_exit(pid: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while _pid_is_running(pid):
        if time.monotonic() >= deadline:
            raise AssertionError(f"child PID {pid} survived its supervisor")
        time.sleep(0.01)


def _start_holding_supervisor(
    output: Path,
    ready: Path,
    worker_pid: Path,
    *,
    run_id: str,
    checkpoint_state: str,
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tests.native_process_harness",
            "hold",
            "--output",
            str(output),
            "--run-id",
            run_id,
            "--ready",
            str(ready),
            "--worker-pid",
            str(worker_pid),
            "--checkpoint-state",
            checkpoint_state,
        ],
        cwd=REPOSITORY_ROOT,
        env=_subprocess_environment(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


@pytest.mark.skipif(os.name == "nt", reason="native POSIX signals are required")
@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
@pytest.mark.parametrize("checkpoint_state", ["before", "after"])
def test_native_signal_joins_worker_and_discards_uncommitted_checkpoint(
    tmp_path,
    signum,
    checkpoint_state,
):
    output = tmp_path / "output"
    ready = tmp_path / "parent.ready"
    worker_pid_path = tmp_path / "worker.pid"
    process = _start_holding_supervisor(
        output,
        ready,
        worker_pid_path,
        run_id=f"signal-{signum}-{checkpoint_state}",
        checkpoint_state=checkpoint_state,
    )
    try:
        _wait_for_path(ready)
        worker_pid = int(worker_pid_path.read_text(encoding="ascii"))
        os.kill(process.pid, signum)
        _stdout, stderr = process.communicate(timeout=10.0)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5.0)

    assert process.returncode != 0
    assert "KeyboardInterrupt" in stderr or "SupervisorSignal" in stderr
    _wait_for_pid_exit(worker_pid)
    assert not (output / "transcript.raw.json").exists()
    assert not list(output.glob("keyframe-run-*"))


@pytest.mark.skipif(os.name == "nt", reason="native advisory-lock test requires POSIX")
def test_concurrent_native_process_cannot_enter_the_same_output(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    previous = output / "transcript.raw.json"
    previous.write_text("previous", encoding="utf-8")
    ready = tmp_path / "holder.ready"
    worker_pid_path = tmp_path / "holder-worker.pid"
    holder = _start_holding_supervisor(
        output,
        ready,
        worker_pid_path,
        run_id="native-holder",
        checkpoint_state="before",
    )
    try:
        _wait_for_path(ready)
        probe = subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.native_process_harness",
                "probe",
                "--output",
                str(output),
                "--run-id",
                "native-contender",
            ],
            cwd=REPOSITORY_ROOT,
            env=_subprocess_environment(),
            text=True,
            capture_output=True,
            timeout=10.0,
            check=False,
        )
        assert probe.returncode == 73
        assert "already in use" in probe.stderr
        assert previous.read_text(encoding="utf-8") == "previous"
        assert not (output / "keyframe-run-native-contender").exists()
    finally:
        holder.send_signal(signal.SIGTERM)
        holder.communicate(timeout=10.0)

    _wait_for_pid_exit(int(worker_pid_path.read_text(encoding="ascii")))
    assert not list(output.glob("keyframe-run-*"))


@pytest.mark.parametrize("checkpoint_state", ["before", "after"])
def test_worker_crash_never_promotes_staged_checkpoint(tmp_path, checkpoint_state):
    output = tmp_path / "output"
    with StageSupervisor(output, run_id=f"crash-{checkpoint_state}") as supervisor:
        assert supervisor.staging is not None
        handle = supervisor._start_stage(
            stage="transcription",
            target=crashing_transcription_worker,
            request={
                "checkpoint_state": checkpoint_state,
                "checkpoint": str(supervisor.staging.transcript_raw),
                "exitcode": 17,
            },
            checkpoint_path=supervisor.staging.transcript_raw,
            validator=transcript.read_raw_transcript_checkpoint,
        )

        with pytest.raises(StageWorkerError, match="status 17"):
            supervisor.complete(handle)

        assert handle.process.exitcode == 17
        assert not handle.process.is_alive()
        assert not (output / "transcript.raw.json").exists()
        assert supervisor.staging.transcript_raw.exists() is (
            checkpoint_state == "after"
        )

    assert not list(output.glob("keyframe-run-*"))


def test_spawned_disk_exhaustion_is_reliable_and_never_published(tmp_path):
    output = tmp_path / "output"
    with StageSupervisor(output, run_id="disk-full") as supervisor:
        assert supervisor.staging is not None
        handle = supervisor._start_stage(
            stage="transcription",
            target=disk_full_transcription_worker,
            request={},
            checkpoint_path=supervisor.staging.transcript_raw,
            validator=transcript.read_raw_transcript_checkpoint,
        )

        with pytest.raises(StageWorkerError, match="disk exhaustion") as raised:
            supervisor.complete(handle)

        assert raised.value.error_type == "OSError"
        assert not handle.process.is_alive()
        assert not (output / "transcript.raw.json").exists()

    assert not list(output.glob("keyframe-run-*"))


def test_checkpoint_promotion_rename_failure_is_controlled_and_preserves_previous(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "output"
    output.mkdir()
    public = output / "transcript.raw.json"
    transcript.write_raw_transcript_checkpoint(
        [transcript.TranscriptSegment(0.0, 1.0, "previous")],
        public,
    )

    with StageSupervisor(output, run_id="rename-failure") as supervisor:
        assert supervisor.staging is not None
        stage_handle = supervisor._start_stage(
            stage="transcription",
            target=successful_transcription_worker,
            request={"checkpoint": str(supervisor.staging.transcript_raw)},
            checkpoint_path=supervisor.staging.transcript_raw,
            validator=transcript.read_raw_transcript_checkpoint,
        )
        monkeypatch.setattr(
            supervisor_module,
            "atomic_promote_file",
            lambda *_args: (_ for _ in ()).throw(OSError("injected rename failure")),
        )

        with pytest.raises(StageCheckpointError, match="promotion failed"):
            supervisor.complete(stage_handle)

        assert public.exists()
        assert transcript.read_raw_transcript_checkpoint(public)[0].text == "previous"
        assert supervisor.staging.transcript_raw.exists()

    assert not list(output.glob("keyframe-run-*"))


@pytest.mark.skipif(
    os.name == "nt" or getattr(os, "geteuid", lambda: 1)() == 0,
    reason="read-only permission behavior requires a non-root POSIX user",
)
@pytest.mark.parametrize("session_type", [StageSupervisor, OutputRunSession])
def test_read_only_output_fails_without_mixing_artifacts(tmp_path, session_type):
    output = tmp_path / "output"
    with session_type(output, run_id="create-lock"):
        pass
    previous = output / "transcript.raw.json"
    previous.write_text("previous", encoding="utf-8")
    runs = output / ".keyframe-work" / "runs"
    output.chmod(0o500)
    runs.chmod(0o500)
    try:
        with pytest.raises(
            OutputSessionError,
            match="failed to create managed run",
        ):
            with session_type(output, run_id="read-only"):
                pytest.fail("read-only output must not admit a run")
    finally:
        runs.chmod(0o700)
        output.chmod(0o700)

    assert previous.read_text(encoding="utf-8") == "previous"
    assert not list(output.glob("keyframe-run-*"))
