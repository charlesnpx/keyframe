from __future__ import annotations

import multiprocessing as mp
import os
import queue
import signal
import stat
import sys
import threading
import time
from pathlib import Path

import pytest

from keyframe import stage_supervisor as supervisor_module
from keyframe import transcript
from keyframe.stage_supervisor import (
    DiarizationWorkerRequest,
    OutputDirectoryLock,
    OutputDirectoryLockedError,
    StageCheckpointError,
    StageHandle,
    StageProgress,
    StageProtocolError,
    StageSupervisor,
    StageSupervisorError,
    StageTerminal,
    StageWorkerError,
    SupervisorSignal,
    TranscriptionWorkerRequest,
    _close_worker_ipc,
    _execute_worker,
    diarization_worker_entry,
    emit_stage_progress,
    transcription_worker_entry,
)


class _FakeTerminal:
    def __init__(self):
        self.messages = []
        self.closed = False

    def send(self, message):
        self.messages.append(message)

    def close(self):
        self.closed = True


class _FakeProgressQueue:
    def __init__(self):
        self.events = []
        self.closed = False
        self.cancelled = False

    def put_nowait(self, event):
        self.events.append(event)

    def close(self):
        self.closed = True

    def cancel_join_thread(self):
        self.cancelled = True


class _DroppingProgressQueue(_FakeProgressQueue):
    def put_nowait(self, event):
        raise queue.Full


class _FakeEvent:
    def __init__(self, value=False):
        self.value = value

    def is_set(self):
        return self.value


def _spawn_test_worker(request, terminal_send, progress_queue, cancellation_event):
    mode = request["mode"]
    checkpoint = Path(request["checkpoint"])
    try:
        if mode == "crash":
            os._exit(7)
        if mode == "block":
            Path(request["started"]).write_text(str(os.getpid()), encoding="utf-8")
            while True:
                time.sleep(0.05)
        if mode == "missing-terminal":
            return
        if mode == "invalid":
            checkpoint.write_text("not valid JSON", encoding="utf-8")
        elif request["stage"] == "transcription":
            transcript.write_raw_transcript_checkpoint(
                [transcript.TranscriptSegment(0.123456789, 1.987654321, "worker")],
                checkpoint,
            )
        else:
            transcript.write_diarization_checkpoint(
                [transcript.DiarizationRow(0.1, 1.9, "SPEAKER_00")],
                checkpoint,
            )

        for index in range(request.get("progress_events", 0)):
            emit_stage_progress(
                progress_queue,
                StageProgress(request["stage"], "progress", str(index)),
            )
        callback_observed = request.get("callback_observed")
        if callback_observed is not None and not callback_observed.wait(15.0):
            raise RuntimeError("progress callback was not observed")
        terminal = StageTerminal.succeeded(
            request["stage"],
            {"record_count": 1, "language": "en"},
        )
        terminal_send.send(terminal)
        if mode == "duplicate-terminal":
            terminal_send.send(terminal)
    finally:
        _close_worker_ipc(terminal_send, progress_queue)


def _spawn_routed_output_worker(
    request,
    terminal_send,
    progress_queue,
    cancellation_event,
):
    stage = request["stage"]
    checkpoint = Path(request["checkpoint"])

    def operation():
        print(f"{stage} stdout")
        print(f"{stage} stderr", file=sys.stderr)
        time.sleep(0.05)
        if stage == "transcription":
            transcript.write_raw_transcript_checkpoint(
                [transcript.TranscriptSegment(0.1, 1.9, "worker")],
                checkpoint,
            )
        else:
            transcript.write_diarization_checkpoint(
                [transcript.DiarizationRow(0.1, 1.9, "SPEAKER_00")],
                checkpoint,
            )
        return {"record_count": 1}

    _execute_worker(
        stage,
        terminal_send,
        progress_queue,
        cancellation_event,
        operation,
    )


def _start_test_stage(supervisor, *, stage="transcription", mode="success", **request_values):
    assert supervisor.staging is not None
    if stage == "transcription":
        checkpoint = supervisor.staging.transcript_raw
        validator = transcript.read_raw_transcript_checkpoint
    else:
        checkpoint = supervisor.staging.diarization
        validator = transcript.read_diarization_checkpoint
    request = {
        "stage": stage,
        "mode": mode,
        "checkpoint": str(checkpoint),
        **request_values,
    }
    return supervisor._start_stage(
        stage=stage,
        target=_spawn_test_worker,
        request=request,
        checkpoint_path=checkpoint,
        validator=validator,
    )


def _wait_for_path(path: Path, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out waiting for {path}")
        time.sleep(0.01)


def test_concurrent_worker_text_is_routed_through_stage_progress(tmp_path, capfd):
    events = []
    output = tmp_path / "out"

    with StageSupervisor(
        output,
        progress_callback=events.append,
        progress_capacity=64,
    ) as supervisor:
        assert supervisor.staging is not None
        transcription_handle = supervisor._start_stage(
            stage="transcription",
            target=_spawn_routed_output_worker,
            request={
                "stage": "transcription",
                "checkpoint": str(supervisor.staging.transcript_raw),
            },
            checkpoint_path=supervisor.staging.transcript_raw,
            validator=transcript.read_raw_transcript_checkpoint,
        )
        diarization_handle = supervisor._start_stage(
            stage="diarization",
            target=_spawn_routed_output_worker,
            request={
                "stage": "diarization",
                "checkpoint": str(supervisor.staging.diarization),
            },
            checkpoint_path=supervisor.staging.diarization,
            validator=transcript.read_diarization_checkpoint,
        )

        supervisor.complete(transcription_handle)
        supervisor.complete(diarization_handle)

    captured = capfd.readouterr()
    assert "transcription stdout" not in captured.out
    assert "transcription stderr" not in captured.err
    assert "diarization stdout" not in captured.out
    assert "diarization stderr" not in captured.err
    for stage in ("transcription", "diarization"):
        assert any(
            event.stage == stage
            and event.event == "output"
            and event.message in {f"{stage} stdout", f"{stage} stderr"}
            for event in events
        )


def test_spawned_worker_is_non_daemon_validated_and_promoted(tmp_path):
    output = tmp_path / "output"
    progress = []

    with StageSupervisor(
        output,
        progress_callback=progress.append,
        run_id="success",
    ) as supervisor:
        before_launch = time.monotonic()
        handle = _start_test_stage(
            supervisor,
            progress_events=3,
        )
        assert handle.process.daemon is False

        completion = supervisor.complete(handle)
        after_completion = time.monotonic()

        assert completion.stage == "transcription"
        assert completion.checkpoint_path == output / "transcript.raw.json"
        assert completion.metadata == {"record_count": 1, "language": "en"}
        with pytest.raises(TypeError):
            completion.metadata["language"] = "changed"
        assert completion.records == (
            transcript.TranscriptSegment(0.123456789, 1.987654321, "worker"),
        )
        assert not handle.process.is_alive()
        assert before_launch <= handle.ended_at <= after_completion
        assert completion.checkpoint_path.exists()
        assert not supervisor.staging.transcript_raw.exists()

    assert not (output / "keyframe-run-success").exists()
    assert (output / "keyframe-output.lock").exists()


def test_lossy_progress_cannot_block_reliable_terminal_during_parent_work(tmp_path):
    output = tmp_path / "output"
    progress = []

    with StageSupervisor(
        output,
        progress_callback=progress.append,
        progress_capacity=1,
        run_id="flood",
    ) as supervisor:
        handle = _start_test_stage(supervisor, progress_events=10_000)

        # Stand in for synchronous frame extraction while the monitor keeps draining.
        time.sleep(0.2)
        completion = supervisor.complete(handle)

    assert completion.metadata["record_count"] == 1
    # Progress is deliberately lossy, including the possibility of no delivery.
    assert len(progress) < 10_000


def test_slow_progress_callback_cannot_delay_reliable_terminal(tmp_path):
    output = tmp_path / "output"
    callback_started = threading.Event()
    callback_observed = mp.get_context("spawn").Event()
    release_callback = threading.Event()

    def block_progress(_event):
        callback_started.set()
        callback_observed.set()
        release_callback.wait()

    try:
        with StageSupervisor(
            output,
            progress_callback=block_progress,
            progress_capacity=1,
            run_id="slow-progress",
        ) as supervisor:
            handle = _start_test_stage(
                supervisor,
                progress_events=1,
                callback_observed=callback_observed,
            )
            assert callback_started.wait(timeout=15.0)

            completion = handle.wait(timeout=15.0)
            promoted = supervisor.complete(handle)

            assert completion.metadata["record_count"] == 1
            assert promoted.checkpoint_path == output / "transcript.raw.json"
    finally:
        release_callback.set()


def test_second_supervisor_fails_nonblocking_without_touching_artifacts(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with StageSupervisor(output, run_id="first"):
        before = {path.name for path in output.iterdir()}
        with pytest.raises(OutputDirectoryLockedError, match="already in use"):
            with StageSupervisor(output, run_id="second"):
                pytest.fail("the second supervisor must not acquire the lock")
        after = {path.name for path in output.iterdir()}

        assert after == before
        assert sentinel.read_text(encoding="utf-8") == "keep"
        assert not (output / "keyframe-run-second").exists()


def test_nonzero_worker_exit_is_controlled_and_never_promoted(tmp_path):
    output = tmp_path / "output"

    with StageSupervisor(output, run_id="crash") as supervisor:
        handle = _start_test_stage(supervisor, mode="crash")

        with pytest.raises(StageWorkerError, match="status 7"):
            supervisor.complete(handle)

        assert not (output / "transcript.raw.json").exists()


def test_successful_exit_without_terminal_message_is_protocol_error(tmp_path):
    output = tmp_path / "output"

    with StageSupervisor(output, run_id="missing") as supervisor:
        handle = _start_test_stage(supervisor, mode="missing-terminal")

        with pytest.raises(StageProtocolError, match="0 terminal messages"):
            supervisor.complete(handle)


def test_duplicate_terminal_message_is_protocol_error(tmp_path):
    output = tmp_path / "output"

    with StageSupervisor(output, run_id="duplicate") as supervisor:
        handle = _start_test_stage(supervisor, mode="duplicate-terminal")

        with pytest.raises(StageProtocolError, match="2 terminal messages"):
            supervisor.complete(handle)


def test_invalid_checkpoint_is_rejected_and_previous_public_artifact_survives(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    public = output / "transcript.raw.json"
    public.write_text("previous", encoding="utf-8")

    with StageSupervisor(output, run_id="invalid") as supervisor:
        handle = _start_test_stage(supervisor, mode="invalid")

        with pytest.raises(StageCheckpointError, match="checkpoint is invalid"):
            supervisor.complete(handle)

        assert public.read_text(encoding="utf-8") == "previous"

    assert not (output / "keyframe-run-invalid").exists()


def test_cancel_escalates_joins_and_cleans_current_run(tmp_path):
    output = tmp_path / "output"
    started = tmp_path / "worker.pid"

    with StageSupervisor(
        output,
        cancellation_grace_seconds=0.05,
        run_id="cancel",
    ) as supervisor:
        handle = _start_test_stage(
            supervisor,
            mode="block",
            started=str(started),
        )
        _wait_for_path(started)

        supervisor.cancel(handle)

        assert not handle.process.is_alive()
        assert handle.process.exitcode is not None

    assert not (output / "keyframe-run-cancel").exists()


def test_parent_exception_joins_live_worker_and_cleans_current_run(tmp_path):
    output = tmp_path / "output"
    started = tmp_path / "exception-worker.pid"
    handle = None

    with pytest.raises(RuntimeError, match="parent failed"):
        with StageSupervisor(
            output,
            cancellation_grace_seconds=0.05,
            run_id="parent-error",
        ) as supervisor:
            handle = _start_test_stage(
                supervisor,
                mode="block",
                started=str(started),
            )
            _wait_for_path(started)
            raise RuntimeError("parent failed")

    assert handle is not None
    assert not handle.process.is_alive()
    assert not (output / "keyframe-run-parent-error").exists()


@pytest.mark.parametrize("failure", [RuntimeError("registration failed"), KeyboardInterrupt()])
def test_failure_after_spawn_joins_unregistered_worker(tmp_path, monkeypatch, failure):
    output = tmp_path / "output"
    processes = []

    def fail_monitor_start(handle):
        processes.append(handle.process)
        raise failure

    monkeypatch.setattr(StageHandle, "start_monitors", fail_monitor_start)

    with StageSupervisor(
        output,
        cancellation_grace_seconds=0.05,
        run_id="registration-failure",
    ) as supervisor:
        with pytest.raises(type(failure), match=str(failure) or None):
            _start_test_stage(supervisor, mode="block")

        assert supervisor._handles == []

    assert len(processes) == 1
    assert not processes[0].is_alive()
    assert processes[0].exitcode is not None
    assert not (output / "keyframe-run-registration-failure").exists()


@pytest.mark.skipif(
    os.name == "nt" or not hasattr(signal, "SIGTERM"),
    reason="Python cannot intercept os.kill(SIGTERM) on Windows",
)
def test_sigterm_unwinds_joins_worker_and_restores_handler(tmp_path):
    output = tmp_path / "output"
    started = tmp_path / "signal-worker.pid"
    previous_handler = signal.getsignal(signal.SIGTERM)
    handle = None

    with pytest.raises(SupervisorSignal):
        with StageSupervisor(
            output,
            cancellation_grace_seconds=0.05,
            run_id="signal",
        ) as supervisor:
            handle = _start_test_stage(
                supervisor,
                mode="block",
                started=str(started),
            )
            _wait_for_path(started)
            os.kill(os.getpid(), signal.SIGTERM)

    assert handle is not None
    assert not handle.process.is_alive()
    assert signal.getsignal(signal.SIGTERM) == previous_handler
    assert not (output / "keyframe-run-signal").exists()


def test_committed_raw_transcript_survives_later_diarization_failure(tmp_path):
    output = tmp_path / "output"

    with StageSupervisor(output, run_id="partial") as supervisor:
        raw_handle = _start_test_stage(supervisor)
        raw_completion = supervisor.complete(raw_handle)
        assert raw_completion.checkpoint_path.exists()

        diarization_handle = _start_test_stage(
            supervisor,
            stage="diarization",
            mode="invalid",
        )
        with pytest.raises(StageCheckpointError):
            supervisor.complete(diarization_handle)

    assert transcript.read_raw_transcript_checkpoint(
        output / "transcript.raw.json"
    ) == (transcript.TranscriptSegment(0.123456789, 1.987654321, "worker"),)
    assert not (output / "diarization.json").exists()
    assert not (output / "keyframe-run-partial").exists()


def test_stale_run_cleanup_is_scoped_and_does_not_follow_symlinks(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    stale = output / "keyframe-run-stale"
    stale.mkdir()
    (stale / "partial.json").write_text("partial", encoding="utf-8")
    unrelated = output / "keep-me"
    unrelated.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    run_symlink = output / "keyframe-run-symlink"
    run_symlink.symlink_to(external, target_is_directory=True)

    with StageSupervisor(output, run_id="current") as supervisor:
        assert not stale.exists()
        assert unrelated.exists()
        assert run_symlink.is_symlink()
        assert external.exists()
        assert supervisor.staging.root.exists()

    assert unrelated.exists()
    assert run_symlink.is_symlink()
    assert external.exists()


@pytest.mark.parametrize("frame_artifact", ["file", "directory-symlink"])
def test_transcript_session_ignores_unrelated_frame_path_without_recovery_backup(
    tmp_path,
    frame_artifact,
):
    output = tmp_path / "output"
    output.mkdir()
    frames = output / "frames"
    external = tmp_path / "external-frames"
    if frame_artifact == "file":
        frames.write_text("user-owned", encoding="utf-8")
    else:
        external.mkdir()
        (external / "sentinel.txt").write_text("user-owned", encoding="utf-8")
        frames.symlink_to(external, target_is_directory=True)

    with StageSupervisor(output, run_id="transcript-only") as supervisor:
        assert supervisor.staging is not None
        if frame_artifact == "file":
            assert frames.read_text(encoding="utf-8") == "user-owned"
        else:
            assert frames.is_symlink()
            assert (frames / "sentinel.txt").read_text(encoding="utf-8") == (
                "user-owned"
            )

    if frame_artifact == "file":
        assert frames.read_text(encoding="utf-8") == "user-owned"
    else:
        assert frames.is_symlink()
        assert external.exists()


def test_failed_entry_preserves_collision_and_releases_output_lock(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    collision = output / "keyframe-run-collision"
    collision.write_text("not a run directory", encoding="utf-8")

    with pytest.raises(StageSupervisorError, match="failed to initialize"):
        with StageSupervisor(output, run_id="collision"):
            pytest.fail("a run directory must not replace an existing file")

    assert collision.read_text(encoding="utf-8") == "not a run directory"
    with StageSupervisor(output, run_id="after-collision") as supervisor:
        assert supervisor.staging is not None
        assert supervisor.staging.root.exists()


def test_interruption_after_output_lock_acquire_is_released(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    original_acquire = OutputDirectoryLock.acquire

    def acquire_then_interrupt(lock):
        original_acquire(lock)
        raise KeyboardInterrupt("interrupted after lock acquisition")

    monkeypatch.setattr(OutputDirectoryLock, "acquire", acquire_then_interrupt)
    with pytest.raises(KeyboardInterrupt, match="after lock acquisition"):
        with StageSupervisor(output, run_id="interrupted-entry"):
            pytest.fail("entry must be interrupted")

    monkeypatch.setattr(OutputDirectoryLock, "acquire", original_acquire)
    with StageSupervisor(output, run_id="after-interrupted-entry"):
        pass


def test_interruption_inside_output_lock_acquire_closes_descriptor(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "output"
    output.mkdir()
    original_acquire_descriptor = OutputDirectoryLock._acquire_descriptor

    def lock_then_interrupt(lock, descriptor):
        original_acquire_descriptor(lock, descriptor)
        raise KeyboardInterrupt("interrupted inside lock acquisition")

    monkeypatch.setattr(
        OutputDirectoryLock,
        "_acquire_descriptor",
        lock_then_interrupt,
    )
    lock = OutputDirectoryLock(output)
    with pytest.raises(KeyboardInterrupt, match="inside lock acquisition"):
        lock.acquire()
    assert lock._descriptor is None

    monkeypatch.setattr(
        OutputDirectoryLock,
        "_acquire_descriptor",
        original_acquire_descriptor,
    )
    with OutputDirectoryLock(output):
        pass


def test_close_from_another_thread_releases_lock_and_can_restore_on_owner(tmp_path):
    output = tmp_path / "output"
    previous_handler = signal.getsignal(signal.SIGTERM)
    supervisor = StageSupervisor(output, run_id="cross-thread-close")
    supervisor.__enter__()
    errors = []

    def close_from_thread():
        try:
            supervisor.close()
        except BaseException as exc:
            errors.append(exc)

    try:
        close_thread = threading.Thread(target=close_from_thread)
        close_thread.start()
        close_thread.join(timeout=5.0)

        assert not close_thread.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], ValueError)
        with StageSupervisor(output, run_id="after-cross-thread-close"):
            pass
    finally:
        supervisor.close()

    assert signal.getsignal(signal.SIGTERM) == previous_handler


def test_complete_rejects_foreign_handle_and_mismatched_checkpoint(tmp_path):
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"

    with (
        StageSupervisor(first_output, run_id="first") as first,
        StageSupervisor(second_output, run_id="second") as second,
    ):
        handle = _start_test_stage(first)

        with pytest.raises(StageProtocolError, match="does not belong"):
            second.complete(handle)
        assert not (second_output / "transcript.raw.json").exists()

        expected_checkpoint = handle.checkpoint_path
        handle.checkpoint_path = first_output / "wrong.json"
        with pytest.raises(StageProtocolError, match="does not match"):
            first.complete(handle)
        handle.checkpoint_path = expected_checkpoint

        completion = first.complete(handle)
        assert completion.checkpoint_path == first_output / "transcript.raw.json"


def test_promotion_preserves_mode_and_repeated_completion_is_stable(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    public = output / "transcript.raw.json"
    transcript.write_raw_transcript_checkpoint(
        [transcript.TranscriptSegment(0.0, 1.0, "previous")],
        public,
    )
    public.chmod(0o600)

    with StageSupervisor(output, run_id="mode") as supervisor:
        handle = _start_test_stage(supervisor)

        first = supervisor.complete(handle)
        second = supervisor.complete(handle)

        assert first == second
        assert handle.wait() == first
        assert stat.S_IMODE(public.stat().st_mode) == 0o600


def test_interrupted_completed_promotion_is_reconciled_on_retry(tmp_path, monkeypatch):
    output = tmp_path / "output"
    original_promote = supervisor_module.atomic_promote_file

    def promote_then_interrupt(staged_path, public_path):
        original_promote(staged_path, public_path)
        raise KeyboardInterrupt("interrupted after checkpoint replacement")

    with StageSupervisor(output, run_id="interrupted-promotion") as supervisor:
        handle = _start_test_stage(supervisor)
        monkeypatch.setattr(
            supervisor_module,
            "atomic_promote_file",
            promote_then_interrupt,
        )

        with pytest.raises(KeyboardInterrupt, match="after checkpoint replacement"):
            supervisor.complete(handle)

        assert not handle.checkpoint_path.exists()
        assert (output / "transcript.raw.json").exists()
        monkeypatch.setattr(
            supervisor_module,
            "atomic_promote_file",
            original_promote,
        )

        recovered = supervisor.complete(handle)

        assert recovered.checkpoint_path == output / "transcript.raw.json"
        assert recovered.records == (
            transcript.TranscriptSegment(0.123456789, 1.987654321, "worker"),
        )
        assert handle.wait() == recovered


def test_transcription_worker_keeps_result_on_disk_and_metadata_off_progress_channel(
    tmp_path,
    monkeypatch,
):
    checkpoint = tmp_path / "transcript.raw.json"
    terminal = _FakeTerminal()
    progress = _DroppingProgressQueue()
    cancellation = _FakeEvent()
    runtime_platform = transcript.RuntimePlatform("Linux", "x86_64", None, 6)
    monkeypatch.setattr(transcript, "current_runtime_platform", lambda: runtime_platform)
    monkeypatch.setattr(
        transcript,
        "resolve_transcription_backend",
        lambda *_args, **_kwargs: "whisper",
    )
    monkeypatch.setattr(
        transcript,
        "_extract_with_transcription_backend",
        lambda *_args, **_kwargs: transcript.TranscriptionResult(
            (transcript.TranscriptSegment(0.1, 2.3, "disk backed"),),
            "en",
            {
                "model_repository": "mlx-community/whisper-medium-mlx",
                "model_revision": "immutable-revision",
                "model_resolution_source": "local-hit",
                "model_resolution_seconds": 0.125,
                "mlx_peak_memory_bytes": 123456789,
            },
        ),
    )
    request = TranscriptionWorkerRequest(
        video_path=str(tmp_path / "video.mp4"),
        model_name="medium",
        requested_backend="auto",
        checkpoint_path=str(checkpoint),
    )

    transcription_worker_entry(request, terminal, progress, cancellation)

    assert transcript.read_raw_transcript_checkpoint(checkpoint) == (
        transcript.TranscriptSegment(0.1, 2.3, "disk backed"),
    )
    assert len(terminal.messages) == 1
    terminal_message = terminal.messages[0]
    terminal_metadata = dict(terminal_message.metadata)
    assert terminal_metadata.pop("process_tree_peak_rss_bytes") > 0
    assert terminal_metadata == {
        "language": "en",
        "segment_count": 1,
        "requested_backend": "auto",
        "effective_backend": "whisper",
        "model_repository": "mlx-community/whisper-medium-mlx",
        "model_revision": "immutable-revision",
        "model_resolution_source": "local-hit",
        "model_resolution_seconds": 0.125,
        "mlx_peak_memory_bytes": 123456789,
    }
    assert terminal_message.ended_at > 0
    assert not hasattr(terminal.messages[0], "segments")
    assert terminal.closed
    assert progress.closed
    assert progress.cancelled


def test_diarization_worker_entry_keeps_bulk_result_on_disk(tmp_path, monkeypatch):
    checkpoint = tmp_path / "diarization.json"
    terminal = _FakeTerminal()
    progress = _FakeProgressQueue()
    cancellation = _FakeEvent()
    calls = []

    def fake_detect(video_path, hf_token, *, device):
        calls.append((video_path, hf_token, device))
        return (
            transcript.DiarizationRow(0.1, 2.3, "SPEAKER_00"),
        )

    monkeypatch.setattr(transcript, "_detect_speakers", fake_detect)
    request = DiarizationWorkerRequest(
        video_path=str(tmp_path / "video.mp4"),
        hf_token=" hf_test ",
        checkpoint_path=str(checkpoint),
        device="cpu",
    )

    diarization_worker_entry(request, terminal, progress, cancellation)

    assert transcript.read_diarization_checkpoint(checkpoint) == (
        transcript.DiarizationRow(0.1, 2.3, "SPEAKER_00"),
    )
    assert calls == [(tmp_path / "video.mp4", "hf_test", "cpu")]
    assert len(terminal.messages) == 1
    terminal_message = terminal.messages[0]
    terminal_metadata = dict(terminal_message.metadata)
    assert terminal_metadata.pop("process_tree_peak_rss_bytes") > 0
    assert terminal_metadata == {"row_count": 1, "device": "cpu"}
    assert terminal_message.ended_at > 0


def test_worker_error_has_reliable_terminal_metadata(tmp_path, monkeypatch):
    terminal = _FakeTerminal()
    progress = _FakeProgressQueue()
    cancellation = _FakeEvent()
    monkeypatch.setattr(
        transcript,
        "_detect_speakers",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("model failed")),
    )
    request = DiarizationWorkerRequest(
        video_path=str(tmp_path / "video.mp4"),
        hf_token="hf_test",
        checkpoint_path=str(tmp_path / "diarization.json"),
    )

    with pytest.raises(RuntimeError, match="model failed"):
        diarization_worker_entry(request, terminal, progress, cancellation)

    assert len(terminal.messages) == 1
    assert terminal.messages[0].status == "error"
    assert terminal.messages[0].error_type == "RuntimeError"
    assert terminal.messages[0].error_message == "model failed"
    assert not terminal.messages[0].fallback_eligible


def test_diarization_worker_marks_only_typed_mps_compute_failures_for_fallback(
    tmp_path,
    monkeypatch,
):
    terminal = _FakeTerminal()
    progress = _FakeProgressQueue()
    cancellation = _FakeEvent()
    failure = transcript.MPSDiarizationInferenceError("MPS kernel failed")
    monkeypatch.setattr(
        transcript,
        "_detect_speakers",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
    )
    request = DiarizationWorkerRequest(
        video_path=str(tmp_path / "video.mp4"),
        hf_token="hf_test",
        checkpoint_path=str(tmp_path / "diarization.json"),
        device="mps",
    )

    with pytest.raises(transcript.MPSDiarizationInferenceError):
        diarization_worker_entry(request, terminal, progress, cancellation)

    assert len(terminal.messages) == 1
    assert terminal.messages[0].fallback_eligible
