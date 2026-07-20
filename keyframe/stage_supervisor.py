"""Spawned transcript-stage workers with disk-backed results and strict cleanup."""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import shutil
import signal
import threading
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from keyframe import transcript as transcript_module
from keyframe.artifacts import (
    RunStagingPaths,
    TranscriptCheckpointPaths,
    atomic_promote_file,
    reject_path_aliases,
    run_staging_paths,
    transcript_checkpoint_paths,
)
from keyframe.transcript import (
    DiarizationRow,
    TranscriptSegment,
    read_diarization_checkpoint,
    read_raw_transcript_checkpoint,
)


LOCK_FILENAME = "keyframe-output.lock"
RUN_DIRECTORY_PREFIX = "keyframe-run-"


class StageSupervisorError(RuntimeError):
    """Base class for controlled parent-side stage failures."""


class OutputDirectoryLockedError(StageSupervisorError):
    """Another Keyframe CLI process owns the output directory."""


class StageProtocolError(StageSupervisorError):
    """A worker exited without one valid terminal status message."""


class StageWorkerError(StageSupervisorError):
    """A worker reported failure or exited unsuccessfully."""

    def __init__(
        self,
        stage: str,
        message: str,
        *,
        exitcode: int | None,
        error_type: str | None = None,
    ) -> None:
        self.stage = stage
        self.exitcode = exitcode
        self.error_type = error_type
        prefix = f"{error_type}: " if error_type else ""
        super().__init__(f"{stage} worker failed (status {exitcode}): {prefix}{message}")


class StageCheckpointError(StageSupervisorError):
    """A successful worker did not leave a valid current-run checkpoint."""


class StageWaitTimeout(StageSupervisorError):
    """A stage did not exit before the caller's explicit wait deadline."""


class SupervisorSignal(BaseException):
    """SIGTERM converted to unwinding so the supervisor can join children."""

    def __init__(self, signum: int) -> None:
        self.signum = signum
        super().__init__(f"Keyframe interrupted by signal {signum}")


@dataclass(frozen=True)
class StageProgress:
    stage: str
    event: str
    message: str = ""


@dataclass(frozen=True)
class StageTerminal:
    stage: str
    status: str
    metadata: Mapping[str, Any]
    error_type: str | None = None
    error_message: str | None = None

    @classmethod
    def succeeded(cls, stage: str, metadata: Mapping[str, Any]) -> StageTerminal:
        return cls(stage=stage, status="success", metadata=dict(metadata))

    @classmethod
    def failed(cls, stage: str, exc: BaseException) -> StageTerminal:
        return cls(
            stage=stage,
            status="error",
            metadata={},
            error_type=type(exc).__name__,
            error_message=str(exc),
        )


@dataclass(frozen=True)
class TranscriptionWorkerRequest:
    video_path: str
    model_name: str
    requested_backend: str
    checkpoint_path: str
    final_output_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class DiarizationWorkerRequest:
    video_path: str
    hf_token: str
    checkpoint_path: str
    final_output_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class StageCompletion:
    stage: str
    checkpoint_path: Path
    metadata: Mapping[str, Any]
    records: tuple[TranscriptSegment, ...] | tuple[DiarizationRow, ...]


def emit_stage_progress(progress_queue: Any, event: StageProgress) -> None:
    """Best-effort progress must never delay inference or terminal delivery."""
    try:
        progress_queue.put_nowait(event)
    except (queue.Full, BrokenPipeError, OSError, ValueError):
        pass


def _send_terminal(terminal_send: Any, terminal: StageTerminal) -> None:
    try:
        terminal_send.send(terminal)
    except (BrokenPipeError, EOFError, OSError):
        pass


def _close_worker_ipc(terminal_send: Any, progress_queue: Any) -> None:
    try:
        terminal_send.close()
    except (AttributeError, OSError):
        pass
    try:
        progress_queue.close()
        progress_queue.join_thread()
    except (AttributeError, OSError, ValueError):
        pass


def _execute_worker(
    stage: str,
    terminal_send: Any,
    progress_queue: Any,
    cancellation_event: Any,
    operation: Callable[[], Mapping[str, Any]],
) -> None:
    try:
        emit_stage_progress(progress_queue, StageProgress(stage, "started"))
        if cancellation_event.is_set():
            raise RuntimeError("cancelled before stage start")
        metadata = operation()
        if cancellation_event.is_set():
            raise RuntimeError("cancelled before checkpoint commit")
        _send_terminal(terminal_send, StageTerminal.succeeded(stage, metadata))
        emit_stage_progress(progress_queue, StageProgress(stage, "completed"))
    except BaseException as exc:
        _send_terminal(terminal_send, StageTerminal.failed(stage, exc))
        raise
    finally:
        _close_worker_ipc(terminal_send, progress_queue)


def transcription_worker_entry(
    request: TranscriptionWorkerRequest,
    terminal_send: Any,
    progress_queue: Any,
    cancellation_event: Any,
) -> None:
    """One-shot spawned entry point for transcription and raw checkpoint write."""

    def transcribe() -> Mapping[str, Any]:
        runtime_platform = transcript_module.current_runtime_platform()
        effective_backend = transcript_module.resolve_transcription_backend(
            request.requested_backend,
            runtime_platform,
        )
        emit_stage_progress(
            progress_queue,
            StageProgress("transcription", "inference", effective_backend),
        )
        segments, language = transcript_module._extract_with_transcription_backend(
            Path(request.video_path),
            request.model_name,
            request.requested_backend,
            runtime_platform,
        )
        if cancellation_event.is_set():
            raise RuntimeError("transcription cancelled")
        emit_stage_progress(
            progress_queue,
            StageProgress("transcription", "checkpoint"),
        )
        transcript_module.write_raw_transcript_checkpoint(
            segments,
            request.checkpoint_path,
            final_output_paths=request.final_output_paths,
        )
        metadata: dict[str, Any] = {
            "language": language,
            "segment_count": len(segments),
            "requested_backend": request.requested_backend,
            "effective_backend": effective_backend,
        }
        if effective_backend == "mlx":
            model_spec = transcript_module.MLX_MODEL_SPECS[request.model_name]
            metadata["model_repository"] = model_spec.repository
            metadata["model_revision"] = model_spec.revision
        return metadata

    _execute_worker(
        "transcription",
        terminal_send,
        progress_queue,
        cancellation_event,
        transcribe,
    )


def diarization_worker_entry(
    request: DiarizationWorkerRequest,
    terminal_send: Any,
    progress_queue: Any,
    cancellation_event: Any,
) -> None:
    """One-shot spawned entry point for pyannote and diarization checkpoint write."""

    def diarize() -> Mapping[str, Any]:
        if not request.hf_token.strip():
            raise ValueError("diarization requires a non-empty Hugging Face token")
        emit_stage_progress(
            progress_queue,
            StageProgress("diarization", "inference"),
        )
        rows = transcript_module._detect_speakers(
            Path(request.video_path), request.hf_token.strip()
        )
        if cancellation_event.is_set():
            raise RuntimeError("diarization cancelled")
        emit_stage_progress(
            progress_queue,
            StageProgress("diarization", "checkpoint"),
        )
        transcript_module.write_diarization_checkpoint(
            rows,
            request.checkpoint_path,
            final_output_paths=request.final_output_paths,
        )
        return {"row_count": len(rows)}

    _execute_worker(
        "diarization",
        terminal_send,
        progress_queue,
        cancellation_event,
        diarize,
    )


class OutputDirectoryLock:
    """Non-blocking advisory lock keyed by the resolved output directory."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.path = self.output_dir / LOCK_FILENAME
        self._descriptor: int | None = None

    def acquire(self) -> None:
        if self._descriptor is not None:
            raise RuntimeError("output directory lock is already held")
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            self._acquire_descriptor(descriptor)
        except BaseException:
            os.close(descriptor)
            raise
        self._descriptor = descriptor

    def _acquire_descriptor(self, descriptor: int) -> None:
        if os.name == "nt":
            import msvcrt

            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"\0")
            os.lseek(descriptor, 0, os.SEEK_SET)
            try:
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise OutputDirectoryLockedError(
                    f"output directory is already in use: {self.output_dir}"
                ) from exc
            return

        import fcntl

        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise OutputDirectoryLockedError(
                f"output directory is already in use: {self.output_dir}"
            ) from exc

    def release(self) -> None:
        descriptor = self._descriptor
        if descriptor is None:
            return
        self._descriptor = None
        try:
            if os.name == "nt":
                import msvcrt

                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def __enter__(self) -> OutputDirectoryLock:
        self.acquire()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.release()


class StageHandle:
    """Parent-side lifecycle and protocol state for one disposable worker."""

    def __init__(
        self,
        *,
        stage: str,
        process: Any,
        terminal_receive: Any,
        progress_queue: Any,
        cancellation_event: Any,
        checkpoint_path: Path,
        validator: Callable[[Path], tuple[Any, ...]],
        progress_callback: Callable[[StageProgress], None] | None,
    ) -> None:
        self.stage = stage
        self.process = process
        self.checkpoint_path = checkpoint_path
        self._terminal_receive = terminal_receive
        self._progress_queue = progress_queue
        self._cancellation_event = cancellation_event
        self._validator = validator
        self._progress_callback = progress_callback
        self._terminal_messages: list[Any] = []
        self._progress_callback_errors: list[Exception] = []
        self._monitor_done = threading.Event()
        self._ipc_closed = False
        self._completion: StageCompletion | None = None
        self._failure: BaseException | None = None
        self._monitor_thread = threading.Thread(
            target=self._monitor,
            name=f"keyframe-{stage}-monitor",
            daemon=True,
        )
        self._monitor_thread.start()

    def _drain_progress(self) -> None:
        while True:
            try:
                event = self._progress_queue.get_nowait()
            except (queue.Empty, EOFError, OSError, ValueError):
                return
            if isinstance(event, StageProgress) and self._progress_callback is not None:
                try:
                    self._progress_callback(event)
                except Exception as exc:
                    self._progress_callback_errors.append(exc)

    def _drain_terminal(self) -> None:
        while True:
            try:
                if not self._terminal_receive.poll(0):
                    return
                self._terminal_messages.append(self._terminal_receive.recv())
            except (EOFError, OSError):
                return

    def _monitor(self) -> None:
        try:
            while self.process.is_alive():
                self._drain_progress()
                self._drain_terminal()
                try:
                    self._terminal_receive.poll(0.05)
                except (OSError, ValueError):
                    break
            self.process.join()
            self._drain_terminal()
            try:
                event = self._progress_queue.get(timeout=0.05)
            except (queue.Empty, EOFError, OSError, ValueError):
                pass
            else:
                if isinstance(event, StageProgress) and self._progress_callback is not None:
                    try:
                        self._progress_callback(event)
                    except Exception as exc:
                        self._progress_callback_errors.append(exc)
            self._drain_progress()
        finally:
            self._monitor_done.set()

    def _close_ipc(self) -> None:
        if self._ipc_closed:
            return
        self._ipc_closed = True
        try:
            self._terminal_receive.close()
        except (OSError, AttributeError):
            pass
        try:
            self._progress_queue.close()
            self._progress_queue.join_thread()
        except (OSError, AttributeError, ValueError):
            pass

    def wait(self, timeout: float | None = None) -> StageCompletion:
        if self._completion is not None:
            return self._completion
        if self._failure is not None:
            raise self._failure

        self.process.join(timeout)
        if self.process.is_alive():
            raise StageWaitTimeout(
                f"{self.stage} worker did not exit within {float(timeout):.1f}s"
            )
        self._monitor_done.wait(timeout=2.0)
        self._monitor_thread.join(timeout=0)
        self._close_ipc()

        try:
            completion = self._validated_completion()
        except BaseException as exc:
            self._failure = exc
            raise
        self._completion = completion
        return completion

    def _validated_completion(self) -> StageCompletion:
        exitcode = self.process.exitcode
        error_terminal = next(
            (
                message
                for message in self._terminal_messages
                if isinstance(message, StageTerminal) and message.status == "error"
            ),
            None,
        )
        if exitcode != 0:
            raise StageWorkerError(
                self.stage,
                error_terminal.error_message if error_terminal else "exited unexpectedly",
                exitcode=exitcode,
                error_type=error_terminal.error_type if error_terminal else None,
            )
        if len(self._terminal_messages) != 1:
            raise StageProtocolError(
                f"{self.stage} worker emitted {len(self._terminal_messages)} terminal messages"
            )
        terminal = self._terminal_messages[0]
        if not isinstance(terminal, StageTerminal) or terminal.stage != self.stage:
            raise StageProtocolError(f"{self.stage} worker emitted an invalid terminal message")
        if terminal.status != "success":
            raise StageWorkerError(
                self.stage,
                terminal.error_message or "worker reported failure",
                exitcode=exitcode,
                error_type=terminal.error_type,
            )
        try:
            records = self._validator(self.checkpoint_path)
        except Exception as exc:
            raise StageCheckpointError(
                f"{self.stage} worker checkpoint is invalid: {exc}"
            ) from exc
        return StageCompletion(
            stage=self.stage,
            checkpoint_path=self.checkpoint_path,
            metadata=dict(terminal.metadata),
            records=records,
        )

    def cancel(self, grace_seconds: float = 1.0) -> None:
        if self.process.is_alive():
            self._cancellation_event.set()
            self.process.join(max(0.0, grace_seconds))
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(max(0.0, grace_seconds))
        if self.process.is_alive():
            self.process.kill()
            self.process.join()
        else:
            self.process.join()
        self._monitor_done.wait(timeout=max(2.0, grace_seconds))
        self._monitor_thread.join(timeout=0)
        self._close_ipc()


class StageSupervisor:
    """Own the output lock, run directory, spawned workers, and promotions."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        progress_callback: Callable[[StageProgress], None] | None = None,
        progress_capacity: int = 32,
        cancellation_grace_seconds: float = 1.0,
        run_id: str | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.progress_callback = progress_callback
        self.progress_capacity = max(1, int(progress_capacity))
        self.cancellation_grace_seconds = max(0.0, float(cancellation_grace_seconds))
        self.run_id = run_id or uuid.uuid4().hex
        self.context = mp.get_context("spawn")
        self.lock: OutputDirectoryLock | None = None
        self.staging: RunStagingPaths | None = None
        self.public: TranscriptCheckpointPaths | None = None
        self._handles: list[StageHandle] = []
        self._entered = False
        self._previous_sigterm_handler: Any = None

    def __enter__(self) -> StageSupervisor:
        if self._entered:
            raise RuntimeError("stage supervisor cannot be entered twice")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = self.output_dir.resolve()
        self.lock = OutputDirectoryLock(self.output_dir)
        self.lock.acquire()
        try:
            self._cleanup_stale_runs()
            self.staging = run_staging_paths(self.output_dir, self.run_id)
            self.staging.root.mkdir()
            self.public = transcript_checkpoint_paths(self.output_dir)
            self._install_sigterm_handler()
            self._entered = True
            return self
        except BaseException:
            self._restore_sigterm_handler()
            if self.staging is not None and self.staging.root.exists():
                shutil.rmtree(self.staging.root)
            self.lock.release()
            raise

    def _install_sigterm_handler(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        self._previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, self._handle_sigterm)

    def _restore_sigterm_handler(self) -> None:
        if self._previous_sigterm_handler is None:
            return
        signal.signal(signal.SIGTERM, self._previous_sigterm_handler)
        self._previous_sigterm_handler = None

    @staticmethod
    def _handle_sigterm(signum: int, _frame: Any) -> None:
        raise SupervisorSignal(signum)

    def _cleanup_stale_runs(self) -> None:
        for candidate in self.output_dir.iterdir():
            if (
                candidate.name.startswith(RUN_DIRECTORY_PREFIX)
                and candidate.is_dir()
                and not candidate.is_symlink()
            ):
                shutil.rmtree(candidate)

    def _require_entered(self) -> tuple[RunStagingPaths, TranscriptCheckpointPaths]:
        if not self._entered or self.staging is None or self.public is None:
            raise RuntimeError("stage supervisor must be entered before starting workers")
        return self.staging, self.public

    def _start_stage(
        self,
        *,
        stage: str,
        target: Callable[..., None],
        request: Any,
        checkpoint_path: Path,
        validator: Callable[[Path], tuple[Any, ...]],
    ) -> StageHandle:
        self._require_entered()
        terminal_receive, terminal_send = self.context.Pipe(duplex=False)
        progress_queue = self.context.Queue(maxsize=self.progress_capacity)
        cancellation_event = self.context.Event()
        process = self.context.Process(
            target=target,
            args=(request, terminal_send, progress_queue, cancellation_event),
            name=f"keyframe-{stage}-{self.run_id[:8]}",
            daemon=False,
        )
        try:
            process.start()
        except BaseException:
            terminal_receive.close()
            terminal_send.close()
            progress_queue.close()
            progress_queue.join_thread()
            raise
        terminal_send.close()
        handle = StageHandle(
            stage=stage,
            process=process,
            terminal_receive=terminal_receive,
            progress_queue=progress_queue,
            cancellation_event=cancellation_event,
            checkpoint_path=checkpoint_path,
            validator=validator,
            progress_callback=self.progress_callback,
        )
        self._handles.append(handle)
        return handle

    def start_transcription(
        self,
        video_path: str | Path,
        *,
        model_name: str,
        requested_backend: str,
        final_output_paths: Iterable[str | Path] = (),
    ) -> StageHandle:
        staging, public = self._require_entered()
        final_paths = tuple(str(Path(path)) for path in final_output_paths)
        reject_path_aliases(public.transcript_raw, final_paths)
        request = TranscriptionWorkerRequest(
            video_path=str(Path(video_path)),
            model_name=model_name,
            requested_backend=requested_backend,
            checkpoint_path=str(staging.transcript_raw),
            final_output_paths=final_paths,
        )
        return self._start_stage(
            stage="transcription",
            target=transcription_worker_entry,
            request=request,
            checkpoint_path=staging.transcript_raw,
            validator=lambda path: read_raw_transcript_checkpoint(
                path,
                final_output_paths=final_paths,
            ),
        )

    def start_diarization(
        self,
        video_path: str | Path,
        *,
        hf_token: str,
        final_output_paths: Iterable[str | Path] = (),
    ) -> StageHandle:
        staging, public = self._require_entered()
        final_paths = tuple(str(Path(path)) for path in final_output_paths)
        reject_path_aliases(public.diarization, final_paths)
        request = DiarizationWorkerRequest(
            video_path=str(Path(video_path)),
            hf_token=hf_token,
            checkpoint_path=str(staging.diarization),
            final_output_paths=final_paths,
        )
        return self._start_stage(
            stage="diarization",
            target=diarization_worker_entry,
            request=request,
            checkpoint_path=staging.diarization,
            validator=lambda path: read_diarization_checkpoint(
                path,
                final_output_paths=final_paths,
            ),
        )

    def complete(self, handle: StageHandle) -> StageCompletion:
        _staging, public = self._require_entered()
        completion = handle.wait()
        if completion.stage == "transcription":
            public_path = public.transcript_raw
        elif completion.stage == "diarization":
            public_path = public.diarization
        else:
            raise StageProtocolError(f"unknown stage cannot be promoted: {completion.stage}")
        atomic_promote_file(completion.checkpoint_path, public_path)
        return replace(completion, checkpoint_path=public_path)

    def cancel(self, handle: StageHandle) -> None:
        handle.cancel(self.cancellation_grace_seconds)

    def close(self) -> None:
        if not self._entered:
            return
        self._entered = False
        first_error: BaseException | None = None
        try:
            for handle in self._handles:
                try:
                    handle.cancel(self.cancellation_grace_seconds)
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
            if self.staging is not None and self.staging.root.exists():
                try:
                    shutil.rmtree(self.staging.root)
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
        finally:
            self._restore_sigterm_handler()
            if self.lock is not None:
                self.lock.release()
        if first_error is not None:
            raise first_error

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()
