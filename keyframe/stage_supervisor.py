"""Spawned transcript-stage workers with disk-backed results and strict cleanup."""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import queue
import signal
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, replace
from multiprocessing.connection import wait as wait_for_connections
from pathlib import Path
from types import MappingProxyType
from typing import Any

from keyframe import transcript as transcript_module
from keyframe.artifacts import (
    RunStagingPaths,
    TranscriptCheckpointPaths,
    atomic_promote_file,
    reject_path_aliases,
    transcript_checkpoint_paths,
)
from keyframe.managed_workspace import ManagedWorkspace
from keyframe.output_session import (
    OutputDirectoryLock,
    OutputSessionError,
    workspace_entry_id,
)
from keyframe.output_session import (
    OutputDirectoryLockedError as OutputDirectoryLockedError,
)
from keyframe.process_memory import process_tree_high_water_rss_bytes
from keyframe.stage_scheduler import configure_worker_thread_budget
from keyframe.transcript import (
    DiarizationRow,
    TranscriptSegment,
    read_diarization_checkpoint,
    read_raw_transcript_checkpoint,
)


class StageSupervisorError(OutputSessionError):
    """Base class for controlled parent-side stage failures."""

    pipeline_evidence: Any | None = None


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
        fallback_eligible: bool = False,
    ) -> None:
        self.stage = stage
        self.exitcode = exitcode
        self.error_type = error_type
        self.fallback_eligible = bool(fallback_eligible)
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
    ended_at: float
    error_type: str | None = None
    error_message: str | None = None
    fallback_eligible: bool = False

    @classmethod
    def succeeded(
        cls,
        stage: str,
        metadata: Mapping[str, Any],
        *,
        ended_at: float | None = None,
    ) -> StageTerminal:
        return cls(
            stage=stage,
            status="success",
            metadata=dict(metadata),
            ended_at=time.monotonic() if ended_at is None else float(ended_at),
        )

    @classmethod
    def failed(
        cls,
        stage: str,
        exc: BaseException,
        *,
        fallback_eligible: bool = False,
        ended_at: float | None = None,
    ) -> StageTerminal:
        return cls(
            stage=stage,
            status="error",
            metadata={},
            ended_at=time.monotonic() if ended_at is None else float(ended_at),
            error_type=type(exc).__name__,
            error_message=str(exc),
            fallback_eligible=fallback_eligible,
        )


@dataclass(frozen=True)
class TranscriptionWorkerRequest:
    video_path: str
    model_name: str
    requested_backend: str
    checkpoint_path: str
    final_output_paths: tuple[str, ...] = ()
    thread_budget: int | None = None


@dataclass(frozen=True)
class DiarizationWorkerRequest:
    video_path: str
    hf_token: str
    checkpoint_path: str
    final_output_paths: tuple[str, ...] = ()
    thread_budget: int | None = None
    device: str | None = None


@dataclass(frozen=True)
class StageCompletion:
    stage: str
    checkpoint_path: Path
    metadata: Mapping[str, Any]
    records: tuple[TranscriptSegment, ...] | tuple[DiarizationRow, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class _PromotionAttempt:
    public_path: Path
    staged_device: int
    staged_inode: int


def emit_stage_progress(progress_queue: Any, event: StageProgress) -> None:
    """Best-effort progress must never delay inference or terminal delivery."""
    try:
        progress_queue.put_nowait(event)
    except (queue.Full, BrokenPipeError, OSError, ValueError):
        pass


class _StageProgressStream:
    """Turn line-oriented worker text into lossy, stage-prefixed events."""

    encoding = "utf-8"
    errors = "replace"
    _MAX_FRAGMENT_LENGTH = 4096

    def __init__(self, stage: str, progress_queue: Any) -> None:
        self.stage = stage
        self.progress_queue = progress_queue
        self._buffer = ""
        self._lock = threading.Lock()

    def write(self, value: str) -> int:
        rendered = str(value)
        with self._lock:
            self._buffer += rendered.replace("\r", "\n")
            self._emit_complete_lines()
            while len(self._buffer) >= self._MAX_FRAGMENT_LENGTH:
                fragment = self._buffer[: self._MAX_FRAGMENT_LENGTH]
                self._buffer = self._buffer[self._MAX_FRAGMENT_LENGTH :]
                self._emit(fragment)
        return len(rendered)

    def flush(self) -> None:
        with self._lock:
            if self._buffer:
                self._emit(self._buffer)
                self._buffer = ""

    def isatty(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def _emit_complete_lines(self) -> None:
        lines = self._buffer.split("\n")
        self._buffer = lines.pop()
        for line in lines:
            self._emit(line)

    def _emit(self, line: str) -> None:
        message = line.strip()
        if message:
            emit_stage_progress(
                self.progress_queue,
                StageProgress(self.stage, "output", message),
            )


def _run_with_routed_output(
    stage: str,
    progress_queue: Any,
    operation: Callable[[], Mapping[str, Any]],
) -> Mapping[str, Any]:
    stream = _StageProgressStream(stage, progress_queue)
    try:
        with redirect_stdout(stream), redirect_stderr(stream):
            return operation()
    finally:
        stream.flush()


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
        progress_queue.cancel_join_thread()
    except (AttributeError, OSError, ValueError):
        pass


def _execute_worker(
    stage: str,
    terminal_send: Any,
    progress_queue: Any,
    cancellation_event: Any,
    operation: Callable[[], Mapping[str, Any]],
    fallback_classifier: Callable[[BaseException], bool] | None = None,
) -> None:
    try:
        emit_stage_progress(progress_queue, StageProgress(stage, "started"))
        if cancellation_event.is_set():
            raise RuntimeError("cancelled before stage start")
        metadata = dict(_run_with_routed_output(stage, progress_queue, operation))
        try:
            metadata["process_tree_peak_rss_bytes"] = (
                process_tree_high_water_rss_bytes()
            )
        except OSError:
            # The release benchmark requires this on Darwin, but unsupported
            # platforms must retain functional stage workers.
            pass
        if cancellation_event.is_set():
            raise RuntimeError("cancelled before checkpoint commit")
        _send_terminal(terminal_send, StageTerminal.succeeded(stage, metadata))
        emit_stage_progress(progress_queue, StageProgress(stage, "completed"))
    except BaseException as exc:
        fallback_eligible = False
        if fallback_classifier is not None:
            try:
                fallback_eligible = fallback_classifier(exc)
            except Exception:
                fallback_eligible = False
        _send_terminal(
            terminal_send,
            StageTerminal.failed(
                stage,
                exc,
                fallback_eligible=fallback_eligible,
            ),
        )
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
        configure_worker_thread_budget(
            request.thread_budget,
            torch_threads=effective_backend == "whisper",
        )
        emit_stage_progress(
            progress_queue,
            StageProgress("transcription", "inference", effective_backend),
        )
        result = transcript_module._extract_with_transcription_backend(
            Path(request.video_path),
            request.model_name,
            request.requested_backend,
            runtime_platform,
        )
        segments, language = result.segments, result.language
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
        metadata: dict[str, Any] = dict(result.metadata)
        metadata.update({
            "language": language,
            "segment_count": len(segments),
            "requested_backend": request.requested_backend,
            "effective_backend": effective_backend,
        })
        return metadata

    _execute_worker(
        "transcription",
        terminal_send,
        progress_queue,
        cancellation_event,
        transcribe,
        lambda exc: request.requested_backend == "auto"
        and transcript_module.is_auto_fallback_eligible(exc),
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
        uses_mps = (request.device or "").strip().lower().startswith("mps")
        if uses_mps:
            # PyTorch reads this before its first MPS operation. Keep unsupported
            # kernels visible to the parent so fallback receives a fresh resource
            # admission decision instead of silently borrowing CPU in this worker.
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        configure_worker_thread_budget(request.thread_budget, torch_threads=True)
        emit_stage_progress(
            progress_queue,
            StageProgress(
                "diarization",
                "inference",
                request.device or "auto",
            ),
        )
        if request.device is None:
            rows = transcript_module._detect_speakers(
                Path(request.video_path), request.hf_token.strip()
            )
        else:
            rows = transcript_module._detect_speakers(
                Path(request.video_path),
                request.hf_token.strip(),
                device=request.device,
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
        metadata = {
            "row_count": len(rows),
            "device": request.device or "auto",
        }
        if uses_mps:
            metadata["pytorch_mps_fallback_enabled"] = (
                os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"
            )
        return metadata

    _execute_worker(
        "diarization",
        terminal_send,
        progress_queue,
        cancellation_event,
        diarize,
        transcript_module.is_auto_diarization_fallback_eligible,
    )


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
        thread_budget: int | None,
    ) -> None:
        self.stage = stage
        self.process = process
        self.checkpoint_path = checkpoint_path
        self._terminal_receive = terminal_receive
        self._progress_queue = progress_queue
        self._cancellation_event = cancellation_event
        self._validator = validator
        self._progress_callback = progress_callback
        self.thread_budget = thread_budget
        self._terminal_messages: list[Any] = []
        self._progress_callback_errors: list[Exception] = []
        self._control_done = threading.Event()
        self._progress_done = threading.Event()
        self._callback_done = threading.Event()
        self._progress_stop = threading.Event()
        self._callback_queue: queue.Queue[StageProgress] = queue.Queue(maxsize=1)
        self._ipc_closed = False
        self._completion: StageCompletion | None = None
        self._failure: BaseException | None = None
        self._promotion_attempt: _PromotionAttempt | None = None
        self._process_ended_at: float | None = None
        self._control_thread = threading.Thread(
            target=self._monitor_control,
            name=f"keyframe-{stage}-control",
            daemon=True,
        )
        self._progress_thread = threading.Thread(
            target=self._monitor_progress,
            name=f"keyframe-{stage}-progress",
            daemon=True,
        )
        self._callback_thread = threading.Thread(
            target=self._monitor_callback,
            name=f"keyframe-{stage}-callback",
            daemon=True,
        )

    def start_monitors(self) -> None:
        """Start independent reliable-control and best-effort progress drains."""
        self._control_thread.start()
        self._progress_thread.start()
        if self._progress_callback is not None:
            self._callback_thread.start()

    def _offer_progress(self, event: Any) -> None:
        if (
            not isinstance(event, StageProgress)
            or self._progress_callback is None
            or self._progress_stop.is_set()
        ):
            return
        try:
            self._callback_queue.put_nowait(event)
            return
        except queue.Full:
            pass
        try:
            self._callback_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._callback_queue.put_nowait(event)
        except queue.Full:
            pass

    def _drain_progress(self) -> None:
        while True:
            try:
                event = self._progress_queue.get_nowait()
            except (queue.Empty, EOFError, OSError, ValueError):
                return
            if self._progress_stop.is_set():
                return
            self._offer_progress(event)

    def _drain_terminal(self) -> None:
        while True:
            try:
                if not self._terminal_receive.poll(0):
                    return
                self._terminal_messages.append(self._terminal_receive.recv())
            except (EOFError, OSError):
                return

    def _process_exited(self, timeout: float | None = 0.0) -> bool:
        if self.process.pid is None:
            return False
        return bool(wait_for_connections([self.process.sentinel], timeout=timeout))

    def _monitor_control(self) -> None:
        try:
            while not self._process_exited():
                self._drain_terminal()
                try:
                    self._terminal_receive.poll(0.05)
                except (OSError, ValueError):
                    break
            self._drain_terminal()
        finally:
            if self._process_exited():
                self._process_ended_at = time.monotonic()
            self._control_done.set()

    def _monitor_progress(self) -> None:
        try:
            while not self._process_exited() and not self._progress_stop.is_set():
                try:
                    event = self._progress_queue.get(timeout=0.05)
                except (queue.Empty, EOFError, OSError, ValueError):
                    continue
                self._offer_progress(event)
            if self._progress_stop.is_set():
                return
            try:
                event = self._progress_queue.get(timeout=0.05)
            except (queue.Empty, EOFError, OSError, ValueError):
                pass
            else:
                self._offer_progress(event)
            self._drain_progress()
        finally:
            self._progress_done.set()

    def _monitor_callback(self) -> None:
        callback = self._progress_callback
        if callback is None:
            self._callback_done.set()
            return
        try:
            while not self._progress_stop.is_set():
                try:
                    event = self._callback_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if self._progress_stop.is_set():
                    return
                try:
                    callback(event)
                except Exception as exc:
                    self._progress_callback_errors.append(exc)
        finally:
            self._callback_done.set()

    @staticmethod
    def _thread_started(thread: threading.Thread) -> bool:
        return thread.ident is not None

    def _finish_control_monitor(self) -> None:
        if self._thread_started(self._control_thread):
            self._control_done.wait()
            self._control_thread.join()
        else:
            self._drain_terminal()

    def _finish_progress_monitors(self) -> None:
        if self._thread_started(self._progress_thread):
            self._progress_done.wait(timeout=1.0)
            if self._progress_done.is_set():
                self._progress_thread.join()
        if (
            self._thread_started(self._callback_thread)
            and self._callback_done.is_set()
        ):
            self._callback_thread.join()

    def _close_ipc(self) -> None:
        if self._ipc_closed:
            return
        self._ipc_closed = True
        self._progress_stop.set()
        try:
            self._terminal_receive.close()
        except (OSError, AttributeError):
            pass
        try:
            self._progress_queue.close()
            self._progress_queue.join_thread()
        except (OSError, AttributeError, ValueError):
            pass
        self._finish_progress_monitors()

    def wait(self, timeout: float | None = None) -> StageCompletion:
        if self._completion is not None:
            return self._completion
        if self._failure is not None:
            raise self._failure

        if not self._process_exited(timeout):
            raise StageWaitTimeout(
                f"{self.stage} worker did not exit within {float(timeout):.1f}s"
            )
        if self._process_ended_at is None:
            self._process_ended_at = time.monotonic()
        self.process.join()
        self._finish_control_monitor()
        self._close_ipc()

        try:
            completion = self._validated_completion()
        except BaseException as exc:
            self._failure = exc
            raise
        self._completion = completion
        return completion

    @property
    def ended_at(self) -> float | None:
        """Reliable worker terminal time on the shared monotonic clock."""

        terminals = tuple(
            message
            for message in self._terminal_messages
            if isinstance(message, StageTerminal) and message.stage == self.stage
        )
        if len(terminals) == 1:
            ended_at = terminals[0].ended_at
            if (
                isinstance(ended_at, (int, float))
                and not isinstance(ended_at, bool)
                and math.isfinite(float(ended_at))
                and float(ended_at) >= 0
            ):
                return float(ended_at)
        return self._process_ended_at

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
                fallback_eligible=(
                    error_terminal.fallback_eligible if error_terminal else False
                ),
            )
        if len(self._terminal_messages) != 1:
            raise StageProtocolError(
                f"{self.stage} worker emitted {len(self._terminal_messages)} terminal messages"
            )
        terminal = self._terminal_messages[0]
        if not isinstance(terminal, StageTerminal) or terminal.stage != self.stage:
            raise StageProtocolError(f"{self.stage} worker emitted an invalid terminal message")
        if (
            isinstance(terminal.ended_at, bool)
            or not isinstance(terminal.ended_at, (int, float))
            or not math.isfinite(float(terminal.ended_at))
            or float(terminal.ended_at) < 0
        ):
            raise StageProtocolError(
                f"{self.stage} worker emitted an invalid terminal timestamp"
            )
        if terminal.status != "success":
            raise StageWorkerError(
                self.stage,
                terminal.error_message or "worker reported failure",
                exitcode=exitcode,
                error_type=terminal.error_type,
                fallback_eligible=terminal.fallback_eligible,
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
        if self.process.pid is not None:
            if not self._process_exited():
                self._cancellation_event.set()
                self._process_exited(max(0.0, grace_seconds))
            if not self._process_exited():
                self.process.terminate()
                self._process_exited(max(0.0, grace_seconds))
            if not self._process_exited():
                self.process.kill()
                self._process_exited(None)
            if self._process_ended_at is None:
                self._process_ended_at = time.monotonic()
            self.process.join()
        self._finish_control_monitor()
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
        self.entry_id = workspace_entry_id(run_id)
        self.run_id = str(self.entry_id)
        self.context = mp.get_context("spawn")
        self.lock: OutputDirectoryLock | None = None
        self.workspace: ManagedWorkspace | None = None
        self.staging: RunStagingPaths | None = None
        self.public: TranscriptCheckpointPaths | None = None
        self._handles: list[StageHandle] = []
        self._entered = False
        self._previous_sigterm_handler: Any = None

    def __enter__(self) -> StageSupervisor:
        if self._entered:
            raise RuntimeError("stage supervisor cannot be entered twice")
        run_created = False
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.output_dir = self.output_dir.resolve()
            self.lock = OutputDirectoryLock(self.output_dir)
            self.lock.acquire()
            self.workspace = ManagedWorkspace.open(self.output_dir, self.lock)
            self.staging = self.workspace.create_run(self.entry_id)
            run_created = True
            self.public = transcript_checkpoint_paths(self.output_dir)
            self._install_sigterm_handler()
            self._entered = True
            return self
        except BaseException as exc:
            try:
                try:
                    self._restore_sigterm_handler()
                except BaseException as cleanup_exc:
                    exc.add_note(f"failed to restore SIGTERM handler: {cleanup_exc}")
                if run_created and self.workspace is not None:
                    try:
                        self.workspace.delete_entry("run", self.entry_id)
                    except BaseException as cleanup_exc:
                        exc.add_note(
                            f"failed to remove managed run entry: {cleanup_exc}"
                        )
            finally:
                if self.lock is not None:
                    self.lock.release()
            if isinstance(exc, OSError):
                raise StageSupervisorError(
                    f"failed to initialize output directory {self.output_dir}: {exc}"
                ) from exc
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
        handle: StageHandle | None = None
        try:
            handle = StageHandle(
                stage=stage,
                process=process,
                terminal_receive=terminal_receive,
                progress_queue=progress_queue,
                cancellation_event=cancellation_event,
                checkpoint_path=checkpoint_path,
                validator=validator,
                progress_callback=self.progress_callback,
                thread_budget=getattr(request, "thread_budget", None),
            )
            self._handles.append(handle)
            process.start()
            terminal_send.close()
            handle.start_monitors()
            return handle
        except BaseException:
            if handle is not None:
                self._handles[:] = [owned for owned in self._handles if owned is not handle]
                handle.cancel(self.cancellation_grace_seconds)
            else:
                terminal_receive.close()
                progress_queue.close()
                progress_queue.join_thread()
            try:
                terminal_send.close()
            except (OSError, AttributeError):
                pass
            raise

    def start_transcription(
        self,
        video_path: str | Path,
        *,
        model_name: str,
        requested_backend: str,
        final_output_paths: Iterable[str | Path] = (),
        thread_budget: int | None = None,
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
            thread_budget=thread_budget,
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
        thread_budget: int | None = None,
        device: str | None = None,
    ) -> StageHandle:
        staging, public = self._require_entered()
        final_paths = tuple(str(Path(path)) for path in final_output_paths)
        reject_path_aliases(public.diarization, final_paths)
        request = DiarizationWorkerRequest(
            video_path=str(Path(video_path)),
            hf_token=hf_token,
            checkpoint_path=str(staging.diarization),
            final_output_paths=final_paths,
            thread_budget=thread_budget,
            device=device,
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
        staging, public = self._require_entered()
        if not any(owned is handle for owned in self._handles):
            raise StageProtocolError("stage handle does not belong to this supervisor")
        if handle.stage == "transcription":
            expected_staged_path = staging.transcript_raw
            public_path = public.transcript_raw
        elif handle.stage == "diarization":
            expected_staged_path = staging.diarization
            public_path = public.diarization
        else:
            raise StageProtocolError(f"unknown stage cannot be promoted: {handle.stage}")
        if handle.checkpoint_path != expected_staged_path:
            raise StageProtocolError(
                f"{handle.stage} handle checkpoint does not match this run"
            )
        completion = handle.wait()
        if completion.checkpoint_path == public_path:
            handle._promotion_attempt = None
            return completion
        attempt = handle._promotion_attempt
        if (
            attempt is not None
            and attempt.public_path == public_path
            and not completion.checkpoint_path.exists()
        ):
            try:
                public_stat = public_path.stat()
                public_records = handle._validator(public_path)
            except Exception as exc:
                raise StageCheckpointError(
                    f"{handle.stage} checkpoint promotion could not be reconciled: {exc}"
                ) from exc
            if (
                public_stat.st_dev != attempt.staged_device
                or public_stat.st_ino != attempt.staged_inode
                or public_records != completion.records
            ):
                raise StageCheckpointError(
                    f"{handle.stage} checkpoint promotion could not be reconciled"
                )
            promoted = replace(completion, checkpoint_path=public_path)
            handle._completion = promoted
            handle._promotion_attempt = None
            return promoted
        try:
            staged_stat = completion.checkpoint_path.stat()
        except OSError as exc:
            raise StageCheckpointError(
                f"{handle.stage} staged checkpoint is no longer available: {exc}"
            ) from exc
        handle._promotion_attempt = _PromotionAttempt(
            public_path=public_path,
            staged_device=staged_stat.st_dev,
            staged_inode=staged_stat.st_ino,
        )
        try:
            atomic_promote_file(completion.checkpoint_path, public_path)
        except OSError as exc:
            raise StageCheckpointError(
                f"{handle.stage} checkpoint promotion failed: {exc}"
            ) from exc
        promoted = replace(completion, checkpoint_path=public_path)
        handle._completion = promoted
        handle._promotion_attempt = None
        return promoted

    def cancel(self, handle: StageHandle) -> None:
        if not any(owned is handle for owned in self._handles):
            raise StageProtocolError("stage handle does not belong to this supervisor")
        handle.cancel(self.cancellation_grace_seconds)

    def completed_stage_peak_rss_bytes(self) -> dict[str, int]:
        """Return conservative high-water reports from completed stage workers."""

        peaks: dict[str, int] = {}
        for handle in self._handles:
            completion = handle._completion
            if completion is None:
                continue
            value = completion.metadata.get("process_tree_peak_rss_bytes")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                continue
            peaks[handle.stage] = max(peaks.get(handle.stage, 0), value)
        return peaks

    def completed_stage_metadata(
        self,
        stage: str,
    ) -> tuple[Mapping[str, Any], ...]:
        """Return immutable metadata snapshots for successful stage attempts."""

        snapshots = []
        for handle in self._handles:
            completion = handle._completion
            if handle.stage != stage or completion is None:
                continue
            snapshots.append(MappingProxyType(dict(completion.metadata)))
        return tuple(snapshots)

    def close(self) -> None:
        if not self._entered:
            self._restore_sigterm_handler()
            return
        self._entered = False
        first_error: BaseException | None = None

        def remember_error(exc: BaseException, context: str) -> None:
            nonlocal first_error
            if first_error is None:
                first_error = exc
            else:
                first_error.add_note(f"{context}: {type(exc).__name__}: {exc}")

        try:
            for handle in self._handles:
                try:
                    handle.cancel(self.cancellation_grace_seconds)
                except BaseException as exc:
                    remember_error(exc, f"failed to cancel {handle.stage} worker")
            if self.workspace is not None:
                try:
                    self.workspace.delete_entry("run", self.entry_id)
                except BaseException as exc:
                    remember_error(exc, "failed to remove managed run entry")
        finally:
            try:
                self._restore_sigterm_handler()
            except BaseException as exc:
                remember_error(exc, "failed to restore SIGTERM handler")
            finally:
                if self.lock is not None:
                    try:
                        self.lock.release()
                    except BaseException as exc:
                        remember_error(exc, "failed to release output lock")
        if first_error is not None:
            raise first_error

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()
