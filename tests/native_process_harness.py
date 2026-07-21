"""Subprocess harness used by native supervisor integration tests."""

from __future__ import annotations

import argparse
import errno
import os
import sys
import time
from pathlib import Path

from keyframe import transcript
from keyframe.stage_supervisor import (
    OutputDirectoryLockedError,
    StageSupervisor,
    StageTerminal,
    _close_worker_ipc,
    _execute_worker,
)


def blocking_transcription_worker(
    request,
    terminal_send,
    progress_queue,
    _cancellation_event,
):
    """Stage an optional checkpoint, report the PID, and ignore cancellation."""

    try:
        if request["checkpoint_state"] == "after":
            transcript.write_raw_transcript_checkpoint(
                [transcript.TranscriptSegment(0.0, 1.0, "staged")],
                request["checkpoint"],
            )
        Path(request["worker_pid_path"]).write_text(
            str(os.getpid()),
            encoding="ascii",
        )
        while True:
            time.sleep(0.05)
    finally:
        _close_worker_ipc(terminal_send, progress_queue)


def crashing_transcription_worker(
    request,
    terminal_send,
    progress_queue,
    _cancellation_event,
):
    """Crash before or after staging without emitting terminal success."""

    if request["checkpoint_state"] == "after":
        transcript.write_raw_transcript_checkpoint(
            [transcript.TranscriptSegment(0.0, 1.0, "never public")],
            request["checkpoint"],
        )
    os._exit(int(request.get("exitcode", 17)))


def successful_transcription_worker(
    request,
    terminal_send,
    progress_queue,
    _cancellation_event,
):
    """Write one valid checkpoint and emit terminal success."""

    try:
        transcript.write_raw_transcript_checkpoint(
            [transcript.TranscriptSegment(0.0, 1.0, "current")],
            request["checkpoint"],
        )
        terminal_send.send(
            StageTerminal.succeeded(
                "transcription",
                {"record_count": 1, "language": "en"},
            )
        )
    finally:
        _close_worker_ipc(terminal_send, progress_queue)


def disk_full_transcription_worker(
    request,
    terminal_send,
    progress_queue,
    cancellation_event,
):
    """Emit a reliable worker failure that models checkpoint disk exhaustion."""

    def fail_checkpoint():
        raise OSError(errno.ENOSPC, "injected checkpoint disk exhaustion")

    _execute_worker(
        "transcription",
        terminal_send,
        progress_queue,
        cancellation_event,
        fail_checkpoint,
    )


def _wait_for_path(path: Path, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {path}")
        time.sleep(0.01)


def _hold_supervisor(args: argparse.Namespace) -> int:
    output = Path(args.output)
    worker_pid = Path(args.worker_pid)
    ready = Path(args.ready)
    with StageSupervisor(
        output,
        run_id=args.run_id,
        cancellation_grace_seconds=0.05,
    ) as supervisor:
        assert supervisor.staging is not None
        supervisor._start_stage(
            stage="transcription",
            target=blocking_transcription_worker,
            request={
                "checkpoint_state": args.checkpoint_state,
                "checkpoint": str(supervisor.staging.transcript_raw),
                "worker_pid_path": str(worker_pid),
            },
            checkpoint_path=supervisor.staging.transcript_raw,
            validator=transcript.read_raw_transcript_checkpoint,
        )
        _wait_for_path(worker_pid)
        ready.write_text(str(os.getpid()), encoding="ascii")
        while True:
            time.sleep(0.05)


def _probe_supervisor(args: argparse.Namespace) -> int:
    try:
        with StageSupervisor(Path(args.output), run_id=args.run_id):
            pass
    except OutputDirectoryLockedError as exc:
        print(exc, file=sys.stderr)
        return 73
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("hold", "probe"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ready")
    parser.add_argument("--worker-pid")
    parser.add_argument(
        "--checkpoint-state",
        choices=("before", "after"),
        default="before",
    )
    args = parser.parse_args(argv)
    if args.mode == "probe":
        return _probe_supervisor(args)
    if not args.ready or not args.worker_pid:
        parser.error("hold mode requires --ready and --worker-pid")
    return _hold_supervisor(args)


if __name__ == "__main__":
    raise SystemExit(main())
