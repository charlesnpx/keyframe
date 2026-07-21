import logging
import os
from pathlib import Path

import pytest

from keyframe import transcript
from keyframe.stage_scheduler import (
    ActiveStage,
    DIARIZATION_MEMORY_GIB,
    FRAME_MEMORY_GIB,
    GIB,
    NATIVE_THREAD_ENVIRONMENT,
    RuntimeResources,
    StageScheduler,
    configure_worker_thread_budget,
    complete_transcription_with_auto_fallback,
    diarization_demand,
    frame_demand,
    transcription_demand,
)
from keyframe.stage_supervisor import (
    StageCheckpointError,
    StageSupervisor,
    StageWorkerError,
    TranscriptionWorkerRequest,
    _execute_worker,
    transcription_worker_entry,
)


AMPLE_MEMORY = 64 * GIB


def _resources(cpus=8, memory=AMPLE_MEMORY):
    return RuntimeResources(cpus, memory)


def _scheduler(*, policy="auto", cpus=8, memory=AMPLE_MEMORY):
    return StageScheduler(
        policy,
        resource_probe=lambda: _resources(cpus, memory),
    )


def test_stage_memory_estimates_match_admission_contract():
    assert {
        model: transcription_demand(model, backend="whisper").memory_bytes // GIB
        for model in ("tiny", "base", "small", "medium", "large")
    } == {"tiny": 3, "base": 3, "small": 4, "medium": 6, "large": 11}
    assert diarization_demand("cpu").memory_bytes == DIARIZATION_MEMORY_GIB * GIB
    assert frame_demand("mps").memory_bytes == FRAME_MEMORY_GIB * GIB


@pytest.mark.parametrize(
    ("first", "second", "expected_mode"),
    [
        (transcription_demand("medium", backend="mlx"), diarization_demand("cpu"), "parallel"),
        (transcription_demand("medium", backend="mlx"), frame_demand("mps"), "serial"),
        (
            transcription_demand("medium", backend="whisper", device="cuda"),
            diarization_demand("cpu"),
            "parallel",
        ),
        (
            transcription_demand("medium", backend="whisper", device="cuda"),
            diarization_demand("cuda:0"),
            "serial",
        ),
        (
            transcription_demand("medium", backend="whisper", device="cpu"),
            diarization_demand("cuda"),
            "parallel",
        ),
        (
            transcription_demand("medium", backend="whisper", device="cpu"),
            diarization_demand("cpu"),
            "parallel",
        ),
        (frame_demand("cuda:1"), diarization_demand("cuda:0"), "parallel"),
        (frame_demand("cuda:0"), diarization_demand("cuda"), "serial"),
    ],
)
def test_auto_schedule_covers_device_pairings(first, second, expected_mode):
    decision = _scheduler().decide((first, second))

    assert decision.mode == expected_mode


@pytest.mark.parametrize("cpus", [1, 2, 3])
def test_auto_cpu_transcription_and_diarization_require_four_cpus(cpus):
    decision = _scheduler(cpus=cpus).decide(
        (
            transcription_demand("tiny", backend="whisper", device="cpu"),
            diarization_demand("cpu"),
        )
    )

    assert not decision.parallel
    assert "at least four CPUs" in decision.reason


def test_auto_cpu_transcription_and_diarization_split_thread_budget():
    decision = _scheduler(cpus=9).decide(
        (
            transcription_demand("small", backend="whisper", device="cpu"),
            diarization_demand("cpu"),
        )
    )

    assert decision.parallel
    assert decision.cpu_threads_for("transcription") == 4
    assert decision.cpu_threads_for("diarization") == 4


def test_parallel_mlx_and_cpu_diarization_keeps_diarization_half_width():
    decision = _scheduler(cpus=10).decide(
        (
            transcription_demand("medium", backend="mlx"),
            diarization_demand("cpu"),
        )
    )

    assert decision.parallel
    assert decision.cpu_threads_for("diarization") == 5


def test_auto_cpu_frames_do_not_overlap_cpu_transcript_stages():
    decision = _scheduler(cpus=16).decide(
        (frame_demand("cpu"), diarization_demand("cpu"))
    )

    assert not decision.parallel
    assert "CPU frame work" in decision.reason


def test_explicit_parallel_cannot_force_cpu_frames_over_cpu_diarization(caplog):
    caplog.set_level(logging.WARNING)

    decision = _scheduler(policy="parallel", cpus=16).decide(
        (frame_demand("cpu"), diarization_demand("cpu"))
    )

    assert not decision.parallel
    assert "CPU frame-stage exclusion" in caplog.text


@pytest.mark.parametrize(
    ("available_memory", "parallel"),
    [
        (int(12.5 * GIB) - 1, False),
        (int(12.5 * GIB), True),
        (None, False),
    ],
)
def test_auto_memory_admission_requires_25_percent_headroom(
    available_memory,
    parallel,
):
    decision = _scheduler(memory=available_memory).decide(
        (
            transcription_demand("medium", backend="whisper", device="cpu"),
            diarization_demand("cpu"),
        )
    )

    assert decision.required_memory_bytes == int(12.5 * GIB)
    assert decision.parallel is parallel


def test_serial_policy_uses_full_cpu_budget():
    decision = _scheduler(policy="serial", cpus=7).decide(
        (
            transcription_demand("tiny", backend="mlx"),
            diarization_demand("cpu"),
        )
    )

    assert not decision.parallel
    assert decision.cpu_threads_for("transcription") == 7
    assert decision.cpu_threads_for("diarization") == 7


def test_explicit_parallel_bypasses_cpu_and_memory_with_warning(caplog):
    caplog.set_level(logging.WARNING)

    decision = _scheduler(policy="parallel", cpus=1, memory=None).decide(
        (
            transcription_demand("large", backend="whisper", device="cpu"),
            diarization_demand("cpu"),
        )
    )

    assert decision.parallel
    assert decision.cpu_threads_for("transcription") == 1
    assert decision.warnings
    assert "forces overlap" in caplog.text


def test_explicit_parallel_never_bypasses_shared_accelerator(caplog):
    caplog.set_level(logging.WARNING)

    decision = _scheduler(policy="parallel").decide(
        (
            transcription_demand("medium", backend="mlx"),
            frame_demand("mps"),
        )
    )

    assert not decision.parallel
    assert "cannot override shared-accelerator" in caplog.text


def test_scheduler_reprobes_resources_for_every_decision():
    probes = iter((_resources(2, None), _resources(8, AMPLE_MEMORY)))
    scheduler = StageScheduler(resource_probe=lambda: next(probes))
    stages = (
        transcription_demand("tiny", backend="whisper", device="cpu"),
        diarization_demand("cpu"),
    )

    assert not scheduler.decide(stages).parallel
    assert scheduler.decide(stages).parallel


def test_resource_probe_failure_serializes_auto_and_is_logged(caplog):
    caplog.set_level(logging.WARNING)
    scheduler = StageScheduler(
        resource_probe=lambda: (_ for _ in ()).throw(OSError("probe unavailable"))
    )

    decision = scheduler.decide(
        (
            transcription_demand("tiny", backend="mlx"),
            diarization_demand("cpu"),
        )
    )

    assert not decision.parallel
    assert "available memory could not be determined" in decision.reason
    assert "resource admission probe failed" in caplog.text


def test_worker_thread_budget_sets_native_environment_before_torch_import(
    monkeypatch,
):
    calls = []
    for variable in NATIVE_THREAD_ENVIRONMENT:
        monkeypatch.setenv(variable, "before")

    class FakeTorch:
        @staticmethod
        def set_num_threads(value):
            calls.append(("intra", value))

        @staticmethod
        def set_num_interop_threads(value):
            calls.append(("interop", value))

    def import_torch(name):
        assert name == "torch"
        assert {name: os.environ[name] for name in NATIVE_THREAD_ENVIRONMENT} == {
            name: "3" for name in NATIVE_THREAD_ENVIRONMENT
        }
        calls.append(("import", name))
        return FakeTorch

    monkeypatch.setattr("keyframe.stage_scheduler.importlib.import_module", import_torch)

    configure_worker_thread_budget(3, torch_threads=True)

    assert calls == [("import", "torch"), ("intra", 3), ("interop", 3)]


def test_mlx_worker_budget_does_not_import_torch(monkeypatch):
    for variable in NATIVE_THREAD_ENVIRONMENT:
        monkeypatch.setenv(variable, "before")
    monkeypatch.setattr(
        "keyframe.stage_scheduler.importlib.import_module",
        lambda _name: pytest.fail("MLX workers must not load Torch for thread setup"),
    )

    configure_worker_thread_budget(2, torch_threads=False)

    assert all(os.environ[name] == "2" for name in NATIVE_THREAD_ENVIRONMENT)


class _TerminalCapture:
    def __init__(self):
        self.messages = []

    def send(self, message):
        self.messages.append(message)

    def close(self):
        pass


class _ProgressCapture:
    def put_nowait(self, _message):
        pass

    def close(self):
        pass

    def cancel_join_thread(self):
        pass


class _UnsetCancellation:
    @staticmethod
    def is_set():
        return False


@pytest.mark.parametrize(
    ("requested_backend", "expected_eligible"),
    [("auto", True), ("mlx", False)],
)
def test_transcription_worker_marks_only_auto_mlx_backend_errors_for_fallback(
    tmp_path,
    monkeypatch,
    requested_backend,
    expected_eligible,
):
    terminal = _TerminalCapture()
    monkeypatch.setattr(
        transcript,
        "current_runtime_platform",
        lambda: transcript.RuntimePlatform("Darwin", "arm64", 14, 23),
    )
    monkeypatch.setattr(
        transcript,
        "_extract_with_transcription_backend",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            transcript.MLXInferenceError("forced inference failure")
        ),
    )
    monkeypatch.setattr(
        "keyframe.stage_supervisor.configure_worker_thread_budget",
        lambda *_args, **_kwargs: None,
    )
    request = TranscriptionWorkerRequest(
        video_path=str(tmp_path / "recording.mp4"),
        model_name="tiny",
        requested_backend=requested_backend,
        checkpoint_path=str(tmp_path / "transcript.raw.json"),
    )

    with pytest.raises(transcript.MLXInferenceError):
        transcription_worker_entry(
            request,
            terminal,
            _ProgressCapture(),
            _UnsetCancellation(),
        )

    assert len(terminal.messages) == 1
    assert terminal.messages[0].fallback_eligible is expected_eligible


@pytest.mark.parametrize(
    "failure",
    [transcript.TranscriptionCancelled("cancelled"), OSError("disk full")],
)
def test_cancellation_and_checkpoint_write_failures_are_not_worker_fallbacks(
    tmp_path,
    monkeypatch,
    failure,
):
    terminal = _TerminalCapture()
    monkeypatch.setattr(
        transcript,
        "current_runtime_platform",
        lambda: transcript.RuntimePlatform("Darwin", "arm64", 14, 23),
    )
    monkeypatch.setattr(
        "keyframe.stage_supervisor.configure_worker_thread_budget",
        lambda *_args, **_kwargs: None,
    )
    if isinstance(failure, transcript.TranscriptionCancelled):
        monkeypatch.setattr(
            transcript,
            "_extract_with_transcription_backend",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
        )
    else:
        monkeypatch.setattr(
            transcript,
            "_extract_with_transcription_backend",
            lambda *_args, **_kwargs: transcript.TranscriptionResult(
                (transcript.TranscriptSegment(0.0, 1.0, "done"),),
                "en",
                {},
            ),
        )
        monkeypatch.setattr(
            transcript,
            "write_raw_transcript_checkpoint",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
        )
    request = TranscriptionWorkerRequest(
        video_path=str(tmp_path / "recording.mp4"),
        model_name="tiny",
        requested_backend="auto",
        checkpoint_path=str(tmp_path / "transcript.raw.json"),
    )

    with pytest.raises(type(failure), match=str(failure)):
        transcription_worker_entry(
            request,
            terminal,
            _ProgressCapture(),
            _UnsetCancellation(),
        )

    assert len(terminal.messages) == 1
    assert not terminal.messages[0].fallback_eligible


class _FakeProcess:
    _next_pid = 1000

    def __init__(self, *, alive=False, exitcode=1):
        type(self)._next_pid += 1
        self.pid = type(self)._next_pid
        self._alive = alive
        self.exitcode = None if alive else exitcode

    def is_alive(self):
        return self._alive


class _FakeHandle:
    def __init__(self, *, alive=False, wait_error=None):
        self.process = _FakeProcess(alive=alive)
        self.wait_error = wait_error
        self.waited = False

    def wait(self):
        self.waited = True
        self.process._alive = False
        self.process.exitcode = 0
        if self.wait_error is not None:
            raise self.wait_error
        return "companion"


class _FakeSupervisor:
    def __init__(self, initial, initial_error):
        self.initial = initial
        self.initial_error = initial_error
        self.started = []
        self.events = []

    def complete(self, handle):
        if handle is self.initial:
            self.events.append("initial-complete")
            raise self.initial_error
        self.events.append("fallback-complete")
        return "fallback-completion"

    def start_transcription(self, video_path, **kwargs):
        self.events.append("fallback-start")
        handle = _FakeHandle()
        handle.process.exitcode = 0
        self.started.append((video_path, kwargs, handle))
        return handle


def _eligible_worker_error():
    return StageWorkerError(
        "transcription",
        "MLX allocation failed",
        exitcode=1,
        error_type="MLXModelLoadError",
        fallback_eligible=True,
    )


def test_auto_fallback_reprobes_waits_when_cpu_overlap_is_rejected_and_relaunches():
    initial = _FakeHandle()
    companion = _FakeHandle(alive=True)
    supervisor = _FakeSupervisor(initial, _eligible_worker_error())
    probes = iter((_resources(8, AMPLE_MEMORY), _resources(2, AMPLE_MEMORY)))
    scheduler = StageScheduler(resource_probe=lambda: next(probes))
    initial_schedule = scheduler.decide(
        (transcription_demand("medium", backend="mlx"), diarization_demand("cpu"))
    )

    result = complete_transcription_with_auto_fallback(
        supervisor,
        initial,
        scheduler=scheduler,
        video_path="recording.mp4",
        model_name="medium",
        requested_backend="auto",
        effective_backend="mlx",
        active_stages=(ActiveStage(diarization_demand("cpu"), companion),),
    )

    assert initial_schedule.parallel
    assert initial_schedule.cpu_threads_for("diarization") == 4
    assert companion.waited
    assert supervisor.events == [
        "initial-complete",
        "fallback-start",
        "fallback-complete",
    ]
    assert result.fallback_used
    assert not result.fallback_schedule.parallel
    assert supervisor.started[0][1]["requested_backend"] == "whisper"
    assert supervisor.started[0][1]["thread_budget"] == 2
    assert result.handle is not initial


def test_auto_fallback_overlaps_running_cpu_diarization_when_admitted():
    initial = _FakeHandle()
    companion = _FakeHandle(alive=True)
    supervisor = _FakeSupervisor(initial, _eligible_worker_error())

    result = complete_transcription_with_auto_fallback(
        supervisor,
        initial,
        scheduler=_scheduler(cpus=8, memory=AMPLE_MEMORY),
        video_path="recording.mp4",
        model_name="small",
        requested_backend="auto",
        effective_backend="mlx",
        active_stages=(ActiveStage(diarization_demand("cpu"), companion),),
    )

    assert result.fallback_schedule.parallel
    assert result.fallback_schedule.cpu_threads_for("transcription") == 4
    assert not companion.waited
    assert supervisor.started[0][1]["thread_budget"] == 4


@pytest.mark.parametrize(
    ("requested_backend", "effective_backend", "error"),
    [
        ("mlx", "mlx", _eligible_worker_error()),
        (
            "auto",
            "mlx",
            StageWorkerError(
                "transcription",
                "cancelled",
                exitcode=1,
                error_type="TranscriptionCancelled",
                fallback_eligible=False,
            ),
        ),
    ],
)
def test_ineligible_or_explicit_mlx_failure_never_falls_back(
    requested_backend,
    effective_backend,
    error,
):
    initial = _FakeHandle()
    supervisor = _FakeSupervisor(initial, error)

    with pytest.raises(StageWorkerError) as raised:
        complete_transcription_with_auto_fallback(
            supervisor,
            initial,
            scheduler=_scheduler(),
            video_path="recording.mp4",
            model_name="medium",
            requested_backend=requested_backend,
            effective_backend=effective_backend,
        )

    assert raised.value is error
    assert supervisor.started == []


def test_checkpoint_failure_never_falls_back():
    initial = _FakeHandle()
    error = StageCheckpointError("disk write did not commit")
    supervisor = _FakeSupervisor(initial, error)

    with pytest.raises(StageCheckpointError):
        complete_transcription_with_auto_fallback(
            supervisor,
            initial,
            scheduler=_scheduler(),
            video_path="recording.mp4",
            model_name="medium",
            requested_backend="auto",
            effective_backend="mlx",
        )

    assert supervisor.started == []


def _spawned_mlx_failure(request, terminal_send, progress_queue, cancellation_event):
    def fail_after_allocation():
        Path(request.video_path).write_text(str(os.getpid()), encoding="ascii")
        allocated_model = bytearray(1024 * 1024)
        assert allocated_model
        raise transcript.MLXModelLoadError("forced failure after allocation")

    _execute_worker(
        "transcription",
        terminal_send,
        progress_queue,
        cancellation_event,
        fail_after_allocation,
        transcript.is_auto_fallback_eligible,
    )


def _spawned_whisper_success(request, terminal_send, progress_queue, cancellation_event):
    def succeed():
        Path(request.video_path).write_text(str(os.getpid()), encoding="ascii")
        transcript.write_raw_transcript_checkpoint(
            [transcript.TranscriptSegment(0.0, 1.0, "fresh worker")],
            request.checkpoint_path,
        )
        return {"language": "en", "effective_backend": "whisper"}

    _execute_worker(
        "transcription",
        terminal_send,
        progress_queue,
        cancellation_event,
        succeed,
    )


class _SpawnFallbackSupervisor(StageSupervisor):
    def start_transcription(
        self,
        video_path,
        *,
        model_name,
        requested_backend,
        final_output_paths=(),
        thread_budget=None,
    ):
        staging, _public = self._require_entered()
        request = TranscriptionWorkerRequest(
            video_path=str(video_path),
            model_name=model_name,
            requested_backend=requested_backend,
            checkpoint_path=str(staging.transcript_raw),
            final_output_paths=tuple(str(path) for path in final_output_paths),
            thread_budget=thread_budget,
        )
        return self._start_stage(
            stage="transcription",
            target=_spawned_whisper_success,
            request=request,
            checkpoint_path=staging.transcript_raw,
            validator=transcript.read_raw_transcript_checkpoint,
        )


def test_eligible_mlx_failure_exits_before_fresh_spawned_whisper_worker(tmp_path):
    output = tmp_path / "output"
    mlx_pid_path = tmp_path / "mlx.pid"
    whisper_pid_path = tmp_path / "whisper.pid"

    with _SpawnFallbackSupervisor(output, run_id="fresh-fallback") as supervisor:
        staging, _public = supervisor._require_entered()
        initial_request = TranscriptionWorkerRequest(
            video_path=str(mlx_pid_path),
            model_name="tiny",
            requested_backend="auto",
            checkpoint_path=str(staging.transcript_raw),
            thread_budget=2,
        )
        initial = supervisor._start_stage(
            stage="transcription",
            target=_spawned_mlx_failure,
            request=initial_request,
            checkpoint_path=staging.transcript_raw,
            validator=transcript.read_raw_transcript_checkpoint,
        )

        result = complete_transcription_with_auto_fallback(
            supervisor,
            initial,
            scheduler=_scheduler(cpus=6),
            video_path=whisper_pid_path,
            model_name="tiny",
            requested_backend="auto",
            effective_backend="mlx",
        )

        assert result.fallback_used
        assert result.completion.records == (
            transcript.TranscriptSegment(0.0, 1.0, "fresh worker"),
        )
        assert not initial.process.is_alive()
        assert initial.process.exitcode is not None
        assert int(mlx_pid_path.read_text(encoding="ascii")) == initial.process.pid
        assert int(whisper_pid_path.read_text(encoding="ascii")) == result.handle.process.pid
        assert initial.process.pid != result.handle.process.pid
