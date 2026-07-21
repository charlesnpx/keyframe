from __future__ import annotations

import json
from itertools import count
from pathlib import Path
from types import SimpleNamespace

import pytest

from keyframe import full_pipeline as full_pipeline_module
from keyframe import transcript
from keyframe.artifacts import transcript_checkpoint_paths
from keyframe.full_pipeline import (
    FullPipelineFrameError,
    resolve_frame_device,
    run_supervised_full_pipeline,
)
from keyframe.output_session import remove_keyframe_owned_directory
from keyframe.stage_scheduler import GIB, RuntimeResources, StageScheduler
from keyframe.stage_supervisor import StageCompletion, StageWorkerError
from keyframe.transcript_cli import (
    TranscriptPreflight,
    TranscriptRunConfig,
)


MAC = transcript.RuntimePlatform("Darwin", "arm64", 14, 23)
LINUX = transcript.RuntimePlatform("Linux", "x86_64", None, 6)
AMPLE_MEMORY = 64 * GIB


def _preflight(
    *,
    runtime=MAC,
    backend="mlx",
    requested_backend=None,
    transcription_device="mlx",
    diarization_device="cpu",
    requested_diarization_device=None,
    policy="auto",
    speaker_detection=True,
    fmt="json",
):
    config = TranscriptRunConfig(
        model_name="medium",
        fmt=fmt,
        transcription_backend=requested_backend or backend,
        diarization_device=(
            requested_diarization_device
            or diarization_device
            or "auto"
        ),
        stage_concurrency=policy,
        speaker_detection=speaker_detection,
    )
    return TranscriptPreflight(
        config=config,
        runtime_platform=runtime,
        effective_backend=backend,
        transcription_device=transcription_device,
        hf_token="hf_test" if diarization_device is not None else None,
        effective_diarization_device=diarization_device,
        missing_hf_token=False,
    )


def _scheduler(policy="auto", *, probes=None, cpus=8, memory=AMPLE_MEMORY):
    if probes is None:
        return StageScheduler(
            policy,
            resource_probe=lambda: RuntimeResources(cpus, memory),
        )
    resources = iter(probes)
    return StageScheduler(policy, resource_probe=lambda: next(resources))


class _FakeProcess:
    _pids = count(7000)

    def __init__(self):
        self.pid = next(self._pids)
        self.exitcode = None
        self.alive = True

    def is_alive(self):
        return self.alive


class _FakeHandle:
    def __init__(self, owner, stage, *, attempt=1, requested_backend=None):
        self.owner = owner
        self.stage = stage
        self.attempt = attempt
        self.requested_backend = requested_backend
        self.process = _FakeProcess()
        self.ended_at = None
        self.completion = None
        self.failure = None

    def wait(self):
        return self.owner.complete(self)


class _FakeSupervisor:
    def __init__(
        self,
        output_dir,
        events,
        *,
        segments=None,
        diarization_rows=None,
        transcription_error=None,
        diarization_error=None,
        effective_backend="mlx",
        fail_first_mlx=False,
        fail_first_mps=False,
        mps_exits_before_transcription=False,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.public = transcript_checkpoint_paths(self.output_dir)
        self.events = events
        self.segments = tuple(
            segments
            if segments is not None
            else (transcript.TranscriptSegment(0.0, 2.0, "hello"),)
        )
        self.diarization_rows = tuple(
            diarization_rows
            if diarization_rows is not None
            else (transcript.DiarizationRow(0.0, 2.0, "SPEAKER_00"),)
        )
        self.transcription_error = transcription_error
        self.diarization_error = diarization_error
        self.effective_backend = effective_backend
        self.fail_first_mlx = fail_first_mlx
        self.fail_first_mps = fail_first_mps
        self.mps_exits_before_transcription = mps_exits_before_transcription
        self.transcription_attempts = 0
        self.diarization_attempts = 0
        self.handles = {}

    def start_transcription(self, _video, **kwargs):
        self.transcription_attempts += 1
        handle = _FakeHandle(
            self,
            "transcription",
            attempt=self.transcription_attempts,
            requested_backend=kwargs["requested_backend"],
        )
        self.handles["transcription"] = handle
        self.events.append(("start-transcription", kwargs))
        return handle

    def start_diarization(self, _video, **kwargs):
        self.diarization_attempts += 1
        handle = _FakeHandle(
            self,
            "diarization",
            attempt=self.diarization_attempts,
        )
        self.handles["diarization"] = handle
        self.events.append(("start-diarization", kwargs))
        return handle

    def complete(self, handle):
        if handle.failure is not None:
            raise handle.failure
        if handle.completion is not None:
            return handle.completion
        self.events.append((f"complete-{handle.stage}", None))
        handle.process.alive = False
        if handle.stage == "transcription":
            if self.fail_first_mlx and handle.attempt == 1:
                handle.process.exitcode = 1
                handle.failure = StageWorkerError(
                    "transcription",
                    "injected MLX load failure",
                    exitcode=1,
                    error_type="MLXModelLoadError",
                    fallback_eligible=True,
                )
                raise handle.failure
            if self.transcription_error is not None:
                handle.process.exitcode = 1
                handle.failure = self.transcription_error
                raise handle.failure
            handle.process.exitcode = 0
            transcript.write_raw_transcript_checkpoint(
                self.segments,
                self.public.transcript_raw,
            )
            effective_backend = (
                "whisper"
                if handle.requested_backend == "whisper"
                else self.effective_backend
            )
            metadata = {
                "language": "en",
                "effective_backend": effective_backend,
            }
            if effective_backend == "mlx":
                metadata.update(
                    {
                        "model_repository": "mlx-community/whisper-medium-mlx",
                        "model_revision": "immutable-revision",
                        "model_resolution_source": "local-hit",
                        "model_resolution_seconds": 0.125,
                    }
                )
            handle.completion = StageCompletion(
                "transcription",
                self.public.transcript_raw,
                metadata,
                self.segments,
            )
            if self.mps_exits_before_transcription:
                diarization = self.handles.get("diarization")
                if diarization is not None:
                    diarization.process.alive = False
                    diarization.process.exitcode = 1
        else:
            if self.fail_first_mps and handle.attempt == 1:
                handle.process.exitcode = 1
                handle.failure = StageWorkerError(
                    "diarization",
                    "injected MPS inference failure",
                    exitcode=1,
                    error_type="MPSDiarizationInferenceError",
                    fallback_eligible=True,
                )
                raise handle.failure
            if self.diarization_error is not None:
                handle.process.exitcode = 1
                handle.failure = self.diarization_error
                raise handle.failure
            handle.process.exitcode = 0
            transcript.write_diarization_checkpoint(
                self.diarization_rows,
                self.public.diarization,
            )
            handle.completion = StageCompletion(
                "diarization",
                self.public.diarization,
                {"row_count": len(self.diarization_rows)},
                self.diarization_rows,
            )
        return handle.completion

    def cancel(self, handle):
        self.events.append((f"cancel-{handle.stage}", None))
        handle.process.alive = False
        handle.process.exitcode = -15


class _ContextFakeSupervisor(_FakeSupervisor):
    def __init__(self, output_dir, events, **kwargs):
        super().__init__(output_dir, events, **kwargs)
        self.staging_root = self.output_dir / "keyframe-run-interruption"

    def __enter__(self):
        self.staging_root.mkdir()
        self.events.append(("enter-supervisor", None))
        return self

    def __exit__(self, *_args):
        for handle in self.handles.values():
            if handle.process.is_alive():
                self.cancel(handle)
        remove_keyframe_owned_directory(self.staging_root)
        self.events.append(("close-supervisor", None))


class _FakeFrameGeneration:
    def __init__(self, output_dir, events):
        self.output_dir = Path(output_dir)
        self.events = events
        self.enriched_segments = None

    def enrich_manifest(self, segments):
        self.events.append(("enrich-manifest", tuple(segments)))
        assert (self.output_dir / "transcript.json").exists()
        self.enriched_segments = tuple(segments)

    def promote(self):
        self.events.append(("promote-frames", None))
        assert self.enriched_segments is not None
        public = self.output_dir / "frames"
        public.mkdir(parents=True, exist_ok=True)
        (public / "current.txt").write_text("current", encoding="utf-8")
        return SimpleNamespace(final_frame_count=1, output_dir=public)


def _event_names(events):
    return [name for name, _detail in events]


def _frame_runner(supervisor, output_dir, events, *, error=None):
    def run():
        assert supervisor.public.transcript_raw.exists()
        diarization = supervisor.handles.get("diarization")
        events.append(
            (
                "start-frames",
                diarization is not None and diarization.process.is_alive(),
            )
        )
        if error is not None:
            events.append(("fail-frames", None))
            raise error
        events.append(("finish-frames", None))
        return _FakeFrameGeneration(output_dir, events)

    return run


@pytest.mark.parametrize(
    ("preflight", "expected"),
    [
        (_preflight(), "mps"),
        (
            _preflight(
                runtime=LINUX,
                backend="whisper",
                transcription_device="cuda",
            ),
            "cuda",
        ),
        (
            _preflight(
                runtime=LINUX,
                backend="whisper",
                transcription_device="cpu",
                diarization_device=None,
                speaker_detection=False,
            ),
            "cpu",
        ),
    ],
)
def test_frame_device_resolution_matches_the_platform_execution_backend(
    preflight,
    expected,
):
    assert resolve_frame_device(preflight) == expected


@pytest.mark.parametrize(
    ("preflight", "frame_device"),
    [
        (_preflight(), "mps"),
        (
            _preflight(
                runtime=LINUX,
                backend="whisper",
                transcription_device="cuda",
            ),
            "cuda",
        ),
    ],
)
def test_accelerated_frames_overlap_running_cpu_diarization_after_transcription(
    tmp_path,
    monkeypatch,
    preflight,
    frame_device,
):
    events = []
    output = tmp_path / "out"
    supervisor = _FakeSupervisor(
        output,
        events,
        effective_backend=preflight.effective_backend,
    )
    original_assign = transcript._assign_speakers

    def assign(segments, rows):
        events.append(("merge-speakers", None))
        return original_assign(segments, rows)

    monkeypatch.setattr(transcript, "_assign_speakers", assign)
    result = run_supervised_full_pipeline(
        tmp_path / "recording.mp4",
        output,
        preflight,
        supervisor=supervisor,
        frame_device=frame_device,
        frame_runner=_frame_runner(supervisor, output, events),
        scheduler=_scheduler(),
    )

    names = _event_names(events)
    assert names.index("start-diarization") < names.index("complete-transcription")
    assert events[names.index("start-frames")][1] is True
    assert names.index("finish-frames") < names.index("complete-diarization")
    assert names.index("complete-diarization") < names.index("merge-speakers")
    assert names.index("merge-speakers") < names.index("enrich-manifest")
    assert names.index("enrich-manifest") < names.index("promote-frames")
    enriched = events[names.index("enrich-manifest")][1]
    assert enriched[0].speaker == "SPEAKER_00"
    assert result.frame_device == frame_device
    assert result.initial_schedule.parallel
    assert result.frame_schedule.parallel
    assert result.critical_path == "max(T + F, D) + M + E"
    assert result.transcript.segments[0].speaker == "SPEAKER_00"
    if preflight.effective_backend == "mlx":
        assert result.transcript.metadata["model_resolution_source"] == "local-hit"
    else:
        assert "model_resolution_source" not in result.transcript.metadata
    assert result.transcription_metadata is result.transcript.metadata
    transcription_interval = result.pipeline_evidence.interval("transcription")
    diarization_interval = result.pipeline_evidence.interval("diarization")
    frame_interval = result.pipeline_evidence.interval("frames")
    assert transcription_interval.launch_wave == "initial"
    assert diarization_interval.launch_wave == "initial"
    assert frame_interval.launch_wave == "post-transcription"
    assert transcription_interval.ended_at <= frame_interval.started_at
    assert diarization_interval.overlaps(frame_interval)


def test_diarization_exit_before_frame_start_does_not_fake_second_wave_overlap(
    tmp_path,
):
    events = []
    output = tmp_path / "out"
    supervisor = _FakeSupervisor(output, events)
    ticks = iter((0.0, 0.0, 2.0, 4.0, 6.0, 7.0, 8.0, 9.0, 10.0))

    def frame_runner():
        diarization = supervisor.handles["diarization"]
        # The worker was alive at admission, then exited before frame inference.
        # Settlement happens after frames, so only its reliable terminal time can
        # distinguish this race from real overlap.
        diarization.ended_at = 3.0
        diarization.process.alive = False
        return _frame_runner(supervisor, output, events)()

    result = run_supervised_full_pipeline(
        tmp_path / "recording.mp4",
        output,
        _preflight(),
        supervisor=supervisor,
        frame_device="mps",
        frame_runner=frame_runner,
        scheduler=_scheduler(),
        clock=lambda: next(ticks),
    )

    diarization_interval = result.pipeline_evidence.interval("diarization")
    frame_interval = result.pipeline_evidence.interval("frames")
    assert diarization_interval.ended_at == 3.0
    assert frame_interval.started_at == 4.0
    assert not diarization_interval.overlaps(frame_interval)
    assert result.critical_path == "max(T, D) + F + M + E"


def test_initial_serial_run_re_admits_post_transcription_diarization_and_frames(
    tmp_path,
    monkeypatch,
):
    events = []
    frame_budget_calls = []
    output = tmp_path / "out"
    supervisor = _FakeSupervisor(output, events)
    probes = (
        RuntimeResources(10, 11 * GIB - 1),
        RuntimeResources(8, 11 * GIB),
    )
    monkeypatch.setattr(
        full_pipeline_module,
        "configure_worker_thread_budget",
        lambda budget, *, torch_threads: frame_budget_calls.append(
            (budget, torch_threads)
        ),
    )

    result = run_supervised_full_pipeline(
        tmp_path / "recording.mp4",
        output,
        _preflight(),
        supervisor=supervisor,
        frame_device="mps",
        frame_runner=_frame_runner(supervisor, output, events),
        scheduler=_scheduler(probes=probes),
    )

    names = _event_names(events)
    assert not result.initial_schedule.parallel
    assert result.frame_schedule.parallel
    assert names.index("complete-transcription") < names.index("start-diarization")
    assert events[names.index("start-frames")][1] is True
    assert names.index("finish-frames") < names.index("complete-diarization")
    diarization_start = next(
        detail for name, detail in events if name == "start-diarization"
    )
    assert diarization_start["thread_budget"] == 4
    assert frame_budget_calls == [(4, True)]
    diarization_interval = result.pipeline_evidence.interval("diarization")
    frame_interval = result.pipeline_evidence.interval("frames")
    assert diarization_interval.launch_wave == "post-transcription"
    assert diarization_interval.overlaps(frame_interval)
    assert result.critical_path == "T + max(D, F) + M + E"


def test_running_diarization_keeps_initial_budget_while_frames_use_second_wave_budget(
    tmp_path,
    monkeypatch,
):
    events = []
    frame_budget_calls = []
    output = tmp_path / "out"
    supervisor = _FakeSupervisor(output, events)
    probes = (
        RuntimeResources(10, AMPLE_MEMORY),
        RuntimeResources(8, AMPLE_MEMORY),
    )
    monkeypatch.setattr(
        full_pipeline_module,
        "configure_worker_thread_budget",
        lambda budget, *, torch_threads: frame_budget_calls.append(
            (budget, torch_threads)
        ),
    )

    result = run_supervised_full_pipeline(
        tmp_path / "recording.mp4",
        output,
        _preflight(),
        supervisor=supervisor,
        frame_device="mps",
        frame_runner=_frame_runner(supervisor, output, events),
        scheduler=_scheduler(probes=probes),
    )

    diarization_start = next(
        detail for name, detail in events if name == "start-diarization"
    )
    assert diarization_start["thread_budget"] == 5
    assert result.frame_schedule.cpu_threads_for("diarization") == 4
    assert frame_budget_calls == [(4, True)]


def test_shared_cuda_stages_remain_serial_through_diarization_and_frames(tmp_path):
    events = []
    output = tmp_path / "out"
    preflight = _preflight(
        runtime=LINUX,
        backend="whisper",
        transcription_device="cuda",
        diarization_device="cuda",
        policy="parallel",
    )
    supervisor = _FakeSupervisor(output, events, effective_backend="whisper")

    result = run_supervised_full_pipeline(
        tmp_path / "recording.mp4",
        output,
        preflight,
        supervisor=supervisor,
        frame_device="cuda",
        frame_runner=_frame_runner(supervisor, output, events),
        scheduler=_scheduler("parallel"),
    )

    names = _event_names(events)
    assert names.index("complete-transcription") < names.index("start-diarization")
    assert names.index("complete-diarization") < names.index("start-frames")
    assert result.critical_path == "T + D + F + M + E"
    assert not result.initial_schedule.parallel
    assert not result.pipeline_evidence.interval("diarization").overlaps(
        result.pipeline_evidence.interval("frames")
    )


def test_automatic_mps_diarization_serializes_between_mlx_and_mps_frames(
    tmp_path,
):
    events = []
    output = tmp_path / "out"
    preflight = _preflight(
        diarization_device="mps",
        requested_diarization_device="auto",
    )
    supervisor = _FakeSupervisor(output, events)

    result = run_supervised_full_pipeline(
        tmp_path / "recording.mp4",
        output,
        preflight,
        supervisor=supervisor,
        frame_device="mps",
        frame_runner=_frame_runner(supervisor, output, events),
        scheduler=_scheduler(),
    )

    names = _event_names(events)
    assert names.index("complete-transcription") < names.index(
        "start-diarization"
    )
    assert names.index("complete-diarization") < names.index("start-frames")
    assert not result.initial_schedule.parallel
    assert not result.frame_schedule.parallel
    assert result.critical_path == "T + D + F + M + E"
    assert result.diarization_attempted_devices == ("mps",)
    assert not result.diarization_fallback_used


@pytest.mark.parametrize(
    ("policy", "expected_path", "expected_parallel"),
    [
        ("auto", "T + R + max(D, F) + M + E", True),
        ("serial", "T + R + D + F + M + E", False),
    ],
)
def test_automatic_mps_failure_retries_cpu_with_fresh_frame_admission(
    tmp_path,
    policy,
    expected_path,
    expected_parallel,
):
    events = []
    output = tmp_path / "out"
    preflight = _preflight(
        diarization_device="mps",
        requested_diarization_device="auto",
        policy=policy,
    )
    supervisor = _FakeSupervisor(output, events, fail_first_mps=True)

    result = run_supervised_full_pipeline(
        tmp_path / "recording.mp4",
        output,
        preflight,
        supervisor=supervisor,
        frame_device="mps",
        frame_runner=_frame_runner(supervisor, output, events),
        scheduler=_scheduler(policy),
    )

    starts = [
        detail for name, detail in events if name == "start-diarization"
    ]
    assert [start["device"] for start in starts] == ["mps", "cpu"]
    assert result.diarization_attempted_devices == ("mps", "cpu")
    assert result.diarization_fallback_used
    assert result.transcript.diarization_fallback_schedule is result.frame_schedule
    assert result.frame_schedule.parallel is expected_parallel
    assert result.critical_path == expected_path
    retry = result.pipeline_evidence.interval("diarization_retry")
    diarization = result.pipeline_evidence.interval("diarization")
    frames = result.pipeline_evidence.interval("frames")
    assert retry.outcome == "failed"
    assert retry.launch_wave == "post-transcription"
    assert diarization.launch_wave == "post-transcription"
    assert diarization.overlaps(frames) is expected_parallel
    assert set(result.timings) >= {
        "transcription",
        "diarization_retry",
        "diarization",
        "frames",
    }


def test_initial_wave_mps_failure_uses_max_t_r_retry_critical_path(tmp_path):
    events = []
    output = tmp_path / "out"
    preflight = _preflight(
        backend="whisper",
        requested_backend="whisper",
        transcription_device="cpu",
        diarization_device="mps",
        requested_diarization_device="auto",
    )
    supervisor = _FakeSupervisor(
        output,
        events,
        effective_backend="whisper",
        fail_first_mps=True,
    )

    result = run_supervised_full_pipeline(
        tmp_path / "recording.mp4",
        output,
        preflight,
        supervisor=supervisor,
        frame_device="mps",
        frame_runner=_frame_runner(supervisor, output, events),
        scheduler=_scheduler(),
    )

    assert result.pipeline_evidence.interval(
        "diarization_retry"
    ).launch_wave == "initial"
    assert result.critical_path == "max(T, R) + max(D, F) + M + E"


def test_mps_failure_during_cpu_frames_retries_cpu_after_frames(tmp_path):
    events = []
    output = tmp_path / "out"
    preflight = _preflight(
        runtime=transcript.RuntimePlatform("Darwin", "arm64", 13, 22),
        backend="whisper",
        requested_backend="whisper",
        transcription_device="cpu",
        diarization_device="mps",
        requested_diarization_device="auto",
    )
    supervisor = _FakeSupervisor(
        output,
        events,
        effective_backend="whisper",
        fail_first_mps=True,
    )
    frame_device = resolve_frame_device(preflight)

    result = run_supervised_full_pipeline(
        tmp_path / "recording.mp4",
        output,
        preflight,
        supervisor=supervisor,
        frame_device=frame_device,
        frame_runner=_frame_runner(supervisor, output, events),
        scheduler=_scheduler(),
    )

    names = _event_names(events)
    starts = [
        detail for name, detail in events if name == "start-diarization"
    ]
    start_indices = [
        index for index, name in enumerate(names) if name == "start-diarization"
    ]
    assert frame_device == "cpu"
    assert [start["device"] for start in starts] == ["mps", "cpu"]
    assert names.index("finish-frames") < start_indices[-1]
    assert result.diarization_attempted_devices == ("mps", "cpu")
    assert result.diarization_fallback_used
    assert result.frame_schedule.parallel
    assert result.transcript.diarization_fallback_schedule is not None
    assert tuple(
        stage.stage
        for stage in result.transcript.diarization_fallback_schedule.stages
    ) == ("diarization",)
    assert result.transcript.diarization_fallback_schedule.stages[0].device == "cpu"
    assert result.critical_path == "max(T + F, R) + D + M + E"
    assert result.pipeline_evidence.interval("diarization_retry").overlaps(
        result.pipeline_evidence.interval("frames")
    )
    assert not result.pipeline_evidence.interval("diarization").overlaps(
        result.pipeline_evidence.interval("frames")
    )
    assert result.transcript.segments[0].speaker == "SPEAKER_00"


def test_empty_transcript_does_not_retry_an_already_failed_mps_worker(tmp_path):
    events = []
    output = tmp_path / "out"
    preflight = _preflight(
        backend="whisper",
        requested_backend="whisper",
        transcription_device="cpu",
        diarization_device="mps",
        requested_diarization_device="auto",
    )
    supervisor = _FakeSupervisor(
        output,
        events,
        segments=(),
        effective_backend="whisper",
        fail_first_mps=True,
        mps_exits_before_transcription=True,
    )

    result = run_supervised_full_pipeline(
        tmp_path / "recording.mp4",
        output,
        preflight,
        supervisor=supervisor,
        frame_device="mps",
        frame_runner=_frame_runner(supervisor, output, events),
        scheduler=_scheduler(),
    )

    starts = [
        detail for name, detail in events if name == "start-diarization"
    ]
    assert [start["device"] for start in starts] == ["mps"]
    assert result.transcript.segments == ()
    assert result.diarization_attempted_devices == ("mps",)
    assert not result.diarization_fallback_used
    assert result.pipeline_evidence.interval("diarization_retry") is None
    assert result.pipeline_evidence.interval("diarization").outcome == "failed"
    assert (output / "frames" / "current.txt").read_text() == "current"


@pytest.mark.parametrize(
    ("requested_device", "fallback_eligible"),
    [("mps", True), ("auto", False)],
)
def test_explicit_or_ineligible_mps_failure_does_not_retry_cpu(
    tmp_path,
    requested_device,
    fallback_eligible,
):
    events = []
    output = tmp_path / "out"
    failure = StageWorkerError(
        "diarization",
        "injected non-retry failure",
        exitcode=1,
        error_type="RuntimeError",
        fallback_eligible=fallback_eligible,
    )
    preflight = _preflight(
        diarization_device="mps",
        requested_diarization_device=requested_device,
    )
    supervisor = _FakeSupervisor(
        output,
        events,
        diarization_error=failure,
    )

    result = run_supervised_full_pipeline(
        tmp_path / "recording.mp4",
        output,
        preflight,
        supervisor=supervisor,
        frame_device="mps",
        frame_runner=_frame_runner(supervisor, output, events),
        scheduler=_scheduler(),
    )

    assert result.diarization_attempted_devices == ("mps",)
    assert not result.diarization_fallback_used
    assert result.pipeline_evidence.interval("diarization_retry") is None
    assert result.transcript.segments[0].speaker is None


def test_failed_cpu_retry_preserves_unlabeled_transcript_and_attempt_evidence(
    tmp_path,
):
    events = []
    output = tmp_path / "out"
    cpu_failure = StageWorkerError(
        "diarization",
        "injected CPU retry failure",
        exitcode=1,
        error_type="RuntimeError",
    )
    preflight = _preflight(
        diarization_device="mps",
        requested_diarization_device="auto",
    )
    supervisor = _FakeSupervisor(
        output,
        events,
        fail_first_mps=True,
        diarization_error=cpu_failure,
    )

    result = run_supervised_full_pipeline(
        tmp_path / "recording.mp4",
        output,
        preflight,
        supervisor=supervisor,
        frame_device="mps",
        frame_runner=_frame_runner(supervisor, output, events),
        scheduler=_scheduler(),
    )

    assert result.diarization_attempted_devices == ("mps", "cpu")
    assert result.diarization_fallback_used
    assert result.pipeline_evidence.interval("diarization_retry").outcome == "failed"
    assert result.pipeline_evidence.interval("diarization").outcome == "failed"
    assert result.transcript.segments[0].speaker is None
    assert not (output / "diarization.json").exists()


def test_explicit_serial_keeps_post_transcription_wave_serial(tmp_path):
    events = []
    output = tmp_path / "out"
    preflight = _preflight(policy="serial")
    supervisor = _FakeSupervisor(output, events)

    result = run_supervised_full_pipeline(
        tmp_path / "recording.mp4",
        output,
        preflight,
        supervisor=supervisor,
        frame_device="mps",
        frame_runner=_frame_runner(supervisor, output, events),
        scheduler=_scheduler("serial"),
    )

    names = _event_names(events)
    assert not result.initial_schedule.parallel
    assert not result.frame_schedule.parallel
    assert names.index("complete-transcription") < names.index("start-diarization")
    assert names.index("complete-diarization") < names.index("start-frames")
    assert result.critical_path == "T + D + F + M + E"


def test_cpu_frames_wait_for_parallel_cpu_transcription_and_diarization(tmp_path):
    events = []
    output = tmp_path / "out"
    preflight = _preflight(
        runtime=LINUX,
        backend="whisper",
        transcription_device="cpu",
        policy="parallel",
    )
    supervisor = _FakeSupervisor(output, events, effective_backend="whisper")

    result = run_supervised_full_pipeline(
        tmp_path / "recording.mp4",
        output,
        preflight,
        supervisor=supervisor,
        frame_device="cpu",
        frame_runner=_frame_runner(supervisor, output, events),
        scheduler=_scheduler("parallel"),
    )

    names = _event_names(events)
    assert names.index("start-diarization") < names.index("complete-transcription")
    assert names.index("complete-diarization") < names.index("start-frames")
    assert not result.frame_schedule.parallel
    assert "CPU frame work" in result.frame_schedule.reason
    assert result.critical_path == "max(T, D) + F + M + E"


def test_frame_overlap_reprobes_memory_after_transcription_exits(tmp_path):
    events = []
    output = tmp_path / "out"
    supervisor = _FakeSupervisor(output, events)
    probes = (
        RuntimeResources(8, AMPLE_MEMORY),
        RuntimeResources(8, GIB),
    )

    result = run_supervised_full_pipeline(
        tmp_path / "recording.mp4",
        output,
        _preflight(),
        supervisor=supervisor,
        frame_device="mps",
        frame_runner=_frame_runner(supervisor, output, events),
        scheduler=_scheduler(probes=probes),
    )

    names = _event_names(events)
    assert result.initial_schedule.parallel
    assert not result.frame_schedule.parallel
    assert "memory admission failed" in result.frame_schedule.reason
    assert names.index("complete-diarization") < names.index("start-frames")
    assert result.critical_path == "max(T, D) + F + M + E"


def test_mlx_fallback_finishes_in_a_fresh_worker_before_frames_start(tmp_path):
    events = []
    output = tmp_path / "out"
    preflight = _preflight(requested_backend="auto")
    supervisor = _FakeSupervisor(output, events, fail_first_mlx=True)

    result = run_supervised_full_pipeline(
        tmp_path / "recording.mp4",
        output,
        preflight,
        supervisor=supervisor,
        frame_device="mps",
        frame_runner=_frame_runner(supervisor, output, events),
        scheduler=_scheduler(),
    )

    names = _event_names(events)
    transcription_starts = [
        detail
        for name, detail in events
        if name == "start-transcription"
    ]
    assert [start["requested_backend"] for start in transcription_starts] == [
        "auto",
        "whisper",
    ]
    assert supervisor.transcription_attempts == 2
    assert names.index("start-frames") > max(
        index
        for index, name in enumerate(names)
        if name == "complete-transcription"
    )
    assert result.transcript.fallback_used
    assert result.transcript.effective_backend == "whisper"


def test_serialized_mlx_fallback_does_not_double_count_diarization_in_path(
    tmp_path,
):
    events = []
    output = tmp_path / "out"
    preflight = _preflight(requested_backend="auto")
    supervisor = _FakeSupervisor(output, events, fail_first_mlx=True)
    probes = (
        RuntimeResources(8, AMPLE_MEMORY),
        RuntimeResources(2, AMPLE_MEMORY),
        RuntimeResources(8, AMPLE_MEMORY),
    )

    result = run_supervised_full_pipeline(
        tmp_path / "recording.mp4",
        output,
        preflight,
        supervisor=supervisor,
        frame_device="mps",
        frame_runner=_frame_runner(supervisor, output, events),
        scheduler=_scheduler(probes=probes),
    )

    assert result.transcript.fallback_used
    assert result.critical_path == "T + F + M + E"
    assert (
        result.pipeline_evidence.interval("diarization").ended_at
        <= result.pipeline_evidence.interval("transcription").ended_at
    )
    assert _event_names(events).index("complete-diarization") < (
        _event_names(events).index("start-frames")
    )


def test_stage_timings_span_launch_through_validated_commit(tmp_path):
    events = []
    output = tmp_path / "out"
    supervisor = _FakeSupervisor(output, events)
    ticks = iter((0.0, 0.0, 5.0, 5.0, 8.0, 10.0, 10.0, 11.0, 11.0, 12.0))
    clock_calls = []

    def clock():
        value = next(ticks)
        clock_calls.append(value)
        return value

    result = run_supervised_full_pipeline(
        tmp_path / "recording.mp4",
        output,
        _preflight(),
        supervisor=supervisor,
        frame_device="mps",
        frame_runner=_frame_runner(supervisor, output, events),
        scheduler=_scheduler(),
        clock=clock,
    )

    assert clock_calls == [0.0, 0.0, 5.0, 5.0, 8.0, 10.0, 10.0, 11.0, 11.0, 12.0]
    assert result.timings == {
        "transcription": 5.0,
        "frames": 3.0,
        "diarization": 10.0,
        "merge": 1.0,
        "manifest": 1.0,
    }
    assert result.transcript.timings == result.timings


def test_empty_serial_transcript_skips_unstarted_diarization_but_keeps_frames(
    tmp_path,
):
    events = []
    output = tmp_path / "out"
    preflight = _preflight(policy="serial")
    supervisor = _FakeSupervisor(output, events, segments=())

    result = run_supervised_full_pipeline(
        tmp_path / "recording.mp4",
        output,
        preflight,
        supervisor=supervisor,
        frame_device="mps",
        frame_runner=_frame_runner(supervisor, output, events),
        scheduler=_scheduler("serial"),
    )

    names = _event_names(events)
    assert "start-diarization" not in names
    assert "start-frames" in names
    assert result.transcript.segments == ()
    assert result.critical_path == "T + F + M + E"
    assert result.pipeline_evidence.interval("diarization") is None
    assert json.loads((output / "transcript.json").read_text(encoding="utf-8")) == []
    assert (output / "frames" / "current.txt").exists()


def test_empty_parallel_transcript_cancels_diarization_and_records_topology(tmp_path):
    events = []
    output = tmp_path / "out"
    supervisor = _FakeSupervisor(output, events, segments=())

    result = run_supervised_full_pipeline(
        tmp_path / "recording.mp4",
        output,
        _preflight(),
        supervisor=supervisor,
        frame_device="mps",
        frame_runner=_frame_runner(supervisor, output, events),
        scheduler=_scheduler(),
    )

    assert "cancel-diarization" in _event_names(events)
    assert result.critical_path == "T + F + M + E"
    assert result.pipeline_evidence.interval("diarization").outcome == "cancelled"
    assert result.pipeline_evidence.interval("frames").outcome == "completed"


@pytest.mark.parametrize("fmt", ["txt", "srt", "vtt", "json"])
def test_full_pipeline_preserves_all_final_transcript_formats(tmp_path, fmt):
    events = []
    output = tmp_path / fmt
    preflight = _preflight(
        runtime=LINUX,
        backend="whisper",
        transcription_device="cpu",
        diarization_device=None,
        speaker_detection=False,
        fmt=fmt,
    )
    supervisor = _FakeSupervisor(output, events, effective_backend="whisper")

    run_supervised_full_pipeline(
        tmp_path / "recording.mp4",
        output,
        preflight,
        supervisor=supervisor,
        frame_device="cpu",
        frame_runner=_frame_runner(supervisor, output, events),
        scheduler=_scheduler(),
    )

    assert (output / f"transcript.{fmt}").exists()
    assert (output / "transcript.raw.json").exists()
    if fmt != "json":
        assert (output / "transcript.json").exists()


@pytest.mark.parametrize("frame_after_diarization", [False, True])
@pytest.mark.parametrize(
    "error_kind",
    ["exception", "system-exit"],
)
def test_frame_failure_finishes_independent_transcript_and_preserves_prior_frames(
    tmp_path,
    frame_after_diarization,
    error_kind,
):
    events = []
    output = tmp_path / "out"
    prior_frames = output / "frames"
    prior_frames.mkdir(parents=True)
    (prior_frames / "previous.txt").write_text("previous", encoding="utf-8")
    if frame_after_diarization:
        preflight = _preflight(
            runtime=LINUX,
            backend="whisper",
            transcription_device="cpu",
        )
        frame_device = "cpu"
        effective_backend = "whisper"
    else:
        preflight = _preflight()
        frame_device = "mps"
        effective_backend = "mlx"
    supervisor = _FakeSupervisor(
        output,
        events,
        effective_backend=effective_backend,
    )
    injected_error = (
        OSError("injected frame failure")
        if error_kind == "exception"
        else SystemExit(1)
    )

    with pytest.raises(FullPipelineFrameError, match="partial output") as raised:
        run_supervised_full_pipeline(
            tmp_path / "recording.mp4",
            output,
            preflight,
            supervisor=supervisor,
            frame_device=frame_device,
            frame_runner=_frame_runner(
                supervisor,
                output,
                events,
                error=injected_error,
            ),
            scheduler=_scheduler(),
        )

    names = _event_names(events)
    if frame_after_diarization:
        assert names.index("complete-diarization") < names.index("start-frames")
    else:
        assert names.index("fail-frames") < names.index("complete-diarization")
    assert "enrich-manifest" not in names
    assert "promote-frames" not in names
    assert (prior_frames / "previous.txt").read_text(encoding="utf-8") == (
        "previous"
    )
    assert json.loads((output / "transcript.json").read_text(encoding="utf-8"))[0][
        "speaker"
    ] == "SPEAKER_00"
    assert raised.value.pipeline_evidence.interval("frames").outcome == "failed"


def test_parent_interruption_cancels_overlap_and_discards_current_frame_stage(
    tmp_path,
):
    events = []
    output = tmp_path / "out"
    prior_frames = output / "frames"
    prior_frames.mkdir(parents=True)
    (prior_frames / "previous.txt").write_text("previous", encoding="utf-8")
    prior_final = output / "transcript.json"
    prior_final.write_text("previous final", encoding="utf-8")
    supervisor = _ContextFakeSupervisor(output, events)

    def interrupt_frames():
        assert supervisor.handles["diarization"].process.is_alive()
        staged_frames = supervisor.staging_root / "frames"
        staged_frames.mkdir()
        (staged_frames / "partial.png").write_bytes(b"partial")
        events.append(("interrupt-frames", None))
        raise KeyboardInterrupt("parent interrupted")

    with pytest.raises(KeyboardInterrupt, match="parent interrupted") as raised:
        with supervisor:
            run_supervised_full_pipeline(
                tmp_path / "recording.mp4",
                output,
                _preflight(),
                supervisor=supervisor,
                frame_device="mps",
                frame_runner=interrupt_frames,
                scheduler=_scheduler(),
            )

    names = _event_names(events)
    assert names.index("interrupt-frames") < names.index("cancel-diarization")
    assert names[-1] == "close-supervisor"
    assert not supervisor.staging_root.exists()
    assert (prior_frames / "previous.txt").read_text(encoding="utf-8") == (
        "previous"
    )
    assert prior_final.read_text(encoding="utf-8") == "previous final"
    assert transcript.read_raw_transcript_checkpoint(
        output / "transcript.raw.json"
    )[0].text == "hello"
    assert not (output / "diarization.json").exists()
    assert raised.value.pipeline_evidence.interval("transcription").outcome == (
        "completed"
    )
    assert raised.value.pipeline_evidence.interval("diarization").outcome == (
        "cancelled"
    )
    assert raised.value.pipeline_evidence.interval("frames").outcome == "cancelled"


def test_transcription_launch_failure_closes_and_attaches_pipeline_evidence(
    tmp_path,
    monkeypatch,
):
    events = []
    output = tmp_path / "out"
    supervisor = _FakeSupervisor(output, events)
    failure = OSError("injected transcription launch failure")

    def fail_transcription_launch(*_args, **_kwargs):
        events.append(("fail-transcription-launch", None))
        raise failure

    monkeypatch.setattr(
        supervisor,
        "start_transcription",
        fail_transcription_launch,
    )

    with pytest.raises(OSError, match="transcription launch") as raised:
        run_supervised_full_pipeline(
            tmp_path / "recording.mp4",
            output,
            _preflight(),
            supervisor=supervisor,
            frame_device="mps",
            frame_runner=lambda: pytest.fail("frames must not start"),
            scheduler=_scheduler(),
        )

    assert raised.value is failure
    assert raised.value.pipeline_evidence.interval("transcription").outcome == "failed"
    assert raised.value.pipeline_evidence.interval("diarization") is None
    assert _event_names(events) == ["fail-transcription-launch"]


def test_diarization_launch_failure_cancels_transcription_and_attaches_evidence(
    tmp_path,
    monkeypatch,
):
    events = []
    output = tmp_path / "out"
    supervisor = _FakeSupervisor(output, events)
    failure = OSError("injected diarization launch failure")

    def fail_diarization_launch(*_args, **_kwargs):
        events.append(("fail-diarization-launch", None))
        raise failure

    monkeypatch.setattr(
        supervisor,
        "start_diarization",
        fail_diarization_launch,
    )

    with pytest.raises(OSError, match="diarization launch") as raised:
        run_supervised_full_pipeline(
            tmp_path / "recording.mp4",
            output,
            _preflight(),
            supervisor=supervisor,
            frame_device="mps",
            frame_runner=lambda: pytest.fail("frames must not start"),
            scheduler=_scheduler(),
        )

    assert raised.value is failure
    assert raised.value.pipeline_evidence.interval("transcription").outcome == (
        "cancelled"
    )
    assert raised.value.pipeline_evidence.interval("diarization").outcome == "failed"
    assert _event_names(events) == [
        "start-transcription",
        "fail-diarization-launch",
        "cancel-transcription",
    ]


def test_transcription_failure_cancels_diarization_and_never_starts_frames(
    tmp_path,
):
    events = []
    output = tmp_path / "out"
    prior_frames = output / "frames"
    prior_frames.mkdir(parents=True)
    (prior_frames / "previous.txt").write_text("previous", encoding="utf-8")
    prior_final = output / "transcript.json"
    prior_final.write_text("previous final", encoding="utf-8")
    prior_raw = output / "transcript.raw.json"
    prior_raw.write_text("previous raw", encoding="utf-8")
    failure = StageWorkerError(
        "transcription",
        "injected failure",
        exitcode=1,
        error_type="RuntimeError",
    )
    preflight = _preflight(backend="mlx")
    supervisor = _FakeSupervisor(
        output,
        events,
        transcription_error=failure,
    )

    with pytest.raises(StageWorkerError) as raised:
        run_supervised_full_pipeline(
            tmp_path / "recording.mp4",
            output,
            preflight,
            supervisor=supervisor,
            frame_device="mps",
            frame_runner=lambda: pytest.fail("frames must not start"),
            scheduler=_scheduler(),
        )

    assert raised.value is failure
    assert raised.value.pipeline_evidence.interval("transcription").outcome == "failed"
    assert raised.value.pipeline_evidence.interval("diarization").outcome == "cancelled"
    names = _event_names(events)
    assert "start-diarization" in names
    assert "cancel-diarization" in names
    assert prior_final.read_text(encoding="utf-8") == "previous final"
    assert (prior_frames / "previous.txt").read_text(encoding="utf-8") == (
        "previous"
    )
    assert prior_raw.read_text(encoding="utf-8") == "previous raw"
    assert not (output / "diarization.json").exists()


def test_diarization_failure_still_promotes_frames_and_writes_unlabeled_transcript(
    tmp_path,
    capsys,
):
    events = []
    output = tmp_path / "out"
    failure = StageWorkerError(
        "diarization",
        "injected failure",
        exitcode=1,
        error_type="RuntimeError",
    )
    supervisor = _FakeSupervisor(output, events, diarization_error=failure)

    result = run_supervised_full_pipeline(
        tmp_path / "recording.mp4",
        output,
        _preflight(),
        supervisor=supervisor,
        frame_device="mps",
        frame_runner=_frame_runner(supervisor, output, events),
        scheduler=_scheduler(),
    )

    assert result.transcript.segments[0].speaker is None
    assert (output / "frames" / "current.txt").exists()
    assert not (output / "diarization.json").exists()
    assert "speaker detection failed" in capsys.readouterr().err


def test_manifest_promotion_failure_is_reported_after_transcript_publication(
    tmp_path,
):
    events = []
    output = tmp_path / "out"
    supervisor = _FakeSupervisor(output, events)
    generation = _FakeFrameGeneration(output, events)

    def fail_promotion():
        events.append(("promote-frames", None))
        raise OSError("injected promotion failure")

    generation.promote = fail_promotion

    with pytest.raises(FullPipelineFrameError, match="partial output"):
        run_supervised_full_pipeline(
            tmp_path / "recording.mp4",
            output,
            _preflight(),
            supervisor=supervisor,
            frame_device="mps",
            frame_runner=lambda: generation,
            scheduler=_scheduler(),
        )

    assert (output / "transcript.json").exists()
    assert _event_names(events)[-2:] == ["enrich-manifest", "promote-frames"]


def test_full_cli_reports_partial_frame_output_with_nonzero_status(
    tmp_path,
    monkeypatch,
    capsys,
):
    from keyframe import cli

    video = tmp_path / "recording.mp4"
    video.write_bytes(b"media")
    output = tmp_path / "out"
    prior_frames = output / "frames"
    prior_frames.mkdir(parents=True)
    (prior_frames / "previous.txt").write_text("previous", encoding="utf-8")
    monkeypatch.setattr(cli, "_preflight_transcript", lambda _args: object())

    def fail_full_pipeline(*_args):
        raise FullPipelineFrameError("partial output: injected frame failure")

    monkeypatch.setattr(cli, "_run_full_pipeline", fail_full_pipeline)
    args = SimpleNamespace(
        video=str(video),
        output=str(output),
        transcript_only=False,
        frames_only=False,
        whisper_model="medium",
        transcript_format="json",
        no_speaker_detection=False,
    )

    with pytest.raises(SystemExit) as raised:
        cli.cmd_extract(args)

    assert raised.value.code == 1
    assert "Error: partial output" in capsys.readouterr().err
    assert (prior_frames / "previous.txt").read_text(encoding="utf-8") == (
        "previous"
    )


def test_cli_full_pipeline_pins_frame_extraction_to_the_scheduled_device(
    tmp_path,
    monkeypatch,
):
    from keyframe import cli
    import keyframe.full_pipeline as full_pipeline

    preflight = _preflight()
    supervisor = object()
    captured = {}

    def fake_frame_generation(
        video,
        output,
        args,
        active_supervisor,
        *,
        frame_device,
    ):
        captured["frame"] = (
            video,
            output,
            args,
            active_supervisor,
            frame_device,
        )
        return "staged-frames"

    def fake_orchestrator(
        video,
        output,
        active_preflight,
        *,
        supervisor,
        frame_device,
        frame_runner,
    ):
        captured["orchestrator"] = (
            video,
            output,
            active_preflight,
            supervisor,
            frame_device,
        )
        assert frame_runner() == "staged-frames"
        return "full-result"

    monkeypatch.setattr(cli, "_run_frame_generation", fake_frame_generation)
    monkeypatch.setattr(
        full_pipeline,
        "run_supervised_full_pipeline",
        fake_orchestrator,
    )
    video = tmp_path / "recording.mp4"
    output = tmp_path / "out"
    args = object()

    result = cli._run_full_pipeline(
        video,
        output,
        args,
        preflight,
        supervisor,
    )

    assert result == "full-result"
    assert captured["orchestrator"][-1] == "mps"
    assert captured["frame"][-1] == "mps"
