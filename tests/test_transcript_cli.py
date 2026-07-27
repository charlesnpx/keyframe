import errno
import json
from types import SimpleNamespace

import pytest

from keyframe import cli, transcript
from keyframe import transcript_cli as transcript_cli_module
from keyframe.artifacts import transcript_checkpoint_paths
from keyframe.output_session import OutputSessionError
from keyframe.stage_scheduler import (
    GIB,
    RuntimeResources,
    StageScheduler,
)
from keyframe.stage_supervisor import (
    StageCompletion,
    StageProgress,
    StageWorkerError,
)
from keyframe.transcript_cli import (
    TranscriptOutputError,
    TranscriptPreflight,
    TranscriptRunConfig,
    _print_schedule,
    _write_final_outputs,
    preflight_transcript_run,
    print_stage_progress,
    run_supervised_transcript,
)


SUPPORTED_MAC = transcript.RuntimePlatform("Darwin", "arm64", 14, 23)
LINUX = transcript.RuntimePlatform("Linux", "x86_64", None, 6)


def _video(tmp_path):
    video = tmp_path / "recording.mp4"
    video.write_bytes(b"not real media")
    return video


def _stub_cli_media_preflight(monkeypatch):
    from keyframe import media_preflight

    monkeypatch.setattr(
        media_preflight,
        "probe_media",
        lambda _path: media_preflight.MediaProbeResult(
            (
                media_preflight.MediaStream(
                    codec_type="audio",
                    codec_name="aac",
                    channels=1,
                ),
            )
        ),
    )


def _config(**overrides):
    values = {
        "model_name": "medium",
        "fmt": "json",
        "transcription_backend": "auto",
        "diarization_device": "auto",
        "stage_concurrency": "auto",
        "speaker_detection": True,
    }
    values.update(overrides)
    return TranscriptRunConfig(**values)


def _preflight(**overrides):
    config = overrides.pop("config", _config())
    values = {
        "config": config,
        "runtime_platform": SUPPORTED_MAC,
        "effective_backend": "mlx",
        "transcription_device": "mlx",
        "hf_token": "hf_test",
        "effective_diarization_device": "cpu",
        "missing_hf_token": False,
    }
    values.update(overrides)
    return TranscriptPreflight(**values)


def _scheduler(policy="auto", *, cpus=8, memory=64 * GIB):
    return StageScheduler(
        policy,
        resource_probe=lambda: RuntimeResources(cpus, memory),
    )


def test_final_output_write_failure_preserves_the_previous_generation(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "out"
    output.mkdir()
    output_paths = (output / "transcript.txt", output / "transcript.json")
    output_paths[0].write_text("previous text", encoding="utf-8")
    output_paths[1].write_text("previous json", encoding="utf-8")
    staging_root = output / "keyframe-run-test"
    staging_root.mkdir()

    def disk_full(*_args, **_kwargs):
        raise OSError(errno.ENOSPC, "injected final-output disk exhaustion")

    monkeypatch.setattr(transcript, "write_json", disk_full)
    with pytest.raises(TranscriptOutputError, match="disk exhaustion"):
        _write_final_outputs(
            (transcript.TranscriptSegment(0.0, 1.0, "current"),),
            output_paths,
            "txt",
            staging_root=staging_root,
        )

    assert output_paths[0].read_text(encoding="utf-8") == "previous text"
    assert output_paths[1].read_text(encoding="utf-8") == "previous json"
    assert not list(staging_root.iterdir())


def test_final_output_promotion_failure_rolls_back_every_representation(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "out"
    output.mkdir()
    output_paths = (output / "transcript.txt", output / "transcript.json")
    output_paths[0].write_text("previous text", encoding="utf-8")
    output_paths[1].write_text("previous json", encoding="utf-8")
    staging_root = output / "keyframe-run-test"
    staging_root.mkdir()
    real_replace = transcript_cli_module._replace_final_output
    calls = 0

    def fail_second_promotion(source, target):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise OSError("injected final-output rename failure")
        return real_replace(source, target)

    monkeypatch.setattr(
        transcript_cli_module,
        "_replace_final_output",
        fail_second_promotion,
    )
    with pytest.raises(TranscriptOutputError, match="rename failure"):
        _write_final_outputs(
            (transcript.TranscriptSegment(0.0, 1.0, "current"),),
            output_paths,
            "txt",
            staging_root=staging_root,
        )

    assert calls == 6
    assert output_paths[0].read_text(encoding="utf-8") == "previous text"
    assert output_paths[1].read_text(encoding="utf-8") == "previous json"
    assert not list(staging_root.iterdir())


def test_preflight_selects_mlx_and_mps_diarization_on_supported_mac_without_cuda_probe():
    result = preflight_transcript_run(
        _config(),
        environment={"HF_TOKEN": "  hf_test  "},
        runtime_platform=SUPPORTED_MAC,
        cuda_probe=lambda: pytest.fail("macOS preflight must not import Torch for CUDA"),
        mps_probe=lambda: True,
    )

    assert result.effective_backend == "mlx"
    assert result.transcription_device == "mlx"
    assert result.hf_token == "hf_test"
    assert result.effective_diarization_device == "mps"


def test_preflight_falls_back_to_cpu_when_mps_is_unavailable_on_supported_mac():
    result = preflight_transcript_run(
        _config(),
        environment={"HF_TOKEN": "hf_test"},
        runtime_platform=SUPPORTED_MAC,
        cuda_probe=lambda: pytest.fail("macOS preflight must not probe CUDA"),
        mps_probe=lambda: False,
    )

    assert result.effective_diarization_device == "cpu"


def test_preflight_selects_cuda_for_whisper_and_diarization_when_available():
    result = preflight_transcript_run(
        _config(),
        environment={"HF_TOKEN": "hf_test"},
        runtime_platform=LINUX,
        cuda_probe=lambda: True,
    )

    assert result.effective_backend == "whisper"
    assert result.transcription_device == "cuda"
    assert result.effective_diarization_device == "cuda"


def test_preflight_honors_forced_cpu_diarization_with_cuda_transcription():
    result = preflight_transcript_run(
        _config(diarization_device="cpu"),
        environment={"HF_TOKEN": "hf_test"},
        runtime_platform=LINUX,
        cuda_probe=lambda: True,
    )

    assert result.transcription_device == "cuda"
    assert result.effective_diarization_device == "cpu"


@pytest.mark.parametrize("token", [None, "   "])
def test_preflight_missing_token_disables_diarization(token):
    environment = {} if token is None else {"HF_TOKEN": token}

    result = preflight_transcript_run(
        _config(),
        environment=environment,
        runtime_platform=SUPPORTED_MAC,
    )

    assert result.hf_token is None
    assert result.missing_hf_token
    assert not result.diarization_enabled


def test_preflight_no_speaker_detection_ignores_cuda_hardware_requirement():
    result = preflight_transcript_run(
        _config(speaker_detection=False, diarization_device="cuda"),
        environment={"HF_TOKEN": "hf_test"},
        runtime_platform=SUPPORTED_MAC,
        cuda_probe=lambda: pytest.fail("disabled diarization must not probe CUDA"),
    )

    assert not result.diarization_enabled
    assert not result.missing_hf_token


def test_preflight_rejects_unsupported_explicit_mlx_before_cuda_probe():
    with pytest.raises(transcript.UnsupportedTranscriptionBackendError):
        preflight_transcript_run(
            _config(transcription_backend="mlx"),
            environment={},
            runtime_platform=LINUX,
            cuda_probe=lambda: pytest.fail("unsupported MLX must fail first"),
        )


def test_preflight_rejects_forced_cuda_diarization_without_cuda():
    with pytest.raises(transcript.UnsupportedDiarizationDeviceError):
        preflight_transcript_run(
            _config(diarization_device="cuda"),
            environment={"HF_TOKEN": "hf_test"},
            runtime_platform=LINUX,
            cuda_probe=lambda: False,
        )


def test_preflight_rejects_forced_mps_diarization_without_mps():
    with pytest.raises(transcript.UnsupportedDiarizationDeviceError, match="MPS"):
        preflight_transcript_run(
            _config(diarization_device="mps"),
            environment={"HF_TOKEN": "hf_test"},
            runtime_platform=SUPPORTED_MAC,
            cuda_probe=lambda: pytest.fail("macOS must not probe CUDA"),
            mps_probe=lambda: False,
        )


def test_preflight_forced_cpu_skips_mps_and_cuda_diarization_probes():
    result = preflight_transcript_run(
        _config(diarization_device="cpu"),
        environment={"HF_TOKEN": "hf_test"},
        runtime_platform=SUPPORTED_MAC,
        cuda_probe=lambda: pytest.fail("forced CPU must not probe CUDA"),
        mps_probe=lambda: pytest.fail("forced CPU must not probe MPS"),
    )

    assert result.effective_diarization_device == "cpu"


def test_direct_and_explicit_extract_alias_share_exact_parser_contract():
    direct = cli._parse_extract_args(
        [
            "recording.mp4",
            "--transcription-backend",
            "whisper",
            "--diarization-device",
            "cpu",
            "--stage-concurrency",
            "parallel",
        ]
    )
    explicit = cli._parse_extract_args(
        [
            "extract",
            "recording.mp4",
            "--transcription-backend",
            "whisper",
            "--diarization-device",
            "cpu",
            "--stage-concurrency",
            "parallel",
        ]
    )

    assert vars(direct) == vars(explicit)
    assert direct.transcription_backend == "whisper"
    assert direct.diarization_device == "cpu"
    assert direct.stage_concurrency == "parallel"


def test_new_cli_options_default_to_auto():
    args = cli._parse_extract_args(["recording.mp4"])

    assert args.transcription_backend == "auto"
    assert args.diarization_device == "auto"
    assert args.stage_concurrency == "auto"


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--transcription-backend", "auto"),
        ("--transcription-backend", "mlx"),
        ("--transcription-backend", "whisper"),
        ("--diarization-device", "auto"),
        ("--diarization-device", "cpu"),
        ("--diarization-device", "mps"),
        ("--diarization-device", "cuda"),
        ("--stage-concurrency", "auto"),
        ("--stage-concurrency", "serial"),
        ("--stage-concurrency", "parallel"),
    ],
)
def test_new_cli_options_accept_every_documented_value(flag, value):
    args = cli._parse_extract_args(["recording.mp4", flag, value])

    assert getattr(args, flag.removeprefix("--").replace("-", "_")) == value


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--transcription-backend", "metal"),
        ("--diarization-device", "metal"),
        ("--stage-concurrency", "unsafe"),
    ],
)
def test_new_cli_options_reject_unknown_values(flag, value):
    with pytest.raises(SystemExit) as raised:
        cli._parse_extract_args(["recording.mp4", flag, value])

    assert raised.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["keyframe", "recording.mp4", "--transcript-only"],
        ["keyframe", "extract", "recording.mp4", "--transcript-only"],
    ],
)
def test_main_routes_both_extract_entry_points_to_the_same_behavior(
    monkeypatch,
    argv,
):
    calls = []
    monkeypatch.setattr("sys.argv", argv)
    monkeypatch.setattr(cli, "cmd_extract", calls.append)

    cli.main()

    assert len(calls) == 1
    assert calls[0].video == "recording.mp4"
    assert calls[0].transcript_only


class _FakeProcess:
    next_pid = 3000

    def __init__(self, *, failure=False):
        type(self).next_pid += 1
        self.pid = type(self).next_pid
        self.exitcode = None
        self.alive = True
        self.failure = failure

    def is_alive(self):
        return self.alive


class _FakeHandle:
    def __init__(self, owner, stage, *, attempt=1, failure=False):
        self.owner = owner
        self.stage = stage
        self.attempt = attempt
        self.process = _FakeProcess(failure=failure)
        self._completion = None
        self._failure = None

    def wait(self):
        if self._failure is not None:
            raise self._failure
        if self._completion is not None:
            return self._completion
        try:
            self._completion = self.owner._finish(self)
        except Exception as exc:
            self._failure = exc
            raise
        return self._completion


class _FakeSupervisor:
    def __init__(
        self,
        output_dir,
        *,
        progress_callback=None,
        transcript_segments=None,
        diarization_rows=None,
        diarization_error=None,
        transcription_error=None,
        fail_first_mlx=False,
        fail_first_mps=False,
    ):
        self.output_dir = output_dir
        self.progress_callback = progress_callback
        self.public = transcript_checkpoint_paths(output_dir)
        self.transcript_segments = tuple(
            transcript_segments
            if transcript_segments is not None
            else (transcript.TranscriptSegment(0.0, 2.0, "hello"),)
        )
        self.diarization_rows = tuple(
            diarization_rows
            if diarization_rows is not None
            else (transcript.DiarizationRow(0.0, 2.0, "SPEAKER_00"),)
        )
        self.diarization_error = diarization_error
        self.transcription_error = transcription_error
        self.fail_first_mlx = fail_first_mlx
        self.fail_first_mps = fail_first_mps
        self.transcription_attempts = 0
        self.diarization_attempts = 0
        self.events = []
        self.started = []

    def __enter__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events.append("enter")
        return self

    def __exit__(self, *_args):
        self.events.append("exit")

    def start_transcription(self, video_path, **kwargs):
        self.transcription_attempts += 1
        failure = bool(self.fail_first_mlx and self.transcription_attempts == 1)
        handle = _FakeHandle(
            self,
            "transcription",
            attempt=self.transcription_attempts,
            failure=failure,
        )
        self.started.append(("transcription", video_path, kwargs, handle))
        self.events.append(f"start-transcription-{self.transcription_attempts}")
        if self.progress_callback is not None:
            self.progress_callback(StageProgress("transcription", "started"))
        return handle

    def start_diarization(self, video_path, **kwargs):
        self.diarization_attempts += 1
        handle = _FakeHandle(
            self,
            "diarization",
            attempt=self.diarization_attempts,
        )
        self.started.append(("diarization", video_path, kwargs, handle))
        self.events.append("start-diarization")
        if self.progress_callback is not None:
            self.progress_callback(StageProgress("diarization", "started"))
        return handle

    def _finish(self, handle):
        handle.process.alive = False
        if handle.stage == "transcription":
            if handle.process.failure:
                handle.process.exitcode = 1
                raise StageWorkerError(
                    "transcription",
                    "forced MLX load failure",
                    exitcode=1,
                    error_type="MLXModelLoadError",
                    fallback_eligible=True,
                )
            if self.transcription_error is not None:
                handle.process.exitcode = 1
                raise self.transcription_error
            handle.process.exitcode = 0
            transcript.write_raw_transcript_checkpoint(
                self.transcript_segments,
                self.public.transcript_raw,
            )
            requested = self.started[handle.attempt - 1][2]["requested_backend"]
            effective = "whisper" if requested == "whisper" else "mlx"
            metadata = {"language": "en", "effective_backend": effective}
            if effective == "mlx":
                metadata.update(
                    {
                        "model_repository": "mlx-community/whisper-medium-mlx",
                        "model_revision": "immutable-revision",
                        "model_resolution_source": "local-hit",
                        "model_resolution_seconds": 0.125,
                    }
                )
            return StageCompletion(
                "transcription",
                self.public.transcript_raw,
                metadata,
                self.transcript_segments,
            )
        if self.fail_first_mps and handle.attempt == 1:
            handle.process.exitcode = 1
            raise StageWorkerError(
                "diarization",
                "forced MPS inference failure",
                exitcode=1,
                error_type="MPSDiarizationInferenceError",
                fallback_eligible=True,
            )
        if self.diarization_error is not None:
            handle.process.exitcode = 1
            raise self.diarization_error
        handle.process.exitcode = 0
        transcript.write_diarization_checkpoint(
            self.diarization_rows,
            self.public.diarization,
        )
        return StageCompletion(
            "diarization",
            self.public.diarization,
            {"row_count": len(self.diarization_rows)},
            self.diarization_rows,
        )

    def complete(self, handle):
        self.events.append(f"complete-{handle.stage}-{handle.attempt}")
        return handle.wait()

    def cancel(self, handle):
        self.events.append(f"cancel-{handle.stage}")
        handle.process.alive = False
        handle.process.exitcode = -15


def _factory(holder, **scenario):
    def build(output_dir, *, progress_callback=None):
        supervisor = _FakeSupervisor(
            output_dir,
            progress_callback=progress_callback,
            **scenario,
        )
        holder.append(supervisor)
        return supervisor

    return build


def test_parallel_run_promotes_current_checkpoints_before_speaker_assignment(
    tmp_path,
    monkeypatch,
):
    video = _video(tmp_path)
    output = tmp_path / "out"
    supervisors = []
    original_assign = transcript._assign_speakers

    def assign_after_promotion(segments, rows):
        assert (output / "transcript.raw.json").exists()
        assert (output / "diarization.json").exists()
        assert not (output / "transcript.json").exists()
        return original_assign(segments, rows)

    monkeypatch.setattr(transcript, "_assign_speakers", assign_after_promotion)

    result = run_supervised_transcript(
        video,
        output,
        _preflight(),
        scheduler=_scheduler(),
        supervisor_factory=_factory(supervisors),
    )

    supervisor = supervisors[0]
    assert result.segments == (
        transcript.TranscriptSegment(0.0, 2.0, "hello", "SPEAKER_00"),
    )
    assert result.metadata["model_resolution_source"] == "local-hit"
    assert result.metadata["model_resolution_seconds"] == 0.125
    with pytest.raises(TypeError):
        result.metadata["model_resolution_source"] = "changed"
    assert supervisor.events.index("start-diarization") < supervisor.events.index(
        "complete-transcription-1"
    )
    diarization_start = next(
        entry for entry in supervisor.started if entry[0] == "diarization"
    )
    assert diarization_start[2]["device"] == "cpu"
    assert json.loads((output / "transcript.raw.json").read_text()) == [
        {"start": 0.0, "end": 2.0, "text": "hello"}
    ]
    assert json.loads((output / "transcript.json").read_text())[0]["speaker"] == "SPEAKER_00"


def test_serial_run_starts_diarization_only_after_raw_transcript_promotion(tmp_path):
    supervisors = []
    output = tmp_path / "out"

    run_supervised_transcript(
        _video(tmp_path),
        output,
        _preflight(config=_config(stage_concurrency="serial")),
        scheduler=_scheduler("serial"),
        supervisor_factory=_factory(supervisors),
    )

    events = supervisors[0].events
    assert events.index("complete-transcription-1") < events.index("start-diarization")
    assert (output / "transcript.raw.json").exists()


def test_transcript_only_automatic_mps_success_records_attempt_and_stays_serial(
    tmp_path,
):
    supervisors = []
    preflight = _preflight(
        config=_config(diarization_device="auto"),
        effective_diarization_device="mps",
    )

    result = run_supervised_transcript(
        _video(tmp_path),
        tmp_path / "out",
        preflight,
        scheduler=_scheduler(),
        supervisor_factory=_factory(supervisors),
    )

    starts = [
        entry for entry in supervisors[0].started if entry[0] == "diarization"
    ]
    assert [start[2]["device"] for start in starts] == ["mps"]
    assert result.diarization_attempted_devices == ("mps",)
    assert not result.diarization_fallback_used
    assert not result.initial_schedule.parallel


def test_transcript_only_automatic_mps_failure_retries_cpu_once(tmp_path):
    supervisors = []
    preflight = _preflight(
        config=_config(diarization_device="auto"),
        effective_diarization_device="mps",
    )

    result = run_supervised_transcript(
        _video(tmp_path),
        tmp_path / "out",
        preflight,
        scheduler=_scheduler(),
        supervisor_factory=_factory(supervisors, fail_first_mps=True),
    )

    starts = [
        entry for entry in supervisors[0].started if entry[0] == "diarization"
    ]
    assert [start[2]["device"] for start in starts] == ["mps", "cpu"]
    assert result.diarization_attempted_devices == ("mps", "cpu")
    assert result.diarization_fallback_used
    assert result.diarization_fallback_schedule is not None
    assert result.diarization_fallback_schedule.stages[0].device == "cpu"
    assert set(result.timings) >= {
        "transcription",
        "diarization_retry",
        "diarization",
    }
    assert result.segments[0].speaker == "SPEAKER_00"


def test_transcript_only_explicit_mps_failure_does_not_retry(tmp_path):
    supervisors = []
    failure = StageWorkerError(
        "diarization",
        "forced explicit MPS failure",
        exitcode=1,
        error_type="MPSDiarizationInferenceError",
        fallback_eligible=True,
    )
    preflight = _preflight(
        config=_config(diarization_device="mps"),
        effective_diarization_device="mps",
    )

    result = run_supervised_transcript(
        _video(tmp_path),
        tmp_path / "out",
        preflight,
        scheduler=_scheduler(),
        supervisor_factory=_factory(
            supervisors,
            diarization_error=failure,
        ),
    )

    assert result.diarization_attempted_devices == ("mps",)
    assert not result.diarization_fallback_used
    assert result.segments[0].speaker is None


def test_transcript_only_failed_cpu_retry_keeps_unlabeled_output(tmp_path):
    cpu_failure = StageWorkerError(
        "diarization",
        "forced CPU retry failure",
        exitcode=1,
        error_type="RuntimeError",
    )
    preflight = _preflight(
        config=_config(diarization_device="auto"),
        effective_diarization_device="mps",
    )

    result = run_supervised_transcript(
        _video(tmp_path),
        tmp_path / "out",
        preflight,
        scheduler=_scheduler(),
        supervisor_factory=_factory(
            [],
            fail_first_mps=True,
            diarization_error=cpu_failure,
        ),
    )

    assert result.diarization_attempted_devices == ("mps", "cpu")
    assert result.diarization_fallback_used
    assert result.segments[0].speaker is None


@pytest.mark.parametrize("fmt", ["txt", "srt", "vtt", "json"])
def test_supervised_run_preserves_all_final_output_formats(tmp_path, fmt):
    config = _config(fmt=fmt, speaker_detection=False)
    preflight = _preflight(
        config=config,
        hf_token=None,
        effective_diarization_device=None,
    )
    output = tmp_path / fmt

    run_supervised_transcript(
        _video(tmp_path),
        output,
        preflight,
        scheduler=_scheduler(),
        supervisor_factory=_factory([]),
    )

    assert (output / f"transcript.{fmt}").exists()
    assert (output / "transcript.raw.json").exists()
    assert not (output / "diarization.json").exists()
    if fmt != "json":
        assert (output / "transcript.json").exists()


@pytest.mark.parametrize("speaker_detection", [True, False])
def test_skipped_diarization_removes_stale_artifact(
    tmp_path,
    capsys,
    speaker_detection,
):
    output = tmp_path / "out"
    output.mkdir()
    (output / "diarization.json").write_text("stale", encoding="utf-8")
    config = _config(speaker_detection=speaker_detection)
    preflight = _preflight(
        config=config,
        hf_token=None,
        effective_diarization_device=None,
        missing_hf_token=speaker_detection,
    )

    run_supervised_transcript(
        _video(tmp_path),
        output,
        preflight,
        scheduler=_scheduler(),
        supervisor_factory=_factory([]),
    )

    assert not (output / "diarization.json").exists()
    assert ("no HF_TOKEN" in capsys.readouterr().err) is speaker_detection


@pytest.mark.parametrize(
    "scenario",
    [
        {
            "diarization_error": StageWorkerError(
                "diarization",
                "model failed",
                exitcode=1,
                error_type="RuntimeError",
            )
        },
        {"diarization_rows": (transcript.DiarizationRow(10.0, 12.0, "SPEAKER_00"),)},
    ],
)
def test_diarization_failure_or_unusable_overlap_writes_unlabeled_fallback(
    tmp_path,
    capsys,
    scenario,
):
    output = tmp_path / "out"
    output.mkdir()
    (output / "diarization.json").write_text("stale", encoding="utf-8")

    result = run_supervised_transcript(
        _video(tmp_path),
        output,
        _preflight(),
        scheduler=_scheduler(),
        supervisor_factory=_factory([], **scenario),
    )

    assert result.segments == (transcript.TranscriptSegment(0.0, 2.0, "hello"),)
    assert not (output / "diarization.json").exists()
    assert "speaker detection failed" in capsys.readouterr().err
    assert "speaker" not in json.loads((output / "transcript.json").read_text())[0]


def test_empty_transcript_cancels_speculative_diarization_and_writes_empty_final(tmp_path):
    supervisors = []
    output = tmp_path / "out"

    result = run_supervised_transcript(
        _video(tmp_path),
        output,
        _preflight(),
        scheduler=_scheduler(),
        supervisor_factory=_factory(supervisors, transcript_segments=()),
    )

    assert result.segments == ()
    assert "cancel-diarization" in supervisors[0].events
    assert json.loads((output / "transcript.raw.json").read_text()) == []
    assert json.loads((output / "transcript.json").read_text()) == []
    assert not (output / "diarization.json").exists()


def test_transcription_failure_preserves_prior_final_and_does_not_promote_diarization(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    final = output / "transcript.json"
    final.write_text("previous final", encoding="utf-8")
    raw = output / "transcript.raw.json"
    raw.write_text("previous raw", encoding="utf-8")
    diarization = output / "diarization.json"
    diarization.write_text("stale", encoding="utf-8")
    failure = StageWorkerError(
        "transcription",
        "whisper failed",
        exitcode=1,
        error_type="RuntimeError",
    )

    with pytest.raises(StageWorkerError) as raised:
        run_supervised_transcript(
            _video(tmp_path),
            output,
            _preflight(
                config=_config(transcription_backend="whisper"),
                effective_backend="whisper",
                transcription_device="cpu",
            ),
            scheduler=_scheduler(),
            supervisor_factory=_factory([], transcription_error=failure),
        )

    assert raised.value is failure
    assert final.read_text(encoding="utf-8") == "previous final"
    assert raw.read_text(encoding="utf-8") == "previous raw"
    assert not diarization.exists()


def test_auto_mlx_failure_uses_fresh_whisper_attempt_and_reports_fallback(tmp_path):
    supervisors = []
    config = _config(speaker_detection=False)

    result = run_supervised_transcript(
        _video(tmp_path),
        tmp_path / "out",
        _preflight(
            config=config,
            hf_token=None,
            effective_diarization_device=None,
        ),
        scheduler=_scheduler(),
        supervisor_factory=_factory(supervisors, fail_first_mlx=True),
    )

    starts = [entry for entry in supervisors[0].started if entry[0] == "transcription"]
    assert len(starts) == 2
    assert starts[0][2]["requested_backend"] == "auto"
    assert starts[1][2]["requested_backend"] == "whisper"
    assert starts[0][3].process.pid != starts[1][3].process.pid
    assert result.fallback_used
    assert result.effective_backend == "whisper"


def test_previous_raw_checkpoint_is_never_consumed_as_current_input(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    transcript.write_raw_transcript_checkpoint(
        [transcript.TranscriptSegment(0.0, 1.0, "stale")],
        output / "transcript.raw.json",
    )

    result = run_supervised_transcript(
        _video(tmp_path),
        output,
        _preflight(config=_config(speaker_detection=False), hf_token=None,
                   effective_diarization_device=None),
        scheduler=_scheduler(),
        supervisor_factory=_factory([], transcript_segments=(
            transcript.TranscriptSegment(0.0, 1.0, "current"),
        )),
    )

    assert result.segments[0].text == "current"
    assert transcript.read_raw_transcript_checkpoint(
        output / "transcript.raw.json"
    )[0].text == "current"


def test_supervised_transcript_borrows_entered_supervisor_without_closing_it(
    tmp_path,
):
    output = tmp_path / "out"
    supervisor = _FakeSupervisor(output)
    preflight = _preflight(
        config=_config(speaker_detection=False),
        hf_token=None,
        effective_diarization_device=None,
    )

    with supervisor:
        result = run_supervised_transcript(
            _video(tmp_path),
            output,
            preflight,
            scheduler=_scheduler(),
            supervisor=supervisor,
        )

        assert result.segments[0].text == "hello"
        assert supervisor.events.count("enter") == 1
        assert "exit" not in supervisor.events

    assert supervisor.events[-1] == "exit"


def test_progress_output_is_stably_stage_prefixed(capsys):
    print_stage_progress(StageProgress("diarization", "inference", "cpu"))

    assert capsys.readouterr().out == "[diarization] inference: cpu\n"


def test_schedule_output_includes_resource_probe_source(capsys):
    decision = StageScheduler(
        resource_probe=lambda: RuntimeResources(
            8,
            16 * GIB,
            source="macos-memory-pressure",
        )
    ).decide(
        (
            transcript_cli_module.transcription_demand(
                "medium",
                backend="mlx",
            ),
            transcript_cli_module.diarization_demand("cpu"),
        )
    )

    _print_schedule(decision)

    assert "source=macos-memory-pressure" in capsys.readouterr().out


def test_cmd_extract_preflight_failure_happens_before_output_creation(
    tmp_path,
    monkeypatch,
):
    video = _video(tmp_path)
    output = tmp_path / "out"
    monkeypatch.setattr(
        transcript,
        "current_runtime_platform",
        lambda: LINUX,
    )
    args = SimpleNamespace(
        video=str(video),
        output=str(output),
        transcript_only=True,
        frames_only=False,
        whisper_model="medium",
        transcript_format="json",
        transcription_backend="mlx",
        diarization_device="auto",
        stage_concurrency="auto",
        no_speaker_detection=True,
    )

    with pytest.raises(SystemExit) as raised:
        cli.cmd_extract(args)

    assert raised.value.code == 2
    assert not output.exists()


def test_transcript_only_output_file_is_a_controlled_session_error(tmp_path):
    video = _video(tmp_path)
    output = tmp_path / "not-a-directory"
    output.write_text("user owned", encoding="utf-8")
    preflight = _preflight(
        config=_config(speaker_detection=False),
        hf_token=None,
        effective_diarization_device=None,
    )

    with pytest.raises(OutputSessionError, match="failed to initialize"):
        run_supervised_transcript(
            video,
            output,
            preflight,
            scheduler=_scheduler(),
        )

    assert output.read_text(encoding="utf-8") == "user owned"


def test_cmd_extract_reports_explicit_output_creation_failure(
    tmp_path,
    monkeypatch,
    capsys,
):
    video = _video(tmp_path)
    _stub_cli_media_preflight(monkeypatch)
    monkeypatch.setattr(cli, "_preflight_transcript", lambda _args: object())
    monkeypatch.setattr(
        cli,
        "_resolve_out_dir",
        lambda *_args: (_ for _ in ()).throw(PermissionError("read-only parent")),
    )
    args = SimpleNamespace(
        video=str(video),
        output=str(tmp_path / "blocked" / "out"),
        transcript_only=True,
        frames_only=False,
        whisper_model="medium",
        transcript_format="json",
        transcription_backend="auto",
        diarization_device="auto",
        stage_concurrency="auto",
        no_speaker_detection=True,
        sample_interval=0.75,
        pass1_clusters=9,
        similarity_threshold=0.85,
        max_output_frames=None,
        verbose_trace=False,
        debug_qa_targets=None,
    )

    with pytest.raises(SystemExit) as raised:
        cli.cmd_extract(args)

    assert raised.value.code == 1
    assert "could not create output directory" in capsys.readouterr().err


def test_cmd_extract_presents_transcription_failure_and_preserves_prior_final(
    tmp_path,
    monkeypatch,
    capsys,
):
    video = _video(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    final = output / "transcript.json"
    final.write_text("previous final", encoding="utf-8")
    failure = StageWorkerError(
        "transcription",
        "whisper failed",
        exitcode=1,
        error_type="RuntimeError",
    )
    monkeypatch.setattr(cli, "_preflight_transcript", lambda _args: object())
    _stub_cli_media_preflight(monkeypatch)
    monkeypatch.setattr(
        cli,
        "_run_transcript",
        lambda *_args: (_ for _ in ()).throw(failure),
    )
    args = SimpleNamespace(
        video=str(video),
        output=str(output),
        transcript_only=True,
        frames_only=False,
        whisper_model="medium",
        transcript_format="json",
        transcription_backend="whisper",
        diarization_device="auto",
        stage_concurrency="auto",
        no_speaker_detection=True,
        sample_interval=0.75,
        pass1_clusters=9,
        similarity_threshold=0.85,
        max_output_frames=None,
        verbose_trace=False,
        debug_qa_targets=None,
    )

    with pytest.raises(SystemExit) as raised:
        cli.cmd_extract(args)

    assert raised.value.code == 1
    assert "Error: transcription worker failed" in capsys.readouterr().err
    assert final.read_text(encoding="utf-8") == "previous final"
