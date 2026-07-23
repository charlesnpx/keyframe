from __future__ import annotations

import math
import subprocess
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

import keyframe.frame_preflight as frame_preflight
import keyframe.media_preflight as media_preflight
from keyframe import cli
from keyframe.frame_preflight import (
    FramePreflightError,
    FrameRuntimePlatform,
    preflight_frame_runtime,
    resolve_frame_execution_device,
)
from keyframe.media_preflight import (
    ExtractionMode,
    MediaPreflightError,
    MediaProbeResult,
    MediaStream,
    parse_ffprobe_payload,
    probe_media,
    resolve_extraction_mode,
    resolve_readable_media_file,
)
from keyframe.pipeline.config import KeyframeExtractionConfig
from keyframe.pipeline.qa_targets import (
    load_targets,
    normalize_targets,
    write_debug_qa_trace,
)


def _video_stream(
    *,
    codec: str | None = "h264",
    width: int = 1920,
    height: int = 1080,
    attached: bool = False,
) -> dict:
    return {
        "codec_type": "video",
        "codec_name": codec,
        "width": width,
        "height": height,
        "disposition": {"attached_pic": int(attached)},
    }


def _audio_stream(*, codec: str | None = "aac", channels: int = 2) -> dict:
    return {
        "codec_type": "audio",
        "codec_name": codec,
        "channels": channels,
    }


def _probe(*streams: dict) -> MediaProbeResult:
    return parse_ffprobe_payload({"streams": list(streams)})


@pytest.mark.parametrize(
    ("streams", "frames_only", "transcript_only", "expected"),
    [
        (
            (_video_stream(), _audio_stream()),
            False,
            False,
            ExtractionMode(True, True),
        ),
        (
            (_video_stream(),),
            False,
            False,
            ExtractionMode(
                True,
                False,
                "no usable audio stream; running frames-only extraction",
            ),
        ),
        (
            (_audio_stream(),),
            False,
            False,
            ExtractionMode(
                False,
                True,
                "no usable video stream; running transcript-only extraction",
            ),
        ),
        (
            (_video_stream(), _audio_stream()),
            True,
            False,
            ExtractionMode(True, False),
        ),
        (
            (_video_stream(), _audio_stream()),
            False,
            True,
            ExtractionMode(False, True),
        ),
        (
            (_video_stream(attached=True), _audio_stream()),
            False,
            False,
            ExtractionMode(
                False,
                True,
                "no usable video stream; running transcript-only extraction",
            ),
        ),
    ],
)
def test_stream_routing_resolves_default_and_explicit_modes(
    streams,
    frames_only,
    transcript_only,
    expected,
):
    assert resolve_extraction_mode(
        _probe(*streams),
        frames_only=frames_only,
        transcript_only=transcript_only,
    ) == expected


def test_stream_routing_accepts_any_usable_stream_among_multiple_candidates():
    result = _probe(
        _video_stream(width=0),
        _video_stream(codec="unknown"),
        _video_stream(codec="hevc", width=1280, height=720),
        _audio_stream(channels=0),
        _audio_stream(codec="opus", channels=1),
    )

    assert result.has_usable_video
    assert result.has_usable_audio


@pytest.mark.parametrize(
    ("streams", "frames_only", "transcript_only", "message"),
    [
        ((), False, False, "neither"),
        ((_audio_stream(),), True, False, "requires a usable"),
        ((_video_stream(),), False, True, "requires a usable"),
        ((_video_stream(attached=True),), True, False, "requires a usable"),
        (
            (_video_stream(), _audio_stream()),
            True,
            True,
            "cannot be used together",
        ),
    ],
)
def test_stream_routing_rejects_unavailable_or_conflicting_modes(
    streams,
    frames_only,
    transcript_only,
    message,
):
    with pytest.raises(MediaPreflightError, match=message):
        resolve_extraction_mode(
            _probe(*streams),
            frames_only=frames_only,
            transcript_only=transcript_only,
        )


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {"streams": {}},
        {"streams": [None]},
        {"streams": [{}]},
        {"streams": [{"codec_type": 1}]},
        {"streams": [{"codec_type": "video", "codec_name": 3}]},
        {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1.5,
                    "height": 100,
                }
            ]
        },
        {
            "streams": [
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "channels": True,
                }
            ]
        },
        {
            "streams": [
                {
                    **_video_stream(),
                    "disposition": {"attached_pic": 2},
                }
            ]
        },
    ],
)
def test_ffprobe_payload_rejects_malformed_shapes_and_fields(payload):
    with pytest.raises(MediaPreflightError):
        parse_ffprobe_payload(payload)


def test_ffprobe_uses_exact_timeout_and_parses_json():
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout='{"streams":[{"codec_type":"audio","codec_name":"aac","channels":2}]}',
            stderr="",
        )

    result = probe_media("/tmp/input.m4a", runner=runner)

    assert result.has_usable_audio
    assert calls[0][1]["timeout"] == 15.0
    assert calls[0][0][:6] == (
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-of",
        "json",
    )


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (subprocess.TimeoutExpired("ffprobe", 15), "timed out after 15 seconds"),
        (FileNotFoundError("ffprobe"), "not installed"),
        (PermissionError("blocked"), "could not be executed"),
    ],
)
def test_ffprobe_execution_failures_are_controlled(failure, message):
    def runner(*_args, **_kwargs):
        raise failure

    with pytest.raises(MediaPreflightError, match=message):
        probe_media("/tmp/input.mp4", runner=runner)


@pytest.mark.parametrize(
    ("completed", "message"),
    [
        (
            subprocess.CompletedProcess([], 7, stdout="", stderr="bad media"),
            "exit status 7",
        ),
        (
            subprocess.CompletedProcess([], 0, stdout="{", stderr=""),
            "malformed JSON",
        ),
    ],
)
def test_ffprobe_failed_status_and_malformed_json_are_controlled(
    completed,
    message,
):
    with pytest.raises(MediaPreflightError, match=message):
        probe_media(
            "/tmp/input.mp4",
            runner=lambda *_args, **_kwargs: completed,
        )


def test_input_preflight_resolves_symlinks_to_readable_regular_files(tmp_path):
    target = tmp_path / "recording.mp4"
    target.write_bytes(b"media")
    alias = tmp_path / "alias.mp4"
    alias.symlink_to(target)

    assert resolve_readable_media_file(alias) == target.resolve()


def test_input_preflight_rejects_directories_and_unreadable_files(
    tmp_path,
    monkeypatch,
):
    with pytest.raises(MediaPreflightError, match="regular file"):
        resolve_readable_media_file(tmp_path)

    target = tmp_path / "recording.mp4"
    target.write_bytes(b"media")
    real_open = Path.open

    def deny_target(path, *args, **kwargs):
        if path == target.resolve():
            raise PermissionError("denied")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", deny_target)
    with pytest.raises(MediaPreflightError, match="not readable"):
        resolve_readable_media_file(target)


def test_cli_resolves_input_file_before_invoking_ffprobe(tmp_path, monkeypatch):
    output = tmp_path / "out"
    calls = []
    monkeypatch.setattr(
        media_preflight,
        "probe_media",
        lambda path: calls.append(path) or pytest.fail("ffprobe must not run"),
    )

    with pytest.raises(cli.ExtractionPreflightError, match="regular file"):
        cli._preflight_extract(_cli_args(tmp_path, output))

    assert calls == []
    assert not output.exists()


@pytest.mark.parametrize(
    ("runtime", "imports"),
    [
        (FrameRuntimePlatform("Darwin", "arm64"), ["keyframe.frames"]),
        (
            FrameRuntimePlatform("Linux", "x86_64"),
            ["keyframe.frames", "paddleocr"],
        ),
    ],
)
def test_frame_preflight_imports_supported_runtime_without_constructing_ocr(
    runtime,
    imports,
):
    calls = []

    class ConstructorMustNotRun:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("PaddleOCR was constructed")

    def importer(name):
        calls.append(name)
        if name == "paddleocr":
            return SimpleNamespace(PaddleOCR=ConstructorMustNotRun)
        return SimpleNamespace()

    assert preflight_frame_runtime(runtime, importer=importer) == runtime
    assert calls == imports


@pytest.mark.parametrize(
    "runtime",
    [
        FrameRuntimePlatform("Windows", "AMD64"),
        FrameRuntimePlatform("Darwin", "x86_64"),
        FrameRuntimePlatform("Linux", "aarch64"),
    ],
)
def test_frame_preflight_rejects_unsupported_platforms_before_import(runtime):
    with pytest.raises(FramePreflightError, match="supported only"):
        preflight_frame_runtime(
            runtime,
            importer=lambda _name: pytest.fail("must not import frame stack"),
        )


def test_linux_frame_preflight_gives_reinstall_and_transcript_guidance():
    def importer(name):
        if name == "paddleocr":
            raise ModuleNotFoundError("paddleocr")
        return SimpleNamespace()

    with pytest.raises(FramePreflightError) as raised:
        preflight_frame_runtime(
            FrameRuntimePlatform("Linux", "x86_64"),
            importer=importer,
        )

    assert "Reinstall" in str(raised.value)
    assert "--transcript-only" in str(raised.value)


@pytest.mark.parametrize(
    ("runtime", "mps", "cuda", "expected"),
    [
        (FrameRuntimePlatform("Darwin", "arm64"), True, False, "mps"),
        (FrameRuntimePlatform("Darwin", "arm64"), False, False, "cpu"),
        (FrameRuntimePlatform("Linux", "x86_64"), False, True, "cuda"),
        (FrameRuntimePlatform("Linux", "x86_64"), False, False, "cpu"),
    ],
)
def test_frame_execution_device_is_resolved_during_preflight(
    runtime,
    mps,
    cuda,
    expected,
):
    torch = SimpleNamespace(
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: mps),
        ),
        cuda=SimpleNamespace(is_available=lambda: cuda),
    )

    assert resolve_frame_execution_device(
        runtime,
        importer=lambda name: torch if name == "torch" else pytest.fail(name),
    ) == expected


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sample_interval": 0}, "sample_interval"),
        ({"sample_interval": -1}, "sample_interval"),
        ({"sample_interval": math.nan}, "sample_interval"),
        ({"sample_interval": math.inf}, "sample_interval"),
        ({"sample_interval": 10**10000}, "sample_interval"),
        ({"sample_interval": True}, "sample_interval"),
        ({"pass1_clusters": 0}, "pass1_clusters"),
        ({"pass1_clusters": 65}, "pass1_clusters"),
        ({"pass1_clusters": True}, "pass1_clusters"),
        ({"pass1_clusters": 1.5}, "pass1_clusters"),
        ({"max_output_frames": 0}, "max_output_frames"),
        ({"max_output_frames": True}, "max_output_frames"),
        ({"max_output_frames": 1.5}, "max_output_frames"),
        ({"max_clustering_memory_mb": 0}, "max_clustering_memory_mb"),
        ({"max_clustering_memory_mb": True}, "max_clustering_memory_mb"),
        ({"max_frame_cache_mb": -1}, "max_frame_cache_mb"),
        ({"max_frame_cache_mb": 1.5}, "max_frame_cache_mb"),
        ({"similarity_threshold": -0.01}, "similarity_threshold"),
        ({"similarity_threshold": 1.01}, "similarity_threshold"),
        ({"similarity_threshold": math.nan}, "similarity_threshold"),
        ({"similarity_threshold": math.inf}, "similarity_threshold"),
        ({"similarity_threshold": 10**10000}, "similarity_threshold"),
        ({"similarity_threshold": True}, "similarity_threshold"),
    ],
)
def test_library_frame_config_rejects_non_strict_numeric_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        KeyframeExtractionConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "sample_interval": 0.001,
            "pass1_clusters": 1,
            "max_output_frames": 1,
            "max_clustering_memory_mb": 1,
            "max_frame_cache_mb": 1,
            "similarity_threshold": 0,
        },
        {
            "sample_interval": 1,
            "pass1_clusters": 64,
            "max_output_frames": None,
            "similarity_threshold": 1,
        },
    ],
)
def test_library_frame_config_accepts_boundary_values(kwargs):
    KeyframeExtractionConfig(**kwargs)


def test_cli_numeric_parsing_and_config_reject_nan_and_fractional_integers():
    parsed = cli._parse_extract_args(
        ["recording.mp4", "--sample-interval", "nan", "--frames-only"]
    )
    with pytest.raises(ValueError, match="sample_interval"):
        cli._frame_config(parsed)

    with pytest.raises(SystemExit) as raised:
        cli._parse_extract_args(
            ["recording.mp4", "--pass1-clusters", "1.5", "--frames-only"]
        )
    assert raised.value.code == 2


def test_qa_targets_normalize_list_and_wrapped_container(tmp_path):
    expected = (
        {
            "time": 3.0,
            "label": "3",
            "tolerance": 2.25,
            "anchor_tokens": [],
        },
        {
            "time": 4.5,
            "label": "checkout",
            "tolerance": 0.5,
            "anchor_tokens": ["total", "submit"],
        },
    )
    payload = [
        {"time": 3},
        {
            "time": 4.5,
            "label": "checkout",
            "tolerance": 0.5,
            "anchor_tokens": ["total", "submit"],
        },
    ]
    path = tmp_path / "targets.json"
    path.write_text(
        __import__("json").dumps({"targets": payload}),
        encoding="utf-8",
    )

    assert normalize_targets(payload) == expected
    assert load_targets(path) == expected


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"targets": {}},
        [None],
        [{}],
        [{"time": True}],
        [{"time": -1}],
        [{"time": math.nan}],
        [{"time": math.inf}],
        [{"time": 10**10000}],
        [{"time": 1, "label": ""}],
        [{"time": 1, "label": "   "}],
        [{"time": 1, "label": 3}],
        [{"time": 1, "label": None}],
        [{"time": 1, "tolerance": 0}],
        [{"time": 1, "tolerance": -1}],
        [{"time": 1, "tolerance": True}],
        [{"time": 1, "tolerance": math.nan}],
        [{"time": 1, "tolerance": math.inf}],
        [{"time": 1, "tolerance": 10**10000}],
        [{"time": 1, "anchor_tokens": "token"}],
        [{"time": 1, "anchor_tokens": ("token",)}],
        [{"time": 1, "anchor_tokens": [""]}],
        [{"time": 1, "anchor_tokens": [1]}],
    ],
)
def test_qa_targets_reject_malformed_containers_and_fields(payload):
    with pytest.raises(ValueError):
        normalize_targets(payload)


def test_frame_config_reuses_preflight_normalized_qa_targets(tmp_path):
    path = tmp_path / "targets.json"
    path.write_text('[{"time": 3, "anchor_tokens": ["submit"]}]', encoding="utf-8")
    config = KeyframeExtractionConfig(debug_qa_targets_path=path)
    path.write_text("not valid JSON anymore", encoding="utf-8")
    output = tmp_path / "trace.json"

    write_debug_qa_trace(
        trace_records=[],
        targets=config.debug_qa_targets or (),
        video="recording.mp4",
        output_path=output,
    )

    payload = __import__("json").loads(output.read_text(encoding="utf-8"))
    assert payload["targets"][0]["label"] == "3"
    assert payload["targets"][0]["anchor_tokens"] == ["submit"]


def _cli_args(video: Path, output: Path, **overrides):
    values = {
        "video": str(video),
        "output": str(output),
        "transcript_only": False,
        "frames_only": True,
        "sample_interval": 0.5,
        "pass1_clusters": 15,
        "similarity_threshold": 0.85,
        "max_output_frames": None,
        "max_clustering_memory_mb": 2048,
        "max_frame_cache_mb": 8192,
        "frame_cache_dir": None,
        "verbose_trace": False,
        "debug_qa_targets": None,
        "whisper_model": "medium",
        "transcript_format": "txt",
        "transcription_backend": "auto",
        "diarization_device": "auto",
        "stage_concurrency": "auto",
        "no_speaker_detection": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _patch_media(monkeypatch, probe):
    monkeypatch.setattr(media_preflight, "probe_media", lambda _path: probe)


def _patch_frame_runtime(monkeypatch):
    monkeypatch.setattr(
        frame_preflight,
        "preflight_frame_runtime",
        lambda: FrameRuntimePlatform("Darwin", "arm64"),
    )
    monkeypatch.setattr(
        frame_preflight,
        "resolve_frame_execution_device",
        lambda _runtime: "cpu",
    )


def test_cli_preflight_resolves_all_default_stream_combinations(
    tmp_path,
    monkeypatch,
):
    video = tmp_path / "recording.mp4"
    video.write_bytes(b"media")
    output = tmp_path / "out"
    _patch_frame_runtime(monkeypatch)
    transcript = SimpleNamespace(
        runtime_platform=SimpleNamespace(supports_mlx_whisper=False),
        transcription_device="cpu",
        effective_diarization_device=None,
    )
    monkeypatch.setattr(cli, "_preflight_transcript", lambda _args: transcript)

    cases = [
        (_probe(_video_stream(), _audio_stream()), (True, True)),
        (_probe(_video_stream()), (True, False)),
        (_probe(_audio_stream()), (False, True)),
    ]
    for probe, expected in cases:
        _patch_media(monkeypatch, probe)
        result = cli._preflight_extract(
            _cli_args(
                video,
                output,
                frames_only=False,
                transcript_only=False,
            )
        )
        assert (result.do_frames, result.do_transcript) == expected
        assert not output.exists()


@pytest.mark.parametrize(
    "failure",
    [
        "probe",
        "transcript",
        "platform",
        "numeric",
        "qa",
        "output",
        "cache",
    ],
)
def test_every_cli_preflight_failure_precedes_output_cache_model_and_worker_side_effects(
    tmp_path,
    monkeypatch,
    failure,
):
    video = tmp_path / "recording.mp4"
    video.write_bytes(b"media")
    output = tmp_path / "nested" / "out"
    cache = tmp_path / "cache" / "nested"
    qa_path = tmp_path / "targets.json"
    qa_path.write_text('[{"time": 1}]', encoding="utf-8")
    args = _cli_args(
        video,
        output,
        frame_cache_dir=str(cache),
        debug_qa_targets=str(qa_path),
    )
    _patch_media(monkeypatch, _probe(_video_stream()))
    _patch_frame_runtime(monkeypatch)

    if failure == "probe":
        monkeypatch.setattr(
            media_preflight,
            "probe_media",
            lambda _path: (_ for _ in ()).throw(
                MediaPreflightError("injected probe failure")
            ),
        )
    elif failure == "transcript":
        _patch_media(monkeypatch, _probe(_audio_stream()))
        args.frames_only = False
        args.transcript_only = True
        monkeypatch.setattr(
            cli,
            "_preflight_transcript",
            lambda _args: (_ for _ in ()).throw(ValueError("bad transcript config")),
        )
    elif failure == "platform":
        monkeypatch.setattr(
            frame_preflight,
            "preflight_frame_runtime",
            lambda: (_ for _ in ()).throw(
                FramePreflightError("unsupported frame platform")
            ),
        )
    elif failure == "numeric":
        args.sample_interval = math.nan
    elif failure == "qa":
        qa_path.write_text('[{"time": -1}]', encoding="utf-8")
    elif failure == "output":
        blocker = tmp_path / "blocker"
        blocker.write_text("file", encoding="utf-8")
        args.output = str(blocker / "out")
    elif failure == "cache":
        blocker = tmp_path / "cache-blocker"
        blocker.write_text("file", encoding="utf-8")
        args.frame_cache_dir = str(blocker / "nested")

    monkeypatch.setattr(
        cli,
        "_frame_session",
        lambda *_args, **_kwargs: pytest.fail("worker/session started"),
    )
    monkeypatch.setattr(
        "keyframe.frames.ModelPreloader",
        lambda *_args, **_kwargs: pytest.fail("model constructed"),
    )
    with pytest.raises(SystemExit) as raised:
        cli.cmd_extract(args)

    assert raised.value.code == 2
    assert not output.exists()
    assert not cache.exists()


def test_transcript_only_ignores_frame_specific_paths_and_configuration(
    tmp_path,
    monkeypatch,
):
    video = tmp_path / "recording.m4a"
    video.write_bytes(b"media")
    output = tmp_path / "out"
    blocker = tmp_path / "blocker"
    blocker.write_text("file", encoding="utf-8")
    _patch_media(monkeypatch, _probe(_audio_stream()))
    monkeypatch.setattr(cli, "_preflight_transcript", lambda _args: object())
    monkeypatch.setattr(
        frame_preflight,
        "preflight_frame_runtime",
        lambda: pytest.fail("frame platform must be ignored"),
    )

    result = cli._preflight_extract(
        _cli_args(
            video,
            output,
            frames_only=False,
            transcript_only=True,
            frame_cache_dir=str(blocker / "cache"),
            debug_qa_targets=str(tmp_path / "missing-targets.json"),
        )
    )

    assert result.do_transcript
    assert not result.do_frames
    assert result.frame_config is None
    assert not output.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sample_interval", 0),
        ("sample_interval", -1),
        ("sample_interval", math.nan),
        ("sample_interval", math.inf),
        ("sample_interval", True),
        ("pass1_clusters", 0),
        ("pass1_clusters", 65),
        ("pass1_clusters", True),
        ("pass1_clusters", 1.5),
        ("max_output_frames", 0),
        ("max_output_frames", True),
        ("max_output_frames", 1.5),
        ("max_clustering_memory_mb", 0),
        ("max_clustering_memory_mb", True),
        ("max_clustering_memory_mb", 1.5),
        ("max_frame_cache_mb", -1),
        ("max_frame_cache_mb", False),
        ("max_frame_cache_mb", 1.5),
        ("similarity_threshold", -0.01),
        ("similarity_threshold", 1.01),
        ("similarity_threshold", math.nan),
        ("similarity_threshold", math.inf),
        ("similarity_threshold", True),
    ],
)
def test_transcript_only_still_rejects_invalid_numeric_configuration(
    tmp_path,
    monkeypatch,
    field,
    value,
):
    video = tmp_path / "recording.m4a"
    video.write_bytes(b"media")
    output = tmp_path / "out"
    _patch_media(monkeypatch, _probe(_audio_stream()))
    monkeypatch.setattr(cli, "_preflight_transcript", lambda _args: object())
    args = _cli_args(
        video,
        output,
        frames_only=False,
        transcript_only=True,
        **{field: value},
    )

    with pytest.raises(cli.ExtractionPreflightError, match=field):
        cli._preflight_extract(args)

    assert not output.exists()


@pytest.mark.parametrize(
    "values",
    [
        {
            "sample_interval": 0.001,
            "pass1_clusters": 1,
            "max_output_frames": 1,
            "max_clustering_memory_mb": 1,
            "max_frame_cache_mb": 1,
            "similarity_threshold": 0,
        },
        {
            "sample_interval": 1,
            "pass1_clusters": 64,
            "max_output_frames": None,
            "max_clustering_memory_mb": 2048,
            "max_frame_cache_mb": 8192,
            "similarity_threshold": 1,
        },
    ],
)
def test_cli_preflight_accepts_boundary_numeric_configuration(
    tmp_path,
    monkeypatch,
    values,
):
    video = tmp_path / "recording.mp4"
    video.write_bytes(b"media")
    output = tmp_path / "out"
    _patch_media(monkeypatch, _probe(_video_stream()))
    _patch_frame_runtime(monkeypatch)

    result = cli._preflight_extract(_cli_args(video, output, **values))

    assert result.frame_config is not None
    for field, expected in values.items():
        assert getattr(result.frame_config, field) == expected
    assert not output.exists()


def test_nested_output_and_cache_are_created_only_after_preflight_and_reuse_config(
    tmp_path,
    monkeypatch,
    capsys,
):
    video = tmp_path / "recording.mp4"
    video.write_bytes(b"media")
    output = tmp_path / "nested" / "output"
    cache = tmp_path / "nested-cache" / "candidate-cache"
    args = _cli_args(
        video,
        output,
        frames_only=False,
        frame_cache_dir=str(cache),
    )
    _patch_media(monkeypatch, _probe(_video_stream()))
    _patch_frame_runtime(monkeypatch)
    created_configs = []
    real_frame_config = cli._frame_config

    def record_config(*config_args, **config_kwargs):
        config = real_frame_config(*config_args, **config_kwargs)
        created_configs.append(config)
        assert not output.exists()
        assert not cache.exists()
        return config

    captured = []

    class Generation:
        def promote(self):
            return SimpleNamespace(final_frame_count=0, output_dir=output / "frames")

    monkeypatch.setattr(cli, "_frame_config", record_config)
    monkeypatch.setattr(cli, "_frame_session", lambda *_args, **_kwargs: nullcontext(object()))
    monkeypatch.setattr(
        cli,
        "_run_frame_generation",
        lambda _video, _output, config, _session: (
            captured.append(config) or Generation()
        ),
    )
    monkeypatch.setattr(
        "keyframe.managed_workspace.known_public_artifact_paths",
        lambda _output: (),
    )

    cli.cmd_extract(args)

    assert output.is_dir()
    assert cache.is_dir()
    assert len(created_configs) == 1
    assert captured == [created_configs[0]]
    assert "Notice: no usable audio stream; running frames-only extraction." in (
        capsys.readouterr().out
    )


@pytest.mark.parametrize(
    ("failure", "exit_code"),
    [(RuntimeError("model failed"), 1), (KeyboardInterrupt(), 130)],
)
def test_cli_maps_execution_failure_and_interruption_to_contract_exit_codes(
    tmp_path,
    monkeypatch,
    failure,
    exit_code,
):
    video = tmp_path / "recording.mp4"
    video.write_bytes(b"media")
    output = tmp_path / "out"
    args = _cli_args(video, output)
    _patch_media(monkeypatch, _probe(_video_stream()))
    _patch_frame_runtime(monkeypatch)
    monkeypatch.setattr(cli, "_frame_session", lambda *_args, **_kwargs: nullcontext(object()))

    def fail(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(cli, "_run_frame_generation", fail)

    with pytest.raises(SystemExit) as raised:
        cli.cmd_extract(args)

    assert raised.value.code == exit_code
