import json
import tomllib
from pathlib import Path

import pytest
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from keyframe import transcript


ROOT = Path(__file__).parents[1]
SUPPORTED_MAC = transcript.RuntimePlatform("Darwin", "arm64", 14, 23)


class FakeLocalEntryNotFoundError(Exception):
    pass


@pytest.mark.parametrize(
    ("runtime_platform", "supported"),
    [
        (transcript.RuntimePlatform("Darwin", "arm64", 14, 23), True),
        (transcript.RuntimePlatform("Darwin", "arm64", 26, 25), True),
        (transcript.RuntimePlatform("Darwin", "arm64", None, 23), True),
        (transcript.RuntimePlatform("Darwin", "arm64", 13, 22), False),
        (transcript.RuntimePlatform("Darwin", "x86_64", 15, 24), False),
        (transcript.RuntimePlatform("Linux", "aarch64", None, 6), False),
        (transcript.RuntimePlatform("Windows", "AMD64", None, 10), False),
    ],
)
def test_runtime_platform_limits_mlx_to_supported_apple_silicon(
    runtime_platform,
    supported,
):
    assert runtime_platform.supports_mlx_whisper is supported
    assert transcript.resolve_transcription_backend("auto", runtime_platform) == (
        "mlx" if supported else "whisper"
    )


def test_explicit_mlx_preflight_fails_before_import_or_download(monkeypatch, tmp_path):
    imported = []
    monkeypatch.setattr(
        transcript,
        "_load_mlx_runtime",
        lambda: imported.append(True) or pytest.fail("MLX must not be imported"),
    )

    with pytest.raises(transcript.UnsupportedTranscriptionBackendError):
        transcript._extract_with_mlx(
            tmp_path / "recording.mp4",
            "medium",
            transcript.RuntimePlatform("Linux", "x86_64", None, 6),
        )

    assert imported == []


def test_non_mlx_auto_backend_never_requests_mlx_weights(monkeypatch, tmp_path):
    calls = []
    video = tmp_path / "recording.mp4"
    monkeypatch.setattr(
        transcript,
        "_extract_with_mlx",
        lambda *_args, **_kwargs: pytest.fail("MLX must not run"),
    )
    monkeypatch.setattr(
        transcript,
        "_extract_with_whisper",
        lambda path, model: calls.append((path, model))
        or transcript.TranscriptionResult((), "en", {}),
    )

    result = transcript._extract_with_transcription_backend(
        video,
        "medium",
        "auto",
        transcript.RuntimePlatform("Windows", "AMD64", None, 10),
    )

    assert result.segments == ()
    assert result.language == "en"
    assert result.metadata == {}
    assert calls == [(video, "medium")]


def test_model_sizes_map_to_immutable_mlx_revisions():
    assert transcript.MLX_MODEL_SPECS == {
        "tiny": transcript.MLXModelSpec(
            "mlx-community/whisper-tiny-mlx",
            "6caf9c55601caafbe6508a8b0d216bdf4783c4e8",
        ),
        "base": transcript.MLXModelSpec(
            "mlx-community/whisper-base-mlx",
            "1e3e249fb8d01c655324bd6841b1deadffd6d04c",
        ),
        "small": transcript.MLXModelSpec(
            "mlx-community/whisper-small-mlx",
            "45f3915923c7a79a5a5b5a7d909d39aeb0e5630e",
        ),
        "medium": transcript.MLXModelSpec(
            "mlx-community/whisper-medium-mlx",
            "7fc08c4eac4c316526498f147dfdee6f6303f975",
        ),
        "large": transcript.MLXModelSpec(
            "mlx-community/whisper-large-mlx",
            "9310354911111f2406ead1478e0139d9c6ea3acc",
        ),
    }


def test_mlx_adapter_resolves_cached_pinned_snapshot_and_preserves_precision(
    monkeypatch,
    tmp_path,
    capsys,
):
    calls = []
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    def snapshot_download(**kwargs):
        calls.append(("download", kwargs))
        return str(model_dir)

    def load_model(path, dtype):
        calls.append(("load", path, dtype))

    def transcribe_mlx(path, **kwargs):
        calls.append(("transcribe", path, kwargs))
        return {
            "language": "fr",
            "text": "ignored aggregate text",
            "segments": [
                {
                    "start": 0.123456789,
                    "end": 1.987654321,
                    "text": "  déjà vu  ",
                    "tokens": [1, 2, 3],
                }
            ],
        }

    runtime = transcript.MLXRuntime(
        snapshot_download=snapshot_download,
        load_model=load_model,
        transcribe=transcribe_mlx,
        float16="float16",
        local_entry_not_found_error=FakeLocalEntryNotFoundError,
    )
    monkeypatch.setattr(transcript, "_load_mlx_runtime", lambda: runtime)

    result = transcript._extract_with_mlx(
        tmp_path / "recording.mp4",
        "medium",
        SUPPORTED_MAC,
    )

    spec = transcript.MLX_MODEL_SPECS["medium"]
    assert calls == [
        (
            "download",
            {
                "repo_id": spec.repository,
                "revision": spec.revision,
                "local_files_only": True,
            },
        ),
        ("load", str(model_dir), "float16"),
        (
            "transcribe",
            str(tmp_path / "recording.mp4"),
            {
                "path_or_hf_repo": str(model_dir),
                "verbose": False,
                "word_timestamps": False,
            },
        ),
    ]
    assert result.language == "fr"
    assert result.segments == (
        transcript.TranscriptSegment(0.123456789, 1.987654321, "déjà vu"),
    )
    assert result.segments[0].to_dict() == {
        "start": 0.123456789,
        "end": 1.987654321,
        "text": "déjà vu",
    }
    assert result.metadata["model_repository"] == spec.repository
    assert result.metadata["model_revision"] == spec.revision
    assert result.metadata["model_resolution_source"] == "local-hit"
    assert 0 <= result.metadata["model_resolution_seconds"] < 1
    with pytest.raises(TypeError):
        result.metadata["model_resolution_source"] = "changed"
    output = capsys.readouterr().out
    assert "Importing MLX runtime" in output
    assert "Resolving cached MLX model" in output
    assert "downloading pinned snapshot" not in output
    assert output.index("Resolving cached MLX model") < output.index(
        "Loading MLX model"
    )
    assert output.index("Loading MLX model") < output.index("Transcribing with MLX")


def test_mlx_cache_miss_permits_one_online_resolution(monkeypatch, tmp_path, capsys):
    calls = []
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            raise FakeLocalEntryNotFoundError("not cached")
        return str(model_dir)

    runtime = transcript.MLXRuntime(
        snapshot_download=snapshot_download,
        load_model=lambda *_args: None,
        transcribe=lambda *_args, **_kwargs: {"segments": [], "language": "en"},
        float16="float16",
        local_entry_not_found_error=FakeLocalEntryNotFoundError,
    )
    monkeypatch.setattr(transcript, "_load_mlx_runtime", lambda: runtime)

    result = transcript._extract_with_mlx(
        tmp_path / "recording.mp4",
        "medium",
        SUPPORTED_MAC,
    )

    spec = transcript.MLX_MODEL_SPECS["medium"]
    assert calls == [
        {
            "repo_id": spec.repository,
            "revision": spec.revision,
            "local_files_only": True,
        },
        {"repo_id": spec.repository, "revision": spec.revision},
    ]
    assert result.metadata["model_resolution_source"] == "downloaded"
    assert 0 <= result.metadata["model_resolution_seconds"] < 1
    output = capsys.readouterr().out
    assert output.index("Resolving cached MLX model") < output.index(
        "downloading pinned snapshot"
    )


@pytest.mark.parametrize("failure", [PermissionError("denied"), OSError("broken cache")])
def test_mlx_local_resolution_failures_do_not_retry_online(
    monkeypatch,
    tmp_path,
    failure,
):
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        raise failure

    runtime = transcript.MLXRuntime(
        snapshot_download=snapshot_download,
        load_model=lambda *_args: pytest.fail("invalid snapshots must not load"),
        transcribe=lambda *_args, **_kwargs: pytest.fail("invalid snapshots must not run"),
        float16="float16",
        local_entry_not_found_error=FakeLocalEntryNotFoundError,
    )
    monkeypatch.setattr(transcript, "_load_mlx_runtime", lambda: runtime)

    with pytest.raises(transcript.MLXModelAcquisitionError) as raised:
        transcript._extract_with_mlx(
            tmp_path / "recording.mp4",
            "medium",
            SUPPORTED_MAC,
        )

    assert raised.value.__cause__ is failure
    assert len(calls) == 1
    assert calls[0]["local_files_only"] is True


@pytest.mark.parametrize("resolved_kind", ["none", "missing", "file"])
def test_mlx_rejects_malformed_or_non_directory_snapshot_paths(
    monkeypatch,
    tmp_path,
    resolved_kind,
):
    if resolved_kind == "none":
        resolved = None
    elif resolved_kind == "missing":
        resolved = tmp_path / "missing"
    else:
        resolved = tmp_path / "model-file"
        resolved.write_text("not a model directory", encoding="utf-8")
    calls = []
    runtime = transcript.MLXRuntime(
        snapshot_download=lambda **kwargs: calls.append(kwargs) or resolved,
        load_model=lambda *_args: pytest.fail("invalid snapshots must not load"),
        transcribe=lambda *_args, **_kwargs: pytest.fail("invalid snapshots must not run"),
        float16="float16",
        local_entry_not_found_error=FakeLocalEntryNotFoundError,
    )
    monkeypatch.setattr(transcript, "_load_mlx_runtime", lambda: runtime)

    with pytest.raises(transcript.MLXModelAcquisitionError):
        transcript._extract_with_mlx(
            tmp_path / "recording.mp4",
            "medium",
            SUPPORTED_MAC,
        )

    assert len(calls) == 1
    assert calls[0]["local_files_only"] is True


@pytest.mark.parametrize(
    ("failing_stage", "expected_error"),
    [
        ("acquire", transcript.MLXModelAcquisitionError),
        ("load", transcript.MLXModelLoadError),
        ("infer", transcript.MLXInferenceError),
        ("normalize", transcript.MLXInferenceError),
    ],
)
def test_mlx_failures_are_typed_and_auto_fallback_eligible(
    monkeypatch,
    tmp_path,
    failing_stage,
    expected_error,
):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    def fail_if(stage, value):
        if failing_stage == stage:
            raise RuntimeError(stage)
        return value

    runtime = transcript.MLXRuntime(
        snapshot_download=lambda **_kwargs: fail_if("acquire", str(model_dir)),
        load_model=lambda *_args: fail_if("load", None),
        transcribe=lambda *_args, **_kwargs: fail_if(
            "infer",
            {"segments": "malformed"} if failing_stage == "normalize" else {"segments": []},
        ),
        float16="float16",
        local_entry_not_found_error=FakeLocalEntryNotFoundError,
    )
    monkeypatch.setattr(transcript, "_load_mlx_runtime", lambda: runtime)

    with pytest.raises(expected_error) as error:
        transcript._extract_with_mlx(tmp_path / "recording.mp4", "tiny", SUPPORTED_MAC)

    assert transcript.is_auto_fallback_eligible(error.value)


def test_import_failure_is_typed_and_auto_fallback_eligible(monkeypatch, tmp_path):
    error = transcript.MLXImportError("missing")
    monkeypatch.setattr(
        transcript,
        "_load_mlx_runtime",
        lambda: (_ for _ in ()).throw(error),
    )

    with pytest.raises(transcript.MLXImportError) as raised:
        transcript._extract_with_mlx(tmp_path / "recording.mp4", "tiny", SUPPORTED_MAC)

    assert raised.value is error
    assert transcript.is_auto_fallback_eligible(raised.value)


def test_cancellation_and_output_failures_are_not_auto_fallback_eligible(
    monkeypatch,
    tmp_path,
):
    cancelled = transcript.TranscriptionCancelled("stop")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    runtime = transcript.MLXRuntime(
        snapshot_download=lambda **_kwargs: str(model_dir),
        load_model=lambda *_args: None,
        transcribe=lambda *_args, **_kwargs: (_ for _ in ()).throw(cancelled),
        float16="float16",
        local_entry_not_found_error=FakeLocalEntryNotFoundError,
    )
    monkeypatch.setattr(transcript, "_load_mlx_runtime", lambda: runtime)

    with pytest.raises(transcript.TranscriptionCancelled) as raised:
        transcript._extract_with_mlx(tmp_path / "recording.mp4", "tiny", SUPPORTED_MAC)

    assert raised.value is cancelled
    assert not transcript.is_auto_fallback_eligible(raised.value)
    assert not transcript.is_auto_fallback_eligible(
        transcript.TranscriptOutputError("disk full")
    )
    assert not transcript.is_auto_fallback_eligible(KeyboardInterrupt())


def test_requested_and_effective_backend_are_logged(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        transcript,
        "_extract_with_whisper",
        lambda *_args, **_kwargs: transcript.TranscriptionResult((), "en", {}),
    )

    transcript._extract_with_transcription_backend(
        tmp_path / "recording.mp4",
        "small",
        "auto",
        transcript.RuntimePlatform("Linux", "x86_64", None, 6),
    )

    assert "requested=auto, effective=whisper" in capsys.readouterr().out


def _project_dependencies():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    return project["dependencies"]


def test_shared_dependency_contract_is_exact():
    dependencies = set(_project_dependencies())
    assert {
        "torch~=2.8.0",
        "torchaudio~=2.8.0",
        "torchvision~=0.23.0",
        "whisperx==3.8.6",
        "huggingface-hub>=0.34,<1",
        "transformers>=4.50,<5",
    } <= dependencies


@pytest.mark.parametrize(
    ("sys_platform", "platform_system", "platform_machine", "platform_release", "selected"),
    [
        ("darwin", "Darwin", "arm64", "23.0.0", True),
        ("darwin", "Darwin", "arm64", "22.6.0", False),
        ("darwin", "Darwin", "x86_64", "24.0.0", False),
        ("linux", "Linux", "x86_64", "6.12.0", False),
        ("win32", "Windows", "AMD64", "10", False),
    ],
)
def test_mlx_dependency_markers_select_only_supported_darwin_arm64(
    sys_platform,
    platform_system,
    platform_machine,
    platform_release,
    selected,
):
    requirements = {
        canonicalize_name(requirement.name): requirement
        for requirement in map(Requirement, _project_dependencies())
    }
    environment = default_environment()
    environment.update(
        {
            "sys_platform": sys_platform,
            "platform_system": platform_system,
            "platform_machine": platform_machine,
            "platform_release": platform_release,
        }
    )

    for dependency in ("mlx", "mlx-whisper"):
        requirement = requirements[dependency]
        assert requirement.marker is not None
        assert requirement.marker.evaluate(environment) is selected


def test_benchmark_baseline_tracks_revisions_timings_quality_and_thresholds():
    baseline = json.loads(
        (ROOT / "tests/fixtures/transcription-benchmark-baseline.json").read_text(
            encoding="utf-8"
        )
    )

    assert baseline["recording"]["duration_seconds"] == 988.75
    assert baseline["model"]["mlx_revision"] == transcript.MLX_MODEL_SPECS["medium"].revision
    assert baseline["runs"]["whisper_cpu_fp32"]["wall_time_seconds"] == 225.43
    assert baseline["runs"]["mlx_whisper"]["wall_time_seconds"] == 75.38
    assert baseline["runs"]["mlx_whisper"]["normalized_word_agreement_vs_cpu"] == 0.99443
    assert baseline["quality_thresholds"]["minimum_normalized_word_agreement_vs_cpu"] == 0.99
    assert baseline["quality_thresholds"]["require_no_long_form_collapse"] is True
