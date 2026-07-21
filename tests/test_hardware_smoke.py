"""Opt-in real-hardware hooks; ordinary test runs skip model execution."""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

from keyframe import transcript


def _smoke_input(environment_name: str) -> Path:
    raw_path = os.environ.get(environment_name)
    if not raw_path:
        pytest.skip(f"set {environment_name} to opt into this hardware smoke test")
    path = Path(raw_path).expanduser()
    if not path.is_file():
        pytest.fail(f"{environment_name} does not name an input file: {path}")
    return path


@pytest.mark.hardware
@pytest.mark.mlx
def test_real_mlx_transcription_smoke():
    video = _smoke_input("KEYFRAME_MLX_SMOKE_INPUT")
    runtime_platform = transcript.current_runtime_platform()
    if not runtime_platform.supports_mlx_whisper:
        pytest.skip("MLX smoke test requires Apple Silicon running macOS 14 or newer")

    model_name = os.environ.get("KEYFRAME_MLX_SMOKE_MODEL", "tiny")
    result = transcript._extract_with_mlx(
        video,
        model_name,
        runtime_platform,
    )

    model_spec = transcript.MLX_MODEL_SPECS[model_name]
    assert result.segments
    assert result.language and result.language != "unknown"
    assert result.metadata["model_repository"] == model_spec.repository
    assert result.metadata["model_revision"] == model_spec.revision
    assert result.metadata["model_resolution_source"] in {"local-hit", "downloaded"}
    assert math.isfinite(result.metadata["model_resolution_seconds"])
    assert result.metadata["model_resolution_seconds"] >= 0


@pytest.mark.hardware
@pytest.mark.cuda
def test_real_cuda_diarization_smoke():
    video = _smoke_input("KEYFRAME_CUDA_SMOKE_INPUT")
    hf_token = (os.environ.get("HF_TOKEN") or "").strip()
    if not hf_token:
        pytest.skip("CUDA diarization smoke test requires HF_TOKEN")
    if not transcript.cuda_is_available():
        pytest.skip("CUDA diarization smoke test requires an available CUDA device")

    rows = transcript._detect_speakers(video, hf_token, device="cuda")

    assert rows
