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


@pytest.mark.hardware
@pytest.mark.cuda
def test_real_paddle_cuda_ocr_smoke():
    if os.environ.get("KEYFRAME_PADDLE_CUDA_SMOKE") != "1":
        pytest.skip("set KEYFRAME_PADDLE_CUDA_SMOKE=1 to opt into this smoke test")

    import numpy as np

    from keyframe.frames import _load_ocr_engine
    from keyframe.paddle_runtime import ensure_paddle_runtime

    selection = ensure_paddle_runtime(force=True)
    if selection.status != "gpu":
        pytest.fail(f"Paddle CUDA smoke selected {selection.status}: {selection.reason}")

    engine = _load_ocr_engine(selection.ocr_device, selection)
    image = np.full((64, 256, 3), 255, dtype=np.uint8)
    result = engine.predict(image)

    assert engine.device.startswith("gpu:")
    assert isinstance(result, list)
