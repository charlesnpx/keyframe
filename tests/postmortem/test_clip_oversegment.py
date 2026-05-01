import numpy as np
from PIL import Image

from keyframe import frames as frames_mod


def _blank_frames(n):
    return [Image.new("RGB", (8, 8), "white") for _ in range(n)]


def test_clip_oversegment_emits_primary_and_alt_for_diverse_candidate(monkeypatch):
    monkeypatch.setattr(frames_mod, "_laplacian_sharpness", lambda img: float(img.info["sharpness"]))
    imgs = _blank_frames(3)
    for img, sharpness in zip(imgs, [1000.0, 900.0, 800.0]):
        img.info["sharpness"] = sharpness
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.99, 0.01],
        ],
        dtype=np.float32,
    )
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    candidates = frames_mod.clip_oversegment(
        embeddings,
        [0.0, 1.0, 2.0],
        [0, 1, 2],
        1,
        imgs,
        max_reps_per_cluster=2,
    )

    assert [c["cluster_role"] for c in candidates] == ["primary", "alt"]
    assert [c["sample_idx"] for c in candidates] == [0, 1]
    assert candidates[1]["cluster_alt_reason"] == "clip"


def test_clip_oversegment_emits_single_when_no_diverse_candidate(monkeypatch):
    monkeypatch.setattr(frames_mod, "_laplacian_sharpness", lambda img: float(img.info["sharpness"]))
    imgs = _blank_frames(3)
    for img, sharpness in zip(imgs, [1000.0, 900.0, 800.0]):
        img.info["sharpness"] = sharpness
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.999, 0.001],
            [0.998, 0.002],
        ],
        dtype=np.float32,
    )
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    candidates = frames_mod.clip_oversegment(
        embeddings,
        [0.0, 1.0, 2.0],
        [0, 1, 2],
        1,
        imgs,
        max_reps_per_cluster=2,
    )

    assert [c["cluster_role"] for c in candidates] == ["single"]
    assert [c["sample_idx"] for c in candidates] == [0]
