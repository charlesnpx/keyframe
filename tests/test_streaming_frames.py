from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from keyframe.pipeline.config import KeyframeExtractionConfig
from keyframe.pipeline.streaming import (
    CandidateFrameCache,
    FrameCacheError,
    cache_candidate_frames,
    stream_video_features,
)


def _video(path: Path, *, frames: int = 5) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (24, 16))
    assert writer.isOpened()
    try:
        for index in range(frames):
            writer.write(np.full((16, 24, 3), index * 30, dtype=np.uint8))
    finally:
        writer.release()
    return path


def test_streaming_pass_keeps_compact_metadata_and_caches_only_candidates(tmp_path):
    video = _video(tmp_path / "recording.mp4")
    batches = []

    def embed(images):
        batches.append(len(images))
        return np.ones((len(images), 3), dtype=np.float32)

    streamed = stream_video_features(video, 0.1, embed_images=embed)

    assert len(streamed.timestamps) == 5
    assert streamed.clip_embeddings.shape == (5, 3)
    assert len(streamed.pixel_digests) == 5
    assert streamed.frame_metrics.full_gray_stack.shape == (0, 0, 0)
    assert streamed.frame_metrics.content_gray_stack.shape == (0, 0, 0)
    assert sum(batches) == 5

    cache = CandidateFrameCache(cache_root=tmp_path, max_bytes=1024 * 1024)
    provider = cache_candidate_frames(
        video,
        0.1,
        candidate_indices={1, 3},
        frame_sizes=streamed.frame_sizes,
        pixel_digests=streamed.pixel_digests,
        cache=cache,
    )
    first = provider[1]
    try:
        assert first.size == (24, 16)
    finally:
        first.close()
    with pytest.raises(KeyError):
        provider[0]
    cache_path = cache.path
    cache.cleanup()
    assert not cache_path.exists()


def test_candidate_cache_rejects_union_above_configured_byte_limit(tmp_path):
    video = _video(tmp_path / "recording.mp4")
    streamed = stream_video_features(
        video,
        0.1,
        embed_images=lambda images: np.ones((len(images), 2), dtype=np.float32),
    )
    cache = CandidateFrameCache(cache_root=tmp_path, max_bytes=1)

    with pytest.raises(FrameCacheError, match="max_frame_cache_mb"):
        cache_candidate_frames(
            video,
            0.1,
            candidate_indices={0},
            frame_sizes=streamed.frame_sizes,
            pixel_digests=streamed.pixel_digests,
            cache=cache,
        )


@pytest.mark.parametrize("clusters", [0, 65])
def test_pass1_cluster_config_has_a_finite_safety_bound(clusters):
    with pytest.raises(ValueError, match="pass1_clusters"):
        KeyframeExtractionConfig(pass1_clusters=clusters)
