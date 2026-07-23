from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from keyframe.pipeline.config import KeyframeExtractionConfig
from keyframe.pipeline.streaming import (
    CandidateFrameCache,
    ClusteringWorkerError,
    FrameCacheError,
    _receive_worker_result,
    average_linkage_labels,
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
    cache_result = cache_candidate_frames(
        video,
        0.1,
        candidate_indices={1, 3},
        frame_indices=streamed.frame_indices,
        timestamps=streamed.timestamps,
        consumed_targets=streamed.consumed_targets,
        next_targets=streamed.next_targets,
        frame_sizes=streamed.frame_sizes,
        pixel_digests=streamed.pixel_digests,
        sampling_timing=streamed.sampling_timing,
        cache=cache,
    )
    provider = cache_result.provider
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


def test_candidate_cache_uses_the_real_directory_behind_a_valid_symlink(tmp_path):
    target = tmp_path / "cache-root"
    target.mkdir()
    alias = tmp_path / "cache-alias"
    alias.symlink_to(target, target_is_directory=True)

    cache = CandidateFrameCache(cache_root=alias, max_bytes=1024)
    try:
        assert cache.root == target.resolve()
        assert cache.path.parent == target.resolve()
    finally:
        cache.cleanup()


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
            frame_indices=streamed.frame_indices,
            timestamps=streamed.timestamps,
            consumed_targets=streamed.consumed_targets,
            next_targets=streamed.next_targets,
            frame_sizes=streamed.frame_sizes,
            pixel_digests=streamed.pixel_digests,
            sampling_timing=streamed.sampling_timing,
            cache=cache,
        )


@pytest.mark.parametrize("clusters", [0, 65])
def test_pass1_cluster_config_has_a_finite_safety_bound(clusters):
    with pytest.raises(ValueError, match="pass1_clusters"):
        KeyframeExtractionConfig(pass1_clusters=clusters)


def test_clustering_result_is_drained_before_the_worker_is_joined():
    calls = []

    class FakeWorker:
        exitcode = 0

        def is_alive(self):
            return True

        def join(self):
            calls.append("join")

    class FakeQueue:
        def get(self, timeout):
            calls.append(("get", timeout))
            return "ok", np.zeros((1024 * 1024,), dtype=np.int8)

    result = _receive_worker_result(FakeWorker(), FakeQueue(), timeout_seconds=1)

    assert result[0] == "ok"
    assert calls == [("get", 0.25)]


def test_clustering_worker_exit_without_result_is_controlled():
    class DeadWorker:
        exitcode = -9

        def is_alive(self):
            return False

        def join(self):
            pass

    class EmptyQueue:
        def get(self, timeout):
            from queue import Empty
            raise Empty

    with pytest.raises(ClusteringWorkerError, match="exited unexpectedly"):
        _receive_worker_result(DeadWorker(), EmptyQueue(), timeout_seconds=1)


def test_clustering_worker_timeout_is_controlled(monkeypatch):
    class LiveWorker:
        def is_alive(self):
            return True

        def join(self):
            pass

    class EmptyQueue:
        def get(self, timeout):
            from queue import Empty
            raise Empty

    clock = iter((0.0, 2.0))
    monkeypatch.setattr("keyframe.pipeline.streaming.time.monotonic", lambda: next(clock))

    with pytest.raises(ClusteringWorkerError, match="timed out"):
        _receive_worker_result(LiveWorker(), EmptyQueue(), timeout_seconds=1)


@pytest.mark.skipif(os.name == "nt", reason="isolated worker regression test requires Unix process limits")
def test_average_linkage_uses_a_spawned_worker_and_returns_labels():
    embeddings = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        dtype=np.float32,
    )

    labels = average_linkage_labels(embeddings, 2, max_memory_mb=2048)

    assert labels.shape == (4,)
    assert len(set(labels.tolist())) == 2
