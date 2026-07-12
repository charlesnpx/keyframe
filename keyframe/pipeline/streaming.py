"""Bounded-retention helpers for the frame-extraction pipeline.

Pass one keeps metadata and CLIP vectors, never source-resolution images.  A
second decode writes only the already-selected candidate union to a temporary,
lossless PPM cache.  The cache provider opens one image at a time, so later
model and dedupe stages do not accidentally retain all sampled frames again.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import multiprocessing as mp
import os
import errno
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable
from queue import Empty

import cv2
import numpy as np
from PIL import Image

from keyframe.visual import (
    FrameMetricTable,
    build_compact_frame_metric_table,
    build_frame_metric_table,
    content_crop,
    laplacian_sharpness,
)


# A sampled frame is briefly present as OpenCV BGR, RGB/PIL, and metric
# workspaces.  Keep the raw RGB ceiling comfortably below the process budget
# so a single pathological source frame is a controlled error rather than an
# OOM before the streaming lifecycle can release it.
_MAX_SINGLE_FRAME_BYTES = 64 * 1024 * 1024
_MAX_CLIP_BATCH_BYTES = 64 * 1024 * 1024
_MIN_TMPFS_HEADROOM_BYTES = 512 * 1024 * 1024


class FramePipelineMemoryError(RuntimeError):
    """A bounded-memory stage cannot safely admit this input."""


class FrameCacheError(RuntimeError):
    """The temporary candidate cache could not be created or verified."""


class ClusteringWorkerError(RuntimeError):
    """The isolated average-linkage worker failed without taking down the run."""


@dataclass(frozen=True)
class StreamedFeatures:
    timestamps: list[float]
    frame_indices: list[int]
    dhashes: list[int]
    pixel_digests: list[str]
    frame_sizes: list[tuple[int, int]]
    clip_embeddings: np.ndarray
    frame_metrics: FrameMetricTable


def _rgb_digest(image: Image.Image) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()


def _frame_bytes(width: int, height: int) -> int:
    if width <= 0 or height <= 0:
        raise FramePipelineMemoryError(f"invalid frame dimensions: {width}x{height}")
    value = int(width) * int(height) * 3
    if value > _MAX_SINGLE_FRAME_BYTES:
        raise FramePipelineMemoryError(
            f"a {width}x{height} frame needs {value / (1024 * 1024):.1f} MiB of RGB memory; "
            "this exceeds the per-frame safety limit"
        )
    return value


def _stream_metric_row(image: Image.Image) -> dict[str, float]:
    """Calculate the existing low-resolution metrics for one disposable image."""
    table = build_frame_metric_table((image,), (0.0,), (0,))
    return {
        "textline_score": float(table.textline_score[0]),
        "edge_score": float(table.edge_score[0]),
        "entropy": float(table.entropy[0]),
        "dark_ratio": float(table.dark_ratio[0]),
        "bright_ratio": float(table.bright_ratio[0]),
        "visual_stddev": float(table.visual_stddev[0]),
        "visual_edge_score": float(table.visual_edge_score[0]),
        "visual_dark_ratio": float(table.visual_dark_ratio[0]),
        "visual_bright_ratio": float(table.visual_bright_ratio[0]),
        "visual_entropy": float(table.visual_entropy[0]),
        "visual_unique_buckets": float(table.visual_unique_buckets[0]),
    }


def stream_video_features(
    video_path: str | Path,
    interval_seconds: float,
    *,
    embed_images: Callable[[Sequence[Image.Image]], np.ndarray],
) -> StreamedFeatures:
    """Decode sampled frames once, retaining compact metadata only.

    ``embed_images`` is called with a byte-bounded batch.  It is intentionally
    passed in from the model wrapper so this module remains inexpensive to
    import in tests and clustering workers.
    """
    from keyframe.dedupe import compute_dhash

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0:
            raise RuntimeError("video reports an invalid frame rate")
        _frame_bytes(width, height)
        duration = total_frames / fps if total_frames > 0 else 0.0
        print(f"Video: {video_path}")
        print(f"  {width}x{height}, {fps:.1f} fps, {total_frames} frames, {duration:.1f}s")

        interval_frames = max(1, int(float(interval_seconds) * fps))
        timestamps: list[float] = []
        frame_indices: list[int] = []
        dhashes: list[int] = []
        pixel_digests: list[str] = []
        frame_sizes: list[tuple[int, int]] = []
        metric_rows: list[dict[str, float]] = []
        content_prev_delta: list[float] = []
        embeddings: list[np.ndarray] = []
        batch: list[Image.Image] = []
        batch_bytes = 0
        previous_content: np.ndarray | None = None

        def flush_batch() -> None:
            nonlocal batch, batch_bytes
            if not batch:
                return
            encoded = np.asarray(embed_images(batch), dtype=np.float32)
            if encoded.ndim != 2 or encoded.shape[0] != len(batch):
                raise RuntimeError("CLIP returned embeddings with an unexpected shape")
            embeddings.append(encoded)
            for image in batch:
                image.close()
            batch = []
            batch_bytes = 0

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % interval_frames != 0:
                frame_idx += 1
                continue

            height_now, width_now = frame.shape[:2]
            current_bytes = _frame_bytes(int(width_now), int(height_now))
            if batch and batch_bytes + current_bytes > _MAX_CLIP_BATCH_BYTES:
                flush_batch()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            timestamp = frame_idx / fps
            timestamps.append(float(timestamp))
            frame_indices.append(int(frame_idx))
            dhashes.append(int(compute_dhash(image)))
            pixel_digests.append(_rgb_digest(image))
            frame_sizes.append((int(width_now), int(height_now)))
            row = _stream_metric_row(image)
            # Representative selection historically used the source-resolution
            # Laplacian score, so retain that scalar rather than changing the
            # algorithm to the compact metric-table approximation.
            row["sharpness"] = float(laplacian_sharpness(image))
            metric_rows.append(row)
            content = np.asarray(
                content_crop(image).convert("L").resize((160, 90), Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
            content_prev_delta.append(
                float(np.mean(np.abs(previous_content - content))) if previous_content is not None else 0.0
            )
            previous_content = content
            batch.append(image)
            batch_bytes += current_bytes
            if batch_bytes >= _MAX_CLIP_BATCH_BYTES:
                flush_batch()
            frame_idx += 1

        flush_batch()
        if not embeddings:
            clip_embeddings = np.empty((0, 0), dtype=np.float32)
        else:
            clip_embeddings = np.vstack(embeddings).astype(np.float32, copy=False)
        content_next_delta = [0.0] * len(content_prev_delta)
        if len(content_prev_delta) > 1:
            content_next_delta[:-1] = content_prev_delta[1:]
        metrics = build_compact_frame_metric_table(
            metric_rows,
            timestamps=timestamps,
            frame_indices=frame_indices,
            content_prev_delta=content_prev_delta,
            content_next_delta=content_next_delta,
        )
        print(f"  Sampled {len(timestamps)} frames at {interval_seconds}s intervals")
        return StreamedFeatures(
            timestamps=timestamps,
            frame_indices=frame_indices,
            dhashes=dhashes,
            pixel_digests=pixel_digests,
            frame_sizes=frame_sizes,
            clip_embeddings=clip_embeddings,
            frame_metrics=metrics,
        )
    finally:
        cap.release()


def _ppm_header(width: int, height: int) -> bytes:
    return f"P6\n{width} {height}\n255\n".encode("ascii")


def _cache_file_size(width: int, height: int) -> int:
    return len(_ppm_header(width, height)) + _frame_bytes(width, height)


def _linux_tmpfs_memory_available(path: Path) -> int | None:
    """Return available RAM when ``path`` is on Linux tmpfs, else ``None``."""
    mounts = Path("/proc/mounts")
    meminfo = Path("/proc/meminfo")
    if not mounts.exists() or not meminfo.exists():
        return None
    try:
        resolved = path.resolve()
        candidates: list[Path] = []
        for line in mounts.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) < 3 or fields[2] != "tmpfs":
                continue
            mountpoint = Path(fields[1].replace("\\040", " "))
            try:
                resolved.relative_to(mountpoint)
            except ValueError:
                continue
            candidates.append(mountpoint)
        if not candidates:
            return None
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


class CandidateImageProvider:
    """Lazy, digest-verified access to the bounded candidate image union."""

    is_disk_backed = True

    def __init__(self, paths: Mapping[int, Path], digests: Mapping[int, str]):
        self._paths = dict(paths)
        self._digests = dict(digests)
        self._length = max(self._paths, default=-1) + 1

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, sample_idx: int) -> Image.Image:
        index = int(sample_idx)
        path = self._paths.get(index)
        if path is None:
            raise KeyError(f"sample {index} was not retained in the candidate cache")
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
        except (OSError, ValueError) as exc:
            raise FrameCacheError(f"could not read cached candidate {index}: {exc}") from exc
        if _rgb_digest(image) != self._digests[index]:
            image.close()
            raise FrameCacheError(f"cached candidate {index} failed its SHA-256 verification")
        return image


class CandidateFrameCache:
    """Private, per-run disk cache for lossless candidate frames only."""

    def __init__(self, *, cache_root: Path | None, max_bytes: int):
        root = Path(cache_root) if cache_root is not None else Path(tempfile.gettempdir())
        try:
            root.lstat()
        except OSError as exc:
            raise FrameCacheError(f"cannot inspect frame cache directory {root}: {exc}") from exc
        if not root.is_dir() or root.is_symlink():
            raise FrameCacheError(f"frame cache directory must be a real directory: {root}")
        self.root = root
        self.max_bytes = int(max_bytes)
        self.path = Path(tempfile.mkdtemp(prefix="keyframe-frame-cache-", dir=root))
        os.chmod(self.path, 0o700)
        self._paths: dict[int, Path] = {}
        self._sizes: dict[int, int] = {}

    def reserve(self, frame_sizes: Mapping[int, tuple[int, int]]) -> None:
        required = sum(_cache_file_size(width, height) for width, height in frame_sizes.values())
        if required > self.max_bytes:
            raise FrameCacheError(
                f"candidate cache requires {required / (1024 * 1024):.1f} MiB, "
                f"above max_frame_cache_mb={self.max_bytes // (1024 * 1024)}"
            )
        available = shutil.disk_usage(self.path).free
        if required > available:
            raise FrameCacheError(
                f"candidate cache requires {required / (1024 * 1024):.1f} MiB but only "
                f"{available / (1024 * 1024):.1f} MiB is available in {self.root}"
            )
        # A tmpfs cache is disk-shaped but RAM-backed.  Count its reservation
        # against currently available memory so selecting the OS temp directory
        # cannot silently reintroduce duration × source-resolution pressure.
        tmpfs_available = _linux_tmpfs_memory_available(self.path)
        if tmpfs_available is not None and required + _MIN_TMPFS_HEADROOM_BYTES > tmpfs_available:
            raise FrameCacheError(
                f"candidate cache needs {required / (1024 * 1024):.1f} MiB on tmpfs, leaving less than "
                f"{_MIN_TMPFS_HEADROOM_BYTES / (1024 * 1024):.0f} MiB of memory headroom; "
                "choose a disk-backed --frame-cache-dir or reduce the candidate cache limit"
            )
        try:
            for sample_idx, (width, height) in frame_sizes.items():
                path = self.path / f"{int(sample_idx):08d}.ppm"
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                try:
                    size = _cache_file_size(width, height)
                    if hasattr(os, "posix_fallocate"):
                        try:
                            os.posix_fallocate(fd, 0, size)
                        except OSError as exc:
                            if exc.errno not in {errno.EOPNOTSUPP, errno.ENOSYS, errno.EINVAL}:
                                raise
                            os.ftruncate(fd, size)
                    else:
                        # APFS does not expose posix_fallocate through Python.
                        # The preflight free-space admission and the checked write
                        # below provide the equivalent controlled-failure path.
                        os.ftruncate(fd, size)
                finally:
                    os.close(fd)
                self._paths[int(sample_idx)] = path
                self._sizes[int(sample_idx)] = size
        except Exception:
            self.cleanup()
            raise

    def write(self, sample_idx: int, image: Image.Image, expected_digest: str) -> None:
        index = int(sample_idx)
        path = self._paths.get(index)
        if path is None:
            return
        image = image.convert("RGB")
        try:
            actual_digest = _rgb_digest(image)
            if actual_digest != expected_digest:
                raise FrameCacheError(
                    f"video changed or decoded inconsistently while caching sample {index}; "
                    "refusing to mix passes"
                )
            header = _ppm_header(*image.size)
            payload = image.tobytes()
            expected_size = self._sizes[index]
            if len(header) + len(payload) != expected_size:
                raise FrameCacheError(f"candidate {index} changed dimensions between decode passes")
            with path.open("r+b") as output:
                output.write(header)
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
        except OSError as exc:
            raise FrameCacheError(f"could not write cached candidate {index}: {exc}") from exc
        finally:
            image.close()

    def provider(self, expected_digests: Mapping[int, str]) -> CandidateImageProvider:
        missing = sorted(set(self._paths) - set(expected_digests))
        if missing:
            raise FrameCacheError(f"cache is missing expected digests for samples: {missing}")
        return CandidateImageProvider(self._paths, expected_digests)

    def cleanup(self) -> None:
        if self.path.exists():
            shutil.rmtree(self.path, ignore_errors=True)


def cache_candidate_frames(
    video_path: str | Path,
    interval_seconds: float,
    *,
    candidate_indices: Iterable[int],
    frame_sizes: Sequence[tuple[int, int]],
    pixel_digests: Sequence[str],
    cache: CandidateFrameCache,
) -> CandidateImageProvider:
    """Second streaming decode: store just the selected lossless candidate union."""
    wanted = {int(index) for index in candidate_indices}
    if not wanted:
        return CandidateImageProvider({}, {})
    if min(wanted) < 0 or max(wanted) >= len(frame_sizes):
        raise FrameCacheError("candidate index is outside the first-pass sample table")
    sizes = {index: frame_sizes[index] for index in wanted}
    expected = {index: pixel_digests[index] for index in wanted}
    cache.reserve(sizes)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not reopen video for candidate cache: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            raise RuntimeError("video reports an invalid frame rate")
        interval_frames = max(1, int(float(interval_seconds) * fps))
        frame_idx = 0
        sample_idx = 0
        remaining = set(wanted)
        while remaining:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % interval_frames == 0:
                if sample_idx in remaining:
                    height, width = frame.shape[:2]
                    _frame_bytes(int(width), int(height))
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    cache.write(sample_idx, image, expected[sample_idx])
                    remaining.remove(sample_idx)
                sample_idx += 1
            frame_idx += 1
        if remaining:
            raise FrameCacheError(f"video ended before candidate samples could be cached: {sorted(remaining)}")
        return cache.provider(expected)
    except Exception:
        cache.cleanup()
        raise
    finally:
        cap.release()


def _cluster_worker(embeddings: np.ndarray, n_clusters: int, memory_limit_bytes: int, queue: Any) -> None:
    """Run sklearn in a disposable process so a native allocation cannot kill parent state."""
    try:
        try:
            import resource

            # RLIMIT_AS is defence in depth.  The analytic admission below is
            # the portable guard; some platforms do not enforce RSS limits.
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
        except (ImportError, AttributeError, OSError, ValueError):
            pass
        from sklearn.cluster import AgglomerativeClustering

        labels = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="average",
        ).fit_predict(embeddings)
        queue.put(("ok", labels))
    except BaseException as exc:  # worker exceptions must become controlled failures
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


def average_linkage_labels(
    embeddings: np.ndarray,
    n_clusters: int,
    *,
    max_memory_mb: int,
) -> np.ndarray:
    """Average-linkage clustering with an explicit admission bound and worker."""
    rows = int(len(embeddings))
    if rows < 2 or n_clusters < 2:
        return np.zeros((rows,), dtype=np.int64)
    if n_clusters > rows:
        raise ClusteringWorkerError("cluster count exceeds available samples")
    # sklearn's average-linkage implementation needs more than the condensed
    # distance representation.  This conservative estimate prevents a known
    # quadratic allocation from being attempted in the parent process.
    estimated = int(embeddings.nbytes) + (rows * rows * 16)
    limit = int(max_memory_mb) * 1024 * 1024
    if estimated > limit:
        raise ClusteringWorkerError(
            f"average-linkage for {rows} samples needs about {estimated / (1024 * 1024):.1f} MiB, "
            f"above max_clustering_memory_mb={max_memory_mb}"
        )

    context = mp.get_context("spawn")
    queue = context.Queue(maxsize=1)
    worker = context.Process(
        target=_cluster_worker,
        args=(np.asarray(embeddings, dtype=np.float32), int(n_clusters), limit, queue),
        daemon=True,
    )
    worker.start()
    worker.join()
    try:
        if worker.exitcode != 0:
            raise ClusteringWorkerError(
                f"average-linkage worker exited unexpectedly (status {worker.exitcode})"
            )
        try:
            status, payload = queue.get(timeout=1)
        except Empty as exc:
            raise ClusteringWorkerError("average-linkage worker returned no result") from exc
        if status != "ok":
            raise ClusteringWorkerError(f"average-linkage worker failed: {payload}")
        return np.asarray(payload, dtype=np.int64)
    finally:
        queue.close()
        queue.join_thread()
