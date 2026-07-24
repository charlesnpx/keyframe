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
import time
from typing import Any, Callable
from queue import Empty

import cv2
import numpy as np
from PIL import Image

from keyframe.media_preflight import VideoTimingMetadata
from keyframe.pipeline.sampling import (
    PRESENTATION_TIME_EPSILON_SECONDS,
    DecoderTimestampNormalizer,
    DecoderTimingRegression,
    DecoderTimingUnavailable,
    FrameTimingError,
    TargetTimeSampler,
)
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
_CLUSTER_WORKER_OVERHEAD_BYTES = 256 * 1024 * 1024
_MIN_CLUSTER_WORKER_TIMEOUT_SECONDS = 30.0
_MAX_CLUSTER_WORKER_TIMEOUT_SECONDS = 600.0


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
    consumed_targets: list[float]
    next_targets: list[float]
    dhashes: list[int]
    pixel_digests: list[str]
    frame_sizes: list[tuple[int, int]]
    source_sharpness: list[float]
    clip_embeddings: np.ndarray
    frame_metrics: FrameMetricTable
    sampling_timing: dict[str, Any]


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
        "sharpness": float(table.sharpness[0]),
    }


def stream_video_features(
    video_path: str | Path,
    interval_seconds: float,
    *,
    embed_images: Callable[[Sequence[Image.Image]], np.ndarray],
    video_timing: VideoTimingMetadata | None = None,
) -> StreamedFeatures:
    """Decode sampled frames once, retaining compact metadata only.

    ``embed_images`` is called with a byte-bounded batch.  It is intentionally
    passed in from the model wrapper so this module remains inexpensive to
    import in tests and clustering workers.
    """
    from keyframe.dedupe import compute_dhash

    def decode_once(
        *,
        timing_source: str,
        nominal_rate: float | None = None,
        fallback_reason: str | None = None,
        announce: bool,
    ) -> StreamedFeatures:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"could not open video: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width > 0 and height > 0:
            _frame_bytes(width, height)
        duration = (
            float(video_timing.duration_seconds)
            if video_timing is not None
            and video_timing.duration_seconds is not None
            else total_frames / fps
            if total_frames > 0 and fps > 0
            else 0.0
        )
        if announce:
            fps_text = f"{fps:.3f}" if fps > 0 else "unknown"
            print(f"Video: {video_path}")
            print(
                f"  {width}x{height}, {fps_text} fps, "
                f"{total_frames} frames, {duration:.1f}s"
            )

        timestamps: list[float] = []
        frame_indices: list[int] = []
        consumed_targets: list[float] = []
        next_targets: list[float] = []
        dhashes: list[int] = []
        pixel_digests: list[str] = []
        frame_sizes: list[tuple[int, int]] = []
        source_sharpness: list[float] = []
        metric_rows: list[dict[str, float]] = []
        content_prev_delta: list[float] = []
        embeddings: list[np.ndarray] = []
        batch: list[Image.Image] = []
        batch_bytes = 0
        previous_content: np.ndarray | None = None
        target_sampler = TargetTimeSampler(interval_seconds)
        decoder_timing = DecoderTimestampNormalizer()
        decoder_diagnostic_error: str | None = None

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
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                raw_decoder_seconds = (
                    float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                )
                if timing_source == "decoder_presentation_time":
                    timestamp = decoder_timing.observe(raw_decoder_seconds)
                    if timestamp is None:
                        frame_idx += 1
                        continue
                else:
                    if nominal_rate is None or nominal_rate <= 0:
                        raise FrameTimingError(
                            "nominal CFR fallback requires a positive probe rate"
                        )
                    if decoder_diagnostic_error is None:
                        try:
                            decoder_timing.observe(raw_decoder_seconds)
                        except FrameTimingError as exc:
                            decoder_diagnostic_error = str(exc)
                    timestamp = frame_idx / nominal_rate

                decision = target_sampler.consider(timestamp)
                if decision is None:
                    frame_idx += 1
                    continue

                height_now, width_now = frame.shape[:2]
                current_bytes = _frame_bytes(int(width_now), int(height_now))
                if batch and batch_bytes + current_bytes > _MAX_CLIP_BATCH_BYTES:
                    flush_batch()

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
                timestamps.append(float(decision.timestamp))
                frame_indices.append(int(frame_idx))
                consumed_targets.append(float(decision.consumed_target))
                next_targets.append(float(decision.next_target))
                dhashes.append(int(compute_dhash(image)))
                pixel_digests.append(_rgb_digest(image))
                frame_sizes.append((int(width_now), int(height_now)))
                row = _stream_metric_row(image)
                metric_rows.append(row)
                # clip_oversegment historically used source-resolution sharpness,
                # while a one-cluster scene used the compact FrameMetricTable
                # score. Retain both scalar forms without retaining the image.
                source_sharpness.append(float(laplacian_sharpness(image)))
                content = np.asarray(
                    content_crop(image)
                    .convert("L")
                    .resize((160, 90), Image.Resampling.BILINEAR),
                    dtype=np.float32,
                )
                content_prev_delta.append(
                    float(np.mean(np.abs(previous_content - content)))
                    if previous_content is not None
                    else 0.0
                )
                previous_content = content
                batch.append(image)
                batch_bytes += current_bytes
                if batch_bytes >= _MAX_CLIP_BATCH_BYTES:
                    flush_batch()
                frame_idx += 1
            if timing_source == "decoder_presentation_time":
                decoder_timing.finalize()
            flush_batch()
        finally:
            for image in batch:
                image.close()
            cap.release()

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
        probe_metadata = (
            video_timing.to_dict()
            if video_timing is not None
            else {
                "classification": "unknown",
                "reason": "no ffprobe timing metadata was supplied",
                "avg_frame_rate": None,
                "r_frame_rate": None,
                "time_base": None,
                "stream_start_seconds": None,
                "duration_seconds": None,
                "duration_source": None,
                "duration_ts": None,
                "nb_frames": None,
                "nominal_frame_rate": None,
                "expected_duration_seconds": None,
                "duration_delta_seconds": None,
                "duration_tolerance_seconds": None,
            }
        )
        sampling_timing = {
            **probe_metadata,
            "source": timing_source,
            "interval_seconds": float(interval_seconds),
            "epsilon_seconds": PRESENTATION_TIME_EPSILON_SECONDS,
            "decoder_origin_seconds": decoder_timing.origin_seconds,
            "fallback_reason": fallback_reason,
            "decoder_diagnostic_error": decoder_diagnostic_error,
            "decoded_frame_count": frame_idx,
            "sample_count": len(timestamps),
            "second_pass": None,
        }
        return StreamedFeatures(
            timestamps=timestamps,
            frame_indices=frame_indices,
            consumed_targets=consumed_targets,
            next_targets=next_targets,
            dhashes=dhashes,
            pixel_digests=pixel_digests,
            frame_sizes=frame_sizes,
            source_sharpness=source_sharpness,
            clip_embeddings=clip_embeddings,
            frame_metrics=metrics,
            sampling_timing=sampling_timing,
        )

    try:
        return decode_once(
            timing_source="decoder_presentation_time",
            announce=True,
        )
    except (DecoderTimingUnavailable, DecoderTimingRegression) as exc:
        confirmed_rate = (
            video_timing.confirmed_cfr_rate
            if video_timing is not None
            else None
        )
        if confirmed_rate is None:
            classification = (
                video_timing.classification
                if video_timing is not None
                else "unknown"
            )
            raise FrameTimingError(
                "decoder presentation timing is unusable and "
                f"{classification} probe evidence does not permit nominal fallback: "
                f"{exc}"
            ) from exc
        print(
            "  Decoder presentation timing unavailable; restarting confirmed "
            f"CFR sampling at {confirmed_rate:.6f} fps"
        )
        return decode_once(
            timing_source="nominal_source_index_cfr",
            nominal_rate=confirmed_rate,
            fallback_reason=str(exc),
            announce=False,
        )


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
            resolved_root = root.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise FrameCacheError(f"cannot inspect frame cache directory {root}: {exc}") from exc
        if not resolved_root.is_dir():
            raise FrameCacheError(f"frame cache directory must be a real directory: {root}")
        self.root = resolved_root
        self.max_bytes = int(max_bytes)
        self.path = Path(
            tempfile.mkdtemp(prefix="keyframe-frame-cache-", dir=resolved_root)
        )
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
            header = _ppm_header(*image.size)
            payload = image.tobytes()
            expected_size = self._sizes[index]
            if len(header) + len(payload) != expected_size:
                raise FrameCacheError(
                    f"candidate {index} changed dimensions between decode passes"
                )
            actual_digest = _rgb_digest(image)
            if actual_digest != expected_digest:
                raise FrameCacheError(
                    f"video changed or decoded inconsistently while caching sample {index}; "
                    "refusing to mix passes"
                )
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


@dataclass(frozen=True)
class CandidateCacheResult:
    provider: CandidateImageProvider
    timing_metadata: dict[str, Any]


def cache_candidate_frames(
    video_path: str | Path,
    interval_seconds: float,
    *,
    candidate_indices: Iterable[int],
    frame_indices: Sequence[int],
    timestamps: Sequence[float],
    consumed_targets: Sequence[float],
    next_targets: Sequence[float],
    frame_sizes: Sequence[tuple[int, int]],
    pixel_digests: Sequence[str],
    sampling_timing: Mapping[str, Any],
    cache: CandidateFrameCache,
) -> CandidateCacheResult:
    """Second streaming decode: store just the selected lossless candidate union."""
    wanted = {int(index) for index in candidate_indices}
    sample_count = len(frame_indices)
    aligned_lengths = {
        sample_count,
        len(timestamps),
        len(consumed_targets),
        len(next_targets),
        len(frame_sizes),
        len(pixel_digests),
    }
    if len(aligned_lengths) != 1:
        raise FrameCacheError("first-pass sample identity tables are misaligned")
    timing_source = str(sampling_timing.get("source") or "")
    if timing_source not in {
        "decoder_presentation_time",
        "nominal_source_index_cfr",
    }:
        raise FrameCacheError("first-pass sampling timing source is invalid")
    nominal_rate_value = sampling_timing.get("nominal_frame_rate")
    nominal_rate = (
        float(nominal_rate_value)
        if timing_source == "nominal_source_index_cfr"
        and nominal_rate_value is not None
        else None
    )
    if timing_source == "nominal_source_index_cfr" and (
        nominal_rate is None or nominal_rate <= 0
    ):
        raise FrameCacheError("nominal first-pass timing is missing its CFR rate")

    empty_timing = {
        "source": timing_source,
        "decoder_origin_seconds": None,
        "decoder_error": None,
        "assignment_schedule_verified": None,
        "replayed_sample_count": 0,
        "candidate_comparisons": [],
    }
    if not wanted:
        return CandidateCacheResult(
            provider=CandidateImageProvider({}, {}),
            timing_metadata=empty_timing,
        )
    if min(wanted) < 0 or max(wanted) >= sample_count:
        raise FrameCacheError("candidate index is outside the first-pass sample table")

    recorded_source_indices: list[int] = []
    previous_source_idx: int | None = None
    for source_value in frame_indices:
        source_idx = int(source_value)
        if source_idx < 0:
            raise FrameCacheError("first-pass source frame index is negative")
        if (
            previous_source_idx is not None
            and source_idx <= previous_source_idx
        ):
            raise FrameCacheError(
                "first-pass source frame indices must be strictly increasing"
            )
        recorded_source_indices.append(source_idx)
        previous_source_idx = source_idx

    source_to_sample: dict[int, int] = {}
    for sample_idx in wanted:
        source_idx = recorded_source_indices[sample_idx]
        source_to_sample[source_idx] = sample_idx

    last_required_source_idx = max(source_to_sample)
    expected_schedule = [
        (
            source_idx,
            float(consumed_targets[sample_idx]),
            float(next_targets[sample_idx]),
        )
        for sample_idx, source_idx in enumerate(recorded_source_indices)
        if source_idx <= last_required_source_idx
    ]

    sizes = {index: frame_sizes[index] for index in wanted}
    expected = {index: pixel_digests[index] for index in wanted}
    cache.reserve(sizes)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not reopen video for candidate cache: {video_path}")
    try:
        if not float(interval_seconds) > 0:
            raise FrameCacheError("sampling interval must be positive")
        frame_idx = 0
        remaining = set(wanted)
        decoder_timing = DecoderTimestampNormalizer()
        decoder_disabled = False
        decoder_error: str | None = None
        replay_sampler = TargetTimeSampler(interval_seconds)
        replayed_schedule: list[tuple[int, float, float]] = []
        comparisons: list[dict[str, Any]] = []
        while remaining:
            ok, frame = cap.read()
            if not ok:
                break
            raw_decoder_seconds = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            normalized_decoder_seconds: float | None = None
            if not decoder_disabled:
                try:
                    normalized_decoder_seconds = decoder_timing.observe(
                        raw_decoder_seconds
                    )
                except DecoderTimingRegression as exc:
                    raise FrameCacheError(
                        f"second-pass decoder timing is non-monotonic: {exc}"
                    ) from exc
                except DecoderTimingUnavailable as exc:
                    decoder_disabled = True
                    decoder_error = str(exc)
                if normalized_decoder_seconds is not None:
                    replay_decision = replay_sampler.consider(
                        normalized_decoder_seconds
                    )
                    if replay_decision is not None:
                        replayed_schedule.append(
                            (
                                frame_idx,
                                float(replay_decision.consumed_target),
                                float(replay_decision.next_target),
                            )
                        )

            sample_idx = source_to_sample.get(frame_idx)
            if sample_idx is not None and sample_idx in remaining:
                height, width = frame.shape[:2]
                _frame_bytes(int(width), int(height))
                actual_size = (int(width), int(height))
                expected_size = tuple(frame_sizes[sample_idx])
                if actual_size != expected_size:
                    raise FrameCacheError(
                        f"candidate {sample_idx} changed dimensions between "
                        f"decode passes: expected {expected_size[0]}x"
                        f"{expected_size[1]}, got {actual_size[0]}x"
                        f"{actual_size[1]}"
                    )
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cache.write(sample_idx, image, expected[sample_idx])

                first_timestamp = float(timestamps[sample_idx])
                effective_timestamp = (
                    frame_idx / float(nominal_rate)
                    if timing_source == "nominal_source_index_cfr"
                    else normalized_decoder_seconds
                )
                decoder_delta = (
                    None
                    if normalized_decoder_seconds is None
                    else normalized_decoder_seconds - first_timestamp
                )
                assignment_changed = (
                    None
                    if normalized_decoder_seconds is None
                    else (
                        normalized_decoder_seconds
                        < float(consumed_targets[sample_idx])
                        or normalized_decoder_seconds
                        >= float(next_targets[sample_idx])
                    )
                )
                comparisons.append(
                    {
                        "sample_idx": sample_idx,
                        "source_index": frame_idx,
                        "first_pass_timestamp_seconds": first_timestamp,
                        "second_pass_decoder_timestamp_seconds": (
                            normalized_decoder_seconds
                        ),
                        "second_pass_effective_timestamp_seconds": (
                            effective_timestamp
                        ),
                        "decoder_delta_seconds": decoder_delta,
                        "consumed_target_seconds": float(
                            consumed_targets[sample_idx]
                        ),
                        "next_target_seconds": float(next_targets[sample_idx]),
                        "assignment_changed": assignment_changed,
                        "comparison_enforced": False,
                    }
                )
                remaining.remove(sample_idx)
            frame_idx += 1
        if remaining:
            missing = [
                {
                    "sample_idx": sample_idx,
                    "source_index": int(frame_indices[sample_idx]),
                }
                for sample_idx in sorted(remaining)
            ]
            raise FrameCacheError(
                "video ended before exact candidate source frames could be cached: "
                f"{missing}"
            )
        decoder_usable = not decoder_disabled
        if decoder_usable:
            try:
                decoder_timing.finalize()
            except DecoderTimingUnavailable as exc:
                decoder_error = str(exc)
                decoder_usable = False

        enforce_assignment_schedule = (
            timing_source == "decoder_presentation_time"
            and decoder_usable
        )
        for comparison in comparisons:
            comparison["comparison_enforced"] = enforce_assignment_schedule

        if enforce_assignment_schedule:
            mismatch_position = None
            for position in range(
                max(len(expected_schedule), len(replayed_schedule))
            ):
                expected_row = (
                    expected_schedule[position]
                    if position < len(expected_schedule)
                    else None
                )
                replayed_row = (
                    replayed_schedule[position]
                    if position < len(replayed_schedule)
                    else None
                )
                if expected_row is None or replayed_row is None:
                    mismatch_position = position
                    break
                source_matches = expected_row[0] == replayed_row[0]
                consumed_matches = (
                    abs(expected_row[1] - replayed_row[1])
                    <= PRESENTATION_TIME_EPSILON_SECONDS
                )
                next_matches = (
                    abs(expected_row[2] - replayed_row[2])
                    <= PRESENTATION_TIME_EPSILON_SECONDS
                )
                if not (source_matches and consumed_matches and next_matches):
                    mismatch_position = position
                    break
            if mismatch_position is not None:
                expected_row = (
                    expected_schedule[mismatch_position]
                    if mismatch_position < len(expected_schedule)
                    else None
                )
                replayed_row = (
                    replayed_schedule[mismatch_position]
                    if mismatch_position < len(replayed_schedule)
                    else None
                )
                raise FrameCacheError(
                    "second-pass decoder timing crossed a recorded sampling "
                    "boundary and changed the sampling target assignment at "
                    f"replay position {mismatch_position}: "
                    f"expected {expected_row}, got {replayed_row}"
                )

        comparisons.sort(key=lambda row: int(row["sample_idx"]))
        return CandidateCacheResult(
            provider=cache.provider(expected),
            timing_metadata={
                "source": timing_source,
                "decoder_origin_seconds": decoder_timing.origin_seconds,
                "decoder_error": decoder_error,
                "assignment_schedule_verified": (
                    True if enforce_assignment_schedule else None
                ),
                "replayed_sample_count": (
                    len(replayed_schedule) if decoder_usable else 0
                ),
                "candidate_comparisons": comparisons,
            },
        )
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


def _receive_worker_result(worker: Any, queue: Any, *, timeout_seconds: float) -> tuple[str, Any]:
    """Drain the queue before joining so a large result cannot deadlock exit."""
    deadline = time.monotonic() + float(timeout_seconds)
    while True:
        try:
            return queue.get(timeout=0.25)
        except Empty:
            if not worker.is_alive():
                worker.join()
                raise ClusteringWorkerError(
                    f"average-linkage worker exited unexpectedly (status {worker.exitcode})"
                )
            if time.monotonic() >= deadline:
                raise ClusteringWorkerError(
                    f"average-linkage worker timed out after {timeout_seconds:.1f}s"
                )


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
    admitted_peak = estimated + _CLUSTER_WORKER_OVERHEAD_BYTES
    if admitted_peak > limit:
        raise ClusteringWorkerError(
            f"average-linkage for {rows} samples needs about {admitted_peak / (1024 * 1024):.1f} MiB, "
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
    try:
        timeout_seconds = min(
            _MAX_CLUSTER_WORKER_TIMEOUT_SECONDS,
            max(_MIN_CLUSTER_WORKER_TIMEOUT_SECONDS, 30.0 + rows * 0.02),
        )
        status, payload = _receive_worker_result(
            worker,
            queue,
            timeout_seconds=timeout_seconds,
        )
        worker.join()
        if worker.exitcode != 0:
            raise ClusteringWorkerError(
                f"average-linkage worker exited unexpectedly (status {worker.exitcode})"
            )
        if status != "ok":
            raise ClusteringWorkerError(f"average-linkage worker failed: {payload}")
        return np.asarray(payload, dtype=np.int64)
    finally:
        if worker.is_alive():
            worker.terminate()
        worker.join()
        queue.close()
        queue.join_thread()
