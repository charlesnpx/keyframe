from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from keyframe import cli, frame_preflight, media_preflight
from keyframe.frame_preflight import FrameRuntimePlatform
from keyframe.media_preflight import (
    MediaProbeResult,
    MediaStream,
    classify_video_timing,
    parse_ffprobe_payload,
)
from keyframe.pipeline.config import KeyframeExtractionConfig
from keyframe.pipeline.context import make_context
from keyframe.pipeline.contracts import (
    FrameStore,
    SampleTable,
    SamplingOutput,
    TemporalOutput,
)
from keyframe.pipeline.orchestrator import OutputStage
from keyframe.pipeline.sampling import (
    DecoderTimestampNormalizer,
    DecoderTimingRegression,
    DecoderTimingUnavailable,
    FrameTimingError,
    TargetTimeSampler,
)
from keyframe.pipeline.streaming import (
    CandidateFrameCache,
    FrameCacheError,
    cache_candidate_frames,
    stream_video_features,
)
from keyframe.pipeline.trace import NoOpTraceSink


def _timing_payload(
    *,
    avg_frame_rate: str | None = "10/1",
    r_frame_rate: str | None = "10/1",
    time_base: str | None = "1/1000",
    start_time: str | None = "0.000000",
    duration: str | None = "0.500000",
    duration_ts: str | None = "500",
    nb_frames: str | None = "5",
):
    return {
        "avg_frame_rate": avg_frame_rate,
        "r_frame_rate": r_frame_rate,
        "time_base": time_base,
        "start_time": start_time,
        "duration": duration,
        "duration_ts": duration_ts,
        "nb_frames": nb_frames,
    }


@pytest.mark.parametrize(
    ("payload", "classification", "reason_fragment"),
    [
        (
            _timing_payload(
                avg_frame_rate="30000/1001",
                r_frame_rate="60000/2002",
                duration="10.010000",
                duration_ts="900900",
                time_base="1/90000",
                nb_frames="300",
            ),
            "cfr",
            "consistent",
        ),
        (
            _timing_payload(
                avg_frame_rate="120/19",
                r_frame_rate="30/1",
            ),
            "vfr",
            "differ",
        ),
        (
            _timing_payload(avg_frame_rate=None),
            "unknown",
            "missing",
        ),
        (
            _timing_payload(duration="2.0"),
            "unknown",
            "inconsistent",
        ),
    ],
)
def test_video_timing_classification(payload, classification, reason_fragment):
    timing = classify_video_timing(payload)

    assert timing.classification == classification
    assert reason_fragment in timing.reason
    assert json.loads(json.dumps(timing.to_dict())) == timing.to_dict()


def test_cfr_duration_consistency_accepts_the_exact_tolerance_boundary():
    timing = classify_video_timing(
        _timing_payload(
            duration="10.1",
            duration_ts=None,
            nb_frames="100",
        )
    )

    assert timing.classification == "cfr"
    assert timing.expected_duration_seconds == pytest.approx(10.0)
    assert timing.duration_delta_seconds == pytest.approx(0.1)
    assert timing.duration_tolerance_seconds == pytest.approx(0.1)


def test_duration_ts_is_used_when_stream_duration_is_missing():
    timing = classify_video_timing(
        _timing_payload(duration=None)
    )

    assert timing.classification == "cfr"
    assert timing.duration_source == "duration_ts"
    assert timing.duration_seconds == pytest.approx(0.5)


def test_selected_usable_video_stream_carries_all_probe_timing_fields():
    result = parse_ffprobe_payload(
        {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 0,
                    "height": 720,
                    **_timing_payload(),
                },
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1280,
                    "height": 720,
                    **_timing_payload(
                        avg_frame_rate="120/19",
                        r_frame_rate="30/1",
                        time_base="1/15360",
                        start_time="2.500000",
                        duration="1.266667",
                        duration_ts="19456",
                        nb_frames="8",
                    ),
                },
            ]
        }
    )

    timing = result.selected_video_timing
    assert timing is not None
    assert timing.classification == "vfr"
    assert timing.avg_frame_rate == "120/19"
    assert timing.r_frame_rate == "30/1"
    assert timing.time_base == "1/15360"
    assert timing.stream_start_seconds == pytest.approx(2.5)
    assert timing.duration_seconds == pytest.approx(1.266667)
    assert timing.duration_ts == 19456
    assert timing.nb_frames == 8


def test_cli_preflight_threads_selected_timing_through_one_frame_config(
    tmp_path,
    monkeypatch,
):
    video = tmp_path / "recording.mp4"
    video.write_bytes(b"media")
    timing = classify_video_timing(_timing_payload())
    probe = MediaProbeResult(
        (
            MediaStream(
                codec_type="video",
                codec_name="h264",
                width=1280,
                height=720,
                video_timing=timing,
            ),
        )
    )
    monkeypatch.setattr(media_preflight, "probe_media", lambda _path: probe)
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
    args = cli._parse_extract_args(
        [
            str(video),
            "--frames-only",
            "--output",
            str(tmp_path / "output"),
        ]
    )

    preflight = cli._preflight_extract(args)

    assert preflight.frame_config is not None
    assert preflight.frame_config.video_timing is timing


def test_target_sampler_selects_each_frame_once_and_skips_missed_targets():
    sampler = TargetTimeSampler(0.5)
    decisions = [
        sampler.consider(timestamp)
        for timestamp in (0.0, 0.1, 0.5, 1.2, 1.21)
    ]
    selected = [decision for decision in decisions if decision is not None]

    assert [decision.timestamp for decision in selected] == [0.0, 0.5, 1.2]
    assert [decision.consumed_target for decision in selected] == [0.0, 0.5, 1.0]
    assert [decision.next_target for decision in selected] == [0.5, 1.0, 1.5]


def test_decoder_timing_normalizes_origin_and_clamps_epsilon_regression():
    timing = DecoderTimestampNormalizer()

    assert timing.observe(7.25) == 0.0
    assert timing.observe(7.35) == pytest.approx(0.1)
    assert timing.observe(7.3499995) == pytest.approx(0.1)
    timing.finalize()
    assert timing.origin_seconds == pytest.approx(7.25)


def test_decoder_timing_allows_provisional_initial_zeros_until_progress():
    timing = DecoderTimestampNormalizer()

    assert [timing.observe(value) for value in (0.0, 0.0, 0.2)] == [
        0.0,
        0.0,
        0.2,
    ]
    timing.finalize()


def test_decoder_timing_uses_the_first_finite_timestamp_as_origin():
    timing = DecoderTimestampNormalizer()

    assert timing.observe(float("nan")) is None
    assert timing.observe(3.5) == 0.0
    assert timing.observe(4.0) == pytest.approx(0.5)
    timing.finalize()
    assert timing.origin_seconds == pytest.approx(3.5)


def test_decoder_timing_rejects_material_regression_and_all_zero_sources():
    regressing = DecoderTimestampNormalizer()
    regressing.observe(0.0)
    regressing.observe(0.2)
    with pytest.raises(DecoderTimingRegression, match="moved backward"):
        regressing.observe(0.1)

    all_zero = DecoderTimestampNormalizer()
    all_zero.observe(0.0)
    all_zero.observe(0.0)
    with pytest.raises(DecoderTimingUnavailable, match="all-zero"):
        all_zero.finalize()


class _FakeCapture:
    def __init__(self, frames, timestamps_ms, *, fps=10.0):
        self.frames = [np.asarray(frame, dtype=np.uint8) for frame in frames]
        self.timestamps_ms = list(timestamps_ms)
        self.fps = float(fps)
        self.position = -1
        self.timestamp_reads = []
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        self.position += 1
        if self.position >= len(self.frames):
            return False, None
        return True, self.frames[self.position].copy()

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return len(self.frames)
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return self.frames[0].shape[1]
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.frames[0].shape[0]
        if prop == cv2.CAP_PROP_POS_MSEC:
            self.timestamp_reads.append(self.position)
            return self.timestamps_ms[self.position]
        return 0.0

    def release(self):
        self.released = True


class _CaptureFactory:
    def __init__(self, frames, timestamp_sequences, *, fps=10.0):
        self.frames = frames
        self.timestamp_sequences = list(timestamp_sequences)
        self.fps = fps
        self.captures = []

    def __call__(self, _path):
        sequence_index = min(
            len(self.captures),
            len(self.timestamp_sequences) - 1,
        )
        capture = _FakeCapture(
            self.frames,
            self.timestamp_sequences[sequence_index],
            fps=self.fps,
        )
        self.captures.append(capture)
        return capture


def _frames(count=5, *, shape=(16, 24, 3), offset=0):
    return [
        np.full(shape, offset + index * 20, dtype=np.uint8)
        for index in range(count)
    ]


def _embed(images):
    return np.asarray(
        [[float(index), 1.0] for index, _image in enumerate(images)],
        dtype=np.float32,
    )


def test_streaming_samples_post_read_presentation_time(monkeypatch):
    factory = _CaptureFactory(
        _frames(),
        [[1000.0, 1000.0, 1200.0, 1700.0, 2200.0]],
    )
    monkeypatch.setattr(
        "keyframe.pipeline.streaming.cv2.VideoCapture",
        factory,
    )

    streamed = stream_video_features(
        "recording.mp4",
        0.5,
        embed_images=_embed,
        video_timing=classify_video_timing(
            _timing_payload(
                avg_frame_rate="120/19",
                r_frame_rate="30/1",
            )
        ),
    )

    assert streamed.frame_indices == [0, 3, 4]
    assert streamed.timestamps == pytest.approx([0.0, 0.7, 1.2])
    assert streamed.consumed_targets == pytest.approx([0.0, 0.5, 1.0])
    assert streamed.next_targets == pytest.approx([0.5, 1.0, 1.5])
    assert factory.captures[0].timestamp_reads == [0, 1, 2, 3, 4]
    assert streamed.sampling_timing["source"] == "decoder_presentation_time"
    assert streamed.sampling_timing["decoder_origin_seconds"] == pytest.approx(1.0)


def test_confirmed_cfr_restarts_with_nominal_timing_and_discards_partial_results(
    monkeypatch,
):
    factory = _CaptureFactory(
        _frames(),
        [
            [0.0] * 5,
            [0.0] * 5,
        ],
    )
    monkeypatch.setattr(
        "keyframe.pipeline.streaming.cv2.VideoCapture",
        factory,
    )
    monkeypatch.setattr(
        "keyframe.pipeline.streaming._MAX_CLIP_BATCH_BYTES",
        1,
    )
    embed_calls = 0

    def distinguish_decode_passes(images):
        nonlocal embed_calls
        embed_calls += 1
        return np.full((len(images), 2), embed_calls, dtype=np.float32)

    streamed = stream_video_features(
        "recording.mp4",
        0.2,
        embed_images=distinguish_decode_passes,
        video_timing=classify_video_timing(_timing_payload()),
    )

    assert len(factory.captures) == 2
    assert streamed.frame_indices == [0, 2, 4]
    assert streamed.timestamps == pytest.approx([0.0, 0.2, 0.4])
    assert streamed.clip_embeddings[:, 0].tolist() == [2.0, 3.0, 4.0]
    assert streamed.sampling_timing["source"] == "nominal_source_index_cfr"
    assert "all-zero" in streamed.sampling_timing["fallback_reason"]


@pytest.mark.parametrize(
    "timing",
    [
        classify_video_timing(
            _timing_payload(
                avg_frame_rate="120/19",
                r_frame_rate="30/1",
            )
        ),
        classify_video_timing(_timing_payload(avg_frame_rate=None)),
    ],
)
def test_vfr_and_unknown_inputs_fail_closed_when_decoder_timing_is_unavailable(
    monkeypatch,
    timing,
):
    factory = _CaptureFactory(_frames(), [[0.0] * 5])
    monkeypatch.setattr(
        "keyframe.pipeline.streaming.cv2.VideoCapture",
        factory,
    )

    with pytest.raises(FrameTimingError, match="does not permit nominal fallback"):
        stream_video_features(
            "recording.mp4",
            0.2,
            embed_images=_embed,
            video_timing=timing,
        )

    assert len(factory.captures) == 1


def _cache_streamed(
    tmp_path,
    monkeypatch,
    *,
    first_timestamps,
    second_timestamps,
    second_frames=None,
    candidate_indices=(1, 2),
):
    frames = _frames(count=len(first_timestamps))
    first_factory = _CaptureFactory(frames, [first_timestamps])
    monkeypatch.setattr(
        "keyframe.pipeline.streaming.cv2.VideoCapture",
        first_factory,
    )
    streamed = stream_video_features(
        "recording.mp4",
        0.5,
        embed_images=_embed,
        video_timing=classify_video_timing(
            _timing_payload(
                avg_frame_rate="120/19",
                r_frame_rate="30/1",
                nb_frames=str(len(frames)),
            )
        ),
    )

    second_factory = _CaptureFactory(
        second_frames or frames,
        [second_timestamps],
    )
    monkeypatch.setattr(
        "keyframe.pipeline.streaming.cv2.VideoCapture",
        second_factory,
    )
    cache = CandidateFrameCache(cache_root=tmp_path, max_bytes=1024 * 1024)
    return streamed, cache, lambda: cache_candidate_frames(
        "recording.mp4",
        0.5,
        candidate_indices=candidate_indices,
        frame_indices=streamed.frame_indices,
        timestamps=streamed.timestamps,
        consumed_targets=streamed.consumed_targets,
        next_targets=streamed.next_targets,
        frame_sizes=streamed.frame_sizes,
        pixel_digests=streamed.pixel_digests,
        sampling_timing=streamed.sampling_timing,
        cache=cache,
    )


def test_candidate_cache_decodes_exact_recorded_source_indices(
    tmp_path,
    monkeypatch,
):
    streamed, cache, run_cache = _cache_streamed(
        tmp_path,
        monkeypatch,
        first_timestamps=[0.0, 100.0, 200.0, 700.0, 1200.0],
        second_timestamps=[0.0, 100.0, 250.0, 750.0, 1250.0],
    )

    result = run_cache()
    try:
        assert streamed.frame_indices == [0, 3, 4]
        assert [
            row["source_index"]
            for row in result.timing_metadata["candidate_comparisons"]
        ] == [3, 4]
        assert result.provider[1].size == (24, 16)
        assert result.provider[2].size == (24, 16)
        assert all(
            not row["assignment_changed"]
            for row in result.timing_metadata["candidate_comparisons"]
        )
    finally:
        cache.cleanup()


def test_candidate_cache_rejects_digest_and_dimension_changes(
    tmp_path,
    monkeypatch,
):
    changed = _frames()
    changed[3] = np.full((16, 24, 3), 255, dtype=np.uint8)
    _streamed, _cache, run_cache = _cache_streamed(
        tmp_path,
        monkeypatch,
        first_timestamps=[0.0, 100.0, 200.0, 700.0, 1200.0],
        second_timestamps=[0.0, 100.0, 200.0, 700.0, 1200.0],
        second_frames=changed,
    )
    with pytest.raises(FrameCacheError, match="decoded inconsistently"):
        run_cache()

    resized = _frames()
    # Same pixel count and digest, but the exact dimensions differ.
    resized[3] = np.full((24, 16, 3), 60, dtype=np.uint8)
    _streamed, _cache, run_cache = _cache_streamed(
        tmp_path,
        monkeypatch,
        first_timestamps=[0.0, 100.0, 200.0, 700.0, 1200.0],
        second_timestamps=[0.0, 100.0, 200.0, 700.0, 1200.0],
        second_frames=resized,
    )
    with pytest.raises(FrameCacheError, match="changed dimensions"):
        run_cache()


def test_candidate_cache_rejects_boundary_crossing_and_nonmonotonic_replay(
    tmp_path,
    monkeypatch,
):
    _streamed, _cache, run_cache = _cache_streamed(
        tmp_path,
        monkeypatch,
        first_timestamps=[0.0, 100.0, 200.0, 700.0, 1200.0],
        second_timestamps=[0.0, 100.0, 200.0, 1050.0, 1200.0],
    )
    with pytest.raises(FrameCacheError, match="crossed"):
        run_cache()

    _streamed, _cache, run_cache = _cache_streamed(
        tmp_path,
        monkeypatch,
        first_timestamps=[0.0, 100.0, 200.0, 700.0, 1200.0],
        second_timestamps=[0.0, 100.0, 800.0, 700.0, 1200.0],
    )
    with pytest.raises(FrameCacheError, match="non-monotonic"):
        run_cache()


def test_candidate_cache_rejects_target_reassigned_to_a_non_candidate_frame(
    tmp_path,
    monkeypatch,
):
    _streamed, _cache, run_cache = _cache_streamed(
        tmp_path,
        monkeypatch,
        first_timestamps=[0.0, 100.0, 200.0, 700.0, 1200.0],
        second_timestamps=[0.0, 100.0, 550.0, 700.0, 1200.0],
    )

    with pytest.raises(FrameCacheError, match="changed the sampling target assignment"):
        run_cache()


def test_candidate_cache_defers_assignment_checks_for_all_zero_replay_timing(
    tmp_path,
    monkeypatch,
):
    _streamed, cache, run_cache = _cache_streamed(
        tmp_path,
        monkeypatch,
        first_timestamps=[0.0, 100.0, 200.0, 700.0, 1200.0],
        second_timestamps=[0.0, 0.0, 0.0, 0.0, 0.0],
    )

    result = run_cache()
    try:
        assert "all-zero" in result.timing_metadata["decoder_error"]
        assert result.timing_metadata["assignment_schedule_verified"] is None
        assert all(
            not row["comparison_enforced"]
            for row in result.timing_metadata["candidate_comparisons"]
        )
    finally:
        cache.cleanup()


def test_candidate_cache_rejects_a_missing_recorded_source_index(
    tmp_path,
    monkeypatch,
):
    frames = _frames()
    first_factory = _CaptureFactory(
        frames,
        [[0.0, 100.0, 200.0, 700.0, 1200.0]],
    )
    monkeypatch.setattr(
        "keyframe.pipeline.streaming.cv2.VideoCapture",
        first_factory,
    )
    streamed = stream_video_features(
        "recording.mp4",
        0.5,
        embed_images=_embed,
        video_timing=classify_video_timing(
            _timing_payload(
                avg_frame_rate="120/19",
                r_frame_rate="30/1",
            )
        ),
    )
    second_factory = _CaptureFactory(
        frames[:4],
        [[0.0, 100.0, 200.0, 700.0]],
    )
    monkeypatch.setattr(
        "keyframe.pipeline.streaming.cv2.VideoCapture",
        second_factory,
    )
    cache = CandidateFrameCache(cache_root=tmp_path, max_bytes=1024 * 1024)

    with pytest.raises(FrameCacheError, match="exact candidate source frames"):
        cache_candidate_frames(
            "recording.mp4",
            0.5,
            candidate_indices={2},
            frame_indices=streamed.frame_indices,
            timestamps=streamed.timestamps,
            consumed_targets=streamed.consumed_targets,
            next_targets=streamed.next_targets,
            frame_sizes=streamed.frame_sizes,
            pixel_digests=streamed.pixel_digests,
            sampling_timing=streamed.sampling_timing,
            cache=cache,
        )


def test_output_manifest_preserves_complete_json_safe_sampling_provenance(
    tmp_path,
    monkeypatch,
):
    timing = classify_video_timing(_timing_payload())
    metadata = {
        **timing.to_dict(),
        "source": "nominal_source_index_cfr",
        "interval_seconds": 0.5,
        "epsilon_seconds": 1e-6,
        "decoder_origin_seconds": 0.0,
        "fallback_reason": "decoder exposed all-zero presentation time",
        "decoder_diagnostic_error": None,
        "decoded_frame_count": 5,
        "sample_count": 3,
        "second_pass": {
            "source": "nominal_source_index_cfr",
            "decoder_origin_seconds": 0.0,
            "decoder_error": None,
            "candidate_comparisons": [],
        },
    }
    sampling = SamplingOutput(
        frame_store=FrameStore(frames=[]),
        samples=SampleTable(
            timestamps=[],
            frame_indices=[],
            timing_metadata=metadata,
        ),
    )
    temporal = TemporalOutput(
        scenes=[],
        scene_coalescence={},
        cluster_allocs=[],
        sample_clusters={},
        sample_scenes={},
        sample_temporal_windows={},
    )
    captured = {}

    monkeypatch.setattr(
        "keyframe.frames.save_results",
        lambda *_args, **_kwargs: tmp_path / "captions.json",
    )

    def write_manifest(_rows, _output, *, metadata):
        captured["metadata"] = metadata
        return tmp_path / "manifest.json"

    monkeypatch.setattr("keyframe.manifest.write_manifest", write_manifest)
    context = make_context(
        KeyframeExtractionConfig(video_timing=timing),
        NoOpTraceSink(),
    )

    artifacts = OutputStage().run(
        (),
        sampling,
        temporal,
        tmp_path,
        0,
        0,
        0,
        0,
        0,
        0,
        context,
    )

    assert artifacts.manifest_metadata["sampling_timing"] == metadata
    assert captured["metadata"]["sampling_timing"] == metadata
    assert json.loads(json.dumps(metadata, sort_keys=True)) == metadata
    assert set(metadata) >= {
        "classification",
        "reason",
        "avg_frame_rate",
        "r_frame_rate",
        "time_base",
        "stream_start_seconds",
        "duration_seconds",
        "duration_ts",
        "nb_frames",
        "source",
        "interval_seconds",
        "epsilon_seconds",
        "decoder_origin_seconds",
        "fallback_reason",
    }


def _raw_decoder_timestamps(path: Path):
    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened()
    rows = []
    source_index = 0
    try:
        while True:
            ok, _frame = capture.read()
            if not ok:
                break
            rows.append(
                (
                    source_index,
                    float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0,
                )
            )
            source_index += 1
    finally:
        capture.release()
    return rows


def test_vfr_scene_boundaries_map_source_indices_not_nominal_timecodes(
    tmp_path,
):
    from keyframe.frames import detect_scenes

    fixture = tmp_path / "vfr-scene-cut.mp4"
    writer = cv2.VideoWriter(
        str(fixture),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (64, 48),
    )
    assert writer.isOpened()
    try:
        for source_index in range(80):
            value = 0 if source_index < 40 else 255
            writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
    finally:
        writer.release()

    # These source-frame samples model VFR presentation times where the hard
    # cut at source frame 40 is presented at 8 seconds, while SceneDetect's
    # nominal 10-fps timecode reports 4 seconds.
    frame_indices = list(range(0, 80, 10))
    timestamps = [float(index) * 0.2 for index in frame_indices]

    assert detect_scenes(
        fixture,
        timestamps,
        frame_indices=frame_indices,
    ) == [(0, 3), (4, 7)]


def test_real_vfr_fixture_uses_production_opencv_timing_and_exact_replay(
    tmp_path,
):
    fixture = Path(__file__).parent / "fixtures" / "vfr-sampling.mp4"
    raw_rows = _raw_decoder_timestamps(fixture)
    raw_times = [timestamp for _index, timestamp in raw_rows]
    assert len(raw_rows) >= 6
    assert len({round(delta, 6) for delta in np.diff(raw_times)}) >= 3

    timing = classify_video_timing(
        {
            "avg_frame_rate": "120/19",
            "r_frame_rate": "30/1",
            "time_base": "1/15360",
            "start_time": "0.000000",
            "duration": "1.266667",
            "duration_ts": "19456",
            "nb_frames": "8",
        }
    )
    assert timing.classification == "vfr"
    streamed = stream_video_features(
        fixture,
        0.4,
        embed_images=_embed,
        video_timing=timing,
    )

    normalized = [
        timestamp - raw_times[0]
        for timestamp in raw_times
    ]
    sampler = TargetTimeSampler(0.4)
    expected_source_indices = [
        source_index
        for (source_index, _raw), timestamp in zip(raw_rows, normalized)
        if sampler.consider(timestamp) is not None
    ]
    assert streamed.frame_indices == expected_source_indices

    capture = cv2.VideoCapture(str(fixture))
    legacy_fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    legacy_step = max(1, int(0.4 * legacy_fps))
    legacy_indices = list(range(0, len(raw_rows), legacy_step))
    assert streamed.frame_indices != legacy_indices

    cache = CandidateFrameCache(cache_root=tmp_path, max_bytes=1024 * 1024)
    result = cache_candidate_frames(
        fixture,
        0.4,
        candidate_indices=range(len(streamed.frame_indices)),
        frame_indices=streamed.frame_indices,
        timestamps=streamed.timestamps,
        consumed_targets=streamed.consumed_targets,
        next_targets=streamed.next_targets,
        frame_sizes=streamed.frame_sizes,
        pixel_digests=streamed.pixel_digests,
        sampling_timing=streamed.sampling_timing,
        cache=cache,
    )
    try:
        assert [
            row["source_index"]
            for row in result.timing_metadata["candidate_comparisons"]
        ] == streamed.frame_indices
        for sample_idx in range(len(streamed.frame_indices)):
            image = result.provider[sample_idx]
            image.close()
    finally:
        cache.cleanup()
