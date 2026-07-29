from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest


RECORDING_SHA256 = "d42ca66e99cfa571334c6a04c42d603819e3f1b034939615aafd3df0ac472225"
RECORDING_DURATION_SECONDS = 3018.6245
TARGET_TOLERANCE_SECONDS = 2.25

DENSE_STATES = (
    ("source_frames", (2453.0, 2454.0), ("source", "frames", "1260")),
    ("samples", (2455.0, 2455.5), ("samples", "84", "interval")),
    ("inspected_candidates", (2460.0, 2465.5), ("inspected", "candidates", "18")),
    ("saved_frames", (2484.0, 2495.0), ("saved", "frames", "10")),
    ("transcript_parts", (2498.0, 2501.0), ("transcript", "parts", "7")),
)

LATER_STATES = (
    ("rescue", 2652.0, 2656.0),
    ("inspect", 2656.0, 2664.0),
    ("save", 2664.0, 2672.0),
    ("limitations", 2672.0, 2710.0),
    ("diarization", 2710.0, 2720.0),
    ("transcript_output", 2720.0, 2728.0),
)

FIRST_PRESENTATION_STATES = (
    ("component_test_method_configuration", 735.0, 790.0),
    ("inspection_execution_or_repair", 945.0, 1035.0),
    ("engineering_review", 1045.0, 1095.0),
    ("cnsc_reporting", 1165.0, 1198.0),
    ("personnel_certifications", 1220.0, 1245.0),
    ("component_library", 1245.0, 1270.0),
)


def _enabled() -> bool:
    return os.environ.get("KEYFRAME_BIWEEKLY_DURABLE_REGRESSION") == "1"


def _recording_path() -> Path:
    configured = os.environ.get("KEYFRAME_BIWEEKLY_DURABLE_INPUT")
    if not configured:
        pytest.skip("set KEYFRAME_BIWEEKLY_DURABLE_INPUT to the private recording")
    return Path(configured).expanduser()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _words(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.casefold()))


def _frame_words(frame: dict) -> set[str]:
    return _words(str(frame.get("ocr_text", "")))


def _frames_in_interval(frames: list[dict], start: float, end: float) -> list[dict]:
    return [frame for frame in frames if start <= float(frame["timestamp"]) < end]


def _dense_state_match(frames: list[dict], targets: tuple[float, ...], anchors: tuple[str, ...]) -> dict | None:
    eligible = [
        frame
        for frame in frames
        if any(
            abs(float(frame["timestamp"]) - target) <= TARGET_TOLERANCE_SECONDS
            for target in targets
        )
    ]
    eligible.sort(
        key=lambda frame: min(
            abs(float(frame["timestamp"]) - target)
            for target in targets
        )
    )
    for frame in eligible:
        words = _frame_words(frame)
        if sum(anchor in words for anchor in anchors) >= 2:
            return frame
    return None


def _dense_state_matches(
    frames: list[dict],
    targets: tuple[float, ...],
    anchors: tuple[str, ...],
) -> list[dict]:
    return [
        frame
        for frame in frames
        if any(
            abs(float(frame["timestamp"]) - target) <= TARGET_TOLERANCE_SECONDS
            for target in targets
        )
        and sum(anchor in _frame_words(frame) for anchor in anchors) >= 2
    ]


def _structural_redundancy(frames: list[dict]) -> float:
    if len(frames) < 2:
        return 0.0
    redundant = 0
    for left, right in zip(frames, frames[1:]):
        left_tokens = set(left.get("cleaned_ocr_tokens", ()))
        right_tokens = set(right.get("cleaned_ocr_tokens", ()))
        overlap = (
            len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)
            if left_tokens or right_tokens
            else 0.0
        )
        redundant += int(
            abs(float(left["timestamp"]) - float(right["timestamp"])) <= 2.0
            and overlap >= 0.9
        )
    return redundant / (len(frames) - 1)


@pytest.mark.slow
@pytest.mark.skipif(
    not _enabled(),
    reason="set KEYFRAME_BIWEEKLY_DURABLE_REGRESSION=1 for the private durable-state gate",
)
def test_biweekly_recording_retains_durable_information_states(tmp_path):
    recording = _recording_path()
    assert recording.is_file()
    assert _sha256(recording) == RECORDING_SHA256

    output = tmp_path / "biweekly-durable-state"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "keyframe.cli",
            str(recording),
            "--frames-only",
            "--output",
            str(output),
        ],
        check=True,
    )
    manifest = json.loads((output / "frames" / "manifest.json").read_text(encoding="utf-8"))
    frames = manifest["frames"]
    metadata = manifest["metadata"]

    dense_matches = {}
    for label, targets, anchors in DENSE_STATES:
        dense_matches[label] = _dense_state_match(frames, targets, anchors)
        assert dense_matches[label] is not None, label
    saved_spec = next(state for state in DENSE_STATES if state[0] == "saved_frames")
    assert len(_dense_state_matches(frames, saved_spec[1], saved_spec[2])) == 1
    assert dense_matches["saved_frames"]["timestamp"] != dense_matches["transcript_parts"]["timestamp"]

    for label, start, end in LATER_STATES:
        assert _frames_in_interval(frames, start, end), label
    for label, start, end in FIRST_PRESENTATION_STATES:
        assert _frames_in_interval(frames, start, end), label

    timestamps = [float(frame["timestamp"]) for frame in frames]
    assert timestamps == sorted(timestamps)
    assert any(timestamp >= RECORDING_DURATION_SECONDS / 2.0 for timestamp in timestamps)

    coverage = metadata["coverage"]
    expected_window_ids = set(range(int(coverage["coverage_window_count"])))
    represented_window_ids = {
        int(window_id)
        for frame in frames
        for window_id in frame.get("coverage_window_ids", ())
    }
    assert represented_window_ids == expected_window_ids
    assert int(coverage["unique_base_primary_count"]) <= 96
    assert int(coverage["durable_state_fill_count"]) <= int(coverage["remaining_primary_capacity"])

    rescue = metadata["rescue"]
    rescue_count = sum(bool(frame.get("rescue_origin")) for frame in frames)
    assert rescue_count <= int(rescue["rescue_budget"])
    assert _structural_redundancy(frames) <= 0.10
