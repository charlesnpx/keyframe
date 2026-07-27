import json
import os
import stat
from pathlib import Path

import pytest

from keyframe import artifacts, transcript


def test_raw_transcript_checkpoint_round_trips_full_precision_and_unicode(tmp_path):
    path = tmp_path / "transcript.raw.json"
    segments = (
        transcript.TranscriptSegment(0.1234567890123, 1.9876543210987, " déjà vu "),
        transcript.TranscriptSegment(2.0000000000001, 3.1415926535897, "東京 🎙️"),
    )

    assert transcript.write_raw_transcript_checkpoint(segments, path) == path

    assert json.loads(path.read_text(encoding="utf-8")) == [
        {
            "start": 0.1234567890123,
            "end": 1.9876543210987,
            "text": "déjà vu",
        },
        {
            "start": 2.0000000000001,
            "end": 3.1415926535897,
            "text": "東京 🎙️",
        },
    ]
    assert "\\u" not in path.read_text(encoding="utf-8")
    assert transcript.read_raw_transcript_checkpoint(path) == segments


def test_diarization_checkpoint_round_trips_precision_and_normalizes_speakers(tmp_path):
    path = tmp_path / "diarization.json"
    rows = (
        transcript.DiarizationRow(0.0000000000001, 1.2345678901234, " SPEAKER_00 "),
        {"start": 1.2345678901234, "end": 9.8765432109876, "speaker": "SPEAKER_01"},
    )

    assert transcript.write_diarization_checkpoint(rows, path) == path

    assert json.loads(path.read_text(encoding="utf-8")) == [
        {
            "start": 0.0000000000001,
            "end": 1.2345678901234,
            "speaker": "SPEAKER_00",
        },
        {
            "start": 1.2345678901234,
            "end": 9.8765432109876,
            "speaker": "SPEAKER_01",
        },
    ]
    assert transcript.read_diarization_checkpoint(path) == (
        transcript.DiarizationRow(0.0000000000001, 1.2345678901234, "SPEAKER_00"),
        transcript.DiarizationRow(1.2345678901234, 9.8765432109876, "SPEAKER_01"),
    )


def test_checkpoint_empty_arrays_round_trip(tmp_path):
    raw = tmp_path / "transcript.raw.json"
    diarization = tmp_path / "diarization.json"

    transcript.write_raw_transcript_checkpoint([], raw)
    transcript.write_diarization_checkpoint([], diarization)

    assert raw.read_text(encoding="utf-8") == "[]"
    assert diarization.read_text(encoding="utf-8") == "[]"
    assert transcript.read_raw_transcript_checkpoint(raw) == ()
    assert transcript.read_diarization_checkpoint(diarization) == ()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("checkpoint", ["raw", "diarization"])
def test_checkpoint_writers_reject_non_finite_timestamps_without_publishing(
    tmp_path,
    value,
    checkpoint,
):
    path = tmp_path / f"{checkpoint}.json"
    path.write_text("previous", encoding="utf-8")
    if checkpoint == "raw":
        writer = transcript.write_raw_transcript_checkpoint
        rows = [{"start": value, "end": 1.0, "text": "bad"}]
    else:
        writer = transcript.write_diarization_checkpoint
        rows = [{"start": 0.0, "end": value, "speaker": "SPEAKER_00"}]

    with pytest.raises(transcript.CheckpointValidationError):
        writer(rows, path)

    assert path.read_text(encoding="utf-8") == "previous"
    assert list(tmp_path.glob(f"{path.name}.tmp-*")) == []


@pytest.mark.parametrize(
    "row",
    [
        None,
        {"start": 0.0, "end": 1.0},
        {"start": 0.0, "end": 1.0, "text": "ok", "speaker": "SPEAKER_00"},
        {"start": True, "end": 1.0, "text": "bad bool"},
        {"start": -1.0, "end": 1.0, "text": "negative"},
        {"start": 1.0, "end": 1.0, "text": "zero duration"},
        {"start": 2.0, "end": 1.0, "text": "backwards"},
        {"start": 0.0, "end": 1.0, "text": "  "},
        {"start": 0.0, "end": 1.0, "text": 42},
    ],
)
def test_raw_checkpoint_rejects_malformed_rows(tmp_path, row):
    with pytest.raises(transcript.CheckpointValidationError):
        transcript.write_raw_transcript_checkpoint([row], tmp_path / "raw.json")


def test_raw_checkpoint_rejects_overlapping_rows(tmp_path):
    with pytest.raises(transcript.CheckpointValidationError, match="overlaps"):
        transcript.write_raw_transcript_checkpoint(
            [
                {"start": 0.0, "end": 2.0, "text": "first"},
                {"start": 1.5, "end": 3.0, "text": "second"},
            ],
            tmp_path / "raw.json",
        )


def test_catastrophic_repetition_detection_is_structural():
    segments = [
        transcript.TranscriptSegment(
            float(index),
            float(index + 1),
            "alpha beta gamma delta epsilon",
        )
        for index in range(30)
    ]

    assert transcript.has_catastrophic_repetition(segments)
    with pytest.raises(transcript.CheckpointValidationError, match="repetition"):
        transcript.validate_transcript_segments(segments)


@pytest.mark.parametrize(
    "row",
    [
        None,
        {"start": 0.0, "end": 1.0},
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "extra"},
        {"start": 0.0, "end": 0.0, "speaker": "SPEAKER_00"},
        {"start": 2.0, "end": 1.0, "speaker": "SPEAKER_00"},
        {"start": 0.0, "end": 1.0, "speaker": " null "},
        {"start": 0.0, "end": 1.0, "speaker": 0},
    ],
)
def test_diarization_checkpoint_rejects_malformed_rows(tmp_path, row):
    with pytest.raises(transcript.CheckpointValidationError):
        transcript.write_diarization_checkpoint([row], tmp_path / "diarization.json")


@pytest.mark.parametrize(
    "payload",
    [
        "not JSON",
        "{}",
        '[{"start": 0, "start": 1, "end": 2, "text": "duplicate"}]',
        '[{"start": NaN, "end": 2, "text": "non-finite"}]',
        '[{"start": 0, "end": 2, "text": "ok", "extra": true}]',
    ],
)
def test_raw_checkpoint_reader_rejects_corrupt_or_malformed_json(tmp_path, payload):
    path = tmp_path / "transcript.raw.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(transcript.CheckpointValidationError):
        transcript.read_raw_transcript_checkpoint(path)


def test_checkpoint_rejects_direct_symlink_and_hardlink_aliases(tmp_path):
    direct = tmp_path / "transcript.raw.json"
    with pytest.raises(artifacts.ArtifactPathCollisionError):
        transcript.write_raw_transcript_checkpoint(
            [],
            direct,
            final_output_paths=[direct],
        )

    target = tmp_path / "transcript.json"
    target.write_text("old", encoding="utf-8")
    symlink = tmp_path / "raw-symlink.json"
    symlink.symlink_to(target)
    with pytest.raises(artifacts.ArtifactPathCollisionError):
        transcript.write_raw_transcript_checkpoint(
            [],
            symlink,
            final_output_paths=[target],
        )

    hardlink = tmp_path / "raw-hardlink.json"
    os.link(target, hardlink)
    with pytest.raises(artifacts.ArtifactPathCollisionError):
        transcript.read_raw_transcript_checkpoint(
            hardlink,
            final_output_paths=[target],
        )

    assert target.read_text(encoding="utf-8") == "old"


def test_atomic_checkpoint_replace_uses_unique_sibling_and_cleans_on_failure(
    monkeypatch,
    tmp_path,
):
    path = tmp_path / "transcript.raw.json"
    path.write_text("previous", encoding="utf-8")
    replace_calls = []

    def fail_replace(source, destination):
        source = Path(source)
        destination = Path(destination)
        replace_calls.append((source, destination, source.read_text(encoding="utf-8")))
        raise OSError("injected rename failure")

    monkeypatch.setattr(artifacts.os, "replace", fail_replace)

    with pytest.raises(OSError, match="rename failure"):
        transcript.write_raw_transcript_checkpoint(
            [transcript.TranscriptSegment(0, 1, "new")],
            path,
        )

    assert len(replace_calls) == 1
    temporary, destination, staged_payload = replace_calls[0]
    assert temporary.parent == path.parent
    assert temporary.name.startswith(f"{path.name}.tmp-")
    assert destination == path
    assert json.loads(staged_payload) == [{"start": 0.0, "end": 1.0, "text": "new"}]
    assert path.read_text(encoding="utf-8") == "previous"
    assert not temporary.exists()


def test_atomic_writer_honors_umask_for_new_outputs(tmp_path):
    path = tmp_path / "new.txt"
    previous_umask = os.umask(0o027)
    try:
        artifacts.atomic_write_text(path, "new")
    finally:
        os.umask(previous_umask)

    assert stat.S_IMODE(path.stat().st_mode) == 0o640


def test_atomic_writer_preserves_existing_output_mode(tmp_path):
    path = tmp_path / "existing.txt"
    path.write_text("old", encoding="utf-8")
    path.chmod(0o604)

    artifacts.atomic_write_text(path, "new")

    assert path.read_text(encoding="utf-8") == "new"
    assert stat.S_IMODE(path.stat().st_mode) == 0o604


def test_huge_integer_writer_failure_uses_checkpoint_validation_contract(tmp_path):
    path = tmp_path / "transcript.raw.json"
    path.write_text("previous", encoding="utf-8")

    with pytest.raises(transcript.CheckpointValidationError, match="finite number"):
        transcript.write_raw_transcript_checkpoint(
            [{"start": 10**400, "end": 10**400, "text": "huge"}],
            path,
        )

    assert path.read_text(encoding="utf-8") == "previous"


def test_huge_integer_reader_failure_uses_checkpoint_validation_contract(tmp_path):
    path = tmp_path / "transcript.raw.json"
    huge = "1" + ("0" * 400)
    path.write_text(
        f'[{{"start": {huge}, "end": {huge}, "text": "huge"}}]',
        encoding="utf-8",
    )

    with pytest.raises(transcript.CheckpointValidationError, match="finite number"):
        transcript.read_raw_transcript_checkpoint(path)


@pytest.mark.parametrize(
    ("writer", "suffix"),
    [
        (transcript.write_txt, "txt"),
        (transcript.write_srt, "srt"),
        (transcript.write_vtt, "vtt"),
        (transcript.write_json, "json"),
    ],
)
def test_final_writers_preserve_previous_output_and_cleanup_on_replace_failure(
    monkeypatch,
    tmp_path,
    writer,
    suffix,
):
    path = tmp_path / f"transcript.{suffix}"
    path.write_text("previous", encoding="utf-8")
    temporary_paths = []

    def fail_replace(source, _destination):
        temporary_paths.append(Path(source))
        raise OSError("injected final replace failure")

    monkeypatch.setattr(artifacts.os, "replace", fail_replace)

    with pytest.raises(OSError, match="final replace failure"):
        writer([transcript.TranscriptSegment(0, 1, "new")], path)

    assert path.read_text(encoding="utf-8") == "previous"
    assert len(temporary_paths) == 1
    assert not temporary_paths[0].exists()


def test_final_writer_formats_remain_byte_compatible(tmp_path):
    segments = (
        transcript.TranscriptSegment(0.0004, 1.2346, "héllo", "SPEAKER_00"),
        transcript.TranscriptSegment(61.2, 62.0, "plain"),
    )
    txt = tmp_path / "transcript.txt"
    srt = tmp_path / "transcript.srt"
    vtt = tmp_path / "transcript.vtt"
    js = tmp_path / "transcript.json"

    transcript.write_txt(segments, txt)
    transcript.write_srt(segments, srt)
    transcript.write_vtt(segments, vtt)
    transcript.write_json(segments, js)

    assert txt.read_bytes() == (
        "[00:00:00.000 --> 00:00:01.235]  SPEAKER_00  héllo\n"
        "[00:01:01.200 --> 00:01:02.000]  plain\n"
    ).encode()
    assert srt.read_bytes() == (
        "1\n00:00:00,000 --> 00:00:01,235\nSPEAKER_00: héllo\n\n"
        "2\n00:01:01,200 --> 00:01:02,000\nplain\n\n"
    ).encode()
    assert vtt.read_bytes() == (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:01.235\nSPEAKER_00: héllo\n\n"
        "00:01:01.200 --> 00:01:02.000\nplain\n\n"
    ).encode()
    assert js.read_bytes() == (
        '[\n'
        '  {\n'
        '    "start": 0.0,\n'
        '    "end": 1.235,\n'
        '    "text": "héllo",\n'
        '    "speaker": "SPEAKER_00"\n'
        '  },\n'
        '  {\n'
        '    "start": 61.2,\n'
        '    "end": 62.0,\n'
        '    "text": "plain"\n'
        '  }\n'
        ']'
    ).encode()


def test_public_and_run_scoped_checkpoint_paths_are_non_hidden_siblings(tmp_path):
    public = artifacts.transcript_checkpoint_paths(tmp_path)
    staged = artifacts.run_staging_paths(tmp_path, "run_20260720-001")

    assert public.transcript_raw == tmp_path / "transcript.raw.json"
    assert public.diarization == tmp_path / "diarization.json"
    assert staged.root == tmp_path / "keyframe-run-run_20260720-001"
    assert staged.transcript_raw == staged.root / "transcript.raw.json"
    assert staged.diarization == staged.root / "diarization.json"
    assert all(not part.startswith(".") for part in staged.root.relative_to(tmp_path).parts)
    assert not staged.root.exists()


@pytest.mark.parametrize("run_id", ["", ".hidden", "../escape", "with/slash", "white space"])
def test_run_staging_paths_reject_unsafe_run_ids(tmp_path, run_id):
    with pytest.raises(ValueError):
        artifacts.run_staging_paths(tmp_path, run_id)
