import json
import sys
from dataclasses import FrozenInstanceError
from types import ModuleType, SimpleNamespace

import pytest

from keyframe import transcript


def _video(tmp_path):
    video = tmp_path / "recording.mp4"
    video.write_bytes(b"not real media")
    return video


def _result(segments=(), language="en"):
    return transcript.TranscriptionResult(tuple(segments), language, {})


def test_transcript_segment_is_immutable():
    segment = transcript.TranscriptSegment(0, 1, "hello", "SPEAKER_00")

    with pytest.raises(FrozenInstanceError):
        segment.text = "changed"


def test_writers_include_speaker_labels_and_omit_absent_speaker(tmp_path):
    segments = (
        transcript.TranscriptSegment(0, 1.25, "hello there", "SPEAKER_00"),
        transcript.TranscriptSegment(1.25, 2.5, "plain segment"),
        transcript.TranscriptSegment(2.5, 3.5, "null-ish segment", " null "),
    )

    txt = tmp_path / "out.txt"
    srt = tmp_path / "out.srt"
    vtt = tmp_path / "out.vtt"
    js = tmp_path / "out.json"

    transcript.write_txt(segments, txt)
    transcript.write_srt(segments, srt)
    transcript.write_vtt(segments, vtt)
    transcript.write_json(segments, js)

    assert txt.read_text(encoding="utf-8") == (
        "[00:00:00.000 --> 00:00:01.250]  SPEAKER_00  hello there\n"
        "[00:00:01.250 --> 00:00:02.500]  plain segment\n"
        "[00:00:02.500 --> 00:00:03.500]  null-ish segment\n"
    )
    assert "SPEAKER_00: hello there" in srt.read_text(encoding="utf-8")
    assert "SPEAKER_00: hello there" in vtt.read_text(encoding="utf-8")
    assert json.loads(js.read_text(encoding="utf-8")) == [
        {"start": 0.0, "end": 1.25, "text": "hello there", "speaker": "SPEAKER_00"},
        {"start": 1.25, "end": 2.5, "text": "plain segment"},
        {"start": 2.5, "end": 3.5, "text": "null-ish segment"},
    ]


def test_select_whisperx_device_uses_cuda_float16(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
    )

    assert transcript._select_whisperx_device() == ("cuda", "float16")


def test_select_whisperx_device_uses_cpu_int8(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )

    assert transcript._select_whisperx_device() == ("cpu", "int8")


@pytest.mark.parametrize("token", ["hf_test", None])
def test_no_speaker_detection_runs_whisper_only_and_emits_no_speaker_warnings(
    tmp_path,
    monkeypatch,
    capsys,
    token,
):
    video = _video(tmp_path)
    out = tmp_path / "transcript.json"
    calls = []

    if token is None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
    else:
        monkeypatch.setenv("HF_TOKEN", token)
    monkeypatch.setattr(
        transcript,
        "_detect_speakers",
        lambda *_args, **_kwargs: pytest.fail("speaker detection should not run"),
    )

    def fake_whisper(video_path, model_name):
        calls.append((video_path, model_name))
        return _result((transcript.TranscriptSegment(0, 1, "hello"),))

    monkeypatch.setattr(transcript, "_extract_with_whisper", fake_whisper)

    segments, language = transcript.extract_transcript(
        video,
        model_name="tiny",
        output=out,
        fmt="json",
        speaker_detection=False,
    )

    assert calls == [(video, "tiny")]
    assert language == "en"
    assert segments == (transcript.TranscriptSegment(0, 1, "hello"),)
    assert "speaker" not in json.loads(out.read_text(encoding="utf-8"))[0]
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("token", [None, "   "])
def test_missing_or_blank_hf_token_warns_after_whisper(tmp_path, monkeypatch, capsys, token):
    video = _video(tmp_path)
    if token is None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
    else:
        monkeypatch.setenv("HF_TOKEN", token)
    monkeypatch.setattr(
        transcript,
        "_detect_speakers",
        lambda *_args, **_kwargs: pytest.fail("speaker detection should not run without HF_TOKEN"),
    )
    monkeypatch.setattr(
        transcript,
        "_extract_with_whisper",
        lambda *_args, **_kwargs: _result(
            (transcript.TranscriptSegment(0, 1, "hello"),)
        ),
    )

    transcript.extract_transcript(video, output=tmp_path / "out.json", fmt="json")

    err = capsys.readouterr().err
    assert "Warning: no HF_TOKEN found" in err
    assert "https://huggingface.co/pyannote/speaker-diarization-community-1" in err
    assert "https://huggingface.co/settings/tokens" in err


def test_empty_whisper_output_skips_diarization_and_speaker_warnings(tmp_path, monkeypatch, capsys):
    video = _video(tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(
        transcript,
        "_detect_speakers",
        lambda *_args, **_kwargs: pytest.fail("speaker detection should not run for empty transcript"),
    )
    monkeypatch.setattr(
        transcript,
        "_extract_with_whisper",
        lambda *_args, **_kwargs: _result(),
    )

    segments, language = transcript.extract_transcript(video, output=tmp_path / "out.json", fmt="json")

    assert segments == ()
    assert language == "en"
    assert json.loads((tmp_path / "out.json").read_text(encoding="utf-8")) == []
    assert capsys.readouterr().err == ""


def test_valid_hf_token_runs_whisper_then_diarization_with_stripped_token(tmp_path, monkeypatch):
    video = _video(tmp_path)
    out = tmp_path / "transcript.json"
    calls = []
    whisper_segments = (
        transcript.TranscriptSegment(1.23456, 2.34567, "Whisper text"),
        transcript.TranscriptSegment(2.34567, 3.0, "unlabeled"),
    )

    def fake_whisper(video_path, model_name):
        calls.append(("whisper", video_path, model_name))
        return _result(whisper_segments)

    def fake_detect(video_path, hf_token):
        calls.append(("detect", video_path, hf_token))
        return (transcript.DiarizationRow(1.0, 2.5, "SPEAKER_00"),)

    monkeypatch.setenv("HF_TOKEN", "  hf_test  ")
    monkeypatch.setattr(transcript, "_extract_with_whisper", fake_whisper)
    monkeypatch.setattr(transcript, "_detect_speakers", fake_detect)

    segments, language = transcript.extract_transcript(video, model_name="tiny", output=out, fmt="json")

    assert calls == [("whisper", video, "tiny"), ("detect", video, "hf_test")]
    assert language == "en"
    assert segments == (
        transcript.TranscriptSegment(1.23456, 2.34567, "Whisper text", "SPEAKER_00"),
        transcript.TranscriptSegment(2.34567, 3.0, "unlabeled", "SPEAKER_00"),
    )
    assert json.loads(out.read_text(encoding="utf-8")) == [
        {"start": 1.235, "end": 2.346, "text": "Whisper text", "speaker": "SPEAKER_00"},
        {"start": 2.346, "end": 3.0, "text": "unlabeled", "speaker": "SPEAKER_00"},
    ]


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("gated model"),
        ModuleNotFoundError("No module named 'whisperx'"),
    ],
)
def test_speaker_detection_failure_warns_and_keeps_whisper_output(
    tmp_path,
    monkeypatch,
    capsys,
    failure,
):
    video = _video(tmp_path)
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setattr(
        transcript,
        "_extract_with_whisper",
        lambda *_args, **_kwargs: _result(
            (transcript.TranscriptSegment(0, 1, "fallback"),)
        ),
    )
    monkeypatch.setattr(
        transcript,
        "_detect_speakers",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
    )

    segments, _language = transcript.extract_transcript(video, output=tmp_path / "out.json", fmt="json")

    err = capsys.readouterr().err
    assert "speaker detection failed" in err
    assert str(failure) in err
    assert segments == (transcript.TranscriptSegment(0, 1, "fallback"),)


def test_empty_diarization_warns_and_keeps_whisper_output(tmp_path, monkeypatch, capsys):
    video = _video(tmp_path)
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setattr(
        transcript,
        "_extract_with_whisper",
        lambda *_args, **_kwargs: _result(
            (transcript.TranscriptSegment(0, 1, "fallback"),)
        ),
    )
    monkeypatch.setattr(transcript, "_detect_speakers", lambda *_args, **_kwargs: ())

    segments, _language = transcript.extract_transcript(video, output=tmp_path / "out.json", fmt="json")

    err = capsys.readouterr().err
    assert "speaker detection failed" in err
    assert "no usable speaker overlaps" in err
    assert segments == (transcript.TranscriptSegment(0, 1, "fallback"),)


def test_detect_speakers_uses_only_whisperx_audio_and_pyannote(monkeypatch, tmp_path):
    video = _video(tmp_path)
    calls = []

    def forbidden(name):
        def fail(*_args, **_kwargs):
            pytest.fail(f"{name} should not be called")
        return fail

    class FakeDiarizationPipeline:
        def __init__(self, model_name, token, device):
            calls.append(("diarization_init", model_name, token, device))

        def __call__(self, audio, progress_callback=None):
            calls.append(("diarize", audio))
            if progress_callback is not None:
                progress_callback(25)
                progress_callback(100)
            return [{"start": 0, "end": 1, "speaker": " SPEAKER_00 "}]

    fake_whisperx = ModuleType("whisperx")
    fake_whisperx.__path__ = []
    fake_whisperx.load_audio = lambda path: calls.append(("load_audio", path)) or "audio"
    fake_whisperx.load_model = forbidden("whisperx.load_model")
    fake_whisperx.load_align_model = forbidden("whisperx.load_align_model")
    fake_whisperx.align = forbidden("whisperx.align")
    fake_whisperx.assign_word_speakers = forbidden("whisperx.assign_word_speakers")

    fake_faster_whisper = ModuleType("faster_whisper")
    fake_faster_whisper.WhisperModel = forbidden("faster_whisper.WhisperModel")

    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
    monkeypatch.setitem(
        sys.modules,
        "whisperx.diarize",
        SimpleNamespace(DiarizationPipeline=FakeDiarizationPipeline),
    )
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_faster_whisper)
    monkeypatch.setattr(transcript, "_select_whisperx_device", lambda: ("cpu", "int8"))

    rows = transcript._detect_speakers(video, "hf_test")

    assert rows == (transcript.DiarizationRow(0, 1, "SPEAKER_00"),)
    assert calls == [
        ("load_audio", str(video)),
        ("diarization_init", "pyannote/speaker-diarization-community-1", "hf_test", "cpu"),
        ("diarize", "audio"),
    ]


def test_detect_speakers_reports_monotonic_progress_and_closes_bar(monkeypatch, tmp_path):
    video = _video(tmp_path)
    updates = []
    closed = []

    class FakeProgress:
        def __init__(self, **kwargs):
            assert kwargs["total"] == 100
            assert kwargs["desc"] == "Detecting speakers"

        def update(self, amount):
            updates.append(amount)

        def close(self):
            closed.append(True)

    class FakeDiarizationPipeline:
        def __init__(self, *_args, **_kwargs):
            pass

        def __call__(self, _audio, progress_callback=None):
            progress_callback(20)
            progress_callback(10)  # WhisperX callbacks may be repeated.
            progress_callback(80)
            progress_callback(100)
            return [{"start": 0, "end": 1, "speaker": "SPEAKER_00"}]

    fake_whisperx = ModuleType("whisperx")
    fake_whisperx.__path__ = []
    fake_whisperx.load_audio = lambda _path: "audio"
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
    monkeypatch.setitem(
        sys.modules,
        "whisperx.diarize",
        SimpleNamespace(DiarizationPipeline=FakeDiarizationPipeline),
    )
    monkeypatch.setitem(sys.modules, "tqdm", SimpleNamespace(tqdm=FakeProgress))
    monkeypatch.setattr(transcript, "_select_whisperx_device", lambda: ("cpu", "int8"))

    rows = transcript._detect_speakers(video, "hf_test")

    assert rows == (transcript.DiarizationRow(0, 1, "SPEAKER_00"),)
    assert updates == [20.0, 60.0, 20.0]
    assert closed == [True]


def test_assign_speakers_uses_largest_summed_overlap():
    segments = (transcript.TranscriptSegment(0, 10, "hello"),)
    diarization = [
        {"start": 0, "end": 3, "speaker": "SPEAKER_00"},
        {"start": 4, "end": 5, "speaker": "SPEAKER_00"},
        {"start": 5, "end": 8, "speaker": "SPEAKER_01"},
    ]

    assert transcript._assign_speakers(segments, diarization) == (
        transcript.TranscriptSegment(0, 10, "hello", "SPEAKER_00"),
    )


def test_assign_speakers_reads_all_tuple_diarization_rows():
    segments = (transcript.TranscriptSegment(0, 10, "hello"),)
    diarization = (
        {"start": 0, "end": 1, "speaker": "SPEAKER_00"},
        {"start": 1, "end": 6, "speaker": "SPEAKER_01"},
    )

    assert transcript._assign_speakers(segments, diarization) == (
        transcript.TranscriptSegment(0, 10, "hello", "SPEAKER_01"),
    )


def test_assign_speakers_ties_by_earliest_clipped_overlap_start_then_label():
    segment = transcript.TranscriptSegment(0, 10, "hello")

    earliest = transcript._assign_speakers(
        (segment,),
        [
            {"start": 2, "end": 4, "speaker": "SPEAKER_B"},
            {"start": 1, "end": 3, "speaker": "SPEAKER_A"},
        ],
    )
    lexical = transcript._assign_speakers(
        (segment,),
        [
            {"start": 1, "end": 3, "speaker": "SPEAKER_B"},
            {"start": 1, "end": 3, "speaker": "SPEAKER_A"},
        ],
    )

    assert earliest == (transcript.TranscriptSegment(0, 10, "hello", "SPEAKER_A"),)
    assert lexical == (transcript.TranscriptSegment(0, 10, "hello", "SPEAKER_A"),)


def test_assign_speakers_ignores_invalid_rows_and_strips_labels():
    segments = (transcript.TranscriptSegment(10, 20, "hello"),)
    diarization = [
        {"start": 0, "end": 5, "speaker": "SPEAKER_NO_OVERLAP"},
        {"start": 20, "end": 25, "speaker": "SPEAKER_ABUTTING"},
        {"start": 11, "end": 12, "speaker": " "},
        {"start": 11, "end": 12, "speaker": None},
        {"start": 11, "end": 12, "speaker": "none"},
        {"start": 11, "end": 12, "speaker": "NULL"},
        {"start": 11, "end": 12, "speaker": "nan"},
        {"start": 11, "end": 12, "speaker": "<NA>"},
        {"start": 11, "end": 12, "speaker": "NaT"},
        {"start": "bad", "end": 12, "speaker": "SPEAKER_BAD_START"},
        {"start": 11, "end": "bad", "speaker": "SPEAKER_BAD_END"},
        {"start": float("nan"), "end": 12, "speaker": "SPEAKER_NAN"},
        {"start": 11, "end": float("inf"), "speaker": "SPEAKER_INF"},
        {"start": 12, "end": 12, "speaker": "SPEAKER_ZERO"},
        {"start": 13, "end": 12, "speaker": "SPEAKER_NEGATIVE"},
        {"start": 11, "end": 12},
        {"end": 12, "speaker": "SPEAKER_MISSING_START"},
        {"start": 11, "speaker": "SPEAKER_MISSING_END"},
        {"start": 11, "end": 12, "speaker": " SPEAKER_02 "},
    ]

    assert transcript._assign_speakers(segments, diarization) == (
        transcript.TranscriptSegment(10, 20, "hello", "SPEAKER_02"),
    )


def test_assign_speakers_returns_new_segments_and_omits_no_overlap_speaker_in_json(tmp_path):
    original = transcript.TranscriptSegment(0, 1, "hello")
    segments = (original, transcript.TranscriptSegment(10, 11, "bye"))

    assigned = transcript._assign_speakers(
        segments,
        [{"start": 0, "end": 1, "speaker": "SPEAKER_00"}],
    )

    assert assigned[0] is not original
    assert original.speaker is None
    assert assigned == (
        transcript.TranscriptSegment(0, 1, "hello", "SPEAKER_00"),
        transcript.TranscriptSegment(10, 11, "bye"),
    )

    out = tmp_path / "out.json"
    transcript.write_json(assigned, out)
    assert json.loads(out.read_text(encoding="utf-8")) == [
        {"start": 0.0, "end": 1.0, "text": "hello", "speaker": "SPEAKER_00"},
        {"start": 10.0, "end": 11.0, "text": "bye"},
    ]


def test_companion_json_uses_same_speaker_semantics(tmp_path, monkeypatch):
    from keyframe import cli
    import keyframe.transcript as transcript_module

    video = _video(tmp_path)
    out_dir = tmp_path / "out"

    segments = (
        transcript_module.TranscriptSegment(0, 1, "hello", "SPEAKER_00"),
        transcript_module.TranscriptSegment(1, 2, "plain"),
    )

    def fake_run_transcript(_video, output, _preflight):
        transcript_module.write_txt(segments, output / "transcript.txt")
        transcript_module.write_json(segments, output / "transcript.json")
        return SimpleNamespace(
            segments=segments,
            language="en",
        )

    monkeypatch.setattr(cli, "_preflight_transcript", lambda _args: object())
    monkeypatch.setattr(cli, "_run_transcript", fake_run_transcript)

    cli.cmd_extract(
        SimpleNamespace(
            video=str(video),
            output=str(out_dir),
            transcript_only=True,
            frames_only=False,
            sample_interval=0.75,
            pass1_clusters=9,
            similarity_threshold=0.85,
            max_output_frames=None,
            verbose_trace=False,
            debug_qa_targets=None,
            whisper_model="medium",
            transcript_format="txt",
            no_speaker_detection=False,
        )
    )

    assert json.loads((out_dir / "transcript.json").read_text(encoding="utf-8")) == [
        {"start": 0.0, "end": 1.0, "text": "hello", "speaker": "SPEAKER_00"},
        {"start": 1.0, "end": 2.0, "text": "plain"},
    ]


def test_transcript_window_remains_text_only_with_speaker_labels():
    from keyframe.manifest import transcript_window

    assert transcript_window(
        [{"start": 0.0, "end": 2.0, "text": "hello", "speaker": "SPEAKER_00"}],
        1.0,
    ) == "hello"
