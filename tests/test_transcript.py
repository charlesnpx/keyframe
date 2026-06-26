import json
import sys
from dataclasses import FrozenInstanceError
from types import ModuleType, SimpleNamespace

import pytest

from keyframe import transcript


def test_transcript_segment_is_immutable():
    segment = transcript.TranscriptSegment(0, 1, "hello", "SPEAKER_00")

    with pytest.raises(FrozenInstanceError):
        segment.text = "changed"


def test_writers_include_speaker_labels(tmp_path):
    segments = (
        transcript.TranscriptSegment(0, 1.25, "hello there", "SPEAKER_00"),
        transcript.TranscriptSegment(1.25, 2.5, "plain segment"),
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
    )
    assert "SPEAKER_00: hello there" in srt.read_text(encoding="utf-8")
    assert "SPEAKER_00: hello there" in vtt.read_text(encoding="utf-8")
    assert json.loads(js.read_text(encoding="utf-8")) == [
        {"start": 0.0, "end": 1.25, "text": "hello there", "speaker": "SPEAKER_00"},
        {"start": 1.25, "end": 2.5, "text": "plain segment"},
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


def test_extract_transcript_uses_whisperx_when_hf_token_exists(tmp_path, monkeypatch):
    video = tmp_path / "recording.mp4"
    video.write_bytes(b"not real media")
    out = tmp_path / "transcript.json"
    calls = []

    def fake_whisperx(video_path, model_name, hf_token):
        calls.append((video_path, model_name, hf_token))
        return (transcript.TranscriptSegment(0, 1, "hello", "SPEAKER_00"),), "en"

    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setattr(transcript, "_extract_with_whisperx", fake_whisperx)
    monkeypatch.setattr(
        transcript,
        "_extract_with_whisper",
        lambda *_args, **_kwargs: pytest.fail("Whisper fallback should not run"),
    )

    segments, language = transcript.extract_transcript(video, model_name="tiny", output=out, fmt="json")

    assert calls == [(video, "tiny", "hf_test")]
    assert language == "en"
    assert segments == (transcript.TranscriptSegment(0, 1, "hello", "SPEAKER_00"),)
    assert json.loads(out.read_text(encoding="utf-8"))[0]["speaker"] == "SPEAKER_00"


def test_no_speaker_detection_forces_whisper_fallback_when_token_exists(tmp_path, monkeypatch):
    video = tmp_path / "recording.mp4"
    video.write_bytes(b"not real media")
    out = tmp_path / "transcript.json"
    calls = []

    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setattr(
        transcript,
        "_extract_with_whisperx",
        lambda *_args, **_kwargs: pytest.fail("WhisperX should not run"),
    )

    def fake_whisper(video_path, model_name):
        calls.append((video_path, model_name))
        return (transcript.TranscriptSegment(0, 1, "hello"),), "en"

    monkeypatch.setattr(transcript, "_extract_with_whisper", fake_whisper)

    segments, _language = transcript.extract_transcript(
        video,
        model_name="tiny",
        output=out,
        fmt="json",
        speaker_detection=False,
    )

    assert calls == [(video, "tiny")]
    assert segments == (transcript.TranscriptSegment(0, 1, "hello"),)
    assert "speaker" not in json.loads(out.read_text(encoding="utf-8"))[0]


def test_missing_hf_token_warns_and_uses_whisper(tmp_path, monkeypatch, capsys):
    video = tmp_path / "recording.mp4"
    video.write_bytes(b"not real media")

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(
        transcript,
        "_extract_with_whisperx",
        lambda *_args, **_kwargs: pytest.fail("WhisperX should not run without HF_TOKEN"),
    )
    monkeypatch.setattr(
        transcript,
        "_extract_with_whisper",
        lambda *_args, **_kwargs: ((transcript.TranscriptSegment(0, 1, "hello"),), "en"),
    )

    transcript.extract_transcript(video, output=tmp_path / "out.json", fmt="json")

    err = capsys.readouterr().err
    assert "Warning: no HF_TOKEN found" in err
    assert "https://huggingface.co/pyannote/speaker-diarization-community-1" in err
    assert "https://huggingface.co/settings/tokens" in err


def test_speaker_detection_failure_warns_and_falls_back(tmp_path, monkeypatch, capsys):
    video = tmp_path / "recording.mp4"
    video.write_bytes(b"not real media")

    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setattr(
        transcript,
        "_extract_with_whisperx",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("gated model")),
    )
    monkeypatch.setattr(
        transcript,
        "_extract_with_whisper",
        lambda *_args, **_kwargs: ((transcript.TranscriptSegment(0, 1, "fallback"),), "en"),
    )

    segments, _language = transcript.extract_transcript(video, output=tmp_path / "out.json", fmt="json")

    err = capsys.readouterr().err
    assert "speaker detection failed" in err
    assert "gated model" in err
    assert segments == (transcript.TranscriptSegment(0, 1, "fallback"),)


def test_whisperx_converter_splits_mixed_speaker_runs():
    result = {
        "segments": [
            {
                "start": 0.0,
                "end": 3.0,
                "text": "hello there yes",
                "words": [
                    {"start": 0.0, "end": 0.5, "word": "hello", "speaker": "SPEAKER_00"},
                    {"start": 0.5, "end": 1.0, "word": "there", "speaker": "SPEAKER_00"},
                    {"start": 1.5, "end": 2.0, "word": "yes", "speaker": "SPEAKER_01"},
                ],
            }
        ]
    }

    assert transcript.whisperx_segments_to_transcript_segments(result) == (
        transcript.TranscriptSegment(0.0, 1.0, "hello there", "SPEAKER_00"),
        transcript.TranscriptSegment(1.5, 2.0, "yes", "SPEAKER_01"),
    )


def test_untimed_diarized_words_still_get_numeric_bounds():
    result = {
        "segments": [
            {
                "start": 10.0,
                "end": 12.5,
                "text": "hello yes",
                "words": [
                    {"word": "hello", "speaker": "SPEAKER_00"},
                    {"word": "yes", "speaker": "SPEAKER_01"},
                ],
            }
        ]
    }

    assert transcript.whisperx_segments_to_transcript_segments(result) == (
        transcript.TranscriptSegment(10.0, 12.5, "hello", "SPEAKER_00"),
        transcript.TranscriptSegment(10.0, 12.5, "yes", "SPEAKER_01"),
    )


def test_partially_untimed_words_use_segment_bounds():
    result = {
        "segments": [
            {
                "start": 10.0,
                "end": 14.0,
                "text": "hello there yes",
                "words": [
                    {"word": "hello", "speaker": "SPEAKER_00"},
                    {"start": 11.0, "end": 12.0, "word": "there", "speaker": "SPEAKER_00"},
                    {"start": 12.5, "word": "yes", "speaker": "SPEAKER_01"},
                ],
            }
        ]
    }

    assert transcript.whisperx_segments_to_transcript_segments(result) == (
        transcript.TranscriptSegment(10.0, 12.0, "hello there", "SPEAKER_00"),
        transcript.TranscriptSegment(12.5, 14.0, "yes", "SPEAKER_01"),
    )


def test_none_words_falls_back_to_segment_level_text_and_speaker():
    result = {
        "segments": [
            {
                "start": 10.0,
                "end": 14.0,
                "text": "segment text",
                "speaker": "SPEAKER_00",
                "words": None,
            }
        ]
    }

    assert transcript.whisperx_segments_to_transcript_segments(result) == (
        transcript.TranscriptSegment(10.0, 14.0, "segment text", "SPEAKER_00"),
    )


def test_extract_with_whisperx_runs_diarization_pipeline(monkeypatch, tmp_path):
    video = tmp_path / "recording.mp4"
    video.write_bytes(b"not real media")
    calls = []

    class FakeModel:
        def transcribe(self, audio, batch_size):
            calls.append(("transcribe", audio, batch_size))
            return {"language": "en", "segments": [{"start": 0, "end": 1, "text": "hello"}]}

    class FakeDiarizationPipeline:
        def __init__(self, model_name, token, device):
            calls.append(("diarization_init", model_name, token, device))

        def __call__(self, audio):
            calls.append(("diarize", audio))
            return "diarized"

    fake_whisperx = ModuleType("whisperx")
    fake_whisperx.__path__ = []
    fake_whisperx.load_model = lambda model_name, device, compute_type: calls.append(
            ("load_model", model_name, device, compute_type)
        ) or FakeModel()
    fake_whisperx.load_audio = lambda path: calls.append(("load_audio", path)) or "audio"
    fake_whisperx.load_align_model = lambda language_code, device: calls.append(
            ("load_align_model", language_code, device)
        ) or ("align_model", "metadata")
    fake_whisperx.align = lambda segments, align_model, metadata, audio, device, return_char_alignments: calls.append(
            ("align", segments, align_model, metadata, audio, device, return_char_alignments)
        ) or {"segments": [{"start": 0, "end": 1, "text": "hello"}]}
    fake_whisperx.assign_word_speakers = lambda diarized, result: calls.append(
            ("assign_word_speakers", diarized, result)
        ) or {
            "segments": [
                {
                    "start": 0,
                    "end": 1,
                    "text": "hello",
                    "words": [{"start": 0, "end": 1, "word": "hello", "speaker": "SPEAKER_00"}],
                }
            ]
        }

    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
    monkeypatch.setitem(
        sys.modules,
        "whisperx.diarize",
        SimpleNamespace(DiarizationPipeline=FakeDiarizationPipeline),
    )
    monkeypatch.setattr(transcript, "_select_whisperx_device", lambda: ("cpu", "int8"))

    segments, language = transcript._extract_with_whisperx(video, "tiny", "hf_test")

    assert language == "en"
    assert segments == (transcript.TranscriptSegment(0, 1, "hello", "SPEAKER_00"),)
    assert ("load_model", "tiny", "cpu", "int8") in calls
    assert (
        "diarization_init",
        "pyannote/speaker-diarization-community-1",
        "hf_test",
        "cpu",
    ) in calls
    assert ("assign_word_speakers", "diarized", {"segments": [{"start": 0, "end": 1, "text": "hello"}]}) in calls
