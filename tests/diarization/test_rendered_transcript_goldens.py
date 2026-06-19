import json
from pathlib import Path

import pytest

from keyframe.diarization import read_recording_json, render_transcript, rendered_transcript_json_dumps


FIXTURE_DIR = Path(__file__).with_name("fixtures")
INPUT_DIR = Path(__file__).with_name("rendered_transcript_inputs")
GOLDEN_DIR = Path(__file__).with_name("goldens") / "rendered_transcripts"

CASES = (
    ("clean_two_speaker", FIXTURE_DIR / "clean_two_speaker.json", "diarization_cluster", {}),
    ("overlap", FIXTURE_DIR / "overlap.json", "diarization_cluster", {}),
    ("missing_confidence", FIXTURE_DIR / "missing_confidence.json", "diarization_cluster", {}),
    ("missing_speaker", FIXTURE_DIR / "missing_speaker.json", "diarization_cluster", {}),
    (
        "needs_review",
        FIXTURE_DIR / "degraded_output.json",
        "diarization_cluster",
        {"degraded_state": "needs_review"},
    ),
    (
        "diagnostic_only",
        FIXTURE_DIR / "degraded_output.json",
        "diarization_cluster",
        {"degraded_state": "diagnostic_only"},
    ),
    ("unsupported", FIXTURE_DIR / "degraded_output.json", "diarization_cluster", {"degraded_state": "unsupported"}),
    (
        "speaker_attribution_unavailable",
        FIXTURE_DIR / "degraded_output.json",
        "diarization_cluster",
        {"degraded_state": "speaker_attribution_unavailable"},
    ),
    ("authenticated_track_metadata", INPUT_DIR / "authenticated_track_metadata.json", "channel_metadata", {}),
)


def _render_case(input_path: Path, label_source: str, render_kwargs: dict) -> str:
    recording = read_recording_json(input_path)
    return rendered_transcript_json_dumps(render_transcript(recording, label_source=label_source, **render_kwargs))


def test_rendered_transcript_golden_case_set_covers_product_behaviors():
    assert [name for name, _, _, _ in CASES] == [
        "clean_two_speaker",
        "overlap",
        "missing_confidence",
        "missing_speaker",
        "needs_review",
        "diagnostic_only",
        "unsupported",
        "speaker_attribution_unavailable",
        "authenticated_track_metadata",
    ]
    assert sorted(path.name for path in GOLDEN_DIR.glob("*.json")) == sorted(
        f"{name}.json" for name, _, _, _ in CASES
    )


@pytest.mark.parametrize(("name", "input_path", "label_source", "render_kwargs"), CASES)
def test_rendered_transcript_json_matches_byte_stable_goldens(name, input_path, label_source, render_kwargs):
    actual = _render_case(input_path, label_source, render_kwargs)
    expected = (GOLDEN_DIR / f"{name}.json").read_text(encoding="utf-8")

    assert actual == expected
    assert actual == _render_case(input_path, label_source, render_kwargs)
    assert "\r\n" not in actual


@pytest.mark.parametrize(("name", "input_path", "label_source", "render_kwargs"), CASES)
def test_displayed_labels_include_provenance_and_recording_scope(name, input_path, label_source, render_kwargs):
    payload = json.loads(_render_case(input_path, label_source, render_kwargs))

    for section_name in ("turns", "words"):
        for item in payload[section_name]:
            if item["label"] is None:
                assert item["display_label"] is None
                continue

            display_label = item["display_label"]
            assert display_label == {
                "confidence": None,
                "label": item["label"],
                "scope": "recording",
                "source": label_source,
                "source_ref": None,
            }


@pytest.mark.parametrize(("name", "input_path", "label_source", "render_kwargs"), CASES)
def test_rendered_transcript_goldens_do_not_leak_reference_speaker_ids(name, input_path, label_source, render_kwargs):
    rendered_text = _render_case(input_path, label_source, render_kwargs)

    assert '"speaker_ref"' not in rendered_text
    assert "spk-" not in rendered_text
    assert "participant_id" not in rendered_text
    assert "reference_speaker_id" not in rendered_text
    assert "corpus_speaker_id" not in rendered_text
