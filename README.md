# keyframe

Extract key frames and timestamped, optionally speaker-labeled transcripts from video files using CLIP, Florence-2, Whisper, and pyannote. All models run locally.

## Install

```bash
# via pipx with Python 3.12 (recommended)
pipx install --python python3.12 git+ssh://git@github.com/charlesnpx/keyframe.git

# or HTTPS
pipx install --python python3.12 git+https://github.com/charlesnpx/keyframe.git

# or from local checkout
pipx install --python python3.12 --force .

# install Claude Code / Codex skills
keyframe install-skills
keyframe install-skills --install --target all --json --install-root /tmp/keyframe-skill-stage
```

`--install-root` is intended for delegated installers such as `mise-en-place`.
It stages files under the supplied directory as if it were `$HOME` and reports
the staged absolute paths in JSON.

### Prerequisites

- Python 3.11, 3.12, or 3.13. Python 3.12 is recommended; Python 3.14 is not supported.
- ffmpeg (required by Whisper for audio extraction)
  ```bash
  brew install ffmpeg
  ```
- Optional for speaker detection: `HF_TOKEN` with access to the pyannote diarization model.
  Accept the model terms at <https://huggingface.co/pyannote/speaker-diarization-community-1>,
  create a token at <https://huggingface.co/settings/tokens>, then export it:
  ```bash
  export HF_TOKEN=hf_...
  ```

### SSL issues

If you hit SSL cert errors when models download for the first time:

```bash
# Install Python's default certificates (fixes most SSL issues)
/Applications/Python\ 3.12/Install\ Certificates.command

# Or if behind a corporate proxy, point to your CA bundle
export SSL_CERT_FILE=/path/to/corporate-ca-bundle.crt
```

### Model downloads (first run)

These download automatically and are cached:
- **CLIP ViT-B-32** (~350MB) — image/text embeddings
- **Florence-2-base** (~450MB) — frame captioning
- **Whisper medium** (~1.4GB) — speech transcription and segment timing
- **pyannote speaker diarization** — segment-level speaker labels when `HF_TOKEN` is configured

## Usage

### Full extraction (frames + transcript)

```bash
keyframe video.mp4
keyframe video.mp4 -o ./output-dir
```

### Frames only

```bash
keyframe video.mp4 --frames-only
```

### Transcript only

```bash
keyframe video.mp4 --transcript-only
keyframe recording.m4a --transcript-only
```

### Whisper-only transcript

Speaker detection is enabled by default when `HF_TOKEN` is present. To force the previous Whisper-only transcript behavior:

```bash
keyframe recording.m4a --transcript-only --no-speaker-detection
python keyframe/transcript.py recording.m4a --no-speaker-detection
```

### As a Claude Code skill

```
/keyframe ~/Downloads/meeting-recording.mp4
```

### As a Codex skill

```
$keyframe ~/Downloads/meeting-recording.mp4
```

## Commands

| Command | Description |
|---------|-------------|
| `keyframe <file>` | Extract frames + transcript |
| `keyframe extract <file>` | Same as above (explicit subcommand) |
| `keyframe install-skills` | Install Claude Code and Codex skills |

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-o, --output` | `<input-folder>/<video>_extracted/` | Output directory (falls back to `/tmp` if the input folder isn't writable) |
| `--frames-only` | | Skip transcript extraction |
| `--transcript-only` | | Skip frame extraction |
| `-i, --sample-interval` | `0.5` | Sample one frame every N seconds |
| `-c, --pass1-clusters` | `15` | CLIP over-segmentation clusters |
| `-t, --similarity-threshold` | `0.85` | Deprecated no-op; deterministic merge vetoes are used |
| `-w, --whisper-model` | `medium` | Whisper model: tiny/base/small/medium/large |
| `--transcript-format` | `txt` | Output format: txt/srt/vtt/json |
| `--no-speaker-detection` | | Force Whisper-only transcription and skip pyannote speaker detection |

## How it works

### Key frame extraction (two-pass)

1. **Pass 1 (CLIP + dHash):** Sample frames at 0.5s intervals, compute dHashes, embed with CLIP ViT-B-32, allocate more clusters to visually novel scenes, and pick scored representatives.

2. **Pass 2 (Florence-2 + OCR):** Caption the candidates with Florence-2, extract OCR, collapse near-time duplicates, and merge with deterministic OCR/time/transcript vetoes.

Scrolling a data table (visually different but semantically identical) gets collapsed, while a dropdown opening (visually similar but semantically distinct) gets preserved.

### Transcript extraction

Whisper always provides the transcript text and segment boundaries. When `HF_TOKEN` is set, pyannote detects speakers with `pyannote/speaker-diarization-community-1`, and Keyframe assigns a dominant speaker label to each Whisper segment based on diarization overlap. If `HF_TOKEN` is missing or speaker detection fails, Keyframe warns and keeps the unlabeled Whisper transcript.

Speaker labels use raw pyannote labels such as `SPEAKER_00`. TXT output places the label after the timestamp, SRT/VTT prefix each caption, and JSON includes `speaker` only on labeled segments.

## Output structure

```
output_dir/
  frames/
    frame_000008_0.50s.png
    frame_000296_18.48s.png
    ...
    captions.json           # Florence-2 captions + merge metadata
    manifest.json           # Deterministic frame triage index
  transcript.txt            # Timestamped transcript, speaker-labeled when available
  transcript.json           # Machine-readable transcript, includes speaker when available
```

## Tips

- For UI recordings with many important states: try `--pass1-clusters 20`
- Default transcription uses `--whisper-model medium`
- Use `--no-speaker-detection` when you want Whisper-only output or do not want to use `HF_TOKEN`
- Florence-2 uses `florence-community/Florence-2-base` (native transformers support). The original `microsoft/Florence-2-base` weights are broken with transformers 4.50+.
- CLIP model is used for image embedding; deterministic OCR/dHash merge logic handles final dedupe.
