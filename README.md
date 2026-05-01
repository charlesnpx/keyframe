# keyframe

Extract key frames and timestamped transcripts from video files using CLIP, Florence-2, and Whisper. All models run locally.

## Install

```bash
# via pipx (recommended)
pipx install git+ssh://git@github.com/charlesnpx/keyframe.git

# or HTTPS
pipx install git+https://github.com/charlesnpx/keyframe.git

# or from local checkout
pipx install --force .

# install Claude Code / Codex skills
keyframe install-skills
```

### Prerequisites

- Python 3.11+
- ffmpeg (required by Whisper for audio extraction)
  ```bash
  brew install ffmpeg
  ```

### SSL issues

If you hit SSL cert errors when models download for the first time:

```bash
# Install Python's default certificates (fixes most SSL issues)
/Applications/Python\ 3.14/Install\ Certificates.command

# Or if behind a corporate proxy, point to your CA bundle
export SSL_CERT_FILE=/path/to/corporate-ca-bundle.crt
```

### Model downloads (first run)

These download automatically and are cached:
- **CLIP ViT-B-32** (~350MB) — image/text embeddings
- **Florence-2-base** (~450MB) — frame captioning
- **Whisper medium** (~1.4GB) — speech transcription

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
| `-o, --output` | `/tmp/<video>_extracted/` | Output directory |
| `--frames-only` | | Skip transcript extraction |
| `--transcript-only` | | Skip frame extraction |
| `-i, --sample-interval` | `0.5` | Sample one frame every N seconds |
| `-c, --pass1-clusters` | `15` | CLIP over-segmentation clusters |
| `-t, --similarity-threshold` | `0.85` | Deprecated no-op; deterministic merge vetoes are used |
| `-w, --whisper-model` | `medium` | Whisper model: tiny/base/small/medium/large |
| `--transcript-format` | `txt` | Output format: txt/srt/vtt/json |

## How it works

### Key frame extraction (two-pass)

1. **Pass 1 (CLIP + dHash):** Sample frames at 0.5s intervals, compute dHashes, embed with CLIP ViT-B-32, allocate more clusters to visually novel scenes, and pick scored representatives.

2. **Pass 2 (Florence-2 + OCR):** Caption the candidates with Florence-2, extract OCR, collapse near-time duplicates, and merge with deterministic OCR/time/transcript vetoes.

Scrolling a data table (visually different but semantically identical) gets collapsed, while a dropdown opening (visually similar but semantically distinct) gets preserved.

### Transcript extraction

Whisper runs locally and extracts timestamped speech segments from the audio track.

## Output structure

```
output_dir/
  frames/
    frame_000008_0.50s.png
    frame_000296_18.48s.png
    ...
    captions.json           # Florence-2 captions + merge metadata
    manifest.json           # Deterministic frame triage index
  transcript.txt            # Timestamped transcript
  transcript.json           # Machine-readable transcript
```

## Tips

- For UI recordings with many important states: try `--pass1-clusters 20`
- Default transcription uses `--whisper-model medium`
- Florence-2 uses `florence-community/Florence-2-base` (native transformers support). The original `microsoft/Florence-2-base` weights are broken with transformers 4.50+.
- CLIP model is used for image embedding; deterministic OCR/dHash merge logic handles final dedupe.
