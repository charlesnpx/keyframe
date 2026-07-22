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

These download automatically on first use and are cached:
- **CLIP ViT-B-32** (~350MB) — image/text embeddings
- **Florence-2-base** (~450MB) — frame captioning
- **Whisper medium** (~1.4GB) — MLX-Whisper on supported Apple Silicon Macs, OpenAI Whisper elsewhere
- **pyannote speaker diarization** — segment-level speaker labels when `HF_TOKEN` is configured

MLX dependencies and weights are gated to Apple Silicon running macOS 14 or
newer. Linux, Windows, Intel Macs, and older macOS releases do not install MLX
and do not request MLX model weights. On a supported Mac, Keyframe resolves the
exact pinned MLX revision from the local Hugging Face cache first. It permits a
network download only when that exact snapshot is genuinely absent, so a warm
run does not wait on online model resolution.

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

### Transcript without speaker detection

Speaker detection is enabled by default when `HF_TOKEN` is present. To keep the transcript unlabeled:

```bash
keyframe recording.m4a --transcript-only --no-speaker-detection
```

### Backend and concurrency controls

```bash
# Automatic: MLX on a supported Mac, OpenAI Whisper elsewhere
keyframe recording.m4a --transcript-only --transcription-backend auto

# Force the portable OpenAI Whisper backend
keyframe recording.m4a --transcript-only --transcription-backend whisper

# Force CPU speaker detection even when MPS or CUDA is available
keyframe recording.m4a --transcript-only --diarization-device cpu

# Force serial stages for a memory-constrained machine
keyframe recording.m4a --transcript-only --stage-concurrency serial
```

Explicit `--transcription-backend mlx` fails during preflight on unsupported
machines, before importing MLX or acquiring a model. `auto` can recover from an
eligible MLX import, acquisition, load, or inference failure by exiting that
worker and starting a fresh OpenAI Whisper CPU worker. Explicit
`--diarization-device mps` likewise fails during preflight when MPS is
unavailable. Automatic MPS diarization retries once on CPU only for a typed MPS
model-initialization or inference compute failure; explicit MPS is strict, and
authentication, acquisition, decoding, and checkpoint failures do not trigger
that fallback. MPS workers force PyTorch's implicit CPU fallback off so every
CPU retry is admitted and reported by Keyframe.

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
| `-o, --output` | `<input-file-folder>/<video>_extracted/` | Output directory (falls back to `/tmp` if the input folder isn't writable) |
| `--frames-only` | | Skip transcript extraction |
| `--transcript-only` | | Skip frame extraction |
| `-i, --sample-interval` | `0.5` | Sample one frame every N seconds |
| `-c, --pass1-clusters` | `15` | CLIP over-segmentation clusters (1-64) |
| `--max-clustering-memory-mb` | `2048` | Memory admission limit for each isolated average-linkage worker |
| `--max-frame-cache-mb` | `8192` | Maximum size of the temporary, lossless candidate-frame cache |
| `--frame-cache-dir` | OS temp directory | Override the directory used for the temporary candidate-frame cache |
| `-t, --similarity-threshold` | `0.85` | Deprecated no-op; deterministic merge vetoes are used |
| `-w, --whisper-model` | `medium` | Whisper model: tiny/base/small/medium/large |
| `--transcript-format` | `txt` | Output format: txt/srt/vtt/json |
| `--transcription-backend` | `auto` | Transcription backend: auto/mlx/whisper |
| `--diarization-device` | `auto` | Speaker-detection device: auto/cpu/mps/cuda |
| `--stage-concurrency` | `auto` | Transcript-stage policy: auto/serial/parallel |
| `--no-speaker-detection` | | Skip pyannote speaker detection |

## How it works

### Key frame extraction (two-pass)

1. **Pass 1 (streaming CLIP + dHash):** Sample frames at 0.5s intervals, compute compact dHash/metric metadata and CLIP vectors in bounded batches, allocate more clusters to visually novel scenes, and pick scored representatives. Source-resolution frames are released immediately.

2. **Pass 2 (candidate cache + Florence-2 + OCR):** Re-decode only the finite candidate union into a private, lossless cache under the OS temporary directory (or `--frame-cache-dir`), verify it against the first-pass SHA-256 metadata, then caption and OCR in bounded batches before deterministic dedupe.

Scrolling a data table (visually different but semantically identical) gets collapsed, while a dropdown opening (visually similar but semantically distinct) gets preserved.

### Transcript extraction

Whisper always provides the transcript text and segment boundaries. `auto`
selects pinned MLX-Whisper weights on Apple Silicon running macOS 14 or newer;
other machines use OpenAI Whisper on CUDA when available and CPU otherwise.
PyTorch MPS is not used for transcription because it was only marginally faster
and materially changed the tested transcript.

When `HF_TOKEN` is set, pyannote detects speakers with
`pyannote/speaker-diarization-community-1`, and Keyframe assigns a dominant
speaker label to each Whisper segment based on diarization overlap. Speaker
detection shows stage-prefixed progress after model loading and audio decoding.
On Darwin ARM64, automatic speaker detection prefers Torch MPS; elsewhere it
prefers CUDA when available and then CPU. Explicit unavailable MPS or CUDA
requests fail during preflight. An automatic MPS compute failure retries once
in a fresh CPU worker after a new scheduler admission decision. Explicit MPS
requests and non-compute failures remain strict.
MPS workers disable PyTorch's implicit CPU fallback so unsupported kernels
cannot bypass scheduler admission or fallback evidence.
If `HF_TOKEN` is missing or speaker detection fails, Keyframe warns and keeps
the unlabeled Whisper transcript. The two known harmless pyannote warnings
about unavailable TorchCodec decoding and a too-short pooling window are
condensed; changed or unrelated warnings remain visible. Every pyannote row is
validated for finite, non-negative timestamps, positive duration, and a
non-empty speaker before the diarization checkpoint can be published.

Each model stage runs in a disposable spawned process. In `auto` concurrency
mode, stages using independent resources may overlap only when the model-aware
memory check (including 10% headroom) admits them. This retains MLX/CUDA
transcription overlap with CPU diarization and CPU transcription overlap with
MPS diarization. CPU transcription plus CPU diarization additionally requires
at least four CPUs. On macOS, automatic admission uses `memory_pressure -Q`
with physical memory from `sysctl`; a bounded `vm_stat` calculation is the
fallback, and swap is never counted. MLX transcription, MPS diarization, and
MPS frame extraction all share the Apple accelerator and therefore run
serially. A full run makes a fresh scheduling decision after transcription and
again before any CPU diarization fallback. CPU diarization can overlap
MPS/CUDA frames when current pressure admits it; if MPS fails while independent
CPU frame work is already running, the CPU retry starts after those frames.
`parallel` may override
CPU-count and memory admission with a warning, but never shared-accelerator
exclusion.

Without a diarization retry, reliable parent-monotonic intervals select one of
five full-run critical paths:
`max(T + F, D) + M + E`, `max(T, D) + F + M + E`,
`T + max(D, F) + M + E`, `T + D + F + M + E`, or
`T + F + M + E`. Here `T`, `D`, and `F` are transcription, diarization, and
frame intervals; `M` is speaker merge/output writing and `E` is manifest
enrichment/promotion. A failed MPS attempt is represented separately as `R`;
ten retry-aware expressions also cover a failed MPS attempt overlapping CPU
frames before the CPU retry begins. The scheduler derives
the expression from recorded stage intervals rather than inferring overlap from
launch intent.

The raw transcript checkpoint is atomically published as soon as transcription
finishes, before speaker assignment. A later diarization or frame failure does
not erase that completed checkpoint. Final transcript formats retain their
existing schemas and millisecond formatting.

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
  transcript.raw.json       # Full-precision start/end/text checkpoint
  diarization.json          # Full-precision start/end/speaker checkpoint, when run
  transcript.txt            # Timestamped transcript, speaker-labeled when available
  transcript.json           # Machine-readable transcript, includes speaker when available
```

`transcript.raw.json` is durable once transcription succeeds.
`diarization.json` is published only for the current successful speaker-detection
run. In full extraction, frames, captions, and the manifest are staged as one
generation and replace the public `frames/` directory only after validation and
transcript-window enrichment complete.

## Tips

- For UI recordings with many important states: try `--pass1-clusters 20`
- Default transcription uses `--whisper-model medium` and `--transcription-backend auto`
- Use `--no-speaker-detection` when you want an unlabeled transcript or do not want to use `HF_TOKEN`
- Florence-2 uses `florence-community/Florence-2-base` (native transformers support). The original `microsoft/Florence-2-base` weights are broken with transformers 4.50+.
- CLIP model is used for image embedding; deterministic OCR/dHash merge logic handles final dedupe.
