---
name: keyframe
description: "Extract key frames and timestamped transcripts from video or audio files. Produces semantically distinct screenshots and optional pyannote speaker labels on Whisper transcript segments."
argument-hint: "<path to video/audio file>"
---

# $keyframe

Extract key frames and/or a timestamped transcript from a video or audio file.

## When to use

- User shares a video file and wants to understand what's in it
- User wants screenshots from a screen recording or meeting recording
- User wants a transcript of spoken content in a video or audio file

## Workflow

1. **Identify the file.** The user provides a path to a video (.mp4, .mov, .mkv) or audio (.m4a, .mp3, .wav) file.

2. **Run the command:**
   ```bash
   keyframe "<path to file>"
   ```
   Output goes to `<input-file-folder>/<filename>_extracted/` by default (falls back to `/tmp` if that folder isn't writable). Use `-o` to override:

   Flags:
   - `--frames-only` — skip transcript
   - `--transcript-only` — skip frames
   - `--no-speaker-detection` — skip pyannote speaker detection
   - `--whisper-model medium` — transcription model (default: medium)
   - `--transcription-backend auto|mlx|whisper` — automatic MLX/Whisper selection or an explicit backend
   - `--diarization-device auto|cpu|mps|cuda` — speaker-detection device
   - `--stage-concurrency auto|serial|parallel` — model-stage scheduling policy
   - `--pass1-clusters 20` — more candidate frames (default: 15)
   - `--similarity-threshold` — deprecated no-op; do not tune with this flag

   When no mode flag is supplied, Keyframe probes the file before creating
   output and selects full extraction for usable video plus audio, frames-only
   for video-only, and transcript-only for audio-only. Attached artwork does
   not count as video. Frame extraction is supported only on Darwin ARM64 and
   Linux x86-64; Linux installs PaddlePaddle and PaddleOCR by default.
   Transcript-only remains available on other platforms. Selected-mode
   configuration, paths, dependencies, and QA targets are validated before
   output/cache creation or model/worker startup.

   The automatic backend uses pinned MLX-Whisper on Apple Silicon running macOS 14 or newer and OpenAI Whisper elsewhere. Supported Macs resolve the exact pinned snapshot from the local cache before permitting a network download; unsupported platforms neither install MLX nor request its weights. Eligible automatic MLX failures retry OpenAI Whisper in a fresh CPU worker. Automatic diarization prefers MPS on Darwin ARM64, CUDA elsewhere when available, and CPU otherwise. Eligible automatic MPS compute failures retry once in a fresh CPU worker; explicit MPS and non-compute failures are strict. MPS workers disable PyTorch's implicit CPU fallback so retries cannot bypass scheduler admission or evidence. Speaker detection is attempted by default when `HF_TOKEN` exists; pyannote then adds segment-level labels to Whisper segments. To enable it, accept the pyannote model terms at `https://huggingface.co/pyannote/speaker-diarization-community-1`, create a Hugging Face token at `https://huggingface.co/settings/tokens`, and export it as `HF_TOKEN`. If `HF_TOKEN` is missing or pyannote access fails, Keyframe warns and keeps the unlabeled Whisper transcript. The two exact known harmless pyannote warnings are condensed; unrelated warnings stay visible, and malformed diarization rows cannot be published.

   In automatic concurrency mode, independent stages can overlap after 10%-headroom admission. macOS uses physical-memory pressure evidence with a bounded `vm_stat` fallback and never counts swap. MLX transcription, MPS diarization, and MPS frames share the Apple accelerator and remain serialized. CPU diarization retains the existing overlap behavior, including a fresh admission decision before a fallback can overlap frames.

   Reliable intervals select one of the five normal topology expressions or ten retry-aware variants. `T`, `R`, `D`, `F`, `M`, and `E` are transcription, failed MPS attempt, successful diarization, frames, transcript merge/output, and manifest enrichment/promotion.

   Release qualification is separate from ordinary user extraction. Keyframe
   0.6.3 includes a synthetic public frame fixture and the
   `keyframe-release-evidence` fresh/replay runner. Fresh artifact evidence is
   limited to Darwin ARM64 and Linux x86-64 and executes the default CLI route
   in an isolated clean artifact environment outside the checkout. Model-free
   replay resolves only regular non-symlinked files beneath the bundle and
   recomputes hashes, normalized OCR target subsets, budgets, redundancy,
   platform, package, and exact source identity instead of trusting stored
   pass fields.

3. **Present results.** Read the transcript first; treat it as narrative authority for what was said. Use `frames/manifest.json` as the frame triage index. Describe only what is visibly shown in frame images, and distinguish “frame visibly shows X” from “speaker said X near this timestamp.”

## Output

```
<output_dir>/
  frames/
    frame_000064_4.00s.png
    captions.json
    manifest.json
  transcript.raw.json         # Durable raw transcript before speaker assignment
  diarization.json            # Current successful pyannote checkpoint, when run
  transcript.txt              # Speaker-labeled when pyannote labels overlap
  transcript.json             # Includes speaker only on labeled segments
```

`transcript.raw.json` remains available if a later independent stage fails.

## Grounding Rules

- Never claim annotations, highlights, arrows, red marks, or callouts unless directly visible.
- If uncertain, say “no annotations visible” or “unclear.”
- Do not describe transcript content as if it appears visually in the frame.

## Installation

If `keyframe` is not found: `pipx install --python python3.12 git+ssh://git@github.com/charlesnpx/keyframe.git && keyframe install-skills`

If Linux frame preflight reports incomplete OCR dependencies, reinstall
Keyframe with its default dependencies or use `--transcript-only`. A missing
or timed-out `ffprobe` requires a working ffmpeg installation.
