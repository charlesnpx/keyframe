---
name: keyframe
description: "Extract key frames and timestamped transcripts from video or audio files. Produces semantically distinct screenshots and optional pyannote speaker labels on Whisper transcript segments."
argument-hint: "<path to video/audio file>"
---

# /keyframe

Extract key frames and/or a timestamped transcript from a video or audio file.

## When to use

- User shares a video file and wants to understand what's in it
- User wants screenshots from a screen recording or meeting recording
- User wants a transcript of spoken content in a video or audio file
- User wants to extract both frames and transcript for documentation

## Workflow

1. **Identify the file.** The user provides a path to a video (.mp4, .mov, .mkv) or audio (.m4a, .mp3, .wav) file. If unclear, ask.

2. **Choose mode.** Decide based on the file type and user request:
   - Video with speech → full extraction (frames + transcript)
   - Video without speech (screen recording) → `--frames-only`
   - Audio only → `--transcript-only`
   - If unsure, omit the mode flag; Keyframe probes usable streams before
     selecting full, frames-only, or transcript-only extraction

3. **Run the command.** Execute via Bash:
   ```bash
   keyframe "<path to file>"
   ```
   Output goes to `<input-file-folder>/<filename>_extracted/` by default (falls back to `/tmp` if that folder isn't writable). Use `-o` to override:

   Common flags:
   - `--frames-only` — skip transcript extraction
   - `--transcript-only` — skip frame extraction
   - `--no-speaker-detection` — skip pyannote speaker detection
   - `--whisper-model medium` — transcription model (default: medium)
   - `--transcription-backend auto|mlx|whisper` — automatic MLX/Whisper selection or an explicit backend
   - `--diarization-device auto|cpu|mps|cuda` — speaker-detection device
   - `--stage-concurrency auto|serial|parallel` — model-stage scheduling policy
   - `--pass1-clusters 20` — more candidate frames before merging (default: 15)
   - `--similarity-threshold` — deprecated no-op; do not tune with this flag

   Frame extraction is supported only on Darwin ARM64 and Linux x86-64. Linux
   installs PaddlePaddle and PaddleOCR by default. Transcript-only extraction
   remains available on other platforms when the input has usable audio.
   Keyframe validates the input, `ffprobe` result, selected platform,
   configuration, paths, dependencies, and QA targets before creating output
   or cache directories or starting models/workers. Attached album artwork
   does not satisfy video routing.

   The automatic backend uses pinned MLX-Whisper on Apple Silicon running macOS 14 or newer and OpenAI Whisper elsewhere. Supported Macs resolve the exact pinned snapshot from the local cache before permitting a network download; unsupported platforms neither install MLX nor request its weights. Eligible automatic MLX failures retry OpenAI Whisper in a fresh CPU worker. Automatic diarization prefers MPS on Darwin ARM64, CUDA elsewhere when available, and CPU otherwise. Eligible automatic MPS compute failures retry once in a fresh CPU worker; explicit MPS and non-compute failures are strict. MPS workers disable PyTorch's implicit CPU fallback so retries cannot bypass scheduler admission or evidence. Speaker detection is attempted by default when `HF_TOKEN` exists; pyannote then adds segment-level labels to Whisper segments. To enable it, accept the pyannote model terms at `https://huggingface.co/pyannote/speaker-diarization-community-1`, create a Hugging Face token at `https://huggingface.co/settings/tokens`, and export it as `HF_TOKEN`. If `HF_TOKEN` is missing or pyannote access fails, Keyframe warns and keeps the unlabeled Whisper transcript. The two exact known harmless pyannote warnings are condensed; unrelated warnings stay visible, and malformed diarization rows cannot be published.

   In automatic concurrency mode, independent stages can overlap after 10%-headroom admission. macOS uses physical-memory pressure evidence with a bounded `vm_stat` fallback and never counts swap. MLX transcription, MPS diarization, and MPS frames share the Apple accelerator and remain serialized. CPU diarization retains the existing overlap behavior, including a fresh admission decision before a fallback can overlap frames.

   Reliable intervals select one of the five normal topology expressions or ten retry-aware variants. `T`, `R`, `D`, `F`, `M`, and `E` are transcription, failed MPS attempt, successful diarization, frames, transcript merge/output, and manifest enrichment/promotion.

   Release qualification is a separate maintainer workflow, not ordinary user
   extraction. Keyframe 0.6.3 ships a synthetic public frame fixture and the
   `keyframe-release-evidence` fresh/replay runner. Fresh artifact evidence is
   supported only on Darwin ARM64 and Linux x86-64, runs the default CLI route
   through a clean isolated artifact environment outside the checkout, and
   records exact source, package, OCR, model, and artifact provenance. Replay
   is model-free, accepts only regular non-symlinked files beneath its bundle,
   and recomputes normalized target-token, timing, budget, redundancy, hash,
   platform, and source conclusions without trusting stored pass fields.

4. **Present the results.** After extraction completes:
   - Read the transcript first; treat it as narrative authority for what was said
   - Use `frames/manifest.json` as the frame triage index before opening every image
   - Read key frame images to describe only what is visibly shown on screen
   - Distinguish “frame visibly shows X” from “speaker said X near this timestamp”
   - If the user asked a specific question about the video, answer it using the extracted content

## Output structure

```
<output_dir>/
  frames/
    frame_000064_4.00s.png    # Key frames named with frame index + timestamp
    frame_000296_18.48s.png
    ...
    captions.json              # Florence-2 captions + merge metadata
    manifest.json              # Deterministic frame triage index
  transcript.raw.json          # Durable raw transcript before speaker assignment
  diarization.json             # Current successful pyannote checkpoint, when run
  transcript.txt               # Timestamped transcript, speaker-labeled when available
  transcript.json              # Machine-readable transcript; includes speaker when available
```

## Tips

- For UI demo recordings with many similar screens, use `--pass1-clusters 20` to capture more detail
- Frame work scales with the recording and the configured sampling interval;
  allow more time for long or high-resolution videos
- Whisper defaults to `medium`; use `large` only when accuracy is worth the extra time and download
- Speaker labels use raw pyannote labels such as `SPEAKER_00`; JSON includes `speaker` only on labeled segments
- `transcript.raw.json` remains available if a later independent stage fails
- The transcript.json file contains structured `[{start, end, text}]` segments for programmatic use when speaker detection is unavailable or disabled
- Audio-only files (.m4a, .mp3) automatically skip frame extraction even without `--transcript-only`

## Grounding Rules

- Never claim annotations, highlights, arrows, red marks, or callouts unless they are directly visible in the frame.
- If uncertain, say “no annotations visible” or “unclear.”
- Do not describe transcript content as if it appears visually in the frame.
- Use `manifest.json` OCR tokens and transcript windows for triage, then verify visual claims against the PNG.

## Error handling

- If `keyframe` is not found, tell the user to install it with Python 3.12: `pipx install --python python3.12 git+ssh://git@github.com/charlesnpx/keyframe.git && keyframe install-skills`
- If Linux frame preflight reports incomplete OCR dependencies, reinstall
  Keyframe with its default dependencies or use `--transcript-only`
- If `ffprobe` is missing or times out, install/repair ffmpeg before retrying
- If models fail to download (SSL errors), suggest: `/Applications/Python\ 3.12/Install\ Certificates.command`
- If ffmpeg is missing (Whisper needs it), suggest: `brew install ffmpeg`
