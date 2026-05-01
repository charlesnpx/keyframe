---
name: keyframe
description: "Extract key frames and timestamped transcripts from video or audio files. Produces a folder of semantically distinct screenshots + a Whisper transcript."
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
   - If unsure, run full extraction — it handles both gracefully

3. **Run the command.** Execute via Bash:
   ```bash
   keyframe "<path to file>"
   ```
   Output goes to `/tmp/<filename>_extracted/` by default. Use `-o` to override:

   Common flags:
   - `--frames-only` — skip transcript extraction
   - `--transcript-only` — skip frame extraction
   - `--whisper-model medium` — transcription model (default: medium)
   - `--pass1-clusters 20` — more candidate frames before merging (default: 15)
   - `--similarity-threshold` — deprecated no-op; do not tune with this flag

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
  transcript.txt               # Timestamped transcript
  transcript.json              # Machine-readable transcript
```

## Tips

- For UI demo recordings with many similar screens, use `--pass1-clusters 20` to capture more detail
- For long videos, the frame extraction takes ~20-30s regardless of length (it samples at 0.5s intervals)
- Whisper defaults to `medium`; use `large` only when accuracy is worth the extra time and download
- The transcript.json file contains structured `[{start, end, text}]` segments for programmatic use
- Audio-only files (.m4a, .mp3) automatically skip frame extraction even without `--transcript-only`

## Grounding Rules

- Never claim annotations, highlights, arrows, red marks, or callouts unless they are directly visible in the frame.
- If uncertain, say “no annotations visible” or “unclear.”
- Do not describe transcript content as if it appears visually in the frame.
- Use `manifest.json` OCR tokens and transcript windows for triage, then verify visual claims against the PNG.

## Error handling

- If `keyframe` is not found, tell the user to install it: `pipx install git+ssh://git@github.com/charlesnpx/keyframe.git && keyframe install-skills`
- If models fail to download (SSL errors), suggest: `/Applications/Python\ 3.14/Install\ Certificates.command`
- If ffmpeg is missing (Whisper needs it), suggest: `brew install ffmpeg`
