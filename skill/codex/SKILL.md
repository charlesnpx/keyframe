---
name: keyframe
description: "Extract key frames and timestamped transcripts from video or audio files. Produces a folder of semantically distinct screenshots + a Whisper transcript."
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
   Output goes to `/tmp/<filename>_extracted/` by default. Use `-o` to override:

   Flags:
   - `--frames-only` — skip transcript
   - `--transcript-only` — skip frames
   - `--whisper-model medium` — transcription model (default: medium)
   - `--pass1-clusters 20` — more candidate frames (default: 15)
   - `--similarity-threshold` — deprecated no-op; do not tune with this flag

3. **Present results.** Read the transcript first; treat it as narrative authority for what was said. Use `frames/manifest.json` as the frame triage index. Describe only what is visibly shown in frame images, and distinguish “frame visibly shows X” from “speaker said X near this timestamp.”

## Output

```
<output_dir>/
  frames/
    frame_000064_4.00s.png
    captions.json
    manifest.json
  transcript.txt
  transcript.json
```

## Grounding Rules

- Never claim annotations, highlights, arrows, red marks, or callouts unless directly visible.
- If uncertain, say “no annotations visible” or “unclear.”
- Do not describe transcript content as if it appears visually in the frame.

## Installation

If `keyframe` is not found: `pipx install git+ssh://git@github.com/charlesnpx/keyframe.git && keyframe install-skills`
