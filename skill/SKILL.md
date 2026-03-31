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
   keyframe "<path to file>" -o "<output directory>"
   ```

   Common flags:
   - `--frames-only` — skip transcript extraction
   - `--transcript-only` — skip frame extraction
   - `--whisper-model medium` — faster transcription (default: large)
   - `--pass1-clusters 20` — more candidate frames before merging (default: 15)
   - `--similarity-threshold 0.80` — less aggressive merging (default: 0.85)

4. **Present the results.** After extraction completes:
   - Read the transcript file and summarize what was said
   - Read the key frame images to describe what's shown on screen
   - Cross-reference timestamps between frames and transcript to build a narrative
   - If the user asked a specific question about the video, answer it using the extracted content

## Output structure

```
<output_dir>/
  frames/
    frame_000064_4.00s.png    # Key frames named with frame index + timestamp
    frame_000296_18.48s.png
    ...
    captions.json              # Florence-2 captions + merge metadata
  transcript.txt               # Timestamped transcript
  transcript.json              # Machine-readable transcript
```

## Tips

- For UI demo recordings with many similar screens, use `--pass1-clusters 20 --similarity-threshold 0.80` to capture more detail
- For long videos, the frame extraction takes ~20-30s regardless of length (it samples at 0.5s intervals)
- Whisper large model gives best accuracy but is slower; use `--whisper-model medium` for faster results
- The transcript.json file contains structured `[{start, end, text}]` segments for programmatic use
- Audio-only files (.m4a, .mp3) automatically skip frame extraction even without `--transcript-only`

## Error handling

- If `keyframe` is not found, tell the user to install it: `pipx install git+ssh://git@github.com/charlesnpx/key-frame.git && keyframe install-skills`
- If models fail to download (SSL errors), suggest: `/Applications/Python\ 3.14/Install\ Certificates.command`
- If ffmpeg is missing (Whisper needs it), suggest: `brew install ffmpeg`
