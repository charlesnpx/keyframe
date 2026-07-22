# Transcription models to test on Apple Silicon

## Goal

Keyframe needs a local transcription backend that preserves Whisper-level text and timestamp quality while making effective use of Apple Silicon. The backend must support long meeting recordings, structured segments, automatic language detection, and a clean handoff to speaker diarization.

## Current baseline

OpenAI Whisper `medium` through PyTorch uses CUDA when available and CPU otherwise. PyTorch MPS was tested, but it is not a suitable transcription backend: it was only 9% faster than CPU, used roughly twice the memory, required an unsupported sparse-alignment workaround, produced NaNs in FP16, and materially degraded the FP32 transcript.

The current 16m 28.75s English meeting benchmark produced:

| Backend | Wall time | Segments | Words | Peak memory footprint |
| --- | ---: | ---: | ---: | ---: |
| OpenAI Whisper `medium`, CPU FP32 | 225.43s | 275 | 2,513 | 5.87 GiB |
| OpenAI Whisper `medium`, PyTorch MPS FP32 | 205.09s | 131 | 2,581 | 12.03 GiB |
| MLX-Whisper `medium`, full-precision model | 75.38s | 273 | 2,515 | 5.41 GiB |

The machine-readable baseline and release thresholds are tracked in `tests/fixtures/transcription-benchmark-baseline.json`.

## 1. MLX-Whisper

[MLX-Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) runs OpenAI Whisper weights through Apple's MLX framework. It retains the same model family, exposes a Python API, detects language automatically, and returns segment- and word-level timestamps.

Status: adopted as the automatic backend on supported Apple Silicon Macs. Keyframe pins `mlx==0.32.0`, `mlx-whisper==0.4.3`, and immutable MLX Community model revisions. Downloads remain lazy and occur only after the runtime confirms Apple Silicon and macOS 14 or newer. Other platforms retain OpenAI Whisper and neither install MLX nor request MLX weights.

### Benchmark result: 2026-07-20

The benchmark used a 16m 28.75s English meeting recording on a 10-core Apple M4 MacBook Pro with 32 GB of unified memory and macOS 26.5.2. It measured transcription only, without frame extraction or diarization. Dependencies and models were prewarmed, so timings exclude installation and download.

| Metric | OpenAI Whisper CPU | MLX-Whisper | MLX change |
| --- | ---: | ---: | ---: |
| Wall time | 225.43s | 75.38s | 150.05s less; 2.99x throughput |
| Real-time factor | 0.2280 | 0.0762 | 66.56% lower |
| Segments | 275 | 273 | -2 |
| Normalized words | 2,513 | 2,515 | +2 |
| Peak memory footprint | 5.87 GiB | 5.41 GiB | 7.91% lower |
| Maximum resident set size | 5.88 GiB | 1.90 GiB | 67.76% lower |

Quality remained effectively equivalent to the CPU baseline:

- A 30-second smoke test reproduced all eight opening segments and timestamps exactly.
- Full-recording normalized word agreement was 99.443%.
- Word edit distance was 21 words, or 0.836% of the CPU transcript's word count.
- Character agreement was 99.532%.
- Both transcripts had 17 excess duplicate five-grams, so MLX introduced no additional repetition by that measure.
- There was no opening loss, repeated passage, or long-form collapse. Most differences were punctuation or segment boundaries.

The adopted medium model is `mlx-community/whisper-medium-mlx` at revision `7fc08c4eac4c316526498f147dfdee6f6303f975` (model SHA-256 `10b597c2bcb1bcc38b2d3d24cd4f0885f461a7cd70e8444d6ad5a763ece549ea`).

### Packaging compatibility

MLX-Whisper supports Python 3.8 or newer, WhisperX supports Python 3.10 through 3.13, and the benchmark ran on Keyframe's Python 3.12.13 runtime. The shared environment uses WhisperX's stricter compatible matrix: `torch~=2.8.0`, `torchaudio~=2.8.0`, `torchvision~=0.23.0`, `whisperx==3.8.6`, `huggingface-hub>=0.34,<1`, and `transformers>=4.50,<5`.

Keyframe 0.6.2 supports Python 3.11 through 3.13. Its clean-install
matrix covers macOS ARM64 and Linux x86-64 on all three versions. MLX and
MLX-Whisper use Darwin ARM64/macOS 14+ environment markers; the Linux jobs
assert that neither distribution is installed. Import validation does not load
models or touch a Hugging Face cache.

### Adopted runtime behavior

The CLI is the supported product surface. `--transcription-backend auto`
selects MLX on supported Macs and OpenAI Whisper elsewhere. Explicit MLX fails
during preflight on unsupported machines, before imports or model acquisition.
An eligible automatic MLX import, acquisition, load, or inference failure exits
that process before a fresh CPU Whisper fallback starts. On supported Macs, the
exact immutable model revision is first resolved with local-only cache access;
network resolution is attempted only for a genuine cache miss. Reliable worker
metadata records the resolution source and duration.

`transcript.raw.json` is atomically published immediately after transcription,
before pyannote speaker assignment. A successful diarization pass separately
publishes `diarization.json`; final TXT, SRT, VTT, and JSON output behavior is
unchanged. A completed raw checkpoint survives failure in a later independent
stage. Only the two exact known pyannote UserWarnings (unavailable TorchCodec
decoding and a too-short pooling window) are condensed. All near misses remain
visible, and every diarization row must pass strict checkpoint validation.

Automatic diarization prefers MPS on Darwin ARM64, CUDA elsewhere when
available, and CPU otherwise. Explicit unavailable MPS and CUDA requests fail
during preflight. If an automatic MPS attempt fails during model initialization
or inference with an eligible compute error, Keyframe exits that worker and
retries once on CPU. Explicit MPS, authentication, acquisition, audio decoding,
protocol, and checkpoint failures do not trigger fallback. MPS workers force
PyTorch's implicit CPU fallback off, so CPU work cannot bypass resource
admission or attempted-device evidence.

Automatic scheduling overlaps stages using independent resources only when
model-aware memory admission with 10% headroom succeeds. On macOS, admission
uses bounded `memory_pressure -Q` and `sysctl hw.memsize` probes, with
physical-page `vm_stat` fallback and no swap. CPU transcription and CPU
diarization additionally require at least four CPUs. MLX transcription, MPS
diarization, and MPS frames all claim the same Apple accelerator and remain
serialized. Existing CPU-diarization overlap remains available, and an MPS to
CPU fallback receives a fresh admission decision. If an MPS attempt fails while
independent CPU frame work is already running, the CPU retry starts after those
frames instead of aborting the extraction.
Diarization remains holistic; Keyframe does not split recordings or reconcile
per-chunk speaker identities.

Without a diarization retry, reliable intervals support five dependency
expressions:
`max(T + F, D) + M + E`, `max(T, D) + F + M + E`,
`T + max(D, F) + M + E`, `T + D + F + M + E`, and
`T + F + M + E`. `T`, `D`, and `F` are the stage intervals, `M` is transcript
merge/output, and `E` is frame-manifest enrichment/promotion. Ten additional
expressions introduce `R` for a failed MPS attempt, including late failures
that overlap independent CPU frame work before the CPU retry begins.

### MPS diarization spike: 2026-07-21

A strict-MPS long-form spike completed pyannote inference in 75.36 seconds and
model initialization in 1.38 seconds. The MPS result contained the same 253
speaker intervals as the CPU reference after label reconciliation, with zero
timestamp delta. Peak Torch MPS driver allocation was 3.80 GiB and peak process
RSS was 1.43 GiB. This established both the quality equivalence and the margin
for the release gate's 335-second MPS diarization ceiling.

The post-implementation full-pipeline candidate then completed in 332.6
seconds with PyTorch's implicit MPS-to-CPU fallback disabled: 79.0 seconds for
MLX transcription, 85.8 seconds for MPS diarization, and 167.1 seconds for MPS
frame extraction.
It used the expected serialized `T + D + F + M + E` path, published every
checkpoint and final artifact, and did not use transcription or diarization
fallback.

### Release pipeline benchmark: 2026-07-21

The 0.6.0 release gate used the same 988.75-second meeting on a 10-core Apple
M4 MacBook Pro with 32 GB unified memory and macOS 26.5.2. Dependencies, model
snapshots, and short hardware paths were prewarmed. The serial reference ran
OpenAI Whisper CPU transcription followed by CPU diarization. The candidate ran
the complete pipeline: MLX transcription and CPU diarization began together,
then MPS frame extraction overlapped the remaining diarization after MLX exited.

| Run | Wall time | Transcription | Diarization | Frames | Peak resident memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| Serial CPU transcript + diarization reference | 847.22s | 204.67s | 642.53s | — | 4.93 GiB |
| Concurrent MLX + diarization + full frames candidate | 613.67s | 89.39s | 613.62s | 189.23s | 5.28 GiB |

The candidate completed 233.55 seconds sooner (27.57% lower wall time, 1.38x
throughput) despite also producing and enriching a 16-frame generation. Speaker
merge took 0.012 seconds and manifest enrichment/promotion took 0.035 seconds.
Its predicted critical path was 613.666 seconds and measured wall time was
613.675 seconds, a 0.009-second delta within the fixed five-second process and
scheduler tolerance. MLX and MPS never overlapped.

The raw candidate transcript retained 99.443% normalized word agreement with
the CPU reference: 21 edits across 2,513 reference words (0.836% word error
rate), 99.699% normalized character agreement, two additional words, and a
0.727% segment-count delta. Both transcripts contained 17 duplicate
five-grams, the first 138 normalized segments matched, and both covered through
987.76 seconds. Diarization was exactly equivalent across all 253 rows after
speaker-label reconciliation, with zero timestamp delta. Both raw and
diarization checkpoints, final TXT/JSON speaker-labeled transcripts, and the
candidate frame manifest were present; the machine-readable report passed all
checked-in thresholds with no failures.

## 2. WhisperKit and SpeakerKit

[WhisperKit](https://github.com/argmaxinc/argmax-oss-swift) runs Whisper through Core ML. The same project includes SpeakerKit, which runs pyannote Community-1 diarization through Core ML.

- Strongest follow-up candidate for an end-to-end Mac-native stack.
- Can use the GPU and Apple Neural Engine without PyTorch MPS limitations.
- Integration would use a small Swift helper, local server, or subprocess boundary.
- Compare a recommended nonquantized `large-v3-turbo` configuration before considering compressed models.

## 3. whisper.cpp

[whisper.cpp](https://github.com/ggml-org/whisper.cpp) is a mature C/C++ Whisper implementation with Metal acceleration and an optional Core ML encoder.

- Offers a stable CLI and C API with low runtime overhead.
- Decoder and segmentation behavior can differ from OpenAI Whisper despite sharing weights.
- Test an unquantized F16 model before treating Q5 or Q8 variants as quality-equivalent.

## 4. Apple SpeechTranscriber

[SpeechTranscriber](https://developer.apple.com/documentation/speech/speechtranscriber) and SpeechAnalyzer are system-native on macOS 26 and use downloadable Apple speech models.

- Designed for general-purpose file and streaming transcription.
- Requires a small Swift helper because the API is Swift-concurrency-oriented.
- It is a different model, so meeting jargon, accents, punctuation, omissions, and timestamps need independent review.

## 5. Different models through MLX-Audio

[MLX-Audio](https://github.com/Blaizzy/mlx-audio) exposes non-Whisper speech models on Apple Silicon.

- NVIDIA Parakeet TDT 0.6B v3 is the first non-Whisper candidate to test.
- Qwen3-ASR is a multilingual follow-up with a separate forced aligner for word timestamps.
- Voxtral and VibeVoice-ASR are heavier and lower priority for Keyframe's focused transcript stage.

## Test order

1. MLX-Whisper `medium`, full-precision model (completed and adopted).
2. WhisperKit plus SpeakerKit.
3. whisper.cpp with Metal, then its Core ML encoder.
4. Apple SpeechTranscriber.
5. Parakeet TDT v3, then Qwen3-ASR.

## Evaluation protocol

Run every backend against the same source recording and retain:

- Backend, package version, model repository, and immutable revision.
- Raw structured transcript before diarization.
- Wall time, real-time factor, CPU time, and peak memory.
- Segment count and duration distribution.
- Normalized word count, sequence agreement, edit distance, omissions, insertions, and repeated n-grams versus the CPU baseline.
- Timestamp coverage and manual review of representative disagreement windows.

A backend passes only if it has no systematic opening loss, repeated passages, long-form drift, or material timestamp regression. The CPU transcript is a comparison baseline rather than ground truth, so material disagreements must be reviewed against the audio.

The release benchmark is parameterized and keeps user-specific paths out of the
repository:

```bash
python scripts/benchmark_transcription.py \
  --input /path/to/meeting.mp4 \
  --baseline tests/fixtures/transcription-benchmark-baseline.json \
  --output /tmp/keyframe-release-benchmark
```

It runs the CPU Whisper/CPU diarization reference serially and uses automatic
scheduling for the complete MLX/MPS-diarization/MPS-frame candidate. The
candidate must obtain serial Apple-accelerator decisions from macOS pressure
evidence, use a local pinned-model cache hit in under one second, make exactly
one MPS diarization attempt, and avoid both transcription and diarization
fallback. The report records reliable intervals, kernel-recorded per-process
RSS high-water marks, MLX allocator peak, transcript agreement,
checkpoint/final artifacts, and diarization equivalence modulo speaker-label
renaming. Its phase-aware conservative tree bound uses the largest serialized
worker in each phase and retains reaped-child peaks, so a short-lived allocation
cannot disappear between polling samples.

The release limits are named in the validator: no more than 115% of the
historical 613.67-second candidate, at least 15% faster than the same-run serial
reference, at most 6.60 GiB for the conservative process-tree RSS high-water
bound, at most 5.96 GiB MLX allocator peak, no more than 335 seconds for MPS
diarization, and at most five seconds between the interval-derived prediction
and wall clock. Existing reports can be revalidated with `--replay-report`;
replay binds the report hash and duration to the explicitly supplied recording.
