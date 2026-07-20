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
