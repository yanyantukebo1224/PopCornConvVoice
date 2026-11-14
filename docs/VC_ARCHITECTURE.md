# One-to-one VC architecture

## Current direction

PopCornConvVoice is a fixed-speaker conversion system:

- Input identity: one enrolled source speaker (A).
- Output identity: one enrolled target speaker (B).
- Audio format: mono, 48 kHz float audio.
- Runtime target: Windows + NVIDIA CUDA.
- Quality and latency are co-equal optimization goals.

Any-to-any and any-to-one conversion are not part of this first model.

## AudioDec integration

The project uses the external AudioDec checkout as a neural codec runtime. We
do not copy AudioDec source into this repository.

The runtime path is:

1. Source waveform -> pretrained AudioDec encoder.
2. Encoder representation -> AudioDec residual vector quantizer.
3. Quantized codes -> AudioDec codebook lookup.
4. Looked-up representation -> **PopCornConvVoice target-speaker decoder**.
5. Decoder waveform -> output.

The source encoder and codebook provide a stable content representation. The
decoder used for conversion must be trained on source-A representations paired
with target-B waveform segments. The ordinary pretrained AudioDec reconstruction
decoder is not a VC model: using it unchanged reconstructs the input speaker.

## First trainable baseline

Train a target decoder with paired, time-aligned A/B recordings:

- Freeze the pretrained source encoder and quantizer initially.
- Encode A segments into quantized representations.
- Supervise the target decoder against the corresponding B segments.
- Use held-out utterances for evaluation; do not split random chunks from the
  same utterance across train and validation.
- Start with reconstruction losses before adding adversarial objectives.
- Keep all training and validation audio at 48 kHz mono.

Parallel or carefully aligned recordings are strongly preferred. If recordings
are not aligned, alignment must be solved explicitly before treating segments as
paired targets.

## Streaming constraints

The AudioDec encoder and decoder expose stateful streaming methods. A loaded
converter instance is therefore intended to process sequential chunks while
preserving model state. Reset behavior, chunk boundaries, buffering, and
lookahead must be handled explicitly when the streaming engine is added.

The 300-sample latent hop corresponds to 6.25 ms at 48 kHz. It is not the total
system latency. Report buffering, past context, compute time, and I/O latency
separately.

## Current implementation boundary

Implemented:
- A lazy-loading adapter that connects an external AudioDec checkout to a
  separately trained target-speaker decoder.
- Input tensor contract checks and a deterministic encoder/quantizer/lookup/
  decoder call path.

Not implemented yet:
- Target-decoder training.
- Dataset alignment and segment sampling.
- Stateful chunk scheduler and audio-device I/O.
- Perceptual / speaker-similarity evaluation.

Tests in this repository validate PopCornConvVoice's adapter contract with
mocked components. They do not run or certify AudioDec itself.
