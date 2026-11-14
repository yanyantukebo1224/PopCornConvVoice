# Research plan

## Scope

Build a high-quality, low-latency 48 kHz **one-to-one** voice-conversion system. A fixed source speaker A is converted to a fixed target speaker B. Any-to-any and any-to-one generalization are out of scope for the initial experiment.

AudioDec remains the core foundation. The project uses an external AudioDec checkout; it does not vendor upstream source.

## Evaluation rules

1. Separate codec reconstruction quality from voice-conversion quality.
2. Separate model compute time from buffering / chunking / lookahead latency.
3. Report end-to-end latency and real-time factor, not only isolated GPU kernel time.
4. Use the same audio, chunk size, warm-up, device, and measurement method for comparisons.
5. Tests must focus on PopCornConvVoice-owned code. Do not make standalone AudioDec reconstruction a required test or milestone.

## Implementation sequence

### E1 — integration contract (in progress)

- Connect the external AudioDec encoder, quantizer, and codebook lookup to a replaceable target-speaker decoder.
- Keep the pretrained source encoder and quantizer frozen initially.
- Make the 48 kHz mono tensor contract explicit.
- Unit-test the adapter with mocked AudioDec components.

### E2 — paired data pipeline

- Define a manifest for source-A and target-B recordings.
- Validate sample rate, channel count, duration, file readability, and pair identity.
- Prefer parallel / time-aligned recordings.
- Keep utterances, not random chunks, separated between train and validation.
- Add deterministic segment sampling and alignment metadata.

### E3 — target-speaker decoder training

- Train the decoder on source-A quantized representations paired with corresponding target-B waveform segments.
- Begin with stable reconstruction losses (waveform / multi-resolution STFT / mel as appropriate).
- Add adversarial or teacher losses only after the supervised baseline is stable.
- Evaluate content preservation, target-speaker similarity, prosody, naturalness, and artifacts separately.

### E4 — streaming and latency

- Preserve AudioDec encoder / decoder state across sequential chunks.
- Make reset behavior and chunk-boundary handling explicit.
- Measure buffering, past context, model compute, output buffering, and device I/O separately.
- Compare chunk sizes and precision settings on the actual RTX 3060 Ti.
- Consider LLVC-style teacher/student distillation only after the conversion quality baseline exists.

## Important design constraint

A decoder trained to reconstruct one target speaker does not automatically perform voice conversion. It must learn to map source-A content representations to target-B acoustic output using correctly paired training data.

The AudioDec 300-sample latent hop is 6.25 ms at 48 kHz. This is not total algorithmic or end-to-end latency.

## Milestone exit criteria

- E1: adapter contract is implemented and covered by project-owned unit tests.
- E2: validated paired manifest and deterministic dataset loader.
- E3: reproducible A-to-B training run and held-out conversion baseline.
- E4: streaming implementation with measured latency and quality regression checks.
