# Research plan

## Scope

Build a high-quality, low-latency 48 kHz **one-to-one** voice-conversion system. A fixed source speaker is converted to a fixed target speaker. Any-to-any and any-to-one generalization are explicitly out of scope for the initial experiment.

AudioDec remains the core foundation. Do not replace it with a different codec as the default plan.

## Non-negotiable evaluation rules

1. Separate codec reconstruction quality from voice-conversion quality.
2. Separate model compute time from buffering / chunking / lookahead latency.
3. Report end-to-end latency and real-time factor, not only isolated GPU kernel time.
4. Use the same audio, chunk size, warm-up, device, and measurement method for comparisons.
5. Keep a known-good AudioDec reconstruction baseline before modifying the architecture.

## Proposed experiment sequence

### E0 — reproduce AudioDec

- Inspect upstream 48 kHz configurations and checkpoint provenance.
- Reproduce the official reconstruction path.
- Verify sample rate, tensor shape, normalization, chunk handling, and causal state behavior.
- Measure encoder, decoder/vocoder, and complete-path latency independently.
- Compare AD v1 and AD v2; v1 is the initial quality reference.

### E1 — characterize the representation

- Determine which source-speaker cues remain in the encoder representation.
- Compare frozen encoder vs fine-tuned encoder using held-out paired utterances.
- Do not assume a multi-speaker codec representation is speaker-invariant.
- Inspect whether the RVQ bottleneck helps or harms conversion quality.

### E2 — one-to-one conversion

- Use source-speaker A and target-speaker B recordings.
- Prefer aligned / parallel utterances when available.
- Establish a direct supervised baseline before introducing teacher-generated pairs.
- Evaluate content preservation, target-speaker similarity, prosody, naturalness, and artifacts separately.

### E3 — quality refinement

- Test multi-resolution STFT and mel losses.
- Test waveform and feature-level teacher losses where a suitable teacher exists.
- Test adversarial training only after a stable non-adversarial baseline.
- Preserve pitch, timing, and expressive prosody rather than optimizing speaker similarity alone.

### E4 — latency optimization

- Only after quality is measured, profile causal chunk sizes and state caching.
- Compare v1 and v2 vocoder paths.
- Evaluate reduced precision and compilation separately from architecture changes.
- Distill to a smaller causal student only if the quality baseline is already strong.

## Important design constraint

A decoder trained to reconstruct one target speaker does not automatically perform voice conversion. The encoder representation must preserve linguistic content and timing while not forcing the decoder to reproduce the source speaker's identity. This must be demonstrated experimentally, not assumed.

## Milestone exit criteria

- E0: reproducible 48 kHz reconstruction and benchmark report.
- E1: evidence-based encoder / quantizer choice.
- E2: measurable A-to-B conversion baseline.
- E3: quality improvements verified on held-out recordings.
- E4: latency improvements without unacceptable quality regression.
