# PopCornConvVoice

Research project for high-quality, low-latency **48 kHz one-to-one voice conversion**, using **AudioDec** as the audio representation and synthesis foundation.

## Current research target

- Conversion: one fixed source speaker -> one fixed target speaker
- Sample rate: 48,000 Hz, mono runtime baseline
- Priorities: conversion quality and end-to-end latency equally
- Core: external AudioDec 48 kHz encoder / quantizer plus a trained target-speaker decoder
- Research reference: LLVC teacher/student training and causal streaming ideas
- Local profiling target: NVIDIA RTX 3060 Ti

This repository is a research implementation. It is not yet a trained or production-ready voice-conversion model.

## Architecture

The current integration path is:

1. AudioDec source encoder extracts a content representation from speaker A.
2. AudioDec quantizer produces discrete codes.
3. AudioDec codebook lookup restores the quantized representation.
4. A separately trained PopCornConvVoice decoder synthesizes speaker B.

**The ordinary AudioDec reconstruction decoder is not a voice-conversion model.** A target-speaker decoder must be trained using source-A representations paired with corresponding target-B audio.

See [VC architecture](docs/VC_ARCHITECTURE.md).

## Repository status

- [x] Project scaffold and benchmark helpers
- [x] Lazy AudioDec adapter for a separately trained target decoder
- [x] Mock-based tests for our adapter contract
- [ ] Paired A/B dataset preparation and alignment
- [ ] Target-speaker decoder training
- [ ] Stateful streaming scheduler and device I/O
- [ ] Held-out voice-conversion quality evaluation and latency profiling

AudioDec itself is treated as an external dependency. We do not require a separate AudioDec demo/reconstruction test as a project milestone. Tests focus on PopCornConvVoice-owned code, using mocks where an external model would otherwise be required.

## Windows development

Use the existing Windows Python environment and existing installed packages. Do not create a new virtual environment unless specifically needed to resolve a dependency conflict.

The adapter loads an existing AudioDec checkout at runtime:

- AudioDec checkout root: directory containing `utils/audiodec.py`
- Source encoder checkpoint: pretrained 48 kHz AudioDec encoder
- Target decoder checkpoint: PopCornConvVoice-trained decoder compatible with the AudioDec generator interface

The adapter API is in `src/popcorn_conv_voice/audiodec_adapter.py`.

## Tests

Run the repository-owned tests from the project root:

```powershell
python -m pytest
```

These tests do not launch AudioDec or validate its pretrained checkpoints.

## Licensing

The repository's own code is not yet assigned a license. AudioDec upstream is distributed under CC BY-NC 4.0; review that license before copying code, redistributing checkpoints, or using adapted work commercially.

Upstream: https://github.com/facebookresearch/AudioDec
Paper: https://arxiv.org/abs/2305.16608
