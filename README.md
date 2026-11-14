# PopCornConvVoice

Research project for high-quality, low-latency **48 kHz one-to-one voice conversion**, using **AudioDec** as the core audio representation / synthesis foundation.

## Current research target

- Conversion: one fixed source speaker -> one fixed target speaker
- Sample rate: 48,000 Hz
- Priorities: conversion quality and end-to-end latency
- Core: AudioDec (48 kHz configuration; start with AD v1 as the quality baseline)
- Research reference: LLVC's teacher/student training and streaming implementation ideas
- Local profiling target: NVIDIA RTX 3060 Ti

This repository is a research scaffold, not a trained voice-conversion model. No model-quality or latency claims are made until measured.

## Why start with AudioDec v1?

AudioDec provides a causal, streaming-capable 48 kHz neural codec with separable encoder and vocoder components. The first milestone is to reproduce its reconstruction path and measure it locally before changing the architecture for voice conversion.

We will compare:
1. AudioDec v1 reconstruction baseline.
2. AudioDec v2 as a speed/quality reference.
3. One-to-one VC adaptation using paired source/target recordings and/or carefully generated teacher targets.
4. Causal streaming and student distillation only after a quality baseline exists.

## Repository status

**Stage 0 — project scaffold.** AudioDec integration, pretrained checkpoint acquisition, dataset preparation, and actual VC training are not implemented yet.

See:
- [Research plan](docs/RESEARCH_PLAN.md)
- [AudioDec notes](docs/AUDIODEC_NOTES.md)
- [Benchmark protocol](docs/BENCHMARKS.md)

## Setup

Python 3.10+ is recommended. Create an isolated environment, then install the project:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e ".[dev]"
```

The initial scaffold intentionally keeps core dependencies small. AudioDec has its own upstream dependency and checkpoint requirements; those will be pinned in a dedicated integration step rather than guessed here.

## Run tests

```bash
python -m pytest
```

## License

The repository's own code is not yet assigned a license. AudioDec upstream is distributed under CC BY-NC 4.0; review that license before copying code, redistributing checkpoints, or using derived work commercially.

Upstream: https://github.com/facebookresearch/AudioDec
Paper: https://arxiv.org/abs/2305.16608
