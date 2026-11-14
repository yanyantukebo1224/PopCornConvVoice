# Benchmark protocol

## Latency

Record all of the following for every run:

- GPU and driver / CUDA versions
- PyTorch version and precision
- sample rate (must be 48,000 Hz for target comparisons)
- input chunk size in samples and milliseconds
- warm-up count and measured iteration count
- encoder, decoder, and full model p50 / p95 / mean
- real-time factor (processing time / audio duration)
- algorithmic buffering latency and measured end-to-end latency

Synchronize CUDA before and after timed GPU regions. Do not report asynchronous launch time as inference time.

## Quality

Use a held-out set that is not used for training. Report at least:

- intelligibility / content preservation
- target-speaker similarity
- naturalness
- prosody and pitch preservation
- transient / consonant artifacts
- clipping, silence handling, and stability across chunk boundaries

Codec reconstruction MOS must be reported separately from VC metrics.

## Comparison discipline

- Keep audio normalization identical.
- Keep source and target utterance splits fixed.
- Do not compare different chunk sizes as if they were the same latency condition.
- Include a non-streaming offline quality reference where useful.
- Store raw metric outputs and configuration alongside summary tables.
