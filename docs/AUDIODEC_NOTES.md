# AudioDec research notes

## Upstream

- Repository: https://github.com/facebookresearch/AudioDec
- Paper: https://arxiv.org/abs/2305.16608
- Demo: https://bigpon.github.io/AudioDec_demo/

AudioDec is a streaming neural audio codec, not a voice-conversion model. Its encoder maps waveform audio to a compact representation and its decoder / vocoder reconstructs waveform audio.

## 48 kHz configuration

The upstream 48 kHz VCTK configuration uses a total encoder stride of 300 samples:

`3 x 4 x 5 x 5 = 300`

At 48,000 samples/s, this corresponds to a 6.25 ms latent frame interval (160 frames/s). The published configuration uses a 64-dimensional latent and 8 residual vector-quantizer codebooks of size 1024, corresponding to 12.8 kbps at that frame rate.

These are codec configuration facts, not a guarantee of 6.25 ms end-to-end VC latency.

## Variants

- symAD: symmetric autoencoder baseline.
- AD v0: autoencoder encoder plus a separately trained HiFi-GAN-style vocoder.
- AD v1: group-convolution vocoder variant; use as the initial quality baseline.
- AD v2: smaller-kernel group-convolution variant; use as a speed/quality comparison.

Published MOS values measure codec reconstruction, not voice-conversion quality or target-speaker similarity.

## Causality and latency

The upstream design uses causal operations for streaming. The latent frame interval is not the same thing as total algorithmic latency. Measure separately:

- required input buffering / chunk duration;
- past-context state;
- encoder compute;
- decoder/vocoder compute;
- any lookahead (which should be zero for strictly causal layers);
- output buffering and device I/O.

The paper's RTX 3090 timing results must not be extrapolated directly to an RTX 3060 Ti. Benchmark the actual implementation on the target hardware.

## Licensing

The upstream AudioDec repository states CC BY-NC 4.0. Before copying upstream source, redistributing checkpoints, or using adapted code in a commercial product, review the license and its implications. This note is not legal advice.
