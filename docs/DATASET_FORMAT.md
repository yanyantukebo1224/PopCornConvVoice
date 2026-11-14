# Paired dataset format

The first VC baseline is supervised one-to-one conversion. Each manifest row
pairs one source-speaker A recording with the corresponding target-speaker B
recording.

## Audio requirements

- WAV audio, readable by SoundFile.
- 48,000 Hz.
- Mono.
- Source and target recordings must contain the same utterance and be
  time-aligned. Similar text alone is not sufficient.
- Duration difference must be no more than 20 ms by default.

Use recordings that you own or have explicit permission to use.

## JSONL manifest

Create one JSON object per line in `pairs.jsonl`:

```json
{"pair_id":"p001","source_utterance_id":"A_0001","target_utterance_id":"B_0001","source_audio":"audio/A_0001.wav","target_audio":"audio/B_0001.wav","split":"train","alignment_verified":true}
{"pair_id":"p002","source_utterance_id":"A_0002","target_utterance_id":"B_0002","source_audio":"audio/A_0002.wav","target_audio":"audio/B_0002.wav","split":"validation","alignment_verified":true}
```

Audio paths are relative to the manifest file. Allowed splits are `train`,
`validation`, and `test`.

The loader rejects missing files, wrong sample rates, stereo files, empty
recordings, unverified alignment, large duration mismatches, duplicate pair
IDs, and the same speaker-side utterance appearing in multiple splits.

## Python API

```python
from popcorn_conv_voice.paired_manifest import load_paired_manifest

pairs = load_paired_manifest("dataset/pairs.jsonl")
train_pairs = [pair for pair in pairs if pair.split == "train"]
```

This stage validates the manifest and metadata; waveform loading, alignment
correction, segment sampling, and training are subsequent implementation steps.
