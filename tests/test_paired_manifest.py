import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from popcorn_conv_voice.paired_manifest import AudioMetadata, load_paired_manifest


def metadata_reader(_path):
    return AudioMetadata(sample_rate=48_000, channels=1, frames=48_000)


def write_manifest(tmp_path: Path, rows):
    (tmp_path / "source.wav").write_bytes(b"placeholder")
    (tmp_path / "target.wav").write_bytes(b"placeholder")
    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    return manifest


def row(**overrides):
    result = {
        "pair_id": "pair-001",
        "source_utterance_id": "source-001",
        "target_utterance_id": "target-001",
        "source_audio": "source.wav",
        "target_audio": "target.wav",
        "split": "train",
        "alignment_verified": True,
    }
    result.update(overrides)
    return result


def test_loads_valid_pair_and_resolves_relative_paths(tmp_path):
    manifest = write_manifest(tmp_path, [row()])

    pairs = load_paired_manifest(manifest, metadata_reader=metadata_reader)

    assert len(pairs) == 1
    assert pairs[0].source_audio == (tmp_path / "source.wav").resolve()
    assert pairs[0].split == "train"


def test_rejects_unverified_alignment(tmp_path):
    manifest = write_manifest(tmp_path, [row(alignment_verified=False)])

    with pytest.raises(ValueError, match="alignment"):
        load_paired_manifest(manifest, metadata_reader=metadata_reader)


def test_rejects_utterance_leakage_between_splits(tmp_path):
    manifest = write_manifest(
        tmp_path,
        [
            row(),
            row(
                pair_id="pair-002",
                target_utterance_id="target-002",
                split="validation",
            ),
        ],
    )

    with pytest.raises(ValueError, match="both"):
        load_paired_manifest(manifest, metadata_reader=metadata_reader)


def test_rejects_duration_mismatch(tmp_path):
    manifest = write_manifest(tmp_path, [row()])

    def mismatched_metadata(path):
        frames = 48_000 if path.name == "source.wav" else 60_000
        return AudioMetadata(sample_rate=48_000, channels=1, frames=frames)

    with pytest.raises(ValueError, match="duration mismatch"):
        load_paired_manifest(manifest, metadata_reader=mismatched_metadata)


def test_rejects_non_mono_audio(tmp_path):
    manifest = write_manifest(tmp_path, [row()])

    def stereo_metadata(_path):
        return AudioMetadata(sample_rate=48_000, channels=2, frames=48_000)

    with pytest.raises(ValueError, match="mono"):
        load_paired_manifest(manifest, metadata_reader=stereo_metadata)
