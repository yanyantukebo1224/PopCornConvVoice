"""Manifest parsing and validation for aligned one-to-one VC training pairs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


SAMPLE_RATE = 48_000
ALLOWED_SPLITS = {"train", "validation", "test"}


@dataclass(frozen=True)
class AudioMetadata:
    sample_rate: int
    channels: int
    frames: int

    @property
    def duration_seconds(self) -> float:
        return self.frames / self.sample_rate


@dataclass(frozen=True)
class PairedUtterance:
    pair_id: str
    source_utterance_id: str
    target_utterance_id: str
    source_audio: Path
    target_audio: Path
    split: str
    alignment_verified: bool


def _read_audio_metadata(path: Path) -> AudioMetadata:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "Reading audio metadata requires soundfile. Install it in the existing "
            "Python environment with: python -m pip install soundfile"
        ) from exc

    info = sf.info(str(path))
    return AudioMetadata(
        sample_rate=int(info.samplerate),
        channels=int(info.channels),
        frames=int(info.frames),
    )


def load_paired_manifest(
    manifest_path: str | Path,
    *,
    metadata_reader: Callable[[Path], AudioMetadata] = _read_audio_metadata,
    duration_tolerance_seconds: float = 0.02,
) -> list[PairedUtterance]:
    """Load JSONL pairs and validate files, format, alignment, and split leakage.

    Paths in the JSONL file are resolved relative to the manifest's directory.
    Each row must contain pair_id, source_utterance_id, target_utterance_id,
    source_audio, target_audio, split, and alignment_verified=true.
    """
    manifest = Path(manifest_path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if duration_tolerance_seconds < 0:
        raise ValueError("duration_tolerance_seconds must be >= 0")

    records: list[PairedUtterance] = []
    seen_pair_ids: set[str] = set()
    utterance_splits: dict[tuple[str, str], str] = {}

    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}") from exc

            required = (
                "pair_id",
                "source_utterance_id",
                "target_utterance_id",
                "source_audio",
                "target_audio",
                "split",
                "alignment_verified",
            )
            missing = [key for key in required if key not in row]
            if missing:
                raise ValueError(
                    f"Line {line_number} is missing required fields: {', '.join(missing)}"
                )

            values = [row[key] for key in required[:6]]
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"Line {line_number} contains an empty or non-string field")
            if row["split"] not in ALLOWED_SPLITS:
                raise ValueError(
                    f"Line {line_number}: split must be one of {sorted(ALLOWED_SPLITS)}"
                )
            if row["alignment_verified"] is not True:
                raise ValueError(
                    f"Line {line_number}: source/target alignment must be explicitly verified"
                )
            if row["pair_id"] in seen_pair_ids:
                raise ValueError(f"Duplicate pair_id on line {line_number}: {row['pair_id']}")
            seen_pair_ids.add(row["pair_id"])

            source_audio = (manifest.parent / row["source_audio"]).resolve()
            target_audio = (manifest.parent / row["target_audio"]).resolve()
            for label, audio_path in (
                ("source_audio", source_audio),
                ("target_audio", target_audio),
            ):
                if not audio_path.is_file():
                    raise FileNotFoundError(
                        f"Line {line_number}: {label} does not exist: {audio_path}"
                    )

            source_meta = metadata_reader(source_audio)
            target_meta = metadata_reader(target_audio)
            for label, metadata in (
                ("source_audio", source_meta),
                ("target_audio", target_meta),
            ):
                if metadata.sample_rate != SAMPLE_RATE:
                    raise ValueError(
                        f"Line {line_number}: {label} must be {SAMPLE_RATE} Hz, "
                        f"got {metadata.sample_rate} Hz"
                    )
                if metadata.channels != 1:
                    raise ValueError(
                        f"Line {line_number}: {label} must be mono, got {metadata.channels} channels"
                    )
                if metadata.frames <= 0:
                    raise ValueError(f"Line {line_number}: {label} is empty")

            duration_delta = abs(
                source_meta.duration_seconds - target_meta.duration_seconds
            )
            if duration_delta > duration_tolerance_seconds:
                raise ValueError(
                    f"Line {line_number}: source/target duration mismatch "
                    f"({duration_delta:.3f}s > {duration_tolerance_seconds:.3f}s)"
                )

            for role, utterance_id in (
                ("source", row["source_utterance_id"]),
                ("target", row["target_utterance_id"]),
            ):
                key = (role, utterance_id)
                previous_split = utterance_splits.get(key)
                if previous_split is not None and previous_split != row["split"]:
                    raise ValueError(
                        f"Utterance {utterance_id!r} for {role} speaker appears in "
                        f"both {previous_split!r} and {row['split']!r} splits"
                    )
                utterance_splits[key] = row["split"]

            records.append(
                PairedUtterance(
                    pair_id=row["pair_id"],
                    source_utterance_id=row["source_utterance_id"],
                    target_utterance_id=row["target_utterance_id"],
                    source_audio=source_audio,
                    target_audio=target_audio,
                    split=row["split"],
                    alignment_verified=True,
                )
            )

    if not records:
        raise ValueError("Manifest contains no paired utterances")
    return records
