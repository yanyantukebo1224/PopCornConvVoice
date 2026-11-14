"""Runtime adapter for a target-speaker decoder built on AudioDec.

The upstream AudioDec repository is an external dependency. This module does not
copy or modify upstream code; it loads the user's existing AudioDec checkout.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any


SAMPLE_RATE = 48_000
EXPECTED_CHANNELS = 1
LATENT_HOP_SAMPLES = 300


class AudioDecAdapterError(RuntimeError):
    """Raised when the external AudioDec runtime cannot be initialized."""


class AudioDecOneToOneConverter:
    """Convert one fixed source speaker to one fixed target speaker.

    The source encoder and codebook are loaded from an AudioDec checkpoint.
    The decoder checkpoint must be a separately trained target-speaker decoder.
    Passing the ordinary AudioDec reconstruction decoder will only reconstruct
    the source; it does not perform voice conversion.
    """

    def __init__(
        self,
        *,
        audiodec_root: str | Path,
        encoder_checkpoint: str | Path,
        target_decoder_checkpoint: str | Path,
        device: str = "cuda:0",
        receptive_length: int = 8192,
    ) -> None:
        self.audiodec_root = Path(audiodec_root).expanduser().resolve()
        self.encoder_checkpoint = Path(encoder_checkpoint).expanduser().resolve()
        self.target_decoder_checkpoint = Path(target_decoder_checkpoint).expanduser().resolve()
        self.device = device
        self.receptive_length = receptive_length
        self._codec: Any | None = None
        self._torch: Any | None = None

    def load(self) -> None:
        """Load AudioDec's encoder/codebook and the trained target decoder."""
        required = {
            "AudioDec checkout": self.audiodec_root / "utils" / "audiodec.py",
            "source encoder checkpoint": self.encoder_checkpoint,
            "target decoder checkpoint": self.target_decoder_checkpoint,
        }
        missing = [f"{label}: {path}" for label, path in required.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Required AudioDec files are missing:\n" + "\n".join(missing))

        root_string = str(self.audiodec_root)
        if root_string not in sys.path:
            sys.path.insert(0, root_string)

        try:
            torch = importlib.import_module("torch")
            audiodec_module = importlib.import_module("utils.audiodec")
            codec = audiodec_module.AudioDec(
                tx_device=self.device,
                rx_device=self.device,
                receptive_length=self.receptive_length,
            )
            codec.load_transmitter(str(self.encoder_checkpoint))
            codec.load_receiver(
                str(self.encoder_checkpoint),
                str(self.target_decoder_checkpoint),
            )
        except Exception as exc:
            raise AudioDecAdapterError(
                "Could not initialize the AudioDec runtime. Check the external "
                "AudioDec checkout, its dependencies, and checkpoint compatibility."
            ) from exc

        self._torch = torch
        self._codec = codec

    def convert(self, waveform: Any) -> Any:
        """Convert a mono float tensor shaped (batch, 1, samples) at 48 kHz.

        The AudioDec encoder/decoder are stateful streaming modules. Repeated
        calls on the same loaded instance continue their streaming state.
        """
        if self._codec is None or self._torch is None:
            raise AudioDecAdapterError("Call load() before convert().")
        if getattr(waveform, "ndim", None) != 3:
            raise ValueError("waveform must have shape (batch, 1, samples)")
        if waveform.shape[1] != EXPECTED_CHANNELS:
            raise ValueError("AudioDec VC currently requires mono audio (1 channel)")
        if waveform.shape[-1] == 0:
            raise ValueError("waveform must contain at least one sample")

        with self._torch.inference_mode():
            encoded = self._codec.tx_encoder.encode(waveform)
            codes = self._codec.tx_encoder.quantize(encoded)
            quantized = self._codec.rx_encoder.lookup(codes)
            converted = self._codec.decoder.decode(quantized)

        return converted[..., : waveform.shape[-1]]
