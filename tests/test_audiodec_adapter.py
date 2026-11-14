from contextlib import nullcontext

import pytest

from popcorn_conv_voice.audiodec_adapter import (
    AudioDecAdapterError,
    AudioDecOneToOneConverter,
)


class FakeTensor:
    ndim = 3

    def __init__(self, shape):
        self.shape = shape
        self.slices = []

    def __getitem__(self, key):
        self.slices.append(key)
        end = key[-1].stop if isinstance(key, tuple) and isinstance(key[-1], slice) else None
        length = self.shape[-1] if end is None else min(self.shape[-1], end)
        return FakeTensor((*self.shape[:-1], length))


class FakeEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, waveform):
        self.calls.append("encode")
        return "encoded"

    def quantize(self, encoded):
        self.calls.append(("quantize", encoded))
        return "codes"

    def lookup(self, codes):
        self.calls.append(("lookup", codes))
        return "quantized"


class FakeDecoder:
    def __init__(self):
        self.calls = []

    def decode(self, quantized):
        self.calls.append(("decode", quantized))
        return FakeTensor((1, 1, 6000))


class FakeTorch:
    inference_mode = staticmethod(nullcontext)


class FakeCodec:
    def __init__(self):
        self.tx_encoder = FakeEncoder()
        self.rx_encoder = FakeEncoder()
        self.decoder = FakeDecoder()


def make_loaded_converter():
    converter = AudioDecOneToOneConverter(
        audiodec_root="unused",
        encoder_checkpoint="unused-encoder.pkl",
        target_decoder_checkpoint="unused-target-decoder.pkl",
    )
    converter._torch = FakeTorch()
    converter._codec = FakeCodec()
    return converter


def test_convert_runs_encoder_quantizer_lookup_and_target_decoder():
    converter = make_loaded_converter()
    waveform = FakeTensor((1, 1, 4800))

    output = converter.convert(waveform)

    assert output.shape == (1, 1, 4800)
    assert converter._codec.tx_encoder.calls == [
        "encode",
        ("quantize", "encoded"),
    ]
    assert converter._codec.rx_encoder.calls == [("lookup", "codes")]
    assert converter._codec.decoder.calls == [("decode", "quantized")]


def test_convert_requires_load_first():
    converter = AudioDecOneToOneConverter(
        audiodec_root="unused",
        encoder_checkpoint="unused-encoder.pkl",
        target_decoder_checkpoint="unused-target-decoder.pkl",
    )

    with pytest.raises(AudioDecAdapterError, match="load"):
        converter.convert(FakeTensor((1, 1, 4800)))


def test_convert_rejects_non_mono_input():
    converter = make_loaded_converter()

    with pytest.raises(ValueError, match="mono"):
        converter.convert(FakeTensor((1, 2, 4800)))


def test_convert_rejects_empty_input():
    converter = make_loaded_converter()

    with pytest.raises(ValueError, match="at least one sample"):
        converter.convert(FakeTensor((1, 1, 0)))
