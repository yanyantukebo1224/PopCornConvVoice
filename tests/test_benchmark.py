import pytest

from popcorn_conv_voice.benchmark import _percentile, benchmark_callable


def test_percentile_interpolates():
    assert _percentile([1.0, 2.0, 3.0, 4.0], 50.0) == 2.5


def test_percentile_rejects_empty_values():
    with pytest.raises(ValueError):
        _percentile([], 95.0)


def test_benchmark_reports_chunk_rtf():
    counter = 0

    def operation():
        nonlocal counter
        counter += 1

    report = benchmark_callable(
        operation,
        warmup=2,
        iterations=5,
        audio_chunk_ms=10.0,
    )

    assert counter == 7
    assert report.iterations == 5
    assert report.warmup == 2
    assert report.real_time_factor is not None
    assert report.real_time_factor >= 0.0
    assert report.p95_ms >= report.min_ms


def test_benchmark_validates_iterations():
    with pytest.raises(ValueError):
        benchmark_callable(lambda: None, iterations=0)
