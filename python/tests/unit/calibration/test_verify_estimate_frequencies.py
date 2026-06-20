"""Regression tests for calibration frequency initialization."""

import numpy as np
import pytest

from adctoolbox.calibration._estimate_frequencies import _estimate_frequencies


def _quantized_sine_bits(freqs, n_samples, bit_width=12, shuffled_indices=None, rng=None, noise=0.0):
    """Build stacked bit segments that mimic ADC output records."""
    if shuffled_indices is None:
        shuffled_indices = np.arange(bit_width)

    t = np.arange(n_samples)
    bits_by_frequency = []

    for freq in freqs:
        signal = 0.5 * np.sin(2 * np.pi * freq * t) + 0.5
        if noise:
            signal = signal + noise * rng.standard_normal(n_samples)

        quantized = np.floor(signal * (2**bit_width)).astype(int)
        quantized = np.clip(quantized, 0, 2**bit_width - 1)
        bits = (quantized[:, None] >> np.arange(bit_width)) & 1
        bits_by_frequency.append(bits[:, shuffled_indices])

    return np.vstack(bits_by_frequency)


def test_scalar_frequency_init_is_broadcast_to_each_segment():
    bits = np.zeros((12, 3), dtype=int)
    segment_lengths = np.array([4, 4, 4])

    freq_array = _estimate_frequencies(bits, segment_lengths, freq_init=0.125)

    np.testing.assert_allclose(freq_array, np.array([0.125, 0.125, 0.125]))


def test_array_frequency_init_is_validated_and_returned():
    bits = np.zeros((12, 3), dtype=int)
    segment_lengths = np.array([4, 4, 4])
    freq_init = np.array([0.125, 0.25, 0.375])

    freq_array = _estimate_frequencies(bits, segment_lengths, freq_init=freq_init)

    np.testing.assert_allclose(freq_array, freq_init)


def test_frequency_init_array_must_match_segment_count():
    bits = np.zeros((12, 3), dtype=int)
    segment_lengths = np.array([4, 4, 4])

    with pytest.raises(ValueError, match="must match number of datasets"):
        _estimate_frequencies(bits, segment_lengths, freq_init=np.array([0.125, 0.25]))


@pytest.mark.parametrize("n_samples", [128, 1024, 4096])
def test_automatic_frequency_estimation_with_shuffled_bits(n_samples):
    rng = np.random.default_rng(12345)
    bit_width = 12
    shuffled_indices = rng.permutation(bit_width)
    freqs = np.array([
        1 / n_samples,
        2 / n_samples,
        0.15,
        0.21,
        0.25,
        0.29,
        0.30,
        0.40,
        (0.5 * n_samples - 2) / n_samples,
        (0.5 * n_samples - 1) / n_samples,
    ])
    bits = _quantized_sine_bits(freqs, n_samples, bit_width, shuffled_indices)

    freq_array = _estimate_frequencies(bits, np.full(len(freqs), n_samples), verbose=0)

    assert freq_array.shape == freqs.shape
    assert np.all(np.isfinite(freq_array))
    np.testing.assert_allclose(freq_array, freqs, atol=1 / n_samples, rtol=0)


@pytest.mark.parametrize("n_samples", [128, 1024, 4096])
def test_automatic_frequency_estimation_with_noisy_shuffled_bits(n_samples):
    rng = np.random.default_rng(67890)
    bit_width = 12
    shuffled_indices = rng.permutation(bit_width)
    freqs = np.array([
        1 / n_samples,
        2 / n_samples,
        0.15,
        0.21,
        0.25,
        0.29,
        0.30,
        0.35,
        0.40,
        (0.5 * n_samples - 2) / n_samples,
        (0.5 * n_samples - 1) / n_samples,
    ])
    bits = _quantized_sine_bits(
        freqs,
        n_samples,
        bit_width,
        shuffled_indices,
        rng=rng,
        noise=0.0005,
    )

    freq_array = _estimate_frequencies(bits, np.full(len(freqs), n_samples), verbose=0)

    assert freq_array.shape == freqs.shape
    assert np.all(np.isfinite(freq_array))
    np.testing.assert_allclose(freq_array, freqs, atol=1 / n_samples, rtol=0)


@pytest.mark.xfail(strict=True, reason="Nyquist estimation is a documented limitation.")
def test_automatic_frequency_estimation_documents_nyquist_limit():
    n_samples = 1024
    freqs = np.array([0.5])
    bits = _quantized_sine_bits(freqs, n_samples)

    freq_array = _estimate_frequencies(bits, np.array([n_samples]), verbose=0)

    np.testing.assert_allclose(freq_array, freqs, atol=1 / n_samples, rtol=0)
