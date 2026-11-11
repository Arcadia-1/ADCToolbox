"""Per-record frequencies with 0 placeholders in calibrate_weight_sine.

A frequency array mixing known values and 0 placeholders skipped the coarse
estimate (it only ran when every entry was 0), so the fine search started at
0 Hz for the placeholder records. The result was all-zero weights reported
with a residual SNR of 200 dB.
"""
import numpy as np
import pytest

from adctoolbox.calibration import calibrate_weight_sine
from adctoolbox.calibration._estimate_frequencies import _interpolate_hann_peak
from adctoolbox.models.sar import sar_convert, sar_ideal_weights

N, BITS = 2048, 12
FREQS = [255 / N, 331 / N, 83 / N]
PHASES = [0.2, 1.9, -2.5]
OFFSETS = [0.5, 0.47, 0.52]
AMPLITUDES = [0.45, 0.40, 0.47]


@pytest.fixture(scope="module")
def records():
    weights = sar_ideal_weights(BITS)
    t = np.arange(N)
    captures = [
        sar_convert(dc + amp * np.sin(2 * np.pi * freq * t + phase), weights)
        for freq, phase, dc, amp in zip(FREQS, PHASES, OFFSETS, AMPLITUDES)
    ]
    return captures, weights / AMPLITUDES[0]


@pytest.mark.parametrize(
    "freq",
    [
        [0, FREQS[1], FREQS[2]],
        [FREQS[0], 0, FREQS[2]],
        [FREQS[0], FREQS[1], 0],
        [0, 0, FREQS[2]],
    ],
)
def test_placeholder_frequencies_are_estimated(records, freq):
    captures, truth = records

    result = calibrate_weight_sine(captures, freq=freq)

    np.testing.assert_allclose(result["refined_frequency"], FREQS, atol=1e-3 / N)
    assert np.max(np.abs(result["weight"] - truth)) / truth[-1] < 0.1
    assert np.all(np.asarray(result["snr_db"]) > 60)
    assert np.all(np.asarray(result["snr_db"]) < 100)


def test_placeholders_match_fully_specified_frequencies(records):
    captures, truth = records

    known = calibrate_weight_sine(captures, freq=FREQS)
    mixed = calibrate_weight_sine(captures, freq=[0, FREQS[1], FREQS[2]])

    # The placeholder run also fine-searches the frequencies, so the two
    # solutions agree to a small fraction of an LSB, not to rounding.
    np.testing.assert_allclose(mixed["weight"], known["weight"], rtol=0, atol=0.01 * truth[-1])


def test_force_search_false_keeps_known_frequencies_and_estimates_the_rest(records):
    captures, truth = records

    result = calibrate_weight_sine(captures, freq=[0, FREQS[1], FREQS[2]], force_search=False)

    assert result["refined_frequency"][1] == FREQS[1]
    assert result["refined_frequency"][2] == FREQS[2]
    assert result["refined_frequency"][0] * N == pytest.approx(FREQS[0] * N, abs=0.05)
    assert np.sum(result["weight"]) == pytest.approx(np.sum(truth), rel=1e-3)


def test_zero_dimensional_frequency_array_is_a_scalar(records):
    captures, _ = records

    as_float = calibrate_weight_sine(captures[0], freq=FREQS[0])
    as_array = calibrate_weight_sine(captures[0], freq=np.array(FREQS[0]))

    np.testing.assert_array_equal(as_array["weight"], as_float["weight"])


@pytest.mark.parametrize("true_bin", [100.0, 100.2, 100.5, 100.8, 100.93])
def test_hann_peak_interpolation_from_the_lower_neighbour(true_bin):
    # The in-band search can be forced onto the bin below the tone when the
    # nearest bin lies outside the band. Interpolating from bin 100 must still
    # land on the tone, up to almost a full bin away.
    n = 4096
    signal = np.cos(2 * np.pi * true_bin / n * np.arange(n) + 0.4)
    magnitude = np.abs(np.fft.rfft(signal * np.hanning(n)))

    assert _interpolate_hann_peak(magnitude, 100) == pytest.approx(true_bin, abs=0.02)


def test_hann_peak_interpolation_at_the_spectrum_edges():
    magnitude = np.array([3.0, 1.0, 0.5, 2.0])

    assert 0.0 <= _interpolate_hann_peak(magnitude, 0) <= 1.0
    assert 2.0 <= _interpolate_hann_peak(magnitude, 3) <= 3.0
    assert _interpolate_hann_peak(np.zeros(4), 2) == 2.0
