"""Value-range inference and the frequency reported by the residual analyses."""
import numpy as np
import pytest

from adctoolbox.aout._infer_signal_range import _infer_signal_range
from adctoolbox.aout.rearrange_error_by_phase import rearrange_error_by_phase
from adctoolbox.aout.rearrange_error_by_value import rearrange_error_by_value
from adctoolbox.fundamentals.validate import validate_dout_data

N = 2**14
CARRIER = np.sin(2 * np.pi * 1637 * np.arange(N) / N + 0.2)


@pytest.mark.parametrize(
    "signal, expected",
    [
        (0.5 + 0.45 * CARRIER, (0.0, 1.0)),      # normalized unipolar: keep [0, 1]
        (0.45 * CARRIER, (-0.5, 0.5)),           # normalized bipolar
        (0.9 * CARRIER, (-1.0, 1.0)),
    ],
)
def test_normalized_signals_keep_the_standard_range(signal, expected):
    assert _infer_signal_range(signal) == expected


@pytest.mark.parametrize(
    "signal",
    [
        1.2 + 0.7 * CARRIER,     # was binned over [0, 1]: 40 % of the samples lost
        0.9 + 0.3 * CARRIER,     # 1.8 V supply, mid-rail
        1.5 * CARRIER,           # was binned over [-1, 1]
        0.005 * CARRIER,         # was binned over [-0.5, 0.5]: 2 of 100 bins used
        0.05 + 0.04 * CARRIER,
        2000 + 1900 * CARRIER,   # ADC codes
    ],
)
def test_other_signals_are_binned_over_their_own_span(signal):
    low, high = _infer_signal_range(signal)

    assert low == pytest.approx(signal.min())
    assert high == pytest.approx(signal.max())


def test_explicit_range_is_returned_unchanged():
    assert _infer_signal_range(1.2 + 0.7 * CARRIER, (0.0, 2.0)) == (0.0, 2.0)


@pytest.mark.parametrize("signal", [1.2 + 0.7 * CARRIER, 0.005 * CARRIER])
def test_value_binning_uses_every_sample_and_bin(signal):
    result = rearrange_error_by_value(signal, norm_freq=1637 / N, clip_percent=0)

    assert np.sum(result["count_per_bin"]) == N
    assert np.count_nonzero(result["count_per_bin"]) == result["n_bins"]
    assert result["value_bin_centers"][0] == pytest.approx(signal.min())
    assert result["value_bin_centers"][-1] == pytest.approx(signal.max())


@pytest.mark.parametrize("analysis", [rearrange_error_by_phase, rearrange_error_by_value])
def test_reported_frequency_is_the_fitted_one(analysis):
    # The tone is 0.05 bin above the caller's estimate. The phase axis and the
    # residual use the fitted frequency, so the result must report that one.
    true_bin = 1001.05
    signal = 0.4 * np.sin(2 * np.pi * true_bin * np.arange(N) / N) + 0.5

    result = analysis(signal, norm_freq=1001 / N)

    assert result["norm_freq"] * N == pytest.approx(true_bin, abs=1e-6)


def test_fixed_frequency_is_reported_unchanged():
    signal = 0.4 * np.sin(2 * np.pi * 1001 * np.arange(N) / N) + 0.5

    result = rearrange_error_by_phase(signal, norm_freq=1001 / N, max_iterations=0)

    assert result["norm_freq"] == 1001 / N


def test_boolean_bit_matrix_is_valid_digital_data():
    bits = np.random.default_rng(0).integers(0, 2, (256, 8)).astype(bool)

    validate_dout_data(bits)  # must not raise


def test_non_binary_digital_data_is_still_rejected():
    bits = np.random.default_rng(0).integers(0, 3, (256, 8))

    with pytest.raises(ValueError, match="only 0 and 1"):
        validate_dout_data(bits)
