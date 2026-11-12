"""Unit tests for overflow residue analysis."""

import numpy as np
import pytest

from adctoolbox.dout import analyze_overflow


def test_analyze_overflow_matches_suffix_weighted_residue_formula():
    bits = np.array(
        [
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 0],
        ],
        dtype=float,
    )
    weights = np.array([4.0, 2.0, 1.0])

    range_min, range_max, ovf_zero, ovf_one = analyze_overflow(
        bits,
        weights,
        create_plot=False,
    )

    np.testing.assert_allclose(range_min, [1.0 / 7.0, 0.0, 0.0])
    np.testing.assert_allclose(range_max, [6.0 / 7.0, 2.0 / 3.0, 1.0])
    np.testing.assert_allclose(ovf_zero, [0.0, 25.0, 75.0])
    np.testing.assert_allclose(ovf_one, [0.0, 0.0, 25.0])


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_all_ones_codes_count_as_overflow_for_any_float_weights(seed):
    # range == 1 exactly in real arithmetic. The dot product and np.sum add
    # the same weights in different orders, so an exact `>= 1` missed these
    # codes at some positions (0 % or 71.4 % instead of 100 %).
    weights = np.random.default_rng(seed).uniform(0.1, 1.0, 10)
    codes = np.ones((7, 10), dtype=np.int8)

    range_min, range_max, ovf_zero, ovf_one = analyze_overflow(codes, weights, create_plot=False)

    np.testing.assert_array_equal(ovf_one, np.full(10, 100.0))
    np.testing.assert_array_equal(ovf_zero, np.zeros(10))
    np.testing.assert_allclose(range_max, 1.0, rtol=0, atol=1e-12)


def test_overflow_percentages_match_bit_patterns_for_calibrated_weights():
    # Sub-radix-2 capture that clips at the top, analysed with weights
    # perturbed by 0.1 % as a calibration would return them.
    rng = np.random.default_rng(1)
    n_samples, n_bits = 4096, 12
    weights = 1.8 ** np.arange(n_bits - 1, -1, -1)
    weights = weights / (weights.sum() + weights[-1])
    codes = (rng.random((n_samples, n_bits)) < 0.5).astype(int)
    codes[: n_samples // 50] = 1   # clipped-high samples
    codes[-(n_samples // 100):] = 0  # clipped-low samples
    calibrated = weights * (1 + 1e-3 * rng.standard_normal(n_bits))

    _, _, ovf_zero, ovf_one = analyze_overflow(codes, calibrated, create_plot=False)

    expected_one = [np.mean(np.all(codes[:, i:] == 1, axis=1)) * 100 for i in range(n_bits)]
    expected_zero = [np.mean(np.all(codes[:, i:] == 0, axis=1)) * 100 for i in range(n_bits)]
    np.testing.assert_allclose(ovf_one, expected_one, rtol=0, atol=1e-12)
    np.testing.assert_allclose(ovf_zero, expected_zero, rtol=0, atol=1e-12)
