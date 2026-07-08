"""Unit tests for overflow residue analysis."""

import numpy as np

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
