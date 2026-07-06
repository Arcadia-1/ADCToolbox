"""Tests for coherent-frequency bin selection policies."""

import numpy as np
import pytest

from adctoolbox import find_coherent_frequency


@pytest.mark.parametrize(
    ("n_fft", "target_bin", "expected_bin"),
    [
        (1024, 256.0, 257),
        (999, 250.0, 250),
        (8192, 1024.0, 1025),
        (1000, 250.0, 251),
        (999, 250.6, 250),
    ],
)
def test_matlab_findbin_policy_matches_floor_upper_first_search(
    n_fft,
    target_bin,
    expected_bin,
):
    fs = 1.0
    fin_target = target_bin / n_fft

    fin_actual, bin_idx = find_coherent_frequency(
        fs,
        fin_target,
        n_fft,
        policy="matlab_findbin",
    )

    assert bin_idx == expected_bin
    np.testing.assert_allclose(fin_actual, expected_bin / n_fft)


def test_default_policy_preserves_odd_bin_selection():
    _, bin_idx = find_coherent_frequency(1.0, 256.0 / 1024.0, 1024)

    assert bin_idx == 255


def test_find_coherent_frequency_rejects_unknown_policy():
    with pytest.raises(ValueError, match="Unknown coherent-frequency policy"):
        find_coherent_frequency(1.0, 0.25, 1024, policy="unknown")
