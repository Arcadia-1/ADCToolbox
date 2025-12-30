"""Unit tests for _create_window helper function."""

import pytest
import numpy as np
from adctoolbox.spectrum._window import _create_window, _calculate_power_correction

@pytest.mark.parametrize("win_type,expected_window_gain,expected_equiv_noise_bw_factor", [
    ("rectangular", 1.000000, 1.000000),
    ("hann", 0.500000, 1.500000),
    ("hamming", 0.540000, 1.362826),
    ("blackman", 0.420000, 1.726757),
    ("blackmanharris", 0.358750, 2.004353),
    ("flattop", 0.215579, 3.770246),
    ("kaiser", 0.202638, 3.507128),
    ("chebwin", 0.370436, 1.940414),
])
def test_create_window(win_type, expected_window_gain, expected_equiv_noise_bw_factor):
    """Test window creation and parameters."""
    N = 10240
    window_vector, window_gain, equiv_noise_bw_factor = _create_window(win_type, N)

    print(f"\n[Window: {win_type:15s}] window_gain={window_gain:.4f}, equiv_noise_bw_factor={equiv_noise_bw_factor:.4f}")

    assert window_vector.shape == (N,)
    assert window_gain > 0
    assert equiv_noise_bw_factor > 0
    assert np.all(np.isfinite(window_vector))

    assert window_gain == pytest.approx(expected_window_gain, abs=1e-3)
    assert equiv_noise_bw_factor == pytest.approx(expected_equiv_noise_bw_factor, abs=1e-3)
