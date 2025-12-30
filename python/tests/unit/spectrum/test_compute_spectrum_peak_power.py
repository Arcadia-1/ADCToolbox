"""Unit tests for power_spectrum_db at fundamental bin (bin_int_n).

Sweep signal amplitude and verify peak power matches expected dBFS.
"""

import pytest
import numpy as np
from adctoolbox.spectrum.compute_spectrum import compute_spectrum


@pytest.mark.parametrize("win_type,side_bin", [
    ('boxcar', 0),
    ('hann', 1),
    ('hamming', 1),
    ('blackman', 2),
    ('blackmanharris', 3),
    ('flattop', 4),
    ('kaiser', 8),
    ('chebwin', 50),
])
def test_peak_power_with_windows(win_type, side_bin):
    """Test power_spectrum_db[bin_int_n] with different window types."""
    # Fixed parameters
    signal_amplitude = 0.5
    N_fft = 2**13
    Fs = 100e6
    Fin = 123 / N_fft * Fs  # Coherent frequency
    t = np.arange(N_fft) / Fs

    # Generate pure tone
    signal = signal_amplitude * np.sin(2 * np.pi * Fin * t) + np.random.randn(N_fft) * 1e-6

    # Compute spectrum
    result = compute_spectrum(signal, fs=Fs, win_type=win_type, side_bin=side_bin)

    # Extract peak bin power
    bin_int_n = result['plot_data']['fundamental_bin']
    spectrum_peak_after = result['plot_data']['power_spectrum_db_plot'][bin_int_n]

    # Expected peak power in dBFS
    expected_peak_db = 0 
    error_db_after = abs(expected_peak_db - spectrum_peak_after)

    print(f"\n[Window test with max_scale_range] win={win_type}, side_bin={side_bin}, A={signal_amplitude:4.1f}")
    print(f"After compensation : Expected=[{expected_peak_db:7.2f} dBFS], Computed=[{spectrum_peak_after:7.2f} dBFS], Error=[{error_db_after:5.2f} dB]")
    assert spectrum_peak_after == pytest.approx(expected_peak_db, abs=0.5)


@pytest.mark.parametrize("win_type,side_bin", [
    ('boxcar', 0),
    ('hann', 1),
    ('hamming', 1),
    ('blackman', 2),
    ('blackmanharris', 3),
    ('flattop', 4),
    ('kaiser', 8),
    ('chebwin', 50),
])
def test_peak_power_with_windows_max_scale_range(win_type, side_bin):
    """Test power_spectrum_db[bin_int_n] with different window types."""
    # Fixed parameters
    signal_amplitude = 0.5
    N_fft = 2**13
    Fs = 100e6
    Fin = 123 / N_fft * Fs  # Coherent frequency
    t = np.arange(N_fft) / Fs

    # Generate pure tone
    signal = signal_amplitude * np.sin(2 * np.pi * Fin * t) + np.random.randn(N_fft) * 1e-6

    # Compute spectrum
    max_scale_range = [-1, 1]

    result = compute_spectrum(signal, fs=Fs, win_type=win_type, side_bin=side_bin, max_scale_range=max_scale_range)

    # Extract peak bin power
    bin_int_n = result['plot_data']['fundamental_bin']
    spectrum_peak_after = result['plot_data']['power_spectrum_db_plot'][bin_int_n]

    # Expected peak power in dBFS
    fs_peak = (max_scale_range[1] - max_scale_range[0]) / 2
    expected_peak_db = 20 * np.log10(signal_amplitude / fs_peak)
    error_db_after = abs(expected_peak_db - spectrum_peak_after)

    print(f"\n[Window test with max_scale_range] win={win_type}, side_bin={side_bin}, A={signal_amplitude:4.1f}")
    print(f"After compensation : Expected=[{expected_peak_db:7.2f} dBFS], Computed=[{spectrum_peak_after:7.2f} dBFS], Error=[{error_db_after:5.2f} dB]")
    assert spectrum_peak_after == pytest.approx(expected_peak_db, abs=0.5)