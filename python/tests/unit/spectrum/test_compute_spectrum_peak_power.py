"""Unit tests for spectrum bin heights and integrated signal power.

Sweep windows and verify plot-bin heights and main-lobe power match dBFS
expectations.
"""

import pytest
import numpy as np
from adctoolbox.spectrum.compute_spectrum import compute_spectrum
from adctoolbox.spectrum._window import _create_window


def test_hann_full_scale_main_lobe_matches_plotspec_height():
    """Hann center bin stays below 0 dBFS; the full main lobe sums to 0 dBFS."""
    n_fft = 2**13
    fs = 100e6
    fundamental_bin = 123
    n = np.arange(n_fft)
    signal = np.sin(2 * np.pi * fundamental_bin * n / n_fft)

    result = compute_spectrum(
        signal,
        fs=fs,
        win_type='hann',
        side_bin=1,
        max_scale_range=[-1, 1],
    )

    plot_data = result['plot_data']
    spec_db = plot_data['power_spectrum_db_plot']
    bin_idx = plot_data['fundamental_bin']
    main_lobe_linear = np.sum(10 ** (spec_db[bin_idx - 1:bin_idx + 2] / 10))

    assert spec_db[bin_idx] == pytest.approx(-10 * np.log10(1.5), abs=1e-9)
    assert spec_db[bin_idx - 1] == pytest.approx(-10 * np.log10(6.0), abs=1e-9)
    assert spec_db[bin_idx + 1] == pytest.approx(-10 * np.log10(6.0), abs=1e-9)
    assert 10 * np.log10(main_lobe_linear) == pytest.approx(0.0, abs=1e-9)
    assert result['metrics']['sig_pwr_dbfs'] == pytest.approx(0.0, abs=1e-9)


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
    """Test plot-bin and integrated signal power with different window types."""
    # Fixed parameters
    signal_amplitude = 0.5
    N_fft = 2**13
    Fs = 100e6
    Fin = 123 / N_fft * Fs  # Coherent frequency
    t = np.arange(N_fft) / Fs
    rng = np.random.default_rng(2026062232)

    # Generate pure tone
    signal = signal_amplitude * np.sin(2 * np.pi * Fin * t) + rng.standard_normal(N_fft) * 1e-6

    # Compute spectrum
    result = compute_spectrum(signal, fs=Fs, win_type=win_type, side_bin=side_bin)

    # Extract peak bin power
    bin_int_n = result['plot_data']['fundamental_bin']
    spectrum_peak_after = result['plot_data']['power_spectrum_db_plot'][bin_int_n]

    _, _, enbw = _create_window(win_type, N_fft)
    expected_signal_db = 0
    expected_peak_db = expected_signal_db - 10 * np.log10(enbw)
    error_db_after = abs(expected_peak_db - spectrum_peak_after)
    sig_start = result['plot_data']['sig_bin_start']
    sig_end = result['plot_data']['sig_bin_end']
    signal_lobe_db = 10 * np.log10(
        np.sum(10 ** (result['plot_data']['power_spectrum_db_plot'][sig_start:sig_end] / 10))
    )

    print(f"\n[Window test] win={win_type}, side_bin={side_bin}, A={signal_amplitude:4.1f}")
    print(f"After compensation : Expected=[{expected_peak_db:7.2f} dBFS], Computed=[{spectrum_peak_after:7.2f} dBFS], Error=[{error_db_after:5.2f} dB]")
    assert spectrum_peak_after == pytest.approx(expected_peak_db, abs=0.5)
    assert signal_lobe_db == pytest.approx(expected_signal_db, abs=0.5)


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
    """Test plot-bin and integrated signal power with different window types."""
    # Fixed parameters
    signal_amplitude = 0.5
    N_fft = 2**13
    Fs = 100e6
    Fin = 123 / N_fft * Fs  # Coherent frequency
    t = np.arange(N_fft) / Fs
    rng = np.random.default_rng(2026062233)

    # Generate pure tone
    signal = signal_amplitude * np.sin(2 * np.pi * Fin * t) + rng.standard_normal(N_fft) * 1e-6

    # Compute spectrum
    max_scale_range = [-1, 1]

    result = compute_spectrum(signal, fs=Fs, win_type=win_type, side_bin=side_bin, max_scale_range=max_scale_range)

    # Extract peak bin power
    bin_int_n = result['plot_data']['fundamental_bin']
    spectrum_peak_after = result['plot_data']['power_spectrum_db_plot'][bin_int_n]

    fs_peak = (max_scale_range[1] - max_scale_range[0]) / 2
    expected_signal_db = 20 * np.log10(signal_amplitude / fs_peak)
    _, _, enbw = _create_window(win_type, N_fft)
    expected_peak_db = expected_signal_db - 10 * np.log10(enbw)
    error_db_after = abs(expected_peak_db - spectrum_peak_after)
    sig_start = result['plot_data']['sig_bin_start']
    sig_end = result['plot_data']['sig_bin_end']
    signal_lobe_db = 10 * np.log10(
        np.sum(10 ** (result['plot_data']['power_spectrum_db_plot'][sig_start:sig_end] / 10))
    )

    print(f"\n[Window test with max_scale_range] win={win_type}, side_bin={side_bin}, A={signal_amplitude:4.1f}")
    print(f"After compensation : Expected=[{expected_peak_db:7.2f} dBFS], Computed=[{spectrum_peak_after:7.2f} dBFS], Error=[{error_db_after:5.2f} dB]")
    assert spectrum_peak_after == pytest.approx(expected_peak_db, abs=0.5)
    assert signal_lobe_db == pytest.approx(expected_signal_db, abs=0.5)
