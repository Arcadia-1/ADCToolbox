"""Unit tests for compute_spectrum signal power and SNDR metrics.

This test suite validates the core metrics calculation:
- Signal power (sig_pwr_dbfs)
- SNDR (Signal-to-Noise-and-Distortion Ratio)

Test methodology:
- Generate sinewaves with known amplitudes and noise levels
- Calculate theoretical values using helper functions
- Compare computed metrics against ground truth
"""

import pytest
import numpy as np
from adctoolbox.spectrum.compute_spectrum import compute_spectrum
from adctoolbox.fundamentals.snr_nsd import amplitudes_to_snr


@pytest.mark.parametrize("signal_amplitude", [0.1, 1.0])
@pytest.mark.parametrize("noise_rms", [10e-6, 100e-6])
def test_compute_spectrum_sndr(signal_amplitude, noise_rms):
   # Test signal power and SNDR with max_scale_range parameter.

    # Generate test signal
    N_fft = 2**13  # 8192 points
    Fs = 100e6     # 100 MHz sampling rate
    Fin = 123 / N_fft * Fs  # Coherent frequency
    t = np.arange(N_fft) / Fs

    signal = signal_amplitude * np.sin(2 * np.pi * Fin * t) + np.random.randn(N_fft) * noise_rms

    # Define full scale range as internal parameter
    max_scale_range = [-1, 1]

    # Use hann window
    result = compute_spectrum(signal, fs=Fs, win_type='hann', max_scale_range=max_scale_range, verbose=0)

    # Extract computed metrics
    sig_pwr_dbfs_computed = result['metrics']['sig_pwr_dbfs']
    sndr_computed = result['metrics']['sndr_dbc']

    fs_peak = (max_scale_range[1] - max_scale_range[0]) / 2
    sig_pwr_dbfs_expected = 20 * np.log10(signal_amplitude / fs_peak)
    sig_pwr_error = abs(sig_pwr_dbfs_expected - sig_pwr_dbfs_computed)

    # For pure sinewave with Gaussian noise (no distortion), SNDR ≈ SNR
    sndr_expected = amplitudes_to_snr(sig_amplitude=signal_amplitude, noise_amplitude=noise_rms)
    sndr_error = abs(sndr_expected - sndr_computed)

    print(f"\n[Test] A={signal_amplitude:6.2f}, noise_rms={noise_rms*1e6:6.2f}uV")
    print(f"Signal: Expected=[{sig_pwr_dbfs_expected:6.2f} dBFS], Computed=[{sig_pwr_dbfs_computed:6.2f} dBFS], Error=[{sig_pwr_error:6.2f} dB]")
    print(f"SNDR  : Expected=[{sndr_expected:6.2f} dBc ], Computed=[{sndr_computed:6.2f} dBc ], Error=[{sndr_error:6.2f} dB]")

    # Hann window applies power correction, allow 2.0 dB tolerance for signal power
    assert sig_pwr_dbfs_computed == pytest.approx(sig_pwr_dbfs_expected, abs=0.5)
    assert sndr_computed == pytest.approx(sndr_expected, abs=0.5)

@pytest.mark.parametrize("signal_amplitude", [0.1, 1.0])
@pytest.mark.parametrize("noise_rms", [10e-6, 100e-6])
def test_compute_spectrum_sndr_no_max_scale_range(signal_amplitude, noise_rms):
    # Test signal power and SNDR without max_scale_range parameter.
    
    N_fft = 2**13  # 8192 points
    Fs = 100e6     # 100 MHz sampling rate
    Fin = 123 / N_fft * Fs  # Coherent frequency
    t = np.arange(N_fft) / Fs

    signal = signal_amplitude * np.sin(2 * np.pi * Fin * t) + np.random.randn(N_fft) * noise_rms

    # Use hann window
    result = compute_spectrum(signal, fs=Fs, win_type='hann', verbose=0)

    # Extract computed metrics
    sig_pwr_dbfs_computed = result['metrics']['sig_pwr_dbfs']
    sndr_computed = result['metrics']['sndr_dbc']

    sig_pwr_dbfs_expected = 0
    sig_pwr_error = abs(sig_pwr_dbfs_expected - sig_pwr_dbfs_computed)

    # For pure sinewave with Gaussian noise (no distortion), SNDR ≈ SNR
    sndr_expected = amplitudes_to_snr(sig_amplitude=signal_amplitude, noise_amplitude=noise_rms)
    sndr_error = abs(sndr_expected - sndr_computed)

    print(f"\n[Test] A={signal_amplitude:6.2f}, noise_rms={noise_rms*1e6:6.2f}uV")
    print(f"Signal: Expected=[{sig_pwr_dbfs_expected:6.2f} dBFS], Computed=[{sig_pwr_dbfs_computed:6.2f} dBFS], Error=[{sig_pwr_error:6.2f} dB]")
    print(f"SNDR  : Expected=[{sndr_expected:6.2f} dBc ], Computed=[{sndr_computed:6.2f} dBc ], Error=[{sndr_error:6.2f} dB]")

    # Hann window applies power correction, allow 2.0 dB tolerance for signal power
    assert sig_pwr_dbfs_computed == pytest.approx(sig_pwr_dbfs_expected, abs=0.5)
    assert sndr_computed == pytest.approx(sndr_expected, abs=0.5)



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
def test_compute_spectrum_window_sweep(win_type, side_bin):
    """Test SNDR with different window types.

    This test sweeps window types with fixed signal amplitude and noise level
    to compare windowing effects on SNDR performance.
    Window configurations match exp_s08_windowing_deep_dive.py
    """
    # Fixed test parameters
    signal_amplitude = 0.5
    noise_rms = 10e-6
    N_fft = 2**13  # 8192 points
    Fs = 100e6     # 100 MHz sampling rate
    Fin = 123 / N_fft * Fs  # Coherent frequency
    t = np.arange(N_fft) / Fs

    signal = signal_amplitude * np.sin(2 * np.pi * Fin * t) + np.random.randn(N_fft) * noise_rms

    # Define full scale range as internal parameter
    max_scale_range = [-2, 2]

    print(f"\n[Window Test] win_type={win_type}, side_bin={side_bin}, A={signal_amplitude:.2f}, noise_rms={noise_rms*1e6:.2f}uV")

    # Compute spectrum with the specified window type and side bins
    result = compute_spectrum(signal, fs=Fs, win_type=win_type, side_bin=side_bin, max_scale_range=max_scale_range, verbose=0)

    # Extract computed metrics
    sig_pwr_dbfs_computed = result['metrics']['sig_pwr_dbfs']
    sndr_computed = result['metrics']['sndr_dbc']

    # Calculate expected values
    fs_peak = (max_scale_range[1] - max_scale_range[0]) / 2
    sig_pwr_dbfs_expected = 20 * np.log10(signal_amplitude / fs_peak)
    sig_pwr_error = abs(sig_pwr_dbfs_expected - sig_pwr_dbfs_computed)

    # For pure sinewave with Gaussian noise (no distortion), SNDR ≈ SNR
    sndr_expected = amplitudes_to_snr(sig_amplitude=signal_amplitude, noise_amplitude=noise_rms)
    sndr_error = abs(sndr_expected - sndr_computed)

    print(f"Signal: Expected=[{sig_pwr_dbfs_expected:6.2f} dBFS], Computed=[{sig_pwr_dbfs_computed:6.2f} dBFS], Error=[{sig_pwr_error:6.2f} dB]")
    print(f"SNDR  : Expected=[{sndr_expected:6.2f} dBc ], Computed=[{sndr_computed:6.2f} dBc ], Error=[{sndr_error:6.2f} dB]")

    assert sig_pwr_dbfs_computed == pytest.approx(sig_pwr_dbfs_expected, abs=0.5)
    assert sndr_computed == pytest.approx(sndr_expected, abs=0.5)


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
def test_compute_spectrum_window_sweep_no_max_scale_range(win_type, side_bin):
    """Test SNDR with different window types.

    This test sweeps window types with fixed signal amplitude and noise level
    to compare windowing effects on SNDR performance.
    Window configurations match exp_s08_windowing_deep_dive.py
    """
    # Fixed test parameters
    signal_amplitude = 0.5
    noise_rms = 10e-6
    N_fft = 2**13  # 8192 points
    Fs = 100e6     # 100 MHz sampling rate
    Fin = 123 / N_fft * Fs  # Coherent frequency
    t = np.arange(N_fft) / Fs

    signal = signal_amplitude * np.sin(2 * np.pi * Fin * t) + np.random.randn(N_fft) * noise_rms

    print(f"\n[Window Test] win_type={win_type}, side_bin={side_bin}, A={signal_amplitude:.2f}, noise_rms={noise_rms*1e6:.2f}uV")

    # Compute spectrum with the specified window type and side bins
    result = compute_spectrum(signal, fs=Fs, win_type=win_type, side_bin=side_bin, verbose=0)

    # Extract computed metrics
    sig_pwr_dbfs_computed = result['metrics']['sig_pwr_dbfs']
    sndr_computed = result['metrics']['sndr_dbc']

    # Calculate expected values
    sig_pwr_dbfs_expected = 0
    sig_pwr_error = abs(sig_pwr_dbfs_expected - sig_pwr_dbfs_computed)

    # For pure sinewave with Gaussian noise (no distortion), SNDR ≈ SNR
    sndr_expected = amplitudes_to_snr(sig_amplitude=signal_amplitude, noise_amplitude=noise_rms)
    sndr_error = abs(sndr_expected - sndr_computed)

    print(f"Signal: Expected=[{sig_pwr_dbfs_expected:6.2f} dBFS], Computed=[{sig_pwr_dbfs_computed:6.2f} dBFS], Error=[{sig_pwr_error:6.2f} dB]")
    print(f"SNDR  : Expected=[{sndr_expected:6.2f} dBc ], Computed=[{sndr_computed:6.2f} dBc ], Error=[{sndr_error:6.2f} dB]")

    assert sig_pwr_dbfs_computed == pytest.approx(sig_pwr_dbfs_expected, abs=0.5)
    assert sndr_computed == pytest.approx(sndr_expected, abs=0.5)

