"""Unit tests for assumed_sig_pwr_dbfs parameter.

Test how overriding signal power assumption affects metrics.
"""

import pytest
import numpy as np
from adctoolbox.spectrum.compute_spectrum import compute_spectrum
from adctoolbox.fundamentals.snr_nsd import amplitudes_to_snr


@pytest.mark.parametrize("assumed_sig_pwr_dbfs", [0, -1, -3, -6, -10])
def test_assumed_signal_power(assumed_sig_pwr_dbfs):
    """Test assumed_sig_pwr_dbfs parameter override."""
    # Fixed parameters
    signal_amplitude = 0.5
    noise_rms = 100e-6
    N_fft = 2**13
    Fs = 100e6
    Fin = 123 / N_fft * Fs  # Coherent frequency
    t = np.arange(N_fft) / Fs

    # Generate pure tone
    signal = signal_amplitude * np.sin(2 * np.pi * Fin * t) + np.random.randn(N_fft) * noise_rms

    # Compute spectrum WITH assumed signal power
    result = compute_spectrum(signal, fs=Fs, win_type='hann', side_bin=1, assumed_sig_pwr_dbfs=assumed_sig_pwr_dbfs)

    # Extract computed metrics
    sig_pwr_computed = result['metrics']['sig_pwr_dbfs']
    sndr_computed = result['metrics']['sndr_dbc']
    bin_int_n = result['plot_data']['fundamental_bin']
    spectrum_peak_after = result['plot_data']['power_spectrum_db_plot'][bin_int_n]

    # Calculate theoretical SNR (based on actual signal and noise amplitudes)
    snr_theoretical = amplitudes_to_snr(sig_amplitude=signal_amplitude, noise_amplitude=noise_rms)

    # Expected SNDR when assuming different signal power:
    # Since actual signal ≈ 0 dBFS and noise is fixed:
    # expected_sndr = snr_theoretical + assumed_sig_pwr_dbfs
    # (because assumed_sig_pwr_dbfs is relative to 0 dBFS baseline)
    sndr_expected = snr_theoretical + assumed_sig_pwr_dbfs
    sndr_error = abs(sndr_expected - sndr_computed)

    print(f"\n[Assumed Power Test] assumed_sig_pwr_dbfs={assumed_sig_pwr_dbfs:6.2f} dBFS")    
    print(f"SNDR        : Expected=[{sndr_expected:7.2f} dBc ], Computed=[{sndr_computed:7.2f} dBc ], Error=[{sndr_error:5.2f} dB]")
    print(f"Signal Power: Expected=[{assumed_sig_pwr_dbfs:7.2f} dBFS], Computed=[{sig_pwr_computed:7.2f} dBFS], plot_peak=[{spectrum_peak_after:7.2f} dBFS]")

    assert sig_pwr_computed == pytest.approx(assumed_sig_pwr_dbfs, abs=0.01)
    assert sndr_computed == pytest.approx(sndr_expected, abs=0.5)
    assert spectrum_peak_after == pytest.approx(assumed_sig_pwr_dbfs, abs=0.5)
