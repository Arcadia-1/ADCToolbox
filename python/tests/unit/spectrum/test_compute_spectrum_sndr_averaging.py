"""Unit tests for compute_spectrum averaging (M runs).

Power averaging: SNDR stays constant. Coherent averaging: SNDR improves by 10*log10(M) dB.
"""

import pytest
import numpy as np
from adctoolbox.spectrum.compute_spectrum import compute_spectrum
from adctoolbox.fundamentals.snr_nsd import amplitudes_to_snr


@pytest.mark.parametrize("M", [1, 2, 4, 8, 10, 100])
def test_power_averaging_sndr(M):
    """Test power averaging: SNDR = baseline SNR (no improvement) for M = 2, 4, 8, 10, 100."""
    # Test parameters
    signal_amplitude = 0.5
    noise_rms = 100e-6
    N_fft = 2**13
    Fs = 100e6
    Fin = 123 / N_fft * Fs

    # Generate M runs with same signal but independent noise
    t = np.arange(N_fft) / Fs
    signal_clean = signal_amplitude * np.sin(2 * np.pi * Fin * t)
    signal_runs = np.zeros((M, N_fft))
    for m in range(M):
        signal_runs[m, :] = signal_clean + np.random.randn(N_fft) * noise_rms

    # Power averaging
    result = compute_spectrum(signal_runs, fs=Fs, coherent_averaging=False)

    # Extract computed metrics
    sndr_computed = result['metrics']['sndr_dbc']
    sig_pwr_computed = result['metrics']['sig_pwr_dbfs']

    # Power averaging: SNDR = theoretical SNR (no improvement)
    snr_baseline = amplitudes_to_snr(sig_amplitude=signal_amplitude, noise_amplitude=noise_rms)
    sndr_expected = snr_baseline
    sndr_error = abs(sndr_expected - sndr_computed)

    # Signal power: Expected 0 dBFS (auto-detect uses peak as reference)
    sig_pwr_expected = 0.0
    sig_pwr_error = abs(sig_pwr_expected - sig_pwr_computed)

    # SNDR change from baseline (should be ~0)
    delta_computed = sndr_computed - snr_baseline

    print(f"\n[Power Avg M={M:3d}] A={signal_amplitude:6.2f}, noise_rms={noise_rms*1e6:6.2f}uV")
    print(f"Signal: Expected=[{sig_pwr_expected:6.2f} dBFS], Computed=[{sig_pwr_computed:6.2f} dBFS], Error=[{sig_pwr_error:6.2f} dB]")
    print(f"SNDR  : Expected=[{sndr_expected:6.2f} dBc], Computed=[{sndr_computed:6.2f} dBc], Error=[{sndr_error:6.2f} dB], Expected Delta=[{0:6.2f} dB], Computed Delta=[{delta_computed:6.2f} dB]")

    # Assertions
    assert sig_pwr_computed == pytest.approx(sig_pwr_expected, abs=0.5)
    assert sndr_computed == pytest.approx(sndr_expected, abs=0.5)


@pytest.mark.parametrize("M", [1, 2, 4, 8, 10, 100])
def test_coherent_averaging_sndr(M):
    """Test coherent averaging: SNDR = baseline SNR + 10*log10(M) dB for M = 2, 4, 8, 10, 100."""
    # Test parameters
    signal_amplitude = 0.5
    noise_rms = 100e-6
    N_fft = 2**13
    Fs = 100e6
    Fin = 123 / N_fft * Fs

    # Generate M runs with same signal but independent noise
    t = np.arange(N_fft) / Fs
    signal_clean = signal_amplitude * np.sin(2 * np.pi * Fin * t)
    signal_runs = np.zeros((M, N_fft))
    for m in range(M):
        signal_runs[m, :] = signal_clean + np.random.randn(N_fft) * noise_rms

    # Coherent averaging
    result = compute_spectrum(signal_runs, fs=Fs, coherent_averaging=True)

    # Extract computed metrics
    sndr_computed = result['metrics']['sndr_dbc']
    sig_pwr_computed = result['metrics']['sig_pwr_dbfs']

    # Coherent averaging: SNDR = theoretical SNR + 10*log10(M)
    snr_baseline = amplitudes_to_snr(sig_amplitude=signal_amplitude, noise_amplitude=noise_rms)
    sndr_expected = snr_baseline + 10 * np.log10(M)
    sndr_error = abs(sndr_expected - sndr_computed)

    # Signal power: Expected 0 dBFS (auto-detect uses peak as reference)
    sig_pwr_expected = 0.0
    sig_pwr_error = abs(sig_pwr_expected - sig_pwr_computed)

    # SNDR improvement over baseline
    delta_expected = 10 * np.log10(M)
    delta_computed = sndr_computed - snr_baseline

    print(f"\n[Coherent Avg M={M:3d}] A={signal_amplitude:6.2f}, noise_rms={noise_rms*1e6:6.2f}uV")
    print(f"Signal: Expected=[{sig_pwr_expected:6.2f} dBFS], Computed=[{sig_pwr_computed:6.2f} dBFS], Error=[{sig_pwr_error:6.2f} dB]")
    print(f"SNDR  : Expected=[{sndr_expected:6.2f} dBc], Computed=[{sndr_computed:6.2f} dBc], Error=[{sndr_error:6.2f} dB], Expected Delta=[{delta_expected:6.2f} dB], Computed Delta=[{delta_computed:6.2f} dB]")

    # Assertions
    assert sig_pwr_computed == pytest.approx(sig_pwr_expected, abs=0.5)
    assert sndr_computed == pytest.approx(sndr_expected, abs=0.5)

