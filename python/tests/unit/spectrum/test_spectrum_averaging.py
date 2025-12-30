"""Unit tests for spectrum averaging functions."""

import pytest
import numpy as np
from adctoolbox.spectrum._spectrum_averaging import _power_average, _coherent_average


@pytest.mark.parametrize("M,N", [
    (1, 1024),
    (8, 1024),
    (1, 256),
    (4, 512),
    (10, 128),
    (10, 127),
])
def test_power_average(M, N):
    """Test power-averaged spectrum computation."""
    data = np.random.randn(M, N)

    spectrum_power, _ = _power_average(data)

    print(f"\n[Power Average Test] M={M}, N={N}, size(spectrum_power)={spectrum_power.shape}")
    assert spectrum_power.shape == (N // 2 + 1,)
    assert np.all(np.isfinite(spectrum_power))

    spectrum_powers_single = []
    for i in range(M):
        sp, _ = _power_average(data[i:i+1, :])
        spectrum_powers_single.append(sp)

    expected_avg = np.mean(spectrum_powers_single, axis=0)
    assert np.allclose(spectrum_power, expected_avg, rtol=1e-10)

    # Parseval's theorem: energy conservation
    # Time domain: sum of squared samples averaged over all runs
    time_total_energy_avg = np.mean(np.sum(data**2, axis=1))

    # Frequency domain: restore DC and Nyquist boundary corrections, and account for negative frequencies
    # _power_average divides DC/Nyquist bins by 2 for single-sided representation
    # For single-sided spectrum: DC and Nyquist are unique, middle bins represent both +/- frequencies
    spectrum_full = spectrum_power.copy()
    spectrum_full[0] *= 2.0  # Restore DC boundary correction
    if N % 2 == 0:
        # Even N: has Nyquist bin at the end
        spectrum_full[1:-1] *= 2.0  # Middle bins (exclude Nyquist)
        spectrum_full[-1] *= 2.0  # Restore Nyquist boundary correction
    else:
        # Odd N: no Nyquist bin, all bins except DC represent both +/- frequencies
        spectrum_full[1:] *= 2.0  # All bins except DC

    freq_total_energy_avg = N * np.sum(spectrum_full)
    assert np.isclose(time_total_energy_avg, freq_total_energy_avg, rtol=1e-10)


@pytest.mark.parametrize("M,N,osr", [
    (1, 1024, 1),
    (8, 1024, 1),
    (4, 512, 2),    
    (4, 511, 2),
    (10, 127, 1),
])
def test_coherent_average(M, N, osr):
    """Test coherent-averaged spectrum with phase alignment."""
    bin_target = N // 15
    A = 1.0  # Sine wave amplitude
    # Vectorized sine wave generation with random phases
    phases = 2 * np.pi * np.random.rand(M, 1)
    t = np.arange(N)
    data = A * np.sin(2 * np.pi * bin_target * t / N + phases)

    spectrum_power, spectrum_complex = _coherent_average(data, osr)

    # Shape and power/voltage duality checks
    print(f"\n[Coherent Average Test] M={M}, N={N}, OSR={osr}, size(spectrum_complex)={spectrum_complex.shape}")
    assert spectrum_power.shape == spectrum_complex.shape == (N // 2 + 1,)
    assert np.allclose(np.abs(spectrum_complex), np.sqrt(spectrum_power), rtol=1e-10)

    # Phase alignment normalization check (peak power should be M-independent)
    peak_idx = np.argmax(spectrum_power)

    if M > 1:
        spectrum_power_single, _ = _coherent_average(data[0:1, :], osr)
        assert np.isclose(spectrum_power[peak_idx], spectrum_power_single[peak_idx], rtol=0.1)

    # SNR check (coherent averaging should suppress uncorrelated noise)
    noise_floor = np.median(np.delete(spectrum_power, range(max(0, peak_idx-5), min(len(spectrum_power), peak_idx+6))))
    assert 10 * np.log10(spectrum_power[peak_idx] / (noise_floor + 1e-20)) > 20

    # Parseval's theorem: energy conservation
    # For a pure sine wave with amplitude A over N points:
    # - Mean square value (power per sample): A²/2
    # - Total energy over N samples: E = N × (A²/2)
    # With A=1.0, expected energy = N/2
    total_energy_theoretical = N * (A**2 / 2.0)
    time_total_energy_avg = np.mean(np.sum(data**2, axis=1))

    # Restore DC and Nyquist boundary corrections, and account for negative frequencies
    # For single-sided spectrum: DC and Nyquist are unique, middle bins represent both +/- frequencies
    spectrum_full = spectrum_power.copy()
    spectrum_full[0] *= 2.0  # Restore DC boundary correction
    if N % 2 == 0:
        # Even N: has Nyquist bin at the end
        spectrum_full[1:-1] *= 2.0  # Middle bins (exclude Nyquist)
        spectrum_full[-1] *= 2.0  # Restore Nyquist boundary correction
    else:
        # Odd N: no Nyquist bin, all bins except DC represent both +/- frequencies
        spectrum_full[1:] *= 2.0  # All bins except DC

    freq_total_energy_avg = N * np.sum(spectrum_full)
    print(f"Theoretical: {total_energy_theoretical:.2f}, Time domain: {time_total_energy_avg:.2f}, Frequency domain: {freq_total_energy_avg:.2f}")

    assert np.isclose(time_total_energy_avg, total_energy_theoretical, rtol=1e-8)
    assert np.isclose(time_total_energy_avg, freq_total_energy_avg, rtol=1e-8)


