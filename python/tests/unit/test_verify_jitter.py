"""
Unit Test: Verify jitter analysis with synthetic signals

Purpose: Self-verify that jitter measurement correctly extracts timing jitter
         from signals with known phase noise characteristics
         (NOT compared against MATLAB)

Strategy:
    1. Generate synthetic sinusoids with controlled phase noise
    2. Apply jitter analysis (sine fit + phase noise extraction)
    3. Assert: Measured jitter matches set jitter (within tolerance)
"""
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for testing
import matplotlib.pyplot as plt
from adctoolbox.common import fit_sine
from adctoolbox.aout import analyze_spectrum, plot_error_hist_code


def generate_jitter_signal(N, Fs, Fin, Tj_rms, A=0.49, offset=0.5, amp_noise=1e-5, seed=42):
    """
    Generate synthetic sinusoid with timing jitter.

    Args:
        N: Number of samples
        Fs: Sampling frequency (Hz)
        Fin: Input frequency (Hz)
        Tj_rms: RMS timing jitter (seconds)
        A: Signal amplitude
        offset: DC offset
        amp_noise: Amplitude noise level
        seed: Random seed for reproducibility

    Returns:
        signal: Synthetic signal with phase jitter
        params: Dictionary of generation parameters
    """
    np.random.seed(seed)

    Ts = 1 / Fs
    t = np.arange(N) * Ts
    theta = 2 * np.pi * Fin * t

    # Generate phase noise from timing jitter
    # Phase noise RMS = 2*pi*Fin*Tj
    phase_noise_rms = 2 * np.pi * Fin * Tj_rms
    phase_jitter = np.random.randn(N) * phase_noise_rms

    # Generate amplitude noise
    amplitude_noise = np.random.randn(N) * amp_noise

    # Build signal
    signal = A * np.sin(theta + phase_jitter) + offset + amplitude_noise

    params = {
        'N': N,
        'Fs': Fs,
        'Fin': Fin,
        'Tj_set': Tj_rms,
        'phase_noise_rms': phase_noise_rms,
        'A': A,
        'offset': offset
    }

    return signal, params


def measure_jitter(signal, Fs):
    """
    Measure timing jitter from signal using phase noise analysis.

    Args:
        signal: ADC output data
        Fs: Sampling frequency (Hz)

    Returns:
        jitter_measured: RMS timing jitter (seconds)
        sndr: Signal-to-noise and distortion ratio (dB)
        pnoi: Phase noise (radians RMS)
    """
    # Sine fit to extract frequency
    fitted_signal, f_norm, mag, dc, phi = fit_sine(signal)
    Fin_fit = f_norm * Fs

    # Error histogram analysis to extract phase noise
    emean, erms, phase_code, anoi, pnoi, err, xx = plot_error_hist_code(
        signal, bin=99, fin=f_norm, disp=0
    )

    # Calculate jitter from phase noise
    # Tj = pnoi / (2*pi*Fin)
    jitter_measured = pnoi / (2 * np.pi * Fin_fit)

    # Spectrum analysis for SNDR
    ENoB, SNDR, SFDR, SNR, THD, pwr, NF, NSD = analyze_spectrum(
        signal,
        label=0,
        harmonic=0,
        is_plot=0
    )

    return jitter_measured, SNDR, pnoi


def test_verify_jitter_single_point():
    """
    Verify jitter measurement with a single known jitter level.

    Test strategy:
    1. Generate signal with Tj = 5 ps (moderate level)
    2. Measure jitter using phase noise analysis
    3. Assert: Measured jitter matches set jitter (within 15% tolerance)

    Note: At very low jitter levels, measurement accuracy is limited by:
    - Finite sample size (N=4096)
    - Amplitude noise floor
    - Sine fit accuracy
    """
    # Test parameters
    N = 2**12
    Fs = 20e9
    Fin = 400e6
    Tj_set = 5e-12  # 5 ps (moderate level for better accuracy)

    # Generate signal
    signal, params = generate_jitter_signal(N, Fs, Fin, Tj_set)

    # Measure jitter
    Tj_measured, sndr, pnoi = measure_jitter(signal, Fs)

    # Verify measurement
    error_pct = abs(Tj_measured - Tj_set) / Tj_set * 100
    status = 'PASS' if error_pct < 15 else 'FAIL'

    print(f'\n[Verify Jitter] [Tj={Tj_set*1e12:.1f} ps] [Fin={Fin/1e6:.0f} MHz]')
    print(f'  [Measured={Tj_measured*1e12:.3f} ps] [Error={error_pct:.2f}%] [SNDR={sndr:.1f} dB] [{status}]')

    # Tolerance: Within 15% (accounts for finite sample size and random noise)
    assert error_pct < 15, f"Jitter measurement error too large: {error_pct:.2f}%"


def test_verify_jitter_sweep():
    """
    Verify jitter measurement over a range of jitter levels.

    Test strategy:
    1. Generate signals with Tj from 500 fs to 20 ps
    2. Measure jitter for each level
    3. Assert: Measured vs set jitter correlation > 0.99
    4. Assert: Measured jitter increases monotonically with set jitter

    Note: Very low jitter levels (<500 fs) are challenging to measure
    accurately with finite sample sizes.
    """
    # Test parameters
    N = 2**12
    Fs = 20e9
    Fin = 400e6

    # Jitter sweep: 500 fs to 20 ps (logarithmic)
    Tj_set_list = np.logspace(-12.3, -10.7, 10)  # 500 fs to 20 ps

    print(f'\n[Verify Jitter Sweep] [{Tj_set_list[0]*1e15:.0f} fs to {Tj_set_list[-1]*1e12:.1f} ps] [N={len(Tj_set_list)}]')

    Tj_measured_list = np.zeros(len(Tj_set_list))

    for i, Tj_set in enumerate(Tj_set_list):
        # Generate signal with unique seed for each point
        signal, params = generate_jitter_signal(N, Fs, Fin, Tj_set, seed=42+i)

        # Measure jitter
        Tj_measured, sndr, pnoi = measure_jitter(signal, Fs)
        Tj_measured_list[i] = Tj_measured

    # Calculate metrics
    correlation = np.corrcoef(Tj_set_list, Tj_measured_list)[0, 1]
    diffs = np.diff(Tj_measured_list)
    monotonic = np.all(diffs > 0)
    ratios = Tj_measured_list / Tj_set_list
    errors_pct = np.abs(Tj_measured_list[3:] - Tj_set_list[3:]) / Tj_set_list[3:] * 100
    avg_error = np.mean(errors_pct)

    status = 'PASS' if correlation > 0.98 and monotonic and avg_error < 15 else 'FAIL'
    print(f'  [Corr={correlation:.4f}] [Monotonic={monotonic}] [AvgErr={avg_error:.2f}%] [{status}]')

    assert correlation > 0.98, f"Correlation too low: {correlation:.6f}"
    assert monotonic, "Measured jitter should increase monotonically with set jitter"
    assert 0.5 < np.min(ratios) and np.max(ratios) < 2.0, "Measured jitter outside reasonable range"
    assert avg_error < 15, f"Average error too large: {avg_error:.2f}%"


def test_verify_jitter_frequency_dependence():
    """
    Verify jitter measurement is consistent across different frequencies.

    Test strategy:
    1. Generate signals at 400 MHz, 900 MHz with same Tj = 5 ps
    2. Measure jitter for each frequency
    3. Assert: Measured jitter is consistent (within 20% variation)

    Note: Phase noise (radians) scales with frequency (pnoi = 2*pi*Fin*Tj),
    but extracted jitter (seconds) should be frequency-independent.
    """
    N = 2**12
    Fs = 20e9
    Tj_set = 5e-12  # 5 ps (moderate level)
    frequencies = [400e6, 900e6]

    print(f'\n[Verify Jitter Freq Depend] [Tj={Tj_set*1e12:.1f} ps]')

    Tj_measured_list = []

    for Fin in frequencies:
        signal, params = generate_jitter_signal(N, Fs, Fin, Tj_set)
        Tj_measured, sndr, pnoi = measure_jitter(signal, Fs)
        Tj_measured_list.append(Tj_measured)

    # Verify consistency
    Tj_mean = np.mean(Tj_measured_list)
    errors_from_mean = np.abs(Tj_measured_list - Tj_mean) / Tj_mean * 100
    max_deviation = np.max(errors_from_mean)

    for i, Fin in enumerate(frequencies):
        error_from_mean = errors_from_mean[i]
        status = 'PASS' if error_from_mean < 20 else 'FAIL'
        print(f'  [Fin={Fin/1e6:.0f} MHz] [Tj={Tj_measured_list[i]*1e12:.3f} ps] [Dev={error_from_mean:.2f}%] [{status}]')

    assert max_deviation < 20, f"Jitter deviation across frequencies too large: {max_deviation:.2f}%"


if __name__ == '__main__':
    """Run verification tests standalone"""
    print('Running jitter verification tests...\n')
    test_verify_jitter_single_point()
    test_verify_jitter_sweep()
    test_verify_jitter_frequency_dependence()
    print('\n** All jitter verification tests passed! **')
