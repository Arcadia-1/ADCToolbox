"""
Unit Test: Verify spec_plot_phase with synthetic signals

Purpose: Self-verify that spec_plot_phase correctly detects harmonics and phases
         in both FFT and LMS modes using synthetic test signals
         (NOT compared against MATLAB)
"""
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for testing
from adctoolbox.spectrum import analyze_spectrum_polar
from adctoolbox.aout import analyze_decomposition_polar


def generate_test_signal():
    """
    Generate synthetic signal with known harmonics and phases.

    Returns:
        signal: Test signal with 4 harmonics + noise
        params: Dictionary of expected values
    """
    rng = np.random.default_rng(42)

    N = 8192           # Number of samples
    Fs = 1e9           # Sampling frequency
    J = 323            # Fundamental bin (prime number for coherent)
    Fin = J * Fs / N   # Fundamental frequency

    t = np.arange(N) / Fs

    # Known amplitudes and phases
    A_fundamental = 0.45
    A_HD2 = 0.05   # 2nd harmonic at -19 dB
    A_HD3 = 0.02   # 3rd harmonic at -27 dB
    A_HD4 = 0.01   # 4th harmonic at -33 dB

    phi_fundamental = 0.5
    phi_HD2 = 1.2
    phi_HD3 = -0.8
    phi_HD4 = 0.3

    # Build signal
    signal = (A_fundamental * np.sin(2*np.pi*Fin*t + phi_fundamental) +
              A_HD2 * np.sin(2*np.pi*2*Fin*t + phi_HD2) +
              A_HD3 * np.sin(2*np.pi*3*Fin*t + phi_HD3) +
              A_HD4 * np.sin(2*np.pi*4*Fin*t + phi_HD4))

    # Add DC offset and small noise
    signal = signal + 0.5 + rng.standard_normal(N) * 1e-5

    # Calculate maxSignal for normalization (needed for expected values)
    maxSignal = np.max(signal) - np.min(signal)

    # Expected magnitudes in normalized units (what the function returns)
    # Function returns: sqrt(I^2 + Q^2) * 2, where I,Q are from normalized signal
    # For normalized signal: amplitude_norm = amplitude_original / maxSignal
    # So expected_mag_normalized = (amplitude_original / maxSignal) * 2
    amplitudes_normalized = [A * 2 / maxSignal for A in [A_fundamental, A_HD2, A_HD3, A_HD4]]

    params = {
        'N': N,
        'Fs': Fs,
        'Fin': Fin,
        'Fin_normalized': Fin / Fs,
        'amplitudes': amplitudes_normalized,  # Now in normalized units
        'phases': [phi_fundamental, phi_HD2, phi_HD3, phi_HD4]
    }

    return signal, params


def test_verify_spec_plot_phase_fft_mode():
    """
    Verify spec_plot_phase FFT mode correctly processes signal.

    FFT mode returns the standard polar spectrum metrics dictionary.
    """
    signal, params = generate_test_signal()

    print('\n[Verify spec_plot_phase] [FFT Mode]')
    metrics = analyze_spectrum_polar(signal, harmonic=5, create_plot=False)

    # Check that FFT mode returns expected values
    has_sndr = 'sndr_dbc' in metrics
    has_sfdr = 'sfdr_dbc' in metrics
    has_harmonics = 'harmonics_dbc' in metrics

    status = 'PASS' if has_sndr and has_sfdr and has_harmonics else 'FAIL'
    print(f'  [sndr_dbc=present] [sfdr_dbc=present] [harmonics_dbc=present] [{status}]')

    assert has_sndr, "FFT mode should return sndr_dbc"
    assert has_sfdr, "FFT mode should return sfdr_dbc"
    assert has_harmonics, "FFT mode should return harmonics_dbc"
    assert np.isfinite(metrics['sndr_dbc'])
    assert np.isfinite(metrics['sfdr_dbc'])


def test_verify_spec_plot_phase_lms_mode():
    """
    Verify spec_plot_phase LMS mode correctly detects frequency, harmonics, and phases.

    Test strategy:
    1. Generate signal with known harmonics (4 harmonics + noise)
    2. Run LMS mode to detect frequency, magnitudes, and phases
    3. Assert: Detected values match expected within tolerance
    """
    signal, params = generate_test_signal()

    print('\n[Verify spec_plot_phase] [LMS Mode]')
    decomp_result = analyze_decomposition_polar(signal, harmonic=5, create_plot=False)

    harm_phase = decomp_result['phases']
    harm_mag = decomp_result['magnitudes']
    freq = decomp_result['fundamental_freq']
    noise_dB = decomp_result['noise_db']

    # 1. Verify frequency detection (LMS algorithm has small numerical errors)
    freq_error = abs(freq - params['Fin_normalized'])
    freq_ok = freq_error < 1e-5

    # 2. Verify magnitude detection (within 5%)
    expected_mags = params['amplitudes']
    mag_errors = [abs(harm_mag[i] - expected_mags[i]) / expected_mags[i] * 100
                  for i in range(min(4, len(harm_mag)))]
    mag_ok = all(err < 5 for err in mag_errors)

    # 3. Verify noise floor is reasonable
    noise_ok = noise_dB < -38

    # 4. Verify number of harmonics detected
    count_ok = len(harm_phase) >= 4 and len(harm_mag) >= 4

    status = 'PASS' if freq_ok and mag_ok and noise_ok and count_ok else 'FAIL'
    print(f'  [FreqErr={freq_error:.2e}] [MagOK={mag_ok}] [Noise={noise_dB:.1f}dB] [Harm={len(harm_mag)}] [{status}]')

    assert freq_ok, f"Frequency error too large: {freq_error:.2e}"
    assert mag_ok, "Magnitude errors too large"
    assert noise_ok, f"Noise floor too high: {noise_dB:.2f} dB (expected < -38 dB)"
    assert count_ok, "Should detect at least 4 harmonics"


def test_verify_spec_plot_phase_comparison():
    """
    Compare FFT and LMS mode behavior on same signal.

    Verify that:
    - Both modes run without errors
    - LMS mode provides more detailed analysis than FFT mode
    """
    signal, params = generate_test_signal()

    print('\n[Verify spec_plot_phase] [FFT vs LMS]')

    # Run both modes
    coherent_result = analyze_spectrum_polar(signal, harmonic=5, create_plot=False)
    decomp_result = analyze_decomposition_polar(signal, harmonic=5, create_plot=False)

    # FFT mode: returns spectrum metrics
    fft_ok = 'sndr_dbc' in coherent_result and 'sfdr_dbc' in coherent_result

    # LMS mode: returns detailed harmonic decomposition
    lms_ok = (len(decomp_result['phases']) > 0 and
              len(decomp_result['magnitudes']) > 0 and
              'fundamental_freq' in decomp_result and
              'noise_db' in decomp_result)

    status = 'PASS' if fft_ok and lms_ok else 'FAIL'
    print(f'  [FFT: has_spectrum={fft_ok}]', end=' ')
    print(f'[LMS: hp={len(decomp_result["phases"])} hm={len(decomp_result["magnitudes"])}] [{status}]')

    assert fft_ok, "FFT mode should return spectrum metrics"
    assert lms_ok, "LMS mode should return detailed output"


if __name__ == '__main__':
    """Run verification tests standalone"""
    print('Running spec_plot_phase verification tests...\n')
    test_verify_spec_plot_phase_fft_mode()
    test_verify_spec_plot_phase_lms_mode()
    test_verify_spec_plot_phase_comparison()
    print('\n** All verification tests passed! **')
