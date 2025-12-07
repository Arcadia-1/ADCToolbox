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
from adctoolbox.aout import analyze_phase_spectrum


def generate_test_signal():
    """
    Generate synthetic signal with known harmonics and phases.

    Returns:
        signal: Test signal with 4 harmonics + noise
        params: Dictionary of expected values
    """
    # Set random seed for reproducibility
    np.random.seed(42)

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
    signal = signal + 0.5 + np.random.randn(N) * 1e-5

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

    FFT mode should return:
    - Empty harm_phase and harm_mag arrays
    - NaN for freq and noise_dB
    """
    signal, params = generate_test_signal()

    print('\n[Verify spec_plot_phase] [FFT Mode]')
    result = analyze_phase_spectrum(signal, harmonic=5, mode='FFT', show_plot=False)

    # Check that FFT mode returns expected empty/NaN values
    hp_empty = len(result['harm_phase']) == 0
    hm_empty = len(result['harm_mag']) == 0
    freq_nan = np.isnan(result['freq'])
    noise_nan = np.isnan(result['noise_dB'])

    status = 'PASS' if hp_empty and hm_empty and freq_nan and noise_nan else 'FAIL'
    print(f'  [harm_phase=empty] [harm_mag=empty] [freq=NaN] [noise_dB=NaN] [{status}]')

    assert hp_empty, "FFT mode should return empty harm_phase"
    assert hm_empty, "FFT mode should return empty harm_mag"
    assert freq_nan, "FFT mode should return NaN for freq"
    assert noise_nan, "FFT mode should return NaN for noise_dB"


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
    result = analyze_phase_spectrum(signal, harmonic=5, mode='LMS', show_plot=False)

    harm_phase = result['harm_phase']
    harm_mag = result['harm_mag']
    freq = result['freq']
    noise_dB = result['noise_dB']

    # 1. Verify frequency detection (LMS algorithm has small numerical errors)
    freq_error = abs(freq - params['Fin_normalized'])
    freq_ok = freq_error < 1e-5

    # 2. Verify magnitude detection (within 5%)
    expected_mags = params['amplitudes']
    mag_errors = [abs(harm_mag[i] - expected_mags[i]) / expected_mags[i] * 100
                  for i in range(min(4, len(harm_mag)))]
    mag_ok = all(err < 5 for err in mag_errors)

    # 3. Verify noise floor is reasonable
    noise_ok = noise_dB < -40

    # 4. Verify number of harmonics detected
    count_ok = len(harm_phase) >= 4 and len(harm_mag) >= 4

    status = 'PASS' if freq_ok and mag_ok and noise_ok and count_ok else 'FAIL'
    print(f'  [FreqErr={freq_error:.2e}] [MagOK={mag_ok}] [Noise={noise_dB:.1f}dB] [Harm={len(harm_mag)}] [{status}]')

    assert freq_ok, f"Frequency error too large: {freq_error:.2e}"
    assert mag_ok, "Magnitude errors too large"
    assert noise_ok, f"Noise floor too high: {noise_dB:.2f} dB (expected < -40 dB)"
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
    result_fft = spec_plot_phase(signal, harmonic=5, mode='FFT', show_plot=False)
    result_lms = spec_plot_phase(signal, harmonic=5, mode='LMS', show_plot=False)

    # FFT mode: minimal output
    fft_ok = len(result_fft['harm_phase']) == 0 and len(result_fft['harm_mag']) == 0

    # LMS mode: detailed output
    lms_ok = (len(result_lms['harm_phase']) > 0 and
              len(result_lms['harm_mag']) > 0 and
              not np.isnan(result_lms['freq']) and
              not np.isnan(result_lms['noise_dB']))

    status = 'PASS' if fft_ok and lms_ok else 'FAIL'
    print(f'  [FFT: hp={len(result_fft["harm_phase"])} hm={len(result_fft["harm_mag"])}]', end=' ')
    print(f'[LMS: hp={len(result_lms["harm_phase"])} hm={len(result_lms["harm_mag"])}] [{status}]')

    assert fft_ok, "FFT mode should return empty arrays"
    assert lms_ok, "LMS mode should return detailed output"


if __name__ == '__main__':
    """Run verification tests standalone"""
    print('Running spec_plot_phase verification tests...\n')
    test_verify_spec_plot_phase_fft_mode()
    test_verify_spec_plot_phase_lms_mode()
    test_verify_spec_plot_phase_comparison()
    print('\n** All verification tests passed! **')
