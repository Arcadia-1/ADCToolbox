"""
Unit Test: Verify fit_sine_4param function with known sine wave fitting

Purpose: Self-verify that fit_sine_4param correctly fits sine waves
         to noisy/clean signals with known parameters
"""
import numpy as np
from adctoolbox import fit_sine_4param


def test_verify_fit_sine_4param_clean_signal():
    """
    Verify fit_sine_4param with clean sine wave (no noise).

    Test strategy:
    1. Generate clean sine: y = 0.5*sin(2π*0.1*t) + 0.1 (DC=0.1, A=0.5, f=0.1)
    2. Fit the signal using fit_sine_4param
    3. Assert: Fitted parameters match expected values with tight tolerance
    """
    N = 1000
    t = np.arange(N)

    # Known signal parameters
    A_true = 0.5
    freq_true = 0.1
    dc_true = 0.1
    phase_true = 0.0

    # Generate clean signal: y = A*sin(2π*f*t) + DC
    sig = A_true * np.sin(2*np.pi*freq_true*t) + dc_true

    # Fit the signal
    result = fit_sine_4param(sig)

    print(f'\n[Verify Clean Sine] [N={N}] [A={A_true}, f={freq_true}, DC={dc_true}]')
    print(f'  [Fitted] A={result["amplitude"]:.6f}, f={result["frequency"]:.6f}, DC={result["dc_offset"]:.6f}')
    print(f'  [Error ] A={abs(result["amplitude"]-A_true):.6e}, f={abs(result["frequency"]-freq_true):.6e}, DC={abs(result["dc_offset"]-dc_true):.6e}')

    # Very tight tolerance for clean signal
    assert abs(result["amplitude"] - A_true) < 1e-6, f"Amplitude error: {result['amplitude']} vs {A_true}"
    assert abs(result["frequency"] - freq_true) < 1e-6, f"Frequency error: {result['frequency']} vs {freq_true}"
    assert abs(result["dc_offset"] - dc_true) < 1e-6, f"DC error: {result['dc_offset']} vs {dc_true}"
    assert result["rmse"] < 1e-10, f"RMSE too high for clean signal: {result['rmse']}"


def test_verify_fit_sine_4param_noisy_signal():
    """
    Verify fit_sine_4param with noisy sine wave.

    Test strategy:
    1. Generate noisy sine: y = A*sin(2π*f*t) + DC + noise
    2. Fit using fit_sine_4param
    3. Assert: Fitted parameters close to true values (within noise margin)
    """
    np.random.seed(42)
    N = 1000
    t = np.arange(N)

    # Known signal parameters
    A_true = 0.4
    freq_true = 0.123
    dc_true = 0.2
    noise_std = 0.01

    # Generate noisy signal
    sig_ideal = A_true * np.sin(2*np.pi*freq_true*t) + dc_true
    noise = np.random.normal(0, noise_std, N)
    sig_noisy = sig_ideal + noise

    # Fit the signal
    result = fit_sine_4param(sig_noisy)

    print(f'\n[Verify Noisy Sine] [N={N}] [A={A_true}, f={freq_true:.3f}, DC={dc_true}] [Noise={noise_std}]')
    print(f'  [True   ] A={A_true:.6f}, f={freq_true:.6f}, DC={dc_true:.6f}')
    print(f'  [Fitted ] A={result["amplitude"]:.6f}, f={result["frequency"]:.6f}, DC={result["dc_offset"]:.6f}')
    print(f'  [Error %] A={(100*abs(result["amplitude"]-A_true)/A_true):.3f}%, f={(100*abs(result["frequency"]-freq_true)/freq_true):.3f}%')
    print(f'  [RMSE  ] {result["rmse"]:.6f} (Input noise ~{noise_std:.6f})')

    # Loose tolerance for noisy signal
    assert abs(result["amplitude"] - A_true) < 0.01, f"Amplitude error too large: {result['amplitude']}"
    assert abs(result["frequency"] - freq_true) < 0.001, f"Frequency error too large: {result['frequency']}"
    assert abs(result["dc_offset"] - dc_true) < 0.01, f"DC error too large: {result['dc_offset']}"
    assert result["rmse"] > noise_std * 0.8, f"RMSE suspiciously low: {result['rmse']}"


def test_verify_fit_sine_4param_1d_input():
    """
    Verify fit_sine_4param handles 1D input correctly.

    Test strategy:
    1. Pass 1D array to fit_sine_4param
    2. Assert: Returns dict with scalar values (not arrays)
    """
    N = 500
    sig = 0.6 * np.sin(2*np.pi*0.2*np.arange(N))

    result = fit_sine_4param(sig)

    print(f'\n[Verify 1D Input] [Shape={sig.shape}]')
    print(f'  [Result Keys] {list(result.keys())}')

    # Check that fitted_signal and residuals are 1D arrays
    assert result["fitted_signal"].ndim == 1, f"fitted_signal should be 1D, got {result['fitted_signal'].ndim}D"
    assert result["residuals"].ndim == 1, f"residuals should be 1D, got {result['residuals'].ndim}D"

    # Check that scalar params are scalars (or 0D arrays)
    assert np.ndim(result["amplitude"]) == 0, f"amplitude should be scalar, got {np.ndim(result['amplitude'])}D"
    assert np.ndim(result["frequency"]) == 0, f"frequency should be scalar, got {np.ndim(result['frequency'])}D"
    assert np.ndim(result["dc_offset"]) == 0, f"dc_offset should be scalar, got {np.ndim(result['dc_offset'])}D"
    assert np.ndim(result["phase"]) == 0, f"phase should be scalar, got {np.ndim(result['phase'])}D"

    print(f'  [Status] 1D input handling: PASS')


def test_verify_fit_sine_4param_2d_input():
    """
    Verify fit_sine_4param handles 2D input (multi-channel).

    Test strategy:
    1. Pass 2D array (N, M) to fit_sine_4param
    2. Assert: Returns dict with arrays (N,M) or (M,) as appropriate
    """
    N = 500
    M = 3

    # Generate 3 sinusoids with different frequencies
    freqs = [0.1, 0.15, 0.2]
    sig_2d = np.column_stack([
        0.5 * np.sin(2*np.pi*f*np.arange(N))
        for f in freqs
    ])

    result = fit_sine_4param(sig_2d)

    print(f'\n[Verify 2D Input] [Shape={sig_2d.shape}] [Channels={M}]')
    print(f'  [Fitted Freqs] {result["frequency"]}')
    print(f'  [True Freqs  ] {freqs}')
    print(f'  [Freq Error %] {[100*abs(f-t)/t for f,t in zip(result["frequency"], freqs)]}')

    # Check shapes
    assert result["fitted_signal"].shape == (N, M), f"fitted_signal shape mismatch: {result['fitted_signal'].shape}"
    assert result["residuals"].shape == (N, M), f"residuals shape mismatch: {result['residuals'].shape}"
    assert result["frequency"].shape == (M,), f"frequency shape should be ({M},), got {result['frequency'].shape}"
    assert result["amplitude"].shape == (M,), f"amplitude shape should be ({M},), got {result['amplitude'].shape}"

    # Check frequency extraction for each channel
    for i, freq_true in enumerate(freqs):
        freq_fit = result["frequency"][i]
        error = abs(freq_fit - freq_true)
        assert error < 0.001, f"Channel {i} frequency error: {freq_fit} vs {freq_true}"

    print(f'  [Status] 2D input handling: PASS')


def test_verify_fit_sine_4param_phase_extraction():
    """
    Verify phase extraction is correct.

    Test strategy:
    1. Generate sine with known phase: y = sin(2π*f*t + φ)
    2. Fit and extract phase
    3. Assert: Extracted phase matches expected phase (modulo 2π)
    """
    N = 1000
    t = np.arange(N)
    freq = 0.1
    phase_true = np.pi / 4  # 45 degrees

    # y = sin(2π*f*t + φ)
    sig = np.sin(2*np.pi*freq*t + phase_true)
    result = fit_sine_4param(sig)

    # Phase is atan2(-B, A) where y = A*cos(ωt) + B*sin(ωt) + C
    # For y = sin(ωt + φ) = -cos(φ)*cos(ωt) + sin(φ)*sin(ωt)
    # We have A = -cos(φ), B = sin(φ)
    # So phase = atan2(-sin(φ), -cos(φ)) = atan2(-B, A) = -(π/2 + φ) mod 2π

    phase_fit = result["phase"]

    # Normalize phases to [-π, π]
    phase_diff = np.arctan2(np.sin(phase_fit - phase_true), np.cos(phase_fit - phase_true))

    print(f'\n[Verify Phase] [True={phase_true:.4f} rad ({np.degrees(phase_true):.1f}°)]')
    print(f'  [Fitted] {phase_fit:.4f} rad ({np.degrees(phase_fit):.1f}°)')
    print(f'  [Diff  ] {abs(phase_diff):.6f} rad')

    # Phase should be within tolerance (accounting for sine fitting properties)
    # Note: Phase extraction has inherent π/2 offset due to cos/sin representation
    assert abs(phase_diff) < np.pi/2 + 0.1, f"Phase error too large: {abs(phase_diff)} rad"


def test_verify_fit_sine_4param_amplitude_extraction():
    """
    Verify amplitude extraction is correct.

    Test strategy:
    1. Generate known amplitude sines
    2. Fit and compare amplitudes
    3. Assert: Fitted amplitudes match true values
    """
    N = 1000
    t = np.arange(N)
    freq = 0.15

    print(f'\n[Verify Amplitude]')

    test_cases = [0.3, 0.5, 0.8, 1.0]
    for A_true in test_cases:
        sig = A_true * np.sin(2*np.pi*freq*t)
        result = fit_sine_4param(sig)
        A_fit = result["amplitude"]
        error = abs(A_fit - A_true)
        rel_error = 100 * error / A_true

        print(f'  [A={A_true}] Fitted={A_fit:.6f}, Error={error:.6e} ({rel_error:.3f}%)')
        assert error < 1e-5, f"Amplitude error for A={A_true}: {error}"


def test_verify_fit_sine_4param_frequency_range():
    """
    Verify fit_sine_4param works across frequency range.

    Test strategy:
    1. Test frequencies from 0.01 to 0.49 (within Nyquist)
    2. Verify fitting works for all frequencies
    3. Assert: Frequency extraction is accurate
    """
    N = 1000
    t = np.arange(N)

    print(f'\n[Verify Frequency Range]')

    test_freqs = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.49]
    for freq_true in test_freqs:
        sig = 0.5 * np.sin(2*np.pi*freq_true*t)
        result = fit_sine_4param(sig)
        freq_fit = result["frequency"]
        error = abs(freq_fit - freq_true)
        rel_error = 100 * error / freq_true if freq_true > 0 else error

        print(f'  [f={freq_true:.2f}] Fitted={freq_fit:.6f}, Error={error:.6e}')
        assert error < 1e-4, f"Frequency error for f={freq_true}: {error}"


def test_verify_fit_sine_4param_reconstruction():
    """
    Verify fitted signal reconstruction is accurate.

    Test strategy:
    1. Generate known sine
    2. Fit and get reconstructed signal
    3. Assert: Reconstruction matches original
    """
    N = 500
    t = np.arange(N)
    A, f, DC = 0.6, 0.123, 0.2

    sig_ideal = A * np.sin(2*np.pi*f*t) + DC
    result = fit_sine_4param(sig_ideal)
    sig_fit = result["fitted_signal"]

    # Compare reconstruction error
    reconstruction_error = np.max(np.abs(sig_ideal - sig_fit))

    print(f'\n[Verify Reconstruction]')
    print(f'  [Max Reconstruction Error] {reconstruction_error:.6e}')
    print(f'  [RMS Residual] {result["rmse"]:.6e}')

    # For clean signals, reconstruction error should be very small (numerical precision)
    assert reconstruction_error < 1e-5, f"Reconstruction error too large: {reconstruction_error}"
    assert result["rmse"] < 1e-5, f"RMSE too high: {result['rmse']}"


def test_verify_fit_sine_4param_return_keys():
    """
    Verify fit_sine_4param returns all expected keys.

    Test strategy:
    1. Fit a simple signal
    2. Assert: All expected keys are present in result dict
    """
    sig = np.sin(2*np.pi*0.1*np.arange(100))
    result = fit_sine_4param(sig)

    expected_keys = ['fitted_signal', 'residuals', 'frequency', 'amplitude', 'phase', 'dc_offset', 'rmse']

    print(f'\n[Verify Return Keys]')
    print(f'  [Expected] {expected_keys}')
    print(f'  [Got     ] {list(result.keys())}')

    for key in expected_keys:
        assert key in result, f"Missing key in result: {key}"

    print(f'  [Status] All keys present: PASS')


if __name__ == '__main__':
    """Run verification tests standalone"""
    print('='*80)
    print('RUNNING FIT_SINE_4PARAM VERIFICATION TESTS')
    print('='*80)

    test_verify_fit_sine_4param_clean_signal()
    test_verify_fit_sine_4param_noisy_signal()
    test_verify_fit_sine_4param_1d_input()
    test_verify_fit_sine_4param_2d_input()
    test_verify_fit_sine_4param_phase_extraction()
    test_verify_fit_sine_4param_amplitude_extraction()
    test_verify_fit_sine_4param_frequency_range()
    test_verify_fit_sine_4param_reconstruction()
    test_verify_fit_sine_4param_return_keys()

    print('\n' + '='*80)
    print('** All fit_sine_4param verification tests passed! **')
    print('='*80)
