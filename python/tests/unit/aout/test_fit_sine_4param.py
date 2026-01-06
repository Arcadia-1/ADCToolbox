"""Unit tests for 4-parameter sine wave fitting with figure generation."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import fit_sine_4param


# Create output directory for test figures
output_dir = Path(__file__).parent / "test_output"
output_dir.mkdir(exist_ok=True)


def test_fit_sine_4param_basic():
    """Test basic 4-parameter sine wave fitting."""
    N = 2**13
    Fs = 800e6
    Fin = 10.1234567e6
    t = np.arange(N) / Fs
    A = 0.499
    DC = 0.5
    noise_rms = 20e-3

    print(f"\n[Sinewave] [Fs={Fs/1e6:.1f} MHz] [Fin={Fin/1e6:.6f} MHz] [Amplitude={A:.3f} V] [DC={DC:.3f} V] [Noise RMS={noise_rms*1e3:.2f} mV]")

    sig_ideal = A * np.cos(2 * np.pi * Fin * t) + DC
    sig_noisy = sig_ideal + np.random.randn(N) * noise_rms

    # Fit 4-parameter sine wave
    result = fit_sine_4param(sig_noisy)

    # Extract fitted parameters
    dc_fit = result['dc_offset']
    mag_fit = result['amplitude']
    phase_fit = result['phase'] * 180 / np.pi
    freq_fit = result['frequency'] * Fs
    sig_fit = result['fitted_signal']

    # Calculate errors
    residual = sig_noisy - sig_fit
    residual_rms = np.sqrt(np.mean(residual**2))
    reconstruction_error = np.sqrt(np.mean((sig_ideal - sig_fit)**2))

    # Parameter errors
    dc_error = abs(DC - dc_fit)
    mag_error = abs(A - mag_fit)
    phase_expected = 0.0

    # Normalize phase difference to [-180, 180] range
    phase_diff = phase_fit - phase_expected
    phase_error = ((phase_diff + 180) % 360) - 180
    freq_error = abs(Fin - freq_fit)

    print(f"\n[Expected] [DC={DC:8.4f} V] [Amplitude={A:8.4f} V] [Freq={Fin/1e6:10.7f} MHz] [Phase={phase_expected:8.4f} deg]")
    print(f"[Fitted  ] [DC={dc_fit:8.4f} V] [Amplitude={mag_fit:8.4f} V] [Freq={freq_fit/1e6:10.7f} MHz] [Phase={phase_fit:8.4f} deg]")

    # Assert that fitting error is within acceptable limits
    residual_rms_pct = abs(100 * (residual_rms - noise_rms)) / noise_rms
    assert residual_rms_pct < 10.0, f"Residual RMS error percentage {residual_rms_pct:.4f}% exceeds 10% threshold!"
    print(f"\n[Residual error ] fitted rms={residual_rms*1e3:.2f} mV, input noise={noise_rms*1e3:.2f} mV, error percentage={residual_rms_pct:.2f}% -> [PASS]")

    # Check frequency error
    freq_error_pct = 100 * freq_error / Fin
    assert freq_error_pct < 1.0, f"Frequency error percentage {freq_error_pct:.6f}% exceeds 1% threshold!"
    print(f"[Frequency error] freq error={freq_error:.4f} Hz, percentage={freq_error_pct:.6f}% -> [PASS]")

    # Check phase error
    phase_error_abs = abs(phase_error)
    assert phase_error_abs < 5.0, f"Phase error {phase_error_abs:.4f} deg exceeds 5 deg threshold!"
    print(f"[Phase error    ] phase error={phase_error:.4f} deg -> [PASS]")

    # Check reconstruction error
    assert reconstruction_error < 1e-2, f"reconstruction error {reconstruction_error:.6f} exceeds 0.01 threshold!"
    print(f"[Reconstruction ] reconstruction error (rms) ={reconstruction_error:.2e} V -> [PASS]")

    # Create figure
    period_samples = Fs / Fin
    num_cycles = 3
    num_samples = int(period_samples * num_cycles)
    sample_range = slice(0, num_samples)

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    ax1.plot(t[sample_range]*1e9, sig_noisy[sample_range]*1e3, 'o', markersize=4, alpha=0.5, label='Raw data (noisy)')
    ax1.plot(t[sample_range]*1e9, sig_fit[sample_range]*1e3, 'r-', linewidth=2, label='Fitted')
    ax1.set_xlabel('Time (ns)', fontsize=12)
    ax1.set_ylabel('Amplitude (mV)', fontsize=12)
    ax1.set_ylim(min(sig_noisy[sample_range]*1e3)-100, max(sig_noisy[sample_range]*1e3)+200)
    ax1.set_title(f'Sine Fit: 3 Periods (Noise={noise_rms*1e3:.2f}mV)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    fig_path = output_dir / 'test_fit_sine_4param.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\n[Save fig] -> [{fig_path.resolve()}]\n")

    # Verify file was created
    assert fig_path.exists(), f"Figure file not created: {fig_path}"
    assert fig_path.stat().st_size > 0, f"Figure file is empty: {fig_path}"


@pytest.mark.parametrize("noise_level", [5e-3, 20e-3, 50e-3])
def test_fit_sine_4param_noise_levels(noise_level):
    """Test sine fitting at different noise levels."""
    N = 2**13
    Fs = 800e6
    Fin = 10.1234567e6
    t = np.arange(N) / Fs
    A = 0.499
    DC = 0.5

    sig_ideal = A * np.cos(2 * np.pi * Fin * t) + DC
    sig_noisy = sig_ideal + np.random.randn(N) * noise_level

    result = fit_sine_4param(sig_noisy)
    sig_fit = result['fitted_signal']
    residual = sig_noisy - sig_fit
    residual_rms = np.sqrt(np.mean(residual**2))

    # Residual RMS should be close to input noise level
    residual_rms_pct = abs(100 * (residual_rms - noise_level)) / noise_level
    assert residual_rms_pct < 15.0, f"Residual RMS error percentage {residual_rms_pct:.4f}% exceeds 15% threshold at noise={noise_level*1e3:.2f}mV"

    print(f"\n[Noise={noise_level*1e3:.2f}mV] Residual RMS error percentage: {residual_rms_pct:.2f}% -> [PASS]")


if __name__ == '__main__':
    """Run fit_sine_4param tests standalone"""
    print('='*80)
    print('RUNNING FIT_SINE_4PARAM TESTS')
    print('='*80)

    test_fit_sine_4param_basic()
    test_fit_sine_4param_noise_levels(5e-3)
    test_fit_sine_4param_noise_levels(20e-3)
    test_fit_sine_4param_noise_levels(50e-3)

    print('\n' + '='*80)
    print(f'** All fit_sine_4param tests completed! **')
    print(f'** Figures saved to: {output_dir.resolve()} **')
    print('='*80)
