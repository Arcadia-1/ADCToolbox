"""Unit tests for error envelope spectrum analysis with figure generation."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import analyze_error_envelope_spectrum, find_coherent_frequency
from adctoolbox.siggen import ADC_Signal_Generator


# Create output directory for test figures
output_dir = Path(__file__).parent / "test_output"
output_dir.mkdir(exist_ok=True)


def _assert_envelope_spectrum_panel(ax, result, title, n_samples):
    assert ax.get_title() == title
    assert ax.get_xlabel() == 'Frequency (Hz)'
    assert ax.get_ylabel() == 'Envelope Spectrum (dB)'
    assert len(ax.lines) >= 1
    assert len(ax.lines[0].get_xdata()) > 0
    assert len(ax.lines[0].get_ydata()) > 0
    assert result['error_signal'].shape == (n_samples,)
    assert result['envelope'].shape == (n_samples,)
    assert np.all(np.isfinite(result['envelope']))
    assert np.all(result['envelope'] >= 0)


def test_analyze_error_envelope_spectrum_basic():
    """Test error envelope spectrum analysis for different noise types."""
    # Setup
    N = 2**14
    Fs = 800e6
    Fin_target = 97e6
    Fin, Fin_bin = find_coherent_frequency(Fs, Fin_target, N)
    A = 0.49
    DC = 0.5

    gen = ADC_Signal_Generator(N=N, Fs=Fs, Fin=Fin, A=A, DC=DC)

    print(f"\n[Config] Fs={Fs/1e6:.0f} MHz, Fin={Fin/1e6:.1f} MHz, Bin={Fin_bin}, N={N}")
    print(f"[Config] A={A:.3f} V, DC={DC:.3f} V")

    # Create 3 test cases
    sig_thermal = gen.apply_thermal_noise(noise_rms=180e-6)
    sig_am_noise = gen.apply_thermal_noise(gen.apply_am_noise(None, strength=0.0005), noise_rms=10e-6)
    sig_am_tone = gen.apply_thermal_noise(gen.apply_am_tone(None, am_tone_freq=500e3, am_tone_depth=0.05), noise_rms=10e-6)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Analyze each case
    result1 = analyze_error_envelope_spectrum(sig_thermal, fs=Fs, frequency=Fin/Fs, ax=axes[0], title='Thermal Noise')
    result2 = analyze_error_envelope_spectrum(sig_am_noise, fs=Fs, frequency=Fin/Fs, ax=axes[1], title='AM Noise')
    result3 = analyze_error_envelope_spectrum(sig_am_tone, fs=Fs, frequency=Fin/Fs, ax=axes[2], title='AM Tone')

    fig.suptitle(f'Error Envelope Spectrum Analysis: ADC Non-idealities (Fs={Fs/1e6:.0f} MHz, Fin={Fin/1e6:.1f} MHz)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    fig_path = output_dir / 'test_analyze_error_envelope_spectrum.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')

    print(f"[Save fig] -> [{fig_path.resolve()}]\n")

    # Verify file was created
    assert fig_path.exists(), f"Figure file not created: {fig_path}"
    assert fig_path.stat().st_size > 0, f"Figure file is empty: {fig_path}"

    assert len(fig.axes) == 3
    for ax, result, title in [
        (axes[0], result1, 'Thermal Noise'),
        (axes[1], result2, 'AM Noise'),
        (axes[2], result3, 'AM Tone'),
    ]:
        _assert_envelope_spectrum_panel(ax, result, title, N)
    plt.close(fig)


if __name__ == '__main__':
    """Run analyze_error_envelope_spectrum tests standalone"""
    print('='*80)
    print('RUNNING ANALYZE_ERROR_ENVELOPE_SPECTRUM TESTS')
    print('='*80)

    test_analyze_error_envelope_spectrum_basic()

    print('\n' + '='*80)
    print(f'** All analyze_error_envelope_spectrum tests completed! **')
    print(f'** Figures saved to: {output_dir.resolve()} **')
    print('='*80)
