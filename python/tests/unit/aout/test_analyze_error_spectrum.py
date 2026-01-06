"""Unit tests for error spectrum analysis with figure generation."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import analyze_error_spectrum, find_coherent_frequency
from adctoolbox.siggen import ADC_Signal_Generator


# Create output directory for test figures
output_dir = Path(__file__).parent / "test_output"
output_dir.mkdir(exist_ok=True)


def test_analyze_error_spectrum_basic():
    """Test error spectrum analysis for thermal noise and nonlinearity."""
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
    sig_jitter = gen.apply_thermal_noise(gen.apply_jitter(None, jitter_rms=2e-12), noise_rms=10e-6)
    sig_nonlin = gen.apply_thermal_noise(gen.apply_static_nonlinearity(None, k2=0, k3=0.01), noise_rms=10e-6)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Analyze each case
    analyze_error_spectrum(sig_thermal, fs=Fs, ax=axes[0], title='Thermal Noise')
    analyze_error_spectrum(sig_jitter, fs=Fs, ax=axes[1], title='Jitter Noise')
    analyze_error_spectrum(sig_nonlin, fs=Fs, ax=axes[2], title='Static HD3')

    fig.suptitle(f'Error Spectrum Analysis: ADC Non-idealities (Fs={Fs/1e6:.0f} MHz, Fin={Fin/1e6:.1f} MHz)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    fig_path = output_dir / 'test_analyze_error_spectrum.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[Save fig] -> [{fig_path.resolve()}]\n")

    # Verify file was created
    assert fig_path.exists(), f"Figure file not created: {fig_path}"
    assert fig_path.stat().st_size > 0, f"Figure file is empty: {fig_path}"


if __name__ == '__main__':
    """Run analyze_error_spectrum tests standalone"""
    print('='*80)
    print('RUNNING ANALYZE_ERROR_SPECTRUM TESTS')
    print('='*80)

    test_analyze_error_spectrum_basic()

    print('\n' + '='*80)
    print(f'** All analyze_error_spectrum tests completed! **')
    print(f'** Figures saved to: {output_dir.resolve()} **')
    print('='*80)
