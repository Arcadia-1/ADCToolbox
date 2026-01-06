"""Unit tests for spectrum analysis with figure generation."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import analyze_spectrum, find_coherent_frequency
from adctoolbox.siggen import ADC_Signal_Generator


# Create output directory for test figures
output_dir = Path(__file__).parent / "test_output"
output_dir.mkdir(exist_ok=True)


def test_analyze_spectrum_basic():
    """Test spectrum analysis for different ADC impairments."""
    # Setup
    N = 2**14
    Fs = 800e6
    Fin_target = 97e6
    Fin, Fin_bin = find_coherent_frequency(Fs, Fin_target, N)
    A = 0.49
    DC = 0.5
    adc_range = [0, 1]

    gen = ADC_Signal_Generator(N=N, Fs=Fs, Fin=Fin, A=A, DC=DC)

    print(f"\n[Config] Fs={Fs/1e6:.0f} MHz, Fin={Fin/1e6:.1f} MHz, Bin={Fin_bin}, N={N}")
    print(f"[Config] A={A:.3f} V, DC={DC:.3f} V, ADC Range={adc_range}")

    # Create test cases
    sig_thermal = gen.apply_thermal_noise(noise_rms=180e-6)
    sig_hd2 = gen.apply_thermal_noise(gen.apply_static_nonlinearity(None, k2=0.01, k3=0), noise_rms=10e-6)
    sig_hd3 = gen.apply_thermal_noise(gen.apply_static_nonlinearity(None, k2=0, k3=0.01), noise_rms=10e-6)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Analyze each case
    plt.sca(axes[0])
    result1 = analyze_spectrum(sig_thermal, fs=Fs, max_scale_range=adc_range)
    axes[0].set_title('Thermal Noise', fontsize=10, fontweight='bold')

    plt.sca(axes[1])
    result2 = analyze_spectrum(sig_hd2, fs=Fs, max_scale_range=adc_range)
    axes[1].set_title('Static HD2', fontsize=10, fontweight='bold')

    plt.sca(axes[2])
    result3 = analyze_spectrum(sig_hd3, fs=Fs, max_scale_range=adc_range)
    axes[2].set_title('Static HD3', fontsize=10, fontweight='bold')

    print(f"\nThermal Noise: SFDR={result1['sfdr_dbc']:6.2f} dB, SNDR={result1['sndr_dbc']:6.2f} dB, THD={result1['thd_dbc']:6.2f} dB")
    print(f"Static HD2   : SFDR={result2['sfdr_dbc']:6.2f} dB, SNDR={result2['sndr_dbc']:6.2f} dB, THD={result2['thd_dbc']:6.2f} dB")
    print(f"Static HD3   : SFDR={result3['sfdr_dbc']:6.2f} dB, SNDR={result3['sndr_dbc']:6.2f} dB, THD={result3['thd_dbc']:6.2f} dB")

    fig.suptitle(f'Spectrum Analysis: ADC Non-idealities (Fs={Fs/1e6:.0f} MHz, Fin={Fin/1e6:.1f} MHz)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    fig_path = output_dir / 'test_analyze_spectrum.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n[Save fig] -> [{fig_path.resolve()}]\n")

    # Verify file was created
    assert fig_path.exists(), f"Figure file not created: {fig_path}"
    assert fig_path.stat().st_size > 0, f"Figure file is empty: {fig_path}"

    # Basic assertions
    assert 'sfdr_dbc' in result1, "Result should contain sfdr_dbc"
    assert 'sndr_dbc' in result1, "Result should contain sndr_dbc"
    assert 'thd_dbc' in result1, "Result should contain thd_dbc"


if __name__ == '__main__':
    """Run analyze_spectrum tests standalone"""
    print('='*80)
    print('RUNNING ANALYZE_SPECTRUM TESTS')
    print('='*80)

    test_analyze_spectrum_basic()

    print('\n' + '='*80)
    print(f'** All analyze_spectrum tests completed! **')
    print(f'** Figures saved to: {output_dir.resolve()} **')
    print('='*80)
