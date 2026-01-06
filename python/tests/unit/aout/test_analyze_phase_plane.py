"""Unit tests for phase plane analysis with figure generation."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency
from adctoolbox.siggen import ADC_Signal_Generator
from adctoolbox.aout import analyze_phase_plane


# Create output directory for test figures
output_dir = Path(__file__).parent / "test_output"
output_dir.mkdir(exist_ok=True)


def test_analyze_phase_plane_basic():
    """Test phase plane analysis for different ADC non-idealities."""
    # Setup
    N = 2**14
    Fs = 800e6
    Fin_target = 97e6
    Fin, Fin_bin = find_coherent_frequency(Fs, Fin_target, N)
    A = 0.49
    DC = 0.5
    B = 12

    gen = ADC_Signal_Generator(N=N, Fs=Fs, Fin=Fin, A=A, DC=DC)

    print(f"\n[Config] Fs={Fs/1e6:.0f} MHz, Fin={Fin/1e6:.1f} MHz, Bin={Fin_bin}, N={N}")
    print(f"[Config] A={A:.3f} V, DC={DC:.3f} V, Resolution={B} bits\n")

    # Create test cases
    sig_thermal = gen.apply_thermal_noise(noise_rms=180e-6)
    sig_glitch = gen.apply_thermal_noise(gen.apply_glitch(None, glitch_prob=0.002, glitch_amplitude=0.1), noise_rms=10e-6)
    sig_quantization = gen.apply_thermal_noise(gen.apply_quantization_noise(None, n_bits=10, quant_range=(0, 1)), noise_rms=10e-6)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Analyze each case
    result1 = analyze_phase_plane(sig_thermal, lag='auto', fs=Fs, detect_outliers=True,
                                  threshold=4.0, ax=axes[0], title='Thermal Noise', create_plot=False)
    result2 = analyze_phase_plane(sig_glitch, lag='auto', fs=Fs, detect_outliers=True,
                                  threshold=4.0, ax=axes[1], title='Glitch', create_plot=False)
    result3 = analyze_phase_plane(sig_quantization, lag='auto', fs=Fs, detect_outliers=True,
                                  threshold=4.0, ax=axes[2], title='Quantization Noise', create_plot=False)

    print(f"Thermal Noise    : Lag={result1['lag']:3d}, Outliers={len(result1['outliers']):5d}")
    print(f"Glitch           : Lag={result2['lag']:3d}, Outliers={len(result2['outliers']):5d}")
    print(f"Quantization     : Lag={result3['lag']:3d}, Outliers={len(result3['outliers']):5d}")

    fig.suptitle(f'Phase Plane Analysis: ADC Non-idealities (Fs={Fs/1e6:.0f} MHz, Fin={Fin/1e6:.1f} MHz, {B}-bit)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    fig_path = output_dir / 'test_analyze_phase_plane.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n[Save fig] -> [{fig_path.resolve()}]\n")

    # Verify file was created
    assert fig_path.exists(), f"Figure file not created: {fig_path}"
    assert fig_path.stat().st_size > 0, f"Figure file is empty: {fig_path}"

    # Basic assertions
    assert 'lag' in result1, "Result should contain lag"
    assert 'outliers' in result1, "Result should contain outliers"


if __name__ == '__main__':
    """Run analyze_phase_plane tests standalone"""
    print('='*80)
    print('RUNNING ANALYZE_PHASE_PLANE TESTS')
    print('='*80)

    test_analyze_phase_plane_basic()

    print('\n' + '='*80)
    print(f'** All analyze_phase_plane tests completed! **')
    print(f'** Figures saved to: {output_dir.resolve()} **')
    print('='*80)
