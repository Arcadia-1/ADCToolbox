"""Unit tests for residual phase plane analysis with figure generation."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency
from adctoolbox.siggen import ADC_Signal_Generator
from adctoolbox.aout import analyze_error_phase_plane


# Create output directory for test figures
output_dir = Path(__file__).parent / "test_output"
output_dir.mkdir(exist_ok=True)


def test_analyze_error_phase_plane_basic():
    """Test residual phase plane analysis for different ADC non-idealities."""
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

    # Create test cases with different harmonics
    hd2_amp = 10**(-80/20)
    hd3_amp = 10**(-70/20)
    k2 = (2 * hd2_amp) / A
    k3 = (4 * hd3_amp) / (A**2)

    sig_thermal = gen.apply_thermal_noise(noise_rms=180e-6)
    sig_hd2 = gen.apply_thermal_noise(gen.apply_static_nonlinearity(None, k2=k2, k3=0), noise_rms=10e-6)
    sig_hd3 = gen.apply_thermal_noise(gen.apply_static_nonlinearity(None, k2=0, k3=k3), noise_rms=10e-6)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Analyze each case
    result1 = analyze_error_phase_plane(sig_thermal, fs=Fs, ax=axes[0], title='Thermal Noise',
                                        create_plot=False, fit_polynomial_order=3)
    result2 = analyze_error_phase_plane(sig_hd2, fs=Fs, ax=axes[1], title='Static HD2 (-80 dBc)',
                                        create_plot=False, fit_polynomial_order=3)
    result3 = analyze_error_phase_plane(sig_hd3, fs=Fs, ax=axes[2], title='Static HD3 (-70 dBc)',
                                        create_plot=False, fit_polynomial_order=3)

    # Print metrics
    rms1_uV = result1['residual'].std() * 1e6
    peak1_uV = abs(result1['residual']).max() * 1e6
    rms2_uV = result2['residual'].std() * 1e6
    peak2_uV = abs(result2['residual']).max() * 1e6
    rms3_uV = result3['residual'].std() * 1e6
    peak3_uV = abs(result3['residual']).max() * 1e6

    print(f"Thermal Noise    : RMS={rms1_uV:8.1f} uV, Peak={peak1_uV:8.1f} uV")
    print(f"Static HD2       : RMS={rms2_uV:8.1f} uV, Peak={peak2_uV:8.1f} uV")
    print(f"Static HD3       : RMS={rms3_uV:8.1f} uV, Peak={peak3_uV:8.1f} uV")

    fig.suptitle(f'Residual Phase Plane Analysis: ADC Non-idealities (Fs={Fs/1e6:.0f} MHz, Fin={Fin/1e6:.1f} MHz, {B}-bit)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    fig_path = output_dir / 'test_analyze_error_phase_plane.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n[Save fig] -> [{fig_path.resolve()}]\n")

    # Verify file was created
    assert fig_path.exists(), f"Figure file not created: {fig_path}"
    assert fig_path.stat().st_size > 0, f"Figure file is empty: {fig_path}"

    # Basic assertions
    assert 'residual' in result1, "Result should contain residual"
    assert 'fitted_sine' in result1, "Result should contain fitted_sine"


if __name__ == '__main__':
    """Run analyze_error_phase_plane tests standalone"""
    print('='*80)
    print('RUNNING ANALYZE_ERROR_PHASE_PLANE TESTS')
    print('='*80)

    test_analyze_error_phase_plane_basic()

    print('\n' + '='*80)
    print(f'** All analyze_error_phase_plane tests completed! **')
    print(f'** Figures saved to: {output_dir.resolve()} **')
    print('='*80)
