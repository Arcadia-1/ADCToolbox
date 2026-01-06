"""Unit tests for error autocorrelation analysis with figure generation."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import analyze_error_autocorr, find_coherent_frequency
from adctoolbox.siggen import ADC_Signal_Generator


# Create output directory for test figures
output_dir = Path(__file__).parent / "test_output"
output_dir.mkdir(exist_ok=True)


def test_analyze_error_autocorrelation_basic():
    """Test error autocorrelation analysis for different noise types."""
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
    sig_memory = gen.apply_thermal_noise(gen.apply_memory_effect(None, memory_strength=0.009), noise_rms=10e-6)
    sig_drift = gen.apply_thermal_noise(gen.apply_drift(None, drift_scale=5e-5), noise_rms=10e-6)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Analyze each case
    result1 = analyze_error_autocorr(sig_thermal, frequency=Fin/Fs, max_lag=100, normalize=True,
                                     create_plot=True, ax=axes[0], title='Thermal Noise')
    result2 = analyze_error_autocorr(sig_memory, frequency=Fin/Fs, max_lag=100, normalize=True,
                                     create_plot=True, ax=axes[1], title='Memory Effect')
    result3 = analyze_error_autocorr(sig_drift, frequency=Fin/Fs, max_lag=100, normalize=True,
                                     create_plot=True, ax=axes[2], title='Drift')

    # Extract ACF values
    for idx, (result, title) in enumerate([(result1, 'Thermal'), (result2, 'Memory'), (result3, 'Drift')]):
        acf = result['acf']
        lags = result['lags']
        acf_0 = acf[lags==0][0]
        acf_1 = acf[lags==1][0] if len(acf[lags==1]) > 0 else 0
        acf_10 = acf[lags==10][0] if len(acf[lags==10]) > 0 else 0
        print(f"{title:15s}: ACF[0]={acf_0:8.4f}, ACF[1]={acf_1:8.4f}, ACF[10]={acf_10:8.4f}")

    fig.suptitle(f'Error Autocorrelation Analysis: ADC Non-idealities (Fs={Fs/1e6:.0f} MHz, Fin={Fin/1e6:.1f} MHz)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    fig_path = output_dir / 'test_analyze_error_autocorrelation.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n[Save fig] -> [{fig_path.resolve()}]\n")

    # Verify file was created
    assert fig_path.exists(), f"Figure file not created: {fig_path}"
    assert fig_path.stat().st_size > 0, f"Figure file is empty: {fig_path}"


if __name__ == '__main__':
    """Run analyze_error_autocorrelation tests standalone"""
    print('='*80)
    print('RUNNING ANALYZE_ERROR_AUTOCORRELATION TESTS')
    print('='*80)

    test_analyze_error_autocorrelation_basic()

    print('\n' + '='*80)
    print(f'** All analyze_error_autocorrelation tests completed! **')
    print(f'** Figures saved to: {output_dir.resolve()} **')
    print('='*80)
