"""Unit tests for error PDF analysis with figure generation."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import analyze_error_pdf, find_coherent_frequency
from adctoolbox.siggen import ADC_Signal_Generator


# Create output directory for test figures
output_dir = Path(__file__).parent / "test_output"
output_dir.mkdir(exist_ok=True)


def _assert_pdf_panel(ax, result, title, n_samples):
    assert ax.get_title() == title
    assert ax.get_xlabel() == 'Error (LSB)'
    assert ax.get_ylabel() == 'Probability Density'
    assert len(ax.lines) == 2
    assert [line.get_label() for line in ax.lines] == ['Actual PDF (KDE)', 'Gaussian Fit']
    assert ax.get_legend() is not None
    assert len(ax.texts) == 1

    assert result['err_lsb'].shape == (n_samples,)
    assert result['x'].shape == result['pdf'].shape == result['gauss_pdf'].shape == (200,)
    assert np.all(np.isfinite(result['pdf']))
    assert np.all(np.isfinite(result['gauss_pdf']))
    assert np.isfinite(result['mu'])
    assert result['sigma'] > 0
    assert result['kl_divergence'] >= 0


def test_analyze_error_pdf_basic():
    """Test error PDF analysis for thermal noise and nonlinearity."""
    # Setup
    N = 2**14
    Fs = 800e6
    Fin_target = 97e6
    Fin, Fin_bin = find_coherent_frequency(Fs, Fin_target, N)
    A = 0.49
    DC = 0.5
    B = 12  # ADC resolution

    gen = ADC_Signal_Generator(N=N, Fs=Fs, Fin=Fin, A=A, DC=DC)

    print(f"\n[Config] Fs={Fs/1e6:.0f} MHz, Fin={Fin/1e6:.1f} MHz, Bin={Fin_bin}, N={N}")
    print(f"[Config] A={A:.3f} V, DC={DC:.3f} V, Resolution={B} bits")

    # Create 3 test cases
    sig_thermal = gen.apply_thermal_noise(noise_rms=180e-6)
    sig_quantization = gen.apply_quantization_noise(None, n_bits=10, quant_range=(0, 1))
    sig_nonlin = gen.apply_static_nonlinearity(None, k2=0.01, k3=0.01)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Analyze each case
    result1 = analyze_error_pdf(sig_thermal, resolution=B, ax=axes[0], title='Thermal Noise')
    result2 = analyze_error_pdf(sig_quantization, resolution=B, ax=axes[1], title='Quantization Noise')
    result3 = analyze_error_pdf(sig_nonlin, resolution=B, ax=axes[2], title='Static Nonlinearity')

    print(f"\nThermal Noise    : mu={result1['mu']:6.3f} LSB, sigma={result1['sigma']:6.3f} LSB, KL={result1['kl_divergence']:6.4f}")
    print(f"Quantization     : mu={result2['mu']:6.3f} LSB, sigma={result2['sigma']:6.3f} LSB, KL={result2['kl_divergence']:6.4f}")
    print(f"Static Nonlin    : mu={result3['mu']:6.3f} LSB, sigma={result3['sigma']:6.3f} LSB, KL={result3['kl_divergence']:6.4f}")

    fig.suptitle(f'Error PDF Analysis: ADC Non-idealities (Fs={Fs/1e6:.0f} MHz, Fin={Fin/1e6:.1f} MHz, {B}-bit)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    fig_path = output_dir / 'test_analyze_error_pdf.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')

    print(f"\n[Save fig] -> [{fig_path.resolve()}]\n")

    # Verify file was created
    assert fig_path.exists(), f"Figure file not created: {fig_path}"
    assert fig_path.stat().st_size > 0, f"Figure file is empty: {fig_path}"

    # Basic assertions
    assert 'mu' in result1, "Result should contain mu"
    assert 'sigma' in result1, "Result should contain sigma"
    assert 'kl_divergence' in result1, "Result should contain kl_divergence"

    assert len(fig.axes) == 3
    for ax, result, title in [
        (axes[0], result1, 'Thermal Noise'),
        (axes[1], result2, 'Quantization Noise'),
        (axes[2], result3, 'Static Nonlinearity'),
    ]:
        _assert_pdf_panel(ax, result, title, N)
    plt.close(fig)


if __name__ == '__main__':
    """Run analyze_error_pdf tests standalone"""
    print('='*80)
    print('RUNNING ANALYZE_ERROR_PDF TESTS')
    print('='*80)

    test_analyze_error_pdf_basic()

    print('\n' + '='*80)
    print(f'** All analyze_error_pdf tests completed! **')
    print(f'** Figures saved to: {output_dir.resolve()} **')
    print('='*80)
