"""Unit tests for harmonic decomposition time-domain analysis with figure generation."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import analyze_decomposition_time


# Create output directory for test figures
output_dir = Path(__file__).parent / "test_output"
output_dir.mkdir(exist_ok=True)


def test_decompose_harmonics_basic():
    """Test harmonic decomposition for thermal noise vs nonlinearity vs glitches."""
    # Setup parameters
    N = 2**13
    Fs = 800e6
    Fin = 10.1234567e6
    t = np.arange(N) / Fs
    A = 0.25
    DC = 0.5
    base_noise = 50e-6

    sig_ac = A * np.sin(2 * np.pi * Fin * t)
    sig_ideal = sig_ac + DC
    print(f"\n[Config] Fs={Fs/1e6:.0f} MHz, Fin={Fin/1e6:.2f} MHz, N={N}")

    # Case 1: Ideal ADC with Thermal Noise
    sig_noise = sig_ideal + np.random.randn(N) * base_noise

    # Case 2: ADC with Nonlinearity
    k2 = 0.001
    k3 = 0.005
    sig_nonlin = DC + sig_ac + k2 * sig_ac**2 + k3 * sig_ac**3 + np.random.randn(N) * base_noise

    # Case 3: ADC with Glitches
    glitch_prob = 0.01
    glitch_amplitude = 0.1
    glitch_mask = np.random.rand(N) < glitch_prob
    glitch = glitch_mask * glitch_amplitude
    sig_glitch = sig_ideal + glitch + np.random.randn(N) * base_noise

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle('Harmonic Decomposition - Thermal Noise vs Nonlinearity vs Glitches',
                 fontsize=14, fontweight='bold')

    analyze_decomposition_time(sig_noise, ax=axes[0], title='Thermal Noise Only')
    analyze_decomposition_time(sig_nonlin, ax=axes[1], title=f'Nonlinearity (k2={k2:.3f}, k3={k3:.3f})')
    analyze_decomposition_time(sig_glitch, ax=axes[2], title=f'Glitches (prob={glitch_prob*100:.2f}%, amp={glitch_amplitude:.1f})')

    plt.tight_layout()

    fig_path = output_dir / 'test_decompose_harmonics.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"[Save fig] -> [{fig_path.resolve()}]\n")

    # Verify file was created
    assert fig_path.exists(), f"Figure file not created: {fig_path}"
    assert fig_path.stat().st_size > 0, f"Figure file is empty: {fig_path}"


@pytest.mark.parametrize("k3_value", [0.001, 0.005, 0.01])
def test_decompose_harmonics_nonlinearity_levels(k3_value):
    """Test harmonic decomposition at different nonlinearity levels."""
    N = 2**13
    Fs = 800e6
    Fin = 10.1234567e6
    t = np.arange(N) / Fs
    A = 0.25
    DC = 0.5
    base_noise = 50e-6

    sig_ac = A * np.sin(2 * np.pi * Fin * t)
    sig_nonlin = DC + sig_ac + k3_value * sig_ac**3 + np.random.randn(N) * base_noise

    # Just run the analysis without creating plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    analyze_decomposition_time(sig_nonlin, ax=ax, title=f'k3={k3_value:.3f}')
    plt.tight_layout()
    plt.close(fig)

    print(f"\n[k3={k3_value:.3f}] -> [PASS]")


if __name__ == '__main__':
    """Run decompose_harmonics tests standalone"""
    print('='*80)
    print('RUNNING DECOMPOSE_HARMONICS TESTS')
    print('='*80)

    test_decompose_harmonics_basic()
    test_decompose_harmonics_nonlinearity_levels(0.001)
    test_decompose_harmonics_nonlinearity_levels(0.005)
    test_decompose_harmonics_nonlinearity_levels(0.01)

    print('\n' + '='*80)
    print(f'** All decompose_harmonics tests completed! **')
    print(f'** Figures saved to: {output_dir.resolve()} **')
    print('='*80)
