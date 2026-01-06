"""Unit tests for analyze_error_by_value with figure generation."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import analyze_error_by_value


# Create output directory for test figures
output_dir = Path(__file__).parent / "test_output"
output_dir.mkdir(exist_ok=True)


def test_analyze_error_by_value_thermal_vs_nonlinearity():
    """Test value error analysis to distinguish thermal noise from static nonlinearity."""
    # Setup parameters
    N = 2**13
    Fs = 800e6
    Fin = 10.1234567e6
    t = np.arange(N) / Fs
    A = 0.49
    DC = 0.5
    base_noise = 50e-6

    print(f"\n[Config] Fs={Fs/1e6:.0f} MHz, Fin={Fin/1e6:.2f} MHz, N={N}")

    # Case 1: Ideal ADC with Thermal Noise
    sig_noise = A * np.sin(2 * np.pi * Fin * t) + DC + np.random.randn(N) * base_noise

    # Case 2: ADC with 3rd Order Nonlinearity
    k3 = 0.01
    sig_nonlin = A * np.sin(2 * np.pi * Fin * t) + DC + k3 * (A * np.sin(2 * np.pi * Fin * t))**3 + np.random.randn(N) * base_noise

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle('Value Error Analysis - Thermal Noise vs 3rd Order Nonlinearity', fontsize=14, fontweight='bold')

    # Analyze each case
    analyze_error_by_value(sig_noise, n_bins=16, ax=axes[0], title='Thermal Noise Only')
    analyze_error_by_value(sig_nonlin, n_bins=16, ax=axes[1], title='3rd Order Nonlinearity (16 bins)')
    analyze_error_by_value(sig_nonlin, n_bins=64, ax=axes[2], title='3rd Order Nonlinearity (64 bins)')

    plt.tight_layout()

    fig_path = output_dir / 'test_analyze_error_by_value_bins.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"[Save fig] -> [{fig_path.resolve()}]")

    # Verify file was created
    assert fig_path.exists(), f"Figure file not created: {fig_path}"
    assert fig_path.stat().st_size > 0, f"Figure file is empty: {fig_path}"


@pytest.mark.parametrize("n_bins", [8, 16, 32, 64])
def test_analyze_error_by_value_bin_count(n_bins):
    """Test analyze_error_by_value with different bin counts."""
    N = 2**13
    Fs = 800e6
    Fin = 10.1234567e6
    t = np.arange(N) / Fs
    A = 0.49
    DC = 0.5
    base_noise = 50e-6
    k3 = 0.01

    sig_nonlin = A * np.sin(2 * np.pi * Fin * t) + DC + k3 * (A * np.sin(2 * np.pi * Fin * t))**3 + np.random.randn(N) * base_noise

    # Create a simple plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    analyze_error_by_value(sig_nonlin, n_bins=n_bins, ax=ax, title=f'Nonlinearity ({n_bins} bins)')
    plt.tight_layout()
    plt.close(fig)

    print(f"\n[Bin Count={n_bins}] -> [PASS]")


def test_analyze_error_by_value_different_nonlinearities():
    """Test analyze_error_by_value with different types of nonlinearities."""
    N = 2**13
    Fs = 800e6
    Fin = 10.1234567e6
    t = np.arange(N) / Fs
    A = 0.49
    DC = 0.5
    base_noise = 50e-6

    # Case 1: Thermal only
    sig_thermal = A * np.sin(2 * np.pi * Fin * t) + DC + np.random.randn(N) * base_noise

    # Case 2: 2nd order nonlinearity
    k2 = 0.01
    sig_k2 = A * np.sin(2 * np.pi * Fin * t) + DC + k2 * (A * np.sin(2 * np.pi * Fin * t))**2 + np.random.randn(N) * base_noise

    # Case 3: 3rd order nonlinearity
    k3 = 0.01
    sig_k3 = A * np.sin(2 * np.pi * Fin * t) + DC + k3 * (A * np.sin(2 * np.pi * Fin * t))**3 + np.random.randn(N) * base_noise

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle('Value Error Analysis - Different Nonlinearity Types', fontsize=14, fontweight='bold')

    analyze_error_by_value(sig_thermal, n_bins=32, ax=axes[0], title='Thermal Only')
    analyze_error_by_value(sig_k2, n_bins=32, ax=axes[1], title='2nd Order Nonlinearity')
    analyze_error_by_value(sig_k3, n_bins=32, ax=axes[2], title='3rd Order Nonlinearity')

    plt.tight_layout()

    fig_path = output_dir / 'test_analyze_error_by_value_nonlinearity_types.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"[Save fig] -> [{fig_path.resolve()}]")

    # Verify file was created
    assert fig_path.exists(), f"Figure file not created: {fig_path}"
    assert fig_path.stat().st_size > 0, f"Figure file is empty: {fig_path}"


if __name__ == '__main__':
    """Run analyze_error_by_value tests standalone"""
    print('='*80)
    print('RUNNING ANALYZE_ERROR_BY_VALUE TESTS')
    print('='*80)

    test_analyze_error_by_value_thermal_vs_nonlinearity()
    test_analyze_error_by_value_bin_count(8)
    test_analyze_error_by_value_bin_count(16)
    test_analyze_error_by_value_bin_count(32)
    test_analyze_error_by_value_bin_count(64)
    test_analyze_error_by_value_different_nonlinearities()

    print('\n' + '='*80)
    print(f'** All analyze_error_by_value tests completed! **')
    print(f'** Figures saved to: {output_dir.resolve()} **')
    print('='*80)
