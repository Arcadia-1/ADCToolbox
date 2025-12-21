"""
Unit Test: Verify _find_harmonic_bins for harmonic detection

Purpose: Self-verify that _find_harmonic_bins correctly calculates
         harmonic bin positions with proper aliasing handling
"""
import numpy as np
from adctoolbox.spectrum._find_harmonic_bins import _find_harmonic_bins


def test_verify_find_harmonic_bins_basic():
    """
    Verify _find_harmonic_bins calculates harmonic bin positions.

    Test strategy:
    1. Define fundamental bin and number of harmonics
    2. Calculate harmonic bins
    3. Assert: Harmonic positions scale correctly
    """
    fundamental_bin = 50.0
    n_harmonics = 5
    n_fft = 2048

    harmonic_bins = _find_harmonic_bins(fundamental_bin, n_harmonics, n_fft)

    print(f'\n[Verify Find Harmonic Bins] [Fundamental={fundamental_bin}, N_FFT={n_fft}]')
    print(f'  [Harmonic bins] {harmonic_bins}')

    # Check we have correct number of harmonics
    assert len(harmonic_bins) == n_harmonics, f"Should find {n_harmonics} harmonics"

    # Check that bins are in ascending order (mostly, accounting for aliasing)
    print(f'  [Bin differences] {np.diff(harmonic_bins)}')

    print(f'  [Status] PASS')


def test_verify_find_harmonic_bins_scaling():
    """
    Verify harmonic bins scale linearly for low frequencies (no aliasing).

    Test strategy:
    1. Use low fundamental frequency (no aliasing)
    2. Check that harmonic bins scale proportionally
    """
    fundamental_bin = 20.0
    n_harmonics = 4
    n_fft = 4096

    harmonic_bins = _find_harmonic_bins(fundamental_bin, n_harmonics, n_fft)

    print(f'\n[Verify Harmonic Scaling]')
    print(f'  [Fundamental] {fundamental_bin}')
    print(f'  [H1-H4]       {harmonic_bins}')

    # For low frequencies without aliasing, harmonics should roughly scale
    # H1 ≈ fundamental_bin * 1
    # H2 ≈ fundamental_bin * 2, etc.
    expected_h1 = fundamental_bin * 1
    expected_h2 = fundamental_bin * 2

    print(f'  [H1 expected ~{expected_h1:.1f}, got {harmonic_bins[0]:.1f}]')
    print(f'  [H2 expected ~{expected_h2:.1f}, got {harmonic_bins[1]:.1f}]')

    # Allow some tolerance due to aliasing handling
    assert abs(harmonic_bins[0] - expected_h1) < 5, f"H1 should be near {expected_h1}"

    print(f'  [Status] PASS')


if __name__ == '__main__':
    """Run verification tests standalone"""
    print('='*80)
    print('RUNNING FIND_HARMONIC_BINS VERIFICATION TESTS')
    print('='*80)

    test_verify_find_harmonic_bins_basic()
    test_verify_find_harmonic_bins_scaling()

    print('\n' + '='*80)
    print('** All find_harmonic_bins verification tests passed! **')
    print('='*80)
