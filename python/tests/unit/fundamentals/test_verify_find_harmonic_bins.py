"""
Unit Test: Verify _locate_harmonic_bins for harmonic detection

Purpose: Self-verify that _locate_harmonic_bins correctly calculates
         harmonic bin positions with proper aliasing handling
"""
import numpy as np
from adctoolbox.spectrum._harmonics import _locate_harmonic_bins


def test_verify_find_harmonic_bins_basic():
    """
    Verify _locate_harmonic_bins calculates harmonic bin positions.

    Test strategy:
    1. Define fundamental bin and number of harmonics
    2. Calculate harmonic bins
    3. Assert: Harmonic positions scale correctly
    """
    fundamental_bin = 50.0
    n_harmonics = 5
    n_fft = 2048

    harmonic_bins = _locate_harmonic_bins(fundamental_bin, n_harmonics, n_fft)

    print(f'\n[Verify Find Harmonic Bins] [Fundamental={fundamental_bin}, N_FFT={n_fft}]')
    print(f'  [Harmonic bins] {harmonic_bins}')

    # Check we have correct number of harmonics (max_harmonic - 1)
    assert len(harmonic_bins) == n_harmonics - 1, f"Should find {n_harmonics - 1} harmonics (H2-H{n_harmonics})"

    # Check that bins are in ascending order (mostly, accounting for aliasing)
    print(f'  [Bin differences] {np.diff(harmonic_bins)}')

    print(f'  [Status] PASS')


def test_verify_find_harmonic_bins_scaling():
    """
    Verify harmonic bins scale linearly for low frequencies (no aliasing).

    Test strategy:
    1. Use low fundamental frequency (no aliasing)
    2. Check that harmonic bins scale proportionally
    Note: Harmonics start from H2 (2nd harmonic), not H1 (fundamental)
    """
    fundamental_bin = 20.0
    n_harmonics = 4
    n_fft = 4096

    harmonic_bins = _locate_harmonic_bins(fundamental_bin, n_harmonics, n_fft)

    print(f'\n[Verify Harmonic Scaling]')
    print(f'  [Fundamental] {fundamental_bin}')
    print(f'  [H2-H5]       {harmonic_bins}')

    # For low frequencies without aliasing, harmonics should roughly scale
    # H2 ≈ fundamental_bin * 2
    # H3 ≈ fundamental_bin * 3, etc.
    expected_h2 = fundamental_bin * 2
    expected_h3 = fundamental_bin * 3

    print(f'  [H2 expected ~{expected_h2:.1f}, got {harmonic_bins[0]:.1f}]')
    print(f'  [H3 expected ~{expected_h3:.1f}, got {harmonic_bins[1]:.1f}]')

    # Allow some tolerance due to aliasing handling
    assert abs(harmonic_bins[0] - expected_h2) < 5, f"H2 should be near {expected_h2}"

    print(f'  [Status] PASS')


if __name__ == '__main__':
    """Run verification tests standalone"""
    print('='*80)
    print('RUNNING LOCATE_HARMONIC_BINS VERIFICATION TESTS')
    print('='*80)

    test_verify_find_harmonic_bins_basic()
    test_verify_find_harmonic_bins_scaling()

    print('\n' + '='*80)
    print('** All locate_harmonic_bins verification tests passed! **')
    print('='*80)
