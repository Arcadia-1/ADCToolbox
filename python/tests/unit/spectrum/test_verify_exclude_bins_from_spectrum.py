"""
Unit Test: Verify _exclude_bins_from_spectrum for spectrum preprocessing

Purpose: Self-verify that _exclude_bins_from_spectrum correctly removes
         specified bins from spectrum for noise calculation
"""
import numpy as np
from adctoolbox.spectrum._exclude_bins import _exclude_bins_from_spectrum


def test_verify_exclude_bins_from_spectrum_basic():
    """
    Verify _exclude_bins_from_spectrum removes specified bins from spectrum.

    Test strategy:
    1. Create spectrum with known bins
    2. Exclude DC and signal bins
    3. Assert: Excluded bins are zeroed, others preserved
    """
    N = 1024
    spectrum = np.ones(N)
    signal_bin = 100
    harmonic_bins = [100, 200, 300]
    side_bin = 2
    max_bin = N // 2

    result = _exclude_bins_from_spectrum(spectrum, signal_bin, harmonic_bins, side_bin, max_bin)

    print(f'\n[Verify Exclude Bins] [N={N}] [Signal={signal_bin}]')
    print(f'  [Original sum] {np.sum(spectrum):.1f}')
    print(f'  [Result sum  ] {np.sum(result):.1f}')
    print(f'  [Excluded at signal_bin] {result[signal_bin-2:signal_bin+3]}')

    # Check that signal bin and side bins are zeroed
    assert result[signal_bin] == 0, f"Signal bin {signal_bin} should be zero"
    assert np.all(result[signal_bin-side_bin:signal_bin+side_bin+1] == 0), "Side bins should be zero"

    # Check that DC (bin 0) is zeroed
    assert result[0] == 0, "DC bin should be zero"

    # Check that some other bins are preserved (beyond excluded region)
    assert result[max_bin - 10] == 1.0, "Bins beyond excluded region should be preserved"

    print(f'  [Status] PASS')


def test_verify_exclude_bins_from_spectrum_harmonics():
    """
    Verify _exclude_bins_from_spectrum excludes multiple harmonic bins.

    Test strategy:
    1. Create spectrum with distinct values
    2. Exclude fundamental and harmonics
    3. Assert: Correct bins are zeroed
    """
    spectrum = np.arange(1000, dtype=float)  # Values 0, 1, 2, ..., 999
    signal_bin = 100
    harmonic_bins = [100, 200, 300, 400]
    side_bin = 1
    max_bin = 500

    result = _exclude_bins_from_spectrum(spectrum, signal_bin, harmonic_bins, side_bin, max_bin)

    print(f'\n[Verify Exclude Harmonics]')
    print(f'  [Original values at 100, 200, 300] {spectrum[[100, 200, 300]]}')
    print(f'  [Result values at 100, 200, 300]   {result[[100, 200, 300]]}')

    # Check that harmonic bins are zeroed (skipping first one since it's same as signal)
    for h_bin in harmonic_bins[1:]:
        assert result[h_bin] == 0, f"Harmonic bin {h_bin} should be zero"

    print(f'  [Status] PASS')


if __name__ == '__main__':
    """Run verification tests standalone"""
    print('='*80)
    print('RUNNING EXCLUDE_BINS_FROM_SPECTRUM VERIFICATION TESTS')
    print('='*80)

    test_verify_exclude_bins_from_spectrum_basic()
    test_verify_exclude_bins_from_spectrum_harmonics()

    print('\n' + '='*80)
    print('** All exclude_bins_from_spectrum verification tests passed! **')
    print('='*80)
