"""Unit tests for bin location functionality in compute_spectrum."""

import pytest
import numpy as np
from adctoolbox.spectrum.compute_spectrum import compute_spectrum
from adctoolbox.fundamentals.frequency import fold_bin_to_nyquist


@pytest.mark.parametrize("bin_target", [1, 100, 1000])
@pytest.mark.parametrize("side_bin", [0, 1])
@pytest.mark.parametrize("max_harmonic", [10])
def test_locate_bins(bin_target, side_bin, max_harmonic):
    """Test bin location (fundamental, signal range, harmonics) with various parameters."""
    noise_rms = 100e-6
    n_fft = 2**13
    fs = 1e6
    osr = 1

    f_sig = bin_target * fs / n_fft
    t = np.arange(n_fft) / fs
    signal = np.sin(2 * np.pi * f_sig * t) + np.random.randn(n_fft) * noise_rms

    result = compute_spectrum(signal, fs=fs, osr=osr, side_bin=side_bin, max_harmonic=max_harmonic)

    # Test fundamental bin location
    fundamental_bin = result['plot_data']['fundamental_bin']
    fundamental_bin_fractional = result['plot_data']['fundamental_bin_fractional']

    # Test signal bin range
    sig_bin_start = result['plot_data']['sig_bin_start']
    sig_bin_end = result['plot_data']['sig_bin_end']
    expected_start = max(fundamental_bin - side_bin, 0)
    expected_end = min(fundamental_bin + side_bin + 1, n_fft // 2 // osr)

    # Test harmonic bins
    harmonic_bins = result['plot_data']['harmonic_bins']

    # Calculate expected harmonic bins (H2, H3, ..., H(max_harmonic))
    # Note: Harmonics start from H2 (fundamental is H1, not included)
    # max_harmonic=5 means H2, H3, H4, H5 (4 values)
    expected_harmonic_bins = np.zeros(max_harmonic - 1, dtype=int)
    for k in range(max_harmonic - 1):
        h_order = k + 2  # Harmonic order: 2, 3, 4, 5, ...
        h_bin = h_order * fundamental_bin_fractional
        # Handle aliasing using fold_bin_to_nyquist (same as _locate_harmonic_bins)
        aliased_bin = fold_bin_to_nyquist(h_bin, n_fft)
        expected_harmonic_bins[k] = round(aliased_bin)

    harmonic_bins_str = '[' + ', '.join(f'{h:3d}' for h in harmonic_bins) + ']'
    expected_harmonic_bins_str = '[' + ', '.join(f'{h:3d}' for h in expected_harmonic_bins) + ']'

    print(f"\n[Bin Location Test] bin_target={bin_target}, side_bin={side_bin}, max_harmonic={max_harmonic}")
    print(f"Fundamental: Expected=[{bin_target}], Computed=[{fundamental_bin}], Fractional=[{fundamental_bin_fractional:.4f}], Signal Range=[{sig_bin_start}:{sig_bin_end}]")
    print(f"Harmonics: Count=[{len(harmonic_bins)}], from HD2 to HD{max_harmonic}:")
    print(f"  Expected={expected_harmonic_bins_str}")
    print(f"  Computed={harmonic_bins_str}")

    assert fundamental_bin == bin_target
    assert abs(fundamental_bin_fractional - bin_target) < 0.5
    assert sig_bin_start == expected_start
    assert sig_bin_end == expected_end
    assert len(harmonic_bins) == max_harmonic - 1
    assert isinstance(harmonic_bins, np.ndarray)
    assert harmonic_bins.dtype == np.dtype('int64') or harmonic_bins.dtype == np.dtype('int32')
    np.testing.assert_array_equal(harmonic_bins, expected_harmonic_bins)


@pytest.mark.parametrize("fin", [1e3, 2e3, 3e3, 4e3, 5e3])  # 1kHz to 5kHz
@pytest.mark.parametrize("max_harmonic", [5, 10])
def test_locate_bins_noncoherent(fin, max_harmonic):
    """Test bin location with non-coherent frequencies (real-world frequencies)."""
    noise_rms = 100e-6
    n_fft = 2**13
    fs = 1e6
    osr = 1
    side_bin = 1

    # Generate signal at non-coherent frequency
    t = np.arange(n_fft) / fs
    signal = np.sin(2 * np.pi * fin * t) + np.random.randn(n_fft) * noise_rms

    result = compute_spectrum(signal, fs=fs, osr=osr, side_bin=side_bin, max_harmonic=max_harmonic)

    # Test fundamental bin location
    fundamental_bin = result['plot_data']['fundamental_bin']
    fundamental_bin_fractional = result['plot_data']['fundamental_bin_fractional']

    # Calculate expected bin (fractional)
    expected_bin_fractional = fin * n_fft / fs
    bin_error = abs(fundamental_bin_fractional - expected_bin_fractional)

    # Test signal bin range
    sig_bin_start = result['plot_data']['sig_bin_start']
    sig_bin_end = result['plot_data']['sig_bin_end']

    # Test harmonic bins
    harmonic_bins = result['plot_data']['harmonic_bins']

    # Calculate expected harmonic bins
    expected_harmonic_bins = np.zeros(max_harmonic - 1, dtype=int)
    for k in range(max_harmonic - 1):
        h_order = k + 2  # Harmonic order: 2, 3, 4, 5, ...
        h_bin = h_order * fundamental_bin_fractional
        # Handle aliasing using fold_bin_to_nyquist (same as _locate_harmonic_bins)
        aliased_bin = fold_bin_to_nyquist(h_bin, n_fft)
        expected_harmonic_bins[k] = round(aliased_bin)

    harmonic_bins_str = '[' + ', '.join(f'{h:3d}' for h in harmonic_bins) + ']'
    expected_harmonic_bins_str = '[' + ', '.join(f'{h:3d}' for h in expected_harmonic_bins) + ']'

    print(f"\n[Bin Location Test] fin={fin/1e3:.1f} kHz, side_bin={side_bin}, max_harmonic={max_harmonic}")
    print(f"Fundamental: Expected=[{expected_bin_fractional:.3f}], Computed=[{fundamental_bin}], Fractional=[{fundamental_bin_fractional:.4f}], Signal Range=[{sig_bin_start}:{sig_bin_end}]")
    print(f"Harmonics: Count=[{len(harmonic_bins)}], from HD2 to HD{max_harmonic}:")
    print(f"  Expected={expected_harmonic_bins_str}")
    print(f"  Computed={harmonic_bins_str}")

    # Verify fundamental bin detection (within 0.5 bins due to interpolation)
    assert bin_error < 0.5, f"Fundamental bin error too large: {bin_error:.3f}"

    # Verify harmonic bins are integers
    assert len(harmonic_bins) == max_harmonic - 1
    assert isinstance(harmonic_bins, np.ndarray)
    assert harmonic_bins.dtype == np.dtype('int64') or harmonic_bins.dtype == np.dtype('int32')

    # Verify harmonic bins match expected (within rounding tolerance)
    np.testing.assert_array_equal(harmonic_bins, expected_harmonic_bins)
