"""Unit tests for _find_fundamental helper function."""

import pytest
import numpy as np
from adctoolbox.spectrum._find_fundamental import _find_fundamental
from adctoolbox.spectrum._find_harmonic_bins import _find_harmonic_bins


class TestFindFundamental:
    """Test suite for _find_fundamental function."""

    def test_simple_sine_magnitude_method(self):
        """Test finding fundamental for simple sine wave using magnitude method."""
        # Create a simple sine at bin 10
        n_fft = 1024
        fs = 1000
        f_sig = 10 * fs / n_fft  # Signal at bin 10
        t = np.arange(n_fft) / fs
        signal = np.sin(2 * np.pi * f_sig * t)

        fft_data = np.fft.fft(signal)
        bin_idx, bin_r = _find_fundamental(fft_data, n_fft, osr=1, method='magnitude')

        assert bin_idx == 10
        # Refined bin should be close to 10
        assert 9.5 <= bin_r <= 10.5

    def test_simple_sine_power_method(self):
        """Test finding fundamental using power method."""
        n_fft = 512
        fs = 1000
        f_sig = 20 * fs / n_fft  # Signal at bin 20
        t = np.arange(n_fft) / fs
        signal = np.sin(2 * np.pi * f_sig * t)

        fft_data = np.fft.fft(signal)
        bin_idx, bin_r = _find_fundamental(fft_data, n_fft, osr=1, method='power')

        assert bin_idx == 20
        assert 19.5 <= bin_r <= 20.5

    def test_simple_sine_log_method(self):
        """Test finding fundamental using log method."""
        n_fft = 256
        fs = 1000
        f_sig = 15 * fs / n_fft  # Signal at bin 15
        t = np.arange(n_fft) / fs
        signal = np.sin(2 * np.pi * f_sig * t)

        fft_data = np.fft.fft(signal)
        bin_idx, bin_r = _find_fundamental(fft_data, n_fft, osr=1, method='log')

        assert bin_idx == 15
        assert 14.5 <= bin_r <= 15.5

    def test_dc_exclusion(self):
        """Test that DC bin (index 0) is excluded from search."""
        n_fft = 128
        # Create spectrum with large DC and smaller signal at bin 5
        spectrum = np.zeros(n_fft, dtype=complex)
        spectrum[0] = 1000  # Large DC
        spectrum[5] = 10    # Smaller signal

        bin_idx, _ = _find_fundamental(spectrum, n_fft, osr=1, method='magnitude')

        # Should find bin 5, not DC
        assert bin_idx == 5

    def test_osr_limiting(self):
        """Test that OSR limits the search range."""
        n_fft = 1024
        osr = 4  # Search up to Nyquist/4

        # Put signals at different locations
        spectrum = np.zeros(n_fft, dtype=complex)
        spectrum[64] = 10   # Within OSR range (1024/2/4 = 128)
        spectrum[200] = 100  # Outside OSR range, but larger

        bin_idx, _ = _find_fundamental(spectrum, n_fft, osr=osr, method='magnitude')

        # Should find bin 64 (within range), not 200 (out of range)
        assert bin_idx == 64

    def test_parabolic_interpolation_accuracy(self):
        """Test parabolic interpolation for sub-bin accuracy."""
        n_fft = 1024
        fs = 1000

        # Create signal slightly off-bin (bin 10.3)
        f_sig = 10.3 * fs / n_fft
        t = np.arange(n_fft) / fs
        signal = np.sin(2 * np.pi * f_sig * t)

        fft_data = np.fft.fft(signal)
        bin_idx, bin_r = _find_fundamental(fft_data, n_fft, osr=1, method='magnitude')

        # Integer bin should be 10
        assert bin_idx == 10
        # Refined bin should be close to 10.3
        assert 10.1 <= bin_r <= 10.5

    def test_edge_case_bin_1(self):
        """Test when signal is at bin 1 (edge case for parabolic interpolation)."""
        n_fft = 256
        spectrum = np.zeros(n_fft, dtype=complex)
        spectrum[1] = 10

        bin_idx, bin_r = _find_fundamental(spectrum, n_fft, osr=1, method='magnitude')

        assert bin_idx == 1
        # Refined value should be reasonable
        assert 0.5 <= bin_r <= 1.5

    def test_edge_case_last_bin(self):
        """Test when signal is at last bin in search range."""
        n_fft = 128
        osr = 2
        max_bin = n_fft // 2 // osr - 1  # Last searchable bin

        spectrum = np.zeros(n_fft, dtype=complex)
        spectrum[max_bin] = 10

        bin_idx, bin_r = _find_fundamental(spectrum, n_fft, osr=osr, method='magnitude')

        assert bin_idx == max_bin

    def test_noisy_signal(self):
        """Test finding fundamental in presence of noise."""
        n_fft = 512
        fs = 1000
        f_sig = 25 * fs / n_fft  # Signal at bin 25
        t = np.arange(n_fft) / fs

        # Signal + noise
        signal = 2.0 * np.sin(2 * np.pi * f_sig * t) + 0.1 * np.random.randn(n_fft)

        fft_data = np.fft.fft(signal)
        bin_idx, bin_r = _find_fundamental(fft_data, n_fft, osr=1, method='magnitude')

        # Should still find bin 25
        assert bin_idx == 25
        # Refined bin might be slightly off due to noise, but should be close
        assert 24 <= bin_r <= 26

    def test_invalid_method_raises_error(self):
        """Test that invalid method raises ValueError."""
        spectrum = np.ones(128, dtype=complex)
        with pytest.raises(ValueError, match="Unknown method"):
            _find_fundamental(spectrum, 128, osr=1, method='invalid_method')

    def test_single_bin_spectrum(self):
        """Test with very small spectrum (edge case)."""
        spectrum = np.array([1.0 + 0j, 2.0 + 0j])
        bin_idx, bin_r = _find_fundamental(spectrum, 2, osr=1, method='magnitude')

        # Should handle gracefully
        assert bin_idx in [0, 1]

    def test_all_zeros_spectrum(self):
        """Test with zero spectrum."""
        spectrum = np.zeros(128, dtype=complex)
        bin_idx, bin_r = _find_fundamental(spectrum, 128, osr=1, method='magnitude')

        # Should return bin 1 (since DC is excluded)
        assert bin_idx == 1


class TestFindHarmonicBins:
    """Test suite for _find_harmonic_bins function."""

    def test_basic_harmonics(self):
        """Test finding harmonics for basic case."""
        fundamental_bin = 10.0
        n_fft = 1024
        harmonic = 5

        harmonic_bins = _find_harmonic_bins(fundamental_bin, harmonic, n_fft)

        assert len(harmonic_bins) == 5
        # Harmonics should be at integer multiples
        # But with aliasing handled
        assert harmonic_bins[0] == pytest.approx(10.0, abs=0.1)  # 1st harmonic (fundamental)
        assert harmonic_bins[1] == pytest.approx(20.0, abs=0.1)  # 2nd harmonic
        assert harmonic_bins[2] == pytest.approx(30.0, abs=0.1)  # 3rd harmonic
        assert harmonic_bins[3] == pytest.approx(40.0, abs=0.1)  # 4th harmonic
        assert harmonic_bins[4] == pytest.approx(50.0, abs=0.1)  # 5th harmonic

    def test_fractional_fundamental(self):
        """Test with fractional fundamental bin."""
        fundamental_bin = 10.5
        n_fft = 512
        harmonic = 3

        harmonic_bins = _find_harmonic_bins(fundamental_bin, harmonic, n_fft)

        assert len(harmonic_bins) == 3
        assert harmonic_bins[0] == pytest.approx(10.5, abs=0.1)
        assert harmonic_bins[1] == pytest.approx(21.0, abs=0.1)  # 2 * 10.5
        assert harmonic_bins[2] == pytest.approx(31.5, abs=0.1)  # 3 * 10.5

    def test_aliasing_near_nyquist(self):
        """Test that harmonics near/above Nyquist are aliased correctly."""
        # Fundamental at 40% of Nyquist
        n_fft = 256
        fundamental_bin = 51.0  # ~51/128 = 0.398 * Nyquist

        harmonic_bins = _find_harmonic_bins(fundamental_bin, 5, n_fft)

        # 3rd harmonic (153) should alias
        # All harmonics should be within [0, n_fft//2]
        assert all(0 <= h <= n_fft // 2 for h in harmonic_bins)

    def test_single_harmonic(self):
        """Test with single harmonic (just fundamental)."""
        fundamental_bin = 15.0
        n_fft = 512

        harmonic_bins = _find_harmonic_bins(fundamental_bin, 1, n_fft)

        assert len(harmonic_bins) == 1
        assert harmonic_bins[0] == pytest.approx(15.0, abs=0.1)

    def test_many_harmonics(self):
        """Test with many harmonics."""
        fundamental_bin = 5.0
        n_fft = 2048
        harmonic = 20

        harmonic_bins = _find_harmonic_bins(fundamental_bin, harmonic, n_fft)

        assert len(harmonic_bins) == 20
        # All should be non-negative and within Nyquist
        assert all(0 <= h <= n_fft // 2 for h in harmonic_bins)

    def test_high_frequency_fundamental(self):
        """Test with fundamental close to Nyquist."""
        n_fft = 512
        fundamental_bin = 240.0  # Close to Nyquist (256)

        harmonic_bins = _find_harmonic_bins(fundamental_bin, 3, n_fft)

        # All harmonics should alias back to valid range
        assert all(0 <= h <= n_fft // 2 for h in harmonic_bins)

    def test_zero_fundamental(self):
        """Test with zero fundamental (edge case)."""
        harmonic_bins = _find_harmonic_bins(0.0, 5, 1024)

        # All harmonics should be zero
        assert all(h == 0.0 for h in harmonic_bins)

    def test_output_type(self):
        """Test that output is numpy array."""
        harmonic_bins = _find_harmonic_bins(10.0, 5, 512)

        assert isinstance(harmonic_bins, np.ndarray)
        assert harmonic_bins.dtype == np.float64


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
