"""Unit tests for _align_spectrum_phase helper function."""

import pytest
import numpy as np
from adctoolbox.spectrum._align_spectrum_phase import _align_spectrum_phase


class TestAlignSpectrumPhase:
    """Test suite for _align_spectrum_phase function."""

    def test_fundamental_aligned_to_zero_phase(self):
        """Test that fundamental frequency is aligned to phase 0."""
        n_fft = 256
        bin_idx = 10
        bin_r = 10.0

        # Create FFT data with arbitrary phase at fundamental
        fft_data = np.zeros(n_fft, dtype=complex)
        fft_data[bin_idx] = 5.0 * np.exp(1j * np.pi / 3)  # Phase = 60 degrees

        fft_aligned = _align_spectrum_phase(fft_data, bin_idx, bin_r, n_fft)

        # Fundamental should now have phase ~0
        fundamental_phase = np.angle(fft_aligned[bin_idx])
        assert np.abs(fundamental_phase) < 1e-10

    def test_dc_bin_zeroed(self):
        """Test that DC bin is zeroed after alignment."""
        n_fft = 128
        bin_idx = 15
        bin_r = 15.0

        fft_data = np.ones(n_fft, dtype=complex)
        fft_data[0] = 100.0  # Large DC component

        fft_aligned = _align_spectrum_phase(fft_data, bin_idx, bin_r, n_fft)

        # DC should be zero
        assert fft_aligned[0] == 0.0

    def test_harmonic_phase_alignment(self):
        """Test that harmonics are aligned by integer multiples."""
        n_fft = 512
        fundamental_bin = 20
        bin_r = 20.0

        # Create signal with fundamental and 2nd harmonic
        fft_data = np.zeros(n_fft, dtype=complex)
        fundamental_phase = np.pi / 4  # 45 degrees
        fft_data[fundamental_bin] = 10.0 * np.exp(1j * fundamental_phase)
        fft_data[2 * fundamental_bin] = 5.0 * np.exp(1j * 2 * fundamental_phase)

        fft_aligned = _align_spectrum_phase(fft_data, fundamental_bin, bin_r, n_fft)

        # Fundamental should be at phase 0
        assert np.abs(np.angle(fft_aligned[fundamental_bin])) < 1e-10

        # 2nd harmonic should also be at phase 0 (aligned by 2x rotation)
        assert np.abs(np.angle(fft_aligned[2 * fundamental_bin])) < 1e-10

    def test_magnitude_preservation(self):
        """Test that magnitude is preserved after alignment."""
        n_fft = 256
        bin_idx = 25
        bin_r = 25.0

        # Create FFT data
        fft_data = np.random.randn(n_fft) + 1j * np.random.randn(n_fft)
        original_mag = np.abs(fft_data)

        fft_aligned = _align_spectrum_phase(fft_data, bin_idx, bin_r, n_fft)

        # Magnitudes should be preserved (except DC which is zeroed)
        aligned_mag = np.abs(fft_aligned)
        aligned_mag[0] = original_mag[0]  # Exclude DC from comparison

        np.testing.assert_array_almost_equal(aligned_mag, original_mag, decimal=10)

    def test_multiple_harmonics_alignment(self):
        """Test alignment with multiple harmonics."""
        n_fft = 1024
        fundamental_bin = 30
        bin_r = 30.0
        phase_offset = np.pi / 6  # 30 degrees

        # Create signal with fundamental and harmonics 2, 3, 4
        fft_data = np.zeros(n_fft, dtype=complex)
        for h in [1, 2, 3, 4]:
            harmonic_bin = fundamental_bin * h
            if harmonic_bin < n_fft // 2:
                fft_data[harmonic_bin] = 10.0 / h * np.exp(1j * h * phase_offset)

        fft_aligned = _align_spectrum_phase(fft_data, fundamental_bin, bin_r, n_fft)

        # All harmonics should be phase-aligned
        for h in [1, 2, 3, 4]:
            harmonic_bin = fundamental_bin * h
            if harmonic_bin < n_fft // 2:
                phase = np.angle(fft_aligned[harmonic_bin])
                assert np.abs(phase) < 1e-8

    def test_non_harmonic_bins_rotated(self):
        """Test that non-harmonic bins are also rotated."""
        n_fft = 256
        bin_idx = 20
        bin_r = 20.0

        # Create FFT with fundamental and non-harmonic frequency
        fft_data = np.zeros(n_fft, dtype=complex)
        fft_data[bin_idx] = 10.0 * np.exp(1j * np.pi / 3)
        fft_data[25] = 5.0 * np.exp(1j * np.pi / 4)  # Non-harmonic

        fft_aligned = _align_spectrum_phase(fft_data, bin_idx, bin_r, n_fft)

        # Non-zero entries should exist (not all zeroed)
        assert np.count_nonzero(fft_aligned) > 1

    def test_aliasing_handling(self):
        """Test that aliasing is handled for harmonics above Nyquist."""
        n_fft = 256
        fundamental_bin = 100  # High frequency
        bin_r = 100.0

        fft_data = np.zeros(n_fft, dtype=complex)
        fft_data[fundamental_bin] = 10.0 * np.exp(1j * np.pi / 4)

        # Should not raise error even though harmonics would alias
        fft_aligned = _align_spectrum_phase(fft_data, fundamental_bin, bin_r, n_fft)

        assert fft_aligned.shape == fft_data.shape

    def test_fractional_bin_idx(self):
        """Test with fractional refined bin position."""
        n_fft = 512
        bin_idx = 30
        bin_r = 30.5  # Fractional bin

        fft_data = np.zeros(n_fft, dtype=complex)
        fft_data[bin_idx] = 10.0 * np.exp(1j * np.pi / 3)

        # Should handle fractional bin_r gracefully
        fft_aligned = _align_spectrum_phase(fft_data, bin_idx, bin_r, n_fft)

        # Fundamental should be aligned
        assert np.abs(np.angle(fft_aligned[bin_idx])) < 1e-10

    def test_zero_fundamental_phase(self):
        """Test when fundamental already has phase 0."""
        n_fft = 128
        bin_idx = 15
        bin_r = 15.0

        fft_data = np.zeros(n_fft, dtype=complex)
        fft_data[bin_idx] = 10.0 + 0j  # Already phase 0

        fft_aligned = _align_spectrum_phase(fft_data, bin_idx, bin_r, n_fft)

        # Should be approximately unchanged (except DC removal)
        fft_data[0] = 0  # Match DC removal
        np.testing.assert_array_almost_equal(fft_aligned, fft_data, decimal=10)

    def test_negative_phase_alignment(self):
        """Test alignment with negative phase."""
        n_fft = 256
        bin_idx = 20
        bin_r = 20.0

        fft_data = np.zeros(n_fft, dtype=complex)
        fft_data[bin_idx] = 8.0 * np.exp(-1j * np.pi / 3)  # Negative phase

        fft_aligned = _align_spectrum_phase(fft_data, bin_idx, bin_r, n_fft)

        # Should still align to phase 0
        assert np.abs(np.angle(fft_aligned[bin_idx])) < 1e-10

    def test_output_is_copy(self):
        """Test that output doesn't modify input."""
        n_fft = 128
        bin_idx = 10
        bin_r = 10.0

        fft_data = np.random.randn(n_fft) + 1j * np.random.randn(n_fft)
        fft_data_original = fft_data.copy()

        _align_spectrum_phase(fft_data, bin_idx, bin_r, n_fft)

        # Original should be unchanged
        np.testing.assert_array_equal(fft_data, fft_data_original)

    def test_complex_dtype_preserved(self):
        """Test that output maintains complex dtype."""
        n_fft = 256
        bin_idx = 15
        bin_r = 15.0

        fft_data = np.ones(n_fft, dtype=complex)
        fft_aligned = _align_spectrum_phase(fft_data, bin_idx, bin_r, n_fft)

        assert fft_aligned.dtype == np.complex128 or fft_aligned.dtype == np.complex64

    def test_edge_case_bin_at_nyquist(self):
        """Test when fundamental is at Nyquist frequency."""
        n_fft = 256
        bin_idx = n_fft // 2
        bin_r = float(bin_idx)

        fft_data = np.zeros(n_fft, dtype=complex)
        fft_data[bin_idx] = 10.0 * np.exp(1j * np.pi / 4)

        # Should handle edge case
        fft_aligned = _align_spectrum_phase(fft_data, bin_idx, bin_r, n_fft)

        assert fft_aligned.shape == fft_data.shape

    def test_small_fft_size(self):
        """Test with very small FFT size."""
        n_fft = 8
        bin_idx = 2
        bin_r = 2.0

        fft_data = np.array([1+1j, 2+2j, 3+3j, 4+4j, 5+5j, 6+6j, 7+7j, 8+8j])
        fft_aligned = _align_spectrum_phase(fft_data, bin_idx, bin_r, n_fft)

        assert fft_aligned.shape == (8,)
        assert fft_aligned[0] == 0  # DC zeroed

    def test_large_fft_size(self):
        """Test with large FFT size (performance test)."""
        n_fft = 16384
        bin_idx = 1000
        bin_r = 1000.0

        fft_data = np.random.randn(n_fft) + 1j * np.random.randn(n_fft)
        fft_data[bin_idx] = 100.0 * np.exp(1j * np.pi / 3)

        # Should complete without error
        fft_aligned = _align_spectrum_phase(fft_data, bin_idx, bin_r, n_fft)

        assert fft_aligned.shape == (n_fft,)
        assert np.abs(np.angle(fft_aligned[bin_idx])) < 1e-10

    def test_phase_continuity_across_spectrum(self):
        """Test that phase alignment maintains continuity."""
        n_fft = 512
        bin_idx = 50
        bin_r = 50.0

        # Create smooth spectrum
        fft_data = np.zeros(n_fft, dtype=complex)
        for i in range(1, n_fft // 2):
            phase = 2 * np.pi * i / n_fft
            fft_data[i] = np.exp(1j * phase)

        fft_aligned = _align_spectrum_phase(fft_data, bin_idx, bin_r, n_fft)

        # Check that alignment worked
        assert np.abs(np.angle(fft_aligned[bin_idx])) < 1e-10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
