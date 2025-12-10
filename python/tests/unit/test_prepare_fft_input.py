"""Unit tests for _prepare_fft_input helper function."""

import pytest
import numpy as np
from adctoolbox.spectrum._prepare_fft_input import _prepare_fft_input


class TestPrepareFftInput:
    """Test suite for _prepare_fft_input function."""

    def test_single_run_1d_input(self):
        """Test with single run 1D input."""
        data = np.array([1, 2, 3, 4, 5])
        processed = _prepare_fft_input(data)

        assert processed.shape == (1, 5)

        # Check DC removal
        assert np.abs(np.mean(processed[0, :])) < 1e-10

    def test_multi_run_2d_input(self):
        """Test with multiple runs as 2D array."""
        data = np.array([
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7]
        ])
        processed = _prepare_fft_input(data)

        assert processed.shape == (3, 5)

        # Check DC removal for each run
        for i in range(3):
            assert np.abs(np.mean(processed[i, :])) < 1e-10

    def test_transpose_handling(self):
        """Test that (N, 1) is transposed to (1, N)."""
        data = np.array([[1], [2], [3], [4], [5]])  # Shape (5, 1)
        processed = _prepare_fft_input(data)

        assert processed.shape == (1, 5)

    def test_custom_max_code(self):
        """Test with custom max_scale_range value."""
        data = np.array([1, 2, 3, 4, 5])
        processed = _prepare_fft_input(data, max_scale_range=10.0)

        # Values should be normalized by 10
        # After DC removal: [-2, -1, 0, 1, 2], divided by 10: [-0.2, -0.1, 0, 0.1, 0.2]
        expected_max = 0.2
        assert np.max(np.abs(processed)) <= expected_max + 0.01

    def test_boxcar_window(self):
        """Test with boxcar (rectangular) window."""
        data = np.array([1, 2, 3, 4, 5])
        processed_boxcar = _prepare_fft_input(data, win_type='boxcar')
        processed_rect = _prepare_fft_input(data, win_type='rectangular')

        # Boxcar and rectangular should give same result
        np.testing.assert_array_almost_equal(processed_boxcar, processed_rect)

    def test_hann_window(self):
        """Test with Hann window."""
        # Create sine wave for better windowing effect
        t = np.arange(128) / 128
        data = np.sin(2 * np.pi * 5 * t)
        processed = _prepare_fft_input(data, win_type='hann')

        assert processed.shape == (1, 128)
        # Hann window should attenuate (DC is removed, so edges near 0)
        # The center should have higher magnitude than edges
        assert np.abs(processed[0, 64]) > np.abs(processed[0, 0])  # Center > Edge

    def test_hamming_window(self):
        """Test with Hamming window."""
        # Create sine wave for better windowing effect
        t = np.arange(128) / 128
        data = np.sin(2 * np.pi * 5 * t) + np.random.randn(128) * 0.01
        processed = _prepare_fft_input(data, win_type='hamming')

        assert processed.shape == (1, 128)
        # Hamming window should reduce power due to tapering
        original_power = np.sum(data**2)
        processed_power = np.sum(processed[0]**2)
        assert processed_power < original_power  # Windowing reduces power

    def test_zero_max_code_handling(self):
        """Test that zero max_scale_range doesn't cause division by zero."""
        data = np.zeros(10)
        processed = _prepare_fft_input(data, max_scale_range=0)

        assert processed.shape == (1, 10)
        # Should not normalize when max_code is 0
        np.testing.assert_array_equal(processed, np.zeros((1, 10)))

    def test_data_length_preserved(self):
        """Test that data length is preserved in output shape."""
        # Use sine wave data which gives non-zero values after normalization
        t = np.arange(8) / 8
        data = np.sin(2 * np.pi * t)
        processed = _prepare_fft_input(data)

        # Processed should maintain original shape
        assert processed.shape == (1, 8)
        # Should have some non-zero values after processing
        assert np.any(np.abs(processed[0, :]) > 1e-10)

    def test_scalar_input(self):
        """Test with scalar input."""
        data = np.array(5)
        processed = _prepare_fft_input(data)

        assert processed.shape == (1, 1)

    def test_empty_input_raises_error(self):
        """Test that invalid input raises appropriate error."""
        # 3D input should raise error
        data = np.zeros((2, 3, 4))
        with pytest.raises(ValueError, match="Input must be 1D or 2D"):
            _prepare_fft_input(data)

    def test_dc_removal_accuracy(self):
        """Test that DC is removed accurately."""
        # Create signal with DC offset
        fs = 1000
        t = np.arange(100) / fs
        signal = 5.0 + 2.0 * np.sin(2 * np.pi * 10 * t)  # 5V DC offset

        processed = _prepare_fft_input(signal)

        # DC should be removed
        assert np.abs(np.mean(processed)) < 1e-10

    def test_power_normalization(self):
        """Test that window power normalization is applied."""
        data = np.ones(100)

        # Boxcar window should preserve power
        processed_boxcar = _prepare_fft_input(data, win_type='boxcar')
        power_boxcar = np.mean(processed_boxcar**2)

        # Hann window with power normalization should also preserve power
        processed_hann = _prepare_fft_input(data, win_type='hann')
        power_hann = np.mean(processed_hann**2)

        # Powers should be similar (within 10%)
        np.testing.assert_allclose(power_boxcar, power_hann, rtol=0.1)

    def test_auto_transpose_when_rows_much_larger(self):
        """Test automatic transpose when rows >> cols (N samples >> M runs)."""
        # Simulate user passing (N, M) = (1024, 8) instead of (M, N) = (8, 1024)
        data = np.random.randn(1024, 8)  # N=1024 samples, M=8 runs (wrong format)

        with pytest.warns(UserWarning, match="Auto-transpose"):
            processed = _prepare_fft_input(data)

        # Should be transposed to (M, N) = (8, 1024)
        assert processed.shape == (8, 1024)

    def test_no_transpose_when_rows_not_much_larger(self):
        """Test no transpose when rows are not significantly larger than cols."""
        # When rows are only slightly larger, don't transpose
        data = np.random.randn(20, 15)  # 20/15 = 1.33 < 2, so no transpose

        # Should NOT trigger warning or transpose
        processed = _prepare_fft_input(data)

        # Should keep original shape (20, 15)
        assert processed.shape == (20, 15)
