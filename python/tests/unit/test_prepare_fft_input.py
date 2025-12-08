"""Unit tests for _prepare_fft_input helper function."""

import pytest
import numpy as np
from adctoolbox.aout._prepare_fft_input import _prepare_fft_input, _get_window_correction


class TestPrepareFftInput:
    """Test suite for _prepare_fft_input function."""

    def test_single_run_1d_input(self):
        """Test with single run 1D input."""
        data = np.array([1, 2, 3, 4, 5])
        processed, max_code, n_samples = _prepare_fft_input(data)

        assert processed.shape == (1, 5)
        assert n_samples == 5
        assert max_code == 4.0  # max(5) - min(1) = 4

        # Check DC removal
        assert np.abs(np.mean(processed[0, :])) < 1e-10

    def test_multi_run_2d_input(self):
        """Test with multiple runs as 2D array."""
        data = np.array([
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7]
        ])
        processed, max_code, n_samples = _prepare_fft_input(data)

        assert processed.shape == (3, 5)
        assert n_samples == 5

        # Check DC removal for each run
        for i in range(3):
            assert np.abs(np.mean(processed[i, :])) < 1e-10

    def test_transpose_handling(self):
        """Test that (N, 1) is transposed to (1, N)."""
        data = np.array([[1], [2], [3], [4], [5]])  # Shape (5, 1)
        processed, max_code, n_samples = _prepare_fft_input(data)

        assert processed.shape == (1, 5)
        assert n_samples == 5

    def test_custom_max_code(self):
        """Test with custom max_code value."""
        data = np.array([1, 2, 3, 4, 5])
        processed, max_code_used, n_samples = _prepare_fft_input(data, max_code=10.0)

        assert max_code_used == 10.0
        # Values should be normalized by 10
        # After DC removal: [-2, -1, 0, 1, 2], divided by 10: [-0.2, -0.1, 0, 0.1, 0.2]
        expected_max = 0.2
        assert np.max(np.abs(processed)) <= expected_max + 0.01

    def test_boxcar_window(self):
        """Test with boxcar (rectangular) window."""
        data = np.array([1, 2, 3, 4, 5])
        processed_boxcar, _, _ = _prepare_fft_input(data, win_type='boxcar')
        processed_rect, _, _ = _prepare_fft_input(data, win_type='rectangular')

        # Boxcar and rectangular should give same result
        np.testing.assert_array_almost_equal(processed_boxcar, processed_rect)

    def test_hann_window(self):
        """Test with Hann window."""
        # Create sine wave for better windowing effect
        t = np.arange(128) / 128
        data = np.sin(2 * np.pi * 5 * t)
        processed, _, _ = _prepare_fft_input(data, win_type='hann')

        assert processed.shape == (1, 128)
        # Hann window should attenuate (DC is removed, so edges near 0)
        # The center should have higher magnitude than edges
        assert np.abs(processed[0, 64]) > np.abs(processed[0, 0])  # Center > Edge

    def test_hamming_window(self):
        """Test with Hamming window."""
        # Create sine wave for better windowing effect
        t = np.arange(128) / 128
        data = np.sin(2 * np.pi * 5 * t) + np.random.randn(128) * 0.01
        processed, _, _ = _prepare_fft_input(data, win_type='hamming')

        assert processed.shape == (1, 128)
        # Hamming window should reduce power due to tapering
        original_power = np.sum(data**2)
        processed_power = np.sum(processed[0]**2)
        assert processed_power < original_power  # Windowing reduces power

    def test_zero_max_code_handling(self):
        """Test that zero max_code doesn't cause division by zero."""
        data = np.zeros(10)
        processed, max_code, n_samples = _prepare_fft_input(data, max_code=0)

        assert processed.shape == (1, 10)
        assert max_code == 0
        # Should not normalize when max_code is 0
        np.testing.assert_array_equal(processed, np.zeros((1, 10)))

    def test_n_fft_equals_data_length(self):
        """Test with n_fft equal to data length."""
        # Use sine wave data which gives non-zero values after normalization
        t = np.arange(8) / 8
        data = np.sin(2 * np.pi * t)
        processed, _, n_samples = _prepare_fft_input(data, n_fft=8)

        # n_samples should be original length
        assert n_samples == 8
        # Processed should maintain original shape
        assert processed.shape[1] == 8
        # Should have some non-zero values after processing
        assert np.any(np.abs(processed[0, :]) > 1e-10)

    def test_scalar_input(self):
        """Test with scalar input."""
        data = np.array(5)
        processed, max_code, n_samples = _prepare_fft_input(data)

        assert processed.shape == (1, 1)
        assert n_samples == 1

    def test_empty_input_raises_error(self):
        """Test that invalid input raises appropriate error."""
        # 3D input should raise error
        data = np.zeros((2, 3, 4))
        with pytest.raises(ValueError, match="Input data must be 1D or 2D"):
            _prepare_fft_input(data)

    def test_dc_removal_accuracy(self):
        """Test that DC is removed accurately."""
        # Create signal with DC offset
        fs = 1000
        t = np.arange(100) / fs
        signal = 5.0 + 2.0 * np.sin(2 * np.pi * 10 * t)  # 5V DC offset

        processed, _, _ = _prepare_fft_input(signal)

        # DC should be removed
        assert np.abs(np.mean(processed)) < 1e-10

    def test_power_normalization(self):
        """Test that window power normalization is applied."""
        data = np.ones(100)

        # Boxcar window should preserve power
        processed_boxcar, _, _ = _prepare_fft_input(data, win_type='boxcar')
        power_boxcar = np.mean(processed_boxcar**2)

        # Hann window with power normalization should also preserve power
        processed_hann, _, _ = _prepare_fft_input(data, win_type='hann')
        power_hann = np.mean(processed_hann**2)

        # Powers should be similar (within 10%)
        np.testing.assert_allclose(power_boxcar, power_hann, rtol=0.1)


class TestGetWindowCorrection:
    """Test suite for _get_window_correction function."""

    def test_boxcar_correction(self):
        """Test correction factor for boxcar window."""
        assert _get_window_correction('boxcar') == 1.0
        assert _get_window_correction('rectangular') == 1.0

    def test_hann_correction(self):
        """Test correction factor for Hann window."""
        assert _get_window_correction('hann') == 2.0
        assert _get_window_correction('hanning') == 2.0

    def test_hamming_correction(self):
        """Test correction factor for Hamming window."""
        assert _get_window_correction('hamming') == 1.85

    def test_blackman_correction(self):
        """Test correction factor for Blackman window."""
        assert _get_window_correction('blackman') == 2.38

    def test_flattop_correction(self):
        """Test correction factor for flat-top window."""
        assert _get_window_correction('flattop') == 4.64

    def test_unknown_window_default(self):
        """Test that unknown window returns default correction."""
        assert _get_window_correction('unknown_window') == 1.0

    def test_case_insensitive(self):
        """Test that window type is case-insensitive."""
        assert _get_window_correction('HANN') == 2.0
        assert _get_window_correction('Hann') == 2.0
        assert _get_window_correction('hAnN') == 2.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
