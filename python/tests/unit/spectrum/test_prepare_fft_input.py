"""Unit tests for _prepare_fft_input helper function.

After refactoring, _prepare_fft_input only handles:
- Data validation and shape normalization
- DC removal
- Amplitude normalization to -1 to 1 range

Window function handling has been moved to compute_spectrum.py
"""

import pytest
import numpy as np
from adctoolbox.spectrum._prepare_fft_input import _prepare_fft_input

@pytest.mark.parametrize("input_shape,expected_output_shape", [
    ((100,), (1, 100)),
    ((1,), (1, 1)),
    ((100, 1), (1, 100)),
    ((1, 100), (1, 100)),
    ((3, 100), (3, 100)),
    ((1024, 8), (8, 1024)),
    ((20, 15), (20, 15)),
])
def test_prepare_fft_input_transpose_handling(input_shape, expected_output_shape):
    """Test automatic transpose and shape handling.

    Verifies correct shape transformation for various input formats.
    """

    test_data = np.random.randn(*input_shape)
    processed = _prepare_fft_input(test_data)

    assert processed.shape == expected_output_shape
    print(f"\n[Test Transpose] Input: {input_shape} -> Expected: {expected_output_shape} -> Processed: {processed.shape}")


@pytest.mark.parametrize("signal_range,max_scale_range,expected_range", [
    ((-0.5, 0.5), None, (-1.0, 1.0)),
    ((-0.99, 0.99), None, (-1.0, 1.0)),
    ((0, 1024), None, (-1.0, 1.0)),
    ((256, 768), None, (-1.0, 1.0)),
    ((0, 2048), 2048, (-1.0, 1.0)),
    ((0, 2048), (0, 2048), (-1.0, 1.0)),
    ((512, 1536), 2048, (-0.5, 0.5)),
    ((256, 768), (0, 2048), (-0.25, 0.25)),
    ((-512, 512), 1024, (-1.0, 1.0)),
    ((-0.5,0.5), (-0.5,0.5), (-1.0, 1.0)),
    ((-0.25,0.25), (-0.5, 0.5), (-0.5, 0.5)),
    ((-0.05,0.05), (-0.5, 0.5), (-0.1, 0.1)),
    ((-0.5,0.5), (-1, 1), (-0.5, 0.5)),
    ((-0.5,0.5), 0, (-0.5, 0.5)),
    ((-0.05,0.05), 0, (-0.05, 0.05)),
])
def test_prepare_fft_input_max_scale_range(signal_range, max_scale_range, expected_range):
    """Test max_scale_range normalization."""
    test_signal = np.random.uniform(*signal_range, 10000)
    processed = _prepare_fft_input(test_signal, max_scale_range=max_scale_range)

    assert np.abs(np.mean(processed)) < 1e-10

    actual_min = np.min(processed)
    actual_max = np.max(processed)
    print(f"\n[Test Normalization] Range={signal_range}, FSR={max_scale_range} -> Expected: [{expected_range[0]:.2f}, {expected_range[1]:.2f}], Actual: [{actual_min:.2f}, {actual_max:.2f}]")

    assert actual_min >= expected_range[0] - 0.1
    assert actual_max <= expected_range[1] + 0.1
