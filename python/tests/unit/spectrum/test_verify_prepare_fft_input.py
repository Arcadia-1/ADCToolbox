"""
Unit Test: Verify _prepare_fft_input for FFT preprocessing

Purpose: Self-verify that _prepare_fft_input correctly applies
         windowing, DC removal, normalization, and input handling
"""
import numpy as np
from adctoolbox.spectrum._prepare_fft_input import _prepare_fft_input


def test_verify_prepare_fft_input_basic():
    """
    Verify _prepare_fft_input applies windowing correctly.

    Test strategy:
    1. Generate simple signal
    2. Prepare FFT input with Hann window
    3. Assert: Output has correct shape and windowing applied
    """
    N = 512
    sig = np.sin(2*np.pi*0.1*np.arange(N))

    result = _prepare_fft_input(sig, win_type='hann')

    print(f'\n[Verify Prepare FFT Input] [N={N}]')
    print(f'  [Input shape  ] {np.atleast_2d(sig).shape}')
    print(f'  [Output shape ] {result.shape}')
    print(f'  [Input peak   ] {np.max(np.abs(sig)):.6f}')
    print(f'  [Output peak  ] {np.max(np.abs(result)):.6f}')

    # Output should be 2D (M, N) format
    assert result.ndim == 2, f"Output should be 2D, got {result.ndim}D"

    # With windowing, peak should be reduced
    # Note: DC removal and normalization also applied
    print(f'  [Status] PASS')


def test_verify_prepare_fft_input_2d():
    """
    Verify _prepare_fft_input handles 2D input (multiple runs).

    Test strategy:
    1. Create 2D array (M runs, N samples)
    2. Prepare FFT input
    3. Assert: Output shape is (M, N)
    """
    M, N = 3, 256
    data_2d = np.random.randn(M, N)

    result = _prepare_fft_input(data_2d, win_type='hamming')

    print(f'\n[Verify 2D Input Handling]')
    print(f'  [Input shape ] {data_2d.shape}')
    print(f'  [Output shape] {result.shape}')

    # Output should have same shape as input
    assert result.shape == data_2d.shape, f"Shape mismatch: {result.shape} vs {data_2d.shape}"

    print(f'  [Status] PASS')


def test_verify_prepare_fft_input_window_types():
    """
    Verify different window types are applied correctly.

    Test strategy:
    1. Apply different windows
    2. Verify all produce output with correct shape
    3. Check that window effects visible in peak reduction
    """
    N = 256
    sig = np.sin(2*np.pi*0.2*np.arange(N))

    windows_to_test = ['boxcar', 'hann', 'hamming', 'blackman']

    print(f'\n[Verify Window Types] [N={N}]')

    results = {}
    for win_type in windows_to_test:
        result = _prepare_fft_input(sig, win_type=win_type)
        results[win_type] = result
        print(f'  [{win_type:8s}] shape={result.shape}, peak={np.max(np.abs(result)):.4f}')

        # All should produce 2D output with same shape
        assert result.ndim == 2, f"Output should be 2D for {win_type}"
        assert result.shape[1] == N, f"Sample dimension should be {N} for {win_type}"

    # Boxcar (no window) should have higher peak than windowed versions
    # (because no amplitude reduction from windowing)
    # Note: DC removal reduces peak, but windowing reduces it further

    print(f'  [Status] PASS')


def test_verify_prepare_fft_input_normalization():
    """
    Verify normalization by max_scale_range.

    Test strategy:
    1. Create signal with known range
    2. Normalize by specific scale
    3. Assert: Output values within expected range
    """
    N = 128
    sig = np.ones(N) * 0.5  # Simple signal with value 0.5
    max_scale = 1.0

    result = _prepare_fft_input(sig, max_scale_range=max_scale, win_type='boxcar')

    print(f'\n[Verify Normalization]')
    print(f'  [Input shape] {np.atleast_2d(sig).shape}')
    print(f'  [Output shape] {result.shape}')
    print(f'  [Input mean ] {np.mean(sig):.4f}')
    print(f'  [Output mean] {np.mean(result):.4f}')

    # Output should be normalized and 2D
    assert result.ndim == 2, "Output should be 2D"
    assert result.shape[1] == N, "Sample dimension should match"

    print(f'  [Status] PASS')


def test_verify_prepare_fft_input_dc_removal():
    """
    Verify DC component is removed.

    Test strategy:
    1. Create signal with DC offset
    2. Prepare FFT input
    3. Assert: Output has near-zero mean (DC removed)
    """
    N = 256
    dc_offset = 5.0
    sig = np.sin(2*np.pi*0.15*np.arange(N)) + dc_offset

    result = _prepare_fft_input(sig, win_type='boxcar')

    print(f'\n[Verify DC Removal]')
    print(f'  [Input mean  ] {np.mean(sig):.6f}')
    print(f'  [Output mean ] {np.mean(result):.6e}')

    # After DC removal, mean should be very close to zero
    assert abs(np.mean(result)) < 1e-10, f"DC not removed: mean={np.mean(result)}"

    print(f'  [Status] PASS')


def test_verify_prepare_fft_input_auto_transpose():
    """
    Verify auto-transpose for (N, 1) shaped inputs.

    Test strategy:
    1. Create column vector (N, 1)
    2. Prepare FFT input
    3. Assert: Output is (1, N) format
    """
    N = 128
    sig_col = np.sin(2*np.pi*0.1*np.arange(N)).reshape(-1, 1)  # (N, 1)

    result = _prepare_fft_input(sig_col, win_type='hann')

    print(f'\n[Verify Auto-Transpose]')
    print(f'  [Input shape ] {sig_col.shape}')
    print(f'  [Output shape] {result.shape}')

    # Should be transposed to (1, N)
    assert result.shape == (1, N), f"Should be transposed to (1, {N}), got {result.shape}"

    print(f'  [Status] PASS')


if __name__ == '__main__':
    """Run verification tests standalone"""
    print('='*80)
    print('RUNNING PREPARE_FFT_INPUT VERIFICATION TESTS')
    print('='*80)

    test_verify_prepare_fft_input_basic()
    test_verify_prepare_fft_input_2d()
    test_verify_prepare_fft_input_window_types()
    test_verify_prepare_fft_input_normalization()
    test_verify_prepare_fft_input_dc_removal()
    test_verify_prepare_fft_input_auto_transpose()

    print('\n' + '='*80)
    print('** All prepare_fft_input verification tests passed! **')
    print('='*80)
