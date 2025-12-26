"""
Unit Test: Verify spec_plot with synthetic quantized signals

Purpose: Self-verify that spec_plot correctly measures ENOB for signals
         with known quantization levels (NOT compared against MATLAB)
"""
import numpy as np
import matplotlib.pyplot as plt
import pytest
from adctoolbox.aout import analyze_spectrum


def test_verify_spec_plot_quantization():
    """
    Verify that spec_plot correctly measures ENOB for quantized signals.

    Test strategy:
    1. Generate ideal sinusoid
    2. Quantize to known bit depths
    3. Measure ENOB with spec_plot
    4. Assert: ENOB ≈ bit depth (within tolerance)
    """
    # Generate ideal sinusoid
    N = 2**13
    J = 323  # Prime number for coherent sampling
    sig = 0.499 * np.sin(np.arange(N) * J * 2 * np.pi / N) + 0.5

    # Test multiple bit depths
    test_bits = [1, 4, 8, 12, 16]

    print(f'\n[Verify ENOB] [Quantization] [N={N}]')

    for nbits in test_bits:
        # Quantize signal to nbits resolution
        sig_quantized = np.floor(sig * 2**nbits) / 2**nbits

        # Measure ENOB
        result = analyze_spectrum(sig_quantized, show_label=False, n_thd=5, create_plot=False)
        enob = result['enob']
        sndr = result['sndr_db']

        # Expected: ENOB should be close to nbits
        error = abs(enob - nbits)

        # Tolerance: ENOB can be slightly below nbits due to quantization noise
        # For low bits (1-2), error can be larger due to harmonics
        tolerance = 0.5 if nbits > 2 else 1.0
        status = 'PASS' if error < tolerance else 'FAIL'

        print(f'  [{nbits:2d} bit] [ENOB={enob:6.3f}] [Error={error:.3f}] [{status}]')

        # Assert with appropriate tolerance
        assert error < tolerance, f"ENOB error too large for {nbits}-bit: {error:.4f}"


def test_verify_spec_plot_sweep():
    """
    Sweep quantization from 1-20 bits and verify ENOB tracks bit depth.

    This generates a visual verification plot (optional).
    """
    # Generate ideal sinusoid
    N = 2**13
    J = 323
    sig = 0.499 * np.sin(np.arange(N) * J * 2 * np.pi / N) + 0.5

    # Sweep over bit depths
    bit_sweep = np.arange(1, 21)
    enob_results = np.zeros(len(bit_sweep))

    print(f'\n[Verify ENOB Sweep] [1-20 bit] [N={N}]')

    for idx, nbits in enumerate(bit_sweep):
        sig_quantized = np.floor(sig * 2**nbits) / 2**nbits
        result = analyze_spectrum(sig_quantized, show_label=False, n_thd=5, create_plot=False)
        enob = result['enob']
        enob_results[idx] = enob

    # Verify ENOB increases with bit depth
    enob_diffs = np.diff(enob_results)
    monotonic = np.all(enob_diffs > 0)

    # Verify ENOB is close to bit depth for higher bits
    high_bit_indices = bit_sweep >= 8
    errors = np.abs(enob_results[high_bit_indices] - bit_sweep[high_bit_indices])
    max_error = np.max(errors)

    status = 'PASS' if monotonic and max_error < 0.3 else 'FAIL'
    print(f'  [Monotonic={monotonic}] [MaxErr(8-20b)={max_error:.4f}] [{status}]')

    assert monotonic, "ENOB should increase monotonically with bit depth"
    assert max_error < 0.3, f"ENOB errors too large for high bits: max={max_error:.4f}"


if __name__ == '__main__':
    """Run verification tests standalone"""
    print('Running spec_plot verification tests...\n')
    test_verify_spec_plot_quantization()
    test_verify_spec_plot_sweep()
    print('\n✅ All verification tests passed!')
