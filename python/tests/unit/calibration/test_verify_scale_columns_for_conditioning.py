import numpy as np
from adctoolbox.calibration._scale_columns_for_conditioning import _scale_columns_for_conditioning, _recover_columns_for_conditioning

def test_scaling_and_near_zero():
    """Verify magnitude scaling and protection for insignificant bits."""
    # 1. Generate random bits (N=1024, M=3)
    # Col 0: Range [0, 1]
    # Col 1: Range [0, 1e6] (Large magnitude)
    # Col 2: Range [0, 0] (Dead bit / Near-zero)
    
    bits = np.random.rand(1024, 12)    
    bits[:, 3] *= 1e6  # Amplify col 3 to test large scaling
    bits[:, 5] *= 0    # Wipe col 5 to test near-zero protection
    bits[:, 11] *= 0    # Wipe col 11 to test near-zero protection
    bits[:, 7] *= 1e-7  # Small magnitude col

    # 2. Run preparation with verbose=2 to see internal scaling process
    print()
    bits_effective, bit_scales = _scale_columns_for_conditioning(bits, verbose=2)
    

    # 3. Validate scaling results
    assert bits_effective.shape == bits.shape
    excepted_scales = np.array([-1, -1, -1, 5, -1, 0, -1, -8, -1, -1, -1, 0])  # Expected scales for each column
    np.testing.assert_array_equal(bit_scales, excepted_scales)

    print(f"[TEST INFO] Computed scales (powers of 10): {bit_scales.tolist()}")


def test_scaling_and_recover():
    n_bits = 16
    nominal_weights = 2 ** np.arange(n_bits-1, -1, -1) / 2**(n_bits-1)
    bits = np.random.randint(0, 2, (1024, n_bits)) * nominal_weights

    print()
    bits_effective, bit_scales = _scale_columns_for_conditioning(bits, verbose=2)

    print("[conditioning] Scaled bits shape:", bits_effective.shape)
    print("[conditioning] Bit scales (powers of 10):", bit_scales.tolist())

    w0 = 1
    mock_w_math = (nominal_weights / w0) * (10.0 ** bit_scales)
    print(f"[conditioning] Mock weights in math space: {mock_w_math.tolist()}")

    recovered_weights = _recover_columns_for_conditioning(mock_w_math, bits.shape[1], 1/w0, bit_scales)
    print(f"[recover] Recovered weights: {recovered_weights.tolist()}")

    print(f"[recover] Nominal weights: {nominal_weights.tolist()}")
    np.testing.assert_allclose(recovered_weights, nominal_weights, rtol=1e-10)
