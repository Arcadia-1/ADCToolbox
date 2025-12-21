import numpy as np
from adctoolbox.calibration._prepare_input import _prepare_input

def test_single_dataset():
    """Verify standard flow with N (samples) > M (bits)."""
    # --- Case 1: Proper Orientation ---
    data = np.random.randint(0, 2, (1024, 8)).astype(float)
    print()
    res = _prepare_input(data, verbose=2)
    
    assert res["bits_stacked"].shape == (1024, 8)
    assert res["bit_width"] == 8
    assert res["num_segments"] == 1
    np.testing.assert_array_equal(res["bits_segments"][0], data)

    # --- Case 2: Transpose Triggered (N < M) ---
    data = np.random.randint(0, 2, (8, 1024)).astype(float)
    print()
    res = _prepare_input(data, verbose=2)
    
    assert res["bits_stacked"].shape == (1024, 8)
    assert res["bit_width"] == 8
    assert res["num_segments"] == 1
    np.testing.assert_array_equal(res["bits_segments"][0], data.T)

    # --- Case 3: Square Matrix Boundary ---
    data = np.random.randint(0, 2, (16, 16)).astype(float)
    print()
    res = _prepare_input(data, verbose=2)
    
    assert res["bits_stacked"].shape == (16, 16)
    assert res["bit_width"] == 16
    assert res["num_segments"] == 1
    np.testing.assert_array_equal(res["bits_segments"][0], data)


def test_multi_dataset():
    """Verify vertical stacking and segment tracking for multiple datasets."""
    # Create 4 datasets with varying sample counts but same bit width
    d1 = np.random.randint(0, 2, (1024, 8)).astype(float)
    d2 = np.random.randint(0, 2, (512, 8)).astype(float)
    d3 = np.random.randint(0, 2, (8, 8192)).astype(float)
    d4 = np.random.randint(0, 2, (8, 16384)).astype(float)
    data = [d1, d2, d3, d4]
    print()
    res = _prepare_input(data, verbose=2)
    
    # Combined samples: 1024 + 512 + 8192 + 16384 = 26112
    assert res["bits_stacked"].shape == (26112, 8)
    assert res["num_segments"] == 4
    np.testing.assert_array_equal(res["segment_lengths"], [1024, 512, 8192, 16384])    
    np.testing.assert_array_equal(res["bits_segments"][0], d1)
    np.testing.assert_array_equal(res["bits_segments"][1], d2)
    np.testing.assert_array_equal(res["bits_segments"][2], d3.T)
    np.testing.assert_array_equal(res["bits_segments"][3], d4.T)
