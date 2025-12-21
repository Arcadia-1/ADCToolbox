import numpy as np
from adctoolbox.calibration._patch_rank_deficiency import _patch_rank_deficiency, _recover_rank_deficiency

def test_patch_rank_deficiency_logic():
    """
    Test the patching logic for ADC calibration including:
    - Independent bits (remain untouched)
    - Dead bits (constant 0 or 1, should be dropped)
    - Correlated bits (synchronized bits, should be merged with nominal ratios)
    """
    nominal_weights = [128, 64, 32, 16, 8, 4, 2, 1]

    bits_input = np.random.randint(0, 2, (2048, len(nominal_weights)))
    # Create dependencies:
    bits_input[:, 1] = 0
    bits_input[:, 2] = 1
    bits_input[:, 6] = bits_input[:, 0].copy()
    
    # --- Run Patching Function ---
    print()
    result = _patch_rank_deficiency(bits_input, nominal_weights, verbose=2)
    
    bits_eff = result["bits_effective"]
    bit_to_col_map = result["bit_to_col_map"]
    bit_weight_ratios = result["bit_weight_ratios"]
    bit_width_effective = result["bit_width_effective"]

    print(f"\n[bits_effective shape]: {bits_eff.shape}")
    print(f"[bit_to_col_map]: {bit_to_col_map}")
    print(f"[bit_weight_ratios]: {bit_weight_ratios}")
    print(f"[bit_width_effective]: {bit_width_effective}")


    # --- Assertions ---
    expected_eff_width = 5
    assert bit_width_effective == expected_eff_width, f"Expected 5 effective columns, got {bit_width_effective}"
    assert bits_eff.shape[1] == expected_eff_width

    expected_map = [0, -1, -1, 1, 2, 3, 0, 4]
    np.testing.assert_array_equal(bit_to_col_map, expected_map)

    expected_ratios = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.015625, 1.0]
    np.testing.assert_allclose(bit_weight_ratios, expected_ratios, atol=1e-10)
    
    expected_col_0 = bits_input[:, 0] + 0.015625 * bits_input[:, 6]
    np.testing.assert_allclose(bits_eff[:, 0], expected_col_0)

def test_patch_rank_deficiency_logic_reverse():
    """
    Test the patching logic for ADC calibration including:
    - Independent bits (remain untouched)
    - Dead bits (constant 0 or 1, should be dropped)
    - Correlated bits (synchronized bits, should be merged with nominal ratios)
    """
    nominal_weights = [128, 64, 32, 16, 8, 4, 2, 1][::-1]

    bits_input = np.random.randint(0, 2, (2048, len(nominal_weights)))
    # Create dependencies:
    bits_input[:, 1] = 0
    bits_input[:, 2] = 1
    bits_input[:, 6] = bits_input[:, 0].copy()
    
    # --- Run Patching Function ---
    print()
    result = _patch_rank_deficiency(bits_input, nominal_weights, verbose=2)
    
    bits_eff = result["bits_effective"]
    bit_to_col_map = result["bit_to_col_map"]
    bit_weight_ratios = result["bit_weight_ratios"]
    bit_width_effective = result["bit_width_effective"]

    print(f"\n[bits_effective shape]: {bits_eff.shape}")
    print(f"[bit_to_col_map]: {bit_to_col_map}")
    print(f"[bit_weight_ratios]: {bit_weight_ratios}")
    print(f"[bit_width_effective]: {bit_width_effective}")


    # --- Assertions ---
    expected_eff_width = 5
    assert bit_width_effective == expected_eff_width, f"Expected 5 effective columns, got {bit_width_effective}"
    assert bits_eff.shape[1] == expected_eff_width

    expected_map = [0, -1, -1, 1, 2, 3, 0, 4]
    np.testing.assert_array_equal(bit_to_col_map, expected_map)

    expected_ratios = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 64, 1.0]
    np.testing.assert_allclose(bit_weight_ratios, expected_ratios, atol=1e-10)
    
    expected_col_0 = bits_input[:, 0] + 64 * bits_input[:, 6]
    np.testing.assert_allclose(bits_eff[:, 0], expected_col_0)


def test_patch_rank_deficiency_logic_recover():
    """
    Test the patching logic for ADC calibration including:
    - Independent bits (remain untouched)
    - Dead bits (constant 0 or 1, should be dropped)
    - Correlated bits (synchronized bits, should be merged with nominal ratios)
    """
    nominal_weights = [128, 64, 32, 16, 8, 4, 2, 1][::-1]

    bits_input = np.random.randint(0, 2, (2048, len(nominal_weights)))
    # Create dependencies:
    bits_input[:, 1] = 0
    bits_input[:, 2] = 1
    bits_input[:, 6] = bits_input[:, 0].copy()
    
    # --- Run Patching Function ---
    print()
    result = _patch_rank_deficiency(bits_input, nominal_weights, verbose=2)
    
    bit_to_col_map = result["bit_to_col_map"]
    bit_weight_ratios = result["bit_weight_ratios"]

    mock_w_eff = np.array([1, 2, 4, 8, 16])
    weights_recovered = _recover_rank_deficiency(
        mock_w_eff, 
        bit_to_col_map, 
        bit_weight_ratios
    )

    print(f"\n[weights_recovered]: {weights_recovered}")


    # --- Assertions ---
    assert weights_recovered[1] == 0.0
    assert weights_recovered[2] == 0.0