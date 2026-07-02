import numpy as np
import pytest
from adctoolbox.calibration.calibrate_weight_sine import calibrate_weight_sine
from adctoolbox.calibration._patch_rank_deficiency import _patch_rank_deficiency, _recover_rank_deficiency

def test_patch_rank_deficiency_logic():
    """
    Test the patching logic for ADC calibration including:
    - Independent bits (remain untouched)
    - Dead bits (constant 0 or 1, should be dropped)
    - Correlated bits (synchronized bits, should be merged with nominal ratios)
    """
    nominal_weights = [128, 64, 32, 16, 8, 4, 2, 1]

    rng = np.random.default_rng(2026062207)
    bits_input = rng.integers(0, 2, (2048, len(nominal_weights)))
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

    rng = np.random.default_rng(2026062208)
    bits_input = rng.integers(0, 2, (2048, len(nominal_weights)))
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

    rng = np.random.default_rng(2026062209)
    bits_input = rng.integers(0, 2, (2048, len(nominal_weights)))
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


def test_all_constant_bits_raise_not_identifiable_value_error():
    """All-constant bit captures have no AC information for sine calibration."""
    bits = np.ones((64, 5), dtype=int)
    nominal_weights = 2.0 ** np.arange(4, -1, -1)

    with pytest.raises(ValueError, match="not identifiable"):
        calibrate_weight_sine(bits, freq=1 / 64, nominal_weights=nominal_weights)


def test_recover_rank_deficiency_empty_effective_weights_raises_value_error():
    """The low-level recovery helper should fail clearly instead of indexing an empty vector."""
    bit_to_col_map = np.array([-1, -1, -1])
    bit_weight_ratios = np.zeros(3)

    with pytest.raises(ValueError, match="No effective bit weights"):
        _recover_rank_deficiency(np.array([]), bit_to_col_map, bit_weight_ratios)


def test_partial_constant_bits_warn_and_report_rank_patch_metadata():
    """Constant columns are unobservable, not physical zero-weight evidence."""
    n_samples = 256
    freq = 7 / n_samples
    n = np.arange(n_samples)
    active_bit = (np.sin(2 * np.pi * freq * n) > 0).astype(int)
    bits = np.column_stack([
        active_bit,
        np.ones(n_samples, dtype=int),
        np.zeros(n_samples, dtype=int),
        np.ones(n_samples, dtype=int),
        np.zeros(n_samples, dtype=int),
    ])
    nominal_weights = 2.0 ** np.arange(4, -1, -1)

    with pytest.warns(UserWarning, match="constant or otherwise unobservable"):
        result = calibrate_weight_sine(bits, freq=freq, nominal_weights=nominal_weights)

    rank_patch = result["rank_patch"]
    assert rank_patch["applied"] is True
    assert rank_patch["bit_width_effective"] == 1
    np.testing.assert_array_equal(rank_patch["bit_to_col_map"], [0, -1, -1, -1, -1])
    np.testing.assert_array_equal(rank_patch["dropped_constant_bits"], [1, 2, 3, 4])
    np.testing.assert_array_equal(rank_patch["unmapped_bits"], [1, 2, 3, 4])
    np.testing.assert_allclose(result["weight"][rank_patch["unmapped_bits"]], 0.0)
    assert result["weight"][0] != pytest.approx(0.0)
