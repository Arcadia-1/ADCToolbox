import numpy as np
import pytest

from adctoolbox import diagnose_calibration_matrix, ifilter
from adctoolbox.calibration import diagnose_calibration_matrix as diagnose_from_submodule


def _binary_counter_matrix(n_bits=8):
    codes = np.arange(2**n_bits, dtype=np.uint16)
    shifts = np.arange(n_bits - 1, -1, -1, dtype=np.uint16)
    return ((codes[:, None] >> shifts[None, :]) & 1).astype(float)


def _thermometer_matrix(n_samples=4096, n_units=64, fin_bin=17, amplitude=0.49):
    n = np.arange(n_samples)
    signal = 0.5 + amplitude * np.sin(2 * np.pi * fin_bin * n / n_samples)
    code = np.clip(np.floor(signal * n_units).astype(int), 0, n_units)
    return (np.arange(n_units)[None, :] < code[:, None]).astype(float)


def test_diagnose_calibration_matrix_exported_from_top_level_and_submodule():
    assert diagnose_calibration_matrix is diagnose_from_submodule


def test_healthy_binary_matrix_reports_rank_and_binary_status():
    bits = _binary_counter_matrix(8)

    diag = diagnose_calibration_matrix(bits)

    assert diag["shape"] == bits.shape
    assert diag["segment_lengths"].tolist() == [bits.shape[0]]
    assert diag["is_binary"] is True
    assert diag["binary_violation_fraction"] == 0.0
    assert diag["near_constant_columns"].size == 0
    assert diag["rank"] == bits.shape[1]
    assert diag["rank_with_offset"] == bits.shape[1] + 1
    assert np.isfinite(diag["condition_number"])
    assert diag["singular_values"].shape == (bits.shape[1],)


def test_continuous_matrix_reports_binary_violations():
    bits = _binary_counter_matrix(5)
    n = np.arange(bits.shape[0])
    continuous = bits + 0.05 * np.sin(2 * np.pi * n[:, None] / bits.shape[0])

    diag = diagnose_calibration_matrix(continuous)

    assert diag["is_binary"] is False
    assert diag["binary_violation_fraction"] > 0.0


def test_near_constant_columns_are_reported():
    bits = _binary_counter_matrix(4)
    bits[:, 1] = 0.0
    bits[:, 3] = 1.0

    diag = diagnose_calibration_matrix(bits)

    np.testing.assert_array_equal(diag["near_constant_columns"], [1, 3])
    assert diag["rank"] == 2


def test_ifiltered_thermometer_matrix_is_continuous_and_more_ill_conditioned():
    bits = _thermometer_matrix()
    filtered = ifilter(bits, [[0, 0.5 / 32]])

    raw_diag = diagnose_calibration_matrix(bits)
    filtered_diag = diagnose_calibration_matrix(filtered)

    assert raw_diag["is_binary"] is True
    assert filtered_diag["is_binary"] is False
    assert filtered_diag["binary_violation_fraction"] > 0.5
    assert filtered_diag["rank"] > 0
    assert filtered_diag["condition_number"] > raw_diag["condition_number"] * 100


def test_weight_physicality_diagnostics_are_scale_invariant_to_nominal():
    nominal = np.ones(8)
    flat = 0.25 * nominal
    oscillatory = np.array([1.0, -2.0, 3.0, -4.0, 3.0, -2.0, 1.0, -0.5])

    flat_diag = diagnose_calibration_matrix(
        _binary_counter_matrix(8),
        nominal_weights=nominal,
        weights=flat,
    )["weight_diagnostics"]
    bad_diag = diagnose_calibration_matrix(
        _binary_counter_matrix(8),
        nominal_weights=nominal,
        weights=oscillatory,
    )["weight_diagnostics"]

    assert flat_diag["negative_fraction"] == 0.0
    assert flat_diag["total_variation"] == pytest.approx(0.0)
    assert flat_diag["relative_deviation_from_nominal"] == pytest.approx(0.0)
    assert flat_diag["nominal_scale"] == pytest.approx(0.25)

    assert bad_diag["negative_fraction"] > 0.0
    assert bad_diag["total_variation"] > flat_diag["total_variation"] + 10.0
    assert bad_diag["relative_deviation_from_nominal"] > 1.0


def test_invalid_inputs_raise_clear_value_errors():
    with pytest.raises(ValueError, match="must not be empty"):
        diagnose_calibration_matrix([])
    with pytest.raises(ValueError, match="NaN or Inf"):
        diagnose_calibration_matrix(np.array([[0.0, np.nan]]))
    with pytest.raises(ValueError, match="length must match"):
        diagnose_calibration_matrix(
            _binary_counter_matrix(3),
            nominal_weights=np.ones(2),
            weights=np.ones(3),
        )
