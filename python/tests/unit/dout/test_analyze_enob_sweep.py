"""Unit tests for ENOB bit-sweep calibration modes."""

import importlib

import numpy as np
import pytest

from adctoolbox.dout import analyze_enob_sweep


def test_analyze_enob_sweep_rejects_unknown_calibration_mode():
    bits = np.array([[0, 1], [1, 0]], dtype=float)

    with pytest.raises(ValueError, match="calibration_mode"):
        analyze_enob_sweep(bits, calibration_mode="unknown", create_plot=False)


def test_prefix_mode_calibrates_full_bit_matrix_once(monkeypatch):
    module = importlib.import_module("adctoolbox.dout.analyze_enob_sweep")
    bits = np.eye(3, dtype=float)
    calls = []

    def fake_calibrate(input_bits, **kwargs):
        calls.append((input_bits.shape, kwargs))
        return {
            "weight": np.array([10.0, 1.0, 0.1]),
            "refined_frequency": 0.25,
        }

    def fake_analyze_spectrum(signal, **kwargs):
        return {"enob": float(np.sum(signal))}

    monkeypatch.setattr(module, "calibrate_weight_sine", fake_calibrate)
    monkeypatch.setattr(module, "analyze_spectrum", fake_analyze_spectrum)

    enob_sweep, n_bits_vec = analyze_enob_sweep(
        bits,
        freq=0,
        calibration_mode="prefix_of_full_calibration",
        create_plot=False,
    )

    assert [shape for shape, _ in calls] == [(3, 3)]
    np.testing.assert_allclose(enob_sweep, [10.0, 11.0, 11.1])
    np.testing.assert_array_equal(n_bits_vec, [1, 2, 3])


def test_recalibrate_mode_fixes_auto_frequency_then_recalibrates_subsets(monkeypatch):
    module = importlib.import_module("adctoolbox.dout.analyze_enob_sweep")
    bits = np.eye(3, dtype=float)
    calls = []

    def fake_calibrate(input_bits, **kwargs):
        calls.append((input_bits.shape, kwargs))
        n_cols = input_bits.shape[1]
        return {
            "refined_frequency": 0.123,
            "calibrated_signal": np.full(input_bits.shape[0], n_cols, dtype=float),
        }

    def fake_analyze_spectrum(signal, **kwargs):
        return {"enob": float(np.mean(signal))}

    monkeypatch.setattr(module, "calibrate_weight_sine", fake_calibrate)
    monkeypatch.setattr(module, "analyze_spectrum", fake_analyze_spectrum)

    enob_sweep, n_bits_vec = analyze_enob_sweep(
        bits,
        freq=0,
        calibration_mode="recalibrate_each_subset",
        frequency_policy="matlab",
        create_plot=False,
    )

    assert [shape for shape, _ in calls] == [(3, 3), (3, 1), (3, 2), (3, 3)]
    assert calls[0][1]["freq"] == 0
    assert calls[0][1]["frequency_policy"] == "matlab"
    assert [call[1]["freq"] for call in calls[1:]] == [0.123, 0.123, 0.123]
    assert [call[1]["force_search"] for call in calls[1:]] == [False, False, False]
    np.testing.assert_allclose(enob_sweep, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(n_bits_vec, [1, 2, 3])


def test_recalibrate_mode_marks_failed_subset_nan_and_continues(monkeypatch):
    module = importlib.import_module("adctoolbox.dout.analyze_enob_sweep")
    bits = np.eye(3, dtype=float)
    calls = []

    def fake_calibrate(input_bits, **kwargs):
        calls.append((input_bits.shape, kwargs))
        n_cols = input_bits.shape[1]
        if n_cols == 1:
            raise ValueError("not identifiable")
        return {
            "refined_frequency": 0.123,
            "calibrated_signal": np.full(input_bits.shape[0], n_cols, dtype=float),
        }

    def fake_analyze_spectrum(signal, **kwargs):
        return {"enob": float(np.mean(signal))}

    monkeypatch.setattr(module, "calibrate_weight_sine", fake_calibrate)
    monkeypatch.setattr(module, "analyze_spectrum", fake_analyze_spectrum)

    enob_sweep, n_bits_vec = analyze_enob_sweep(
        bits,
        freq=0,
        calibration_mode="recalibrate_each_subset",
        create_plot=False,
    )

    assert [shape for shape, _ in calls] == [(3, 3), (3, 1), (3, 2), (3, 3)]
    assert np.isnan(enob_sweep[0])
    np.testing.assert_allclose(enob_sweep[1:], [2.0, 3.0])
    np.testing.assert_array_equal(n_bits_vec, [1, 2, 3])
