import numpy as np
import pytest

from adctoolbox.calibration import calibrate_weight_sine


def _load_wcalsin_reference(project_root):
    bits = np.loadtxt(
        project_root / "reference_dataset/dout_SAR_12b_weight_2.csv",
        delimiter=",",
    )
    freq = float(
        np.loadtxt(
            project_root
            / "reference_output/dout_SAR_12b_weight_2/test_wcalsine/freqCal_matlab.csv",
            delimiter=",",
        )
    )
    weights = np.loadtxt(
        project_root
        / "reference_output/dout_SAR_12b_weight_2/test_wcalsine/weight_matlab.csv",
        delimiter=",",
    )
    return bits, freq, weights


def test_auto_search_converges_to_matlab_wcalsin_reference(project_root):
    bits, freq_ref, weights_ref = _load_wcalsin_reference(project_root)

    result = calibrate_weight_sine(bits, freq=0, harmonic_order=5)

    assert result["frequency_policy"] == "python"
    assert result["initial_frequency"] == pytest.approx(
        0.299927615783513,
        abs=1e-12,
    )
    assert result["refined_frequency"] == pytest.approx(freq_ref, abs=1e-10)
    np.testing.assert_allclose(result["weight"], weights_ref, atol=1e-10, rtol=0)


def test_matlab_frequency_policy_uses_wcalsin_coarse_estimator(project_root):
    bits, freq_ref, weights_ref = _load_wcalsin_reference(project_root)

    result = calibrate_weight_sine(
        bits,
        freq=0,
        harmonic_order=5,
        frequency_policy="matlab",
    )

    assert result["frequency_policy"] == "matlab"
    assert result["initial_frequency"] == pytest.approx(
        0.299926916527268,
        abs=1e-12,
    )
    assert result["refined_frequency"] == pytest.approx(freq_ref, abs=1e-10)
    np.testing.assert_allclose(result["weight"], weights_ref, atol=1e-10, rtol=0)


def test_frequency_policy_does_not_change_explicit_fixed_frequency(project_root):
    bits, freq_ref, weights_ref = _load_wcalsin_reference(project_root)

    result_python = calibrate_weight_sine(
        bits,
        freq=freq_ref,
        harmonic_order=5,
        frequency_policy="python",
    )
    result_matlab = calibrate_weight_sine(
        bits,
        freq=freq_ref,
        harmonic_order=5,
        frequency_policy="matlab",
    )

    assert result_python["initial_frequency"] == freq_ref
    assert result_matlab["initial_frequency"] == freq_ref
    assert result_python["refined_frequency"] == freq_ref
    assert result_matlab["refined_frequency"] == freq_ref
    np.testing.assert_allclose(
        result_python["weight"],
        result_matlab["weight"],
        atol=0,
        rtol=0,
    )
    np.testing.assert_allclose(
        result_python["weight"],
        weights_ref,
        atol=2e-9,
        rtol=0,
    )


def test_frequency_policy_rejects_unknown_values():
    bits = np.array([[0, 1], [1, 0], [1, 1]], dtype=float)

    with pytest.raises(ValueError, match="frequency_policy"):
        calibrate_weight_sine(bits, frequency_policy="octave")
