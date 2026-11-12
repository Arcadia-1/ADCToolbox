"""Foreground calibration models must not be applied to a different feature layout.

A model used to store only its number of feature columns. Two channels with
history 1 and one channel with history 3 both give four columns, so a model
trained on one was applied to the other without complaint and returned an rms
error of 0.82 where the correct layout gives 5e-16.
"""
import json

import numpy as np
import pytest

from adctoolbox.calibration import (
    apply_calibration,
    build_calibration_features,
    fit_calibration_mlp,
    fit_calibration_polynomial,
)

RNG = np.random.default_rng(0)
TWO_CHANNELS = RNG.standard_normal((400, 2))
ONE_CHANNEL = RNG.standard_normal(400)


def _two_channel_case():
    built = build_calibration_features(TWO_CHANNELS, history=1)
    features = built["features"]
    reference = features @ np.array([1.0, -0.5, 0.25, 0.1]) + 0.3
    return built, reference


def test_builder_reports_channels_and_history():
    assert build_calibration_features(TWO_CHANNELS, history=1)["layout"] == {"channels": 2, "history": 1}
    assert build_calibration_features(ONE_CHANNEL, history=3)["layout"] == {"channels": 1, "history": 3}
    assert build_calibration_features(ONE_CHANNEL)["layout"] == {"channels": 1, "history": 0}


def test_model_records_layout_and_survives_json():
    built, reference = _two_channel_case()

    model = fit_calibration_polynomial(built, reference, degree=1)["model"]
    restored = json.loads(json.dumps(model))

    assert restored["feature_layout"] == {"channels": 2, "history": 1}
    np.testing.assert_allclose(
        apply_calibration(built, restored)["calibrated_signal"], reference, atol=1e-6)


def test_same_column_count_with_another_layout_is_rejected():
    built, reference = _two_channel_case()
    model = fit_calibration_polynomial(built, reference, degree=1)["model"]
    other = build_calibration_features(ONE_CHANNEL, history=3)
    assert other["features"].shape[1] == built["features"].shape[1]

    with pytest.raises(ValueError, match="feature layout mismatch"):
        apply_calibration(other, model)


def test_dict_and_matrix_inputs_give_the_same_fit_and_output():
    built, reference = _two_channel_case()

    from_dict = fit_calibration_polynomial(built, reference, degree=2)["model"]
    from_matrix = fit_calibration_polynomial(built["features"], reference, degree=2)["model"]

    assert "feature_layout" not in from_matrix
    np.testing.assert_allclose(from_dict["coefficients"], from_matrix["coefficients"])
    np.testing.assert_array_equal(
        apply_calibration(built, from_dict)["calibrated_signal"],
        apply_calibration(built["features"], from_dict)["calibrated_signal"],
    )


def test_models_without_a_recorded_layout_still_apply():
    # Models saved before the layout was recorded, or fitted from a matrix.
    built, reference = _two_channel_case()
    model = fit_calibration_polynomial(built["features"], reference, degree=1)["model"]

    np.testing.assert_allclose(
        apply_calibration(built, model)["calibrated_signal"], reference, atol=1e-6)


def test_mlp_rejects_validation_features_with_another_layout():
    built, reference = _two_channel_case()
    other = build_calibration_features(ONE_CHANNEL, history=3)

    with pytest.raises(ValueError, match="feature layout mismatch"):
        fit_calibration_mlp(
            built, reference,
            validation_features=other, validation_reference=reference[: len(other["features"])],
            hidden_units=2, max_iterations=2,
        )


def test_mlp_records_layout_and_reports_a_missing_seed():
    built, reference = _two_channel_case()

    fit = fit_calibration_mlp(
        built, reference, validation_features=built, validation_reference=reference,
        hidden_units=2, max_iterations=3, seed=None,
    )

    assert fit["model"]["feature_layout"] == {"channels": 2, "history": 1}
    assert fit["diagnostics"]["seed"] is None


def test_mlp_rejects_an_unusable_seed_before_fitting():
    built, reference = _two_channel_case()

    with pytest.raises((TypeError, ValueError)):
        fit_calibration_mlp(
            built, reference, validation_features=built, validation_reference=reference,
            hidden_units=2, max_iterations=3, seed="not-a-seed",
        )


def test_feature_dict_without_features_is_rejected():
    with pytest.raises(ValueError, match='"features"'):
        fit_calibration_polynomial({"layout": {"channels": 1, "history": 0}}, np.zeros(3))
