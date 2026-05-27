import numpy as np

from adctoolbox import (
    calibrate_weight_sine,
    quick_sndr,
    sar_apply_cap_mismatch,
    sar_convert,
    sar_reconstruct,
)


def _radix18_integer_weights_16bit() -> np.ndarray:
    raw = np.array(
        [
            29127,
            16182,
            8990,
            4995,
            2775,
            1542,
            856,
            476,
            264,
            147,
            82,
            45,
            25,
            14,
            8,
            4,
            2,
            1,
        ],
        dtype=float,
    )
    return raw / (raw.sum() + raw[-1])


def _enob(trace: np.ndarray) -> float:
    centered = trace - np.mean(trace)
    return quick_sndr(
        centered,
        fs=1.0,
        win_type="rectangular",
        side_bin=0,
        max_scale_range=None,
    )["enob"]


def test_auto_frequency_defaults_to_fine_search_for_sar_weight_calibration():
    n_samples = 2048
    train_bin = 251
    test_bin = 307
    n = np.arange(n_samples)

    nominal_weights = _radix18_integer_weights_16bit()
    actual_weights = sar_apply_cap_mismatch(
        nominal_weights,
        sigma=0.10,
        rng=np.random.default_rng(20260526),
    )

    vin_train = 0.5 + 0.499 * np.sin(2 * np.pi * train_bin * n / n_samples)
    vin_test = 0.5 + 0.499 * np.sin(
        2 * np.pi * test_bin * n / n_samples + 0.37
    )
    bits_train = sar_convert(vin_train, actual_weights)
    bits_test = sar_convert(vin_test, actual_weights)

    before = sar_reconstruct(bits_test, nominal_weights)
    no_search = calibrate_weight_sine(
        bits_train,
        nominal_weights=nominal_weights,
        force_search=False,
    )
    default_search = calibrate_weight_sine(bits_train, nominal_weights=nominal_weights)

    before_enob = _enob(before)
    no_search_enob = _enob(bits_test.astype(float) @ no_search["weight"])
    default_enob = _enob(bits_test.astype(float) @ default_search["weight"])

    assert before_enob < 12.0
    assert default_enob > 15.8
    assert default_enob > no_search_enob + 1.0


def test_explicit_frequency_stays_fixed_unless_force_search_is_enabled():
    n_samples = 2048
    train_bin = 251
    freq = train_bin / n_samples
    n = np.arange(n_samples)

    nominal_weights = _radix18_integer_weights_16bit()
    actual_weights = sar_apply_cap_mismatch(
        nominal_weights,
        sigma=0.10,
        rng=np.random.default_rng(20260526),
    )
    vin_train = 0.5 + 0.499 * np.sin(2 * np.pi * freq * n)
    bits_train = sar_convert(vin_train, actual_weights)

    result = calibrate_weight_sine(
        bits_train,
        freq=freq,
        nominal_weights=nominal_weights,
    )

    assert result["refined_frequency"] == freq
