"""Unit tests for ramp-histogram INL/DNL analysis."""

import numpy as np
import pytest

from adctoolbox import analyze_inl_from_ramp
from adctoolbox.aout import compute_inl_from_ramp


def _codes_from_counts(counts, code_min=0):
    counts = np.asarray(counts, dtype=int)
    codes = np.arange(code_min, code_min + len(counts))
    return np.repeat(codes, counts)


def test_ideal_ramp_codes_have_zero_dnl_and_inl():
    codes = np.repeat(np.arange(16), 10)

    result = compute_inl_from_ramp(
        codes,
        num_bits=4,
        exclude_endpoints=False,
        endpoint="none",
    )

    assert np.array_equal(result["code"], np.arange(16))
    assert np.array_equal(result["counts"], np.full(16, 10))
    assert result["missing_codes"].size == 0
    assert np.allclose(result["dnl"], 0.0)
    assert np.allclose(result["inl"], 0.0)
    assert result["dnl_pp"] == pytest.approx(0.0)
    assert result["inl_pp"] == pytest.approx(0.0)


def test_missing_code_is_reported_with_minus_one_dnl():
    counts = np.full(16, 10)
    counts[5] = 0
    codes = _codes_from_counts(counts)

    result = compute_inl_from_ramp(
        codes,
        num_bits=4,
        exclude_endpoints=False,
        endpoint="none",
    )

    missing_idx = np.where(result["code"] == 5)[0][0]
    assert np.array_equal(result["missing_codes"], np.array([5]))
    assert result["counts"][missing_idx] == 0
    assert result["dnl"][missing_idx] == pytest.approx(-1.0)


def test_code_width_error_maps_to_expected_dnl():
    counts = np.full(16, 20)
    counts[6] = 30
    counts[7] = 10
    codes = _codes_from_counts(counts)

    result = compute_inl_from_ramp(
        codes,
        num_bits=4,
        exclude_endpoints=False,
        endpoint="none",
    )

    assert result["ideal_count"] == pytest.approx(20.0)
    assert result["dnl"][6] == pytest.approx(0.5)
    assert result["dnl"][7] == pytest.approx(-0.5)


def test_endpoint_exclusion_removes_partial_ramp_edges():
    counts = np.full(16, 10)
    counts[0] = 1
    counts[-1] = 1
    codes = _codes_from_counts(counts)

    included = compute_inl_from_ramp(
        codes,
        num_bits=4,
        exclude_endpoints=False,
        endpoint="none",
    )
    excluded = compute_inl_from_ramp(
        codes,
        num_bits=4,
        exclude_endpoints=True,
        endpoint="none",
    )

    assert included["dnl"][0] < 0
    assert included["dnl"][-1] < 0
    assert np.array_equal(excluded["code"], np.arange(1, 15))
    assert np.array_equal(excluded["counts"], np.full(14, 10))
    assert np.allclose(excluded["dnl"], 0.0)
    assert np.allclose(excluded["inl"], 0.0)


def test_endpoint_corrections_are_explicit():
    counts = np.full(8, 10)
    counts[2] = 14
    counts[5] = 6
    codes = _codes_from_counts(counts)

    raw = compute_inl_from_ramp(
        codes,
        num_bits=3,
        exclude_endpoints=False,
        endpoint="none",
    )
    endpoints = compute_inl_from_ramp(
        codes,
        num_bits=3,
        exclude_endpoints=False,
        endpoint="endpoints",
    )
    fit = compute_inl_from_ramp(
        codes,
        num_bits=3,
        exclude_endpoints=False,
        endpoint="fit",
    )

    assert np.array_equal(raw["raw_inl"], raw["inl"])
    assert endpoints["inl"][0] == pytest.approx(0.0)
    assert endpoints["inl"][-1] == pytest.approx(0.0)
    assert abs(np.polyfit(fit["code"], fit["inl"], deg=1)[0]) < 1e-12


def test_analyze_inl_from_ramp_plots_with_existing_dnl_inl_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    codes = np.repeat(np.arange(16), 8)
    fig, ax = plt.subplots()

    result = analyze_inl_from_ramp(
        codes,
        num_bits=4,
        exclude_endpoints=False,
        create_plot=True,
        ax=ax,
    )

    assert {"code", "counts", "dnl", "inl", "missing_codes"} <= set(result)
    assert len(fig.axes) == 2
    assert fig.axes[0].get_ylabel() == "DNL (LSB)"
    assert fig.axes[1].get_ylabel() == "INL (LSB)"
    assert len(fig.axes[0].lines) >= 1
    assert len(fig.axes[1].lines) >= 1
    plt.close(fig)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"codes": []}, "must not be empty"),
        ({"codes": [0, 1.25, 2]}, "integer ADC codes"),
        ({"codes": [0, 1, 16], "num_bits": 4}, "within"),
        ({"codes": [0, 1, 2], "num_bits": 0}, "positive integer"),
        ({"codes": [0, 1, 2], "code_min": 4, "code_max": 3}, "code_max"),
        ({"codes": [0, 1, 2], "endpoint": "mean"}, "endpoint"),
        ({"codes": [0, 1, 2], "endpoint": None}, "endpoint"),
    ],
)
def test_compute_inl_from_ramp_validates_inputs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        compute_inl_from_ramp(**kwargs)


def test_analyze_inl_from_ramp_rejects_future_input_types():
    with pytest.raises(ValueError, match="input_type='codes'"):
        analyze_inl_from_ramp([0, 1, 2], input_type="voltage", create_plot=False)
