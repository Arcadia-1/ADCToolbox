"""Auto side_bin: string alias and the warning for components hidden in the signal window.

With non-coherent sampling and a Hann window, the auto side_bin follows the
tone's leakage down to the noise floor and can span tens of bins. A spur inside
that window is summed into the signal power instead of being reported as the
SFDR spur, so compute_spectrum/quick_sndr warn about it. The warning must not
change any metric and must stay quiet on clean records.
"""
import warnings

import numpy as np
import pytest

from adctoolbox import analyze_spectrum, quick_sndr
from adctoolbox.spectrum.compute_spectrum import compute_spectrum

N = 1024
F_TONE = 0.2937  # non-coherent: 300.75 bins
HIDDEN = "auto side_bin"


def _quantize(x, bits=16):
    scale = 2 ** (bits - 1)
    return np.round(x * scale) / scale


def _tone(freq, amplitude=0.9, phase=0.0):
    n = np.arange(N)
    return amplitude * np.sin(2 * np.pi * freq * n + phase)


def _record_with_hidden_spur():
    # -80 dBc spur about 45 bins from the carrier, inside the ~66-bin auto window.
    return _quantize(_tone(F_TONE) + _tone(0.25, amplitude=0.9e-4))


def _assert_same_metrics(left, right):
    assert left.keys() == right.keys()
    for key in left:
        np.testing.assert_array_equal(left[key], right[key], err_msg=key)


def _hidden_warnings(func, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = func(*args, **kwargs)
    return result, [w for w in caught if HIDDEN in str(w.message)]


def test_hidden_spur_inside_wide_auto_window_warns():
    result, caught = _hidden_warnings(compute_spectrum, _record_with_hidden_spur(), max_scale_range=2)

    side_bin = result["plot_data"]["side_bin"]
    assert side_bin > 45, "precondition: the spur must sit inside the auto signal window"
    assert len(caught) == 1
    assert issubclass(caught[0].category, UserWarning)
    message = str(caught[0].message)
    assert "bin 256" in message
    assert "explicit side_bin" in message


def test_analyze_spectrum_and_quick_sndr_auto_report_the_hidden_spur():
    record = _record_with_hidden_spur()

    _, caught = _hidden_warnings(analyze_spectrum, record, max_scale_range=2, create_plot=False)
    assert len(caught) == 1

    _, caught = _hidden_warnings(quick_sndr, record, max_scale_range=2, side_bin="auto")
    assert len(caught) == 1
    assert str(caught[0].message).startswith("quick_sndr:")


def test_warning_does_not_change_auto_metrics():
    record = _record_with_hidden_spur()
    auto, _ = _hidden_warnings(compute_spectrum, record, max_scale_range=2)
    explicit, caught = _hidden_warnings(
        compute_spectrum, record, max_scale_range=2, side_bin=auto["plot_data"]["side_bin"]
    )

    assert caught == [], "an explicit side_bin is the user's choice and is not second-guessed"
    _assert_same_metrics(auto["metrics"], explicit["metrics"])


@pytest.mark.parametrize("seed", range(5))
def test_clean_noncoherent_records_do_not_warn(seed):
    rng = np.random.default_rng(seed)
    freq = rng.uniform(0.05, 0.45)
    record = _quantize(_tone(freq, phase=rng.uniform(0, 2 * np.pi)), bits=14)

    _, caught = _hidden_warnings(compute_spectrum, record, max_scale_range=2)

    assert caught == []


@pytest.mark.parametrize(
    "win_type",
    ["boxcar", "hann", "hamming", "blackman", "blackmanharris", "flattop", "kaiser", "chebwin"],
)
def test_noise_free_noncoherent_tone_does_not_warn_for_any_window(win_type):
    record = 0.4 * np.sin(2 * np.pi * 0.1 * np.arange(512))

    _, caught = _hidden_warnings(compute_spectrum, record, win_type=win_type)

    assert caught == []


def test_coherent_spur_next_to_carrier_does_not_warn():
    n = np.arange(N)
    record = np.sin(2 * np.pi * 301 / N * n) + 1e-4 * np.sin(2 * np.pi * 304 / N * n)

    result, caught = _hidden_warnings(compute_spectrum, record, max_scale_range=2)

    assert caught == []
    assert result["metrics"]["sfdr_dbc"] == pytest.approx(80.0, abs=1e-6)


def test_side_bin_auto_string_matches_none():
    record = _quantize(_tone(F_TONE), bits=12)

    by_none = compute_spectrum(record, max_scale_range=2, side_bin=None)
    by_auto = compute_spectrum(record, max_scale_range=2, side_bin="auto")

    assert by_auto["plot_data"]["side_bin"] == by_none["plot_data"]["side_bin"]
    _assert_same_metrics(by_auto["metrics"], by_none["metrics"])

    via_analyze = analyze_spectrum(record, max_scale_range=2, side_bin="auto", create_plot=False)
    assert via_analyze["sndr_dbc"] == by_none["metrics"]["sndr_dbc"]


@pytest.mark.parametrize("side_bin", ["bad", "3", "AUTO"])
def test_side_bin_rejects_other_strings(side_bin):
    with pytest.raises(TypeError, match="side_bin"):
        compute_spectrum(_tone(F_TONE), side_bin=side_bin)


def test_numeric_side_bin_is_coerced_like_quick_sndr():
    record = _quantize(_tone(F_TONE), bits=12)

    as_float = compute_spectrum(record, max_scale_range=2, side_bin=3.9)
    as_int = compute_spectrum(record, max_scale_range=2, side_bin=3)
    negative = compute_spectrum(record, max_scale_range=2, side_bin=-2)
    zero = compute_spectrum(record, max_scale_range=2, side_bin=0)

    _assert_same_metrics(as_float["metrics"], as_int["metrics"])
    _assert_same_metrics(negative["metrics"], zero["metrics"])
