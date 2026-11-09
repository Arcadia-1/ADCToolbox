"""Requested HD2-HD5 levels hold relative to the output fundamental.

Each level is measured independently from a coherent rectangular-window FFT
of the generator output, so there is no leakage or window estimate involved.
"""

import numpy as np
import pytest

from adctoolbox.siggen import ADC_Signal_Generator

N = 8192
BIN = 37  # coherent, coprime with N, and 5 * BIN stays below Nyquist
A = 0.4
DC = 0.5
ORDERS = (2, 3, 4, 5)


def _generator(amplitude=A):
    return ADC_Signal_Generator(N=N, Fs=float(N), Fin=float(BIN), A=amplitude, DC=DC)


def _phasors(y):
    """Complex amplitudes of the fundamental (index 1) and harmonics 2-5."""
    spectrum = np.fft.rfft(np.asarray(y, dtype=float) - DC) * 2 / N
    return {order: spectrum[order * BIN] for order in range(1, 6)}


def _assert_levels(y, requested_dbc):
    phasors = _phasors(y)
    fundamental = abs(phasors[1])
    for order in ORDERS:
        ratio = abs(phasors[order]) / fundamental
        if order in requested_dbc:
            assert 20 * np.log10(ratio) == pytest.approx(requested_dbc[order], abs=1e-9)
        else:
            assert ratio < 1e-12


def test_issue_example_hd3_is_exact_relative_to_output_fundamental():
    n, k = 65536, 137
    gen = ADC_Signal_Generator(N=n, Fs=float(n), Fin=float(k), A=.4, DC=.5)
    sp = np.fft.rfft(gen.apply_static_nonlinearity_hd(hd3_dB=-20) - .5)

    assert 20 * np.log10(abs(sp[3 * k]) / abs(sp[k])) == pytest.approx(-20, abs=1e-9)


@pytest.mark.parametrize("level_dbc", [-20.0, -40.0, -60.0])
@pytest.mark.parametrize("order", ORDERS)
def test_single_order_hits_its_level_without_coupling(order, level_dbc):
    gen = _generator()
    clean = gen.get_clean_signal()
    y = gen.apply_static_nonlinearity_hd(**{f"hd{order}_dB": level_dbc})

    _assert_levels(y, {order: level_dbc})
    np.testing.assert_allclose(_phasors(y)[1], _phasors(clean)[1], rtol=1e-12)

    # The harmonic keeps the phase produced by a positive x**order coefficient.
    power_term = _phasors(DC + (clean - DC) ** order)[order]
    harmonic = _phasors(y)[order]
    np.testing.assert_allclose(harmonic / abs(harmonic), power_term / abs(power_term), atol=1e-9)


@pytest.mark.parametrize(
    "requested_dbc",
    [
        {2: -30.0, 3: -25.0, 4: -40.0, 5: -35.0},
        {2: -20.0, 3: -20.0, 4: -20.0, 5: -20.0},
        {3: -40.0, 5: -60.0},
        {2: -60.0, 4: -20.0},
    ],
)
def test_combined_orders_hit_every_level(requested_dbc):
    gen = _generator()
    y = gen.apply_static_nonlinearity_hd(**{f"hd{order}_dB": v for order, v in requested_dbc.items()})

    _assert_levels(y, requested_dbc)
    np.testing.assert_allclose(_phasors(y)[1], _phasors(gen.get_clean_signal())[1], rtol=1e-12)


def test_operating_point_maps_to_itself():
    gen = _generator()
    y = gen.apply_static_nonlinearity_hd(np.full(N, DC), hd2_dB=-20, hd3_dB=-25, hd4_dB=-30, hd5_dB=-35)

    np.testing.assert_array_equal(y, np.full(N, DC))


def test_hd2_only_keeps_the_square_law_polynomial():
    gen = _generator()
    x = gen.get_clean_signal() - DC
    ratio = 10 ** (-30 / 20)

    y = gen.apply_static_nonlinearity_hd(hd2_dB=-30)

    np.testing.assert_allclose(y, DC + x + 2 * ratio / A * x ** 2, rtol=1e-12, atol=1e-15)


def test_explicit_clean_input_matches_internal_signal():
    gen = _generator()
    settings = dict(hd2_dB=-50, hd3_dB=-45, hd5_dB=-70)

    np.testing.assert_array_equal(
        gen.apply_static_nonlinearity_hd(gen.get_clean_signal(), **settings),
        gen.apply_static_nonlinearity_hd(None, **settings),
    )


def test_integer_amplitude_is_supported():
    gen = ADC_Signal_Generator(N=N, Fs=float(N), Fin=float(BIN), A=1, DC=DC)

    _assert_levels(gen.apply_static_nonlinearity_hd(hd3_dB=-30, hd4_dB=-50), {3: -30.0, 4: -50.0})


def test_zero_amplitude_rejects_requested_harmonics():
    gen = _generator(amplitude=0.0)

    with pytest.raises(ValueError, match="nonzero amplitude"):
        gen.apply_static_nonlinearity_hd(hd3_dB=-40)
    np.testing.assert_array_equal(gen.apply_static_nonlinearity_hd(), np.full(N, DC))
