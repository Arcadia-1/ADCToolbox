"""MATLAB-compatible oversampling API tests."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
from scipy import signal

from adctoolbox import (
    ifilter,
    ntf_analyzer,
    ntfperf,
    perfosr,
    sweep_performance_vs_osr,
)
from adctoolbox.oversampling import ifilter as ifilter_from_submodule


def test_ifilter_keeps_requested_band_and_rejects_others():
    n = 64
    t = np.arange(n)
    in_band = np.sin(2 * np.pi * 3 * t / n)
    out_of_band = 0.25 * np.sin(2 * np.pi * 20 * t / n)
    x = in_band + out_of_band

    y = ifilter(x, [0, 0.1]).ravel()

    np.testing.assert_allclose(y, in_band, atol=1e-12)


def test_ifilter_matrix_orientation_matches_matlab_rule():
    n = 32
    t = np.arange(n)
    x1 = np.sin(2 * np.pi * 2 * t / n)
    x2 = np.sin(2 * np.pi * 10 * t / n)

    wide_input = np.vstack([x1, x2])
    y = ifilter_from_submodule(wide_input, [0, 0.2])

    assert y.shape == (n, 2)
    np.testing.assert_allclose(y[:, 0], x1, atol=1e-12)
    np.testing.assert_allclose(y[:, 1], 0.0, atol=1e-12)


def test_ifilter_rejects_invalid_frequency_bands():
    x = np.ones(16)

    for bands in ([[0.1, 0.6]], [[-0.1, 0.2]], [[0.1, 0.2, 0.3]]):
        try:
            ifilter(x, bands)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for bands={bands}")


def test_perfosr_returns_matlab_order_and_matches_existing_core():
    n = 1024
    t = np.arange(n)
    data = 0.5 * np.sin(2 * np.pi * 0.05 * t) + 0.002 * np.random.RandomState(0).normal(size=n)
    osr = np.array([2, 4, 8, 16])

    osr_out, sndr, sfdr, enob = perfosr(data, osr=osr, disp=False)
    core = sweep_performance_vs_osr(data, osr=osr, create_plot=False)

    np.testing.assert_allclose(osr_out, core["osr"])
    np.testing.assert_allclose(sndr, core["sndr"])
    np.testing.assert_allclose(sfdr, core["sfdr"])
    np.testing.assert_allclose(enob, core["enob"])


def test_perfosr_rejects_invalid_matlab_style_parameters():
    data = np.sin(2 * np.pi * 0.05 * np.arange(128))

    for kwargs in ({"osr": [1, 0]}, {"harmonic": 0}, {"smooth": 0}):
        try:
            perfosr(data, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for kwargs={kwargs}")


def test_ntfperf_alias_matches_ntf_analyzer():
    ntf = signal.TransferFunction([1, -1], [1, 0], dt=1)

    expected = ntf_analyzer(ntf, 0, 0.5 / 16, is_plot=0)
    actual = ntfperf(ntf, 0, 0.5 / 16, disp=False)

    assert actual == expected
