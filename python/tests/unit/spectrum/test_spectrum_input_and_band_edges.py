"""Spectrum input handling and auto side_bin next to DC / the band edge."""
import warnings

import numpy as np
import pytest

from adctoolbox.spectrum.compute_spectrum import compute_spectrum
from adctoolbox.spectrum._window import _get_default_side_bin


def _noisy_tone(n, tone_bin, snr_db=60.0, amplitude=0.5, seed=0):
    rng = np.random.default_rng(seed)
    sigma = amplitude / np.sqrt(2) * 10 ** (-snr_db / 20)
    return amplitude * np.sin(2 * np.pi * tone_bin * np.arange(n) / n + 0.3) + sigma * rng.standard_normal(n)


@pytest.mark.parametrize(
    "win_type, tone_bin",
    [("blackmanharris", 2), ("blackmanharris", 3), ("kaiser", 5), ("blackman", 1), ("hann", 1)],
)
def test_auto_side_bin_keeps_window_lobe_for_tone_near_dc(win_type, tone_bin):
    # The auto width was clamped to the distance from DC, so a
    # Blackman-Harris tone at bin 2 kept side_bin=2 and the rest of its own
    # lobe was counted as distortion: SNDR 38.9 dB for a 60 dB capture.
    signal = _noisy_tone(4096, tone_bin)
    minimum = _get_default_side_bin(win_type)

    auto = compute_spectrum(signal, max_scale_range=1.0, win_type=win_type)
    explicit = compute_spectrum(signal, max_scale_range=1.0, win_type=win_type, side_bin=minimum)

    assert auto["plot_data"]["side_bin"] >= minimum
    assert auto["metrics"]["sndr_dbc"] == pytest.approx(explicit["metrics"]["sndr_dbc"], abs=1e-9)


def test_auto_side_bin_keeps_window_lobe_for_tone_near_band_edge():
    # OSR 16, N = 8192: the band ends at bin 256. Old result: SNDR 16.9 dB.
    signal = _noisy_tone(8192, 253)

    auto = compute_spectrum(signal, max_scale_range=1.0, win_type="blackmanharris", osr=16)
    explicit = compute_spectrum(
        signal, max_scale_range=1.0, win_type="blackmanharris", osr=16, side_bin=3,
    )

    assert auto["plot_data"]["side_bin"] == 3
    assert auto["metrics"]["sndr_dbc"] == pytest.approx(explicit["metrics"]["sndr_dbc"], abs=1e-9)
    assert auto["metrics"]["sndr_dbc"] > 65


@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.uint16, np.float32])
def test_integer_codes_give_the_same_metrics_as_float(dtype):
    # (max - min) of int16 codes spanning +/-30000 wrapped around, putting the
    # signal at +20.7 dBFS and the noise floor 20.7 dB too high.
    n = np.arange(4096)
    codes = np.round(30000 * np.sin(2 * np.pi * 301 * n / 4096)).astype(np.int64)
    if dtype is np.uint16:
        codes = codes + 30000

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        typed = compute_spectrum(codes.astype(dtype))["metrics"]
    reference = compute_spectrum(codes.astype(np.float64))["metrics"]

    assert typed["sig_pwr_dbfs"] == pytest.approx(0.0, abs=1e-3)
    for key in ("sig_pwr_dbfs", "noise_floor_dbfs", "sndr_dbc", "sfdr_dbc"):
        assert typed[key] == pytest.approx(reference[key], abs=1e-3), key


def test_range_as_numpy_array_matches_tuple():
    signal = 0.5 + 0.4 * np.sin(2 * np.pi * 301 * np.arange(1024) / 1024)

    from_array = compute_spectrum(signal, max_scale_range=np.array([0.0, 1.0]))["metrics"]
    from_tuple = compute_spectrum(signal, max_scale_range=(0.0, 1.0))["metrics"]

    assert from_array["sig_pwr_dbfs"] == from_tuple["sig_pwr_dbfs"]
    assert from_array["sig_pwr_dbfs"] == pytest.approx(20 * np.log10(0.8), abs=1e-6)


def test_range_array_of_wrong_length_is_rejected():
    with pytest.raises(ValueError, match="2 elements"):
        compute_spectrum(np.sin(np.arange(64.0)), max_scale_range=np.array([0.0, 0.5, 1.0]))


@pytest.mark.parametrize("coherent_averaging", [False, True])
def test_all_zero_records_are_left_out_of_the_average(coherent_averaging):
    # Power averaging counted the empty record and read -7.78 dBFS.
    tone = 0.5 * np.sin(2 * np.pi * 301 * np.arange(1024) / 1024)
    records = np.stack([tone, np.zeros(1024), tone])

    metrics = compute_spectrum(
        records, max_scale_range=1.0, coherent_averaging=coherent_averaging,
    )["metrics"]

    assert metrics["sig_pwr_dbfs"] == pytest.approx(-6.0206, abs=1e-3)


def test_osr_sweep_rejects_several_records_instead_of_splicing_them():
    # Four 60 dB records used to be concatenated and reported as -2 dB SNDR.
    from adctoolbox import sweep_performance_vs_osr

    records = np.stack([_noisy_tone(1024, 37, seed=s) for s in range(4)])

    with pytest.raises(ValueError, match="single 1-D record"):
        sweep_performance_vs_osr(records, create_plot=False)

    single = sweep_performance_vs_osr(records[0], create_plot=False)
    column = sweep_performance_vs_osr(records[0][:, None], create_plot=False)
    np.testing.assert_array_equal(single["sndr"], column["sndr"])
