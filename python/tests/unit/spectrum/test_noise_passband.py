"""Noise estimates are taken over the passband, the in-band bins above the cutoff.

The first group builds spectra bin by bin, so every expected value follows from
counting bins. The second group injects known signals: the true in-band noise
is the power of the injected noise in the passband bins of its own unwindowed
DFT, never the spectrum under test.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest

from adctoolbox.spectrum import compute_spectrum
from adctoolbox.spectrum._estimate_noise_power import _estimate_noise_power
from adctoolbox.spectrum.plot_spectrum import plot_spectrum
from adctoolbox.spectrum.plot_spectrum_virtuoso import plot_spectrum_virtuoso


def _parts(spectrum_power, nf_method=3, **kwargs):
    return _estimate_noise_power(
        spectrum_power, nf_method, M=1, return_parts=True, **kwargs
    )


def test_cutoff_bins_are_neither_averaged_nor_extrapolated():
    # Every bin from 40 up is unit-power noise, so the true in-band noise is 60.
    # compute_spectrum writes the cutoff bins 0..39 as 1e-20.
    spectrum_power = np.ones(100)
    spectrum_power[:40] = 1e-20
    kwargs = dict(n_inband=100, bin_idx=50, harmonic_bins=np.array([70, 85]), side_bin=5,
                  passband_start=40)

    noise, parts = _parts(spectrum_power, **kwargs)

    # 60 passband bins: the signal lobe 45-55 (11) cannot be measured; the
    # harmonic lobes 65-75 and 80-90 (22) are filled at the mean of the 27
    # remaining bins.
    assert parts["exclude_direct"] == pytest.approx(27.0)
    assert parts["exclude_extrapolated"] == pytest.approx(22.0)
    assert noise == pytest.approx(49.0)
    # The median and the trimmed mean are scaled to the 60 passband bins.
    assert parts["trimmed"] == pytest.approx(60.0)
    assert parts["median"] == pytest.approx(60.0 / np.log(2))
    assert _estimate_noise_power(spectrum_power, 0, M=1, **kwargs) == pytest.approx(60.0)


@pytest.mark.parametrize(
    ("passband_start", "direct", "extrapolated"),
    [
        (9, 41, 9),   # cutoff through the H at 10: only its bins 9-12 are filled
        (14, 40, 5),  # the H at 10 lies wholly below the cutoff: no fill for it
        (29, 26, 5),  # cutoff through the signal lobe 28-32
    ],
)
def test_cutoff_overlapping_signal_and_harmonic_lobes(passband_start, direct, extrapolated):
    spectrum_power = np.ones(64)
    spectrum_power[10] = 100.0  # harmonic
    spectrum_power[50] = 100.0  # harmonic
    kwargs = dict(n_inband=64, bin_idx=30, harmonic_bins=np.array([10, 50]), side_bin=2,
                  passband_start=passband_start)

    noise, parts = _parts(spectrum_power, **kwargs)

    assert parts["exclude_direct"] == pytest.approx(direct)
    assert parts["exclude_extrapolated"] == pytest.approx(extrapolated)
    assert noise == pytest.approx(direct + extrapolated)


def test_method_3_is_undefined_without_remaining_bins():
    spectrum_power = np.ones(16)
    spectrum_power[14] = 10.0
    kwargs = dict(n_inband=16, bin_idx=8, harmonic_bins=np.array([14]), side_bin=3,
                  passband_start=12)

    # Passband 12-15 is entirely the H lobe at 14 (11-17): nothing is left to
    # tell its noise from its distortion.
    with pytest.warns(RuntimeWarning, match="undefined for nf_method=3"):
        noise, parts = _parts(spectrum_power, **kwargs)

    assert np.isnan(noise)
    assert parts["exclude_direct"] == 0.0
    # The auto method keeps using the direct sum.
    assert np.isfinite(_estimate_noise_power(spectrum_power, 0, M=1, **kwargs))


def test_empty_passband_is_undefined():
    spectrum_power = np.ones(32)
    for method in (0, 1, 2, 3):
        with pytest.warns(RuntimeWarning, match="above the cutoff"):
            noise, parts = _parts(spectrum_power, method, n_inband=16, bin_idx=8,
                                  harmonic_bins=np.array([12]), side_bin=1, passband_start=16)
        assert np.isnan(noise)
        assert all(np.isnan(v) for v in parts.values())


def test_auto_keeps_the_direct_sum_of_method_3():
    rng = np.random.default_rng(3)
    spectrum_power = rng.exponential(1.0, 256)
    spectrum_power[[40, 80, 120]] += 50.0
    kwargs = dict(n_inband=256, bin_idx=40, harmonic_bins=np.array([80, 120]), side_bin=3)

    _, parts = _parts(spectrum_power, **kwargs)
    auto = _estimate_noise_power(spectrum_power, 0, M=1, **kwargs)

    assert parts["exclude_extrapolated"] > 0
    assert auto == np.median([parts["median"], parts["trimmed"], parts["exclude_direct"]])


N = 8192
TONE_BIN = 809
AMPLITUDE = 0.9


def _passband_noise_dbfs(noise, passband_start, n_inband):
    """Injected noise power in the passband, in dBFS for a +/-1 full scale."""
    n = len(noise)
    p = np.abs(np.fft.rfft(noise)) ** 2 / n**2
    p[1:] *= 2
    if n % 2 == 0:
        p[-1] /= 2
    # A full-scale sine has power 1/2.
    return 10 * np.log10(2 * np.sum(p[max(passband_start, 1):n_inband]))


def _passband_width_bins(passband_start):
    """Width of the truth passband in FFT bins: bins max(passband_start, 1) up
    to Nyquist, where the Nyquist bin counts half."""
    return N // 2 - max(passband_start, 1) + 0.5


def _spectrum(x, nf_method, cutoff_bin=0, **kwargs):
    kw = dict(max_scale_range=[-1, 1], max_harmonic=5, nf_method=nf_method, **kwargs)
    if cutoff_bin:
        kw["cutoff_freq"] = (cutoff_bin - 0.5) / N
    return compute_spectrum(x, **kw)


def _noise_floor_error(x, noise, nf_method, cutoff_bin=0, **kwargs):
    metrics = _spectrum(x, nf_method, cutoff_bin, **kwargs)["metrics"]
    return metrics["noise_floor_dbfs"] - _passband_noise_dbfs(noise, cutoff_bin, N // 2 + 1)


def _white_capture():
    rng = np.random.default_rng(20261110)
    noise = rng.normal(0, AMPLITUDE / np.sqrt(2) * 1e-4, N)
    # The tone at bin 2809 stays in the passband; its H3 folds to bin 235,
    # below the cutoff.
    x = AMPLITUDE * np.sin(2 * np.pi * 2809 * np.arange(N) / N + 0.4) + noise
    return x, noise


def _noise_floor_line_level(ax):
    """Level of the dashed per-bin noise floor line drawn across [0, fs/2]."""
    levels = [
        line.get_ydata()[0]
        for line in ax.lines
        if len(line.get_xdata()) == 2
        and list(line.get_xdata()) == [0, 0.5]
        and line.get_ydata()[0] == line.get_ydata()[1]
    ]
    assert len(levels) == 1
    return levels[0]


@pytest.mark.parametrize("nf_method", [1, 2, 3])
def test_white_noise_estimates_do_not_move_with_the_cutoff(nf_method):
    x, noise = _white_capture()

    without = _noise_floor_error(x, noise, nf_method)
    with_cutoff = _noise_floor_error(x, noise, nf_method, cutoff_bin=N // 8)

    # A cutoff over a quarter of the band used to leave methods 1 and 2
    # 1.25 dB high: they still scaled to every in-band bin. The tolerance
    # covers the sampling spread of the median over 3072 rather than 4096 bins.
    assert with_cutoff == pytest.approx(without, abs=0.15)


@pytest.mark.parametrize("nf_method", [0, 1, 2, 3])
def test_white_noise_density_does_not_move_with_the_cutoff(nf_method):
    # NSD is noise per hertz, so white noise has the same NSD whatever band it
    # is integrated over. Dividing the passband noise by the full band width
    # read 1.25 dB low with a quarter of the band below the cutoff.
    x, _ = _white_capture()

    without = _spectrum(x, nf_method)["metrics"]["nsd_dbfs_hz"]
    with_cutoff = _spectrum(x, nf_method, cutoff_bin=N // 8)["metrics"]["nsd_dbfs_hz"]

    assert with_cutoff == pytest.approx(without, abs=0.15)


@pytest.mark.parametrize("cutoff_bin", [0, N // 8])
def test_nsd_and_noise_floor_line_match_the_injected_density(cutoff_bin):
    x, noise = _white_capture()
    result = _spectrum(x, 3, cutoff_bin)
    true_noise_dbfs = _passband_noise_dbfs(noise, cutoff_bin, N // 2 + 1)
    width_bins = _passband_width_bins(cutoff_bin)
    true_per_bin_dbfs = true_noise_dbfs - 10 * np.log10(width_bins)

    # fs = 1, so one bin is 1/N Hz wide.
    assert result["metrics"]["nsd_dbfs_hz"] == pytest.approx(
        true_per_bin_dbfs + 10 * np.log10(N), abs=0.1
    )
    for plotter in (plot_spectrum, plot_spectrum_virtuoso):
        fig, ax = plt.subplots()
        plotter(result, show_title=False, show_label=True, ax=ax)
        level = _noise_floor_line_level(ax)
        plt.close(fig)
        assert level == pytest.approx(true_per_bin_dbfs, abs=0.1), plotter.__name__


@pytest.mark.parametrize("cutoff_bin", [0, 32])
def test_harmonic_lobe_fill_recovers_noise_under_wide_lobes(cutoff_bin):
    # A non-coherent tone at 100 dB SNR widens the auto side_bin to about 120
    # bins, so the four harmonic lobes cover about a quarter of the band.
    # Without the fill the noise floor read 1.5 dB low. The rest of the error
    # is noise under the DC and signal lobes, which SNDR leaves out as well.
    rng = np.random.default_rng(100)
    noise = rng.normal(0, AMPLITUDE / np.sqrt(2) * 1e-5, N)
    x = AMPLITUDE * np.sin(2 * np.pi * (TONE_BIN + 0.37) * np.arange(N) / N + 0.4) + noise

    error = _noise_floor_error(x, noise, 3, cutoff_bin=cutoff_bin)

    assert abs(error) < 0.5


def test_cutoff_above_the_band_edge_leaves_metrics_undefined():
    x = AMPLITUDE * np.sin(2 * np.pi * 101 * np.arange(N) / N)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics = compute_spectrum(x, max_scale_range=[-1, 1], osr=4, cutoff_freq=0.2)["metrics"]

    for key in ("sndr_dbc", "enob", "snr_dbc", "noise_floor_dbfs", "nsd_dbfs_hz"):
        assert np.isnan(metrics[key]), key
