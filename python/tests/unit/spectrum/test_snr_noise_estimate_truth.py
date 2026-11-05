"""The default SNR must match an injected ground truth.

Each capture is a known tone plus a known impairment: quantization error,
spurs, white or colored noise. The truth is the usual SNR definition. A
time-domain least-squares fit of DC and harmonics 1..5 at the known frequency
gives the fundamental power, and its residual (everything except DC, the
fundamental and H2..H5) is the noise. For OSR > 1 the tone is coherent and the
noise is generated circularly, so the in-band residual power is exact from its
DFT. The oracle never looks at the FFT estimate under test.
"""

import numpy as np
import pytest

from adctoolbox import analyze_spectrum
from adctoolbox.spectrum import compute_spectrum

N_SAMPLES = 8192
TONE_BIN = 809  # odd, so coprime with N_SAMPLES
AMPLITUDE = 0.9


def _tone(bin_position, amplitude=AMPLITUDE, phase=0.4, n_samples=N_SAMPLES):
    n = np.arange(n_samples)
    return amplitude * np.sin(2 * np.pi * bin_position * n / n_samples + phase)


def _quantize(x, bits):
    lsb = 2.0 / 2**bits
    return (np.floor(x / lsb) + 0.5) * lsb


def _circular_noise(rng, b, a, sigma, n_samples=N_SAMPLES):
    white = np.fft.fft(rng.normal(0, sigma, n_samples))
    response = np.fft.fft(b, n_samples) / np.fft.fft(a, n_samples)
    return np.real(np.fft.ifft(white * response))


def _truth_snr(x, bin_position, osr=1):
    n_samples = len(x)
    n = np.arange(n_samples)
    freq = bin_position / n_samples
    columns = [np.ones(n_samples)]
    for h in range(1, 6):
        columns += [np.sin(2 * np.pi * h * freq * n), np.cos(2 * np.pi * h * freq * n)]
    basis = np.column_stack(columns)
    coef = np.linalg.lstsq(basis, x, rcond=None)[0]
    residual = x - basis @ coef
    if osr == 1:
        noise = np.mean(residual**2)
    else:
        in_band = n_samples // (2 * osr) + 1
        noise = 2 * np.sum(np.abs(np.fft.fft(residual)[1:in_band]) ** 2) / n_samples**2
    return 10 * np.log10((coef[1] ** 2 + coef[2] ** 2) / 2 / noise)


def _default_metrics(x, **kwargs):
    return compute_spectrum(x, max_scale_range=[-1, 1], **kwargs)["metrics"]


@pytest.mark.parametrize("phase", [0.0, 0.3])
@pytest.mark.parametrize("win_type", ["hann", "boxcar"])
@pytest.mark.parametrize("bits", [8, 12, 16])
def test_default_snr_of_noiseless_quantizer_is_its_quantization_noise(bits, win_type, phase):
    x = _quantize(_tone(TONE_BIN, 0.98, phase), bits)

    metrics = _default_metrics(x, win_type=win_type)

    assert metrics["snr_dbc"] == pytest.approx(_truth_snr(x, TONE_BIN), abs=0.1)
    assert metrics["snr_dbc"] >= metrics["sndr_dbc"]


@pytest.mark.parametrize("win_type", ["hann", "boxcar"])
def test_default_snr_counts_noiseless_spurs_as_noise(win_type):
    quarter, half = N_SAMPLES // 4, N_SAMPLES // 2
    x = (
        _tone(TONE_BIN)
        + _tone(quarter - TONE_BIN, AMPLITUDE * 10 ** (-40 / 20), 0.2)
        + _tone(quarter + TONE_BIN, AMPLITUDE * 10 ** (-45 / 20), 0.9)
        + _tone(half - TONE_BIN, AMPLITUDE * 10 ** (-50 / 20), 1.7)
    )

    metrics = _default_metrics(x, win_type=win_type)

    assert metrics["snr_dbc"] == pytest.approx(_truth_snr(x, TONE_BIN), abs=0.1)
    assert metrics["snr_dbc"] >= metrics["sndr_dbc"]


def test_default_snr_counts_spurs_on_white_noise():
    rng = np.random.default_rng(20261105)
    x = (
        _tone(TONE_BIN)
        + _tone(1796.5, AMPLITUDE * 10 ** (-75 / 20), 0.3)
        + _tone(3001.5, AMPLITUDE * 10 ** (-85 / 20), 1.1)
        + rng.normal(0, AMPLITUDE / np.sqrt(2) * 10 ** (-80 / 20), N_SAMPLES)
    )

    metrics = _default_metrics(x)

    assert metrics["snr_dbc"] == pytest.approx(_truth_snr(x, TONE_BIN), abs=0.1)
    assert metrics["snr_dbc"] >= metrics["sndr_dbc"]


@pytest.mark.parametrize("snr_db", [60, 80])
@pytest.mark.parametrize("win_type", ["hann", "blackmanharris", "boxcar"])
def test_default_snr_of_white_noise_matches_truth(win_type, snr_db):
    rng = np.random.default_rng(snr_db)
    sigma = AMPLITUDE / np.sqrt(2) * 10 ** (-snr_db / 20)
    x = _tone(TONE_BIN) + rng.normal(0, sigma, N_SAMPLES)

    metrics = _default_metrics(x, win_type=win_type)

    assert metrics["snr_dbc"] == pytest.approx(_truth_snr(x, TONE_BIN), abs=0.2)


@pytest.mark.parametrize(
    ("b", "a", "osr", "win_type", "tolerance_db"),
    [
        ([1], [1, -0.9], 1, "hann", 0.2),  # low-pass noise
        ([1], [1, -0.9], 1, "boxcar", 0.2),
        ([1, -2, 1], [1], 1, "hann", 0.2),  # second-order noise shaping
        # In band at OSR 8 the shaped noise sits in a few bins near the band
        # edge, where Hann cross terms between neighboring bins add a random
        # spread of a few tenths of a dB.
        ([1, -2, 1], [1], 8, "hann", 0.5),
    ],
)
def test_default_snr_of_colored_noise_matches_truth(b, a, osr, win_type, tolerance_db):
    rng = np.random.default_rng(osr)
    tone_bin = TONE_BIN if osr == 1 else 41
    x = _tone(tone_bin) + _circular_noise(rng, b, a, 1e-4)

    metrics = _default_metrics(x, win_type=win_type, osr=osr)

    assert metrics["snr_dbc"] == pytest.approx(_truth_snr(x, tone_bin, osr), abs=tolerance_db)


def test_direct_integration_fills_noise_under_wide_harmonic_lobes():
    # A non-coherent tone at 100 dB SNR widens the auto side_bin to about 120
    # bins, so the four harmonic lobes cover about a quarter of the band.
    # Without the fill the SNR would read about 1.2 dB above SNDR.
    rng = np.random.default_rng(100)
    sigma = AMPLITUDE / np.sqrt(2) * 10 ** (-100 / 20)
    x = _tone(TONE_BIN + 0.37) + rng.normal(0, sigma, N_SAMPLES)

    result = compute_spectrum(x, max_scale_range=[-1, 1], nf_method=3)

    assert result["plot_data"]["side_bin"] > 100
    assert result["metrics"]["snr_dbc"] == pytest.approx(result["metrics"]["sndr_dbc"], abs=0.1)


def test_direct_integration_falls_back_to_sndr_when_lobes_cover_the_band():
    rng = np.random.default_rng(7)
    n_samples = 256
    x = _tone(23, n_samples=n_samples) + rng.normal(0, 1e-4, n_samples)

    metrics = _default_metrics(x, side_bin=40)

    assert np.isfinite(metrics["snr_dbc"])
    assert metrics["snr_dbc"] == pytest.approx(metrics["sndr_dbc"], abs=1e-9)


def test_direct_integration_is_the_default():
    rng = np.random.default_rng(3)
    x = _quantize(_tone(TONE_BIN, 0.98, 0.3), 12) + rng.normal(0, 1e-5, N_SAMPLES)

    default = _default_metrics(x)
    explicit = _default_metrics(x, nf_method=3)
    analyzed = analyze_spectrum(x, max_scale_range=[-1, 1], create_plot=False)

    for key in ("snr_dbc", "noise_floor_dbfs", "nsd_dbfs_hz"):
        assert default[key] == explicit[key]
        assert analyzed[key] == explicit[key]
