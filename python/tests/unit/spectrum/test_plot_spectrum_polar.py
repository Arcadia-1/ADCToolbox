import numpy as np
import matplotlib.pyplot as plt
import pytest

from adctoolbox.spectrum._harmonics import _calculate_harmonic_phases
from adctoolbox.spectrum.analyze_spectrum_polar import analyze_spectrum_polar
from adctoolbox.spectrum.compute_spectrum import compute_spectrum
from adctoolbox.spectrum.plot_spectrum_polar import plot_spectrum_polar


def test_calculate_harmonic_phases_uses_hd2_hd3_bins():
    spec = np.ones(16, dtype=complex)
    harmonic_bins = np.array([2, 3, 4])
    spec[2] = np.exp(1j * np.deg2rad(20.0))
    spec[3] = np.exp(1j * np.deg2rad(-45.0))
    spec[4] = np.exp(1j * np.deg2rad(120.0))

    hd2_phase, hd3_phase = _calculate_harmonic_phases(spec, harmonic_bins)

    assert hd2_phase == pytest.approx(20.0)
    assert hd3_phase == pytest.approx(-45.0)


def test_plot_spectrum_polar_metrics_text_uses_harmonics_dbc():
    n_fft = 2**13
    fs = 100e6
    amplitude = 0.5
    fin = 123 / n_fft * fs
    n = np.arange(n_fft)
    t = n / fs

    hd2_target = -80.0
    hd3_target = -66.0
    hd2_amp = 10 ** (hd2_target / 20)
    hd3_amp = 10 ** (hd3_target / 20)
    k2 = hd2_amp / (amplitude / 2)
    k3 = hd3_amp / (amplitude**2 / 4)

    sine = amplitude * np.sin(2 * np.pi * fin * t)
    signal = sine + k2 * sine**2 + k3 * sine**3
    result = compute_spectrum(
        signal,
        fs=fs,
        win_type="rectangular",
        side_bin=0,
        max_harmonic=5,
        coherent_averaging=True,
    )

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    plot_spectrum_polar(result, harmonic=5, fixed_radial_range=120, ax=ax)
    metrics_text = next(text.get_text() for text in ax.texts if "SNR =" in text.get_text())
    plt.close(fig)

    assert "HD2 =  -80." in metrics_text
    assert "HD3 =  -66." in metrics_text
    assert "HD4 =" not in metrics_text
    assert "HD5 =" not in metrics_text
    assert "HD2 = -150." not in metrics_text
    assert "HD3 = -150." not in metrics_text


def test_analyze_spectrum_polar_harmonic_controls_compute_depth():
    n_fft = 2**13
    fs = 100e6
    fin = 123 / n_fft * fs
    n = np.arange(n_fft)
    signal = 0.5 * np.sin(2 * np.pi * fin * n / fs)

    metrics = analyze_spectrum_polar(signal, fs=fs, harmonic=10, create_plot=False)

    assert len(metrics["harmonics_dbc"]) == 9
