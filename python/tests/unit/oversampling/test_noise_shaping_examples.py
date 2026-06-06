"""Tests for oversampling noise-shaping examples and generators."""

import runpy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
from scipy import signal

from adctoolbox import analyze_spectrum, find_coherent_frequency, ifilter, ntfperf, perfosr
from adctoolbox.siggen import ADC_Signal_Generator


def test_apply_noise_shaping_accepts_custom_ntf_coefficients():
    n = 512
    fs = 10e6
    fin, _ = find_coherent_frequency(fs, fs / 64, n)
    gen = ADC_Signal_Generator(N=n, Fs=fs, Fin=fin, A=0.4, DC=0.0)

    clean = gen.get_clean_signal()
    shaped = gen.apply_noise_shaping(clean, n_bits=8, quant_range=(-0.5, 0.5), ntf=[1, -1.5, 0.5])

    assert shaped.shape == clean.shape
    assert np.std(shaped - clean) > 0


def test_noise_shaping_improves_inband_snr_over_unshaped_quantization():
    n = 4096
    fs = 100e6
    osr = 32
    fin, _ = find_coherent_frequency(fs, fs / (2 * osr) / 5, n)
    gen = ADC_Signal_Generator(N=n, Fs=fs, Fin=fin, A=0.4, DC=0.0)

    unshaped = gen.apply_quantization_noise(n_bits=8, quant_range=(-0.5, 0.5))
    shaped = gen.apply_noise_shaping(n_bits=8, quant_range=(-0.5, 0.5), order=2)

    r0 = analyze_spectrum(unshaped, fs=fs, osr=osr, max_scale_range=[-0.5, 0.5], create_plot=False)
    r1 = analyze_spectrum(shaped, fs=fs, osr=osr, max_scale_range=[-0.5, 0.5], create_plot=False)

    assert r1["snr_dbc"] > r0["snr_dbc"]


def test_ifilter_extracts_noise_shaped_inband_waveform():
    n = 1024
    fs = 20e6
    osr = 16
    fin, _ = find_coherent_frequency(fs, fs / (2 * osr) / 5, n)
    gen = ADC_Signal_Generator(N=n, Fs=fs, Fin=fin, A=0.4, DC=0.0)
    sig = gen.apply_noise_shaping(n_bits=8, quant_range=(-0.5, 0.5), order=1)
    sig_ib = ifilter(sig, [[0, 0.5 / osr]]).ravel()

    assert sig_ib.shape == sig.shape
    assert np.std(sig_ib) > 0


def test_ntfperf_and_perfosr_example_path_runs():
    n = 1024
    fs = 20e6
    fin, _ = find_coherent_frequency(fs, fs / 128, n)
    gen = ADC_Signal_Generator(N=n, Fs=fs, Fin=fin, A=0.4, DC=0.0)
    sig = gen.apply_noise_shaping(n_bits=8, quant_range=(-0.5, 0.5), order=1)

    ntf = signal.TransferFunction([1, -1], [1, 0], dt=1)
    gain = ntfperf(ntf, 0, 0.5 / 16)
    osr, sndr, sfdr, enob = perfosr(sig, osr=np.array([2, 4, 8, 16]), disp=False)

    assert gain > 0
    assert len(osr) == len(sndr) == len(sfdr) == len(enob) == 4


def test_oversampling_example_scripts_execute(tmp_path, monkeypatch):
    examples_dir = Path(__file__).parents[3] / "src" / "adctoolbox" / "examples" / "10_oversampling"
    monkeypatch.chdir(tmp_path)

    for script in [
        "exp_o01_noise_shaping_spectrum.py",
        "exp_o02_ifilter_band_analysis.py",
        "exp_o03_ntfperf_perfosr.py",
    ]:
        runpy.run_path(str(examples_dir / script), run_name="__main__")

