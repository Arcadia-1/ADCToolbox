"""Coherent averaging must keep the window's main lobe coherent.

Every bin other than the harmonic centres used to be rotated as if it were a
separate high-order harmonic. The side bins of a windowed line carry that
line's phase, so across runs with different start phases they averaged away:
the signal read 1.7 dB low with Hann (the default window), 3 dB with
Blackman-Harris and 5 dB with Kaiser, and SNR dropped by the same amount.
"""
import numpy as np
import pytest

from adctoolbox.spectrum.compute_spectrum import compute_spectrum
from adctoolbox.spectrum._align_spectrum_phase import _align_spectrum_phase

N, FUND_BIN, AMPLITUDE, SNR_DB = 4096, 1021, 0.5, 60.0
HD2_DBC, HD3_DBC = -70.0, -65.0


def _runs(n_runs, seed=1):
    rng = np.random.default_rng(seed)
    n = np.arange(N)
    sigma = AMPLITUDE / np.sqrt(2) * 10 ** (-SNR_DB / 20)
    records = []
    for _ in range(n_runs):
        theta = 2 * np.pi * FUND_BIN * n / N + rng.uniform(0, 2 * np.pi)
        records.append(
            AMPLITUDE * np.cos(theta)
            + AMPLITUDE * 10 ** (HD2_DBC / 20) * np.cos(2 * theta + 0.7)
            + AMPLITUDE * 10 ** (HD3_DBC / 20) * np.cos(3 * theta + 1.1)
            + sigma * rng.standard_normal(N)
        )
    return np.array(records)


@pytest.mark.parametrize("win_type", ["boxcar", "hann", "hamming", "blackman", "blackmanharris", "kaiser"])
@pytest.mark.parametrize("n_runs", [4, 16])
def test_signal_power_and_snr_gain_with_random_start_phases(win_type, n_runs):
    metrics = compute_spectrum(
        _runs(n_runs), max_scale_range=1.0, win_type=win_type, coherent_averaging=True,
    )["metrics"]

    assert metrics["sig_pwr_dbfs"] == pytest.approx(-6.02, abs=0.05)
    # Noise averages down by the number of runs; the signal does not.
    assert metrics["snr_dbc"] == pytest.approx(SNR_DB + 10 * np.log10(n_runs), abs=1.0)
    assert metrics["harmonics_dbc"][0] == pytest.approx(HD2_DBC, abs=0.8)
    assert metrics["harmonics_dbc"][1] == pytest.approx(HD3_DBC, abs=0.5)


@pytest.mark.parametrize("win_type", ["hann", "blackmanharris"])
def test_coherent_and_power_averaging_agree_on_signal_and_harmonics(win_type):
    records = _runs(16)

    power = compute_spectrum(records, max_scale_range=1.0, win_type=win_type)["metrics"]
    coherent = compute_spectrum(
        records, max_scale_range=1.0, win_type=win_type, coherent_averaging=True,
    )["metrics"]

    assert coherent["sig_pwr_dbfs"] == pytest.approx(power["sig_pwr_dbfs"], abs=0.05)
    np.testing.assert_allclose(coherent["harmonics_dbc"][:2], power["harmonics_dbc"][:2], atol=0.8)


def test_lobe_bins_rotate_with_their_harmonic():
    # Hann-windowed fundamental and third harmonic with arbitrary phases: after
    # alignment the fundamental's whole lobe is real and positive at its
    # centre, and every lobe keeps the relative phase between its own bins.
    n = np.arange(N)
    theta = 2 * np.pi * FUND_BIN * n / N + 2.1
    window = np.hanning(N + 1)[:N]
    spectrum = np.fft.fft((np.cos(theta) + 0.01 * np.cos(3 * theta + 0.4)) * window)

    aligned = _align_spectrum_phase(spectrum, FUND_BIN, float(FUND_BIN), N, lobe_half_width=1)

    assert np.angle(aligned[FUND_BIN]) == pytest.approx(0.0, abs=1e-9)
    third = (3 * FUND_BIN) % N
    third = N - third if third > N // 2 else third
    for centre in (FUND_BIN, third):
        before = spectrum[centre - 1: centre + 2] / spectrum[centre]
        after = aligned[centre - 1: centre + 2] / aligned[centre]
        np.testing.assert_allclose(after, before, atol=1e-9)
    np.testing.assert_allclose(np.abs(aligned[1: N // 2]), np.abs(spectrum[1: N // 2]), rtol=1e-12)
    assert aligned[0] == 0


def test_zero_lobe_width_reproduces_harmonic_only_alignment():
    # With a rectangular window nothing changes: each bin is rotated as the
    # lowest-order harmonic that folds onto it.
    rng = np.random.default_rng(4)
    n_fft, fund = 64, 5
    spectrum = np.fft.fft(rng.standard_normal(n_fft))
    phasor = np.conj(spectrum[fund] / abs(spectrum[fund]))

    aligned = _align_spectrum_phase(spectrum, fund, float(fund), n_fft, lobe_half_width=0)

    expected = spectrum.copy()
    done = np.zeros(n_fft, dtype=bool)
    for order in range(1, n_fft + 1):
        folded = (fund * order) % n_fft
        if 2 * folded < n_fft:
            position, rotation = folded, phasor**order
        else:
            position, rotation = n_fft - folded, np.conj(phasor**order)
        if not done[position]:
            expected[position] *= rotation
            done[position] = True
    np.testing.assert_allclose(aligned[1: n_fft // 2 + 1], expected[1: n_fft // 2 + 1], atol=1e-9)
