"""Tests for calibrate_weight_sine_lite.

The fast tests below are deterministic regression checks.  The broad sweeps
are still useful for algorithm exploration, but are marked slow so they do not
blur the default pytest signal.
"""

from dataclasses import dataclass
import time

import numpy as np
import pytest

from adctoolbox import analyze_spectrum
from adctoolbox.calibration import calibrate_weight_sine_lite


BIT_WIDTH = 12
DEFAULT_PHASE = np.pi / 4
FULL_SCALE_WEIGHT_TOL = 2e-3


@dataclass(frozen=True)
class LiteMetrics:
    bit_width: int
    n_samp: int
    freq: float
    phase: float
    recovered_weights: np.ndarray
    recovered_weights_scaled: np.ndarray
    true_weights: np.ndarray
    max_weight_error: float
    sndr_before: float
    sndr_calc: float
    sndr_after: float
    enob_before: float
    enob_calc: float
    enob_after: float
    elapsed_ms: float
    signal_range_codes: int


def _binary_weights(bit_width=BIT_WIDTH):
    return 2.0 ** np.arange(bit_width - 1, -1, -1)


def _quantized_sine(
    n_samp,
    fin_bin,
    phase=DEFAULT_PHASE,
    amplitude=0.5,
    bit_width=BIT_WIDTH,
    noise_rms=0.0,
    rng=None,
):
    freq = fin_bin / n_samp
    t = np.arange(n_samp)
    signal = amplitude * np.sin(2 * np.pi * freq * t + phase) + 0.5
    if noise_rms:
        signal = signal + rng.normal(0.0, noise_rms, n_samp)

    max_code = 2**bit_width - 1
    codes = np.clip(np.floor(signal * (2**bit_width)), 0, max_code).astype(int)
    return t, freq, codes


def _bits_from_codes(codes, bit_width=BIT_WIDTH):
    return (codes[:, None] >> np.arange(bit_width - 1, -1, -1)) & 1


def _redundant_bits_from_codes(codes, weights):
    bits = np.zeros((len(codes), len(weights)), dtype=int)

    for i, code in enumerate(codes):
        remaining = float(code)
        for j, weight in enumerate(weights):
            if remaining >= weight - 0.5:
                bits[i, j] = 1
                remaining -= weight

    return bits


def _snr_db(reference, error):
    return 10 * np.log10(np.mean(reference**2) / (np.mean(error**2) + 1e-30))


def _metrics_from_signals(
    *,
    n_samp,
    freq,
    phase,
    t,
    codes,
    bits,
    true_weights,
    recovered_weights,
    recovered_weights_scaled,
    bit_width=BIT_WIDTH,
    amplitude=0.5,
    elapsed_ms=0.0,
):
    normalized_weights = true_weights / np.max(true_weights)
    max_weight_error = float(np.max(np.abs(recovered_weights - normalized_weights)))
    calibrated_signal = bits @ recovered_weights_scaled

    adc_amplitude = (2**bit_width - 1) / 2.0
    ideal_signal = (
        adc_amplitude * (amplitude / 0.5) * np.sin(2 * np.pi * freq * t + phase)
        + adc_amplitude
    )
    error_signal = calibrated_signal - ideal_signal

    sndr_before = float(analyze_spectrum(codes)["sndr_db"])
    sndr_calc = float(_snr_db(ideal_signal, error_signal))
    sndr_after = float(analyze_spectrum(calibrated_signal)["sndr_db"])

    return LiteMetrics(
        bit_width=bit_width,
        n_samp=n_samp,
        freq=freq,
        phase=phase,
        recovered_weights=recovered_weights,
        recovered_weights_scaled=recovered_weights_scaled,
        true_weights=true_weights,
        max_weight_error=max_weight_error,
        sndr_before=sndr_before,
        sndr_calc=sndr_calc,
        sndr_after=sndr_after,
        enob_before=(sndr_before - 1.76) / 6.02,
        enob_calc=(sndr_calc - 1.76) / 6.02,
        enob_after=(sndr_after - 1.76) / 6.02,
        elapsed_ms=elapsed_ms,
        signal_range_codes=int(codes.max() - codes.min()),
    )


def _run_binary_lite_case(
    n_samp=2**13,
    fin_bin=13,
    phase=DEFAULT_PHASE,
    amplitude=0.5,
    bit_width=BIT_WIDTH,
    noise_rms=0.0,
    rng=None,
):
    t, freq, codes = _quantized_sine(
        n_samp,
        fin_bin,
        phase=phase,
        amplitude=amplitude,
        bit_width=bit_width,
        noise_rms=noise_rms,
        rng=rng,
    )
    true_weights = _binary_weights(bit_width)
    bits = _bits_from_codes(codes, bit_width)

    start = time.perf_counter()
    recovered_weights = calibrate_weight_sine_lite(bits, freq=freq)
    elapsed_ms = (time.perf_counter() - start) * 1e3

    recovered_weights_scaled = recovered_weights * np.max(true_weights)
    return _metrics_from_signals(
        n_samp=n_samp,
        freq=freq,
        phase=phase,
        t=t,
        codes=codes,
        bits=bits,
        true_weights=true_weights,
        recovered_weights=recovered_weights,
        recovered_weights_scaled=recovered_weights_scaled,
        bit_width=bit_width,
        amplitude=amplitude,
        elapsed_ms=elapsed_ms,
    )


def _run_redundant_lite_case(n_samp=2**13, fin_bin=13, phase=DEFAULT_PHASE):
    true_weights = np.array(
        [2048.0, 1024.0, 512.0, 256.0, 128.0, 128.0, 64.0, 32.0, 16.0, 8.0, 4.0, 2.0]
    )
    t, freq, codes = _quantized_sine(n_samp, fin_bin, phase=phase, bit_width=BIT_WIDTH)
    bits = _redundant_bits_from_codes(codes, true_weights)

    start = time.perf_counter()
    recovered_weights = calibrate_weight_sine_lite(bits, freq=freq)
    elapsed_ms = (time.perf_counter() - start) * 1e3

    recovered_weights_scaled = recovered_weights * np.max(true_weights)
    return _metrics_from_signals(
        n_samp=n_samp,
        freq=freq,
        phase=phase,
        t=t,
        codes=codes,
        bits=bits,
        true_weights=true_weights,
        recovered_weights=recovered_weights,
        recovered_weights_scaled=recovered_weights_scaled,
        bit_width=BIT_WIDTH,
        elapsed_ms=elapsed_ms,
    )


def _assert_finite_metrics(metrics):
    assert metrics.recovered_weights.shape == metrics.true_weights.shape
    assert metrics.recovered_weights_scaled.shape == metrics.true_weights.shape
    assert np.all(np.isfinite(metrics.recovered_weights))
    assert np.all(np.isfinite(metrics.recovered_weights_scaled))
    for value in [
        metrics.max_weight_error,
        metrics.sndr_before,
        metrics.sndr_calc,
        metrics.sndr_after,
        metrics.enob_before,
        metrics.enob_calc,
        metrics.enob_after,
    ]:
        assert np.isfinite(value)


def _assert_full_scale_regression(metrics, max_weight_error=FULL_SCALE_WEIGHT_TOL):
    _assert_finite_metrics(metrics)
    assert metrics.signal_range_codes > 0
    assert metrics.max_weight_error < max_weight_error
    np.testing.assert_allclose(metrics.sndr_before, metrics.sndr_calc, atol=3.0)
    assert metrics.sndr_after > metrics.sndr_before - 3.0
    assert metrics.enob_after > 10.0


def _print_summary(label, metrics):
    print(
        f"{label}: N={metrics.n_samp}, freq={metrics.freq:.6f}, "
        f"phase={metrics.phase:.3f}, range={metrics.signal_range_codes}, "
        f"runtime={metrics.elapsed_ms:6.2f}ms, "
        f"weight_err={metrics.max_weight_error:.2e}, "
        f"SNDR={metrics.sndr_before:.1f}/{metrics.sndr_calc:.1f}/{metrics.sndr_after:.1f} dB, "
        f"ENOB={metrics.enob_before:.2f}/{metrics.enob_calc:.2f}/{metrics.enob_after:.2f}"
    )


def test_calibration_lite():
    """Validate weight recovery for a representative full-scale sine input."""
    metrics = _run_binary_lite_case()

    _assert_full_scale_regression(metrics)


@pytest.mark.parametrize(
    "n_samp, fin_bin, phase",
    [
        (2**10, 3, np.pi / 4),
        (2**13, 13, np.pi / 4),
        (2**13, 257, np.pi / 3),
    ],
)
def test_calibration_lite_representative_cases(n_samp, fin_bin, phase):
    """Small deterministic cases cover the default regression contract."""
    metrics = _run_binary_lite_case(n_samp=n_samp, fin_bin=fin_bin, phase=phase)

    _assert_full_scale_regression(metrics)


@pytest.mark.slow
def test_calibration_lite_sweep_N():
    """Explore sample-count sensitivity across a broad N range."""
    for n_exp in range(3, 17):
        metrics = _run_binary_lite_case(n_samp=2**n_exp, fin_bin=3)
        _assert_finite_metrics(metrics)
        if metrics.n_samp >= 2**8:
            _assert_full_scale_regression(metrics)
        _print_summary(f"N=2^{n_exp:02d}", metrics)


@pytest.mark.slow
def test_calibration_lite_sweep_fin():
    """Explore frequency-bin sensitivity across the available band."""
    n_samp = 2**13

    for fin_bin in range(1, n_samp // 2, 32):
        metrics = _run_binary_lite_case(n_samp=n_samp, fin_bin=fin_bin)
        _assert_finite_metrics(metrics)
        if 3 <= fin_bin <= n_samp // 2 - 32:
            assert metrics.sndr_after > metrics.sndr_before - 3.0
        _print_summary(f"bin={fin_bin:04d}", metrics)


@pytest.mark.slow
def test_calibration_lite_sweep_phase():
    """Explore phase sensitivity; exact zero/pi phases are known edge cases."""
    n_samp = 2**13
    n_phases = 36

    for i in range(n_phases + 1):
        phase = i * 2 * np.pi / n_phases
        metrics = _run_binary_lite_case(n_samp=n_samp, fin_bin=13, phase=phase)
        _assert_finite_metrics(metrics)
        if abs(np.sin(phase)) > 0.25:
            _assert_full_scale_regression(metrics)
        _print_summary(f"phase={i:02d}/{n_phases}", metrics)


@pytest.mark.slow
def test_calibration_lite_sweep_phase_noise():
    """Explore noisy phase sensitivity using deterministic noise."""
    rng = np.random.default_rng(20260620)
    n_samp = 2**13
    n_phases = 36

    for i in range(n_phases + 1):
        phase = i * 2 * np.pi / n_phases
        metrics = _run_binary_lite_case(
            n_samp=n_samp,
            fin_bin=13,
            phase=phase,
            noise_rms=0.0002,
            rng=rng,
        )
        _assert_finite_metrics(metrics)
        if abs(np.sin(phase)) > 0.25:
            assert metrics.sndr_calc > 40.0
            assert metrics.sndr_after > metrics.sndr_before - 10.0
        _print_summary(f"noisy_phase={i:02d}/{n_phases}", metrics)


@pytest.mark.slow
def test_calibration_lite_sweep_amplitude():
    """Explore amplitude sensitivity from very small signals to full scale."""
    n_samp = 2**13

    for amplitude_db in range(-60, 1, 1):
        amplitude = 0.5 * 10 ** (amplitude_db / 20.0)
        metrics = _run_binary_lite_case(n_samp=n_samp, fin_bin=13, amplitude=amplitude)
        _assert_finite_metrics(metrics)
        assert metrics.signal_range_codes > 0
        if amplitude_db >= -10:
            assert metrics.sndr_after > metrics.sndr_before - 3.0
        _print_summary(f"amplitude={amplitude_db:03d}dBFS", metrics)


@pytest.mark.slow
def test_calibration_lite_sweep_phase_redundancy():
    """Explore phase sensitivity with redundant ADC weights."""
    n_phases = 36

    for i in range(n_phases + 1):
        phase = i * 2 * np.pi / n_phases
        metrics = _run_redundant_lite_case(phase=phase)
        _assert_finite_metrics(metrics)
        assert metrics.sndr_after > 55.0
        if abs(np.sin(phase)) > 0.25:
            assert metrics.sndr_calc > 55.0
        _print_summary(f"redundant_phase={i:02d}/{n_phases}", metrics)
