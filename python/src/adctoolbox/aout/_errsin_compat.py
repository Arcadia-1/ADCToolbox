"""MATLAB errsin-compatible residual binning helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from adctoolbox.fundamentals.fit_sine_4param import fit_sine_4param
from adctoolbox.oversampling.ifilter import ifilter


def errsin_compat(
    signal: np.ndarray,
    *,
    bins: int = 100,
    freq: float | None = None,
    xaxis: str = "phase",
    disp: bool | int = False,
    erange: tuple[float, float] | None = None,
    osr: float = 1,
    window: str = "hann",
    max_iterations: int | None = None,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Return MATLAB ``errsin``-style binned residual diagnostics.

    This helper is intentionally separate from ``analyze_error_by_phase`` and
    ``analyze_error_by_value``. Those functions expose the redesigned Python
    diagnostics; this helper preserves the legacy MATLAB tuple contract used by
    comparison fixtures and compatibility wrappers.
    """

    signal = np.asarray(signal, dtype=float).reshape(-1)
    if signal.size == 0:
        raise ValueError("signal must not be empty")
    if not np.all(np.isfinite(signal)):
        raise ValueError("signal must contain only finite values")

    bins = int(round(bins))
    if bins <= 0:
        raise ValueError("bins must be a positive integer")
    if osr < 1:
        raise ValueError("osr must be greater than or equal to 1")

    fit = _fit_like_errsin(signal, freq, max_iterations, tolerance)
    fitted = np.asarray(fit["fitted_signal"], dtype=float)
    fitted_freq = float(fit["frequency"])
    amplitude = float(fit["amplitude"])
    phase0 = float(fit["phase"])

    error = fitted - signal
    if osr > 1:
        error = _filter_error_like_errsin(error, osr, window)

    sample_index = np.arange(signal.size)
    phase_deg = np.mod(phase0 / np.pi * 180.0 + sample_index * fitted_freq * 360.0, 360.0)

    xaxis = xaxis.lower()
    if xaxis == "phase":
        result = _bin_by_phase(signal, error, phase_deg, bins)
    elif xaxis == "value":
        result = _bin_by_value(signal, error, phase_deg, bins)
    else:
        raise ValueError("xaxis must be 'phase' or 'value'")

    errxx = result["errxx"]
    filtered_error = error
    plot_signal = signal
    plot_phase_deg = phase_deg
    if erange is not None:
        if len(erange) != 2:
            raise ValueError("erange must contain lower and upper bounds")
        lo, hi = erange
        mask = (errxx >= lo) & (errxx <= hi)
        errxx = errxx[mask]
        filtered_error = error[mask]
        plot_signal = signal[mask]
        plot_phase_deg = phase_deg[mask]

    result.update(
        {
            "error": filtered_error,
            "errxx": errxx,
            "phase": phase_deg,
            "plot_signal": plot_signal,
            "plot_phase": plot_phase_deg,
            "fit": fit,
            "frequency": fitted_freq,
            "amplitude": amplitude,
        }
    )

    if disp:
        if xaxis == "phase":
            _plot_phase(result)
        else:
            _plot_value(result)

    return result


def _fit_like_errsin(
    signal: np.ndarray,
    freq: float | None,
    max_iterations: int | None,
    tolerance: float,
) -> dict[str, Any]:
    if freq is None:
        iterations = 100 if max_iterations is None else max_iterations
        return fit_sine_4param(signal, max_iterations=iterations, tolerance=tolerance)

    iterations = 0 if max_iterations is None else max_iterations
    return fit_sine_4param(
        signal,
        frequency_estimate=float(freq),
        max_iterations=iterations,
        tolerance=tolerance,
    )


def _filter_error_like_errsin(error: np.ndarray, osr: float, window: str) -> np.ndarray:
    n = error.size
    if isinstance(window, str):
        if window == "hann":
            win = np.hanning(n)
        elif window == "rect":
            win = np.ones(n)
        else:
            win = np.ones(n)
    else:
        raise ValueError("window must be 'hann' or 'rect'")

    filtered = ifilter(error * win, np.array([[0.0, 0.5 / osr]])).reshape(-1)
    return filtered / np.maximum(win, 0.01)


def _bin_by_phase(signal: np.ndarray, error: np.ndarray, phase_deg: np.ndarray, bins: int) -> dict[str, Any]:
    xx = np.arange(bins, dtype=float) / bins * 360.0
    bin_indices = np.mod(np.floor(phase_deg / 360.0 * bins + 0.5).astype(int), bins)

    emean, erms = _centered_bin_stats(error, bin_indices, bins)

    asen = np.cos(xx / 360.0 * 2.0 * np.pi) ** 2
    psen = np.sin(xx / 360.0 * 2.0 * np.pi) ** 2
    valid = np.isfinite(erms)
    if np.count_nonzero(valid) >= 2:
        coeffs, _, _, _ = np.linalg.lstsq(
            np.column_stack((asen[valid], psen[valid])),
            erms[valid] ** 2,
            rcond=None,
        )
        anoi = float(np.sqrt(max(coeffs[0], 0.0)))
        pnoi = float(np.sqrt(max(coeffs[1], 0.0)))
    else:
        anoi = np.nan
        pnoi = np.nan

    return {
        "emean": emean,
        "erms": erms,
        "xx": xx,
        "anoi": anoi,
        "pnoi": pnoi,
        "errxx": phase_deg,
        "bin_indices": bin_indices,
        "signal": signal,
    }


def _bin_by_value(signal: np.ndarray, error: np.ndarray, phase_deg: np.ndarray, bins: int) -> dict[str, Any]:
    dat_min = float(np.min(signal))
    dat_max = float(np.max(signal))
    bin_width = (dat_max - dat_min) / bins
    if bin_width <= 0:
        raise ValueError("signal value range must be non-zero")

    xx = dat_min + (np.arange(bins, dtype=float) + 0.5) * bin_width
    bin_indices = np.minimum(np.floor((signal - dat_min) / bin_width).astype(int), bins - 1)

    emean, erms = _centered_bin_stats(error, bin_indices, bins)
    emean_rise = _bin_mean(error, bin_indices, bins, phase_deg < 180.0)
    emean_fall = _bin_mean(error, bin_indices, bins, phase_deg >= 180.0)

    return {
        "emean": emean,
        "erms": erms,
        "xx": xx,
        "anoi": np.nan,
        "pnoi": np.nan,
        "errxx": signal,
        "bin_indices": bin_indices,
        "emean_rise": emean_rise,
        "emean_fall": emean_fall,
        "dat_min": dat_min,
        "dat_max": dat_max,
    }


def _centered_bin_stats(error: np.ndarray, bin_indices: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(bin_indices, minlength=bins).astype(float)
    sums = np.bincount(bin_indices, weights=error, minlength=bins)

    emean = np.full(bins, np.nan)
    valid = counts > 0
    emean[valid] = sums[valid] / counts[valid]

    centered = error - emean[bin_indices]
    sum_centered_sq = np.bincount(bin_indices, weights=centered**2, minlength=bins)
    erms = np.full(bins, np.nan)
    erms[valid] = np.sqrt(sum_centered_sq[valid] / counts[valid])
    return emean, erms


def _bin_mean(error: np.ndarray, bin_indices: np.ndarray, bins: int, mask: np.ndarray) -> np.ndarray:
    counts = np.bincount(bin_indices[mask], minlength=bins).astype(float)
    sums = np.bincount(bin_indices[mask], weights=error[mask], minlength=bins)
    result = np.full(bins, np.nan)
    valid = counts > 0
    result[valid] = sums[valid] / counts[valid]
    return result


def _plot_phase(result: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    plot_signal = result["plot_signal"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    ax1_left = ax1
    ax1_right = ax1.twinx()
    ax1_left.plot(result["errxx"], plot_signal, "k.", markersize=2)
    ax1_right.plot(result["errxx"], result["error"], "r.", markersize=2)
    ax1_right.plot(result["xx"], result["emean"], "b-", linewidth=1.5)
    ax1_left.set_xlim(0, 360)
    ax1_right.set_xlim(0, 360)
    ax1_left.set_ylabel("data")
    ax1_right.set_ylabel("error")
    ax1.set_xlabel("phase [deg]")

    ax2.bar(result["xx"], result["erms"], width=360.0 / len(result["xx"]) * 0.8)
    ax2.set_xlim(0, 360)
    ax2.set_xlabel("phase [deg]")
    ax2.set_ylabel("RMS error")
    fig.tight_layout()


def _plot_value(result: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    plot_signal = result["plot_signal"]
    plot_phase_deg = result["plot_phase"]
    plot_error = result["error"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    idx_rise = plot_phase_deg < 180.0
    idx_fall = ~idx_rise
    ax1.plot(plot_signal[idx_rise], plot_error[idx_rise], ".", color=[1.0, 0.5, 0.5], markersize=2)
    ax1.plot(plot_signal[idx_fall], plot_error[idx_fall], ".", color=[0.5, 0.5, 1.0], markersize=2)
    ax1.plot(result["xx"], result["emean_rise"], "r-", linewidth=1.5)
    ax1.plot(result["xx"], result["emean_fall"], "b-", linewidth=1.5)
    ax1.plot(result["xx"], result["emean"], "k-", linewidth=1.5)
    ax1.set_xlim(result["dat_min"], result["dat_max"])
    ax1.set_xlabel("value")
    ax1.set_ylabel("error")

    width = (result["dat_max"] - result["dat_min"]) / len(result["xx"]) * 0.8
    ax2.bar(result["xx"], result["erms"], width=width)
    ax2.set_xlim(result["dat_min"], result["dat_max"])
    ax2.set_xlabel("value")
    ax2.set_ylabel("RMS error")
    fig.tight_layout()
