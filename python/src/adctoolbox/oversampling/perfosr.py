"""MATLAB-compatible performance-vs-OSR sweep."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from adctoolbox.spectrum.sweep_performance_vs_osr import sweep_performance_vs_osr


def perfosr(
    sig: np.ndarray,
    *,
    disp: bool | int | None = None,
    osr: np.ndarray | None = None,
    logscale: bool | int = True,
    harmonic: int = 5,
    smooth: int | None = None,
    ax: plt.Axes | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sweep ADC performance metrics versus oversampling ratio.

    This is the Python counterpart of MATLAB ``perfosr``. It returns the same
    four outputs in the same order: ``osr, sndr, sfdr, enob``.

    Parameters mirror the MATLAB name-value arguments where practical.
    ``disp`` controls plotting; when omitted it defaults to ``False`` because
    Python callers commonly consume returned values programmatically.
    """

    if harmonic < 1 or int(harmonic) != harmonic:
        raise ValueError("harmonic must be a positive integer")
    if smooth is not None and smooth < 1:
        raise ValueError("smooth must be at least 1")
    if osr is not None:
        osr_arr = np.asarray(osr, dtype=float)
        if np.any(osr_arr <= 0):
            raise ValueError("osr values must be positive")
    else:
        osr_arr = None

    create_plot = bool(disp) if disp is not None else False
    result = sweep_performance_vs_osr(
        sig,
        osr=osr_arr,
        harmonic=int(harmonic),
        create_plot=create_plot,
        ax=ax,
        logscale=bool(logscale),
        smooth=smooth,
    )
    return result["osr"], result["sndr"], result["sfdr"], result["enob"]
