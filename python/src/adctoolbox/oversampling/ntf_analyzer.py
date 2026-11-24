"""NTF performance analysis helpers."""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def _frequency_grid(n_grid, grid_policy):
    n_grid = int(n_grid)
    if n_grid <= 0:
        raise ValueError("n_grid must be a positive integer")

    if grid_policy == "python":
        return np.linspace(0, 0.5, n_grid)

    if grid_policy == "matlab":
        return np.arange(1, n_grid + 1, dtype=float) / n_grid / 2

    raise ValueError("grid_policy must be 'python' or 'matlab'")


def ntf_analyzer(ntf, flow, fhigh, is_plot=None, *, n_grid=2**16, grid_policy="python"):
    """
    Analyze the performance of NTF (Noise Transfer Function)

    Args:
        ntf: The noise transfer function (in z domain) - scipy.signal.TransferFunction or tuple (num, den)
        flow: Low bound frequency of signal band (relative to Fs)
        fhigh: High bound frequency of signal band (relative to Fs)
        is_plot: Optional plotting flag (1 to plot, None or 0 to skip)
        n_grid: Number of frequency grid points used for integration.
        grid_policy: "python" keeps the historical linspace grid; "matlab"
            uses the same grid as MATLAB ntfperf.m.

    Returns:
        noiSup: Integrated noise suppression of NTF in signal band in dB (compared to NTF=1)
    """
    flow = float(flow)
    fhigh = float(fhigh)
    if not (np.isfinite(flow) and np.isfinite(fhigh)) or flow < 0.0 or fhigh > 0.5 or flow >= fhigh:
        raise ValueError(
            f"Frequency bounds must satisfy 0 <= flow < fhigh <= 0.5; got flow={flow}, fhigh={fhigh}"
        )

    w = _frequency_grid(n_grid, grid_policy)

    # Convert NTF to transfer function if needed
    if isinstance(ntf, tuple):
        # ntf is (numerator, denominator) tuple
        num, den = ntf
        tf = signal.TransferFunction(num, den, dt=1)  # dt=1 for discrete time
    else:
        tf = ntf

    # Calculate frequency response for discrete-time system
    w_rad = w * 2 * np.pi
    _, mag = signal.dfreqresp(tf, w_rad)
    mag = np.abs(mag)

    # Calculate noise suppression in signal band
    if grid_policy == "matlab":
        band_mask = (w > flow) & (w < fhigh)
        np_val = float(np.sum(mag[band_mask] ** 2) / len(w))
    else:
        # Include exact band endpoints and integrate with trapezoidal quadrature
        # (factor 2 normalizes integral over [0, 0.5] to 1 for flat NTF = 1).
        interior = w[(w > flow) & (w < fhigh)]
        w_band = np.concatenate(([flow], interior, [fhigh]))
        _, mag_band = signal.dfreqresp(tf, w_band * 2 * np.pi)
        mag_band = np.abs(mag_band)
        trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        np_val = float(2.0 * trapz(mag_band ** 2, w_band))

    noiSup = float(-10 * np.log10(max(np_val, np.finfo(float).tiny)))

    # Plot if requested
    if is_plot == 1:
        mag_db = 20 * np.log10(np.maximum(mag, 1e-15))
        y_lo = float(np.min(mag_db))
        y_hi = float(np.max(mag_db))
        plt.semilogx(w[w > 0], mag_db[w > 0])

        if flow > 0:
            plt.semilogx([flow, flow], [y_lo, y_hi], 'k--')

        plt.semilogx([fhigh, fhigh], [y_lo, y_hi], 'k--')
        plt.xlabel('Normalized Frequency')
        plt.ylabel('Magnitude (dB)')
        plt.title('NTF Frequency Response')
        plt.grid(True)

    return noiSup
