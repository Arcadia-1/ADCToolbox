"""
Lean SNDR + ENOB computation.

Implements SNDR by its raw definition — signal_power divided by all
remaining in-band power — without breaking the "remainder" into
harmonic vs noise vs spurious categories.  No plotting, no auxiliary
metrics, no per-bin diagnostics.  Use for optimization loops,
parameter sweeps, and spec gates where only ENOB matters and the
per-call cost adds up.

For the full breakdown (SFDR / THD / HD2-HDk / NSD / noise floor)
use `analyze_spectrum`.
"""

import numpy as np

from adctoolbox.spectrum._window import _create_window, _get_default_side_bin
from adctoolbox.spectrum._locate_fundamental import _locate_fundamental


def quick_sndr(data, fs=1.0, win_type='hann', side_bin=None):
    """
    SNDR + ENOB from a single 1-D capture.

    By definition: ``SNDR = signal_power / (total_power - signal_power)``.
    Everything not in the fundamental's main lobe — harmonics, spurs,
    noise, DC offset — counts as noise+distortion.

    Parameters
    ----------
    data : np.ndarray
        Time-domain samples, shape (N,).
    fs : float
        Sample rate (Hz).  Not used in the SNDR ratio (it cancels) but
        preserved so callers can swap `quick_sndr(...)` in place of
        `analyze_spectrum(...)` without re-shuffling kwargs.
    win_type : str
        Window function name ('hann', 'rectangular', 'blackman', ...).
        Default 'hann' matches `analyze_spectrum`.
    side_bin : int, optional
        Number of bins on each side of the fundamental to count as
        signal.  Default None → the coherent main-lobe width for the
        selected window.  Non-coherent captures should pass a larger
        ``side_bin`` explicitly.

    Returns
    -------
    dict
        ``{'sndr_dbc': float, 'enob': float}``
    """
    x = np.asarray(data, dtype=np.float64)
    N = len(x)

    w, _, _ = _create_window(win_type, N)
    Xw = np.fft.fft(x * w)
    P = np.abs(Xw[:N // 2]) ** 2

    fund_bin, _ = _locate_fundamental(P, len(P))
    if side_bin is None:
        side_bin = _get_default_side_bin(win_type)

    sig_start = max(fund_bin - side_bin, 0)
    sig_end = min(fund_bin + side_bin + 1, len(P))
    sig_power = P[sig_start:sig_end].sum()
    nd_power = P.sum() - sig_power

    sndr_dbc = 10.0 * np.log10(sig_power / max(nd_power, 1e-30))
    enob = (sndr_dbc - 1.76) / 6.02
    return {'sndr_dbc': float(sndr_dbc), 'enob': float(enob)}
