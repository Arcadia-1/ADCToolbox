"""
Lean SNDR + ENOB computation.

Delegates SNDR/ENOB to compute_spectrum (plotspec-aligned) so results match
analyze_spectrum. Use for optimization loops and spec gates.
"""

from adctoolbox.spectrum.compute_spectrum import compute_spectrum


def quick_sndr(data, fs=1.0, win_type="hann", side_bin=None, max_scale_range=None):
    """
    SNDR + ENOB from a single 1-D capture (same SNDR definition as analyze_spectrum).

    Parameters
    ----------
    data
        Time-domain samples, shape (N,)
    fs
        Sample rate (Hz)
    win_type
        Window name ('hann', 'rectangular', ...)
    side_bin
        Side bins around fundamental; None uses auto detection
    max_scale_range
        Optional full-scale range passed through to compute_spectrum

    Returns
    -------
    dict
        ``{'sndr_dbc': float, 'enob': float}``
    """
    results = compute_spectrum(
        data,
        fs=fs,
        win_type=win_type,
        side_bin=side_bin,
        max_scale_range=max_scale_range,
    )
    metrics = results["metrics"]
    return {"sndr_dbc": float(metrics["sndr_dbc"]), "enob": float(metrics["enob"])}
