"""Predict TI-ADC spur locations and magnitudes from per-channel mismatch.

Uses a first-order model:

- **Offset mismatch**  ->  spurs at ``k * fs / M`` (``k = 1..M-1``)
- **Gain + skew mismatch**  ->  spurs at ``fin + k * fs / M`` (``k = 1..M-1``)

For each, spur amplitude equals the appropriate coefficient of the M-point DFT
of the per-channel mismatch sequence, divided by M.

Skew is folded into a first-order complex-gain ``alpha_m = gain_m · e^{j 2π fin · t_m}``,
so the ``gain_skew`` spurs cover both gain and timing mismatches at once.
"""
from __future__ import annotations

import numpy as np

from adctoolbox.fundamentals.frequency import fold_frequency_to_nyquist


def predict_spurs(
    params: dict,
    fs: float,
    fin: float | None = None,
    full_scale: float = 1.0,
) -> list[dict]:
    """
    Predict TI-ADC spur frequencies and magnitudes.

    Parameters
    ----------
    params : dict
        Output of :func:`extract_mismatch_sine`. Must contain ``gain``, ``offset``,
        ``skew``, and at least one of ``fin`` / ``A``; ``fin`` can also be supplied
        via the argument below.
    fs : float
        Aggregate sample rate (Hz).
    fin : float, optional
        Input-tone frequency. Defaults to ``params['fin']`` if present.
    full_scale : float, default 1.0
        Signal full-scale (peak) used as the dBFS reference. If your ``x`` was
        scaled so the ideal sine has peak amplitude ``A``, set
        ``full_scale = A_full_scale`` — usually the converter's peak code.

    Returns
    -------
    spurs : list of dict
        One dict per predicted spur with keys:

        - ``freq_hz`` : spur frequency folded to ``[0, fs/2]``
        - ``kind``    : ``'offset'`` or ``'gain_skew'``
        - ``k``       : spur index in the M-point DFT (1..M-1)
        - ``amp``     : spur amplitude in the same units as ``x``
        - ``dbfs``    : spur magnitude in dBFS (relative to ``full_scale``)
        - ``dbc``     : spur magnitude in dBc (relative to fundamental)
    """
    gain = np.asarray(params["gain"], dtype=float)
    offset = np.asarray(params["offset"], dtype=float)
    skew = np.asarray(params["skew"], dtype=float)
    A_in = float(params.get("A", 1.0))

    if fin is None:
        fin = float(params.get("fin"))
        if fin is None:
            raise ValueError("fin must be provided via argument or params['fin']")

    M = gain.size
    if not (offset.size == M and skew.size == M):
        raise ValueError("gain / offset / skew must all have length M")

    fundamental_amp = A_in * gain.mean()
    fund_dbfs = 20 * np.log10(max(fundamental_amp / full_scale, 1e-300))

    def _to_db(amp: float) -> tuple[float, float]:
        if amp <= 0:
            return (-np.inf, -np.inf)
        dbfs = 20 * np.log10(amp / full_scale)
        dbc = dbfs - fund_dbfs
        return dbfs, dbc

    spurs: list[dict] = []

    # ---- offset spurs: DFT of offset sequence ----
    O = np.fft.fft(offset)
    for k in range(1, M):
        f = float(fold_frequency_to_nyquist(k * fs / M, fs))
        amp = np.abs(O[k]) / M
        dbfs, dbc = _to_db(amp)
        spurs.append(
            {"freq_hz": f, "kind": "offset", "k": int(k),
             "amp": float(amp), "dbfs": float(dbfs), "dbc": float(dbc)}
        )

    # ---- gain + skew spurs: DFT of centered complex gain ----
    alpha = gain * np.exp(1j * 2 * np.pi * fin * skew)
    alpha_ac = alpha - alpha.mean()
    A_fft = np.fft.fft(alpha_ac)
    for k in range(1, M):
        f = float(fold_frequency_to_nyquist(fin + k * fs / M, fs))
        amp = A_in * np.abs(A_fft[k]) / M
        dbfs, dbc = _to_db(amp)
        spurs.append(
            {"freq_hz": f, "kind": "gain_skew", "k": int(k),
             "amp": float(amp), "dbfs": float(dbfs), "dbc": float(dbc)}
        )

    spurs.sort(key=lambda s: s["freq_hz"])
    return spurs
