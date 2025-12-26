"""Fit phase error using AM/PM separation (pure math kernel).

Fitting model (Cosine basis): error²(φ) = am² * cos²(φ) + pm² * sin²(φ) + base_noise
- am, pm: Both in Volts (same unit as error)
- base_noise: Constant noise floor (variance, V²)

This kernel is mode-agnostic: it only solves least-squares on whatever (phi, y) you feed it.
"""

import numpy as np

def _fit_error_phase(
    y: np.ndarray,
    phi: np.ndarray,
    include_base_noise: bool = True
) -> dict[str, float]:
    """Pure least-squares solver for AM/PM model.

    Model (Cosine basis): y = am² * cos²(φ) + pm² * sin²(φ) + base_noise

    Parameters
    ----------
    y : np.ndarray
        Target values (error² for raw mode, RMS² for binned mode)
    phi : np.ndarray
        Phase values in radians (same length as y)
    include_base_noise : bool, default=True
        Include base noise floor term in fitting.

    Returns
    -------
    dict
        am_v: Amplitude noise RMS in Volts
        pm_v: Phase noise RMS in Volts
        base_noise_var: Base noise variance (V²)
        r_squared: Coefficient of determination (goodness-of-fit)
        coeffs: Raw coefficients [am², pm², base_noise]
    """
    y = np.asarray(y).flatten()
    phi = np.asarray(phi).flatten()

    if len(y) < 2:
        return {"am_v": 0.0, "pm_v": 0.0, "base_noise_var": 0.0, "r_squared": 0.0, "coeffs": [0.0, 0.0, 0.0]}

    # Build sensitivity matrix (Cosine basis: s = A·cos(φ))
    # AM noise: error² ∝ cos²(φ) → peaks at signal peaks
    # PM noise: error² ∝ sin²(φ) → peaks at zero-crossings
    am_sensitivity = np.cos(phi) ** 2
    pm_sensitivity = np.sin(phi) ** 2

    # Build design matrix
    if include_base_noise:
        X = np.column_stack([am_sensitivity, pm_sensitivity, np.ones_like(y)])
    else:
        X = np.column_stack([am_sensitivity, pm_sensitivity])

    # Solve least-squares
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        coeffs = np.zeros(3 if include_base_noise else 2)

    # Extract parameters (ensure non-negative)
    am_variance_v2 = max(0.0, coeffs[0])
    pm_variance_v2 = max(0.0, coeffs[1])
    base_noise_var = max(0.0, coeffs[2]) if include_base_noise else 0.0

    # Compute R² (coefficient of determination)
    y_pred = X @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-20 else 0.0

    return {
        "am_v": float(np.sqrt(am_variance_v2)),
        "pm_v": float(np.sqrt(pm_variance_v2)),
        "base_noise_var": float(base_noise_var),
        "r_squared": float(r_squared),
        "coeffs": [float(am_variance_v2), float(pm_variance_v2), float(base_noise_var)],
    }
