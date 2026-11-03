"""Public gain calibration imports and an example-only sine-fit baseline."""
import numpy as np
from adctoolbox.calibration import DitherGainTracker, apply_dither_gain

# Preserve historical example imports; the algorithm lives in calibration.
apply_gain_calibration = apply_dither_gain


def fit_pipeline_gain_sine(captures, frequency):
    """Dedicated sine calibration, without an ideal input or true gain.

    Probe is off. Fit q + c*r = A*sin(w*n) + B*cos(w*n) + DC, fixing the
    coarse-code coefficient to one. Captures are consecutive blocks of that
    dedicated sine; its amplitude and phase are NOT supplied to the fit.
    Return G_est=1/c, to be frozen during subsequent normal operation.
    The constant coarse DAC scale and known tone frequency are assumptions.
    """
    if not np.isfinite(frequency) or not 0 < frequency < .5:
        raise ValueError("frequency must be normalized in (0, .5)")
    triangular, count = np.empty((0, 5)), 0
    for capture in captures:
        q, r = np.asarray(capture["coarse"]), np.asarray(capture["backend"])
        if (q.ndim != 1 or not q.size or q.shape != r.shape
                or not np.isfinite(q).all() or not np.isfinite(r).all()):
            raise ValueError("Invalid foreground capture")
        pn = np.asarray(capture["pn"])
        if pn.shape != q.shape or np.any(pn != 0):
            raise ValueError("Foreground fit requires probe off")
        n = np.arange(count, count+len(q))
        design = np.column_stack([r, -np.sin(2*np.pi*frequency*n), -np.cos(2*np.pi*frequency*n),
                                  -np.ones(len(q))])
        # Streaming QR preserves least squares without squaring conditioning.
        triangular = np.linalg.qr(np.vstack([triangular, np.column_stack([design, -q])]), mode="r")
        count += len(q)
    coefficients, _, rank, singular = np.linalg.lstsq(triangular[:, :4], triangular[:, 4],
                                                      rcond=1e-12)
    if count < 4 or rank < 4 or coefficients[0] <= 0:
        raise ValueError("Foreground record does not identify a positive gain")
    return {"gain_estimate": float(1/coefficients[0]), "samples": count,
            "condition_number": float(singular[0]/singular[-1]),
            "coefficients": coefficients.tolist(),
            "reference": "joint sine fit; frequency known, amplitude/phase/DC fitted"}
