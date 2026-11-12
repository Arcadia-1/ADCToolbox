"""Minimal ADC calibration using sinewave input at known frequency."""

import numpy as np
from scipy.linalg import lstsq
from adctoolbox.calibration._calibration_filter import _make_calibration_filter

def calibrate_weight_sine_lite(
    bits: np.ndarray,
    freq: float,
    *,
    osr: float = 1,
) -> np.ndarray:
    """
    Minimal calibration at known frequency. Returns normalized weights.
    Expects well-conditioned binary data (N samples x M bits).

    The returned weights are in the same solver-unit-sine convention as
    ``calibrate_weight_sine``: the fitted fundamental sine magnitude is fixed
    to one for identifiability. Apply an explicit scale factor before using
    the reconstructed waveform for physical dBFS or NSD interpretation.

    ``osr > 1`` fits a lowpass-filtered residual, using the same finite-record
    FIR as ``calibrate_weight_sine``. Filter both the bit columns and the
    sine/DC reference; returned weights still apply to the original bits.
    The default ``osr=1`` preserves the unfiltered solve.
    """
    if freq is not None and freq > 0.5:
        raise ValueError(
            f"freq must be normalized Fin/Fs (Nyquist range [0, 0.5]); got {freq}. "
            f"If you have Fin in Hz, pass freq=Fin/Fs instead."
        )
    n_samples, bit_width = bits.shape
    fit_filter = _make_calibration_filter(osr, [n_samples])
    if fit_filter is not None:
        fit_filter.validate_frequencies(freq)

    # Build fundamental basis
    t = np.arange(n_samples)
    phase = 2.0 * np.pi * freq * t
    cos_basis = np.cos(phase)
    sin_basis = np.sin(phase)

    # The fundamental's overall scale is not identifiable, so one of its two
    # coefficients is fixed to one. Fixing the cosine fails when the input has
    # (almost) no cosine component, e.g. a sine starting at zero phase, and
    # fixing the sine fails for a pure cosine. Solve both and keep the
    # better-conditioned one, as calibrate_weight_sine does.
    offset_col = np.ones((n_samples, 1))

    def solve(unity_basis, other_basis):
        A = np.column_stack([bits, offset_col, other_basis])
        b = -unity_basis
        if fit_filter is not None:
            A = fit_filter.apply(A)
            b = fit_filter.apply(b)
            if len(b) <= A.shape[1]:
                raise ValueError("Too few samples remain after calibration filtering for the fit.")
        solution, _, _, _ = lstsq(A, b)
        return solution, np.linalg.norm(A @ solution - b)

    coeffs, residual = solve(cos_basis, sin_basis)
    coeffs_sin, residual_sin = solve(sin_basis, cos_basis)
    if residual_sin < residual:
        coeffs = coeffs_sin

    # Extract weights and normalization
    weights_raw = coeffs[:bit_width]
    other_coeff = coeffs[-1]
    norm_factor = np.sqrt(1.0 + other_coeff**2)
    weights = weights_raw / norm_factor

    # Polarity correction (ensure positive sum)
    if np.sum(weights) < 0:
        weights = -weights

    return weights
