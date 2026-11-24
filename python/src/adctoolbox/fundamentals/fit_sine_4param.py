"""
Least Squares Sine Wave Fitting (IEEE Std 1057/1241).

Supports 3-parameter (frequency-fixed) and 4-parameter (frequency-optimized) fitting modes.
"""

import warnings
import numpy as np

# Matches the ``niter`` default of MATLAB ``sinfit``. The refinement stops as soon
# as the frequency update falls below ``tolerance``, typically after 2 to 5
# iterations, so the limit only matters for records that are not a single tone.
DEFAULT_MAX_ITERATIONS = 100

# A refined fit counts as worse than the initial one only when its residual
# energy grows by more than this fraction of the record's energy (-180 dB).
# Smaller differences are floating-point rounding between two exact fits.
_DIVERGENCE_ENERGY_FRACTION = 1e-18

# Frequency refinement must stay strictly between DC and Nyquist.
_MIN_FREQUENCY = 1e-10
_MAX_FREQUENCY = 0.5 - 1e-10


def fit_sine_4param(data, frequency_estimate=None, max_iterations=DEFAULT_MAX_ITERATIONS,
                    tolerance=1e-9, verbose=0):
    """
    Fit sine wave: y = A*cos(wt) + B*sin(wt) + C.

    Args:
        data: Input signal (1D or 2D array).
        frequency_estimate: Initial normalized frequency (0 to 0.5). If None, estimated via FFT.
        max_iterations: Maximum number of frequency-refinement iterations
            (default 100). 0 keeps the frequency fixed (3-parameter fit).
            Refinement stops early once the update is below ``tolerance``.
        tolerance: Convergence threshold for frequency updates.
        verbose: Verbosity level (0=silent, >0=print iteration info).

    Returns:
        Dictionary with fitted parameters:
            - fitted_signal: Reconstructed sine wave
            - residuals: Data - fitted_signal
            - frequency: Normalized frequency (0 to 0.5)
            - amplitude: sqrt(A² + B²)
            - phase: atan2(-B, A) in radians
            - dc_offset: DC component
            - rmse: Root mean square error
            - converged: Whether frequency refinement reached tolerance
              (or no refinement was requested)
            - n_iterations: Number of frequency-refinement iterations run
            - initial_frequency: Initial normalized frequency before refinement
            - last_delta_freq: Last normalized frequency update applied

        For 2D input, all values (except fitted_signal, residuals) are 1D arrays.

    Notes:
        A, B and C are always the least-squares solution at the returned
        frequency. If refinement ends on a frequency that fits worse than the
        initial estimate (for example on a chirp or on noise), the fit at the
        initial estimate is returned, ``converged`` is False and a
        RuntimeWarning is raised.
    """
    data = np.asarray(data)

    if data.ndim == 1:
        return _fit_core(data, frequency_estimate, max_iterations, tolerance, verbose)

    if data.ndim == 2:
        results = [_fit_core(ch, frequency_estimate, max_iterations, tolerance, verbose) for ch in data.T]
        return _merge_results(results)

    raise ValueError(f"Input must be 1D or 2D, got {data.ndim}D.")


def _solve_three_param(y, t, freq):
    """Least-squares A, B, C at a fixed frequency, with the fitted signal."""
    omega = 2 * np.pi * freq
    cos_vec = np.cos(omega * t)
    sin_vec = np.sin(omega * t)
    design_matrix = np.column_stack((cos_vec, sin_vec, np.ones(len(y))))
    a, b, c = np.linalg.lstsq(design_matrix, y, rcond=None)[0]
    return a, b, c, a * cos_vec + b * sin_vec + c


def _fit_core(y, freq_init, max_iter, tol, verbose=0):
    """Fit sine wave to 1D signal using least squares."""
    n = len(y)
    t = np.arange(n)
    freq = freq_init if freq_init is not None else _estimate_frequency_fft(y)
    initial_frequency = freq
    converged = max_iter == 0
    n_iterations = 0
    last_delta_freq = 0.0
    left_valid_range = False

    # 3-parameter fit at the initial frequency. It seeds the refinement and is
    # the fallback if the refinement makes the fit worse.
    a, b, c, fitted_sig = _solve_three_param(y, t, freq)
    initial_fit = (a, b, c, fitted_sig)
    initial_sse = float(np.sum((y - fitted_sig) ** 2))
    energy = float(np.sum(np.square(y, dtype=float)))

    # At exact Nyquist (freq >= _MAX_FREQUENCY) sin(pi*t) == 0 identically for
    # integer t, which makes the 4-param Jacobian column vanish and deadlocks.
    # If the 3-param fit at Nyquist is already exact, converge immediately;
    # otherwise nudge the starting frequency slightly below Nyquist so
    # Gauss-Newton can move freely (while keeping initial_fit as fallback).
    if max_iter > 0 and freq >= _MAX_FREQUENCY:
        if energy == 0.0 or initial_sse <= _DIVERGENCE_ENERGY_FRACTION * energy:
            converged = True
            max_iter = 0
        else:
            freq = 0.5 - 0.25 / max(n, 2)
            a, b, c, fitted_sig = _solve_three_param(y, t, freq)

    for i in range(1, max_iter + 1):
        omega = 2 * np.pi * freq
        cos_vec = np.cos(omega * t)
        sin_vec = np.sin(omega * t)

        # 4-parameter step: normalize the frequency-derivative column by
        # N * hypot(a, b) so its singular value stays O(sqrt(N)), preventing
        # lstsq SVD truncation when A * N^2 is large (e.g. 16-24 bit codes).
        col_scale = max(float(n) * float(np.hypot(a, b)), 1.0)
        freq_corr = (t / col_scale) * (-a * sin_vec + b * cos_vec)
        design_matrix = np.column_stack((cos_vec, sin_vec, np.ones(n), freq_corr))
        coeffs = np.linalg.lstsq(design_matrix, y, rcond=None)[0]
        a, b, c = coeffs[:3]

        delta_freq = (coeffs[3] / col_scale) / (2 * np.pi)
        last_delta_freq = delta_freq
        n_iterations += 1
        freq += delta_freq

        if verbose > 0:
            print(f"Freq iterating ({i}): freq = {freq}, "
                  f"delta_freq = {delta_freq}")

        if not _MIN_FREQUENCY < freq < _MAX_FREQUENCY:
            # The step left (0, 0.5). At DC and at Nyquist the cos and sin
            # columns are no longer independent, so a fit there can match the
            # record with an arbitrarily large amplitude.
            left_valid_range = True
            break

        if abs(delta_freq) < tol:
            converged = True
            break

    reverted = False
    if left_valid_range:
        reverted = True
    elif n_iterations > 0:
        # The 4-parameter step returns A, B for the linearized model at the
        # previous frequency. Re-solve them at the final frequency so the
        # returned fit is the least-squares optimum for that frequency.
        a, b, c, fitted_sig = _solve_three_param(y, t, freq)
        refined_sse = float(np.sum((y - fitted_sig) ** 2))
        reverted = refined_sse - initial_sse > _DIVERGENCE_ENERGY_FRACTION * energy

    if reverted:
        freq = initial_frequency
        a, b, c, fitted_sig = initial_fit
        converged = False

    if reverted:
        warnings.warn(
            "fit_sine_4param did not converge to a better fit than the initial "
            "frequency estimate; returning the fit at the initial estimate.",
            RuntimeWarning,
            stacklevel=3,
        )
    elif max_iter > 0 and not converged:
        warnings.warn(
            f"fit_sine_4param did not converge in {max_iter} iterations.",
            RuntimeWarning,
            stacklevel=3,
        )

    residuals = y - fitted_sig

    return {
        'fitted_signal': fitted_sig,
        'residuals': residuals,
        'frequency': freq,
        'amplitude': np.sqrt(a**2 + b**2),
        'phase': np.arctan2(-b, a),
        'dc_offset': c,
        'rmse': np.sqrt(np.mean(residuals**2)),
        'converged': converged,
        'n_iterations': n_iterations,
        'initial_frequency': initial_frequency,
        'last_delta_freq': last_delta_freq,
    }


def _merge_results(results_list):
    """Merge list of result dicts into single dict with stacked arrays."""
    merged = {}
    for key in results_list[0].keys():
        values = [r[key] for r in results_list]
        if np.ndim(values[0]) == 0:  # Scalar
            merged[key] = np.array(values)
        else:  # Array (fitted_signal, residuals)
            merged[key] = np.column_stack(values)
    return merged

def _estimate_frequency_fft(y):
    """Estimate frequency using FFT peak with parabolic interpolation."""
    n = len(y)
    spec = np.abs(np.fft.fft(y))
    spec[0] = 0
    # Bins 1 .. n//2. For odd n the last one is an ordinary bin just below
    # Nyquist; for even n it is the Nyquist bin itself.
    spec = spec[:n // 2 + 1]
    k = np.argmax(spec)

    # Parabolic interpolation for sub-bin precision (including left-sided
    # interpolation when the peak sits on the top bin so near-Nyquist tones
    # like (N/2 - 0.3)/N do not lock onto exact Nyquist 0.5).
    if 0 < k < len(spec) - 1:
        r = 1 if spec[k + 1] > spec[k - 1] else -1
        delta = r * spec[k + r] / (spec[k] + spec[k + r])
        k += delta
    elif k == len(spec) - 1 and k > 1 and spec[k - 1] > 1e-12 * spec[k]:
        k -= spec[k - 1] / (spec[k] + spec[k - 1])

    return k / n
