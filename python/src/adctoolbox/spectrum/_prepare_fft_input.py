"""Prepare FFT input data - shared helper for spectrum analysis."""

import numpy as np
import warnings

_OVERRANGE_ATOL = 1e-12
_OVERRANGE_RTOL = 1e-3


def _prepare_fft_input(
    data: np.ndarray,
    max_scale_range: float | list[float] | tuple[float | float | None | list[float]] = None
) -> np.ndarray:
    """Prepare input data for FFT analysis.

    Handles data validation, shape normalization, DC removal, and amplitude
    normalization to ±1 range.

    Parameters
    ----------
    data
        Input ADC data, shape (N,) or (M, N). Standard format: (M runs, N samples).
        Auto-transposes if N >> M (with warning).
    max_scale_range
        Full scale range for normalization:

        - ``None`` : auto-detect from data as (max - min)
        - float : peak amplitude (e.g., 0.5 for ±0.5V range)
        - tuple/list : (min, max) ADC range

    Returns
    -------
    np.ndarray
        Processed data (DC removed, normalized to ±1), shape (M, N).
    """
    data = np.atleast_2d(data)
    if data.ndim > 2:
        raise ValueError(f"Input must be 1D or 2D, got {data.ndim}D")

    n_rows, n_cols = data.shape

    # Auto-transpose if shape is (N, 1) or (N, M) with N >> M
    if n_cols == 1 and n_rows > 1:
        data = data.T
    elif n_rows > 1 and n_cols > 1 and n_rows > n_cols * 2:
        warnings.warn(f"[Auto-transpose] Input shape [{n_rows}, {n_cols}] -> [{n_cols}, {n_rows}]. Standard format is (M runs, N samples).", UserWarning, stacklevel=3)
        data = data.T

    # Determine peak amplitude for normalization. Tuple/list ranges declare
    # absolute ADC limits before DC removal; scalar ranges declare centered
    # peak amplitude after DC removal.
    range_limits = None
    if max_scale_range is None:
        peak_amplitude = (np.max(data) - np.min(data)) / 2
    elif isinstance(max_scale_range, (list, tuple)):
        if len(max_scale_range) != 2:
            raise ValueError(f"Range tuple/list must have 2 elements, got {len(max_scale_range)}")
        range_limits = (float(max_scale_range[0]), float(max_scale_range[1]))
        fsr = range_limits[1] - range_limits[0]
        peak_amplitude = fsr / 2
    else:
        # When max_scale_range is a scalar, treat it as the peak amplitude directly
        peak_amplitude = max_scale_range

    if range_limits is not None:
        _warn_if_raw_overrange(data, range_limits)

    data_dc_removed = data - np.mean(data, axis=1, keepdims=True)
    if max_scale_range is not None and range_limits is None:
        _warn_if_centered_overrange(data_dc_removed, peak_amplitude)

    data_normalized = data_dc_removed / peak_amplitude if peak_amplitude != 0 else data_dc_removed

    return data_normalized


def _warn_if_raw_overrange(data: np.ndarray, range_limits: tuple[float, float]) -> None:
    lo, hi = range_limits
    data_min = float(np.min(data))
    data_max = float(np.max(data))
    tol = _overrange_tolerance(hi - lo)
    if data_min < lo - tol or data_max > hi + tol:
        warnings.warn(
            (
                "Input data exceeds declared max_scale_range before DC removal "
                f"(data range [{data_min:g}, {data_max:g}], declared [{lo:g}, {hi:g}])."
            ),
            RuntimeWarning,
            stacklevel=3,
        )


def _warn_if_centered_overrange(data_dc_removed: np.ndarray, peak_amplitude: float) -> None:
    peak = abs(float(peak_amplitude))
    if peak == 0.0:
        return
    centered_peak = float(np.max(np.abs(data_dc_removed)))
    tol = _overrange_tolerance(peak)
    if centered_peak > peak + tol:
        warnings.warn(
            (
                "Centered input data exceeds declared max_scale_range peak "
                f"(centered peak {centered_peak:g}, declared peak {peak:g})."
            ),
            RuntimeWarning,
            stacklevel=3,
        )


def _overrange_tolerance(reference: float) -> float:
    return max(_OVERRANGE_ATOL, _OVERRANGE_RTOL * max(abs(float(reference)), 1.0))
