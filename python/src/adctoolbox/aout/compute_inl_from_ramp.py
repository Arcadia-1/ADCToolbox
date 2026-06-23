"""
Ramp-histogram INL/DNL analysis for ADC output codes.

This method assumes a slow monotonic linear ramp sampled uniformly in time.
Under that assumption, the number of samples landing in each output code is
proportional to that code's input width.
"""

from __future__ import annotations

import numpy as np


def _validate_code_limits(
    num_bits: int | None,
    code_min: int,
    code_max: int | None,
) -> tuple[int | None, int, int | None]:
    if num_bits is not None:
        if not isinstance(num_bits, (int, np.integer)) or num_bits <= 0:
            raise ValueError(f"num_bits must be a positive integer, got {num_bits!r}")
        num_bits = int(num_bits)

    if not isinstance(code_min, (int, np.integer)):
        raise ValueError(f"code_min must be an integer, got {code_min!r}")
    code_min = int(code_min)

    if code_max is None:
        if num_bits is None:
            return num_bits, code_min, code_max
        code_max = code_min + 2**num_bits - 1
    elif not isinstance(code_max, (int, np.integer)):
        raise ValueError(f"code_max must be an integer, got {code_max!r}")

    code_max = int(code_max)
    if code_max < code_min:
        raise ValueError(f"code_max must be >= code_min, got {code_max} < {code_min}")

    return num_bits, code_min, code_max


def _coerce_integer_codes(codes) -> np.ndarray:
    codes_arr = np.asarray(codes)
    if codes_arr.size == 0:
        raise ValueError("codes must not be empty")

    codes_flat = codes_arr.reshape(-1)
    if not np.all(np.isfinite(codes_flat)):
        raise ValueError("codes must be finite")

    rounded = np.rint(codes_flat)
    if not np.allclose(codes_flat, rounded, rtol=0.0, atol=0.0):
        raise ValueError("codes must contain integer ADC codes")

    return rounded.astype(np.int64)


def _correct_inl(raw_inl: np.ndarray, code: np.ndarray, endpoint: str) -> np.ndarray:
    if not isinstance(endpoint, str):
        raise ValueError("endpoint must be one of 'fit', 'endpoints', or 'none'")
    endpoint = endpoint.lower()
    if endpoint == "none":
        return raw_inl

    if len(raw_inl) == 0:
        return raw_inl

    if endpoint in {"endpoint", "endpoints"}:
        if len(raw_inl) == 1:
            return raw_inl - raw_inl[0]
        correction = np.linspace(raw_inl[0], raw_inl[-1], len(raw_inl))
        return raw_inl - correction

    if endpoint == "fit":
        if len(raw_inl) == 1:
            return raw_inl - raw_inl[0]
        coeff = np.polyfit(code.astype(float), raw_inl, deg=1)
        return raw_inl - np.polyval(coeff, code)

    raise ValueError("endpoint must be one of 'fit', 'endpoints', or 'none'")


def compute_inl_from_ramp(
    codes,
    num_bits: int | None = None,
    code_min: int = 0,
    code_max: int | None = None,
    endpoint: str = "fit",
    exclude_endpoints: bool = True,
) -> dict:
    """
    Compute static INL/DNL from ADC output codes collected during a ramp test.

    A linear monotonic ramp sampled at uniform time intervals gives uniform
    spacing on the input-voltage axis. Therefore each code's histogram count is
    proportional to that code's transition width. DNL is estimated as
    ``counts / mean(counts) - 1`` over the analyzed code range.

    The mean count is the average over the analyzed code range, not an
    independently known full-scale 1 LSB width. This is unbiased for a full
    range ramp with uniform sampling, but a partial-range ramp reports DNL
    relative to that partial range's average code width.

    The returned ``inl`` uses the selected ``endpoint`` baseline. The default
    ``endpoint='fit'`` reports best-fit-corrected INL. Use ``endpoint='none'``
    for raw cumulative INL, including direct comparisons against the current
    sine-histogram analyzer's raw ``cumsum(dnl)`` convention.

    This function does not verify that ``codes`` came from a linear monotonic
    ramp. Non-ramp or non-uniformly swept data will produce mathematically
    valid histogram numbers, but they are not meaningful ramp DNL/INL.

    Parameters
    ----------
    codes : array_like
        Integer ADC output codes from a ramp simulation or measurement.
    num_bits : int, optional
        ADC resolution. When provided and ``code_max`` is omitted, the analyzed
        full range is ``code_min`` through ``code_min + 2**num_bits - 1``.
    code_min : int, default=0
        Lowest allowed ADC code.
    code_max : int, optional
        Highest allowed ADC code. If omitted with ``num_bits=None``, it is
        inferred from the maximum observed code.
    endpoint : {'fit', 'endpoints', 'none'}, default='fit'
        INL baseline correction. ``'fit'`` removes a best-fit line from the raw
        cumulative DNL. ``'endpoints'`` removes the line through the first and
        last raw INL samples. ``'none'`` returns the raw cumulative DNL.
    exclude_endpoints : bool, default=True
        Exclude the lowest and highest codes from the reported DNL/INL. This is
        useful for ramp captures where the first and last codes are only
        partially exercised by the ramp start/stop points.

    Returns
    -------
    dict
        Dictionary with ``code``, ``counts``, ``dnl``, ``inl``,
        ``missing_codes``, and summary metrics such as ``dnl_min`` and
        ``inl_pp``.
    """
    num_bits, code_min, code_max = _validate_code_limits(num_bits, code_min, code_max)
    codes_int = _coerce_integer_codes(codes)

    if code_max is None:
        code_max = int(np.max(codes_int))
        if code_max < code_min:
            raise ValueError(f"observed codes are below code_min={code_min}")

    if np.any((codes_int < code_min) | (codes_int > code_max)):
        raise ValueError(
            f"codes must be within [{code_min}, {code_max}] for ramp INL/DNL analysis"
        )

    full_code = np.arange(code_min, code_max + 1, dtype=np.int64)
    full_counts = np.bincount(
        codes_int - code_min,
        minlength=len(full_code),
    )[:len(full_code)].astype(float)

    if exclude_endpoints and len(full_code) > 2:
        analyzed_slice = slice(1, -1)
    else:
        analyzed_slice = slice(None)

    code = full_code[analyzed_slice]
    counts = full_counts[analyzed_slice]
    if counts.size == 0 or np.sum(counts) == 0:
        raise ValueError("no ramp samples remain in the analyzed code range")

    ideal_count = float(np.mean(counts))
    if ideal_count <= 0:
        raise ValueError("ideal code count must be positive")

    dnl = counts / ideal_count - 1.0
    dnl = np.maximum(dnl, -1.0)
    raw_inl = np.cumsum(dnl)
    inl = _correct_inl(raw_inl, code, endpoint)
    missing_codes = code[counts == 0].astype(int)

    return {
        "code": code.astype(int),
        "counts": counts.astype(int),
        "dnl": dnl,
        "inl": inl,
        "raw_inl": raw_inl,
        "missing_codes": missing_codes,
        "ideal_count": ideal_count,
        "dnl_min": float(np.min(dnl)),
        "dnl_max": float(np.max(dnl)),
        "dnl_pp": float(np.max(dnl) - np.min(dnl)),
        "inl_min": float(np.min(inl)),
        "inl_max": float(np.max(inl)),
        "inl_pp": float(np.max(inl) - np.min(inl)),
        "endpoint": endpoint,
        "exclude_endpoints": bool(exclude_endpoints),
        "code_min": int(code_min),
        "code_max": int(code_max),
        "num_bits": num_bits,
    }
