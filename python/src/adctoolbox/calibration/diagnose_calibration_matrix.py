"""Diagnostics for calibration input matrices and recovered weights."""

from __future__ import annotations

import numpy as np


def diagnose_calibration_matrix(
    bits,
    nominal_weights=None,
    weights=None,
    *,
    binary_tol: float = 1e-12,
    constant_tol: float = 1e-12,
    rank_tol: float | None = None,
    weight_tol: float = 1e-12,
) -> dict:
    """Summarize rank, conditioning, binary-ness, and weight-shape diagnostics.

    This helper is intentionally diagnostic-only: it does not reject inputs and
    does not change calibration behavior. It is useful before or after
    sine-based weight calibration when checking whether the design matrix has
    enough observable information, whether a matrix still looks like a physical
    0/1 bit-code matrix, and whether solved weights look physically plausible.

    Parameters
    ----------
    bits
        Calibration design matrix, or a list of matrices to stack vertically.
        Matrices are interpreted as ``(samples, columns)``; wide matrices are
        transposed to match the calibration helpers' orientation convention.
    nominal_weights
        Optional nominal weights used for scale-invariant shape comparison.
    weights
        Optional recovered weights to diagnose.
    binary_tol
        Tolerance for considering an entry close to 0 or 1.
    constant_tol
        Column range at or below this value is treated as near-constant.
    rank_tol
        Optional absolute SVD/rank tolerance. If omitted, NumPy-style default
        tolerance ``max(shape) * eps * s_max`` is used.
    weight_tol
        Tolerance for treating weights as negative.

    Returns
    -------
    dict
        Diagnostic fields for the stacked matrix and optional weights.
    """

    matrix, segment_lengths, was_transposed = _prepare_diagnostic_matrix(bits)

    column_min = np.min(matrix, axis=0)
    column_max = np.max(matrix, axis=0)
    column_range = column_max - column_min
    near_constant_columns = np.flatnonzero(column_range <= constant_tol)

    near_zero = np.abs(matrix) <= binary_tol
    near_one = np.abs(matrix - 1.0) <= binary_tol
    binary_mask = near_zero | near_one
    binary_violation_fraction = float(1.0 - np.mean(binary_mask))

    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    sv_tol = _svd_tolerance(singular_values, centered.shape, rank_tol)
    effective = singular_values > sv_tol
    rank = int(np.count_nonzero(effective))

    if singular_values.size == 0 or rank == 0:
        condition_number = np.inf
        singular_value_ratio = singular_values.copy()
    else:
        condition_number = float(singular_values[0] / singular_values[rank - 1])
        singular_value_ratio = singular_values / singular_values[0]

    with_offset = np.column_stack([matrix, np.ones(matrix.shape[0])])
    if rank_tol is None:
        rank_with_offset = int(np.linalg.matrix_rank(with_offset))
    else:
        rank_with_offset = int(np.linalg.matrix_rank(with_offset, tol=rank_tol))

    result = {
        "shape": matrix.shape,
        "segment_lengths": segment_lengths,
        "was_transposed": was_transposed,
        "is_binary": bool(binary_violation_fraction == 0.0),
        "binary_violation_fraction": binary_violation_fraction,
        "column_min": column_min,
        "column_max": column_max,
        "column_range": column_range,
        "near_constant_columns": near_constant_columns,
        "rank": rank,
        "rank_with_offset": rank_with_offset,
        "condition_number": condition_number,
        "singular_values": singular_values,
        "singular_value_ratio": singular_value_ratio,
        "weight_diagnostics": None,
    }

    if weights is not None:
        result["weight_diagnostics"] = _diagnose_weights(
            weights,
            nominal_weights=nominal_weights,
            weight_tol=weight_tol,
        )

    return result


def _prepare_diagnostic_matrix(bits) -> tuple[np.ndarray, np.ndarray, bool]:
    if isinstance(bits, list):
        if len(bits) == 0:
            raise ValueError("bits list must not be empty")
        segments = bits
    else:
        segments = [bits]

    prepared = []
    lengths = []
    expected_width = None
    was_transposed = False

    for idx, segment in enumerate(segments):
        arr = np.asarray(segment, dtype=float)
        if arr.size == 0:
            raise ValueError(f"Dataset {idx} is empty")
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        if arr.ndim != 2:
            raise ValueError(f"Dataset {idx} must be 1D or 2D, got {arr.ndim}D")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Dataset {idx} contains NaN or Inf")

        n_rows, n_cols = arr.shape
        if n_rows < n_cols:
            arr = arr.T
            n_rows, n_cols = arr.shape
            was_transposed = True

        if expected_width is None:
            expected_width = n_cols
        elif n_cols != expected_width:
            raise ValueError(
                f"Inconsistent column counts: dataset {idx} has {n_cols}, "
                f"expected {expected_width}"
            )

        prepared.append(arr)
        lengths.append(n_rows)

    return np.vstack(prepared), np.asarray(lengths, dtype=int), was_transposed


def _svd_tolerance(
    singular_values: np.ndarray,
    shape: tuple[int, int],
    rank_tol: float | None,
) -> float:
    if rank_tol is not None:
        return float(rank_tol)
    if singular_values.size == 0:
        return 0.0
    return float(max(shape) * np.finfo(float).eps * singular_values[0])


def _diagnose_weights(
    weights,
    *,
    nominal_weights=None,
    weight_tol: float = 1e-12,
) -> dict:
    w = np.asarray(weights, dtype=float).ravel()
    if w.size == 0:
        raise ValueError("weights must not be empty")
    if not np.all(np.isfinite(w)):
        raise ValueError("weights contains NaN or Inf")

    diagnostics = {
        "min": float(np.min(w)),
        "max": float(np.max(w)),
        "negative_fraction": float(np.mean(w < -weight_tol)),
        "total_variation": float(np.sum(np.abs(np.diff(w)))),
        "relative_deviation_from_nominal": None,
        "nominal_scale": None,
    }

    if nominal_weights is None:
        return diagnostics

    nominal = np.asarray(nominal_weights, dtype=float).ravel()
    if nominal.size != w.size:
        raise ValueError("nominal_weights length must match weights length")
    if not np.all(np.isfinite(nominal)):
        raise ValueError("nominal_weights contains NaN or Inf")

    denom = float(np.dot(nominal, nominal))
    if denom <= 0.0:
        diagnostics["relative_deviation_from_nominal"] = np.inf
        diagnostics["nominal_scale"] = np.nan
        return diagnostics

    scale = float(np.dot(w, nominal) / denom)
    reference = scale * nominal
    reference_norm = float(np.linalg.norm(reference))
    if reference_norm <= 0.0:
        rel = np.inf
    else:
        rel = float(np.linalg.norm(w - reference) / reference_norm)

    diagnostics["relative_deviation_from_nominal"] = rel
    diagnostics["nominal_scale"] = scale
    return diagnostics
