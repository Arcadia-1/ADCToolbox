"""Scale sine-calibration results from solver units to an ADC reference."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np


_SCALED_FIELDS = ("weight", "offset", "calibrated_signal", "ideal", "error")


def scale_calibration_output(
    result: dict[str, Any],
    *,
    scale: float | None = None,
    target_weights: np.ndarray | list[float] | tuple[float, ...] | None = None,
    target_sine_peak: float | None = None,
    inplace: bool = False,
) -> dict[str, Any]:
    """Scale ``calibrate_weight_sine`` output to a chosen ADC convention.

    ``calibrate_weight_sine`` fixes the first dataset's fitted fundamental sine
    magnitude to one so that the least-squares problem is identifiable. The returned
    ``weight``, ``calibrated_signal``, ``ideal``, and ``error`` are therefore
    in solver-unit-sine scale by default, not automatically in an ADC voltage
    or code full-scale convention.

    This helper applies a single linear scale factor after calibration. Ratio
    metrics such as ``snr_db`` and ``enob`` are intentionally left unchanged.

    Exactly one scale source must be supplied:

    - ``scale``: direct linear factor.
    - ``target_weights``: use ``sum(target_weights) / sum(result["weight"])``.
    - ``target_sine_peak``: map the solver unit sine peak to this peak value.
    """
    if not isinstance(result, dict):
        raise TypeError("result must be the dict returned by calibrate_weight_sine")

    scale_factor = _resolve_scale_factor(
        result,
        scale=scale,
        target_weights=target_weights,
        target_sine_peak=target_sine_peak,
    )

    output = result if inplace else deepcopy(result)
    for field in _SCALED_FIELDS:
        if field in output:
            output[field] = _scale_value(output[field], scale_factor)

    output["source_scale_convention"] = result.get(
        "scale_convention", "solver_unit_sine"
    )
    output["scale_convention"] = "adc_reference_scale"
    output["scale_factor"] = scale_factor
    return output


def _resolve_scale_factor(
    result: dict[str, Any],
    *,
    scale: float | None,
    target_weights: np.ndarray | list[float] | tuple[float, ...] | None,
    target_sine_peak: float | None,
) -> float:
    supplied = [
        scale is not None,
        target_weights is not None,
        target_sine_peak is not None,
    ]
    if sum(supplied) != 1:
        raise ValueError(
            "provide exactly one of scale, target_weights, or target_sine_peak"
        )

    if scale is not None:
        return _validate_scale(scale, "scale")

    if target_sine_peak is not None:
        return _validate_scale(target_sine_peak, "target_sine_peak")

    if "weight" not in result:
        raise KeyError('result must contain "weight" when target_weights is used')

    source_weights = np.asarray(result["weight"], dtype=float)
    target_weights_arr = np.asarray(target_weights, dtype=float)
    if source_weights.size == 0:
        raise ValueError('result["weight"] must not be empty')
    if target_weights_arr.size == 0:
        raise ValueError("target_weights must not be empty")
    if not np.all(np.isfinite(source_weights)):
        raise ValueError('result["weight"] must contain only finite values')
    if not np.all(np.isfinite(target_weights_arr)):
        raise ValueError("target_weights must contain only finite values")

    source_sum = float(np.sum(source_weights))
    target_sum = float(np.sum(target_weights_arr))
    if not np.isfinite(source_sum) or source_sum == 0.0:
        raise ValueError('sum(result["weight"]) must be finite and non-zero')
    if not np.isfinite(target_sum) or target_sum == 0.0:
        raise ValueError("sum(target_weights) must be finite and non-zero")
    return target_sum / source_sum


def _validate_scale(value: float, name: str) -> float:
    scale = float(value)
    if not np.isfinite(scale) or scale == 0.0:
        raise ValueError(f"{name} must be finite and non-zero")
    return scale


def _scale_value(value: Any, scale: float) -> Any:
    if isinstance(value, list):
        return [_scale_value(item, scale) for item in value]
    if isinstance(value, tuple):
        return tuple(_scale_value(item, scale) for item in value)
    return np.asarray(value) * scale
