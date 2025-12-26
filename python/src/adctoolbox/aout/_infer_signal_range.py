"""Infer signal range from data or user-provided full-scale range."""

import numpy as np

def _infer_signal_range(
    signal: np.ndarray,
    full_scale_range: tuple[float, float] = None,
) -> tuple[float, float]:
    """
    Determine the logical boundaries (min, max) of the signal.
    """
    # 1. Use user-provided range if available
    if full_scale_range is not None:
        assert full_scale_range[0] < full_scale_range[1], "Invalid full_scale_range: min must be less than max."
        return full_scale_range

    # 2. Analyze signal stats
    s_min = np.min(signal)
    s_max = np.max(signal)
    # Check peak-to-peak to handle signed integers correctly
    abs_max = np.max(np.abs([s_min, s_max]))
    mean_val = np.mean(signal)

    # Case A: Large values -> Treat as Raw ADC Codes
    # Use actual min/max as boundaries
    if abs_max > 2.0:
        return float(s_min), float(s_max)

    # Case B: Small values -> Treat as Voltage / Normalized
    # Infer standard ranges
    if mean_val > 0.1:
        return 0.0, 1.0  # Unipolar [0, 1]
    elif abs_max <= 0.6:
        return -0.5, 0.5 # Bipolar small [-0.5, 0.5]
    else:
        return -1.0, 1.0 # Bipolar large [-1, 1]