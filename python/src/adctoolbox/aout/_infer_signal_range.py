"""Infer signal range from data or user-provided full-scale range."""

import numpy as np

# Smallest fraction of a guessed standard range that the signal must span
# (-12 dBFS) for the guess to be kept.
_MIN_RANGE_FILL = 0.25


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
    # Guess a standard full-scale range, so that a signal normalized to it is
    # binned over the whole ADC range rather than over its own swing.
    if mean_val > 0.1:
        guess = (0.0, 1.0)    # Unipolar [0, 1]
    elif abs_max <= 0.6:
        guess = (-0.5, 0.5)   # Bipolar small [-0.5, 0.5]
    else:
        guess = (-1.0, 1.0)   # Bipolar large [-1, 1]

    # The guess is only usable if the signal lies inside it and occupies a
    # reasonable part of it. A 1.2 +/- 0.7 V signal would otherwise lose every
    # sample above 1 V, and a +/-5 mV signal would land in two bins. Fall back
    # to the signal's own span in those cases.
    fits = s_min >= guess[0] and s_max <= guess[1]
    fills = (s_max - s_min) >= _MIN_RANGE_FILL * (guess[1] - guess[0])
    if fits and fills:
        return guess
    return float(s_min), float(s_max)
