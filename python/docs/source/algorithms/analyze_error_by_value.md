# analyze_error_by_value

## Overview

`analyze_error_by_value` bins sine-fit residuals by signal value, revealing value-dependent patterns such as static nonlinearity trends and residual noise changes.

This is a residual diagnostic, not strict code-domain INL/DNL extraction. Use the dedicated sine/ramp INL/DNL tools when transfer-curve or code-domain linearity accuracy is required.

## Syntax

```python
from adctoolbox import analyze_error_by_value

# Basic usage
result = analyze_error_by_value(signal, create_plot=True)

# Increase bin count to inspect finer value-scale structure
result = analyze_error_by_value(signal, n_bins=256, create_plot=True)
```

## Parameters

- **`signal`** (array_like) — Input ADC signal (sine wave excitation)
- **`norm_freq`** (float, optional) — Normalized input frequency. If omitted, the sine fit estimates it.
- **`n_bins`** (int, default=100) — Number of value bins. Too few bins can average away code-scale errors; too many bins can produce sparse/noisy estimates.
- **`clip_percent`** (float, default=0.01) — Fraction of value bins clipped from each edge.
- **`value_range`** (tuple, optional) — Explicit value range mapped to the first/last bins.
- **`create_plot`** (bool, default=True) — Display value-binned residual plots.
- **`ax`** (matplotlib axis, optional) — Axis for plotting

## Returns

Dictionary containing:
- **`error_mean`** — Mean residual per value bin
- **`error_rms`** — RMS residual per value bin
- **`value_bin_centers`** — Physical signal value at each bin center
- **`count_per_bin`** — Number of samples contributing to each bin
- **`bin_indices`** — Value-bin index assigned to each sample
- **`error`** — Raw residual, `signal - fitted_signal`

## Use Cases

- Identify value-dependent residual trends
- Reveal systematic nonlinearity patterns
- Check whether bins are sufficiently populated via `count_per_bin`
- Validate calibration effectiveness

## Interpretation Notes

- The plotted mean curve is a value-binned conditional mean residual, not an INL curve.
- Too few bins can hide alternating or code-scale errors by averaging adjacent structure together.
- Too many bins can leave low-count bins and noisy estimates.
- For strict static INL/DNL, use ramp or sine histogram analysis instead.

## See Also

- [`analyze_error_by_phase`](../api/aout.rst) — Error vs. signal phase
- [`analyze_inl_from_sine`](analyze_inl_from_sine.md) — INL/DNL analysis

## References

1. IEEE Std 1241-2010, "IEEE Standard for Terminology and Test Methods for ADCs"
