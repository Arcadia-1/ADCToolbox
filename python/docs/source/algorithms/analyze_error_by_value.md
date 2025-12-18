# analyze_error_by_value

## Overview

`analyze_error_by_value` rearranges ADC errors by their corresponding output codes, revealing code-dependent patterns like DNL-related errors, missing codes, and systematic nonlinearity.

## Syntax

```python
from adctoolbox import analyze_error_by_value

# Basic usage
result = analyze_error_by_value(signal, show_plot=True)

# With resolution specification
result = analyze_error_by_value(signal, resolution=12, show_plot=True)
```

## Parameters

- **`signal`** (array_like) — Input ADC signal (sine wave excitation)
- **`resolution`** (int, optional) — ADC resolution in bits
- **`show_plot`** (bool, default=False) — Display error vs. code plot
- **`ax`** (matplotlib axis, optional) — Axis for plotting

## Returns

Dictionary containing:
- **`error_by_code`** — Errors grouped by ADC code
- **`codes`** — Unique code values
- **`mean_error`** — Mean error per code
- **`std_error`** — Standard deviation per code

## Use Cases

- Identify code-dependent errors (DNL, missing codes)
- Reveal systematic nonlinearity patterns
- Validate calibration effectiveness

## See Also

- [`analyze_error_by_phase`](../api/aout.rst) — Error vs. signal phase
- [`analyze_inl_from_sine`](analyze_inl_from_sine.md) — INL/DNL analysis

## References

1. IEEE Std 1241-2010, "IEEE Standard for Terminology and Test Methods for ADCs"
