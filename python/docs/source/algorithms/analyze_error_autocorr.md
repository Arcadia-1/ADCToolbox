# analyze_error_autocorr

## Overview

`analyze_error_autocorr` computes the autocorrelation of ADC errors to reveal temporal patterns. White noise has zero autocorrelation at all lags; correlated errors indicate memory effects, settling issues, or periodic disturbances.

## Syntax

```python
from adctoolbox import analyze_error_autocorr

# Basic usage
result = analyze_error_autocorr(signal, max_lag=100, show_plot=True)

# With custom parameters
result = analyze_error_autocorr(signal, max_lag=200, resolution=12)
```

## Parameters

- **`signal`** (array_like) — Input ADC signal
- **`max_lag`** (int, default=100) — Maximum lag for autocorrelation
- **`resolution`** (int, optional) — ADC resolution in bits
- **`show_plot`** (bool, default=False) — Display autocorrelation plot
- **`ax`** (matplotlib axis, optional) — Axis for plotting

## Returns

Dictionary containing:
- **`autocorr`** — Autocorrelation values
- **`lags`** — Lag values
- **`error`** — Error signal used

## Interpretation

| Autocorrelation Pattern | Likely Cause |
|-------------------------|--------------|
| **Peak at lag=0 only** | White noise (ideal) |
| **Decay over few samples** | Low-pass filtering, bandwidth limit |
| **Periodic peaks** | Switching artifacts, clock coupling |
| **Slow decay** | 1/f noise, drift |

## Use Cases

- Distinguish white noise from colored noise
- Detect memory effects in pipelined ADCs
- Identify periodic disturbances

## See Also

- [`analyze_error_pdf`](analyze_error_pdf.md) — Error distribution
- [`analyze_error_spectrum`](analyze_error_spectrum.md) — Frequency domain view
- [`analyze_error_envelope_spectrum`](analyze_error_envelope_spectrum.md) — AM modulation

## References

1. B. Razavi, "Principles of Data Conversion System Design," IEEE Press, 1995
