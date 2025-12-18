# analyze_inl_from_sine

## Overview

`analyze_inl_from_sine` computes Integral Nonlinearity (INL) and Differential Nonlinearity (DNL) from sine wave histogram test using the inverse cosine method. This is the Python implementation following IEEE Std 1241-2010.

## Syntax

```python
from adctoolbox import analyze_inl_from_sine

# Basic usage with auto-detected parameters
result = analyze_inl_from_sine(data)

# Specify resolution
result = analyze_inl_from_sine(data, num_bits=12)

# Adjust clipping
result = analyze_inl_from_sine(data, clip_percent=0.05, show_plot=True)

# No plotting, just computation
result = analyze_inl_from_sine(data, show_plot=False)
```

## Parameters

- **`data`** (array_like) — ADC output signal
  - Analog: Float values (normalized 0-1 or full-scale voltage)
  - Digital: Integer codes or float representation
- **`num_bits`** (int, optional) — ADC resolution in bits. If None, inferred from data range
- **`full_scale`** (float, optional) — Full-scale voltage for quantization
  - If provided: codes = round(data × 2^num_bits / full_scale)
  - If None: assumes normalized input (0-1 range)
- **`clip_percent`** (float, default=0.01) — Fraction of code range to exclude from edges (0.01 = 1%)
  - Removes unreliable bins near saturation
- **`show_plot`** (bool, default=True) — Display INL/DNL plots
- **`show_title`** (bool, default=True) — Show auto-generated title with min/max ranges
- **`col_title`** (str, optional) — Column title above DNL plot (e.g., "N = 2^10")
- **`ax`** (matplotlib axis, optional) — Axis to plot on. If None, uses current axis

## Returns

Dictionary containing:

- **`inl`** — Integral nonlinearity (LSB units)
- **`dnl`** — Differential nonlinearity (LSB units)
- **`code`** — Corresponding code values (x-axis)

## Algorithm

### 1. Histogram Construction

```python
code_range = np.arange(np.floor(data.min()), np.ceil(data.max()) + 1)
clip_bins = int(clip_percent * len(code_range) / 2)
code_valid = code_range[clip_bins : -clip_bins]

hist_counts, _ = np.histogram(data, bins=len(code_valid))
```

### 2. Inverse Cosine Transform

Sine wave PDF is `p(x) ∝ 1/√(1 - x²)`, corresponding CDF:

```python
CDF = np.cumsum(hist_counts) / np.sum(hist_counts)
Linearized_CDF = -np.cos(np.pi * CDF)
```

This maps the nonlinear sine wave distribution to a linear ideal ADC response.

### 3. DNL Calculation

```python
DNL_raw = np.diff(Linearized_CDF)
DNL_normalized = DNL_raw / np.mean(DNL_raw) * (num_codes - 1) - 1
DNL = DNL_normalized - np.mean(DNL_normalized)  # Remove offset
```

**Units**: LSB (Least Significant Bit)
- DNL = 0: Ideal step size
- DNL = -1: Missing code
- DNL > 0: Code width > 1 LSB

### 4. INL Calculation

```python
INL = np.cumsum(DNL)
```

**Units**: LSB
- INL = 0: Ideal transfer function
- INL > 0: Output higher than ideal
- INL < 0: Output lower than ideal

## Examples

### Example 1: Basic Usage

```python
import numpy as np
from adctoolbox import analyze_inl_from_sine

# Analyze 12-bit ADC output
result = analyze_inl_from_sine(adc_data, num_bits=12, show_plot=True)

print(f"Peak INL: {np.max(np.abs(result['inl'])):.3f} LSB")
print(f"Peak DNL: {np.max(np.abs(result['dnl'])):.3f} LSB")
print(f"DNL range: [{result['dnl'].min():.3f}, {result['dnl'].max():.3f}] LSB")
```

### Example 2: Tight Clipping

```python
# Exclude top/bottom 5% to avoid saturation effects
result = analyze_inl_from_sine(data, clip_percent=0.05)
```

### Example 3: Custom Plotting

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 8))
result = analyze_inl_from_sine(data, ax=ax, col_title="Test Condition A")
plt.tight_layout()
plt.show()
```

### Example 4: Batch Analysis

```python
# Analyze multiple datasets
datasets = [data1, data2, data3]
results = [analyze_inl_from_sine(d, show_plot=False) for d in datasets]

for i, res in enumerate(results):
    print(f"Dataset {i+1}: INL={np.max(np.abs(res['inl'])):.2f} LSB")
```

## Interpretation

### DNL Analysis

| DNL Value | Meaning |
|-----------|---------|
| `DNL ≈ 0` | Ideal uniform code width |
| `DNL < -0.5` | Code width < 0.5 LSB → potential missing code |
| `DNL = -1` | Missing code (zero histogram hits) |
| `DNL > 0.5` | Code width > 1.5 LSB → significant nonlinearity |

### INL Analysis

| INL Value | ADC Quality |
|-----------|-------------|
| `max(\|INL\|) < 0.5` | Excellent (< 0.5 LSB error) |
| `max(\|INL\|) < 1.0` | Good (< 1 LSB error) |
| `max(\|INL\|) > 2.0` | Poor, needs calibration |
| INL shape: bow | Gain/offset error |
| INL shape: S-curve | 2nd-order nonlinearity |
| INL shape: periodic | Cyclic error (e.g., flash ADC) |

## Limitations

- **Sine wave input required**: Assumes sine wave histogram PDF `∝ 1/√(1 - x²)`
  - Ramp or triangle inputs require different methods
- **Sufficient samples**: Requires `>> 2^N` samples for N-bit ADC (typically 10× minimum)
  - For 12-bit ADC: need > 40,000 samples for reliable results
- **Clipping sensitive**: Edge bins unreliable due to saturation → adjust `clip_percent`
- **Missing codes**: DNL = -1 causes CDF discontinuities, may need special handling

## Common Issues

### Low Sample Count
```python
# BAD: Only 1000 samples for 12-bit ADC
result = analyze_inl_from_sine(data[:1000], num_bits=12)  # Noisy results

# GOOD: Sufficient samples
result = analyze_inl_from_sine(data[:50000], num_bits=12)  # Clean results
```

### Signal Clipping
```python
# If signal clips at rails, increase clip_percent
result = analyze_inl_from_sine(data, clip_percent=0.05)  # Exclude 5% from edges
```

## See Also

- [`fit_sine_4param`](fit_sine_4param.md) — Sine wave parameter extraction
- [`calibrate_weight_sine`](calibrate_weight_sine.md) — Digital weight calibration to reduce INL/DNL
- [`analyze_spectrum`](analyze_spectrum.md) — Frequency-domain linearity (SFDR, THD)
- [`compute_inl_from_sine`](../api/aout.rst) — Core computation without plotting

## References

1. IEEE Std 1241-2010, Section 5.5, "Histogram Test Method"
2. J. Doernberg et al., "Full-Speed Testing of A/D Converters," IEEE JSSC, 1984
3. M. F. Wagdy and W. Ng, "Validity of Uniform Quantization Error Model for Sinusoidal Signals Without and With Dither," IEEE Trans. IM, 1989
