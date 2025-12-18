# fit_static_nonlin

## Overview

`fit_static_nonlin` extracts static nonlinearity coefficients (k2, k3) from a distorted sine wave signal. This quantifies the 2nd-order and 3rd-order nonlinearity of an ADC's transfer function, which are the primary sources of harmonic distortion.

## Syntax

```python
from adctoolbox import fit_static_nonlin

# Extract quadratic nonlinearity only
k2, k3, fitted_sine, fitted_transfer = fit_static_nonlin(distorted_signal, order=2)

# Extract both quadratic and cubic
k2, k3, fitted_sine, fitted_transfer = fit_static_nonlin(distorted_signal, order=3)

# Higher order polynomial (advanced)
k2, k3, fitted_sine, fitted_transfer = fit_static_nonlin(distorted_signal, order=5)
```

## Parameters

- **`sig_distorted`** (array_like) — Distorted sine wave signal samples
- **`order`** (int) — Polynomial order for fitting (typically 2-3)
  - order=2: Quadratic nonlinearity only (k2)
  - order=3: Quadratic + cubic nonlinearity (k2, k3)
  - order>3: Higher-order terms (advanced use)

## Returns

Tuple containing:
- **`k2_extracted`** (float) — Quadratic nonlinearity coefficient
  - For ideal ADC: k2 = 0
  - Represents 2nd-order distortion (HD2)
  - Returns NaN if order < 2

- **`k3_extracted`** (float) — Cubic nonlinearity coefficient
  - For ideal ADC: k3 = 0
  - Represents 3rd-order distortion (HD3)
  - Returns NaN if order < 3

- **`fitted_sine`** (array) — Fitted ideal sine wave input (reference signal)
  - Same length as sig_distorted
  - This is the ideal sine extracted from distorted signal

- **`fitted_transfer`** (tuple) — Fitted transfer curve for plotting
  - (x, y) where x is 1000 smooth input points, y is output
  - For ideal system: y = x (straight line)

## Transfer Function Model

The ADC transfer function is modeled as:

```
y = x + k2·x² + k3·x³
```

where:
- **x** = ideal input (zero-mean sine)
- **y** = actual output (zero-mean)
- **k2** = 2nd-order nonlinearity coefficient
- **k3** = 3rd-order nonlinearity coefficient

## Algorithm

```python
# 1. Fit sine to get ideal input reference
result = fit_sine_4param(sig_distorted)
fitted_sine = result['fitted_signal']

# 2. Remove DC, normalize
x = fitted_sine - mean(fitted_sine)
y = sig_distorted - mean(sig_distorted)

# 3. Construct polynomial basis
# For order=3: [x, x², x³]
A = np.column_stack([x**i for i in range(1, order+1)])

# 4. Least-squares solve: A × coeffs = y
coeffs = np.linalg.lstsq(A, y)[0]

# 5. Extract k2, k3
k2 = coeffs[1] if order >= 2 else np.nan
k3 = coeffs[2] if order >= 3 else np.nan
```

## Examples

### Example 1: Extract Nonlinearity Coefficients

```python
import numpy as np
from adctoolbox import fit_static_nonlin

# Analyze distorted signal
k2, k3, fitted_sine, fitted_transfer = fit_static_nonlin(distorted_signal, order=3)

print(f"2nd-order coefficient k2: {k2:.6f}")
print(f"3rd-order coefficient k3: {k3:.6f}")

# Interpret
if abs(k2) > abs(k3):
    print("Dominant: 2nd-order (even) nonlinearity")
else:
    print("Dominant: 3rd-order (odd) nonlinearity")
```

### Example 2: Plot Nonlinearity Curve

```python
import matplotlib.pyplot as plt

k2, k3, fitted_sine, fitted_transfer = fit_static_nonlin(signal, order=3)
transfer_x, transfer_y = fitted_transfer

# Plot transfer function
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Transfer curve
ax1.plot(transfer_x, transfer_y, 'b-', linewidth=2, label='Actual')
ax1.plot(transfer_x, transfer_x, 'k--', linewidth=1, label='Ideal')
ax1.set_xlabel('Input (V)')
ax1.set_ylabel('Output (V)')
ax1.set_title('Transfer Function')
ax1.legend()
ax1.grid(True)

# Nonlinearity error
ax2.plot(transfer_x, transfer_y - transfer_x, 'r-', linewidth=2)
ax2.set_xlabel('Input (V)')
ax2.set_ylabel('Nonlinearity Error (V)')
ax2.set_title(f'Static Nonlinearity\nk2={k2:.6f}, k3={k3:.6f}')
ax2.grid(True)
ax2.axhline(0, color='k', linestyle='--', linewidth=1)

plt.tight_layout()
plt.show()
```

### Example 3: Relate Coefficients to Harmonic Distortion

```python
from adctoolbox import fit_static_nonlin, analyze_spectrum

# Extract nonlinearity coefficients
k2, k3, _, _ = fit_static_nonlin(signal, order=3)

# Measure harmonic distortion
result = analyze_spectrum(signal, fs=800e6, harmonic=5)
hd2_db = result['harmonic_powers'][0]  # 2nd harmonic
hd3_db = result['harmonic_powers'][1]  # 3rd harmonic

print(f"k2 = {k2:.6f} → HD2 = {hd2_db:.2f} dBc")
print(f"k3 = {k3:.6f} → HD3 = {hd3_db:.2f} dBc")

# Theoretical relationship (for small distortion):
# HD2 ≈ 20·log10(k2/4)
# HD3 ≈ 20·log10(k3/6)
```

### Example 4: Before/After Calibration

```python
# Before calibration
k2_before, k3_before, _, _ = fit_static_nonlin(signal_uncal, order=3)

# After calibration
k2_after, k3_after, _, _ = fit_static_nonlin(signal_cal, order=3)

print("Before Calibration:")
print(f"  k2 = {k2_before:.6f}, k3 = {k3_before:.6f}")
print("After Calibration:")
print(f"  k2 = {k2_after:.6f}, k3 = {k3_after:.6f}")
print(f"Improvement: {20*np.log10(abs(k2_before/k2_after)):.1f} dB (k2)")
print(f"             {20*np.log10(abs(k3_before/k3_after)):.1f} dB (k3)")
```

## Interpretation

### Coefficient Magnitude

| Coefficient | Typical Range | Interpretation |
|-------------|---------------|----------------|
| **k2** | < 0.001 | Excellent linearity (HD2 < -80 dBc) |
| **k2** | 0.001 - 0.01 | Good linearity (HD2: -60 to -80 dBc) |
| **k2** | > 0.01 | Poor linearity (HD2 > -60 dBc) |
| **k3** | < 0.0001 | Excellent linearity (HD3 < -80 dBc) |
| **k3** | 0.0001 - 0.001 | Good linearity (HD3: -60 to -80 dBc) |
| **k3** | > 0.001 | Poor linearity (HD3 > -60 dBc) |

### Coefficient Sign

| Sign | Physical Meaning |
|------|------------------|
| **k2 > 0** | Upward curvature (compression → expansion) |
| **k2 < 0** | Downward curvature (expansion → compression) |
| **k3 > 0** | Positive cubic term (soft saturation) |
| **k3 < 0** | Negative cubic term (hard limiting) |

### Dominant Nonlinearity

| Condition | ADC Characteristic |
|-----------|-------------------|
| **\|k2\| >> \|k3\|** | Even-order dominant (differential pair mismatch) |
| **\|k3\| >> \|k2\|** | Odd-order dominant (single-ended nonlinearity) |
| **\|k2\| ≈ \|k3\|** | Mixed nonlinearity |

## Limitations

- **Cannot extract gain error**: Sine fitting absorbs amplitude variations
  - Gain calibration requires DC sweep or multi-amplitude testing
- **Assumes static nonlinearity**: Dynamic effects (memory, settling) not captured
- **Single-tone only**: Multi-tone distortion (IMD) requires different analysis
- **Numerical stability**: Order > 10 may cause ill-conditioning

## Common Issues

### High-Order Fitting (order > 5)
```python
# BAD: May overfit or become numerically unstable
k2, k3, _, _ = fit_static_nonlin(signal, order=15)

# GOOD: Use order 2-3 for most applications
k2, k3, _, _ = fit_static_nonlin(signal, order=3)
```

### Interpreting NaN Results
```python
k2, k3, _, _ = fit_static_nonlin(signal, order=2)
# k3 will be NaN because order < 3

if not np.isnan(k3):
    print(f"k3 = {k3}")
else:
    print("k3 not extracted (order < 3)")
```

## Use Cases

- **Characterize ADC linearity** quantitatively
- **Compare designs** (architecture, process, layout)
- **Validate calibration** (before/after)
- **Root cause analysis** (k2 vs k3 dominance)
- **Predict harmonic distortion** from coefficients

## See Also

- [`analyze_spectrum`](analyze_spectrum.md) — Measure HD2, HD3 in frequency domain
- [`analyze_decomposition_time`](analyze_decomposition_time.md) — Time-domain harmonic extraction
- [`analyze_inl_from_sine`](analyze_inl_from_sine.md) — INL/DNL (code-level nonlinearity)
- [`calibrate_weight_sine`](calibrate_weight_sine.md) — Calibrate to reduce k2/k3

## References

1. S. Max, "Fast Accurate and Complete ADC Testing," Proc. IEEE ITC, 1989
2. J. Doernberg et al., "Full-Speed Testing of A/D Converters," IEEE JSSC, 1984
3. IEEE Std 1241-2010, "IEEE Standard for Terminology and Test Methods for ADCs"
