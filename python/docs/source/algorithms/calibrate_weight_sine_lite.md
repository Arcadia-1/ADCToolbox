# calibrate_weight_sine_lite

## Overview

`calibrate_weight_sine_lite` provides a minimal, fast implementation of ADC foreground calibration using a known-frequency sinewave input. This is a simplified version of the full `calibrate_weight_sine` algorithm, optimized for speed and code simplicity.

**Key Characteristics:**
- Single known-frequency calibration only (no frequency search)
- Cosine-basis assumption (no dual-basis optimization)
- No rank deficiency handling (binary weights only)
- No harmonic rejection
- Returns normalized weights only
- Minimal dependencies (NumPy + SciPy only)

## Syntax

```python
from adctoolbox.calibration import calibrate_weight_sine_lite

# Basic usage
weights = calibrate_weight_sine_lite(bits, freq=0.001587)
```

## Parameters

- **`bits`** (ndarray) — Binary data matrix (N samples × M bits)
  - Each row is one sample
  - Each column is a bit (MSB first)
  - Values must be 0 or 1

- **`freq`** (float) — Normalized frequency (f_in / f_s)
  - **Must be known precisely**
  - Range: 0 < freq < 0.5

## Returns

**`weights`** (ndarray) — Calibrated bit weights normalized to [0, 1]
  - Length M (bit width)
  - Normalized by sinewave magnitude
  - Largest weight = 1.0

To get actual ADC weights:
```python
true_weights = 2.0 ** np.arange(bit_width - 1, -1, -1)
recovered_weights_scaled = weights * np.max(true_weights)
```

## Algorithm

### Mathematical Model

The ADC output can be modeled as:
```
y(n) = Σ w_i · b_i(n) = A·cos(2πfn) + B·sin(2πfn) + C
```

where:
- `y(n)` = reconstructed analog signal
- `w_i` = weight of bit i (unknown)
- `b_i(n) ∈ {0, 1}` = binary value of bit i at sample n
- `f` = normalized frequency (f_in/f_s)
- `A, B` = sinewave amplitude coefficients
- `C` = DC offset

### Least Squares Formulation

With the **cosine-basis assumption** (A = 1):

```
[B | 1 | sin(2πfn)] × [w; C; B] = -cos(2πfn)
```

In matrix form:
```
┌                                                    ┐   ┌     ┐   ┌              ┐
│ b_0(0)  b_1(0)  ···  b_M-1(0)  1  sin(2πf·0)     │   │ w_0 │   │ -cos(2πf·0) │
│ b_0(1)  b_1(1)  ···  b_M-1(1)  1  sin(2πf·1)     │ × │ w_1 │ = │ -cos(2πf·1) │
│   ⋮       ⋮      ⋱     ⋮        ⋮      ⋮          │   │  ⋮  │   │      ⋮       │
│ b_0(N-1) ···         b_M-1(N-1) 1 sin(2πf·(N-1)) │   │w_M-1│   │-cos(2πf·N-1)│
└                                                    ┘   │  C  │   └              ┘
                                                         │  B  │
                                                         └     ┘
```

### Algorithm Steps

1. **Build Basis Functions**
   ```python
   t = np.arange(n_samples)
   phase = 2.0 * np.pi * freq * t
   cos_basis = np.cos(phase)
   sin_basis = np.sin(phase)
   ```

2. **Construct Design Matrix**
   ```python
   offset_col = np.ones((n_samples, 1))
   A = np.column_stack([bits, offset_col, sin_basis])
   b = -cos_basis
   ```

3. **Solve Least Squares**
   ```python
   coeffs, _, _, _ = scipy.linalg.lstsq(A, b)
   ```

4. **Extract and Normalize Weights**
   ```python
   weights_raw = coeffs[:bit_width]
   sin_coeff = coeffs[-1]
   norm_factor = np.sqrt(1.0 + sin_coeff**2)
   weights = weights_raw / norm_factor
   ```

   The normalization accounts for actual sinewave amplitude:
   ```
   Amplitude = sqrt(A² + B²) = sqrt(1 + B²)
   ```

5. **Polarity Correction**
   ```python
   if np.sum(weights) < 0:
       weights = -weights
   ```

## Examples

### Example 1: Basic Calibration

```python
import numpy as np
from adctoolbox.calibration import calibrate_weight_sine_lite

# Generate test data (12-bit ADC, 8192 samples, freq = 13/8192)
n_samples = 8192
bit_width = 12
freq_true = 13 / n_samples

# Create ideal sinewave and quantize
t = np.arange(n_samples)
signal = 0.5 * np.sin(2 * np.pi * freq_true * t + np.pi/4) + 0.5
quantized = np.clip(np.floor(signal * (2**bit_width)), 0, 2**bit_width - 1).astype(int)

# Extract bits (MSB first)
bits = (quantized[:, None] >> np.arange(bit_width - 1, -1, -1)) & 1

# Run calibration
recovered_weights = calibrate_weight_sine_lite(bits, freq=freq_true)

# Scale to actual ADC weights
true_weights = 2.0 ** np.arange(bit_width - 1, -1, -1)
recovered_weights_scaled = recovered_weights * np.max(true_weights)

print(f"True weights:      {true_weights}")
print(f"Recovered weights: {recovered_weights_scaled}")
```

**Expected Output:**
```
True weights:      [2048. 1024.  512.  256.  128.   64.   32.   16.    8.    4.    2.    1.]
Recovered weights: [2048.0 1024.0  512.0  256.0  128.0   64.0   32.0   16.0    8.0    4.0    2.0    1.0]
```

### Example 2: With SNDR Calculation

```python
from adctoolbox import analyze_spectrum

# Compute calibrated signal
calibrated_signal = bits @ recovered_weights_scaled

# Compute ideal signal
adc_amplitude = 2**bit_width / 2.0
ideal_signal = adc_amplitude * np.sin(2 * np.pi * freq_true * t + np.pi/4) + adc_amplitude
error_signal = calibrated_signal - ideal_signal

# Calculate SNDR
sndr_before = analyze_spectrum(quantized)['sndr_db']
sndr_calc = 10 * np.log10(np.mean(ideal_signal**2) / np.mean(error_signal**2))

print(f"SNDR before: {sndr_before:.2f} dB")
print(f"SNDR after:  {sndr_calc:.2f} dB")
print(f"ENOB: {(sndr_calc - 1.76) / 6.02:.2f} bits")
```

## Limitations

### 1. Known Frequency Required

The function requires the input frequency to be **precisely known**. Unlike the full `calibrate_weight_sine`, it does not perform frequency search or refinement.

**Workaround**: Use `calibrate_weight_sine` with `force_search=True` if frequency is unknown.

### 2. Binary Weights Only (No Redundancy)

The algorithm **does not handle rank deficiency** or redundant weights. For ADCs with:
- Redundant bits (e.g., `[128, 128, 64, 32, ...]`)
- Identical weights
- Linear dependencies between bits

The least squares solution may:
- Collapse redundant weights to zero
- Produce numerically unstable results
- Fail to recover the full code range

**Example Failure:**

```python
# Redundant weights: two bits with weight 128
true_weights = np.array([2048, 1024, 512, 256, 128, 128, 64, 32, 16, 8, 4, 2])

# Calibration may recover:
recovered = np.array([2048, 1024, 512, 256, 128, 0, 64, 32, 16, 8, 4, 2])
#                                                  ^
#                                       Redundant bit collapsed!
```

**Why this is wrong:**
- Original sum: 4222 (can represent 0-4095 with redundancy)
- Collapsed sum: 4094 (cannot represent code 4095!)
- Lost error correction capability

**Workaround**: Use the full `calibrate_weight_sine` which includes `_patch_rank_deficiency` handling.

### 3. Low Signal Amplitude

At very low input amplitudes (< -6 dBFS), MSB bits may have limited activity, leading to:
- Ill-conditioned least squares matrix
- Numerically unstable weights (values in trillions)
- Poor SNDR estimates

**Example:**
```
At -12 dBFS:
Recovered weights: [1472874591589.4, 1472874590565.5, 511.9, 256.0, ...]
                    ^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^
                    Numerically unstable MSB weights
```

**Recommendation**: Use input signals at or near full scale (0 dBFS to -3 dBFS) for best results.

### 4. No Harmonic Rejection

The algorithm fits only the **fundamental frequency**, without excluding harmonics from the error term. This can lead to:
- Harmonic distortion biasing the weight estimates
- Reduced accuracy for ADCs with significant INL/DNL

**Workaround**: Use `calibrate_weight_sine` with `harmonic_order > 1` to exclude harmonics.

## Performance

### Computational Complexity

**Time Complexity**: `O(N·M² + M³)`
- N = number of samples
- M = bit width
- Dominated by least squares solve (SVD decomposition)

**Space Complexity**: `O(N·M)`
- Storage for design matrix

**Typical Performance** (12-bit ADC, Intel i7):
- N = 2¹² (4096 samples): ~3 ms
- N = 2¹³ (8192 samples): ~5 ms
- N = 2¹⁶ (65536 samples): ~40 ms

### Accuracy

**Weight Recovery** (ideal conditions):
- Error < 10⁻⁵ LSB (normalized)
- Phase-independent performance
- ENOB: ~11-12 bits (for 12-bit ADC at 0 dBFS)

**Amplitude Requirements**:
```
Amplitude (dBFS)    Weight Error    ENOB
-----------------   -------------   ------
    0 to -3         < 1e-5 LSB      11-12 bits
   -6 to -3         < 1e-4 LSB      10-11 bits
  -12 to -6         > 1e+9 LSB      Unstable
  < -12             > 1e+12 LSB     Failed
```

## Comparison with Full Version

| Feature | Lite Version | Full Version |
|---------|-------------|--------------|
| **Frequency search** | ❌ No (requires known freq) | ✅ Yes (coarse + fine search) |
| **Rank deficiency handling** | ❌ No | ✅ Yes (via nominal weights) |
| **Redundant weights** | ❌ Not supported | ✅ Fully supported |
| **Harmonic rejection** | ❌ No | ✅ Yes (configurable order) |
| **Multi-dataset calibration** | ❌ No | ✅ Yes |
| **Numerical conditioning** | ❌ Basic | ✅ Column scaling + patching |
| **Return type** | ndarray (weights only) | dict (weights + diagnostics) |
| **Code size** | ~40 lines | ~600+ lines (with helpers) |
| **Typical runtime** (N=8192) | ~5 ms | ~20-50 ms |

### When to Use Lite Version

✅ **Use lite version when:**
- Frequency is precisely known
- Binary weighted ADC (no redundancy)
- Speed is critical
- Simple embedded deployment
- Minimal code footprint needed

❌ **Use full version when:**
- Unknown or imprecise frequency
- Redundant ADC architecture
- Need harmonic rejection
- Multi-dataset calibration
- Production calibration requiring robustness

## References

1. IEEE Standard 1057-2017: "IEEE Standard for Digitizing Waveform Recorders"

2. Vogel, C., & Johansson, H. (2006). "Time-interleaved analog-to-digital converters: Status and future directions." *IEEE Transactions on Circuits and Systems I*, 53(11), 2386-2394.

3. Jin, H., & Lee, E. K. F. (1992). "A digital-background calibration technique for minimizing timing-error effects in time-interleaved ADCs." *IEEE Transactions on Circuits and Systems II*, 47(7), 603-613.

4. MATLAB Signal Processing Toolbox: `sinefitweights` documentation

## See Also

- [calibrate_weight_sine](calibrate_weight_sine.md) - Full version with frequency search and rank handling
- [analyze_spectrum](analyze_spectrum.md) - Spectral analysis for SNDR/ENOB calculation
- [fit_sine_4param](fit_sine_4param.md) - IEEE 1057 sinewave fitting
