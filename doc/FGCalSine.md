# FGCalSine

## Overview

`FGCalSine` estimates per-bit weights and DC offset for ADC calibration using sinewave input. It performs least-squares fitting to match bit-weighted output to a sine series, automatically handling rank deficiency and frequency estimation.

## Syntax

```matlab
[weight, offset, postCal, ideal, err, freqCal] = FGCalSine(bits)
[weight, offset, postCal, ideal, err, freqCal] = FGCalSine(bits, Name, Value)
[weight, offset, postCal, ideal, err, freqCal] = FGCalSine({bits1, bits2, ...}, Name, Value)
```

## Description

`FGCalSine(bits)` calibrates ADC bit weights using sinewave input with automatic frequency detection.

`FGCalSine(bits, Name, Value)` specifies calibration options.

`FGCalSine({bits1, bits2, ...})` performs joint calibration across multiple datasets with shared weights but independent frequencies/harmonics.

## Input Arguments

### Required

- **`bits`** — Binary data matrix or cell array
  - Single dataset: N×M matrix (N samples × M bits, MSB to LSB)
  - Multi-dataset: Cell array `{bits1, bits2, ...}` for joint calibration

### Name-Value Arguments

- **`freq`** — Normalized frequency (Fin/Fs), default: `0`
  - 0 triggers automatic frequency search
  - For multi-dataset: scalar (shared) or vector (per-dataset)

- **`order`** — Harmonic exclusion order, default: `1`
  - 1: fundamental only (no harmonic exclusion)
  - N: exclude harmonics 2 through N from error

- **`fsearch`** — Force fine frequency search, default: `0`
  - 0: skip if frequency provided
  - 1: refine even with known frequency

- **`rate`** — Frequency update rate, default: `0.5`
  - Adaptive rate for iterative frequency refinement (0-1)

- **`reltol`** — Relative error tolerance, default: `1e-12`
  - Stop criterion for frequency search

- **`niter`** — Max iterations, default: `100`
  - Maximum fine-search iterations

- **`nomWeight`** — Nominal bit weights, default: `2.^(M-1:-1:0)`
  - Used for rank deficiency patching

## Output Arguments

- **`weight`** — Calibrated bit weights (1×M vector)
- **`offset`** — DC offset (scalar)
- **`postCal`** — Calibrated signal (1×N vector or cell array)
- **`ideal`** — Fitted sinewave (1×N vector or cell array)
- **`err`** — Residual error after calibration (1×N vector or cell array)
- **`freqCal`** — Refined frequency estimate (scalar or vector)

## Algorithm

### 1. Frequency Estimation (if `freq = 0`)

**Coarse search**: Tests frequency using top 5 MSBs weighted by `nomWeight`, takes median:
```
for each MSB subset (1 to 5 bits):
    freq_est[i] = findFin(bits(:,1:i) * nomWeight(1:i)')
freq = median(freq_est)
```

**Fine search**: Iteratively refines using augmented least-squares with frequency derivative:
```
for iter = 1:niter:
    Solve: [bits, DC, harmonics, ∂/∂freq] × x = sinewave
    Update: freq ← freq + x_freq * rate / N
    Stop if: relative_error < reltol
```

### 2. Rank Deficiency Handling

Detects rank deficiency in `[bits, DC]` matrix and applies patching:

| Condition | Action |
|-----------|--------|
| Constant column | Discard (weight = 0) |
| Adds rank | Keep as independent column |
| Correlated to existing | Merge: `bits_patch(:,j) += bits(:,i) * nomWeight(i)/nomWeight(j)` |
| Cannot merge | Discard with warning (weight = 0) |

### 3. Weight Estimation via Least-Squares

Solves two formulations to resolve sin/cos ambiguity:

**Assumption 1** (cosine = unity):
```
[bits, DC, cos(2θ), cos(3θ), ..., sin(θ), sin(2θ), ...] × x = -cos(θ)
```

**Assumption 2** (sine = unity):
```
[bits, DC, sin(2θ), sin(3θ), ..., cos(θ), cos(2θ), ...] × x = -sin(θ)
```

where `θ = 2π × freq × (0:N-1)'`

**Selection**: Choose assumption with lower residual RMS.

**Normalization**: Divide by fundamental magnitude `w0 = sqrt(1 + x_quad^2)` where `x_quad` is the quadrature coefficient.

### 4. Multi-Dataset Extension

For cell array input:
1. Estimate per-dataset frequencies independently
2. Concatenate all datasets for unified rank patching
3. Build joint least-squares system with per-dataset harmonic basis
4. Solve for shared `weight` and `offset`, normalized by dataset-1 magnitude

## Examples

### Example 1: Basic Calibration

```matlab
bits = readmatrix('dout_SAR_10bit.csv');
[weight, offset, postCal, ideal, err, freq] = FGCalSine(bits);

fprintf('Frequency: %.6f\n', freq);
fprintf('Weights: %s\n', mat2str(weight, 4));
fprintf('Error RMS: %.4f\n', rms(err));
```

### Example 2: With Known Frequency

```matlab
bits = readmatrix('adc_output.csv');
freq = 0.1234;  % Known Fin/Fs

[weight, offset, postCal] = FGCalSine(bits, 'freq', freq, 'order', 5);
```

### Example 3: Multi-Dataset Joint Calibration

```matlab
bits1 = readmatrix('dataset1.csv');
bits2 = readmatrix('dataset2.csv');

% Joint calibration with different frequencies
[weight, offset, postCal, ~, ~, freq] = FGCalSine({bits1, bits2});

fprintf('Dataset 1 freq: %.6f\n', freq(1));
fprintf('Dataset 2 freq: %.6f\n', freq(2));
fprintf('Shared weights: %s\n', mat2str(weight, 4));
```

## Interpretation

### Weight Analysis

| Observation | Interpretation |
|-------------|----------------|
| `weight(i) ≈ 2^(M-i)` | Binary-weighted ADC, ideal behavior |
| `weight(i) ≈ 0` | Bit discarded due to rank deficiency or constant value |
| `sum(weight) < 0` | Auto-corrected by polarity flip |
| Large `weight(i)` deviation | Non-binary weighting or calibration error |

### Error Metrics

- **RMS(err) << RMS(postCal)**: Good calibration
- **High RMS(err)**: May need higher `order` or indicates non-sine distortion
- **Non-convergence**: Try adjusting `rate` or increasing `niter`

### Rank Deficiency

- **Warning "Rank deficiency detected"**: ADC has redundant/correlated bits
- **"Patch warning"**: Some bits cannot be merged → weight set to 0
- **"Patch failed"**: Adjust `nomWeight` to reflect actual bit hierarchy

## Limitations

- **Single-tone only**: Requires pure sinewave input (multi-tone not supported)
- **Harmonic assumption**: Assumes distortion is harmonic; non-harmonic noise becomes residual error
- **Frequency precision**: Coarse search uses only top 5 MSBs → may fail for highly non-binary weights
- **Rank patching**: Relies on `nomWeight` accuracy for correlated bit merging

## See Also

- [`ENoB_bitSweep`](ENoB_bitSweep.md) — Analyze calibration quality vs number of bits
- [`specPlot`](specPlot.md) — Evaluate calibrated signal spectrum
- [`sineFit`](sineFit.md) — Simpler sinewave fitting without bit-level calibration
- [`tomDecomp`](tomDecomp.md) — Time-domain error decomposition

## References

1. M. Hesener et al., "A Digital Offset Calibration Technique for Multi-bit Pipeline ADCs," IEEE J. Solid-State Circuits, 2007.
2. B. Murmann, "ADC Performance Survey 1997-2023," Stanford University.

## Version History

| Version | Date | Key Features |
|---------|------|--------------|
| **v2.0** | 2025-01-26 | Multi-dataset joint calibration support |
| **v1.5** | 2025-01-20 | Improved rank deficiency patching |
| **v1.0** | 2024-12-15 | Initial release: single-dataset calibration |
