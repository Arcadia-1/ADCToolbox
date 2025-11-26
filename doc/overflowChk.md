# overflowChk

## Overview

`overflowChk` visualizes SAR ADC residue distribution at each bit decision to detect overflow and quantization redundancy. Identifies bit-level saturation issues.

## Syntax

```matlab
data_decom = overflowChk(raw_code, weight)
data_decom = overflowChk(raw_code, weight, OFB)
```

## Input Arguments

- **`raw_code`** — Binary bit matrix (N×M, N samples × M bits)
- **`weight`** — Calibrated bit weights (1×M vector)
- **`OFB`** — Overflow bit position (default: `M`)
  - Bit position from LSB to check for overflow (typically LSB for redundancy)

## Output Arguments

- **`data_decom`** — Normalized residue distribution (N×M matrix)
  - Each column: residue after bit `i` decision, normalized to [0, 1]

## Algorithm

### 1. Residue Calculation

For each bit position `i = 1:M`:
```
residue(:, i) = sum(raw_code(:, i:M) .* weight(i:M)) / sum(weight(i:M))
```

This computes the normalized accumulated weight from bit `i` to LSB.

### 2. Overflow Detection

At bit `M - OFB + 1`:
```
overflow_high = (residue(:, M - OFB + 1) >= 1)
overflow_low  = (residue(:, M - OFB + 1) <= 0)
normal        = ~(overflow_high | overflow_low)
```

### 3. Visualization

Generates scatter plot showing:
- **Blue dots**: Normal samples (residue ∈ [0, 1])
- **Red dots** (offset left): High overflow (residue ≥ 1)
- **Yellow dots** (offset right): Low overflow (residue ≤ 0)
- **Red envelope**: `[min(residue), max(residue)]` range per bit
- **Black lines**: Ideal bounds at 0 and 1
- **Annotations**: Overflow percentages at top/bottom of each bit

## Examples

### Example 1: Basic Overflow Check

```matlab
bits = readmatrix('sar_output.csv');
[weight, ~, ~] = FGCalSine(bits);

figure;
data_decom = overflowChk(bits, weight);
title('SAR ADC Residue Distribution');
```

### Example 2: Custom Overflow Bit

```matlab
% Check overflow at 3rd LSB (for 3-bit redundancy)
data_decom = overflowChk(bits, weight, 3);
```

## Interpretation

### Normal Operation

| Observation | Interpretation |
|-------------|----------------|
| Residue ∈ [0, 1] for all bits | No overflow, ideal operation |
| Tight scatter around 0.5 | Low quantization noise, good SNR |
| Uniform vertical distribution | Proper bit weighting |

### Overflow Conditions

| Observation | Cause | Action |
|-------------|-------|--------|
| Red dots (top) | `residue ≥ 1` → insufficient MSB weight | Increase MSB weight or input range |
| Yellow dots (bottom) | `residue ≤ 0` → negative residue | Check for bit inversions or weight errors |
| Red envelope outside [0,1] | Saturation at bit level | Redesign comparator thresholds |
| High % at MSB | Input signal clipping | Reduce input amplitude |
| High % at LSB | Redundancy bits not utilized | Adjust DAC settling or bit cycling |

### Redundancy Analysis

**Ideal redundancy**: Residue range `[0, 1 + redundancy_factor]` where:
```
redundancy_factor = weight(i) / sum(weight(i+1:M)) - 1
```

Example: For 1-bit redundancy, `weight(i) ≈ 2 × sum(weight(i+1:M))` → range [0, 2].

**Observations**:
- Range [0, 1]: No redundancy (binary-weighted)
- Range [0, 1.5]: 0.5-bit redundancy
- Range [0, 2]: 1-bit redundancy
- Range > [0, 2]: Over-redundancy (may indicate weight calibration issues)

## Limitations

- **Calibrated weights required**: Uses `weight` vector → must run FGCalSine first
- **Sinewave input**: Works best with full-scale sinewave for uniform code coverage
- **No root cause**: Only detects overflow, doesn't identify circuit cause (comparator offset, DAC settling, etc.)
- **Visualization only**: Provides qualitative analysis; quantitative metrics need custom extraction

## See Also

- [`FGCalSine`](FGCalSine.md) — Obtain calibrated weights
- [`ENoB_bitSweep`](ENoB_bitSweep.md) — Analyze per-bit ENoB contribution (complementary analysis)
- [`INLsine`](INLsine.md) — Linearity analysis

## References

1. B. Murmann, "The Race for the Extra Decibel: A Brief Review of Current ADC Performance Trajectories," IEEE SSCS 2015.
2. M. Hesener et al., "Digital Calibration Techniques for SAR ADCs," JSSC 2007.
