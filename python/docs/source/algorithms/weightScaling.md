# weightScaling

## Overview

`weightScaling` visualizes absolute bit weights with radix annotations, helping designers verify bit weight distribution and identify calibration errors or architectural deviations. The radix (scaling factor between consecutive bits) reveals whether the ADC uses pure binary, sub-radix, or redundant bit weighting.

## Syntax

```matlab
radix = weightScaling(weights)
```

```python
# Python equivalent: python/src/adctoolbox/weight_scaling.py
from adctoolbox import weight_scaling
radix = weight_scaling(weights)
```

## Input Arguments

- **`weights`** — Bit weights (1 × B array), from MSB to LSB
  - Typically obtained from `FGCalSine` calibration
  - Example: `[2048, 1024, 512, 256, ..., 2, 1]` for 12-bit binary

## Output Arguments

- **`radix`** — Radix between consecutive bits (1 × B array)
  - `radix(i) = weights(i-1) / weights(i)` for i = 2:B
  - `radix(1) = NaN` (no radix for first bit)
  - Expected value: ~2.00 for pure binary weighting

## Algorithm

```
1. Calculate radix for each bit pair:
   radix(1) = NaN  # No reference for MSB
   For i = 2 to B:
       radix(i) = weights(i-1) / weights(i)

2. Plot weights on log scale (line + markers)

3. Annotate each bit with its radix:
   text(i, weights(i), sprintf('/%.2f', radix(i)))

4. Return radix array
```

## Visualization

The plot shows:
- **X-axis**: Bit index (1=MSB, B=LSB)
- **Y-axis**: Absolute weight (log scale for wide dynamic range)
- **Markers**: Data points at each bit position
- **Annotations**: Radix values displayed as "/2.00", "/1.95", etc.

## Examples

### Example 1: Verify Binary-Weighted SAR ADC

```matlab
% Load digital bits and calibrate
bits = readmatrix('sar_adc_12bit.csv');
[weight_cal, ~] = FGCalSine(bits, 'freq', 0, 'order', 5);

% Visualize weight scaling
radix = weightScaling(weight_cal);

fprintf('Radix Analysis (Bit i / Bit i+1):\n');
for i = 2:length(radix)
    fprintf('  Bit %2d → %2d: radix = %.3f\n', i-1, i, radix(i));
end

% Check if radix is close to 2.00 (binary)
mean_radix = mean(radix(2:end), 'omitnan');
fprintf('\nMean radix: %.3f\n', mean_radix);

if abs(mean_radix - 2.0) < 0.05
    fprintf('✓ Pure binary weighting confirmed\n');
else
    fprintf('⚠ Non-binary weighting detected\n');
end
```

**Output:**
```
Radix Analysis (Bit i / Bit i+1):
  Bit  1 →  2: radix = 2.003
  Bit  2 →  3: radix = 1.998
  Bit  3 →  4: radix = 2.005
  ...
  Bit 11 → 12: radix = 2.001

Mean radix: 2.001
✓ Pure binary weighting confirmed
```

### Example 2: Identify Sub-Radix Pipeline ADC

```matlab
% Pipeline ADC with 1.5-bit/stage (sub-radix = ~1.90)
bits = readmatrix('pipeline_adc_10bit.csv');
[weight_cal, ~] = FGCalSine(bits, 'freq', 0, 'order', 5);

radix = weightScaling(weight_cal);

% Sub-radix architecture has radix < 2.0
mean_radix = mean(radix(2:end), 'omitnan');
fprintf('Mean radix: %.3f\n', mean_radix);

if mean_radix < 1.95
    fprintf('✓ Sub-radix architecture detected (1.5-bit/stage)\n');
    fprintf('  Provides digital error correction capability\n');
end
```

**Output:**
```
Mean radix: 1.903
✓ Sub-radix architecture detected (1.5-bit/stage)
  Provides digital error correction capability
```

### Example 3: Detect Calibration Error

```matlab
% ADC with calibration failure on bit 5
bits = readmatrix('calibration_error.csv');
[weight_cal, ~] = FGCalSine(bits, 'freq', 0, 'order', 3);

radix = weightScaling(weight_cal);

% Check for outliers (radix deviating >10% from 2.0)
outliers = find(abs(radix - 2.0) > 0.2);

if ~isempty(outliers)
    fprintf('⚠ Abnormal radix detected at bits:\n');
    for i = 1:length(outliers)
        b = outliers(i);
        fprintf('  Bit %d → %d: radix = %.3f (expected ~2.00)\n', ...
                b-1, b, radix(b));
    end
    fprintf('\nPossible causes:\n');
    fprintf('  - Calibration failure\n');
    fprintf('  - Bit stuck or faulty\n');
    fprintf('  - Insufficient calibration polynomial order\n');
end
```

**Output:**
```
⚠ Abnormal radix detected at bits:
  Bit 4 → 5: radix = 3.245 (expected ~2.00)
  Bit 5 → 6: radix = 1.234 (expected ~2.00)

Possible causes:
  - Calibration failure
  - Bit stuck or faulty
  - Insufficient calibration polynomial order
```

### Example 4: Compare Nominal vs Calibrated Weights

```matlab
bits = readmatrix('sar_with_mismatch.csv');

% Nominal binary weights
nBits = size(bits, 2);
weights_nominal = 2.^(nBits-1:-1:0);

% Calibrated weights
[weights_cal, ~] = FGCalSine(bits, 'freq', 0, 'order', 5);

% Plot both
figure;
subplot(1,2,1);
radix_nom = weightScaling(weights_nominal);
title('Nominal Weights (Perfect Binary)');
ylim([0.5, max(weights_nominal)*2]);

subplot(1,2,2);
radix_cal = weightScaling(weights_cal);
title('Calibrated Weights (After FGCalSine)');
ylim([0.5, max(weights_nominal)*2]);

% Quantify improvement
error_nominal = std(radix_nom(2:end) - 2.0, 'omitnan');
error_calibrated = std(radix_cal(2:end) - 2.0, 'omitnan');

fprintf('Radix std dev (nominal): %.4f\n', error_nominal);
fprintf('Radix std dev (calibrated): %.4f\n', error_calibrated);
fprintf('Improvement: %.1f×\n', error_nominal / error_calibrated);
```

**Output:**
```
Radix std dev (nominal): 0.0000  # Perfect binary
Radix std dev (calibrated): 0.0123  # Slight variation from mismatch
Improvement: 1.0×  # (Nominal is ideal reference)
```

## Interpretation

### Radix Values

| Radix Range | Interpretation | Architecture |
|-------------|----------------|--------------|
| **1.95 - 2.05** | Pure binary weighting | SAR, Flash, Folding |
| **1.85 - 1.95** | Sub-radix with redundancy | 1.5-bit/stage Pipeline |
| **1.75 - 1.85** | Strong redundancy | 1.67-bit/stage (Radix-1.8) |
| **> 2.10** | Calibration error or unusual architecture | Check calibration |
| **< 1.70** | Extreme redundancy or error | Verify ADC type |

### Diagnostic Patterns

**Pure Binary-Weighted SAR:**
```
Bit  1 → 2: /2.00
Bit  2 → 3: /2.00
Bit  3 → 4: /2.00
...
All radix ≈ 2.00 ± 0.02
```

**1.5-Bit/Stage Pipeline (Sub-Radix):**
```
Bit  1 → 2: /1.90
Bit  2 → 3: /1.91
Bit  3 → 4: /1.89
...
All radix ≈ 1.90 ± 0.03
```

**Calibration Failure:**
```
Bit  1 → 2: /2.01
Bit  2 → 3: /2.00
Bit  3 → 4: /3.45  ← Outlier!
Bit  4 → 5: /1.23  ← Outlier!
Bit  5 → 6: /2.02
...
Irregular pattern indicates calibration error
```

**Capacitor Mismatch (Uncalibrated SAR):**
```
Bit  1 → 2: /1.97
Bit  2 → 3: /2.05
Bit  3 → 4: /1.93
Bit  4 → 5: /2.08
...
Random variation around 2.00 indicates mismatch
```

### Physical Interpretation

**Radix = weight[i-1] / weight[i]**

For ideal binary:
- MSB = 2048, MSB-1 = 1024 → radix = 2.00
- MSB-1 = 1024, MSB-2 = 512 → radix = 2.00

For 1.5-bit/stage pipeline:
- Each stage resolves 1.5 bits (radix ≈ 2^1.5 ≈ 2.83 analog, but digital weights show radix ≈ 1.90 after redundancy)

For SAR with capacitor mismatch:
- C_MSB = 100.5 fF (nominal 100 fF) → weight slightly off → radix ≠ 2.00

## Limitations

1. **Requires Calibrated Weights**: Input weights should come from `FGCalSine` or similar calibration. Using nominal weights will show perfect radix = 2.00 with no useful information.

2. **Does Not Detect Dynamic Errors**: This is a static weight analysis. Dynamic errors (settling, hysteresis) are not visible.

3. **Log Scale Visualization**: Log scale is necessary for wide dynamic range (MSB to LSB), but small weight variations are less visible.

4. **Single Data Point Per Bit**: Cannot show weight variation across different input levels or temperatures.

5. **Interpretation Depends on Architecture**: Radix deviation from 2.00 may be intentional (sub-radix) or unintentional (mismatch). Requires knowledge of ADC architecture.

## Use Cases

### Verify Calibration Correctness
Ensure calibrated weights follow expected radix pattern.

```matlab
[weights_cal, ~] = FGCalSine(bits, 'freq', 0, 'order', 5);
radix = weightScaling(weights_cal);

% Check if all radix values are reasonable
if all(radix(2:end) > 1.5 & radix(2:end) < 2.5)
    fprintf('✓ Calibration produced valid weights\n');
else
    fprintf('✗ Calibration failed - abnormal weights detected\n');
end
```

### Identify ADC Architecture
Determine if ADC uses binary or sub-radix weighting.

```matlab
[weights_cal, ~] = FGCalSine(bits, 'freq', 0);
radix = weightScaling(weights_cal);

mean_radix = mean(radix(2:end), 'omitnan');

if abs(mean_radix - 2.0) < 0.05
    fprintf('Architecture: Pure binary (SAR/Flash)\n');
elseif abs(mean_radix - 1.9) < 0.05
    fprintf('Architecture: 1.5-bit/stage pipeline with redundancy\n');
else
    fprintf('Architecture: Unknown or mixed radix\n');
end
```

### Debug Specific Bit Errors
Identify which bit has incorrect weight.

```matlab
[weights_cal, ~] = FGCalSine(bits, 'freq', 0, 'order', 5);
radix = weightScaling(weights_cal);

% Find bits with radix deviation >10% from mean
mean_radix = mean(radix(2:end), 'omitnan');
outliers = find(abs(radix - mean_radix) > 0.2);

if ~isempty(outliers)
    fprintf('Suspected faulty bits: %s\n', mat2str(outliers));
end
```

## See Also

- [`toolset_dout`](toolset_dout.md) — Digital output analysis suite (includes weightScaling)
- [`FGCalSine`](FGCalSine.md) — Foreground calibration to extract weights
- [`bitActivity`](bitActivity.md) — Bit usage analysis
- [`ENoB_bitSweep`](ENoB_bitSweep.md) — ENoB vs number of bits
- [`cap2weight`](UtilityFunctions.md#cap2weight) — Calculate weights from capacitor network

## References

1. **IEEE Std 1241-2010** — Standard for Terminology and Test Methods for Analog-to-Digital Converters
2. Murmann, B., "Digitally Assisted Analog Circuits," *IEEE Micro*, vol. 26, no. 2, pp. 38-47, Mar-Apr 2006
3. Lee, H.S., Hodges, D.A., and Gray, P.R., "A Self-Calibrating 15 Bit CMOS A/D Converter," *IEEE Journal of Solid-State Circuits*, vol. 19, no. 6, pp. 813-819, Dec 1984
4. Lewis, S.H., et al., "A 10-b 20-Msample/s Analog-to-Digital Converter," *IEEE Journal of Solid-State Circuits*, vol. 27, no. 3, pp. 351-358, Mar 1992

---

## Version History

| Version | Date | Key Features |
|---------|------|--------------|
| **v1.0** | 2025-01-28 | Initial documentation for weightScaling |
