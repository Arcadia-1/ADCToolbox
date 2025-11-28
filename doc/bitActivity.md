# bitActivity

## Overview

`bitActivity` analyzes and visualizes the percentage of 1's in each bit position of ADC digital output. This simple but powerful diagnostic tool helps identify stuck bits, DC offset patterns, and abnormal bit usage that indicate hardware faults or improper biasing.

## Syntax

```matlab
bit_usage = bitActivity(bits)
bit_usage = bitActivity(bits, 'AnnotateExtremes', true)
```

```python
# Python equivalent: python/src/adctoolbox/bit_activity.py
from adctoolbox import bit_activity
bit_usage = bit_activity(bits, annotate_extremes=True)
```

## Input Arguments

- **`bits`** — Binary matrix (N × B)
  - N = number of samples
  - B = number of bits
  - Bit order: MSB to LSB (column 1 = MSB, column B = LSB)
  - Values: 0 or 1

### Optional Parameters

- **`'AnnotateExtremes'`** — Annotate bits with >95% or <5% activity (default: `true`)

## Output Arguments

- **`bit_usage`** — Percentage of 1's for each bit (1 × B array)
  - Values range from 0% to 100%
  - Ideal value: ~50% (bit toggles equally between 0 and 1)

## Algorithm

```
1. For each bit position b = 1 to B:
   bit_usage(b) = mean(bits(:, b)) × 100

2. Create bar chart with bit_usage values

3. Add reference line at 50% (ideal)

4. If AnnotateExtremes enabled:
   For each bit:
     If bit_usage > 95%: Label as "stuck high"
     If bit_usage < 5%:  Label as "stuck low"
```

## Examples

### Example 1: Basic Usage

```matlab
% Load SAR ADC bit outputs (N×12 matrix)
bits = readmatrix('sar_adc_12bit.csv');

% Analyze bit activity
bit_usage = bitActivity(bits);

fprintf('Bit Activity (MSB → LSB):\n');
for b = 1:length(bit_usage)
    fprintf('  Bit %2d: %5.1f%%\n', b, bit_usage(b));
end
```

**Output:**
```
Bit Activity (MSB → LSB):
  Bit  1:  49.2%
  Bit  2:  50.8%
  Bit  3:  48.7%
  ...
  Bit 12:  51.3%
```

### Example 2: Detect Stuck Bit

```matlab
% Data with bit 5 stuck high (manufacturing defect)
bits = readmatrix('faulty_adc.csv');
bit_usage = bitActivity(bits, 'AnnotateExtremes', true);

% Check for extreme values
stuck_high = find(bit_usage > 95);
stuck_low = find(bit_usage < 5);

if ~isempty(stuck_high)
    fprintf('⚠ Bits stuck HIGH: %s\n', mat2str(stuck_high));
end

if ~isempty(stuck_low)
    fprintf('⚠ Bits stuck LOW: %s\n', mat2str(stuck_low));
end
```

**Output:**
```
⚠ Bits stuck HIGH: [5]
```

### Example 3: Identify DC Offset Pattern

```matlab
% ADC with positive DC offset
bits = readmatrix('dc_offset_positive.csv');
bit_usage = bitActivity(bits);

% MSB-to-LSB trend indicates DC offset direction
if bit_usage(1) > 70  % MSB highly active
    fprintf('Large positive DC offset detected\n');
    fprintf('MSB activity: %.1f%% (>>50%%)\n', bit_usage(1));
elseif bit_usage(1) < 30
    fprintf('Large negative DC offset detected\n');
    fprintf('MSB activity: %.1f%% (<<50%%)\n', bit_usage(1));
end
```

**Output:**
```
Large positive DC offset detected
MSB activity: 78.5% (>>50%)
```

### Example 4: Verify Test Signal Coverage

```matlab
% Check if test sine wave covers full ADC range
bits = readmatrix('sinewave_test.csv');
bit_usage = bitActivity(bits);

% For sine wave, MSB should be ~50%, LSBs slightly higher
% due to more time spent near mid-range
fprintf('Signal Coverage Check:\n');
fprintf('  MSB activity: %.1f%% (expect ~50%%)\n', bit_usage(1));
fprintf('  Mid-bits activity: %.1f%% (expect 50-55%%)\n', mean(bit_usage(4:8)));
fprintf('  LSB activity: %.1f%% (expect ~50%%)\n', bit_usage(end));

if all(bit_usage > 20 & bit_usage < 80)
    fprintf('  ✓ Full-scale coverage confirmed\n');
else
    fprintf('  ✗ WARNING: Signal may not cover full ADC range\n');
end
```

**Output:**
```
Signal Coverage Check:
  MSB activity: 49.8% (expect ~50%)
  Mid-bits activity: 51.2% (expect 50-55%)
  LSB activity: 50.3% (expect ~50%)
  ✓ Full-scale coverage confirmed
```

## Interpretation

### Expected Patterns

| Bit Activity | Interpretation | Action |
|--------------|----------------|--------|
| **45-55%** | Normal operation | No action |
| **>95%** | Bit stuck HIGH | Hardware fault, replace device |
| **<5%** | Bit stuck LOW | Hardware fault, replace device |
| **MSB >70%** | Positive DC offset or clipping | Adjust input biasing |
| **MSB <30%** | Negative DC offset or clipping | Adjust input biasing |
| **Gradual trend** | DC offset pattern | Check input common-mode voltage |
| **All bits ~50%** | Ideal (zero DC, full coverage) | Good test setup |

### Diagnostic Patterns

**Healthy ADC with Zero DC:**
```
Bit 1 (MSB):  49.5%
Bit 2:        50.2%
Bit 3:        49.8%
...
Bit 12 (LSB): 50.1%
```

**ADC with Positive DC Offset:**
```
Bit 1 (MSB):  75.3%  ← High activity
Bit 2:        62.1%
Bit 3:        54.8%
Bit 4:        51.2%
...           ≈50%   ← Returns to normal
```

**ADC with Bit 3 Stuck HIGH:**
```
Bit 1 (MSB):  49.2%
Bit 2:        50.5%
Bit 3:        98.7%  ← Stuck bit!
Bit 4:        49.1%
...
```

**Insufficient Input Amplitude (not full-scale):**
```
Bit 1 (MSB):  15.2%  ← Low activity (signal not reaching high codes)
Bit 2:        28.5%
Bit 3:        42.1%
...
Bit 12 (LSB): 49.8%
```

### Physical Causes

| Symptom | Likely Cause |
|---------|--------------|
| Single bit >95% | Open circuit, shorted to VDD |
| Single bit <5% | Open circuit, shorted to GND |
| MSB >70%, others normal | Input clipping high, large +DC offset |
| MSB <30%, others normal | Input clipping low, large -DC offset |
| All bits >60% | Input biased too high |
| All bits <40% | Input biased too low |
| Random bits ~50%, some ~0/100% | Multiple stuck bits (severe fault) |

## Limitations

1. **Requires Adequate Sample Size**: At least N > 1000 samples recommended for accurate percentage calculation. For N < 100, percentages may not be representative.

2. **Input Signal Dependent**: Bit activity depends on input signal characteristics:
   - Sine wave: MSB ~50%, mid-bits slightly higher
   - Ramp: All bits ~50% (ideal)
   - DC: MSB determined by DC level, LSBs random

3. **Does Not Detect Dynamic Faults**: This tool only detects static stuck bits. Dynamic faults (bit flips at high speed, metastability) are not captured.

4. **Cannot Distinguish DC Offset from Clipping**: High MSB activity could indicate either positive DC offset or input signal clipping high.

## Use Cases

### Production Test: Stuck Bit Detection
Fast pass/fail screening for manufacturing defects.

```matlab
bits = readmatrix('production_unit_456.csv');
bit_usage = bitActivity(bits, 'AnnotateExtremes', false);

% Fail if any bit has <5% or >95% activity
if any(bit_usage < 5 | bit_usage > 95)
    fprintf('FAIL: Stuck bit detected\n');
    fail_bits = find(bit_usage < 5 | bit_usage > 95);
    fprintf('  Faulty bits: %s\n', mat2str(fail_bits));
else
    fprintf('PASS: All bits functional\n');
end
```

### Debug Setup: Verify Input Signal Range
Ensure test signal covers full ADC range.

```matlab
bits = readmatrix('test_signal.csv');
bit_usage = bitActivity(bits);

% MSB should be ~50% for full-scale sine wave
if bit_usage(1) < 40 || bit_usage(1) > 60
    warning('MSB activity %.1f%% - signal may not be full-scale', bit_usage(1));
    fprintf('  Increase input amplitude to cover full ADC range\n');
end
```

### Characterization: DC Offset Measurement
Quantify DC offset from bit activity.

```matlab
bits = readmatrix('characterization_data.csv');
bit_usage = bitActivity(bits);

% Estimate DC offset from MSB activity
% For 12-bit ADC: MSB weight = 2048 codes
nBits = size(bits, 2);
msb_weight = 2^(nBits - 1);

% DC offset ≈ (MSB_activity - 50%) × MSB_weight / 50%
dc_offset_codes = (bit_usage(1) - 50) / 50 * msb_weight;

fprintf('Estimated DC offset: %.1f codes (%.2f LSB)\n', ...
        dc_offset_codes, dc_offset_codes);
```

## See Also

- [`toolset_dout`](toolset_dout.md) — Digital output analysis suite (includes bitActivity)
- [`weightScaling`](weightScaling.md) — Bit weight visualization
- [`overflowChk`](overflowChk.md) — Overflow detection
- [`FGCalSine`](FGCalSine.md) — Foreground calibration

## References

1. **IEEE Std 1241-2010** — Standard for Terminology and Test Methods for Analog-to-Digital Converters, Section 5.4 (Digital Output Testing)
2. Kester, W., "ADC Input Noise: The Good, The Bad, and The Ugly," *Analog Devices Tutorial MT-004*
3. Baker, R.J., "CMOS Circuit Design, Layout, and Simulation," Wiley-IEEE Press, 4th ed., 2019

---

## Version History

| Version | Date | Key Features |
|---------|------|--------------|
| **v1.0** | 2025-01-28 | Initial documentation for bitActivity |
