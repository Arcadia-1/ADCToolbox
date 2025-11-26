# ENoB_bitSweep

## Overview

`ENoB_bitSweep` evaluates how ENoB improves as more bits are used for FGCalSine calibration. It identifies the optimal bit count for calibration and diagnoses problematic bit positions through color-coded visualization.

## Syntax

```matlab
[ENoB_sweep, nBits_vec] = ENoB_bitSweep(bits)
[ENoB_sweep, nBits_vec] = ENoB_bitSweep(bits, Name, Value)
```

## Description

`ENoB_bitSweep(bits)` sweeps through different bit configurations, using progressively more bits (1, 2, 3, ..., M) for FGCalSine calibration. For each configuration, it computes the ENoB metric to assess calibration quality.

`ENoB_bitSweep(bits, Name, Value)` specifies additional options using name-value pair arguments.

## Input Arguments

### Required

- **`bits`** — Binary data matrix
  N×M numeric matrix
  Each row represents one sample, each column represents one bit. The function assumes bits are ordered from MSB (column 1) to LSB (column M).

### Name-Value Arguments

- **`freq`** — Normalized frequency (Fin/Fs)
  Scalar in range [0, 0.5], default: `0`
  When set to 0, the function automatically estimates the frequency using all available bits.

- **`order`** — Harmonic exclusion order for FGCalSine
  Positive integer, default: `5`
  Number of harmonics to exclude during calibration.

- **`harmonic`** — Number of harmonics for specPlot
  Positive integer, default: `5`
  Number of harmonics to analyze in the spectrum.

- **`OSR`** — Oversampling ratio
  Positive scalar, default: `1`
  Oversampling ratio for spectrum analysis.

- **`winType`** — Window function for specPlot
  Function handle, default: `@hamming`
  Window function applied during FFT. Common options: `@hamming`, `@hanning`, `@rectwin`, `@blackman`.

- **`plot`** — Enable plotting
  0 or 1, default: `1`
  When set to 1, creates a plot showing ENoB vs number of bits used.

## Output Arguments

- **`ENoB_sweep`** — ENoB values
  1×M numeric vector
  Contains the ENoB value for each bit configuration. NaN entries indicate failed calibrations.

- **`nBits_vec`** — Bit count vector
  1×M numeric vector
  Vector [1, 2, 3, ..., M] indicating the number of bits used.

## Examples

### Example 1: Basic Usage

```matlab
% Read digital bit data
bits = readmatrix('dout_SAR_10bit.csv');

% Run ENoB sweep with automatic frequency detection
[ENoB_sweep, nBits_vec] = ENoB_bitSweep(bits);

% Find optimal number of bits
[maxENoB, optimalBits] = max(ENoB_sweep);
fprintf('Optimal: %d bits, ENoB = %.2f\n', optimalBits, maxENoB);
```

### Example 2: With Known Frequency

```matlab
% Read data
bits = readmatrix('dout_SAR_12bit.csv');

% Known normalized frequency
freq = 0.1234;

% Run sweep with specified frequency
[ENoB_sweep, nBits_vec] = ENoB_bitSweep(bits, 'freq', freq, ...
    'order', 5, 'winType', @hamming, 'plot', 1);
```

### Example 3: Without Plotting

```matlab
% Run sweep without generating plot
[ENoB_sweep, nBits_vec] = ENoB_bitSweep(bits, 'plot', 0);

% Custom analysis
figure;
subplot(2,1,1);
plot(nBits_vec, ENoB_sweep, 'o-', 'LineWidth', 2);
xlabel('Number of Bits'); ylabel('ENoB (bits)');
title('ENoB vs Bits Used');
grid on;

subplot(2,1,2);
bar(nBits_vec, diff([0, ENoB_sweep]));
xlabel('Bit Position'); ylabel('ENoB Improvement (bits)');
title('ENoB Gain per Additional Bit');
grid on;
```

## Algorithm

1. **Frequency Determination**: If `freq = 0`, estimates frequency using all bits via FGCalSine
2. **Bit Sweep**: For n = 1 to M, calibrate using bits(:, 1:n) and compute ENoB
3. **Visualization**: Plots ENoB vs bits with color-coded delta annotations:
   - **Black** (Δ ≥ 1.0): Critical bit, adds ≥1 effective bit
   - **Dark red** (Δ ≈ 0.5): Moderate contribution
   - **Bright red** (Δ ≤ 0.0): Minimal/harmful, investigate for errors
   - Failed calibrations marked as NaN

## Interpretation

### Reading the Plot

Each data point is annotated with its ENoB contribution using an **absolute color scale (Δ: 0→1)**:
- **Bit 1**: Absolute ENoB in black (e.g., "0.76")
- **Bit 2+**: Incremental Δ with "+" prefix (e.g., "+1.02", "+0.38")
- **Color coding**: Black (Δ ≥ 1.0) → Dark red (Δ ≈ 0.5) → Bright red (Δ ≤ 0.0)

### Example Analysis

The following figures show real ENoB sweep analysis from two 12-bit SAR ADC datasets:

<table>
<tr>
<td width="50%">
<img src="doc\ENoB_sweep_matlab.png" width="80%">
<br><b>Case 1: Ideal Calibration (no redundancy)</b>
<ul>
<li>All bits contribute ~1.0 ENoB (black annotations)</li>
<li>Monotonic increase from 0.76 → 11.99 bits</li>
<li>Clean convergence, no problematic bits</li>
<li>Optimal: use all 12 bits for calibration</li>
</ul>
</td>
<td width="50%">
<img src="doc\ENoB_sweep_matlab.png" width="80%">
<br><b>Case 2: With Bit Issues (redundancy detected)</b>
<ul>
<li><b>Bits 4-5</b>: Degraded (+0.58, +0.38 red) → Weight errors</li>
<li><b>Bits 11-12</b>: Plateau (+0.00 bright red) → Noise floor</li>
<li><b>Bits 13-15</b>: Resume adding ~1.0 each (black) → Redundancy</li>
<li>Max ENoB = 12.21 at bit 15 (vs 11.99 at bit 12)</li>
</ul>
</td>
</tr>
</table>

**Key Insight**: Case 1 shows optimal binary-weighted performance. Case 2 reveals redundancy architecture where bits 13-15 compensate for earlier bit errors, achieving higher ENoB but requiring more calibration bits.

**Color Scale Interpretation:**
- **Black** (Δ ≥ 1.0): Essential bit, adds ≥1 effective bit
- **Dark red** (Δ ~ 0.5): Worth including, adds half an effective bit
- **Bright red** (Δ < 0.2): Near noise floor, minimal value
- **Bright red** (Δ < 0): Degrades performance, investigate for errors

#### Common Patterns

| Pattern | Interpretation |
|---------|----------------|
| **Monotonic increase with plateau** | All bits useful; calibration converges properly |
| **Early plateau at bit k** | LSBs beyond k hit noise floor; can reduce calibration cost |
| **Peak then drop** | Later bits corrupted/noisy; check for rank deficiency |
| **Large jumps at specific bits** | Reveals critical bits and ADC weight hierarchy |

### Practical Applications

| Application | Insight |
|-------------|---------|
| **Calibration Optimization** | Black annotations (Δ ≥ 1.0) identify critical bits; stop at plateau to reduce computation |
| **Bit Quality Assessment** | Bright red annotations flag problematic bits with minimal/negative contribution |
| **Troubleshooting** | Red annotations quantify degradation; locate rank deficiency issues |
| **Architecture Analysis** | Reveals bit weight hierarchy and identifies redundant/noisy bits |
| **Design Verification** | Binary-weighted ADCs should show decreasing Δ from MSB to LSB |

## Limitations

- **Computational Cost**: O(M) FGCalSine calls can be slow for large M
- **Single-tone Only**: Requires sinewave input; multi-tone not supported
- **Rank Deficiency**: May fail for rank-deficient bit matrices
- **Sequential Only**: Tests bits 1:n, not arbitrary subsets

## See Also

- [`FGCalSine`](FGCalSine.md) — Foreground calibration using sinewave input
- [`specPlot`](specPlot.md) — Spectrum analysis and ENoB computation
- [`overflowChk`](overflowChk.md) — Check for overflow in SAR ADC
- [`INLsine`](INLsine.md) — Compute INL/DNL from sinewave data

## References

1. M. Inerfield, "Effective Number of Bits (ENoB) as a Metric for ADC Performance," IEEE Standards Association.

2. G. Leger and A. Rueda, "Statistical Analysis of SAR ADCs with Digital Calibration," IEEE Transactions on Circuits and Systems, 2018.

## Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| **v1.3** | 2025-01-26 | Red-to-black gradient (Δ: 0→1), y-axis [min-0.5, max+2], font sizes 12-16pt |
| **v1.2** | 2025-01-26 | Absolute color scale (0-1 range), "+" prefix for Δ ≥ 2, annotations above points |
| **v1.1** | 2025-01-26 | Color-coded delta annotations showing per-bit ENoB contribution |
| **v1.0** | 2025-01-26 | Initial release: ENoB sweep, auto frequency detection, error handling |
