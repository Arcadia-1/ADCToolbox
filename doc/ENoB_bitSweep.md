# ENoB_bitSweep

## Overview

`ENoB_bitSweep` is a diagnostic tool that evaluates the Effective Number of Bits (ENoB) as a function of the number of bits used for foreground calibration with FGCalSine. This analysis helps determine the optimal number of bits needed for calibration and identify potential issues with specific bit positions.

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

The `ENoB_bitSweep` function performs the following steps:

1. **Frequency Determination**: If `freq = 0`, runs FGCalSine on all bits to estimate the input frequency.

2. **Bit Sweep Loop**: For each configuration using n bits (n = 1 to M):
   - Extract bits(:, 1:n) subset
   - Run FGCalSine with fixed frequency to calibrate
   - Compute ENoB using specPlot on calibrated signal
   - Store result in ENoB_sweep(n)

3. **Error Handling**: Failed calibrations (e.g., due to rank deficiency) are marked as NaN.

4. **Visualization** (if `plot = 1`):
   - Plots ENoB vs number of bits
   - Annotates each point with incremental ENoB contribution
     - First bit: Shows absolute ENoB value in black (e.g., "5.20")
     - Subsequent bits: Shows delta with "+" prefix (e.g., "+1.50", "+0.35")
   - Color-coded annotations based on **absolute** delta magnitude (fixed 0-1 scale)
     - **Black** (delta ≥ 1.0): Excellent contribution, critical bit
     - **Dark red/gray** (delta ≈ 0.5): Moderate contribution
     - **Bright red** (delta ≤ 0.0): Minimal/negative contribution, problematic bit
     - Red-to-black gradient between these values
   - All annotations positioned above data points
   - Y-axis range: [min - 0.5, max + 1] to accommodate annotations
   - Adds horizontal reference line at maximum ENoB
   - Displays grid for easy reading

## Interpretation

### Understanding the Results

#### Reading the Annotations

Each data point on the plot is annotated with its ENoB contribution and color-coded using an **absolute scale**:

- **Bit 1**: Shows the absolute ENoB in black (e.g., "5.20")
- **Bit 2 onwards**: Shows the incremental change with "+" prefix (e.g., "+2.15", "+0.35")
  - **Color based on absolute delta value (0 to 1 scale):**
    - **Black** (delta ≥ 1.0): Adds ≥1 bit of effective resolution (critical)
    - **Dark red** (delta ≈ 0.5): Adds ~0.5 bits (moderate value)
    - **Bright red** (delta ≤ 0.0): Adds ≤0 bits (minimal/harmful)
    - **Red-to-black gradient** for values between 0 and 1

**Example interpretation:**
```
Bit 1: 5.20 (black)         → Using only MSB gives 5.20 ENoB
Bit 2: +2.15 (black)        → Adds 2.15 bits → Total: 7.35 (excellent, ≥1.0!)
Bit 3: +0.95 (near black)   → Adds 0.95 bits → Total: 8.30 (very good, nearly 1.0)
Bit 4: +0.50 (dark red)     → Adds 0.50 bits → Total: 8.80 (moderate, half a bit)
Bit 5: +0.20 (red)          → Adds 0.20 bits → Total: 9.00 (low value)
Bit 6: +0.05 (bright red)   → Adds 0.05 bits → Total: 9.05 (minimal, noise floor)
Bit 7: -0.10 (bright red)   → Subtracts 0.10 bits → Total: 8.95 (harmful!)
```

**Key insight from absolute color scale:**
- **Black (≥1.0)**: Bit adds at least 1 effective bit → Essential for calibration
- **Dark red (~0.5)**: Bit adds half an effective bit → Worth including
- **Bright red (<0.2)**: Bit adds minimal value → May skip to reduce computation
- **Bright red (negative)**: Bit degrades performance → Investigate for errors

#### Common Patterns

The ENoB sweep plot typically shows one of the following patterns:

#### Pattern 1: Monotonic Increase
```
ENoB increases steadily as more bits are added, plateauing near the end.
→ All bits contribute useful information
→ Calibration converges as expected
```

#### Pattern 2: Early Plateau
```
ENoB plateaus after using only k < M bits.
→ LSBs beyond bit k add minimal information
→ May indicate noise floor or redundant bits
→ Calibration can use fewer bits without loss
```

#### Pattern 3: Drop After Peak
```
ENoB increases, peaks at bit k, then decreases.
→ Bits beyond k may be corrupted or noisy
→ Rank deficiency issues possible
→ Consider investigating bit k+1 and beyond
```

#### Pattern 4: Large Jumps
```
Sudden ENoB increases when adding specific bits.
→ Identifies critical bits for calibration
→ May reveal bit weight hierarchy
→ Useful for understanding ADC architecture
```

### Practical Applications

1. **Calibration Optimization**
   Determine the minimum number of bits needed for effective calibration, reducing computational cost. Black annotations (delta ≥ 1.0) indicate critical bits; bright red (delta < 0.1) suggests diminishing returns.

2. **Bit Quality Assessment**
   Identify problematic bits that degrade performance when included in calibration. Bright red annotations immediately highlight bad bits that add minimal or negative value.

3. **Troubleshooting**
   Diagnose rank deficiency issues by observing where ENoB drops or fails to improve. Red annotations quantify the exact degradation, helping pinpoint problem bits.

4. **Architecture Analysis**
   Understand ADC bit weight structure by observing ENoB contribution per bit. Black annotations indicate significant bits; red annotations suggest redundancy or noise floor.

5. **Design Verification**
   Verify that bit weights follow expected patterns. In a binary-weighted ADC, you'd expect decreasing delta values (MSB contributes most → black, LSB contributes least → red).

## Limitations

- **Computational Cost**: Runs FGCalSine and specPlot M times, which can be slow for large datasets or many bits.

- **Frequency Dependency**: Assumes a single-tone sinewave input. Multi-tone or modulated signals are not supported.

- **Rank Deficiency**: May fail for certain bit configurations if the bit matrix is rank-deficient.

- **Sequential Bits**: Only tests configurations using bits 1:n. Does not test arbitrary bit subsets (e.g., skipping the middle bits).

## See Also

- [`FGCalSine`](FGCalSine.md) — Foreground calibration using sinewave input
- [`specPlot`](specPlot.md) — Spectrum analysis and ENoB computation
- [`overflowChk`](overflowChk.md) — Check for overflow in SAR ADC
- [`INLsine`](INLsine.md) — Compute INL/DNL from sinewave data

## References

1. M. Inerfield, "Effective Number of Bits (ENoB) as a Metric for ADC Performance," IEEE Standards Association.

2. G. Leger and A. Rueda, "Statistical Analysis of SAR ADCs with Digital Calibration," IEEE Transactions on Circuits and Systems, 2018.

## Version History

- **v1.3** (2025-01-26): Fixed y-axis range and improved visibility
  - Y-axis limit: [min - 0.5, max + 1] to prevent annotations from going beyond range
  - Changed color gradient from red-to-green to **red-to-black**
    - Black: delta ≥ 1.0 (excellent)
    - Dark red: delta ≈ 0.5 (moderate)
    - Bright red: delta ≤ 0.0 (poor)
  - Reduced annotation offset to 6% to better fit in plot area
  - **Increased font sizes for better readability**
    - Axes labels, title: FontSize 16
    - Tick labels: FontSize 16
    - Annotations: FontSize 12
  - Figure size: 800×800 pixels for better quality

- **v1.2** (2025-01-26): Improved annotation with absolute color scale
  - First bit shows absolute value (no sign) in black
  - Subsequent bits show delta with "+" prefix
  - **Color based on absolute delta (0-1 scale)**, not relative
  - All annotations positioned above data points
  - White semi-transparent backgrounds for readability

- **v1.1** (2025-01-26): Enhanced plotting
  - Added incremental ENoB annotations on plot
  - Shows delta contribution of each bit
  - Color-coded annotations

- **v1.0** (2025-01-26): Initial release
  - Basic ENoB sweep functionality
  - Automatic frequency detection
  - Plotting support
  - Error handling for failed calibrations
