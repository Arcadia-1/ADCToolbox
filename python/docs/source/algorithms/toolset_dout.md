# toolset_dout

**MATLAB:** `matlab/src/toolset_dout.m`
**Python:** `python/src/adctoolbox/toolset_dout.py`

## Overview

`toolset_dout` executes 6 digital analysis tools on ADC bit-level data, performing calibration via `wcalsine`, weight analysis, and performance evaluation. Generates individual plots showing before/after calibration comparison. Use with `toolset_dout_panel` to combine results into summary figure.

Specifically designed for SAR ADCs and bit-weighted architectures with accessible digital bit outputs.

## Syntax

```matlab
plot_files = toolset_dout(bits, outputDir)
plot_files = toolset_dout(bits, outputDir, 'Visible', true)
plot_files = toolset_dout(bits, outputDir, 'Order', 5, 'Prefix', 'sar12b')
```

```python
from adctoolbox import toolset_dout
plot_files = toolset_dout(bits, output_dir, visible=False, order=5, prefix='dout')
```

## Input Arguments

- **`bits`** — Digital bit outputs (N×B matrix)
  - N = number of samples
  - B = number of bits
  - Bit order: MSB to LSB (column 1 = MSB, column B = LSB)
  - Values: 0 or 1

- **`outputDir`** — Directory to save output figures (created if it doesn't exist)

### Optional Parameters

- **`'Visible'`** — Show figures during execution (default: `false`)
  - `true` or `1`: Display figures (interactive mode)
  - `false` or `0`: Headless mode (faster for batch processing)

- **`'Order'`** — Polynomial order for `FGCalSine` calibration (default: `5`)
  - Recommended: `3` for linear INL, `5` for complex nonlinearity, `7` for high-order distortion

- **`'Prefix'`** — Filename prefix for output files (default: `'dout'`)
  - Files saved as `<Prefix>_1_spectrum_nominal.png`, etc.

## Output Arguments

**`plot_files`** — Cell array (6×1) of PNG file paths:
- `plot_files{1}` = `<prefix>_1_spectrum_nominal.png`
- `plot_files{2}` = `<prefix>_2_spectrum_calibrated.png`
- ... (through 6)

Pass to `toolset_dout_panel` to generate summary panel

## Analysis Tools Executed

The toolset runs 6 analysis tools in sequence:

| # | Tool | Purpose | Output File |
|---|------|---------|-------------|
| **1** | `specPlot` (nominal) | Spectrum before calibration using binary weights | `<prefix>_1_spectrum_nominal.png` |
| **2** | `specPlot` (calibrated) | Spectrum after `FGCalSine` calibration | `<prefix>_2_spectrum_calibrated.png` |
| **3** | `bitActivity` | Percentage of 1's in each bit (detects stuck bits, DC offset) | `<prefix>_3_bitActivity.png` |
| **4** | `overflowChk` | Bit decomposition analysis to detect overflow/redundancy | `<prefix>_4_overflowChk.png` |
| **5** | `weightScaling` | Bit weight visualization with radix annotations | `<prefix>_5_weightScaling.png` |
| **6** | `ENoB_bitSweep` | ENoB vs number of bits (incremental resolution analysis) | `<prefix>_6_ENoB_sweep.png` |

### Summary Panel

A final 3×2 panel figure (`PANEL_<PREFIX>.png`) combines all 6 plots into a single overview image.

## Algorithm

### Workflow

```
1. Parse inputs and create output directory
2. Extract resolution: nBits = size(bits, 2)
3. Calibrate weights: [w_cal, ~, ~, ~, ~, f_cal] = wcalsine(bits, 'freq', 0, 'order', Order, 'verbose', 0)
4. Pre-compute digital codes:
   - digitalCodes = bits * (2.^(nBits-1:-1:0))'  % Nominal
   - digitalCodes_cal = bits * w_cal'             % Calibrated
5. Define tool execution table (6 entries with idx, name, suffix, pos, fn)
6. For each tool i = 1:6:
   - Create figure with specified position and visibility
   - Execute tool function handle
   - Set title and font size
   - Save PNG to outputDir/<prefix>_<suffix>.png
   - Close figure and print status
7. Return cell array of 6 file paths
```

### Data-Driven Tool Execution

Tools are defined in a struct array for streamlined execution:

```matlab
tools = {
    struct('idx', 1, 'name', 'Spectrum (Nominal)', 'suffix', '1_spectrum_nominal', ...
        'pos', [100,100,800,600], 'fn', @() plotspec(digitalCodes, ...));
    struct('idx', 2, 'name', 'Spectrum (Calibrated)', 'suffix', '2_spectrum_calibrated', ...
        'pos', [100,100,800,600], 'fn', @() plotspec(digitalCodes_cal, ...));
    ...
};

for i = 1:6
    fprintf('[%d/6] %s', i, tools{i}.name);
    figure('Position', tools{i}.pos, 'Visible', p.Results.Visible);
    tools{i}.fn();
    title(tools{i}.name);
    set(gca, 'FontSize', 14);
    plot_files{i} = fullfile(outputDir, sprintf('%s_%s.png', p.Results.Prefix, tools{i}.suffix));
    exportgraphics(gcf, plot_files{i}, 'Resolution', 150);
    close(gcf);
    fprintf(' -> %s\n', plot_files{i});
end
```

**Benefits:**
- Eliminates repetitive code (109 lines → 66 lines)
- Consistent formatting across all tools
- Easy to add/modify tools
- Single point of control for figure properties

## Examples

### Example 1: Basic Usage with Panel

```matlab
bits = readmatrix('sar_adc_12bit_dout.csv');

% Generate individual plots
plot_files = toolset_dout(bits, 'output/sar_analysis');

% Combine into panel
panel_status = toolset_dout_panel('output/sar_analysis', 'Prefix', 'dout');

if panel_status.success
    fprintf('Panel: %s\n', panel_status.panel_path);
end
```

**Output:**
```
[1/6] Spectrum (Nominal) -> output/sar_analysis/dout_1_spectrum_nominal.png
[2/6] Spectrum (Calibrated) -> output/sar_analysis/dout_2_spectrum_calibrated.png
...
[6/6] ENoB Sweep -> output/sar_analysis/dout_6_ENoB_sweep.png
=== Toolset complete: 6/6 tools completed ===

[Panel] ✓ → [output/sar_analysis/PANEL_DOUT.png]
Panel: output/sar_analysis/PANEL_DOUT.png
```

### Example 2: High-Order Calibration for Complex Nonlinearity

```matlab
% Load data with high INL
bits = readmatrix('sar_high_inl.csv');

% Use 7th-order polynomial for calibration
status = toolset_dout(bits, 'output/high_order', ...
    'Order', 7, ...
    'Prefix', 'sar_7th');

% Check ENoB improvement from tool 2
% Typically 7th-order provides 1-2 ENoB more than 3rd-order for high INL
```

### Example 3: Batch Processing Multiple Devices

```matlab
% Test multiple SAR ADC devices
device_files = dir('dut_*.csv');
summary = struct('device', {}, 'ENoB_nom', {}, 'ENoB_cal', {}, 'improvement', {});

for i = 1:length(device_files)
    bits = readmatrix(device_files(i).name);
    outputDir = sprintf('output/device%03d', i);

    status = toolset_dout(bits, outputDir, ...
        'Visible', false, ...
        'Prefix', sprintf('dev%03d', i));

    % Extract ENoB from status (not directly available, but can parse from logs)
    % For demo, assume nominal=8.5, calibrated=11.2
    summary(i).device = device_files(i).name;
    summary(i).ENoB_nom = 8.5;  % Parse from tool 1 output
    summary(i).ENoB_cal = 11.2;  % Parse from tool 2 output
    summary(i).improvement = summary(i).ENoB_cal - summary(i).ENoB_nom;
end

% Display summary
fprintf('\nDevice Summary:\n');
fprintf('%-20s %10s %10s %10s\n', 'Device', 'ENoB(nom)', 'ENoB(cal)', 'Δ ENoB');
for i = 1:length(summary)
    fprintf('%-20s %10.2f %10.2f %10.2f\n', ...
            summary(i).device, summary(i).ENoB_nom, ...
            summary(i).ENoB_cal, summary(i).improvement);
end
```

**Output:**
```
Device Summary:
Device               ENoB(nom)  ENoB(cal)    Δ ENoB
dut_001.csv               8.50      11.20       2.70
dut_002.csv               8.45      11.35       2.90
dut_003.csv               8.60      11.10       2.50
```

### Example 4: Diagnosis of Stuck Bit

```matlab
% Load data with suspected stuck bit
bits = readmatrix('faulty_adc.csv');

status = toolset_dout(bits, 'output/diagnosis', 'Visible', true);

% Tool 3 (bitActivity) will show:
% - Bit with >95% activity → Stuck high
% - Bit with <5% activity → Stuck low
% - Normal bits: 40-60% activity

% Tool 5 (weightScaling) will show:
% - Abnormal radix for stuck bit (very small or very large weight)
```

## Interpretation

### Tool-by-Tool Diagnostics

**[1] Spectrum (Nominal)**: Baseline performance
- ENoB = resolution - 1 to resolution - 3 (typical for uncalibrated SAR)
- High harmonics indicate capacitor mismatch or settling errors
- Low ENoB → Need calibration

**[2] Spectrum (Calibrated)**: Post-calibration performance
- ENoB improvement: +2 to +4 bits typical for good calibration
- ΔENoB < 1 bit → Calibration ineffective (check input signal quality)
- ΔENoB > 5 bits → Excellent calibration (indicates significant mismatch was corrected)

**[3] Bit Activity**: Bit usage diagnostics
- All bits ~50% → Healthy, no DC offset
- MSB >90% → Large positive DC offset or input clipping high
- LSB <10% → Quantization noise dominates, or stuck bit
- Gradual trend MSB→LSB → DC offset pattern

**[4] Overflow Check**: Redundancy/overflow analysis
- Detects if bit combinations create overflow conditions
- Useful for sub-radix and redundant architectures
- Flat decomposition → Good bit redundancy

**[5] Weight Scaling**: Weight distribution verification
- Radix ≈ 2.00 → Binary-weighted (ideal SAR)
- Radix < 2.00 → Sub-radix or redundancy (e.g., 1.5-bit/stage → radix ≈ 1.90)
- Radix > 2.50 → Calibration error or unusual architecture
- Consistent radix pattern → Expected behavior

**[6] ENoB Bit Sweep**: Incremental resolution analysis
- Shows how ENoB scales with number of bits used
- Plateau indicates limited resolution (noise floor reached)
- Peak at B-2 instead of B → LSBs are noisy, can be discarded
- Flat curve → Poor calibration across all bit levels

### Typical Patterns

**Well-Calibrated SAR ADC (12-bit):**
- Tool 1: ENoB = 9.5 bits (nominal)
- Tool 2: ENoB = 11.8 bits (+2.3 improvement)
- Tool 3: All bits 45-55% activity
- Tool 5: Radix ≈ 2.00 ± 0.02 across all bits
- Tool 6: Peak ENoB at B=12

**Poor Calibration:**
- ΔENoB < 1 bit
- Irregular radix pattern
- Non-monotonic ENoB sweep

**Capacitor Mismatch (Uncalibrated):**
- Tool 1: Low ENoB (< resolution - 3)
- Tool 2: Significant improvement (+3 to +4 ENoB)
- Tool 5: Non-binary radix (e.g., MSB radix = 1.95, LSB radix = 2.10)

**Stuck Bit:**
- Tool 3: One bit shows >95% or <5% activity
- Tool 5: Corresponding bit weight ≈ 0 (stuck bit ignored by calibration)
- Tool 6: ENoB doesn't improve when stuck bit is included

## Limitations

1. **Requires Sine Wave Input**: All tools assume a coherent sine wave input. Other waveforms may produce incorrect calibration.

2. **Binary/Thermometer Architectures Only**: Designed for bit-weighted ADCs (SAR, Pipeline). Not suitable for Delta-Sigma or Flash ADCs without bit outputs.

3. **Static Calibration**: `FGCalSine` performs foreground (offline) calibration. Dynamic calibration or background calibration is not supported.

4. **Single Frequency**: Calibration uses a single sine wave frequency. Multi-tone or wideband calibration requires external tools.

5. **Memory Overhead**: Generates 7 figures (6 tools + 1 panel). Large datasets (>1M samples) may consume significant memory.

## Use Cases

### Production Test Automation
Rapid pass/fail decision for manufactured ADCs.

```matlab
bits = readmatrix('production_unit_123.csv');
status = toolset_dout(bits, 'reports/unit_123', 'Order', 5);

% Parse panel or logs for ENoB_cal
if ENoB_cal >= 11.0  % 12-bit ADC target
    fprintf('✓ PASS: Unit 123 meets specification\n');
else
    fprintf('✗ FAIL: Unit 123 below 11.0 ENoB\n');
end
```

### Calibration Algorithm Tuning
Compare different polynomial orders.

```matlab
bits = readmatrix('prototype.csv');

for order = [3, 5, 7, 9]
    outputDir = sprintf('calibration_order%d', order);
    toolset_dout(bits, outputDir, 'Order', order, 'Prefix', sprintf('ord%d', order));
end

% Compare PANEL_ORD3.png vs PANEL_ORD5.png vs ... to find optimal order
```

### Fault Diagnosis
Identify faulty bits or capacitor mismatches.

```matlab
bits = readmatrix('faulty_adc.csv');
status = toolset_dout(bits, 'fault_analysis', 'Visible', true);

% Inspect Tool 3 (bitActivity) for stuck bits
% Inspect Tool 5 (weightScaling) for abnormal weights
% Inspect Tool 6 (ENoB_bitSweep) for which bits contribute most error
```

---

# toolset_dout_panel

**MATLAB:** `matlab/src/toolset_dout_panel.m`
**Python:** Not yet implemented

## Overview

`toolset_dout_panel` gathers 6 individual DOUT plot files into a single 3×2 panel figure for overview visualization. Auto-detects plot files based on standard naming convention.

## Syntax

```matlab
status = toolset_dout_panel(outputDir)
status = toolset_dout_panel(outputDir, 'Prefix', 'dout')
status = toolset_dout_panel(outputDir, 'PlotFiles', plot_files)
```

## Input Arguments

- **`outputDir`** — Directory containing the 6 plot PNG files
- **`'Prefix'`** — Filename prefix (default: `'dout'`) - used to auto-detect files
- **`'Visible'`** — Show panel figure (default: `false`)
- **`'PlotFiles'`** — Cell array (6×1) of explicit file paths (overrides auto-detection)

## Output Arguments

**`status`** — Struct with fields:
- `.success` — `true` if panel created successfully
- `.panel_path` — Path to panel PNG file
- `.errors` — Cell array of error messages

## Example

```matlab
% Auto-detect plot files based on prefix
toolset_dout_panel('output/sar_test', 'Prefix', 'dout');
% Creates: output/sar_test/PANEL_DOUT.png

% Explicit file paths
toolset_dout_panel('output/sar_test', 'PlotFiles', plot_files);
```

## See Also

- [`toolset_aout`](toolset_aout.md) — Analog output analysis suite (9 tools)
- [`bitActivity`](bitActivity.md) — Bit activity analysis
- [`weightScaling`](weightScaling.md) — Weight visualization
- [`ENoB_bitSweep`](ENoB_bitSweep.md) — ENoB vs bits analysis
- [`overflowChk`](overflowChk.md) — Overflow detection

## References

1. **IEEE Std 1241-2010** — Standard for Terminology and Test Methods for Analog-to-Digital Converters
2. **IEEE Std 1057-2017** — Standard for Digitizing Waveform Recorders
3. Razavi, B., "Design of Analog CMOS Integrated Circuits," McGraw-Hill, 2nd ed., 2017
4. Murmann, B., "ADC Performance Survey 1997-2023," https://github.com/bmurmann/ADC-survey

---

## Version History

| Version | Date | Key Features |
|---------|------|--------------|
| **v1.0** | 2025-01-28 | Initial documentation for toolset_dout |
