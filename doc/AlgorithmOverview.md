# ADCToolbox Algorithm Overview

**Last Updated:** 2025-12-03

## Introduction

This document describes the working principles and architectural design of the MATLAB algorithms in ADCToolbox. The toolbox has been streamlined for efficiency, maintainability, and consistent output formatting.

---

## Toolset Architecture

### Design Principles

1. **Separation of Concerns**: Tool execution and panel generation are separate functions
2. **Data-Driven Execution**: Tool lists defined as data structures to eliminate repetitive code
3. **Auto-Detection**: Panel functions automatically locate plot files using naming conventions
4. **Consistent Formatting**: All tools use standardized figure properties and output

### File Organization

```
matlab/src/
├── toolset_aout.m          # 9 analog analysis tools
├── toolset_aout_panel.m    # Combine AOUT plots into 3×3 panel
├── toolset_dout.m          # 6 digital analysis tools
├── toolset_dout_panel.m    # Combine DOUT plots into 3×2 panel
└── [individual tool functions]
```

---

## toolset_aout: Analog Output Analysis

**Files:**
- MATLAB: `matlab/src/toolset_aout.m`
- Python: `python/src/adctoolbox/toolset_aout.py`

### Purpose

Executes 9 diagnostic tools on calibrated ADC analog output data (sine wave). Covers time-domain, frequency-domain, and statistical error analysis.

### Algorithm Workflow

```
1. Parse inputs (aout_data, outputDir, optional parameters)
2. Create output directory if needed
3. Pre-compute common parameters:
   - freqCal = findfreq(aout_data)          % Auto-detect input frequency
   - FullScale = max(aout_data) - min(aout_data)
   - err_data = aout_data - sinfit(aout_data)  % Error signal
4. For each tool i = 1:9:
   - Create figure (800×600, visible/hidden)
   - Execute tool function with pre-computed params
   - Format (title, font size 14)
   - Save PNG: <prefix>_<i>_<toolname>.png
   - Close figure
   - Print progress message
5. Return cell array of 9 file paths
```

### Tool Execution Pattern

Each tool follows a standardized pattern:

```matlab
fprintf('[%d/9] toolname');
figure('Position', [100, 100, 800, 600], 'Visible', opts.Visible);
tool_function(params...);
title('Tool Name');
set(gca, 'FontSize', 14);
plot_files{i} = fullfile(outputDir, sprintf('%s_%d_toolname.png', opts.Prefix, i));
saveas(gcf, plot_files{i});
close(gcf);
fprintf(' -> %s\n', plot_files{i});
```

### Tools Executed

| # | Tool | Key Parameters |
|---|------|----------------|
| 1 | `tomdec` | `freqCal`, `10` harmonics, `1` OSR |
| 2 | `plotspec` | `label=1`, `harmonic=5`, `OSR=1`, `window=@hann` |
| 3 | `plotphase` | `harmonic=10`, `mode='FFT'` |
| 4 | `errsin` (code) | `bin=20`, `fin=freqCal`, `xaxis='value'` |
| 5 | `errsin` (phase) | `bin=99`, `fin=freqCal`, `xaxis='phase'` |
| 6 | `errpdf` | `Resolution`, `FullScale` |
| 7 | `errac` | `MaxLag=200`, `Normalize=true` |
| 8 | `plotspec` (error) | `label=0` |
| 9 | `errevspec` | `Fs=1` |

### Key Features

- **Pre-computation**: Expensive calculations (frequency detection, sine fit) done once and reused
- **Memory Management**: Figures closed immediately after save to prevent memory buildup
- **Fail-Fast**: Execution stops on first error (use `try-catch` for fail-safe execution)
- **Standardized Output**: All plots have consistent size and formatting

---

## toolset_dout: Digital Output Analysis

**Files:**
- MATLAB: `matlab/src/toolset_dout.m`
- Python: `python/src/adctoolbox/toolset_dout.py`

### Purpose

Executes 6 diagnostic tools on ADC digital bit outputs (for SAR ADCs and bit-weighted architectures). Performs calibration, weight analysis, and performance evaluation.

### Algorithm Workflow

```
1. Parse inputs (bits, outputDir, optional parameters)
2. Create output directory if needed
3. Extract resolution: nBits = size(bits, 2)
4. Calibrate bit weights:
   [w_cal, ~, ~, ~, ~, f_cal] = wcalsine(bits, 'freq', 0, 'order', Order, 'verbose', 0)
5. Pre-compute digital codes:
   - digitalCodes = bits * (2.^(nBits-1:-1:0))'  % Nominal binary weights
   - digitalCodes_cal = bits * w_cal'             % Calibrated weights
6. Define tool execution table (struct array with 6 entries)
7. For each tool i = 1:6:
   - Create figure (variable size, visible/hidden)
   - Execute tool function handle from struct
   - Format (title, font size 14)
   - Save PNG with exportgraphics (150 DPI)
   - Close figure
   - Print progress message
8. Return cell array of 6 file paths
```

### Data-Driven Tool Execution

Unlike the old repetitive approach, tools are defined as a struct array:

```matlab
tools = {
    struct('idx', 1, 'name', 'Spectrum (Nominal)', 'suffix', '1_spectrum_nominal', ...
        'pos', [100, 100, 800, 600], ...
        'fn', @() plotspec(digitalCodes, 'label', 1, 'harmonic', 5, 'OSR', 1, 'window', @hann));
    struct('idx', 2, 'name', 'Spectrum (Calibrated)', 'suffix', '2_spectrum_calibrated', ...
        'pos', [100, 100, 800, 600], ...
        'fn', @() plotspec(digitalCodes_cal, 'label', 1, 'harmonic', 5, 'OSR', 1, 'window', @hann));
    % ... 4 more tools
};

for i = 1:6
    fprintf('[%d/6] %s', i, tools{i}.name);
    figure('Position', tools{i}.pos, 'Visible', p.Results.Visible);
    tools{i}.fn();  % Execute function handle
    title(tools{i}.name);
    set(gca, 'FontSize', 14);
    plot_files{i} = fullfile(outputDir, sprintf('%s_%s.png', p.Results.Prefix, tools{i}.suffix));
    exportgraphics(gcf, plot_files{i}, 'Resolution', 150);
    close(gcf);
    fprintf(' -> %s\n', plot_files{i});
end
```

**Benefits of Data-Driven Approach:**
- **Code Reduction**: 109 lines → 66 lines (40% reduction)
- **Consistency**: Single point of control for formatting
- **Maintainability**: Easy to add/remove/modify tools
- **Readability**: Tool properties clearly visible in struct definition

### Tools Executed

| # | Tool | Purpose | Figure Size |
|---|------|---------|-------------|
| 1 | `plotspec` (nominal) | Spectrum before calibration | 800×600 |
| 2 | `plotspec` (calibrated) | Spectrum after `wcalsine` calibration | 800×600 |
| 3 | `bitact` | Bit toggle rate analysis | 1000×750 |
| 4 | `ovfchk` | Overflow/redundancy check | 1000×600 |
| 5 | `weightScaling` | Radix visualization | 800×600 |
| 6 | `bitsweep` | ENoB vs number of bits | 800×600 |

---

## Panel Functions

### toolset_aout_panel

**File:** `matlab/src/toolset_aout_panel.m`

Combines 9 individual AOUT plots into a 3×3 panel figure.

#### Algorithm

```
1. Parse inputs (outputDir, optional Prefix, PlotFiles, Visible)
2. If PlotFiles not provided:
   - Auto-construct file paths: <outputDir>/<prefix>_<i>_<name>.png
3. Validate 9 files expected
4. Create 3×3 tiled layout figure (1800×1000)
5. For each plot i = 1:9:
   - nexttile
   - If file exists: imread, imshow, title
   - Else: display "Missing" text in red
6. Add super title "AOUT Toolset Overview"
7. Export panel: PANEL_<PREFIX>.png (300 DPI)
8. Close figure
9. Return status struct (.success, .panel_path, .errors)
```

#### File Naming Convention

Auto-detected files:
```
<prefix>_1_tomdec.png
<prefix>_2_plotspec.png
<prefix>_3_plotphase.png
<prefix>_4_errsin_code.png
<prefix>_5_errsin_phase.png
<prefix>_6_errPDF.png
<prefix>_7_errAutoCorrelation.png
<prefix>_8_errSpectrum.png
<prefix>_9_errEnvelopeSpectrum.png
```

### toolset_dout_panel

**File:** `matlab/src/toolset_dout_panel.m`

Combines 6 individual DOUT plots into a 3×2 panel figure.

#### Algorithm

Same as `toolset_aout_panel` but with 6 plots in 3×2 layout (1200×1000).

#### File Naming Convention

Auto-detected files:
```
<prefix>_1_spectrum_nominal.png
<prefix>_2_spectrum_calibrated.png
<prefix>_3_bitActivity.png
<prefix>_4_overflowChk.png
<prefix>_5_weightScaling.png
<prefix>_6_ENoB_sweep.png
```

### Key Features

1. **Auto-Detection**: Uses prefix to find files - no need to manually specify paths
2. **Flexible**: Can override with explicit `PlotFiles` parameter
3. **Robust**: Handles missing files gracefully (shows placeholder)
4. **Reusable**: Can re-generate panels without re-running tools

---

## Common Utility Functions

### findfreq

**File:** `matlab/src/findfreq.m`

Auto-detects sine wave frequency from time-domain signal using FFT.

**Algorithm:**
1. Compute FFT of input signal
2. Find peak in magnitude spectrum
3. Convert bin index to normalized frequency
4. Return frequency (0 to 0.5, normalized to sampling rate)

### sinfit

**File:** `matlab/src/sinfit.m`

Fits a 4-parameter sine wave to data: `A*sin(2πft + φ) + DC`

**Algorithm:**
1. Use `findfreq` to detect frequency
2. Set up nonlinear least squares problem
3. Optimize amplitude, frequency, phase, DC offset
4. Return fitted sine wave

---

## Naming Conventions

### File Naming

**Individual tool outputs:**
```
<prefix>_<index>_<toolname>.png
```

Examples:
- `aout_1_tomdec.png`
- `dout_3_bitActivity.png`

**Panel outputs:**
```
PANEL_<PREFIX>.png
```

Examples:
- `PANEL_AOUT.png`
- `PANEL_DOUT.png`

### Variable Naming

- `plot_files` — Cell array of PNG file paths
- `outputDir` — Directory for output files
- `freqCal` — Calibrated/detected frequency
- `w_cal` — Calibrated bit weights
- `digitalCodes` — Digital output codes (nominal weights)
- `digitalCodes_cal` — Digital output codes (calibrated weights)

---

## Performance Optimizations

### Memory Management

- Figures closed immediately after save
- Use `'Visible', false` for batch processing (faster)
- Pre-allocate cell arrays for file paths

### Computation Efficiency

- **Pre-computation**: Expensive operations (frequency detection, sine fit, calibration) done once
- **Data-driven loops**: Eliminates repetitive code
- **Minimal disk I/O**: Each plot saved once

### Code Efficiency

**Before (repetitive):**
```matlab
% Tool 1
fprintf('[1/6] Tool1...');
f = figure('Visible', figVis, 'Position', [100, 100, 800, 600]);
tool1_function(params);
save_and_close(f, outputDir, opts.Prefix, '1_tool1', 'Tool 1');

% Tool 2
fprintf('[2/6] Tool2...');
f = figure('Visible', figVis, 'Position', [100, 100, 800, 600]);
tool2_function(params);
save_and_close(f, outputDir, opts.Prefix, '2_tool2', 'Tool 2');

% ... repeated 4 more times
```

**After (data-driven):**
```matlab
tools = {struct('name', 'Tool1', 'suffix', '1_tool1', 'pos', [100,100,800,600], 'fn', @() tool1_function(params)); ...};

for i = 1:6
    fprintf('[%d/6] %s', i, tools{i}.name);
    figure('Position', tools{i}.pos, 'Visible', figVis);
    tools{i}.fn();
    % ... standardized save
end
```

**Result:** 109 lines → 66 lines (40% reduction)

---

## Usage Patterns

### Pattern 1: Generate plots and panel together

```matlab
% AOUT
plot_files = toolset_aout(aout_data, 'output/test1');
panel_status = toolset_aout_panel('output/test1', 'Prefix', 'aout');

% DOUT
plot_files = toolset_dout(bits, 'output/test1');
panel_status = toolset_dout_panel('output/test1', 'Prefix', 'dout');
```

### Pattern 2: Batch generate plots, then gather panels

```matlab
% Generate all plots first
for i = 1:N
    toolset_aout(data{i}, sprintf('output/run%d', i), 'Prefix', sprintf('run%d', i));
end

% Then generate all panels
for i = 1:N
    toolset_aout_panel(sprintf('output/run%d', i), 'Prefix', sprintf('run%d', i));
end
```

### Pattern 3: Re-generate panels only (plots already exist)

```matlab
% No need to re-run toolset - just regenerate panels
toolset_aout_panel('output/test1', 'Prefix', 'aout');
toolset_dout_panel('output/test2', 'Prefix', 'dout');
```

---

## Error Handling

### Current Behavior

Both toolsets use **fail-fast** error handling:
- Execution stops on first error
- MATLAB displays error message
- Partial results may be saved

### Implementing Fail-Safe Execution

To continue execution after errors, wrap tools in `try-catch`:

```matlab
for i = 1:length(tools)
    try
        % ... tool execution
        success(i) = true;
    catch ME
        fprintf(' ✗ %s\n', ME.message);
        errors{end+1} = sprintf('Tool %d: %s', i, ME.message);
        success(i) = false;
    end
end
```

---

## Future Enhancements

### Planned Features

1. **Python panel functions**: Implement `toolset_aout_panel` and `toolset_dout_panel` in Python
2. **Parallel execution**: Run independent tools in parallel using `parfor`
3. **Progress callbacks**: Optional callback function for progress updates
4. **Custom tool selection**: Allow users to specify which tools to run
5. **Fail-safe mode**: Option to continue execution after errors

### Architectural Improvements

1. **Tool registry**: Central registry of all available tools
2. **Plugin system**: Easy addition of custom tools
3. **Configuration files**: YAML/JSON config for tool parameters
4. **Result caching**: Cache expensive computations for reuse

---

## References

- **IEEE Std 1241-2010**: Standard for ADC Test Methods
- **IEEE Std 1057-2017**: Standard for Digitizing Waveform Recorders
- MATLAB documentation: Function handles, struct arrays, exportgraphics

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| **2.0** | 2025-12-03 | Streamlined toolsets with data-driven execution, separated panel functions |
| **1.0** | 2025-01-28 | Initial implementation |
