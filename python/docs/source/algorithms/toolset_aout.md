# toolset_aout

**MATLAB:** `matlab/src/toolset_aout.m`
**Python:** `python/src/adctoolbox/toolset_aout.py`

## Overview

`toolset_aout` executes 9 analog analysis tools on calibrated ADC output data, generating individual diagnostic plots for time-domain, frequency-domain, and statistical error analysis. Use with `toolset_aout_panel` to combine results into a summary figure.

Designed for rapid ADC characterization with data-driven execution and consistent output formatting.

## Syntax

```matlab
plot_files = toolset_aout(aout_data, outputDir)
plot_files = toolset_aout(aout_data, outputDir, 'Visible', true)
plot_files = toolset_aout(aout_data, outputDir, 'Resolution', 12, 'Prefix', 'test1')
```

```python
from adctoolbox import toolset_aout
plot_files = toolset_aout(aout_data, output_dir, visible=False, resolution=11, prefix='aout')
```

## Input Arguments

- **`aout_data`** — Analog output signal (1D vector or N×M matrix for multirun)
  - If 2D, only the first row is used
- **`outputDir`** — Directory to save output figures (created if it doesn't exist)

### Optional Parameters

- **`'Visible'`** — Show figures during execution (default: `false`)
  - `true` or `1`: Display figures (useful for interactive debugging)
  - `false` or `0`: Headless mode (faster for batch processing)
- **`'Resolution'`** — ADC resolution in bits (default: `11`)
  - Used by `errPDF` for quantization noise normalization
- **`'Prefix'`** — Filename prefix for output files (default: `'aout'`)
  - Files saved as `<Prefix>_1_tomDecomp.png`, etc.

## Output Arguments

**`plot_files`** — Cell array (9×1) of PNG file paths:
- `plot_files{1}` = `<prefix>_1_tomdec.png`
- `plot_files{2}` = `<prefix>_2_plotspec.png`
- ... (through 9)

Pass to `toolset_aout_panel` to generate summary panel

## Analysis Tools Executed

The toolset runs 9 analysis tools in sequence:

| # | Tool | Purpose | Output File |
|---|------|---------|-------------|
| **1** | `tomDecomp` | Time-domain error decomposition into signal, independent error, and dependent error | `<prefix>_1_tomDecomp.png` |
| **2** | `specPlot` | Frequency spectrum with ENoB, SNDR, SFDR, SNR, THD | `<prefix>_2_specPlot.png` |
| **3** | `specPlotPhase` | Phase-domain error analysis showing harmonic phase shifts | `<prefix>_3_specPlotPhase.png` |
| **4** | `errHistSine` (code) | Error histogram binned by ADC code level | `<prefix>_4_errHistSine_code.png` |
| **5** | `errHistSine` (phase) | Error histogram binned by sine wave phase | `<prefix>_5_errHistSine_phase.png` |
| **6** | `errPDF` | Error probability density function with Gaussian fit | `<prefix>_6_errPDF.png` |
| **7** | `errAutoCorrelation` | Error autocorrelation to detect periodic patterns | `<prefix>_7_errAutoCorrelation.png` |
| **8** | `errSpectrum` | Error signal spectrum (after removing fitted sine) | `<prefix>_8_errSpectrum.png` |
| **9** | `errEnvelopeSpectrum` | Envelope spectrum using Hilbert transform | `<prefix>_9_errEnvelopeSpectrum.png` |

## Algorithm

### Workflow

```
1. Parse inputs and create output directory
2. Pre-compute common parameters:
   - freqCal = findfreq(aout_data)
   - FullScale = max(aout_data) - min(aout_data)
   - err_data = aout_data - sinfit(aout_data)
3. For each tool i = 1:9:
   - Create figure with specified visibility
   - Execute tool function
   - Set title and font size
   - Export PNG to outputDir/<prefix>_<i>_<name>.png
   - Close figure
   - Print status message
4. Return cell array of 9 file paths
```

### Tool Execution Structure

Tools are executed sequentially using a standardized pattern:
- Create figure with `Position` and `Visible` parameters
- Call tool function with pre-computed parameters
- Standardize formatting (`title`, `set(gca, 'FontSize', 14)`)
- Save with `saveas(gcf, plot_files{i})`
- Close figure to free memory

### Error Handling

Current implementation uses MATLAB's default error handling - execution stops on first error. To implement fail-safe execution, wrap each tool in `try-catch`

## Examples

### Example 1: Basic Usage with Panel

```matlab
aout_data = readmatrix('sinewave_calibrated.csv');

% Generate individual plots
plot_files = toolset_aout(aout_data, 'output/analysis1');

% Combine into panel
panel_status = toolset_aout_panel('output/analysis1', 'Prefix', 'aout');

if panel_status.success
    fprintf('Panel: %s\n', panel_status.panel_path);
end
```

**Output:**
```
[1/9] tomdec -> output/analysis1/aout_1_tomdec.png
[2/9] plotspec -> output/analysis1/aout_2_plotspec.png
...
[9/9] errEnvelopeSpectrum -> output/analysis1/aout_9_errEnvelopeSpectrum.png
=== Toolset complete: 9/9 tools completed ===

[Panel] ✓ → [output/analysis1/PANEL_AOUT.png]
Panel: output/analysis1/PANEL_AOUT.png
```

### Example 2: Batch Processing

```matlab
datasets = {'run1.csv', 'run2.csv', 'run3.csv'};

for i = 1:length(datasets)
    aout_data = readmatrix(datasets{i});
    outputDir = sprintf('output/run%d', i);

    plot_files = toolset_aout(aout_data, outputDir, 'Visible', false, 'Prefix', sprintf('run%d', i));
    toolset_aout_panel(outputDir, 'Prefix', sprintf('run%d', i));
end
```

### Example 3: Re-generate Panels Only

```matlab
% If plots already exist, just re-create panels
toolset_aout_panel('output/analysis1', 'Prefix', 'aout');
toolset_aout_panel('output/analysis2', 'Prefix', 'aout');
```

## Interpretation

### Tool-by-Tool Diagnostics

**[1] tomDecomp**: Identifies sources of error
- Independent error: Random noise, thermal noise
- Dependent error: Nonlinearity, harmonic distortion
- High dependent error → Check calibration, linearity

**[2] specPlot**: Overall performance metrics
- ENoB: Effective resolution
- SNDR: Total error (noise + distortion)
- SFDR: Largest spur
- THD: Harmonic distortion

**[3] specPlotPhase**: Nonlinearity characterization
- Phase shifts in harmonics indicate even/odd nonlinearity
- Useful for identifying specific distortion mechanisms

**[4-5] errHistSine**: Error distribution patterns
- Code-based: DNL, missing codes, stuck bits
- Phase-based: Sine-wave specific artifacts
- Non-uniform histogram → Linearity issues

**[6] errPDF**: Noise distribution analysis
- KL divergence: Deviation from Gaussian
- High KL → Non-Gaussian noise (spurs, interference)
- sigma: RMS noise level

**[7] errAutoCorrelation**: Temporal error patterns
- Peaks at non-zero lag → Periodic interference
- Exponential decay → Memory effects
- Random (delta function) → White noise

**[8] errSpectrum**: Error spectral content
- Spurs: Switching noise, clock feedthrough
- Baseband noise: Thermal, quantization
- Broadband noise: Jitter, random sources

**[9] errEnvelopeSpectrum**: Modulation analysis
- Detects amplitude modulation in error signal
- Cyclostationary noise patterns
- Switching artifacts

### Typical Patterns

**Well-Calibrated ADC:**
- All 9 tools complete ✓
- ENoB close to resolution - 1
- SNDR > 60 dB for 10-bit
- White noise in autocorrelation
- Gaussian error PDF

**Poor Calibration:**
- High dependent error in tomDecomp
- Low ENoB (< resolution - 3)
- Non-Gaussian PDF
- Structured errors in errHistSine

**Environmental Interference:**
- Spurs in specPlot, errSpectrum
- Periodic autocorrelation
- High-amplitude envelope spectrum
- Non-Gaussian PDF with long tails

## Limitations

1. **Requires Sine Wave Input**: All tools assume a calibrated sine wave input. Other waveforms are not supported.

2. **Single-Channel Only**: Multirun data uses only the first row. For averaging across runs, pre-process data externally.

3. **Fixed Tool Sequence**: Tools run in a fixed order. Cannot skip individual tools (but failures are tolerated).

4. **Memory Usage**: Generates 10 figures (9 tools + 1 panel). In headless mode, figures are closed immediately.

5. **No Cross-Tool Data Sharing**: Each tool runs independently. Computed values (e.g., frequency) are recalculated per tool.

## Use Cases

### Rapid ADC Characterization
Generate complete diagnostic report in one function call.

```matlab
aout_data = readmatrix('production_test.csv');
toolset_aout(aout_data, 'production_report', 'Resolution', 12);
% Review PANEL_AOUT.png for pass/fail decision
```

### Regression Testing
Compare before/after calibration or design changes.

```matlab
toolset_aout(data_before, 'reports/before', 'Prefix', 'before');
toolset_aout(data_after, 'reports/after', 'Prefix', 'after');
% Compare before vs after panels side-by-side
```

### Debugging Calibration Issues
Identify which error source dominates.

```matlab
status = toolset_aout(data, 'debug', 'Visible', true);
% Inspect tomDecomp (tool 1) for dependent vs independent error
% Check errHistSine (tools 4-5) for code-dependent patterns
```

---

# toolset_aout_panel

**MATLAB:** `matlab/src/toolset_aout_panel.m`
**Python:** Not yet implemented

## Overview

`toolset_aout_panel` gathers 9 individual AOUT plot files into a single 3×3 panel figure for overview visualization. Auto-detects plot files based on standard naming convention.

## Syntax

```matlab
status = toolset_aout_panel(outputDir)
status = toolset_aout_panel(outputDir, 'Prefix', 'aout')
status = toolset_aout_panel(outputDir, 'PlotFiles', plot_files)
```

## Input Arguments

- **`outputDir`** — Directory containing the 9 plot PNG files
- **`'Prefix'`** — Filename prefix (default: `'aout'`) - used to auto-detect files
- **`'Visible'`** — Show panel figure (default: `false`)
- **`'PlotFiles'`** — Cell array (9×1) of explicit file paths (overrides auto-detection)

## Output Arguments

**`status`** — Struct with fields:
- `.success` — `true` if panel created successfully
- `.panel_path` — Path to panel PNG file
- `.errors` — Cell array of error messages

## Example

```matlab
% Auto-detect plot files based on prefix
toolset_aout_panel('output/test1', 'Prefix', 'aout');
% Creates: output/test1/PANEL_AOUT.png

% Explicit file paths
toolset_aout_panel('output/test1', 'PlotFiles', plot_files);
```

## See Also

- [`toolset_dout`](toolset_dout.md) — Digital output analysis suite (6 tools)
- [`tomDecomp`](tomDecomp.md) — Time-domain error decomposition
- [`specPlot`](specPlot.md) — Frequency spectrum analysis
- [`errHistSine`](errHistSine.md) — Error histogram analysis
- [`errPDF`](errPDF.md) — Error PDF analysis

## References

1. **IEEE Std 1241-2010** — Standard for Terminology and Test Methods for Analog-to-Digital Converters
2. **IEEE Std 1057-2017** — Standard for Digitizing Waveform Recorders
3. Kester, W., "ADC Testing Techniques," *Analog Devices Application Note AN-1210*

---

## Version History

| Version | Date | Key Features |
|---------|------|--------------|
| **v1.0** | 2025-01-28 | Initial documentation for toolset_aout |
