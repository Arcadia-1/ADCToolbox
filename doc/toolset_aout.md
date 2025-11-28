# toolset_aout

## Overview

`toolset_aout` is a comprehensive batch runner that executes 9 analog analysis tools on calibrated ADC output data. It automatically generates a complete diagnostic report with visualizations for time-domain, frequency-domain, and statistical error analysis.

This toolset is designed for rapid ADC characterization, producing a 3×3 panel of diagnostic plots in a single function call.

## Syntax

```matlab
status = toolset_aout(aout_data, outputDir)
status = toolset_aout(aout_data, outputDir, 'Visible', true)
status = toolset_aout(aout_data, outputDir, 'Resolution', 12, 'Prefix', 'test1')
```

```python
# Python equivalent: python/src/adctoolbox/toolset_aout.py
from adctoolbox import toolset_aout
status = toolset_aout(aout_data, output_dir, visible=False, resolution=11, prefix='aout')
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

**`status`** — Struct with execution results:
- **`.success`** — `true` if all 9 tools completed successfully
- **`.tools_completed`** — `1×9` array of success flags (1 = success, 0 = failed)
- **`.errors`** — Cell array of error messages (empty if all succeeded)
- **`.panel_path`** — Path to summary panel figure (`PANEL_<PREFIX>.png`)

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

### Summary Panel

A final 3×3 panel figure (`PANEL_<PREFIX>.png`) combines all 9 plots into a single overview image.

## Algorithm

### Workflow

```
1. Validate input data using validateAoutData()
2. Handle multirun data (extract first row if 2D)
3. Auto-detect frequency using findFin()
4. Calculate full-scale range: FullScale = max - min
5. For each tool [1-9]:
     a. Create figure (visible or hidden)
     b. Run analysis tool
     c. Save PNG to outputDir
     d. Log success/failure
6. Generate error signal for tools 4-9:
     err_data = aout_data - sineFit(aout_data)
7. Create summary panel from 9 individual PNGs
8. Return status struct
```

### Data Validation

Before processing, `validateAoutData()` checks:
- Input is numeric
- No NaN or Inf values
- Sufficient data length (> 100 samples recommended)

### Error Handling

If a tool fails:
- Error is logged to `status.errors{}`
- Execution continues with remaining tools
- Panel shows "Missing" placeholder for failed tools

## Examples

### Example 1: Basic Usage

```matlab
% Load calibrated ADC output
aout_data = readmatrix('sinewave_calibrated.csv');

% Run all 9 analysis tools
status = toolset_aout(aout_data, 'output/analysis1');

% Check results
if status.success
    fprintf('✓ All 9 tools completed successfully\n');
    fprintf('Panel saved to: %s\n', status.panel_path);
else
    fprintf('✗ %d/%d tools failed\n', 9 - sum(status.tools_completed), 9);
    disp(status.errors);
end
```

**Output:**
```
[Validation] ✓
[1/9][tomDecomp] ✓ → [output/analysis1/aout_1_tomDecomp.png]
[2/9][specPlot] ✓ → [output/analysis1/aout_2_specPlot.png]
...
[9/9][errEnvelopeSpectrum] ✓ → [output/analysis1/aout_9_errEnvelopeSpectrum.png]
[Panel] ✓ → [output/analysis1/PANEL_AOUT.png]
=== Toolset complete: 9/9 tools succeeded ===

✓ All 9 tools completed successfully
Panel saved to: output/analysis1/PANEL_AOUT.png
```

### Example 2: Interactive Mode with Custom Parameters

```matlab
% Load data
aout_data = readmatrix('adc_12bit_output.csv');

% Run with visible figures and custom resolution
status = toolset_aout(aout_data, 'output/test2', ...
    'Visible', true, ...
    'Resolution', 12, ...
    'Prefix', 'adc12b');

% Figures will display during execution
% Output files: adc12b_1_tomDecomp.png, ..., PANEL_ADC12B.png
```

### Example 3: Batch Processing Multiple Datasets

```matlab
% Process multiple datasets in batch mode
datasets = {'run1.csv', 'run2.csv', 'run3.csv'};
results = cell(1, length(datasets));

for i = 1:length(datasets)
    fprintf('Processing %s...\n', datasets{i});

    aout_data = readmatrix(datasets{i});
    outputDir = sprintf('output/run%d', i);

    results{i} = toolset_aout(aout_data, outputDir, ...
        'Visible', false, ...
        'Prefix', sprintf('run%d', i));

    if results{i}.success
        fprintf('  ✓ Complete\n');
    else
        fprintf('  ✗ Failed: %s\n', strjoin(results{i}.errors, ', '));
    end
end

% Summary
success_count = sum(cellfun(@(x) x.success, results));
fprintf('\n%d/%d datasets processed successfully\n', success_count, length(datasets));
```

### Example 4: Error Recovery

```matlab
aout_data = readmatrix('noisy_data.csv');
status = toolset_aout(aout_data, 'output/noisy', 'Visible', false);

% Check which tools failed
if ~status.success
    fprintf('Failed tools:\n');
    tool_names = {'tomDecomp', 'specPlot', 'specPlotPhase', ...
                  'errHistSine(code)', 'errHistSine(phase)', ...
                  'errPDF', 'errAutoCorrelation', 'errSpectrum', ...
                  'errEnvelopeSpectrum'};

    for i = 1:9
        if status.tools_completed(i) == 0
            fprintf('  [%d] %s: %s\n', i, tool_names{i}, ...
                    status.errors{find(contains(status.errors, sprintf('Tool %d', i)), 1)});
        end
    end

    % Partial results are still saved
    fprintf('\nPartial panel saved to: %s\n', status.panel_path);
end
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

## See Also

- [`toolset_dout`](toolset_dout.md) — Digital output analysis suite (6 tools)
- [`tomDecomp`](tomDecomp.md) — Time-domain error decomposition
- [`specPlot`](specPlot.md) — Frequency spectrum analysis
- [`errHistSine`](errHistSine.md) — Error histogram analysis
- [`errPDF`](errPDF.md) — Error PDF analysis
- [`validateAoutData`] — Input data validation

## References

1. **IEEE Std 1241-2010** — Standard for Terminology and Test Methods for Analog-to-Digital Converters
2. **IEEE Std 1057-2017** — Standard for Digitizing Waveform Recorders
3. Kester, W., "ADC Testing Techniques," *Analog Devices Application Note AN-1210*

---

## Version History

| Version | Date | Key Features |
|---------|------|--------------|
| **v1.0** | 2025-01-28 | Initial documentation for toolset_aout |
