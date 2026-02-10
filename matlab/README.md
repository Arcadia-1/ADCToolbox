# ADCToolbox - MATLAB

A comprehensive MATLAB toolbox for ADC (Analog-to-Digital Converter) testing, characterization, and debugging. This toolbox provides advanced functions for spectral analysis, calibration, linearity testing, signal processing, and performance evaluation of ADCs.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Function Categories](#function-categories)
  - [Spectral Analysis](#spectral-analysis)
  - [Signal Fitting and Frequency Analysis](#signal-fitting-and-frequency-analysis)
  - [Calibration Functions](#calibration-functions)
  - [Linearity Analysis](#linearity-analysis)
  - [Noise Transfer Function Analysis](#noise-transfer-function-analysis)
  - [Shortcut Functions](#shortcut-functions)
  - [Utility Functions](#utility-functions)
- [Detailed Function Reference](#detailed-function-reference)
- [Usage Examples](#usage-examples)
- [Legacy Functions](#legacy-functions)
- [Requirements](#requirements)
- [Contributing](#contributing)

## Installation

### Option 1: Install Toolbox Package (Recommended)

1. Navigate to the `toolbox/` directory
2. Double-click `ADCToolbox_1v30.mltbx` to install
3. The toolbox will be automatically added to your MATLAB path
4. You can also download this toolbox from MATLAB Add-Ons

### Option 2: Add to Path Manually

```matlab
% Add the toolbox to your MATLAB path
addpath(genpath('path/to/ADCToolbox/matlab/src'))
savepath  % Optional: save path for future sessions
```

### Option 3: Use Setup Script

```matlab
% Run the setup script (from the matlab/ directory)
run('setupLib.m')
```

## Quick Start

```matlab
% Load ADC output data
load('adc_data.mat');  % Assume this contains variable 'sig'

% Option 1: Use the comprehensive dashboard (recommended for first-time analysis)
rep = adcpanel(sig, 'fs', 100e6);  % One-stop analysis with all metrics

% Option 2: Individual function calls for specific analyses or tools, like:
% Perform comprehensive spectral analysis
[enob, sndr, sfdr, snr, thd] = plotspec(sig, 'Fs', 100e6);

% Find the dominant frequency
freq = findfreq(sig, 100e6);

% Fit a sine wave to the data
[fitout, freq, mag, dc, phi] = sinfit(sig);

% Calculate INL and DNL using histogram method
[inl, dnl, code] = inlsin(sig);

% Analyze phase spectrum
plotphase(sig);
```

## Function Categories

### Dashboard and Comprehensive Analysis

Unified analysis functions that combine multiple testing methods.

- **`adcpanel`** - Comprehensive ADC analysis dashboard with automatic data format handling

### Spectral Analysis

Functions for analyzing the frequency-domain characteristics of ADC output data.

- **`plotspec`** - Comprehensive spectrum analysis with ENOB, SNDR, SFDR, SNR, and THD calculations
- **`plotphase`** - Coherent phase spectrum analysis with polar display

### Signal Fitting and Frequency Analysis

Functions for extracting signal parameters and frequency information.

- **`sinfit`** - Four-parameter iterative sine wave fitting (amplitude, phase, DC, frequency)
- **`findfreq`** - Find dominant frequency using sine wave fitting
- **`findbin`** - Find coherent FFT bin for a given signal frequency
- **`tomdec`** - Thompson decomposition of signal into sinewave, harmonics, and errors

### Calibration Functions

Functions for calibrating ADC bit weights and correcting errors.

- **`wcalsin`** - Weight calibration using sine wave input (single or multi-dataset)
- **`cdacwgt`** - Calculate bit weights for multi-segment capacitive DAC
- **`plotwgt`** - Visualize bit weights with radix annotations, compute optimal scaling and effective resolution
- **`plotres`** - Plot partial-sum residuals of an ADC bit matrix as scatter plots

### Linearity and Error Analysis

Functions for analyzing ADC linearity performance.

- **`inlsin`** - Calculate INL and DNL from sine wave data using histogram method
- **`errsin`** - Analyze sine wave fit errors with histogram binning

### Noise Transfer Function Analysis

Functions for analyzing noise-shaping ADCs (Delta-Sigma modulators).

- **`ntfperf`** - Analyze noise transfer function performance and SNR improvement

### Shortcut Functions

Convenience wrappers that combine multiple steps into a single call.

- **`plotressin`** - Plot partial-sum residuals directly from bit matrix (auto-calibrates via `wcalsin`)
- **`errsinv`** - Shortcut for `errsin` with value-mode binning (`xaxis='value'`)

### Utility Functions

Supporting functions for signal processing and analysis.

- **`alias`** - Calculate aliased frequency after sampling
- **`ifilter`** - Ideal FFT-based filter to retain specified frequency bands
- **`bitchk`** - Check ADC overflow by analyzing bit segment residue distributions

## Detailed Function Reference

### adcpanel

**Purpose:** Comprehensive ADC analysis dashboard that automatically detects input data type and runs appropriate analysis pipelines.

**Syntax:**
```matlab
rep = adcpanel(dat)
rep = adcpanel(dat, 'Name', Value)
```

**Key Features:**
- Unified panel displaying multiple ADC analysis results in a single dashboard
- Automatic data type detection (value-waveform vs. bit-wise data)
- Three analysis pipelines:
  - **Pipeline A**: Value-waveform + sinewave (full characterization)
  - **Pipeline B**: Value-waveform + other signal (basic analysis)
  - **Pipeline C**: Bit-wise data (calibration + full characterization)
- Generates organized tiledlayout figures (3×4 grid for sinewave analysis)
- Returns comprehensive report structure with all metrics and figure handles

**Analysis Pipelines:**

**Pipeline A - Value-Waveform + Sinewave:**
1. `plotspec` - Spectrum analysis (ENOB, SNDR, SFDR, SNR, THD)
2. `tomdec` - Thompson decomposition for time-domain error waveform
3. Time-domain plot - Signal and error waveforms (zoomed to max error region)
4. `errsin` - Sinewave error analysis (both phase and value modes)
5. `inlsin` - INL/DNL calculation
6. `perfosr` - Performance vs OSR sweep
7. `plotphase` - Harmonic phase analysis (both FFT and LMS modes)

**Pipeline B - Value-Waveform + Other Signal:**
1. Time-domain waveform plot
2. `plotspec` - Basic spectrum display

**Pipeline C - Bit-wise Data:**
1. `bitchk` - Overflow/underflow detection
2. `wcalsin` - Weight calibration from sinewave
3. `plotwgt` - Visualize calibrated weights, compute optimal scaling and effective resolution
4. Auto-detect `maxCode` from weight ratios (identifies significant vs noise bits)
5. If calibration successful: run Pipeline A on calibrated values

**Parameters:**
- `'dataType'` - How to interpret input: 'auto' (default), 'values', or 'bits'
- `'signalType'` - Type of input signal: 'sinewave' (default) or 'other'
- `'OSR'` - Oversampling ratio (default: 1)
- `'fs'` - Sampling frequency in Hz (default: 1)
- `'maxCode'` - Full scale range (default: auto-detected; for bit data uses weight ratio analysis to identify significant bits)
- `'harmonic'` - Number of harmonics to analyze (default: 5)
- `'window'` - Window function: 'hann' (default), 'rect', or function handle
- `'fin'` - Normalized input frequency (default: 0 for auto-detect)
- `'disp'` - Enable figure display (default: true)
- `'verbose'` - Enable verbose output (default: false)

**Outputs:**
- `rep` - Report structure containing:
  - `.dataType` - 'values' or 'bits'
  - `.signalType` - 'sinewave' or 'other'
  - `.spectrum` - Spectral metrics (ENOB, SNDR, SFDR, SNR, THD, etc.)
  - `.decomp` - Thompson decomposition (sine, err, har, oth, freq)
  - `.errorPhase` - Error analysis with phase binning (emean, erms, anoi, pnoi)
  - `.errorValue` - Error analysis with value binning (emean, erms)
  - `.linearity` - INL/DNL results (inl, dnl, code)
  - `.osr` - OSR sweep results (osr, sndr, sfdr, enob)
  - `.phaseFFT` - Phase analysis in FFT mode
  - `.phaseLMS` - Phase analysis in LMS mode (with noise circle)
  - `.bits` - Bit-wise analysis (weights, offset, overflow, effres) if bit data
  - `.figures` - Handles to all generated figures and axes

**Examples:**
```matlab
% Basic usage with value-waveform data
sig = sin(2*pi*0.123*(0:4095)') + 0.01*randn(4096,1);
rep = adcpanel(sig);

% Bit-wise data analysis (quantize sig to bits)
bits = dec2bin(round((sig/2.1+0.5)*2^12), 12) - '0';  % Convert to 12-bit representation
rep = adcpanel(bits);

% Oversampled data with specific parameters
rep = adcpanel(sig, 'OSR', 4, 'fs', 100e6, 'harmonic', 7);

% Non-sinewave signal (time-domain + spectrum only)
noise_sig = randn(4096, 1);
rep = adcpanel(noise_sig, 'signalType', 'other');

% Access specific results from report
fprintf('ENOB: %.2f bits\n', rep.spectrum.enob);
fprintf('SNDR: %.2f dB\n', rep.spectrum.sndr);
fprintf('Max INL: %.3f LSB\n', max(abs(rep.linearity.inl)));
```

**Notes:**
- INL/DNL analysis requires integer codes; non-integer data is automatically rounded
- Warning issued when N < maxCode, as INL/DNL may be unreliable
- For bit-wise data, a separate figure shows bitchk and plotwgt results
- Time-domain display zooms to ~3 sine cycles around maximum error point

**See also:** plotspec, tomdec, errsin, inlsin, perfosr, plotphase, bitchk, wcalsin, plotwgt

---

### plotspec

**Purpose:** Comprehensive power spectrum analysis and ADC performance metric calculation.

**Syntax:**
```matlab
[enob, sndr, sfdr, snr, thd, sigpwr, noi, nsd, h] = plotspec(sig)
[enob, sndr, sfdr, snr, thd, sigpwr, noi, nsd, h] = plotspec(sig, Fs, maxCode, harmonic)
[enob, sndr, sfdr, snr, thd, sigpwr, noi, nsd, h] = plotspec(sig, 'Name', Value)
```

**Key Features:**
- Calculates ENOB (Effective Number of Bits)
- Computes SNDR (Signal-to-Noise and Distortion Ratio)
- Measures SFDR (Spurious-Free Dynamic Range)
- Calculates SNR (Signal-to-Noise Ratio) and THD (Total Harmonic Distortion)
- Supports oversampling ratio (OSR) for noise-shaping ADCs
- Multiple averaging modes: normal (power averaging) and coherent (phase-aligned)
- Flexible windowing: built-in Hanning/rectangle or custom window functions
- Noise floor estimation modes: auto (median of methods), median-based, trimmed mean, or exclude harmonics
- Configurable signal bandwidth and flicker noise removal

**Parameters:**
- `'OSR'` - Oversampling ratio (default: 1)
- `'window'` - Window function: 'hann', 'rect', or function handle (default: 'hann')
- `'averageMode'` - 'normal' or 'coherent' averaging (default: 'normal')
- `'NFMethod'` - Noise floor estimation: 'auto', 'median', 'mean', or 'exclude' (default: 'auto'); numeric: 0=auto, 1=median, 2=mean, 3=exclude
- `'sideBin'` - Extra bins on each side of signal peak (default: 'auto')
- `'cutoff'` - High-pass cutoff frequency for flicker noise removal (default: 0)
- `'label'` - Enable plot annotations (default: true)
- `'disp'` - Enable plotting (default: true)
- `'dispItem'` - Display item selector (default: 'sfedutrlyhop', all items)
  - String where each character (case insensitive) enables a specific annotation:
  - `'s'` - Signal power text and signal bin marker
  - `'f'` - Input frequency and sampling frequency (Fin/Fs)
  - `'e'` - Effective Number of Bits (ENOB)
  - `'d'` - Signal-to-Noise and Distortion Ratio (SNDR)
  - `'u'` - Spurious-Free Dynamic Range (SFDR)
  - `'t'` - Total Harmonic Distortion (THD)
  - `'r'` - Signal-to-Noise Ratio (SNR)
  - `'l'` - Noise floor level
  - `'y'` - Noise Spectral Density (NSD) and horizontal dash line
  - `'o'` - Oversampling Ratio (OSR) and vertical bandwidth line
  - `'h'` - Harmonic markers
  - `'p'` - Maximum spur marker
  - Example: `'sfe'` shows only signal, frequency, and ENOB annotations

**Example:**
```matlab
% Basic usage with 32x oversampling and coherent averaging
[enob, sndr, sfdr] = plotspec(sig, 100e6, 2^16, 'OSR', 32, 'averageMode', 'coherent');

% Multiple measurement runs with custom window
sig_multi = randn(10, 1024);  % 10 runs of 1024 samples
[enob, sndr] = plotspec(sig_multi, 'window', @blackman);

% Customize plot annotations - show only essential metrics
[enob, sndr] = plotspec(sig, 'dispItem', 'fedu');  % Show Fin/Fs, ENOB, SNDR, SFDR only
```

### plotphase

**Purpose:** Visualize coherent phase spectrum with polar display, showing harmonics on a polar coordinate system.

**Syntax:**
```matlab
h = plotphase(sig)
h = plotphase(sig, harmonic, maxSignal)
h = plotphase(sig, 'Name', Value)
```

**Key Features:**
- Two analysis modes: FFT-based and LMS-based (default)
- **LMS Mode**: Uses least squares harmonic fitting, displays noise circle as reference
- **FFT Mode**: Traditional coherent averaging with phase alignment
- Polar plot shows magnitude (radius) and phase (angle)
- Fundamental signal in red, harmonics in blue
- Supports oversampling and flicker noise removal

**Parameters:**
- `'mode'` - Analysis mode: 'LMS' (default) or 'FFT'
- `'OSR'` - Oversampling ratio (default: 1)
- `'Fs'` - Sampling frequency (default: 1)
- `'cutoff'` - High-pass cutoff frequency (default: 0)

**Example:**
```matlab
% LMS mode with noise circle display
plotphase(sig, 7, 2^16, 'mode', 'LMS');

% FFT mode with oversampling
plotphase(sig, 10, 'mode', 'FFT', 'OSR', 64);
```

### sinfit

**Purpose:** Four-parameter iterative sine wave fitting to extract amplitude, phase, DC offset, and frequency.

**Syntax:**
```matlab
[fitout, freq, mag, dc, phi] = sinfit(sig)
[fitout, freq, mag, dc, phi] = sinfit(sig, f0, tol, rate, fsearch, verbose, niter)
[fitout, freq, mag, dc, phi] = sinfit(sig, 'Name', Value, ...)
```

**Key Features:**
- Iterative least-squares refinement with frequency gradient descent
- Automatic frequency estimation using FFT with parabolic interpolation
- Configurable convergence tolerance and learning rate
- Optional fine frequency search control (`fsearch`)
- Verbose output for debugging iteration progress
- Handles both single-channel and multi-channel (averaged) input

**Algorithm:**
1. Initial 3-parameter fit (cosine, sine, DC) using linear least squares
2. Iterative frequency refinement by computing frequency gradient (if `fsearch=1`)
3. Convergence when relative error < tolerance (default: 1e-12)
4. Convergence warning if maximum iterations reached without meeting tolerance

**Parameters (positional or Name-Value):**
- `f0` - Initial frequency estimate (normalized, default: 0 for auto-detect)
- `tol` - Convergence tolerance (default: 1e-12)
- `rate` - Learning rate for frequency update (default: 0.5)
- `fsearch` - Force fine frequency search iteration (default: 0, auto-enabled when f0=0)
- `verbose` - Enable verbose output during iteration (default: 0)
- `niter` - Maximum iterations for frequency refinement (default: 100)

**Example:**
```matlab
% Automatic frequency estimation (auto-enables fsearch)
[fitout, freq, mag, dc, phi] = sinfit(sig);

% Known frequency with custom tolerance (positional)
[fitout, freq] = sinfit(sig, 0.123, 1e-10, 0.7);

% 3-parameter fit only at known frequency (no iteration)
[fitout, freq] = sinfit(sig, 0.123);

% Force iteration with verbose output (Name-Value)
[fitout, freq] = sinfit(sig, 'f0', 0.123, 'fsearch', 1, 'verbose', 1);

% Custom iteration limit and tolerance
[fitout, freq] = sinfit(sig, 'niter', 200, 'tol', 1e-15);
```

### findfreq

**Purpose:** Find the dominant frequency in a signal using sine wave fitting.

**Syntax:**
```matlab
freq = findfreq(sig, fs)
```

**Key Features:**
- Uses iterative sine fitting algorithm (sinfit) internally
- Returns fitted frequency, not FFT peak frequency
- Handles both absolute (with fs) and normalized frequency (fs=1)

**Example:**
```matlab
% Find frequency of a 1 kHz signal sampled at 10 kHz
freq = findfreq(sig, 10000);  % Returns ~1000 Hz

% Get normalized frequency
freq_norm = findfreq(sig);  % Returns ~0.1
```

### findbin

**Purpose:** Find the nearest FFT bin that ensures coherent sampling (integer cycles in FFT window).

**Syntax:**
```matlab
b = findbin(fs, fin, n)
```

**Key Features:**
- Ensures gcd(bin, n) = 1 for coherent sampling
- Prevents repeated phase sampling in FFT window
- Searches both upward and downward from initial estimate
- Returns upper bin when two bins are equidistant

**Example:**
```matlab
% Find coherent bin for 1 kHz signal
fs = 10000;  % 10 kHz sampling
n = 1024;    % 1024-point FFT
fin = 1000;  % 1 kHz signal
b = findbin(fs, fin, n);  % Returns 103

% Determine actual coherent frequency
fin_actual = b * fs / n;  % = 1006.8 Hz
```

### wcalsin

**Purpose:** Estimate per-bit weights and DC offset for ADC calibration using sine wave input.

**Syntax:**
```matlab
[weight, offset, postcal, ideal, err, freqcal] = wcalsin(bits)
[weight, offset, postcal, ideal, err, freqcal] = wcalsin(bits, 'Name', Value)
[weight, offset, postcal, ideal, err, freqcal] = wcalsin({bits1, bits2, ...}, 'Name', Value)
```

**Key Features:**
- Single-dataset or multi-dataset joint calibration
- Automatic frequency search with coarse and fine iteration
- Handles rank-deficient bit matrices by merging correlated columns
- Harmonic exclusion up to specified order
- Automatic polarity enforcement
- SNR check with warning when calibration quality is poor (< 20 dB)

**Algorithm:**
1. If frequency unknown: coarse search using multiple bit combinations, then fine iterative search
2. Build sine/cosine basis for fundamental and harmonics
3. Solve two least-squares formulations (cosine-based and sine-based)
4. Choose solution with lower residual
5. Iteratively refine frequency using gradient descent (if fsearch=1)
6. Handle rank deficiency by identifying and merging correlated columns

**Parameters:**
- `'freq'` - Normalized input frequency (default: 0 for auto-search)
- `'order'` - Number of harmonics to exclude from fitting (default: 1)
- `'rate'` - Adaptive rate for frequency iteration (default: 0.5)
- `'reltol'` - Relative error tolerance (default: 1e-12)
- `'niter'` - Maximum iterations for fine search (default: 100)
- `'fsearch'` - Force fine frequency search (default: 0)
- `'verbose'` - Enable verbose output (default: 0)
- `'nomWeight'` - Nominal weights for rank deficiency handling

**Example:**
```matlab
% Basic calibration with automatic frequency search
[wgt, off] = wcalsin(bits);

% Known frequency with 3rd harmonic exclusion
[wgt, off, cal, ideal, err, freq] = wcalsin(bits, 'freq', 0.123, 'order', 3);

% Multi-dataset joint calibration
[wgt, off] = wcalsin({bits1, bits2}, 'freq', [0.1, 0.2], 'order', 5);
```

### cdacwgt

**Purpose:** Calculate bit weights for multi-segment capacitive DAC with bridge capacitors and parasitics.

**Syntax:**
```matlab
[weight, ctot] = cdacwgt(cd, cb, cp)
```

**Key Features:**
- Models multi-segment CDAC architecture
- Accounts for capacitive divider effects
- Handles bridge capacitors between segments
- Includes parasitic capacitances
- Processes LSB to MSB for accurate weight attenuation

**Circuit Model:**
```
MSB side <---||------------||---< LSB side
             cb   |    |   Cl (load from previous bits)
                 ---  ---
             cp  ---  ---  cd
                  |    |
                 gnd   Vbot
```

**Parameters:**
- `cd` - DAC capacitors [MSB ... LSB]
- `cb` - Bridge capacitors [MSB ... LSB] (0 for no bridge)
- `cp` - Parasitic capacitors [MSB ... LSB]

**Example:**
```matlab
% Simple binary-weighted 4-bit DAC
cd = [8 4 2 1];
cb = [0 0 0 0];
cp = [0 0 0 0];
[weight, ctot] = cdacwgt(cd, cb, cp);
% Returns: weight = [0.5333 0.2667 0.1333 0.0667], ctot = 15

% 6-bit CDAC with 3+3 segments
cd = [4 2 1  4 2 1];
cb = [0 4 0  8/7 0 0];  % Bridge between segments
cp = [0 0 0  0 0 1];
[weight, ctot] = cdacwgt(cd, cb, cp);
% Returns: weight = [0.5000 0.2500 0.1250 0.0625 0.0312 0.0156]
```

### plotwgt

**Purpose:** Visualize absolute bit weights with radix annotations to identify ADC architecture and detect calibration errors. Also computes optimal weight scaling factor and effective resolution.

**Syntax:**
```matlab
radix = plotwgt(weights)
radix = plotwgt(weights, disp)
[radix, wgtsca] = plotwgt(weights)
[radix, wgtsca, effres] = plotwgt(weights)
```

**Key Features:**
- Plots absolute bit weights on logarithmic Y-axis
- Annotates radix (scaling factor) between consecutive bits
- Negative weights displayed in red to indicate sign errors
- MSB (largest weight) marked in red, LSB (smallest significant weight) marked in green
- Dual x-axis labels: ascending array order (bottom) and descending significance order (top)
- Displays effective resolution in the plot
- Computes optimal weight scaling factor (`wgtsca`) that minimizes rounding error
- Estimates effective resolution (`effres`) from significant bit weights
- Optional `disp` argument to disable plotting

**Algorithm for wgtsca and effres:**
1. Sort absolute weights descending to identify bit significance
2. Find "significant" bits by detecting ratio jumps >= 3 (large jumps indicate noise/redundant bits)
3. Initial scaling normalizes the smallest significant weight to 1
4. Refine scaling to minimize rounding error across significant weights
5. Compute effres as `log2(sum(absW_sig)/absW_LSB + 1)`

**Parameters:**
- `weights` - Bit weights from MSB to LSB, vector (1 x B)
- `disp` - Display flag (optional, default: 1). Set to 0 to disable plotting.

**Outputs:**
- `radix` - Radix between consecutive bits, vector (1 x B-1)
  - `radix(i) = |weight(i) / weight(i+1)|`
  - Binary ADC: radix ≈ 2.00 for all bits
  - Sub-radix ADC: radix < 2.00 (e.g., 1.5-bit/stage → ~1.90)
- `wgtsca` - Optimal weight scaling factor that normalizes weights to minimize rounding error
- `effres` - Effective resolution in bits, estimated from significant weight ratios

**Example:**
```matlab
% Visualize ideal 12-bit binary weights
weights_ideal = 2.^(11:-1:0);
[radix, wgtsca, effres] = plotwgt(weights_ideal);

% Visualize CDAC weights (6-bit with 3+3 segments)
cd = [4 2 1 4 2 1];       % Two 3-bit segments [MSB ... LSB]
cb = [0 0 0 8/7 0 0];     % Bridge cap between segments
cp = [0 0 0 0 0 1];       % Parasitic at LSB
weight = cdacwgt(cd, cb, cp);
radix = plotwgt(weight);

% Compute scaling without displaying plot
[~, wgtsca, effres] = plotwgt(weights, 0);
```

### plotres

**Purpose:** Plot partial-sum residuals of an ADC bit matrix as scatter plots, revealing correlations, nonlinearity patterns, and redundancy between bit stages.

**Syntax:**
```matlab
plotres(sig, bits)
plotres(sig, bits, wgt)
plotres(sig, bits, wgt, xy)
plotres(sig, bits, wgt, xy, alpha)
plotres(sig, bits, 'Name', Value)
```

**Key Features:**
- Tiled scatter plots of residuals at different bit stages
- Residual at stage k = sig - bits(:,1:k) * wgt(1:k)'
- Translucent markers with automatic or manual alpha control
- Supports custom bit weights and arbitrary bit-pair selections

**Parameters:**
- `sig` - Ideal input signal (N x 1 or 1 x N)
- `bits` - Raw ADC bit matrix (N x M), MSB first
- `wgt` - Bit weights (optional, default: binary weights `[2^(M-1), ..., 1]`)
- `xy` - Bit-pair indices to plot (optional, default: `[(0:(M-1))', ones(M,1)*M]`)
- `alpha` - Marker transparency (optional, default: `'auto'`)
  - `'auto'`: scales as `clamp(1000/N, 0.1, 1)`
  - Numeric scalar in (0, 1]: fixed transparency

**Example:**
```matlab
% Basic residual plot with binary weights
N = 1024; M = 6;
sig = (sin(2*pi*(0:N-1)'/N * 3)/2 + 0.5) * (2^M - 1);
code = round(sig);
bits = dec2bin(code, M) - '0';
plotres(sig, bits);

% Specific bit pairs with custom transparency
plotres(sig, bits, 2.^(M-1:-1:0), [2 4; 4 6], 0.3);
```

**See also:** bitchk, plotwgt, plotressin

---

### plotressin

**Purpose:** Convenience wrapper that calibrates bit weights via `wcalsin` and then forwards the results to `plotres`, eliminating the manual calibration step.

**Syntax:**
```matlab
plotressin(bits)
plotressin(bits, xy)
plotressin(bits, ..., 'Name', Value)
```

**Key Features:**
- Internally calls `wcalsin` to recover calibrated weights and ideal signal
- Forwards the reconstructed reference signal (`ideal + offset`) and weights to `plotres`
- Accepts the same `xy` format as `plotres`
- Forwards `freq`, `order`, `verbose` parameters to `wcalsin`
- Forwards `alpha` parameter to `plotres`

**Parameters:**
- `bits` - Raw ADC bit matrix (N x M), MSB first
- `xy` - Bit-pair indices to plot (optional, same format as `plotres`)
- `'freq'` - Normalized input frequency for `wcalsin` (default: 0 for auto)
- `'order'` - Number of harmonics in fitting model (default: 1)
- `'verbose'` - Verbose output flag (default: 0)
- `'alpha'` - Marker transparency, forwarded to `plotres` (default: `'auto'`)

**Example:**
```matlab
% Basic usage (automatic frequency search and calibration)
N = 1024; M = 6;
sig = (sin(2*pi*(0:N-1)'/N * 3)/2 + 0.5) * (2^M - 1);
code = round(sig);
bits = dec2bin(code, M) - '0';
plotressin(bits)

% Specific bit pairs with known frequency
plotressin(bits, [0 6; 3 6], 'freq', 3/1024)

% Forward calibration parameters
plotressin(bits, 'order', 3)
```

**See also:** plotres, wcalsin, plotwgt

---

### errsinv

**Purpose:** Shortcut for `errsin` that defaults to value-mode binning (`xaxis='value'`), useful for quick INL-style error visualization without specifying the `xaxis` parameter.

**Syntax:**
```matlab
[emean, erms, xx, anoi, pnoi, err, errxx] = errsinv(sig)
[emean, erms, xx, anoi, pnoi, err, errxx] = errsinv(sig, 'Name', Value)
```

**Key Features:**
- Calls `errsin` with `'xaxis', 'value'` pre-set
- When called with no outputs, automatically enables display (`'disp', 1`)
- All other `errsin` name-value parameters are forwarded as-is

**Parameters:**
- `sig` - Input signal (same as `errsin`)
- All name-value arguments accepted by `errsin` (e.g., `'bin'`, `'fin'`, `'erange'`, `'disp'`)

**Outputs:**
- Same as `errsin`: `emean`, `erms`, `xx`, `anoi`, `pnoi`, `err`, `errxx`

**Example:**
```matlab
% Quick value-mode error plot (auto-display)
sig = sin(2*pi*0.12345*(0:999)') + 0.01*randn(1000,1);
errsinv(sig)

% With custom bins
[emean, erms, xx] = errsinv(sig, 'bin', 50);
```

**See also:** errsin

---

### inlsin

**Purpose:** Calculate ADC's INL (Integral Nonlinearity) and DNL (Differential Nonlinearity) using sine wave histogram method.

**Syntax:**
```matlab
[inl, dnl, code] = inlsin(data)
[inl, dnl, code] = inlsin(data, excl, disp)
[inl, dnl, code] = inlsin(data, 'name', value)
```

**Key Features:**
- Histogram-based method assuming ideal sine wave input
- Automatic endpoint exclusion to avoid clipping noise
- Detects and highlights missing codes (DNL ≤ -1)
- Zero-mean normalized DNL output
- INL computed as cumulative sum of DNL

**Algorithm:**
1. Histogram the ADC output codes
2. Apply cosine transform to linearize sine distribution
3. Calculate DNL from differences in linearized histogram
4. Normalize to LSB = 1 and remove DC offset
5. Compute INL as cumulative sum of DNL

**Parameters:**
- `excl` - Exclusion ratio for endpoints (default: 0.01 = 1%)
- `disp` - Display plots (default: auto when no outputs)

**Example:**
```matlab
% Generate 8-bit ADC output and analyze
t = linspace(0, 2*pi, 10000);
data = round(127.5 + 127.5*sin(t));
[inl, dnl, code] = inlsin(data);

% Custom exclusion ratio
[inl, dnl, code] = inlsin(data, 0.05);  % Exclude 5% from each end
```

### errsin

**Purpose:** Analyze sine wave fit errors with histogram binning to identify amplitude and phase noise.

**Syntax:**
```matlab
[emean, erms, xx, anoi, pnoi, err, errxx] = errsin(sig)
[emean, erms, xx, anoi, pnoi, err, errxx] = errsin(sig, 'Name', Value)
```

**Key Features:**
- Two binning modes: phase (default) or value
- **Phase mode**: Estimates amplitude and phase noise components
- **Value mode**: Useful for INL analysis
- Configurable number of bins and error range filtering
- Least-squares decomposition of noise sources

**Noise Estimation (Phase Mode):**
- Amplitude noise affects all phases equally (cos² pattern)
- Phase noise creates errors proportional to slope (sin² pattern)
- Fits: `erms² = anoi²·cos²(θ) + pnoi²·sin²(θ)`

**Parameters:**
- `'bin'` - Number of histogram bins (default: 100)
- `'fin'` - Normalized input frequency (default: 0 for auto)
- `'xaxis'` - Binning mode: 'phase' (default) or 'value'
- `'erange'` - Error range filter [min, max] (default: [])
- `'disp'` - Display plots (default: auto when no outputs)

**Example:**
```matlab
% Phase mode analysis with noise estimation
sig = sin(2*pi*0.12345*(0:999)') + 0.01*randn(1000,1);
[emean, erms, xx, anoi, pnoi] = errsin(sig);

% Value mode for INL visualization
[emean, erms, xx] = errsin(sig, 'xaxis', 'value', 'bin', 50);

% Filter to specific phase range
[~, ~, ~, ~, ~, err, phase] = errsin(sig, 'erange', [90, 180]);
```

### tomdec

**Purpose:** Thompson decomposition of single-tone signal into fundamental sine wave, harmonic distortions, and other errors.

**Syntax:**
```matlab
[sine, err, har, oth, freq] = tomdec(sig)
[sine, err, har, oth, freq] = tomdec(sig, freq, order, disp)
[sine, err, har, oth, freq] = tomdec(sig, 'name', value)
```

**Key Features:**
- Least-squares fitting to separate signal components
- Automatic frequency detection if not specified
- Decomposes into: fundamental + harmonics + residual
- Configurable harmonic order (default: 10)

**Decomposition:**
- `sig = sine + err`
- `sine` = DC + fundamental only
- `err = har + oth`
- `har` = 2nd through nth harmonics
- `oth` = all remaining errors (noise, non-harmonic distortion)

**Parameters:**
- `freq` - Normalized signal frequency (default: auto-detect)
- `order` - Number of harmonics to fit (default: 10)
- `disp` - Display results (default: auto when no outputs)

**Example:**
```matlab
% Auto-detect frequency and decompose
[sine, err, har, oth] = tomdec(sig);

% Fit only 5 harmonics with known frequency
[sine, err, har, oth] = tomdec(sig, 0.123, 5);
```

### ntfperf

**Purpose:** Analyze noise transfer function performance for noise-shaping ADCs (Delta-Sigma modulators).

**Syntax:**
```matlab
snr = ntfperf(ntf, fl, fh)
snr = ntfperf(ntf, fl, fh, disp)
```

**Key Features:**
- Evaluates SNR improvement from noise-shaping and oversampling
- Supports lowpass and bandpass NTF analysis
- High-resolution frequency evaluation (1M points)
- Automatic plot generation with signal band markers

**Parameters:**
- `ntf` - Noise transfer function (tf, zpk, or ss object)
- `fl` - Low frequency bound [0, 0.5]
- `fh` - High frequency bound (fl, 0.5]
- `disp` - Display plot (default: 0)

**Example:**
```matlab
% 1st-order lowpass delta-sigma with 16x OSR
ntf = tf([1 -1], [1 0], 1);  % NTF = 1 - z^-1
snr = ntfperf(ntf, 0, 0.5/16);  % Returns ~31 dB improvement

% Bandpass NTF with visualization
ntf = tf([1 0 1], [1 0 0], 1);  % NTF = 1 + z^-2
snr = ntfperf(ntf, 0.24, 0.26, 1);
```

### alias

**Purpose:** Calculate the aliased frequency after sampling, accounting for Nyquist zone effects.

**Syntax:**
```matlab
fal = alias(fin, fs)
```

**Key Features:**
- Handles signals in any Nyquist zone
- Even zones (0,2,4,...): normal aliasing
- Odd zones (1,3,5,...): mirrored aliasing
- Supports scalar or vector input

**Example:**
```matlab
% Signal at 0.7*fs aliases to 0.3*fs (mirrored)
fal = alias(70, 100);  % Returns 30

% Signal at 1.3*fs aliases to 0.3*fs (normal)
fal = alias(130, 100);  % Returns 30

% Multiple frequencies
fal = alias([30 70 130], 100);  % Returns [30 30 30]
```

### ifilter

**Purpose:** Ideal FFT-based brickwall filter to extract specified frequency bands.

**Syntax:**
```matlab
sigout = ifilter(sigin, passband)
```

**Key Features:**
- FFT-based filtering with sharp transitions
- Multiple passband support (union of bands)
- Filters each column independently
- Maintains Hermitian symmetry for real output

**Parameters:**
- `sigin` - Input signal matrix (each column filtered independently)
- `passband` - Frequency bands [fLow, fHigh] as rows (normalized to [0, 0.5])

**Example:**
```matlab
% Single passband from 0.1*Fs to 0.2*Fs
sigout = ifilter(sigin, [0.1, 0.2]);

% Multiple passbands
sigout = ifilter(sigin, [0.05, 0.15; 0.25, 0.35]);
```

### bitchk

**Purpose:** Check ADC overflow by analyzing bit segment residue distributions.

**Syntax:**
```matlab
bitchk(bits)
bitchk(bits, wgt, chkpos)
bitchk(bits, 'name', value)
```

**Key Features:**
- Visualizes normalized residue distribution for each bit
- Detects overflow (≥1) and underflow (≤0) conditions
- Color-coded visualization: blue (normal), red (overflow), yellow (underflow)
- Shows percentage of samples at boundaries

**Parameters:**
- `bits` - Raw ADC bit matrix [MSB ... LSB] (N×M)
- `wgt` - Bit weights (default: binary weights)
- `chkpos` - Bit position to check (default: MSB)

**Example:**
```matlab
% Check with default binary weights
bits = randi([0 1], 10000, 10);
bitchk(bits);

% Custom weights and check position
wgt = 2.^(9:-1:0);
bitchk(bits, wgt, 8);  % Check segment from 8th-bit to LSB
```

## Usage Examples

### Example 1: Quick Dashboard Analysis with adcpanel

```matlab
% Load ADC data
load('adc_capture.mat');  % Contains 'data' variable

% Run comprehensive analysis with one function call
rep = adcpanel(data, 'fs', 100e6);

% Access results from the report structure
fprintf('=== ADC Performance Summary ===\n');
fprintf('ENOB: %.2f bits\n', rep.spectrum.enob);
fprintf('SNDR: %.2f dB\n', rep.spectrum.sndr);
fprintf('SFDR: %.2f dB\n', rep.spectrum.sfdr);
fprintf('SNR: %.2f dB\n', rep.spectrum.snr);
fprintf('THD: %.2f dB\n', rep.spectrum.thd);
fprintf('Max INL: %.3f LSB\n', max(abs(rep.linearity.inl)));
fprintf('Max DNL: %.3f LSB\n', max(abs(rep.linearity.dnl)));
fprintf('Amplitude noise: %.2e\n', rep.errorPhase.anoi);
fprintf('Phase noise: %.2e rad\n', rep.errorPhase.pnoi);
```

### Example 2: Complete ADC Characterization

```matlab
% Load ADC data (assume 12-bit ADC at 100 MHz sampling)
load('adc_capture.mat');  % Contains 'data' variable

% 1. Spectral analysis
[enob, sndr, sfdr, snr, thd] = plotspec(data, 100e6, 2^12, 5);
fprintf('ADC Performance:\n');
fprintf('  ENOB: %.2f bits\n', enob);
fprintf('  SNDR: %.2f dB\n', sndr);
fprintf('  SFDR: %.2f dB\n', sfdr);
fprintf('  SNR: %.2f dB\n', snr);
fprintf('  THD: %.2f dB\n', thd);

% 2. Find input frequency
freq = findfreq(data, 100e6);
fprintf('  Input Frequency: %.2f MHz\n', freq/1e6);

% 3. Linearity analysis
[inl, dnl, code] = inlsin(data, 0.02);  % Exclude 2% endpoints
fprintf('  Max INL: %.3f LSB\n', max(abs(inl)));
fprintf('  Max DNL: %.3f LSB\n', max(abs(dnl)));
fprintf('  Missing codes: %d\n', sum(dnl <= -1));

% 4. Error analysis
[emean, erms, phase, anoi, pnoi] = errsin(data, 'bin', 200);
fprintf('  Amplitude noise: %.2e\n', anoi);
fprintf('  Phase noise: %.2e rad\n', pnoi);
```

### Example 3: Oversampling ADC Analysis

```matlab
% Analyze 16-bit Delta-Sigma ADC with 64x oversampling
OSR = 64;
[enob, sndr, ~, snr] = plotspec(data, 1e6, 2^16, 'OSR', OSR, ...
                                  'window', @blackman, ...
                                  'averageMode', 'coherent');

% Analyze noise transfer function
ntf = tf([1 -1], [1 -0.5], 1);  % 1st-order NTF
snr_gain = ntfperf(ntf, 0, 0.5/OSR, 1);
fprintf('NTF provides %.2f dB SNR improvement\n', snr_gain);
```

### Example 4: SAR ADC Calibration

```matlab
% Load raw bit data from SAR ADC
load('sar_bits.mat');  % Contains 'bits' matrix

% 1. Calibrate weights using sine wave
[weight, offset, postcal] = wcalsin(bits, 'freq', 0.1234, 'order', 3);

% 2. Check for overflow in calibrated data
bitchk(bits, weight);

% 3. Calculate theoretical CDAC weights for comparison
% Assume 12-bit with 6+6 split capacitor array
cd = [32 16 8 4 2 1,  32 16 8 4 2 1];
cb = [0 0 0 0 0 32,  0 0 0 0 0 0];
cp = zeros(1, 12);
[weight_ideal, ctot] = cdacwgt(cd, cb, cp);

% 4. Compare calibrated vs. ideal weights
figure;
subplot(2,1,1);
bar(weight);
title('Calibrated Weights');
subplot(2,1,2);
bar(weight_ideal);
title('Ideal Weights');

% 5. Analyze calibrated performance
[enob_cal, sndr_cal] = plotspec(postcal, 'disp', true);
fprintf('After calibration: ENOB = %.2f, SNDR = %.2f dB\n', enob_cal, sndr_cal);
```

### Example 5: Multi-Frequency Testing

```matlab
% Test ADC at multiple input frequencies
frequencies = [1e6, 5e6, 10e6, 20e6, 40e6];  % 1 to 40 MHz
fs = 100e6;  % 100 MHz sampling
N = 8192;    % FFT length

results = struct();
for i = 1:length(frequencies)
    fin = frequencies(i);

    % Find coherent bin
    bin = findbin(fs, fin, N);
    fin_coherent = bin * fs / N;

    % Generate test signal (simulated)
    t = (0:N-1)'/fs;
    sig = sin(2*pi*fin_coherent*t) + 0.001*randn(N,1);

    % Measure performance
    [enob, sndr, sfdr] = plotspec(sig, fs, 2, 'disp', false);

    % Store results
    results(i).freq = fin;
    results(i).freq_actual = fin_coherent;
    results(i).enob = enob;
    results(i).sndr = sndr;
    results(i).sfdr = sfdr;

    fprintf('Freq: %.1f MHz -> ENOB: %.2f, SNDR: %.2f dB\n', ...
            fin/1e6, enob, sndr);
end

% Plot ENOB vs. frequency
figure;
plot([results.freq]/1e6, [results.enob], 'o-');
xlabel('Input Frequency (MHz)');
ylabel('ENOB (bits)');
title('ADC Performance vs. Frequency');
grid on;
```

### Example 6: Thompson Decomposition Analysis

```matlab
% Decompose ADC output into components
[sine, err, har, oth, freq] = tomdec(data, 'order', 10, 'disp', true);

% Analyze power of each component
sig_power = rms(sine)^2;
har_power = rms(har)^2;
oth_power = rms(oth)^2;

fprintf('Power Decomposition:\n');
fprintf('  Signal: %.2e (%.1f dB)\n', sig_power, 10*log10(sig_power));
fprintf('  Harmonics: %.2e (%.1f dB)\n', har_power, 10*log10(har_power));
fprintf('  Other: %.2e (%.1f dB)\n', oth_power, 10*log10(oth_power));
fprintf('  THD: %.2f dB\n', 10*log10(har_power/sig_power));
fprintf('  SNR: %.2f dB\n', 10*log10(sig_power/oth_power));
```

### Example 7: Bandpass Filtering

```matlab
% Extract specific frequency bands for noise analysis
% Assume 100 MHz ADC with signal at 20 MHz

% Define multiple passbands
passbands = [
    0.15, 0.25;   % Signal band (15-25 MHz)
    0.40, 0.45;   % High-frequency noise band
];

% Filter signal to these bands
sig_filtered = ifilter(data, passbands);

% Analyze filtered content
plotspec(sig_filtered, 100e6, 2^12);
```

## Legacy Functions

The `legacy/` directory contains older function names for backward compatibility. These functions call the new implementations with updated names:

| Legacy Function | New Function | Notes |
|----------------|--------------|-------|
| `specPlot.m` | `plotspec.m` | Spectral analysis (old camelCase) |
| `specPlotPhase.m` | `plotphase.m` | Phase spectrum |
| `findBin.m` | `findbin.m` | Coherent bin finder |
| `findFin.m` | `findfreq.m` | Frequency finder |
| `FGCalSine.m` | `wcalsin.m` | Weight calibration |
| `cap2weight.m` | `cdacwgt.m` | CDAC weight calculator |
| `weightScaling.m` | `plotwgt.m` | Weight visualization |
| `INLsine.m` | `inlsin.m` | INL/DNL analysis |
| `errHistSine.m` | `errsin.m` | Error histogram |
| `sineFit.m` | `sinfit.m` | Sine fitting |
| `tomDecomp.m` | `tomdec.m` | Thompson decomposition |
| `NTFAnalyzer.m` | `ntfperf.m` | NTF performance |
| `overflowChk.m` | `bitchk.m` | Overflow checker |
| `bitInBand.m` | N/A | Bits-wise filter (deprecated) |

**Note:** It's recommended to use the new function names in new code. Legacy functions are provided for compatibility only.

## Requirements

### Minimum Requirements
- MATLAB R2016b or later
- No toolboxes required for core functionality

### Optional (for extended features)
- **Signal Processing Toolbox**: Required only for custom window functions (e.g., `@blackman`, `@kaiser`)
  - Built-in Hanning and rectangle windows do not require this toolbox
- **Control System Toolbox**: Required for `ntfperf` function (transfer function analysis)

## File Structure

```
matlab/
├── README.md                 # This file
├── setupLib.m               # Setup script for adding to path
├── src/                     # Source code directory
│   ├── adcpanel.m          # Comprehensive ADC analysis dashboard
│   ├── plotspec.m          # Spectral analysis
│   ├── plotphase.m         # Phase spectrum analysis
│   ├── plotwgt.m           # Weight visualization
│   ├── sinfit.m            # Sine wave fitting
│   ├── findfreq.m          # Frequency finder
│   ├── findbin.m           # Coherent bin finder
│   ├── wcalsin.m          # Weight calibration
│   ├── cdacwgt.m           # CDAC weight calculator
│   ├── inlsin.m            # INL/DNL analysis
│   ├── errsin.m            # Error histogram analysis
│   ├── tomdec.m            # Thompson decomposition
│   ├── perfosr.m           # Performance vs OSR sweep
│   ├── ntfperf.m           # NTF performance analyzer
│   ├── alias.m             # Alias calculator
│   ├── ifilter.m           # Ideal filter
│   ├── plotres.m           # Partial-sum residual scatter plots
│   ├── bitchk.m            # Overflow checker
│   ├── shortcut/           # Convenience wrapper functions
│   │   ├── plotressin.m    # plotres + wcalsin in one call
│   │   └── errsinv.m       # errsin value-mode shortcut
│   ├── legacy/             # Legacy function names (for compatibility)
│   │   ├── specPlot.m
│   │   ├── specPlotPhase.m
│   │   ├── findBin.m
│   │   ├── findFin.m
│   │   ├── FGCalSine.m
│   │   ├── cap2weight.m
│   │   ├── weightScaling.m
│   │   ├── INLsine.m
│   │   ├── errHistSine.m
│   │   ├── sineFit.m
│   │   ├── tomDecomp.m
│   │   ├── NTFAnalyzer.m
│   │   ├── overflowChk.m
│   │   └── bitInBand.m
│   └── toolbox.ignore
├── toolbox/                 # Packaged toolbox files
│   ├── ADCToolbox_1v30.mltbx  # Latest toolbox package
│   ├── ADCToolbox_1v21.mltbx  # Previous versions
│   ├── ADCToolbox_1v2.mltbx
│   ├── ADCToolbox_1v1.mltbx
│   ├── ADCToolbox_1v0.mltbx
│   ├── ADCToolbox_0v12.mltbx
│   ├── ADCToolbox_0v11.mltbx
│   ├── ADCToolbox_0v1.mltbx
│   ├── deploymentLog.html
│   └── icon.png
└── .gitignore
```

## Common Workflows

### 1. Quick Performance Check (Dashboard Approach)
```matlab
% Use adcpanel for one-stop comprehensive analysis
rep = adcpanel(data, 'fs', fs);
% All metrics available in rep structure with organized figure panels
```

### 2. Quick Performance Check (Individual Functions)
```matlab
% Perform spectrum analysis only
[enob, sndr, sfdr] = plotspec(data);
```

### 3. Detailed Characterization (Manual Workflow)
```matlab
% Comprehensive analysis using individual functions
freq = findfreq(data, fs);
[enob, sndr, sfdr, snr, thd] = plotspec(data, fs, 2^bits, 5);
[inl, dnl] = inlsin(data);
[emean, erms, phase, anoi, pnoi] = errsin(data);
```

### 4. Calibration Workflow (Bit-wise Data)
```matlab
% Use adcpanel for automatic calibration and analysis
rep = adcpanel(bits, 'dataType', 'bits');
% Or manual calibration workflow:
[weight, offset, postcal] = wcalsin(bits);
bitchk(bits, weight);
[enob_after_cal, ~] = plotspec(postcal, 'disp', false);
```

### 5. Frequency Sweep Test
```matlab
% Test at multiple frequencies
for fin = test_frequencies
    bin = findbin(fs, fin, N);
    [enob(i), sndr(i)] = specplot(data{i}, fs, maxCode, 'disp', false);
end
plot(test_frequencies, enob);
```

## Tips and Best Practices

1. **Coherent Sampling**: Use `findbin` to ensure coherent sampling for accurate FFT analysis
2. **Oversampling**: Set `'OSR'` parameter in `plotspec` when analyzing noise-shaping ADCs
3. **Averaging**: Use `'averageMode', 'coherent'` for better noise floor in repeated measurements
4. **Window Selection**: Hanning window (default) is good for general use; use rectangle for coherent signals
5. **Endpoint Exclusion**: Increase `excl` parameter in `inlsin` if data has clipping or saturation
6. **Frequency Accuracy**: For calibration, let `wcalsin` auto-search frequency or use fine search
7. **Multi-dataset Calibration**: Use cell array input to `wcalsin` for better statistical convergence
8. **Rank Deficiency**: If `wcalsin` warns about rank, adjust `'nomWeight'` based on expected bit weights

## Troubleshooting

### Issue: "Rank deficiency detected" in wcalsin
**Solution:**
- Check that bit data has sufficient variation
- Adjust `'nomWeight'` parameter to match actual bit weights
- Ensure input data covers full code range

### Issue: "SNR below 20 dB" warning in wcalsin
**Solution:**
- Verify that input data contains a clean sine wave signal
- Check for excessive noise or clipping in the data
- Ensure signal amplitude is appropriate for the ADC range
- This warning indicates calibration may have failed to correctly extract the sine wave

### Issue: Poor ENOB in plotspec
**Possible causes:**
- Non-coherent sampling (use `window` and `sideBin` to apply proper windowing)
- Clipping or saturation (use `errsin` or `bitchk` to check)

### Issue: NaN or Inf results
**Solution:**
- Check for zero-valued or constant input data
- Ensure proper data scaling
- Verify sampling frequency is positive

### Issue: inlsin returns unexpected DNL
**Solution:**
- Ensure input is integer codes (will auto-round with warning)
- Increase exclusion ratio if endpoints are noisy
- Verify input is from sine wave (not other waveforms)
- Uses enough data to analysis (typically 64x more than the quantization levels)

## Contributing

Contributions are welcome! Please follow these guidelines:
1. Maintain backward compatibility with legacy function names
2. Add comprehensive help documentation following MATLAB standards
3. Include input validation and error checking
4. Add examples in function help
5. Update this README when adding new functions

## Version History

- **v1.0** (Current) - First formal release

## Contact

For questions, issues, or feature requests, please contact jielu@tsinghua.edu.cn or zhangzs21@mails.tsinghua.edu.cn.

## See Also

- Python implementation: `../python/` - Python port of ADC analysis tools
- Documentation: `../doc/` - Additional documentation and theory
- Test datasets: `../dataset/` - Example ADC captures for testing

---

**Note:** This toolbox is designed for ADC testing and characterization. For production use, ensure proper validation with your specific ADC architecture and test setup.
