# errHistSine

## Overview

Analyzes ADC errors by comparing measured data against a fitted sine wave. Provides statistical error analysis and decomposes noise into amplitude and phase components.

## Syntax

```matlab
[emean, erms, phase_code, anoi, pnoi, err, xx] = errHistSine(data, ...)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data` | (required) | Input ADC data vector |
| `bin` | 100 | Number of histogram bins |
| `fin` | 0 (auto) | Normalized frequency (0-1), cycles per sample |
| `disp` | 1 | Display plots: 1=yes, 0=no |
| `mode` | 0 | 0=phase mode, ≥1=code mode |
| `erange` | [] | Filter errors: `[min, max]` range on x-axis |

### Outputs

| Output | Description |
|--------|-------------|
| `emean` | Mean error per bin (reveals INL in code mode) |
| `erms` | RMS error per bin |
| `phase_code` | Bin centers (phase in degrees or code values) |
| `anoi` | Amplitude noise RMS (phase mode only) |
| `pnoi` | Phase noise RMS in radians (phase mode only) |
| `err` | Raw errors (filtered if `erange` specified) |
| `xx` | X-axis values for `err` |

## Algorithm

### 1. Sine Wave Fitting

```
data_fit[n] = A · sin(2π · f_in · n + φ) + DC
```

Where: $A$ = amplitude, $f_{in}$ = normalized frequency, $\varphi$ = phase, $DC$ = offset

### 2. Error Calculation

```
err[n] = data_fit[n] - data[n]
```

### 3. Binning

**Phase Mode** (θ in degrees):

```
θ[n] = mod((φ/π) × 180 + n · f_in · 360, 360)
```

**Code Mode**:

```
bin_index = min(floor((data[n] - data_min) / bin_width) + 1, N_bins)
```

### 4. Statistics per Bin

**Mean error:**

```
emean[b] = (1/N_b) Σ err[i]  for i in bin b
```

**RMS error:**

```
erms[b] = sqrt((1/N_b) Σ (err[i] - emean[b])²)  for i in bin b
```

### 5. Noise Decomposition (Phase Mode Only)

Models RMS error variance as a combination of amplitude and phase noise:

```
erms²(θ) = σ²_A · cos²(θ) + (A · σ_φ)² · sin²(θ) + σ²_bl
```

Solved via least-squares:

```
┌                              ┐   ┌         ┐   ┌            ┐
│ cos²(θ₁)  sin²(θ₁)  1        │   │ σ²_A    │   │ erms²(θ₁)  │
│ cos²(θ₂)  sin²(θ₂)  1        │ · │ A²σ²_φ  │ = │ erms²(θ₂)  │
│    ⋮         ⋮      ⋮        │   │ σ²_bl   │   │     ⋮      │
│ cos²(θₙ)  sin²(θₙ)  1        │   └         ┘   │ erms²(θₙ)  │
└                              ┘                   └            ┘
```

**Outputs:**
- `anoi = σ_A` (amplitude noise)
- `pnoi = σ_φ` (phase noise in radians)

**Robust fallback**: If solution yields negative/imaginary values, tries phase-only, amplitude-only, or baseline-only fits.

## Physical Interpretation

### Amplitude Noise (anoi)
- **Sources**: reference noise, comparator noise, thermal noise
- **Units**: same as input data (LSB or volts)
- **Normalized**: `anoi/mag` (unitless fraction)

### Phase Noise (pnoi)
- **Sources**: sampling clock jitter, timing uncertainty
- **Units**: radians
- **Convert to timing jitter**:

```
δt_rms = pnoi / (2π · f_in · f_s)
```

Where `f_s` = sampling rate in Hz

**Example**: If `pnoi = 0.001` rad, `f_in = 0.1`, `f_s = 1 GHz`:
```
δt_rms = 0.001 / (2π · 0.1 · 1e9) ≈ 1.59 ps
```

### SNR Contributions

**From amplitude noise:**
```
SNR_amp [dBc] = 20·log₁₀(A / (√2 · anoi))
```

**From phase noise:**
```
SNR_phase [dBc] = 20·log₁₀(1 / (√2 · pnoi))
```

**Combined (uncorrelated noise):**
```
SNR_total = -10·log₁₀(10^(-SNR_amp/10) + 10^(-SNR_phase/10))
```

## Usage Examples

### Phase Mode: Noise Decomposition

```matlab
% Generate noisy sine wave
N = 10000; fin = 0.1;
data = sin(2*pi*fin*(0:N-1)) + 0.01*randn(1,N);

% Analyze
[emean, erms, phase, anoi, pnoi] = errHistSine(data, 'fin', fin);
fprintf('Amplitude Noise: %.4f\n', anoi);
fprintf('Phase Noise: %.4f rad\n', pnoi);
```

### Code Mode: INL Measurement

```matlab
% ADC output codes
adc_codes = load('adc_output.mat');

% Measure INL
[INL, erms, codes] = errHistSine(adc_codes, 'mode', 1, 'bin', 256);
fprintf('Peak INL: %.3f LSB\n', max(abs(INL)));
```

## Key Formulas

### Normalized Frequency

```
f_in = f_signal / f_sampling ∈ (0, 0.5)  [Nyquist criterion]
```

**Coherent sampling**: `f_in = M/N` where M and N are integers (M cycles in N samples)

### INL/DNL (Code Mode)

```
INL[k] = emean[k]

DNL[k] ≈ INL[k] - INL[k-1]

INL_peak = max|emean[k]| over all codes k

INL_rms = sqrt((1/N) Σ emean[k]²)
```

### ENOB (Effective Number of Bits)

```
ENOB = log₂(mag / (√2 · erms_total))
```

where `erms_total = sqrt(anoi² + (pnoi·mag)² + σ²_bl)`

**Relationship to SINAD:**
```
SINAD [dB] = 6.02 · ENOB + 1.76
ENOB = (SINAD - 1.76) / 6.02
```

## Applications

- **ADC Characterization**: INL/DNL measurement (code mode)
- **Noise Analysis**: Separate amplitude vs. phase noise sources
- **Jitter Testing**: Phase noise → aperture jitter quantification
- **Clock Quality**: Phase noise indicates clock performance

## Implementation Notes

- Auto-transposes row vectors
- Phase mode: ideal for dynamic testing with coherent sine inputs
- Code mode: ideal for static testing (ramp/histogram method)
- Noise decomposition uses least-squares with robust fallback
- Plots generated when `disp=1` show error vs. phase/code and RMS profiles

## See Also

- `sineFit` - Sine wave fitting (used internally)
- `errPDF` - Error probability distribution
- `errAutoCorrelation` - Error autocorrelation analysis
