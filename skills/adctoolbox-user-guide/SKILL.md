---
name: adctoolbox-user-guide
description: >
  Guide for writing correct ADCToolbox (`adctoolbox`) Python code — choosing the right
  functions, parameters, and interpreting dictionary results. Covers ADC testing, spectrum
  analysis, FFT, ENOB, SNDR, SFDR, SNR, THD, NSD, INL/DNL, harmonic decomposition,
  two-tone IMD (future), signal generation with jitter/noise, SAR ADC calibration, bit activity,
  noise transfer functions, coherent sampling, unit conversions, mixed-signal test, and
  data converter evaluation. Use this skill whenever the user asks to write code using
  adctoolbox, analyze ADC data, generate ADC test signals, calibrate an ADC, compute
  figures of merit, or anything involving the adctoolbox library — even if they don't
  mention it by name. Also trigger when the user mentions specific metric names (ENOB,
  SNDR, SFDR, THD, IMD, NSD) in a Python coding context.
---

# ADCToolbox Code Guide

This skill teaches how to write correct `adctoolbox` Python code. ADCToolbox is a
library for ADC characterization, calibration, and visualization.

For the full function reference, read `references/api-quickref.md`.
For runnable end-to-end examples, read `references/workflow-recipes.md`.

---

## 1. Quick Start Pattern

### Imports

All public functions are flat-exported — import directly from `adctoolbox`:

```python
from adctoolbox import analyze_spectrum, find_coherent_frequency, fit_sine_4param
```

The signal generator class is the one exception — use a submodule import:

```python
from adctoolbox.siggen import ADC_Signal_Generator
```

### Return Convention

Almost all analysis functions return **dictionaries**, not tuples. Access results by key:

```python
result = analyze_spectrum(data, fs=800e6, create_plot=False)
enob = result['enob']
sndr = result['sndr_dbc']
```

Exceptions that return arrays/tuples:
- `find_coherent_frequency` → `(fin_actual, best_bin)` tuple
- `analyze_bit_activity` → ndarray
- `analyze_overflow` → 4-element tuple of arrays
- `analyze_weight_radix` → ndarray
- `analyze_enob_sweep` → `(enob_sweep, n_bits_vec)` tuple
- `fit_static_nonlin` → `(k2, k3, fitted_sine, fitted_transfer)` tuple

### Plotting Control

Every plotting function supports these patterns:

```python
# Suppress all plots (computation only)
result = analyze_spectrum(data, fs=Fs, create_plot=False)

# Plot into an existing subplot axis
fig, axes = plt.subplots(1, 2)
result = analyze_spectrum(data, fs=Fs, create_plot=True, ax=axes[0])

# Let the function create its own figure (default)
result = analyze_spectrum(data, fs=Fs)
```

For headless/CI execution, set the environment variable:
```bash
MPLBACKEND=Agg python script.py
```

---

## 2. Coherent Sampling (Critical First Step)

Spectral leakage destroys measurement accuracy. Always use coherent sampling when
generating test signals or analyzing known-frequency inputs.

### The Rule

The input frequency must satisfy: **Fin = J × Fs / N**, where J is an integer
coprime to N, and ideally odd.

### Using `find_coherent_frequency`

```python
from adctoolbox import find_coherent_frequency

Fs = 800e6       # Sampling frequency
fin_target = 100e6  # Desired input frequency
N = 8192         # FFT size / number of samples

Fin, bin_idx = find_coherent_frequency(Fs, fin_target, N)
# Fin is the exact coherent frequency (very close to fin_target)
# bin_idx is the FFT bin where the signal peak appears
```

Parameters:
- `force_odd=True` (default): only odd bin indices — standard practice
- `search_radius=200`: how far to search from ideal bin

### When You Cannot Use Coherent Sampling

If the data comes from a real measurement at a non-coherent frequency, apply
windowing instead:

```python
result = analyze_spectrum(data, fs=Fs, win_type='hann')  # default
```

Available windows: `'hann'`, `'hamming'`, `'blackman'`, `'boxcar'` (no window).
Use `'boxcar'` only when the signal is already coherently sampled.

### Undersampling

`find_coherent_frequency` supports Fin > Fs/2 for undersampling applications.
Use `fold_frequency_to_nyquist` to find the aliased frequency:

```python
from adctoolbox import fold_frequency_to_nyquist
f_alias = fold_frequency_to_nyquist(900e6, 800e6)  # → 100 MHz
```

---

## 3. Core Workflows

### 3.1 Single-Tone Spectrum Analysis

The primary measurement workflow. `analyze_spectrum` computes ENOB, SNDR, SFDR,
SNR, THD, noise floor, and NSD in one call.

```python
from adctoolbox import analyze_spectrum

result = analyze_spectrum(
    data,                    # (N,) or (M, N) array
    fs=800e6,                # Sampling frequency (Hz)
    max_scale_range=None,    # Full scale: scalar, [min,max], or None (auto)
    win_type='hann',         # Window function
    max_harmonic=5,          # Harmonics for THD calculation
    osr=1,                   # Oversampling ratio
    create_plot=True,        # Generate plot
    ax=None                  # Matplotlib axes (None = new figure)
)
```

Key return dict keys:
- `enob` — Effective Number of Bits
- `sndr_dbc` — Signal-to-Noise-and-Distortion Ratio (dBc)
- `sfdr_dbc` — Spurious-Free Dynamic Range (dBc)
- `snr_dbc` — Signal-to-Noise Ratio (dBc)
- `thd_dbc` — Total Harmonic Distortion (dBc)
- `nsd_dbfs_hz` — Noise Spectral Density (dBFS/Hz)
- `sig_pwr_dbfs` — Signal power (dBFS)
- `noise_floor_dbfs` — Noise floor level (dBFS)

All metrics are in **dBc** (relative to carrier/signal), not dBFS.

### 3.2 Analog Error Analysis Pipeline

The typical flow: sine fit → error decomposition → detailed error analysis.

```python
from adctoolbox import (fit_sine_4param, analyze_decomposition_time,
                         analyze_error_by_phase, analyze_error_pdf,
                         analyze_error_spectrum, analyze_inl_from_sine)

# Step 1: Sine fit to extract residual error
fit = fit_sine_4param(data)
# fit['frequency'] is normalized (0 to 0.5), NOT Hz
# fit['residuals'] is the error signal

# Step 2: Harmonic decomposition
decomp = analyze_decomposition_time(data, harmonic=7, create_plot=False)
# decomp['magnitudes_db'] — per-harmonic magnitudes in dB

# Step 3: Phase-domain error separation (AM vs PM noise)
phase = analyze_error_by_phase(data, create_plot=False)
# phase['am_noise_rms_v'], phase['pm_noise_rms_v']

# Step 4: Error PDF (Gaussian-ness check)
pdf = analyze_error_pdf(data, resolution=12, create_plot=False)
# pdf['kl_divergence'] — lower means more Gaussian

# Step 5: Error spectrum (frequency content of error)
err_spec = analyze_error_spectrum(data, fs=800e6, create_plot=False)

# Step 6: INL/DNL from sine wave
inl_result = analyze_inl_from_sine(data, num_bits=12, create_plot=False)
# inl_result['inl'], inl_result['dnl'], inl_result['code']
```

### 3.3 Digital Analysis & Calibration

For SAR ADC bit-level analysis and foreground calibration:

```python
from adctoolbox import (calibrate_weight_sine, analyze_bit_activity,
                         analyze_weight_radix, analyze_enob_sweep)

# bit_matrix: shape (N_samples, N_bits) — raw bit decisions

# Step 1: Check bit activity (should be ~50% per bit)
activity = analyze_bit_activity(bit_matrix, create_plot=False)

# Step 2: Calibrate weights using sine fitting
# IMPORTANT: freq is normalized Fin/Fs, NOT Hz
cal = calibrate_weight_sine(bit_matrix, freq=Fin/Fs)
calibrated_signal = cal['calibrated_signal']
weights = cal['weight']

# Step 3: Visualize weight radix
radix = analyze_weight_radix(weights, create_plot=True)

# Step 4: ENOB sweep — how ENOB changes vs number of bits used
enob_sweep, n_bits_vec = analyze_enob_sweep(bit_matrix, freq=Fin/Fs)
```

### 3.4 Signal Generation

`ADC_Signal_Generator` creates test signals with configurable non-idealities
using the Applier Pattern (method chaining):

```python
from adctoolbox.siggen import ADC_Signal_Generator

gen = ADC_Signal_Generator(N=8192, Fs=800e6, Fin=Fin, A=0.49, DC=0.5)

# Option A: Single effect from clean signal
sig = gen.apply_thermal_noise(noise_rms=100e-6)

# Option B: Chain multiple effects
sig = gen.get_clean_signal()
sig = gen.apply_static_nonlinearity(input_signal=sig, k2=0.02, k3=0.01)
sig = gen.apply_thermal_noise(input_signal=sig, noise_rms=50e-6)
sig = gen.apply_jitter(input_signal=sig, jitter_rms=1e-12)
sig = gen.apply_quantization_noise(input_signal=sig, n_bits=12, quant_range=(0.0, 1.0))
```

Each `apply_*()` method:
- With `input_signal=None` (default): starts from the internal clean sine wave
- With `input_signal=<array>`: applies the effect on top of the given signal

Available effects: thermal noise, quantization, jitter, static nonlinearity
(by k-coefficients or by HD levels in dBc), memory effect, incomplete sampling,
residue amplifier gain error (static and dynamic), reference error, AM noise,
AM tone, clipping, drift, glitch, noise shaping (delta-sigma orders 1–5).

---

## 4. Unit Conversions

All conversion functions are flat-imported from `adctoolbox`:

| Conversion | Function | Notes |
|------------|----------|-------|
| dB ↔ magnitude | `db_to_mag(db)`, `mag_to_db(mag)` | Voltage/amplitude domain (20·log10) |
| dB ↔ power | `db_to_power(db)`, `power_to_db(power)` | Power domain (10·log10) |
| SNR ↔ ENOB | `snr_to_enob(snr_db)`, `enob_to_snr(enob)` | Standard formula: (SNR−1.76)/6.02 |
| SNR ↔ NSD | `snr_to_nsd(snr, fs, ...)`, `nsd_to_snr(nsd, fs, ...)` | Needs fs; supports osr parameter |
| LSB ↔ Volts | `lsb_to_volts(lsb, vref, n_bits)`, `volts_to_lsb(v, vref, n_bits)` | Requires Vref and bit count |
| dBm ↔ Vrms | `dbm_to_vrms(dbm, z_load=50)`, `vrms_to_dbm(vrms, z_load=50)` | Default 50Ω load |
| dBm ↔ mW | `dbm_to_mw(dbm)`, `mw_to_dbm(mw)` | Direct power conversion |
| Bin ↔ Freq | `bin_to_freq(bin, fs, nfft)`, `freq_to_bin(freq, fs, nfft)` | FFT bin to/from Hz |
| Amplitudes → SNR | `amplitudes_to_snr(sig_amp, noise_amp)` | From two amplitude values |
| Amplitude → Power | `sine_amplitude_to_power(amp, z_load=50)` | Peak amplitude to watts |

---

## 5. Figures of Merit

```python
from adctoolbox import (calculate_walden_fom, calculate_schreier_fom,
                         calculate_thermal_noise_limit, calculate_jitter_limit)

# Walden FoM (lower is better) — for medium-resolution ADCs
fom_w = calculate_walden_fom(power=1e-3, fs=100e6, enob=10.5)
# Returns J/conv-step

# Schreier FoM (higher is better) — for high-resolution / ΔΣ ADCs
fom_s = calculate_schreier_fom(power=1e-3, sndr_db=65, bw=50e6)
# Returns dB

# Theoretical limits
snr_thermal = calculate_thermal_noise_limit(cap_pf=2.0, v_fs=1.0)
snr_jitter = calculate_jitter_limit(freq=100e6, jitter_rms_sec=0.5e-12)
```

---

## 6. Common Pitfalls

### Coherent Sampling

Not using coherent sampling is the #1 source of incorrect results. If you skip
`find_coherent_frequency`, you must use windowing (`win_type='hann'`), but this
still sacrifices ~1.5 dB of SNR accuracy compared to coherent sampling. The
default window is already `'hann'`, so basic usage is safe — but for precise
measurements, always compute a coherent frequency first.

### `create_plot` in Loops

When calling analysis functions in a loop, always pass `create_plot=False` or
provide an explicit `ax=` parameter. Otherwise each call creates a new figure,
consuming memory and potentially crashing in headless environments.

```python
# BAD: creates N figures
for sig in signals:
    analyze_spectrum(sig, fs=Fs)  # new figure each time

# GOOD: plot into subplots
fig, axes = plt.subplots(1, len(signals))
for i, sig in enumerate(signals):
    analyze_spectrum(sig, fs=Fs, ax=axes[i])
```

### dBc vs dBFS

`analyze_spectrum` returns metrics in **dBc** (relative to carrier power):
`sndr_dbc`, `sfdr_dbc`, `snr_dbc`, `thd_dbc`. The noise floor and signal
power are in **dBFS**: `noise_floor_dbfs`, `sig_pwr_dbfs`. Do not confuse
these reference levels.

### Normalized Frequency in Calibration

`calibrate_weight_sine` expects `freq` as **Fin/Fs** (normalized, range 0–0.5),
not in Hz. This is different from `analyze_spectrum` which takes `fs` in Hz.

```python
# WRONG
calibrate_weight_sine(bits, freq=100e6)

# CORRECT
calibrate_weight_sine(bits, freq=Fin/Fs)    # e.g., 0.125
calibrate_weight_sine(bits, freq=None)       # auto-detect (safe default)
```

### `fit_sine_4param` Returns Normalized Frequency

The `frequency` key in the return dict of `fit_sine_4param` is normalized
(0 to 0.5), not in Hz. Multiply by Fs to get physical frequency:

```python
result = fit_sine_4param(data)
freq_hz = result['frequency'] * Fs
```

Or use the convenience wrapper:
```python
freq_hz = estimate_frequency(data, fs=Fs)
```

### Signal Generator Amplitude Convention

`ADC_Signal_Generator` uses amplitude + DC offset for a signal ranging over
[DC−A, DC+A]. For a typical 0-to-1 range ADC: A=0.49, DC=0.5 gives a signal
from 0.01 to 0.99 (near full-scale without clipping).
