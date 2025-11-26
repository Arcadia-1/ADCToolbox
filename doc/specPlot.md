# specPlot

## Overview

`specPlot` computes FFT-based spectrum analysis with windowing, extracts key ADC performance metrics (ENoB, SNDR, SFDR, THD, SNR, NF), and generates annotated frequency-domain plots.

## Syntax

```matlab
[ENoB, SNDR, SFDR, SNR, THD, pwr, NF, h] = specPlot(data)
[ENoB, SNDR, SFDR, SNR, THD, pwr, NF, h] = specPlot(data, Name, Value)
```

## Input Arguments

- **`data`** — ADC output signal (1×N or N×M for averaging)
- **`Fs`** — Sampling frequency (Hz), default: `1`
- **`maxCode`** — Full-scale range, default: `max(data) - min(data)`
- **`harmonic`** — Number of harmonics to annotate, default: `5`
- **`OSR`** — Oversampling ratio, default: `1`
- **`winType`** — Window function handle, default: `@hann`
- **`sideBin`** — Bins around fundamental to include in signal power, default: `1`
- **`label`** — Show annotations (0/1), default: `1`
- **`nTHD`** — Harmonics for THD calculation, default: `5`
- **`NFMethod`** — Noise floor method (0: median, 1: trimmed mean, 2: exclude THD), default: `0`
- **`noFlicker`** — Suppress DC to this frequency (Hz), default: `0`

## Output Arguments

- **`ENoB`** — Effective number of bits: `(SNDR - 1.76) / 6.02`
- **`SNDR`** — Signal-to-noise+distortion ratio (dB)
- **`SFDR`** — Spurious-free dynamic range (dB)
- **`SNR`** — Signal-to-noise ratio (dB)
- **`THD`** — Total harmonic distortion (dB)
- **`pwr`** — Signal power (dBFS)
- **`NF`** — Noise floor (dB): `SNR - pwr`
- **`h`** — Plot handle (empty if `isPlot = 0`)

## Algorithm

### 1. Windowing & FFT

```
For each run in data:
    1. Normalize: data ← data / maxCode
    2. Remove DC: data ← data - mean(data)
    3. Apply window: data ← data × window / RMS(window)
    4. FFT: spec ← |FFT(data)|² (power spectrum)
    5. Average: spec_avg ← mean(spec across runs)
```

Window preserves power: `mean(win^2) = 1` after normalization.

### 2. Fundamental Detection

**Bin refinement** via 3-point parabolic interpolation:
```
bin = argmax(spec in-band)
sig_log = [log10(spec(bin-1)), log10(spec(bin)), log10(spec(bin+1))]
bin_refined = bin + (sig_r - sig_l) / (2*sig_c - sig_l - sig_r) / 2
```

**Signal power**: Sum `spec[bin - sideBin : bin + sideBin]`

### 3. Metric Calculations

| Metric | Formula |
|--------|---------|
| **SNDR** | `10*log10(signal_power / (noise + distortion))` |
| **SFDR** | `10*log10(signal_peak / max_spur)` where max_spur excludes fund & harmonics |
| **THD** | `10*log10(sum(harmonics 2:nTHD) / signal_peak)` |
| **SNR** | `10*log10(signal_power / noise_floor)` |
| **ENoB** | `(SNDR - 1.76) / 6.02` |
| **NF** | `SNR - pwr` (noise floor relative to 0 dBFS) |

### 4. Noise Floor Estimation

| Method | Description |
|--------|-------------|
| **0** (default) | `median(spec) / sqrt((1 - 2/(9*N_runs))^3) × N_bins` (assumes Gaussian) |
| **1** | Trimmed mean: `mean(spec[5% : 95%]) × N_bins` (robust to outliers) |
| **2** | Exclude harmonics: sum all bins except fundamental and harmonics 2:nTHD |

### 5. Harmonic Aliasing

For each harmonic `k = 2:harmonic`:
```
bin_harmonic = alias(round(bin_fund × k), N_fft)
```
where `alias(b, N) = b mod N` with Nyquist folding.

## Examples

### Example 1: Basic Usage

```matlab
data = readmatrix('adc_output.csv');
[ENoB, SNDR, SFDR] = specPlot(data, 'Fs', 1e9, 'harmonic', 5);
fprintf('ENoB: %.2f, SNDR: %.2f dB, SFDR: %.2f dB\n', ENoB, SNDR, SFDR);
```

### Example 2: Oversampled ADC

```matlab
[ENoB, ~, ~, SNR, ~, ~, NF] = specPlot(data, 'OSR', 32, 'winType', @blackman);
fprintf('OSR=32: ENoB=%.2f, NF=%.2f dB\n', ENoB, NF);
```

### Example 3: Multiple Runs with Coherent Averaging

```matlab
data_runs = [run1; run2; run3];  % 3×N matrix
[ENoB, SNDR] = specPlot(data_runs, 'coAvg', 1);  % Phase-aligned averaging
```

## Interpretation

### Metric Guidelines

| ENoB Range | ADC Quality |
|------------|-------------|
| ENoB ≈ N | Ideal N-bit performance |
| ENoB < N-2 | Significant noise/distortion, needs calibration |
| SFDR > 80 dB | Excellent linearity |
| THD < -60 dB | Low harmonic distortion |

### Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Low SNDR, high SFDR | Noise-dominated | Check thermal/quantization noise |
| Low SFDR, high SNR | Distortion-dominated | Investigate harmonics, check linearity |
| Negative ENoB | Clipping or invalid signal | Check `maxCode`, inspect waveform |
| NF >> expected | Insufficient averaging | Increase `N_runs` or check coherence |

## Limitations

- **Coherent sampling**: Best results require `Fin/Fs` to be rational with prime factors of N
- **Windowing loss**: Non-rectangular windows reduce SNR by ~1-2 dB
- **Averaging**: Incoherent averaging (`coAvg=0`) doesn't preserve phase → harmonic artifacts
- **OSR mode**: Assumes in-band is `[0, Fs/(2*OSR)]` → verify with your ADC architecture

## See Also

- [`specPlotPhase`](specPlotPhase.md) — Phase-domain spectrum analysis
- [`FGCalSine`](FGCalSine.md) — Calibration using sinewave input
- [`tomDecomp`](tomDecomp.md) — Time-domain error decomposition
- [`errHistSine`](errHistSine.md) — Error histogram analysis

## References

1. IEEE Std 1241-2010, "IEEE Standard for Terminology and Test Methods for Analog-to-Digital Converters"
2. M. Tian et al., "A Low-Power 12-bit 10-MS/s SAR ADC," JSSC 2019.
