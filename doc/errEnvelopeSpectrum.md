# errEnvelopeSpectrum

## Overview

`errEnvelopeSpectrum` computes and plots the frequency spectrum of the error signal's envelope using Hilbert transform. Detects amplitude modulation, burst errors, and low-frequency drift.

## Syntax

```matlab
errEnvelopeSpectrum(err_data)
errEnvelopeSpectrum(err_data, Name, Value)
```

## Input Arguments

- **`err_data`** — Error samples (1×N or N×1 vector)
- **`Fs`** — Sampling frequency (Hz), default: `1`

## Algorithm

### 1. Hilbert Transform

Computes analytic signal:
```
z(t) = e(t) + j × H[e(t)]
```
where `H[·]` is the Hilbert transform.

### 2. Envelope Extraction

```
envelope(t) = |z(t)| = sqrt(e²(t) + H[e(t)]²)
```

This extracts the instantaneous amplitude.

### 3. Spectrum Computation

Calls `specPlot(envelope, 'Fs', Fs)` to compute and display the envelope power spectrum.

## Examples

### Example 1: Basic Usage

```matlab
[~, ~, ~, ~, err] = FGCalSine(bits);
figure;
errEnvelopeSpectrum(err, 'Fs', 1e9);  % 1 GHz sampling
title('Error Envelope Spectrum');
```

### Example 2: Detect Burst Errors

```matlab
errEnvelopeSpectrum(err);
% Peaks in envelope spectrum indicate periodic amplitude modulation
```

## Interpretation

### Spectral Features

| Observation | Cause |
|-------------|-------|
| **DC component only** | Constant envelope (white noise) |
| **Low-frequency peak** | Flicker noise or temperature drift |
| **Peak at f_mod** | Amplitude modulation at f_mod |
| **Harmonic peaks** | Periodic bursts or switching artifacts |
| **Broadband hump** | Intermittent errors or metastability |

### Comparison: Time vs Frequency

| Domain | Tool | Best For |
|--------|------|----------|
| **Time** | `errAutoCorrelation` | Detect periodicity, decorrelation time |
| **Frequency** | `errEnvelopeSpectrum` | Identify modulation frequencies, drift rates |

Both are Fourier duals: ACF ↔ Power Spectral Density.

### Common Patterns

**Pattern 1**: Sharp peak at low frequency
→ Thermal drift, supply ripple, or flicker noise

**Pattern 2**: Peak at Fs/2
→ Aliased interference from above Nyquist

**Pattern 3**: Multiple harmonics
→ Clock coupling or switching noise modulating the error

## Use Cases

1. **Flicker Noise**: Low-frequency (<1 kHz) envelope modulation
2. **Supply Noise**: Peaks at power supply ripple frequency (e.g., 60 Hz, 100 Hz)
3. **Clock Coupling**: Envelope modulated at clock frequency or harmonics
4. **Burst Errors**: Intermittent glitches appear as broadband envelope energy

## Limitations

- **Phase information lost**: Envelope is magnitude-only
- **No causality**: Hilbert transform requires full signal (non-causal)
- **Edge effects**: Boundary artifacts if signal not periodic
- **Resolution**: Frequency resolution limited by signal length

## See Also

- [`errAutoCorrelation`](errAutoCorrelation.md) — Time-domain correlation (ACF ↔ envelope spectrum)
- [`errPDF`](errPDF.md) — Amplitude distribution
- [`specPlot`](specPlot.md) — Direct error spectrum (not envelope)

## References

1. S. L. Marple, "Digital Spectral Analysis," Prentice Hall 1987.
2. A. V. Oppenheim, R. W. Schafer, "Discrete-Time Signal Processing," 3rd ed.
