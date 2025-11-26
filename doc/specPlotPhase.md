# specPlotPhase

## Overview

`specPlotPhase` displays FFT spectrum in polar coordinates, showing both magnitude and phase. Uses coherent averaging to preserve phase information across harmonics.

## Syntax

```matlab
[h, spec, phi, bin] = specPlotPhase(data)
[h, spec, phi, bin] = specPlotPhase(data, Name, Value)
```

## Input Arguments

- **`data`** — ADC output signal (1×N or N×M for averaging)
- **`N_fft`** — FFT length, default: `length(data)`
- **`harmonic`** — Number of harmonics to annotate, default: `5`
- **`OSR`** — Oversampling ratio, default: `1`

## Output Arguments

- **`h`** — Polar plot handle
- **`spec`** — Complex spectrum (magnitude + phase)
- **`phi`** — Phase reference (fundamental phasor)
- **`bin`** — Fundamental bin index

## Algorithm

### 1. Phase-Coherent Averaging

For each run:
```
1. FFT: tspec = fft(data - mean(data))
2. Find fundamental: bin = argmax(|tspec[1 : N/2/OSR]|)
3. Extract phase: phi = tspec(bin) / |tspec(bin)|
4. Align harmonics: tspec(k×bin) × conj(phi)^k
5. Align non-harmonics: tspec(f) × conj(phi)^(f/bin)
6. Accumulate: spec += tspec
```

This preserves phase relationships across multiple acquisitions.

### 2. Polar Display

- **Radial axis**: Magnitude in dBFS (logarithmic)
- **Angular axis**: Phase in radians
- **Fundamental**: Red circle with red line from origin
- **Harmonics 2:N**: Blue squares with blue lines

## Examples

### Example 1: Phase Analysis

```matlab
data = adc_output;
[h, spec, phi, bin] = specPlotPhase(data, 'harmonic', 10);
title('ADC Spectrum with Phase');
```

### Example 2: Multi-Run Averaging

```matlab
runs = [run1; run2; run3];  % 3×N matrix
[h, spec] = specPlotPhase(runs);
% Phase-coherent averaging preserves harmonic relationships
```

## Interpretation

| Observation | Meaning |
|-------------|---------|
| Harmonics aligned radially | In-phase distortion (even-order symmetry) |
| Harmonics at 90° offset | Quadrature distortion |
| Random phase scatter | Uncorrelated noise/spurs |
| Phase drift with run | Clock frequency instability |

### Harmonic Phase Patterns

- **All harmonics at 0°**: Clipping or saturation
- **Alternating 0°/180°**: Odd-harmonic distortion
- **Progressive phase shift**: Group delay or filter response

## Comparison: specPlot vs specPlotPhase

| Feature | specPlot | specPlotPhase |
|---------|----------|---------------|
| **Display** | Magnitude-only (Cartesian) | Magnitude + Phase (Polar) |
| **Averaging** | Power averaging | Coherent (phase-preserving) |
| **Use case** | Metrics (SNDR, SFDR) | Phase analysis, jitter |
| **Harmonic info** | Magnitude only | Full phasor |

## Limitations

- **Phase ambiguity**: 2π wrapping may obscure trends
- **No metrics**: Doesn't compute SNDR/SFDR (use `specPlot` for that)
- **Coherent requirement**: Best with phase-locked acquisitions
- **Polar visualization**: Harder to read exact magnitudes vs Cartesian

## See Also

- [`specPlot`](specPlot.md) — Standard magnitude spectrum with metrics
- [`FGCalSine`](FGCalSine.md) — Calibration preserving phase info
- [`errHistSine`](errHistSine.md) — Phase-domain error analysis

## References

1. IEEE Std 1241-2010, "IEEE Standard for Terminology and Test Methods for ADCs"
