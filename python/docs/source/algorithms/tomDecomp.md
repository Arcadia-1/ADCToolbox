# tomDecomp

## Overview

`tomDecomp` performs Thompson time-domain error decomposition, separating ADC errors into signal-dependent (harmonics) and signal-independent (noise) components.

## Syntax

```matlab
[signal, error, indep, dep, phi] = tomDecomp(data)
[signal, error, indep, dep, phi] = tomDecomp(data, re_fin)
[signal, error, indep, dep, phi] = tomDecomp(data, re_fin, order)
[signal, error, indep, dep, phi] = tomDecomp(data, re_fin, order, disp)
```

## Input Arguments

- **`data`** — ADC output signal (1×N or N×1 vector)
- **`re_fin`** — Relative frequency (Fin/Fs), optional
  - If omitted or `NaN`: auto-detect using `findFin(data)`
- **`order`** — Harmonic order for dependent error, default: `10`
  - Order 1: fundamental only
  - Order N: include harmonics 2 through N
- **`disp`** — Display plot (0/1), default: `1`

## Output Arguments

- **`signal`** — Reconstructed signal: `DC + fundamental`
- **`error`** — Total error: `data - signal`
- **`indep`** — Independent error (noise): `data - (DC + all harmonics)`
- **`dep`** — Dependent error (distortion): `signal_all - signal`
- **`phi`** — Fundamental phase (radians): `-atan2(WQ, WI)`

## Algorithm

### 1. Fundamental Extraction

Uses quadrature demodulation:
```
SI = cos(2π × freq × (0:N-1))
SQ = sin(2π × freq × (0:N-1))
WI = 2 × mean(data .* SI)  % In-phase weight
WQ = 2 × mean(data .* SQ)  % Quadrature weight
DC = mean(data)
signal = DC + WI × SI + WQ × SQ
```

**Phase**: `phi = -atan2(WQ, WI)` (negative for lag convention)

### 2. Harmonic Fitting

For harmonics `k = 1:order`:
```
Basis: [cos(kθ), sin(kθ)] for k = 1, 2, ..., order
Solve: [SI_all, SQ_all] × W = data
signal_all = DC + [SI_all, SQ_all] × W
```

where `θ = 2π × freq × (0:N-1)`.

### 3. Error Decomposition

```
Total error:       error = data - signal
Dependent error:   dep   = signal_all - signal  (harmonics 2:order)
Independent error: indep = data - signal_all    (residual noise)
```

**Interpretation**:
- `dep`: Signal-dependent distortion (harmonics)
- `indep`: Signal-independent noise (thermal, quantization, jitter)

## Examples

### Example 1: Basic Decomposition

```matlab
data = calibrated_output;
[sig, err, indep, dep, phi] = tomDecomp(data);

fprintf('Fundamental phase: %.2f°\n', rad2deg(phi));
fprintf('Total error RMS: %.4f\n', rms(err));
fprintf('Independent (noise) RMS: %.4f\n', rms(indep));
fprintf('Dependent (distortion) RMS: %.4f\n', rms(dep));
```

### Example 2: High-Order Harmonics

```matlab
[~, ~, indep, dep] = tomDecomp(data, 0.1234, 20, 0);  % 20 harmonics, no plot
SNR = 20*log10(rms(sig) / rms(indep));
THD = 20*log10(rms(dep) / rms(sig));
fprintf('SNR: %.2f dB, THD: %.2f dB\n', SNR, THD);
```

### Example 3: Auto-Frequency Detection

```matlab
[sig, ~, indep, dep] = tomDecomp(data, NaN, 10);  % Auto-detect frequency
```

## Interpretation

### Error Contribution Analysis

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Total Error** | RMS(error) | Overall inaccuracy |
| **Noise** | RMS(indep) | Signal-independent component |
| **Distortion** | RMS(dep) | Signal-dependent harmonics |
| **SNR** | `20*log10(RMS(signal) / RMS(indep))` | Noise-limited performance |
| **THD** | `20*log10(RMS(dep) / RMS(signal))` | Harmonic distortion |
| **SNDR** | `20*log10(RMS(signal) / RMS(error))` | Combined metric |

### Decomposition Validation

```
Verify: error ≈ indep + dep
Check: RMS(error)² ≈ RMS(indep)² + RMS(dep)²  (orthogonal components)
```

### Common Patterns

| Observation | Interpretation |
|-------------|----------------|
| `RMS(indep) >> RMS(dep)` | Noise-dominated ADC |
| `RMS(dep) >> RMS(indep)` | Distortion-dominated ADC |
| `dep` shows periodic pattern | Harmonic distortion from nonlinearity |
| `indep` shows white noise | Thermal or quantization noise |
| `indep` shows colored noise | Insufficient `order` (harmonics leaking into noise) |

## Limitations

- **Sinewave input only**: Assumes single-tone input
- **Order selection**: Too low → harmonics leak into `indep`; too high → noise absorbed into `dep`
- **Stationary assumption**: Assumes time-invariant ADC characteristics
- **No phase noise**: Phase jitter appears in `indep` (cannot separate from amplitude noise)

## See Also

- [`errHistSine`](errHistSine.md) — Separate amplitude vs phase noise
- [`sineFit`](sineFit.md) — Sinewave fitting without harmonics
- [`specPlot`](specPlot.md) — Frequency-domain analysis

## References

1. S. Thompson, "Time-domain error decomposition for ADC testing," IEEE Trans. Instrumentation and Measurement, 2005.
2. IEEE Std 1057-2017, "IEEE Standard for Digitizing Waveform Recorders."
