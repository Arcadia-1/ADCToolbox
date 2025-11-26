# errAutoCorrelation

## Overview

`errAutoCorrelation` computes and plots the autocorrelation function (ACF) of ADC error signals to detect temporal patterns, periodicity, and noise color.

## Syntax

```matlab
[acf, lags] = errAutoCorrelation(err_data)
[acf, lags] = errAutoCorrelation(err_data, Name, Value)
```

## Input Arguments

- **`err_data`** — Error samples (1×N or N×1 vector)
- **`MaxLag`** — Maximum lag to compute, default: `100`
- **`Normalize`** — Normalize ACF so ACF(0) = 1, default: `true`

## Output Arguments

- **`acf`** — Autocorrelation function (1×(2×MaxLag+1) vector)
- **`lags`** — Lag indices: `-MaxLag : MaxLag`

## Algorithm

### 1. Mean Removal

```
e = err_data - mean(err_data)
```

### 2. Autocorrelation Calculation

For each lag `τ`:
```
ACF(τ) = E[e(t) × e(t + τ)]
       ≈ (1/(N - |τ|)) × Σ e(i) × e(i + τ)
```

### 3. Normalization (if enabled)

```
ACF_norm(τ) = ACF(τ) / ACF(0)
```

This scales ACF(0) = 1 (variance).

## Examples

### Example 1: White Noise Check

```matlab
[~, ~, ~, ~, err] = FGCalSine(bits);
[acf, lags] = errAutoCorrelation(err, 'MaxLag', 200);

if max(abs(acf(lags ~= 0))) < 0.1
    disp('Error appears to be white noise');
else
    disp('Correlated error detected');
end
```

### Example 2: Detect Periodic Patterns

```matlab
[acf, lags] = errAutoCorrelation(err, 'MaxLag', 500, 'Normalize', true);
[peaks, locs] = findpeaks(acf(lags > 0));
if ~isempty(peaks)
    fprintf('Periodic pattern detected at lag: %d samples\n', locs(1));
end
```

## Interpretation

### ACF Patterns

| Pattern | Interpretation |
|---------|----------------|
| **ACF ≈ δ(τ)** (spike at τ=0 only) | White noise (ideal) |
| **Exponential decay** | Colored noise (1/f, thermal settling) |
| **Sinusoidal oscillation** | Periodic interference (clock coupling, aliasing) |
| **Slow decay** | Low-frequency drift or flicker noise |
| **Multiple peaks** | Multi-tone interference or harmonic patterns |

### Quantitative Metrics

**Decorrelation time**: Smallest `τ` where `|ACF(τ)| < 0.1`
- Short decorrelation → white noise
- Long decorrelation → correlated (memory effect)

**Effective noise bandwidth**:
```
BW_eff = 1 / (2 × Σ ACF(τ))
```

### Noise Classification

| ACF Decay | Noise Type |
|-----------|------------|
| ACF(1) < 0.1 | White noise |
| 0.1 ≤ ACF(1) < 0.5 | Weakly correlated |
| 0.5 ≤ ACF(1) < 0.9 | Strongly correlated (1/f noise) |
| ACF(1) > 0.9 | Deterministic pattern or drift |

## Limitations

- **Stationarity assumption**: Assumes error statistics don't change over time
- **Linear correlation only**: Doesn't capture nonlinear dependencies
- **Memory requirements**: ACF length = 2×MaxLag+1 → use reasonable MaxLag
- **No frequency info**: Use `errEnvelopeSpectrum` for frequency-domain view

## See Also

- [`errEnvelopeSpectrum`](errEnvelopeSpectrum.md) — Frequency-domain error analysis (dual of ACF)
- [`errPDF`](errPDF.md) — Error distribution analysis
- [`errHistSine`](errHistSine.md) — Phase/code-dependent error patterns

## References

1. Box, Jenkins, Reinsel, "Time Series Analysis: Forecasting and Control," Wiley 2015.
2. Oppenheim, Schafer, "Discrete-Time Signal Processing," Prentice Hall 2009.
