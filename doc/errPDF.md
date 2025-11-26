# errPDF

## Overview

`errPDF` estimates error probability density function (PDF) using kernel density estimation (KDE) and compares it to a Gaussian fit using Kullback-Leibler divergence.

## Syntax

```matlab
[noise_lsb, mu, sigma, KL, x, fx, gauss_pdf] = errPDF(err_data)
[noise_lsb, mu, sigma, KL, x, fx, gauss_pdf] = errPDF(err_data, Name, Value)
```

## Input Arguments

- **`err_data`** — Error samples (1×N or N×1 vector)
- **`Resolution`** — ADC resolution (bits), default: `12`
- **`FullScale`** — ADC full-scale range, default: `1`

## Output Arguments

- **`noise_lsb`** — Error in LSB units: `err_data / LSB`
- **`mu`** — Mean of error distribution (LSB)
- **`sigma`** — Standard deviation (LSB)
- **`KL_divergence`** — Kullback-Leibler divergence: `D_KL(KDE || Gaussian)`
- **`x`** — PDF x-axis (LSB units)
- **`fx`** — Estimated PDF via KDE
- **gauss_pdf`** — Fitted Gaussian PDF

## Algorithm

### 1. Normalize to LSB

```
LSB = FullScale / 2^Resolution
noise_lsb = err_data / LSB
```

### 2. Kernel Density Estimation (KDE)

Uses Gaussian kernel with Silverman's rule-of-thumb bandwidth:
```
h = 1.06 × σ × N^(-1/5)  % Bandwidth
x = linspace(-max(|noise|), max(|noise|), 200)

For each x[i]:
    u = (x[i] - noise_lsb) / h
    fx[i] = mean(exp(-0.5 × u²)) / (h × sqrt(2π))
```

**Bandwidth selection**: Silverman's rule balances bias vs variance for Gaussian-like distributions.

### 3. Gaussian Fitting

```
mu = mean(noise_lsb)
sigma = std(noise_lsb)
gauss_pdf = (1 / (sigma × sqrt(2π))) × exp(-(x - mu)² / (2 × sigma²))
```

### 4. Kullback-Leibler Divergence

Measures how KDE deviates from Gaussian:
```
KL = ∫ fx(x) × log(fx(x) / gauss_pdf(x)) dx
   ≈ Σ fx[i] × log(fx[i] / gauss_pdf[i]) × Δx
```

**Interpretation**:
- KL ≈ 0: Error distribution is Gaussian (thermal/quantization noise)
- KL > 0.01: Non-Gaussian (e.g., spurs, clipping, non-white noise)

## Examples

### Example 1: Basic Error PDF

```matlab
[~, ~, postCal, ideal, err] = FGCalSine(bits);
[~, mu, sigma, KL] = errPDF(postCal - ideal, 'Resolution', 10);
fprintf('Error: μ=%.2f LSB, σ=%.2f LSB, KL=%.4f\n', mu, sigma, KL);
```

### Example 2: Custom Resolution

```matlab
[~, ~, ~, KL] = errPDF(err_data, 'Resolution', 14, 'FullScale', 2);
if KL < 0.01
    disp('Error distribution is approximately Gaussian');
else
    disp('Non-Gaussian error detected');
end
```

## Interpretation

### KL Divergence Thresholds

| KL Value | Interpretation |
|----------|----------------|
| KL < 0.001 | Highly Gaussian (ideal thermal/quantization noise) |
| 0.001 ≤ KL < 0.01 | Approximately Gaussian (acceptable) |
| 0.01 ≤ KL < 0.1 | Moderately non-Gaussian (investigate) |
| KL ≥ 0.1 | Strongly non-Gaussian (clipping, spurs, or systematic errors) |

### Distribution Shapes

| PDF Shape | Cause |
|-----------|-------|
| Single Gaussian peak | Thermal + quantization noise (ideal) |
| Bimodal (two peaks) | Metastability or comparator offset |
| Heavy tails | Occasional large errors (spurs, glitches) |
| Truncated (cut-off) | Clipping or saturation |
| Uniform-like | Quantization-dominated (high-resolution ADC with low noise) |

### Sigma Analysis

```
Theoretical quantization noise: σ_Q = 1/sqrt(12) ≈ 0.29 LSB
If σ >> σ_Q: Thermal/jitter noise dominates
If σ ≈ σ_Q: Quantization-limited ADC
If σ < σ_Q: Under-sampling or correlated noise
```

## Limitations

- **Bandwidth selection**: Silverman's rule optimal for Gaussian-like data; may over-smooth for multi-modal distributions
- **Sample size**: KDE requires `N >> 100` for reliable estimation
- **KL sensitivity**: Sensitive to tail behavior → outliers can inflate KL
- **No time information**: PDF discards temporal structure (use `errAutoCorrelation` for time-domain analysis)

## See Also

- [`errAutoCorrelation`](errAutoCorrelation.md) — Temporal correlation analysis
- [`errHistSine`](errHistSine.md) — Error histogram by phase or code
- [`tomDecomp`](tomDecomp.md) — Decompose signal-dependent vs independent error

## References

1. B. W. Silverman, "Density Estimation for Statistics and Data Analysis," Chapman & Hall, 1986.
2. S. Kullback, R. A. Leibler, "On Information and Sufficiency," Annals of Math. Statistics, 1951.
