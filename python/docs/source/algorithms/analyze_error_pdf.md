# analyze_error_pdf

## Overview

`analyze_error_pdf` analyzes the probability distribution (PDF) of ADC errors by comparing the input signal against a fitted sine wave. This reveals error characteristics: Gaussian noise, uniform quantization, non-Gaussian distortion, etc.

## Syntax

```python
from adctoolbox import analyze_error_pdf

# Basic usage
result = analyze_error_pdf(signal, resolution=12, show_plot=True)

# With custom parameters
result = analyze_error_pdf(signal, resolution=12, bins=100, show_plot=True)
```

## Parameters

- **`signal`** (array_like) — Input ADC signal (sine wave excitation)
- **`resolution`** (int, optional) — ADC resolution in bits
- **`bins`** (int, default=auto) — Number of histogram bins
- **`show_plot`** (bool, default=False) — Display PDF plot with reference distributions
- **`ax`** (matplotlib axis, optional) — Axis for plotting

## Returns

Dictionary containing:
- **`error`** — Error signal (data - fitted_sine)
- **`sigma`** — Standard deviation of error (LSB)
- **`histogram_counts`** — PDF histogram counts
- **`histogram_edges`** — PDF bin edges
- **`kl_divergence`** — KL divergence from Gaussian (measure of non-Gaussianity)

## Algorithm

### 1. Fit Sine Wave

```python
from adctoolbox import fit_sine_4param
result = fit_sine_4param(signal)
fitted_sine = result['fitted_signal']
```

### 2. Compute Error

```python
error = signal - fitted_sine
error_lsb = error * 2**resolution  # Convert to LSB units
```

### 3. Build Histogram

```python
counts, edges = np.histogram(error_lsb, bins=bins, density=True)
```

### 4. Compare to References

Compare measured PDF against:
- **Gaussian**: For thermal noise
- **Uniform**: For quantization noise
- **Other**: For specific distortions

## Examples

### Example 1: Error PDF Analysis

```python
import numpy as np
from adctoolbox import analyze_error_pdf

# Analyze 12-bit ADC with noise
result = analyze_error_pdf(adc_data, resolution=12, show_plot=True)

print(f"Error std: {result['sigma']:.3f} LSB")
print(f"KL divergence: {result['kl_divergence']:.4f}")
```

### Example 2: Multiple Non-Idealities

```python
# Compare different error sources
datasets = {
    'Thermal Noise': signal_with_noise,
    'Quantization': signal_with_quant,
    'Jitter': signal_with_jitter,
}

for name, data in datasets.items():
    result = analyze_error_pdf(data, resolution=12, show_plot=False)
    print(f"{name:15s}: σ={result['sigma']:.3f} LSB, "
          f"KL={result['kl_divergence']:.4f}")
```

## Interpretation

### Error Distribution Shapes

| PDF Shape | Likely Cause |
|-----------|--------------|
| **Gaussian** | Thermal noise (random, white) |
| **Uniform** | Quantization noise (ideal ADC) |
| **Bimodal** | Missing codes, DNL issues |
| **Heavy tails** | Impulsive noise, glitches |
| **Asymmetric** | Systematic offset, drift |

### KL Divergence

- **KL < 0.1**: Very close to Gaussian (thermal noise dominant)
- **0.1 < KL < 0.5**: Moderately Gaussian
- **KL > 0.5**: Non-Gaussian (distortion, deterministic errors)

## Use Cases

- Distinguish thermal noise from quantization noise
- Identify non-Gaussian error sources (glitches, interference)
- Validate ADC noise models
- Compare error characteristics across different non-idealities

## See Also

- [`analyze_error_autocorr`](analyze_error_autocorr.md) — Temporal correlation in errors
- [`analyze_error_spectrum`](analyze_error_spectrum.md) — Frequency content of errors
- [`fit_sine_4param`](fit_sine_4param.md) — Sine wave fitting

## References

1. IEEE Std 1241-2010, "IEEE Standard for Terminology and Test Methods for ADCs"
2. S. Kullback and R. A. Leibler, "On Information and Sufficiency," Annals of Mathematical Statistics, 1951
