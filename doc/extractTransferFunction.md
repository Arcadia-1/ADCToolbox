# extractTransferFunction

## Overview

`extractTransferFunction` extracts the actual transfer function `y = f(x)` from ADC output data by fitting a polynomial to the relationship between ideal input and actual output. This is useful for characterizing ADC nonlinearity, measuring transfer function coefficients, and understanding distortion mechanisms.

## Syntax

```matlab
[k0, k1, k2, k3, k4, k5, x_ideal, y_actual, polycoeff] = extractTransferFunction(data, fin, polyorder)
```

```python
# Python implementation not yet available
```

## Input Arguments

- **`data`** — ADC output signal (1×N vector), typically a sine wave with distortion
- **`fin`** — Normalized frequency (0 to 1), set to `0` to auto-detect using `sineFit`
- **`polyorder`** — Polynomial order for fitting (e.g., `3` for cubic, `5` for quintic)

## Output Arguments

- **`k0`** — DC offset coefficient (constant term)
- **`k1`** — Linear gain coefficient
- **`k2`** — 2nd-order nonlinearity coefficient (quadratic)
- **`k3`** — 3rd-order nonlinearity coefficient (cubic)
- **`k4`** — 4th-order nonlinearity coefficient (quartic)
- **`k5`** — 5th-order nonlinearity coefficient (quintic)
- **`x_ideal`** — Ideal input signal (zero-mean fitted sine wave)
- **`y_actual`** — Actual output signal (zero-mean ADC output)
- **`polycoeff`** — Raw polynomial coefficients from `polyfit`

## Algorithm

### 1. Ideal Input Reconstruction

```
1. Fit sine wave to ADC output using sineFit
2. x_ideal ← fitted_sine - mean(fitted_sine)  # Zero-mean ideal input
3. y_actual ← data - mean(data)               # Zero-mean actual output
```

The fitted sine wave represents what the "perfect" input would have been.

### 2. Polynomial Fitting

```
1. Normalize: x_norm ← x_ideal / max(|x_ideal|)  # Scale to [-1, 1]
2. Fit: polycoeff ← polyfit(x_norm, y_actual, polyorder)
3. Denormalize coefficients:
   k_i = polycoeff[i] / (x_max^i)
```

Normalization to `[-1, 1]` ensures numerical stability for high-order polynomials.

### 3. Transfer Function Model

The extracted transfer function is:

```
y_actual = k0 + k1·x + k2·x² + k3·x³ + k4·x⁴ + k5·x⁵ + ...
```

Where:
- **k1 ≈ 1**: Linear gain (should be close to 1 for well-calibrated ADC)
- **k2**: Even-order nonlinearity (generates 2nd harmonic, DC shift)
- **k3**: Odd-order nonlinearity (generates 3rd harmonic, gain compression)
- **k4, k5**: Higher-order nonlinearities

## Examples

### Example 1: Extract 3rd-Order Transfer Function

```matlab
% Load ADC output data
data = readmatrix('adc_output_sinewave.csv');

% Extract up to cubic nonlinearity
[k0, k1, k2, k3] = extractTransferFunction(data, 0, 3);

fprintf('Transfer function: y = %.6f + %.6f·x + %.6f·x² + %.6f·x³\n', ...
        k0, k1, k2, k3);
```

**Output:**
```
Transfer function: y = 0.000124 + 0.998765·x + 0.001234·x² + -0.000456·x³
```

### Example 2: Analyze Nonlinearity vs Polynomial Order

```matlab
data = readmatrix('adc_data.csv');

% Test different polynomial orders
for order = 1:5
    [k0, k1, k2, k3, k4, k5, x_ideal, y_actual, polycoeff] = ...
        extractTransferFunction(data, 0, order);

    % Reconstruct transfer function
    x_norm = x_ideal / max(abs(x_ideal));
    y_fitted = polyval(polycoeff, x_norm);
    residual = y_actual - y_fitted;

    fprintf('Order %d: RMS residual = %.6f\n', order, rms(residual));
end
```

**Output:**
```
Order 1: RMS residual = 0.045123
Order 2: RMS residual = 0.012456
Order 3: RMS residual = 0.003789
Order 4: RMS residual = 0.003201
Order 5: RMS residual = 0.003145
```

### Example 3: Visualize Transfer Function

```matlab
data = readmatrix('adc_data.csv');
[k0, k1, k2, k3, ~, ~, x_ideal, y_actual] = extractTransferFunction(data, 0, 3);

% Plot ideal vs actual transfer function
figure;
subplot(2,1,1);
scatter(x_ideal, y_actual, 1, 'b.');
hold on;
x_fit = linspace(min(x_ideal), max(x_ideal), 1000);
y_fit = k0 + k1*x_fit + k2*x_fit.^2 + k3*x_fit.^3;
plot(x_fit, y_fit, 'r-', 'LineWidth', 2);
xlabel('Ideal Input');
ylabel('Actual Output');
title('ADC Transfer Function');
legend('Actual Data', sprintf('y = %.4f + %.4f·x + %.4f·x²', k0, k1, k2));
grid on;

% Plot residual error
subplot(2,1,2);
y_model = k0 + k1*x_ideal + k2*x_ideal.^2 + k3*x_ideal.^3;
residual = y_actual - y_model;
plot(x_ideal, residual, 'b.');
xlabel('Ideal Input');
ylabel('Residual Error');
title('Transfer Function Residual');
grid on;
```

## Interpretation

### Transfer Function Coefficients

| Coefficient | Physical Meaning | Typical Value | Causes |
|-------------|------------------|---------------|--------|
| **k0** | DC offset | < 0.01 LSB | Comparator offset, mismatch |
| **k1** | Linear gain | ≈ 1.00 | Ideal gain is 1.0 |
| **k2** | 2nd-order NL | < 0.001 | Even-order distortion, capacitor mismatch |
| **k3** | 3rd-order NL | < 0.0001 | Odd-order distortion, settling errors |
| **k4, k5** | Higher-order NL | < 0.00001 | Complex nonlinearities |

### Relationship to Harmonics

- **k2** (quadratic): Generates 2nd harmonic in frequency domain
- **k3** (cubic): Generates 3rd harmonic
- **k4** (quartic): Generates 2nd and 4th harmonics
- **k5** (quintic): Generates 3rd and 5th harmonics

For a sine wave input `x = A·sin(ωt)`:
```
y = k1·A·sin(ωt) + k2·A²·sin²(ωt) + k3·A³·sin³(ωt) + ...
  ≈ k1·A·sin(ωt)                           # Fundamental
    + (k2·A²/2)                            # DC from 2nd-order
    - (k2·A²/2)·cos(2ωt)                  # 2nd harmonic
    + (3k3·A³/4)·sin(ωt)                  # Gain compression
    - (k3·A³/4)·sin(3ωt)                  # 3rd harmonic
```

### Diagnostic Patterns

**Well-Calibrated ADC:**
- k1 ≈ 1.00 (within 0.1%)
- k0, k2, k3 < 0.001
- Residual RMS < 0.01 LSB

**2nd-Order Dominant (Even Distortion):**
- k2 >> k3
- Strong 2nd harmonic in spectrum
- Asymmetric transfer function

**3rd-Order Dominant (Odd Distortion):**
- k3 >> k2
- Strong 3rd harmonic in spectrum
- Gain compression/expansion

**Poor Fit (Need Higher Order):**
- Residual RMS >> 0.01 LSB
- Increasing polynomial order significantly reduces residual

## Limitations

1. **Requires Sine Wave Input**: Transfer function extraction assumes a clean sine wave input. Other waveforms (ramp, noise) are not supported.

2. **Numerical Stability**: Very high polynomial orders (>7) may suffer from numerical conditioning issues. Use normalized fitting to mitigate.

3. **Coherent Sampling**: Best results require coherent sampling (integer number of periods). Non-coherent sampling adds spectral leakage.

4. **Static Nonlinearity Only**: This method captures static (DC) transfer function. Dynamic effects (memory, hysteresis) are not modeled.

5. **Limited by SNR**: Noise floor limits the accuracy of extracted coefficients. High SNR data (>60 dB) recommended.

## Use Cases

### Nonlinearity Characterization
Extract and quantify ADC transfer function for datasheet specifications.

```matlab
[k0, k1, k2, k3] = extractTransferFunction(data, 0, 3);
fprintf('INL contributors: k2=%.6f, k3=%.6f\n', k2, k3);
```

### Pre-Distortion Calibration
Use extracted coefficients to design inverse transfer function for digital calibration.

```matlab
% Extract forward transfer function
[~, k1, k2, k3] = extractTransferFunction(data, 0, 3);

% Design inverse (linearization)
inverse_k2 = -k2 / k1;
inverse_k3 = -k3 / k1;
```

### Harmonic Source Identification
Determine whether harmonics come from even-order (k2) or odd-order (k3) nonlinearity.

```matlab
[~, ~, k2, k3] = extractTransferFunction(data, 0, 3);
if abs(k2) > abs(k3)
    fprintf('Dominant distortion: 2nd-order (even)\n');
else
    fprintf('Dominant distortion: 3rd-order (odd)\n');
end
```

## See Also

- [`specPlot`](specPlot.md) — Frequency-domain analysis to measure harmonics
- [`INLsine`](INLsine.md) — Code-domain INL/DNL measurement
- [`tomDecomp`](tomDecomp.md) — Time-domain error decomposition
- [`sineFit`](sineFit.md) — Sine wave fitting for ideal input reconstruction

## References

1. **IEEE Std 1241-2010** — Standard for Terminology and Test Methods for Analog-to-Digital Converters
2. Kester, W., "Taking the Mystery out of the Infamous Formula, 'SNR = 6.02N + 1.76dB,' and Why You Should Care," *Analog Devices Tutorial MT-001*
3. Doernberg, J., Lee, H.S., and Hodges, D.A., "Full-Speed Testing of A/D Converters," *IEEE Journal of Solid-State Circuits*, vol. 19, no. 6, Dec 1984

---

## Version History

| Version | Date | Key Features |
|---------|------|--------------|
| **v1.0** | 2025-01-28 | Initial documentation for extractTransferFunction |
