# INLsine

## Overview

`INLsine` computes Integral Nonlinearity (INL) and Differential Nonlinearity (DNL) from sinewave histogram test using the inverse cosine method.

## Syntax

```matlab
[INL, DNL, code] = INLsine(data)
[INL, DNL, code] = INLsine(data, clip)
```

## Input Arguments

- **`data`** — ADC output codes (1×N or N×1 vector)
- **`clip`** — Fraction of range to exclude from edges, default: `0.01`
  - Removes unreliable bins near saturation (top/bottom 1%)

## Output Arguments

- **`INL`** — Integral nonlinearity (LSB units, 1×M vector)
- **`DNL`** — Differential nonlinearity (LSB units, 1×M vector)
- **`code`** — Corresponding code values (1×M vector)

## Algorithm

### 1. Histogram Construction

```
code_range = floor(min(data)) : ceil(max(data))
clip_bins = round(clip × length(code_range) / 2)
code_valid = code_range[clip_bins : end - clip_bins]

hist_counts = histogram(data, code_valid ± 0.5)
```

### 2. Inverse Cosine Transform

Sinewave PDF is `p(x) ∝ 1/sqrt(1 - x²)`, corresponding CDF:
```
CDF(code) = cumsum(hist_counts) / sum(hist_counts)
Linearized_CDF = -cos(π × CDF)
```

This maps the nonlinear sinewave distribution to a linear ideal ADC response.

### 3. DNL Calculation

```
DNL_raw = diff(Linearized_CDF)
DNL_normalized = DNL_raw / mean(DNL_raw) × (num_codes - 1) - 1
DNL = DNL_normalized - mean(DNL_normalized)  % Remove offset
```

**Units**: LSB (Least Significant Bit)
- DNL = 0: Ideal step size
- DNL = -1: Missing code
- DNL > 0: Code width > 1 LSB

### 4. INL Calculation

```
INL = cumsum(DNL)
```

**Units**: LSB
- INL = 0: Ideal transfer function
- INL > 0: Output higher than ideal
- INL < 0: Output lower than ideal

## Examples

### Example 1: Basic Usage

```matlab
data = calibrated_adc_output;
[INL, DNL, code] = INLsine(data);

figure;
subplot(2,1,1);
plot(code, DNL);
ylabel('DNL (LSB)'); xlabel('Code');
title('Differential Nonlinearity');
grid on;

subplot(2,1,2);
plot(code, INL);
ylabel('INL (LSB)'); xlabel('Code');
title('Integral Nonlinearity');
grid on;
```

### Example 2: Tight Clipping

```matlab
[INL, DNL] = INLsine(data, 0.05);  % Exclude top/bottom 5%
fprintf('Peak INL: %.3f LSB\n', max(abs(INL)));
fprintf('Peak DNL: %.3f LSB\n', max(abs(DNL)));
```

## Interpretation

### DNL Analysis

| DNL Value | Meaning |
|-----------|---------|
| `DNL ≈ 0` | Ideal uniform code width |
| `DNL < -0.5` | Code width < 0.5 LSB → potential missing code |
| `DNL = -1` | Missing code (zero hits) |
| `DNL > 0.5` | Code width > 1.5 LSB → nonlinearity |

### INL Analysis

| INL Value | ADC Quality |
|-----------|-------------|
| `max(\|INL\|) < 0.5` | Excellent (< 0.5 LSB error) |
| `max(\|INL\|) < 1.0` | Good (< 1 LSB error) |
| `max(\|INL\|) > 2.0` | Poor, needs calibration |
| INL shape: bow | Gain/offset error |
| INL shape: S-curve | 2nd-order nonlinearity |

## Limitations

- **Sinewave input required**: Assumes sinewave histogram PDF `∝ 1/sqrt(1 - x²)`
  - Ramp or triangle inputs need different methods
- **Sufficient samples**: Requires `>> 2^N` samples for N-bit ADC (typically 10× minimum)
- **Clipping sensitive**: Edge bins unreliable due to saturation → adjust `clip` parameter
- **No missing code handling**: Missing codes (DNL = -1) cause CDF discontinuities

## See Also

- [`FGCalSine`](FGCalSine.md) — Calibration to reduce INL/DNL
- [`errHistSine`](errHistSine.md) — Error histogram with phase/code binning
- [`specPlot`](specPlot.md) — Frequency-domain linearity (SFDR, THD)

## References

1. IEEE Std 1241-2010, Section 5.5, "Histogram Test Method"
2. J. Doernberg et al., "Full-Speed Testing of A/D Converters," JSSC 1984.
