# errHistSine

## Overview

Analyzes ADC errors by comparing measured data against a fitted sine wave. Provides statistical error analysis and decomposes noise into amplitude and phase components.

## Syntax

```matlab
[emean, erms, phase_code, anoi, pnoi, err, xx] = errHistSine(data, ...)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data` | (required) | Input ADC data vector |
| `bin` | 100 | Number of histogram bins |
| `fin` | 0 (auto) | Normalized frequency (0-1), cycles per sample |
| `disp` | 1 | Display plots: 1=yes, 0=no |
| `mode` | 0 | 0=phase mode, ≥1=code mode |
| `erange` | [] | Filter errors: `[min, max]` range on x-axis |

### Outputs

| Output | Description |
|--------|-------------|
| `emean` | Mean error per bin (reveals INL in code mode) |
| `erms` | RMS error per bin |
| `phase_code` | Bin centers (phase in degrees or code values) |
| `anoi` | Amplitude noise RMS (phase mode only) |
| `pnoi` | Phase noise RMS in radians (phase mode only) |
| `err` | Raw errors (filtered if `erange` specified) |
| `xx` | X-axis values for `err` |

## Algorithm

### 1. Sine Wave Fitting

$$\text{data\_fit}[n] = A \cdot \sin(2\pi f_{in} n + \varphi) + DC$$

### 2. Error Calculation

$$\text{err}[n] = \text{data\_fit}[n] - \text{data}[n]$$

### 3. Binning

**Phase Mode** ($\theta$ in degrees):

$$\theta[n] = \text{mod}\left(\frac{\varphi}{\pi} \times 180 + n \cdot f_{in} \cdot 360, 360\right)$$

**Code Mode**:

$$\text{bin\_index} = \min\left(\left\lfloor\frac{\text{data}[n] - \text{data}_{\min}}{\text{bin\_width}}\right\rfloor + 1, N_{bins}\right)$$

### 4. Statistics per Bin

$$\text{emean}[b] = \frac{1}{N_b} \sum_{i \in b} \text{err}[i]$$

$$\text{erms}[b] = \sqrt{\frac{1}{N_b} \sum_{i \in b} \left(\text{err}[i] - \text{emean}[b]\right)^2}$$

### 5. Noise Decomposition (Phase Mode Only)

Models RMS error variance as a combination of amplitude and phase noise:

$$\text{erms}^2(\theta) = \sigma^2_A \cdot \cos^2(\theta) + (A \cdot \sigma_\varphi)^2 \cdot \sin^2(\theta) + \sigma^2_{bl}$$

Solved via least-squares:

$$\begin{bmatrix}
\cos^2(\theta_1) & \sin^2(\theta_1) & 1 \\
\vdots & \vdots & \vdots \\
\cos^2(\theta_n) & \sin^2(\theta_n) & 1
\end{bmatrix}
\begin{bmatrix}
\sigma^2_A \\
A^2 \sigma^2_\varphi \\
\sigma^2_{bl}
\end{bmatrix}
=
\begin{bmatrix}
\text{erms}^2(\theta_1) \\
\vdots \\
\text{erms}^2(\theta_n)
\end{bmatrix}$$

Output: $\text{anoi} = \sigma_A$, $\text{pnoi} = \sigma_\varphi$

**Robust fallback**: If solution yields negative/imaginary values, tries phase-only, amplitude-only, or baseline-only fits.

## Physical Interpretation

### Amplitude Noise (anoi)
- Sources: reference noise, comparator noise, thermal noise
- Units: same as input data (LSB or volts)

### Phase Noise (pnoi)
- Sources: sampling clock jitter, timing uncertainty
- Units: radians
- Convert to timing jitter:

$$\delta t_{\text{rms}} = \frac{\text{pnoi}}{2\pi f_{in} f_s}$$

**Example**: $\text{pnoi} = 0.001$ rad, $f_{in} = 0.1$, $f_s = 1$ GHz → $\delta t_{\text{rms}} \approx 1.59$ ps

### SNR Contributions

$$\text{SNR}_{\text{amp}} = 20 \log_{10}\left(\frac{A}{\sqrt{2} \cdot \text{anoi}}\right) \text{ [dBc]}$$

$$\text{SNR}_{\text{phase}} = 20 \log_{10}\left(\frac{1}{\sqrt{2} \cdot \text{pnoi}}\right) \text{ [dBc]}$$

$$\text{SNR}_{\text{total}} = -10 \log_{10}\left(10^{-\text{SNR}_{\text{amp}}/10} + 10^{-\text{SNR}_{\text{phase}}/10}\right)$$

## Usage Examples

### Phase Mode: Noise Decomposition

```matlab
% Generate noisy sine wave
N = 10000; fin = 0.1;
data = sin(2*pi*fin*(0:N-1)) + 0.01*randn(1,N);

% Analyze
[emean, erms, phase, anoi, pnoi] = errHistSine(data, 'fin', fin);
fprintf('Amplitude Noise: %.4f\n', anoi);
fprintf('Phase Noise: %.4f rad\n', pnoi);
```

### Code Mode: INL Measurement

```matlab
% ADC output codes
adc_codes = load('adc_output.mat');

% Measure INL
[INL, erms, codes] = errHistSine(adc_codes, 'mode', 1, 'bin', 256);
fprintf('Peak INL: %.3f LSB\n', max(abs(INL)));
```

## Key Formulas

### Normalized Frequency

$$f_{in} = \frac{f_{\text{signal}}}{f_{\text{sampling}}} \in (0, 0.5) \quad \text{(Nyquist)}$$

Coherent sampling: $f_{in} = M/N$ (integers)

### INL/DNL (Code Mode)

$$\text{INL}[k] = \text{emean}[k]$$

$$\text{DNL}[k] \approx \text{INL}[k] - \text{INL}[k-1]$$

$$\text{INL}_{\text{peak}} = \max_k |\text{emean}[k]|$$

### ENOB

$$\text{ENOB} = \log_2\left(\frac{\text{mag}}{\sqrt{2} \cdot \text{erms}_{\text{total}}}\right)$$

where $\text{erms}_{\text{total}} = \sqrt{\text{anoi}^2 + (\text{pnoi} \cdot \text{mag})^2 + \sigma^2_{bl}}$

$$\text{SINAD [dB]} = 6.02 \cdot \text{ENOB} + 1.76$$

## Applications

- **ADC Characterization**: INL/DNL measurement (code mode)
- **Noise Analysis**: Separate amplitude vs. phase noise sources
- **Jitter Testing**: Phase noise → aperture jitter quantification
- **Clock Quality**: Phase noise indicates clock performance

## Implementation Notes

- Auto-transposes row vectors
- Phase mode: ideal for dynamic testing with coherent sine inputs
- Code mode: ideal for static testing (ramp/histogram method)
- Noise decomposition uses least-squares with robust fallback
- Plots generated when `disp=1` show error vs. phase/code and RMS profiles

## See Also

- `sineFit` - Sine wave fitting (used internally)
- `errPDF` - Error probability distribution
- `errAutoCorrelation` - Error autocorrelation analysis
