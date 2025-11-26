# errHistSine

Analyzes ADC errors by comparing measured data against a fitted sine wave. Decomposes noise into amplitude and phase components.

## Syntax

```matlab
[emean, erms, phase_code, anoi, pnoi, err, xx] = errHistSine(data, ...)
```

### Parameters

| Name | Default | Description |
|------|---------|-------------|
| `data` | required | Input ADC data vector |
| `bin` | 100 | Number of histogram bins |
| `fin` | 0 | Normalized frequency (0-1), 0 = auto-detect |
| `disp` | 1 | Show plots (1) or not (0) |
| `mode` | 0 | 0 = phase mode, ≥1 = code mode |
| `erange` | [] | Filter errors to `[min, max]` range |

### Returns

| Output | Description |
|--------|-------------|
| `emean` | Mean error per bin (INL in code mode) |
| `erms` | RMS error per bin |
| `phase_code` | Bin centers (degrees or codes) |
| `anoi` | Amplitude noise RMS (phase mode only) |
| `pnoi` | Phase noise RMS in radians (phase mode only) |
| `err` | Raw errors (filtered by erange if set) |
| `xx` | X-axis values for err |

## How It Works

**1. Fits a sine wave** to your data: `data_fit[n] = A·sin(2πfn + φ) + DC`

**2. Calculates errors** between fit and actual data: `err = data_fit - data`

**3. Groups errors into bins** by phase (0-360°) or by code value

**4. Computes statistics** for each bin:
   - Mean error: average systematic error
   - RMS error: total variation including noise

**5. Decomposes noise** (phase mode only):
   - Separates **amplitude noise** (voltage noise) from **phase noise** (timing jitter)
   - Uses least-squares to solve: `erms²(θ) = σ²_A·cos²(θ) + (A·σ_φ)²·sin²(θ) + baseline`
   - Amplitude noise dominates at peaks, phase noise dominates at zero crossings

## Usage Examples

### Measure Amplitude and Phase Noise

```matlab
% Generate test signal
N = 10000; fin = 0.1;
data = sin(2*pi*fin*(0:N-1)) + 0.01*randn(1,N);

% Analyze
[~, ~, ~, anoi, pnoi] = errHistSine(data, 'fin', fin);
fprintf('Amplitude Noise: %.4f\n', anoi);
fprintf('Phase Noise: %.4f rad\n', pnoi);
```

### Measure ADC INL

```matlab
% Load ADC codes
adc_codes = load('adc_output.mat');

% Measure INL
[INL, ~, codes] = errHistSine(adc_codes, 'mode', 1, 'bin', 256);
fprintf('Peak INL: %.3f LSB\n', max(abs(INL)));
plot(codes, INL); xlabel('Code'); ylabel('INL [LSB]');
```

## Physical Interpretation

### Amplitude Noise (anoi)
Voltage noise from references, comparators, thermal sources. Units match your data (LSB or volts).

### Phase Noise (pnoi)
Timing jitter in radians. Convert to time jitter: **δt_rms = pnoi / (2π·f_in·f_s)**

**Example:** pnoi = 0.001 rad at f_in = 0.1 and f_s = 1 GHz → δt_rms ≈ 1.59 ps

### SNR from Noise Components

- **Amplitude SNR:** `20·log₁₀(A/(√2·anoi))` [dBc]
- **Phase SNR:** `20·log₁₀(1/(√2·pnoi))` [dBc]
- **Total SNR:** `-10·log₁₀(10^(-SNR_amp/10) + 10^(-SNR_phase/10))`

## Key Metrics

### Normalized Frequency
**f_in = f_signal / f_sampling** where f_in < 0.5 (Nyquist)

For coherent sampling use f_in = M/N (M cycles in N samples)

### INL/DNL (Code Mode)
- **INL[k] = emean[k]** - Integral nonlinearity at code k
- **DNL[k] ≈ INL[k] - INL[k-1]** - Differential nonlinearity
- **Peak INL = max|emean|**

### ENOB (Effective Bits)
**ENOB = log₂(mag / (√2·erms_total))** where erms_total combines all noise sources

**SINAD = 6.02·ENOB + 1.76** [dB]

## Applications

- **ADC Testing** - Measure INL/DNL in code mode
- **Noise Analysis** - Separate voltage noise from timing jitter
- **Jitter Testing** - Quantify aperture jitter from phase noise
- **Clock Quality** - Evaluate sampling clock performance

## Notes

- Automatically handles row or column vectors
- **Phase mode**: Best for dynamic testing with sine waves
- **Code mode**: Best for static testing (histogram method)
- Noise decomposition uses robust least-squares with fallbacks
- Set `disp=0` for batch processing without plots

## See Also

`sineFit` · `errPDF` · `errAutoCorrelation`
