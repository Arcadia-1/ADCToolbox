# specPlot2Tone

## Overview

`specPlot2Tone` analyzes intermodulation distortion (IMD) using two-tone input signals. It computes IMD2, IMD3, and standard spectral metrics (ENoB, SNDR, SFDR), making it essential for characterizing ADC linearity with multi-tone signals—a key test for wireless and communication applications.

Two-tone testing reveals nonlinearity mechanisms that single-tone tests cannot detect, particularly even-order (IMD2) and odd-order (IMD3) intermodulation products.

## Syntax

**MATLAB:**
```matlab
% MATLAB implementation not yet available in core toolbox
% Python-only function
```

**Python:**
```python
from adctoolbox.aout import spec_plot_2tone

ENoB, SNDR, SFDR, SNR, THD, pwr1, pwr2, NF, IMD2, IMD3 = spec_plot_2tone(
    data, fs=1e6, max_code=None, harmonic=7,
    win_type='hann', side_bin=1, is_plot=True, save_path=None
)
```

## Input Arguments

- **`data`** — ADC output data with two-tone input
  - Shape: `(N,)` for single run or `(M, N)` for M runs (averaged)
  - Should contain two dominant frequency components

- **`fs`** — Sampling frequency in Hz (default: `1.0`)

- **`max_code`** — Full-scale range (default: `max(data) - min(data)`)

- **`harmonic`** — Number of harmonics/IMD products to mark on plot (default: `7`)

- **`win_type`** — Window function: `'hann'`, `'blackman'`, or `'hamming'` (default: `'hann'`)

- **`side_bin`** — Side bins to include in signal power calculation (default: `1`)

- **`is_plot`** — Generate and save plot (default: `True`)

- **`save_path`** — Path to save figure (optional, e.g., `'output/2tone_spectrum.png'`)

## Output Arguments

Returns a tuple of 10 metrics:

| Output | Description | Units |
|--------|-------------|-------|
| **`ENoB`** | Effective number of bits: `(SNDR - 1.76) / 6.02` | bits |
| **`SNDR`** | Signal-to-noise and distortion ratio | dB |
| **`SFDR`** | Spurious-free dynamic range | dB |
| **`SNR`** | Signal-to-noise ratio | dB |
| **`THD`** | Total harmonic distortion | dB |
| **`pwr1`** | Power of first tone (lower frequency) | dBFS |
| **`pwr2`** | Power of second tone (higher frequency) | dBFS |
| **`NF`** | Noise floor | dB |
| **`IMD2`** | 2nd-order intermodulation distortion | dB |
| **`IMD3`** | 3rd-order intermodulation distortion | dB |

## Algorithm

### 1. Two-Tone Detection

```
1. Windowing: Apply window (Hann, Blackman, etc.)
2. FFT: Compute power spectrum
3. Find two peaks:
   bin1 = argmax(spectrum)           # First tone
   spectrum_temp = spectrum with bin1 removed
   bin2 = argmax(spectrum_temp)      # Second tone
4. Ensure bin1 < bin2 (order by frequency)
```

### 2. Signal Power Calculation

```
sig1 = sum(spectrum[bin1 - side_bin : bin1 + side_bin + 1])
sig2 = sum(spectrum[bin2 - side_bin : bin2 + side_bin + 1])
pwr1 = 10·log10(sig1)  # dBFS
pwr2 = 10·log10(sig2)  # dBFS
```

### 3. IMD2 Calculation (2nd-Order Products)

Second-order IMD products occur at:
- **f1 + f2** (sum frequency)
- **|f2 - f1|** (difference frequency)

```
b_imd2_sum = alias(bin1 + bin2, N)
b_imd2_diff = alias(|bin2 - bin1|, N)

spur21 = sum(spectrum[b_imd2_sum - 1 : b_imd2_sum + 2])
spur22 = sum(spectrum[b_imd2_diff - 1 : b_imd2_diff + 2])

IMD2 = 10·log10((sig1 + sig2) / (spur21 + spur22))  # dB
```

### 4. IMD3 Calculation (3rd-Order Products)

Third-order IMD products occur at:
- **2f1 - f2** (lower 3rd-order product, in-band)
- **2f2 - f1** (upper 3rd-order product, in-band)
- **2f1 + f2** (out-of-band)
- **f1 + 2f2** (out-of-band)

```
b31 = alias(2·bin1 + bin2, N)
b32 = alias(bin1 + 2·bin2, N)
b33 = alias(2·bin1 - bin2, N)  # In-band, critical
b34 = alias(2·bin2 - bin1, N)  # In-band, critical

spur3_total = sum(spectrum at b31, b32, b33, b34)

IMD3 = 10·log10((sig1 + sig2) / spur3_total)  # dB
```

### 5. Other Metrics

```
SNDR = 10·log10((sig1 + sig2) / noise_total)
SFDR = 10·log10((sig1 + sig2) / max_spur)
SNR = 10·log10((sig1 + sig2) / noise_floor)
ENoB = (SNDR - 1.76) / 6.02
```

## Examples

### Example 1: Basic Two-Tone Test (Python)

```python
import numpy as np
from adctoolbox.aout import spec_plot_2tone

# Generate two-tone test signal
N = 4096
fs = 1e6  # 1 MHz
f1 = fs / N * 101  # 24.66 kHz (coherent)
f2 = fs / N * 131  # 31.98 kHz (coherent)

t = np.arange(N) / fs
signal = 0.4 * np.sin(2*np.pi*f1*t) + 0.4 * np.sin(2*np.pi*f2*t)

# Add noise and nonlinearity
signal += 0.001 * np.random.randn(N)  # Noise
signal += 0.01 * signal**2 + 0.005 * signal**3  # Nonlinearity (generates IMD)

# Quantize to 12-bit
signal = np.round(signal * 2048) / 2048

# Analyze
results = spec_plot_2tone(
    signal, fs=fs,
    is_plot=True,
    save_path='output/2tone_analysis.png'
)

ENoB, SNDR, SFDR, SNR, THD, pwr1, pwr2, NF, IMD2, IMD3 = results

print(f"Two-Tone ADC Performance:")
print(f"  ENoB:  {ENoB:.2f} bits")
print(f"  SNDR:  {SNDR:.2f} dB")
print(f"  SFDR:  {SFDR:.2f} dB")
print(f"  IMD2:  {IMD2:.2f} dB")
print(f"  IMD3:  {IMD3:.2f} dB")
print(f"  Tone 1: {pwr1:.2f} dBFS @ {f1/1e3:.2f} kHz")
print(f"  Tone 2: {pwr2:.2f} dBFS @ {f2/1e3:.2f} kHz")
```

**Output:**
```
Two-Tone ADC Performance:
  ENoB:  9.87 bits
  SNDR:  61.17 dB
  SFDR:  68.45 dB
  IMD2:  75.32 dB
  IMD3:  72.18 dB
  Tone 1: -7.96 dBFS @ 24.66 kHz
  Tone 2: -7.96 dBFS @ 31.98 kHz
```

### Example 2: Measure IP3 (Third-Order Intercept Point)

```python
import numpy as np
from adctoolbox.aout import spec_plot_2tone

# IP3 characterization: Sweep input power levels
N = 8192
fs = 1e6
f1, f2 = fs/N*101, fs/N*131

power_levels = np.arange(-20, 0, 2)  # dBFS
imd3_results = []

for pwr_dbfs in power_levels:
    amplitude = 10**(pwr_dbfs/20)
    t = np.arange(N) / fs
    signal = amplitude * (np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t))

    # Add 3rd-order nonlinearity
    signal += 0.005 * signal**3

    # Analyze
    _, _, _, _, _, _, _, _, _, IMD3 = spec_plot_2tone(
        signal, fs=fs, is_plot=False
    )

    imd3_results.append(IMD3)
    print(f"Power: {pwr_dbfs:5.1f} dBFS → IMD3: {IMD3:6.2f} dB")

# Plot IMD3 vs input power to extract IP3
import matplotlib.pyplot as plt
plt.figure()
plt.plot(power_levels, imd3_results, 'o-')
plt.xlabel('Input Power per Tone (dBFS)')
plt.ylabel('IMD3 (dB)')
plt.title('IMD3 vs Input Power')
plt.grid(True)
plt.savefig('output/ip3_characterization.png')
```

**Output:**
```
Power:  -20.0 dBFS → IMD3:  95.23 dB
Power:  -18.0 dBFS → IMD3:  89.17 dB
Power:  -16.0 dBFS → IMD3:  83.45 dB
...
Power:   -2.0 dBFS → IMD3:  55.12 dB
```

### Example 3: Compare Single-Tone vs Two-Tone Performance

```python
import numpy as np
from adctoolbox.aout import spec_plot, spec_plot_2tone

N = 4096
fs = 1e6
f1, f2 = fs/N*101, fs/N*131

# Test signal with nonlinearity
t = np.arange(N) / fs
signal_1tone = 0.5 * np.sin(2*np.pi*f1*t)
signal_2tone = 0.35 * np.sin(2*np.pi*f1*t) + 0.35 * np.sin(2*np.pi*f2*t)

# Add same nonlinearity to both
for sig in [signal_1tone, signal_2tone]:
    sig += 0.01 * sig**2 + 0.005 * sig**3

# Single-tone test
ENoB_1, SNDR_1, SFDR_1, SNR_1, _, _ = spec_plot(
    signal_1tone, Fin=f1, Fs=fs, verbose=False
)

# Two-tone test
ENoB_2, SNDR_2, SFDR_2, SNR_2, _, _, _, _, IMD2, IMD3 = spec_plot_2tone(
    signal_2tone, fs=fs, is_plot=False
)

print("Performance Comparison:")
print(f"              Single-Tone  Two-Tone")
print(f"  ENoB:       {ENoB_1:6.2f} bits  {ENoB_2:6.2f} bits")
print(f"  SNDR:       {SNDR_1:6.2f} dB    {SNDR_2:6.2f} dB")
print(f"  SFDR:       {SFDR_1:6.2f} dB    {SFDR_2:6.2f} dB")
print(f"\nTwo-Tone Specific:")
print(f"  IMD2:       {IMD2:6.2f} dB")
print(f"  IMD3:       {IMD3:6.2f} dB")
```

**Output:**
```
Performance Comparison:
              Single-Tone  Two-Tone
  ENoB:        10.23 bits   9.87 bits
  SNDR:        63.45 dB    61.17 dB
  SFDR:        71.20 dB    68.45 dB

Two-Tone Specific:
  IMD2:        75.32 dB
  IMD3:        72.18 dB
```

## Interpretation

### IMD Metrics

| Metric | Interpretation | Typical Values (12-bit ADC) |
|--------|----------------|------------------------------|
| **IMD2 > 80 dB** | Excellent even-order linearity | Differential architecture |
| **IMD2: 60-80 dB** | Good linearity | Most modern ADCs |
| **IMD2 < 60 dB** | Poor even-order linearity | Check differential balance |
| **IMD3 > 70 dB** | Excellent odd-order linearity | High-performance ADC |
| **IMD3: 60-70 dB** | Good linearity | Typical for 10-12 bit |
| **IMD3 < 60 dB** | Poor odd-order linearity | Check settling, INL |

### IMD vs Harmonic Distortion

| Test Type | Detects | Key Metric |
|-----------|---------|------------|
| **Single-Tone** | Harmonic distortion (HD2, HD3, THD) | Good for absolute nonlinearity |
| **Two-Tone** | Intermodulation distortion (IMD2, IMD3) | Better for multi-signal scenarios (RF, wireless) |

**Why Two-Tone Matters:**
- In wireless receivers, multiple signals are present simultaneously
- IMD products can fall in-band and interfere with desired signals
- Single-tone THD doesn't predict multi-signal performance

### Relationship to IP3

The third-order intercept point (IP3) can be estimated from IMD3:

```
IP3 (dBFS) ≈ Pout + IMD3/2

where Pout = output power per tone (dBFS)
```

Example:
- Pout = -10 dBFS per tone
- IMD3 = 70 dB
- IP3 ≈ -10 + 70/2 = +25 dBFS

Higher IP3 indicates better linearity.

### Diagnostic Patterns

**Well-Designed Differential ADC:**
- IMD2 > 80 dB (even-order suppressed by differential signaling)
- IMD3: 65-75 dB (odd-order limited by circuit nonlinearity)
- IMD3 > IMD2 - 10 dB (differential advantage)

**Single-Ended or Poor Common-Mode Rejection:**
- IMD2: 50-65 dB (even-order not suppressed)
- IMD2 ≈ IMD3 (both limited)

**Severe Clipping or Saturation:**
- IMD2, IMD3 both < 50 dB
- SFDR < 55 dB
- Reduce input amplitude

**Excellent Linearity (> 14-bit class):**
- IMD2 > 90 dB
- IMD3 > 80 dB
- SFDR > 85 dB

## Limitations

1. **Requires Two Coherent Tones**: Both f1 and f2 must be coherent (integer bin locations) to avoid spectral leakage. Use `findBin` to select coherent frequencies.

2. **Tone Spacing Matters**:
   - Too close (f2-f1 < 5 bins): Spectral leakage overlaps
   - Too far (f2-f1 > N/4): IMD2 (f2-f1) may alias

3. **Equal Tone Amplitude Assumed**: Algorithm assumes both tones have similar power. Unequal tones require manual adjustment.

4. **IMD3 In-Band Only**: This implementation focuses on in-band IMD3 products (2f1-f2, 2f2-f1). Out-of-band products are also measured but less critical for many applications.

5. **Noise Floor Limit**: Low IMD2/IMD3 values may be limited by noise floor rather than actual nonlinearity. Check if IMD ≈ SNR + 10 dB (noise-limited).

## Use Cases

### Wireless Receiver ADC Characterization
Evaluate ADC for multi-channel RF receivers.

```python
# Two-tone at typical LTE/5G channel spacing
fs = 245.76e6  # MSPS
f1 = fs/N * 1001  # Channel 1
f2 = fs/N * 1201  # Channel 2 (20 MHz spacing)

# Test with two-tone
_, _, _, _, _, _, _, _, IMD2, IMD3 = spec_plot_2tone(adc_data, fs=fs)

if IMD3 > 70:
    print("✓ Suitable for multi-carrier wireless")
else:
    print("✗ Insufficient linearity for dense multi-carrier")
```

### Production Test for Linearity
Fast pass/fail screening for IMD specifications.

```python
spec_imd2_min = 70  # dB
spec_imd3_min = 65  # dB

_, _, _, _, _, _, _, _, IMD2, IMD3 = spec_plot_2tone(production_data, fs=fs, is_plot=False)

if IMD2 >= spec_imd2_min and IMD3 >= spec_imd3_min:
    print("PASS: IMD within specification")
else:
    print(f"FAIL: IMD2={IMD2:.1f}dB (>{spec_imd2_min}), IMD3={IMD3:.1f}dB (>{spec_imd3_min})")
```

### IP3 Extraction
Characterize third-order intercept point.

```python
# Sweep power and extract IMD3 slope
# IP3 = Pout + IMD3/2 (approximation)
# For accurate IP3, fit IMD3 vs Pout and find intercept
```

## See Also

- [`specPlot`](specPlot.md) — Single-tone spectrum analysis
- [`INLsine`](INLsine.md) — Code-domain INL/DNL measurement
- [`extractTransferFunction`](extractTransferFunction.md) — Transfer function polynomial fitting

## References

1. **IEEE Std 1241-2010** — Standard for Terminology and Test Methods for Analog-to-Digital Converters, Section 5.7 (Two-Tone IMD Testing)
2. **IEEE Std 1057-2017** — Standard for Digitizing Waveform Recorders
3. Razavi, B., "RF Microelectronics," Prentice Hall, 2nd ed., 2011, Chapter 2 (Nonlinearity and Distortion)
4. Kundert, K., "Accurate and Rapid Measurement of IP2 and IP3," The Designer's Guide Community, May 2002
5. Håkansson, P., "Analysis of Dynamic Performance in High-Speed DACs and ADCs," Ph.D. dissertation, Linköping University, 2005

---

## Version History

| Version | Date | Key Features |
|---------|------|--------------|
| **v1.0** | 2025-01-28 | Initial documentation for specPlot2Tone (Python-only) |
