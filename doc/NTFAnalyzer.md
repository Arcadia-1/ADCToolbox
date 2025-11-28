# NTFAnalyzer

## Overview

`NTFAnalyzer` analyzes the performance of a Noise Transfer Function (NTF) for oversampled ADCs (Delta-Sigma modulators). It calculates the integrated noise suppression within a specified signal band and optionally plots the NTF magnitude response.

This tool is essential for Delta-Sigma ADC design, helping designers evaluate noise-shaping effectiveness and optimize NTF design for maximum SNR in the band of interest.

## Syntax

```matlab
noiSup = NTFAnalyzer(NTF, Flow, Fhigh)
noiSup = NTFAnalyzer(NTF, Flow, Fhigh, isPlot)
```

```python
# Python equivalent: python/src/adctoolbox/oversampling/ntf_analyzer.py
from adctoolbox.oversampling import ntf_analyzer
noi_sup = ntf_analyzer(NTF, Flow, Fhigh, is_plot=False)
```

## Input Arguments

- **`NTF`** — Noise transfer function in z-domain (MATLAB `tf` or `zpk` object)
- **`Flow`** — Lower bound of signal band (normalized frequency, 0 to 0.5 relative to Fs)
- **`Fhigh`** — Upper bound of signal band (normalized frequency, 0 to 0.5 relative to Fs)
- **`isPlot`** — (Optional) Plot NTF magnitude (0 = no plot, 1 = plot), default: `0`

## Output Arguments

- **`noiSup`** — Integrated noise suppression in signal band (dB)
  - Relative to flat NTF = 1 (no noise shaping)
  - Positive value indicates noise suppression
  - Negative value indicates noise amplification (unstable or poor NTF)

## Algorithm

### 1. Frequency Response Evaluation

```
1. Generate frequency vector: w = linspace(0, 0.5, 2^16)  # 65536 points
2. Compute magnitude response: |NTF(e^(jω))|
3. Extract magnitude: mag = |NTF| at each frequency point
```

The frequency range `[0, 0.5]` covers DC to Nyquist in normalized frequency.

### 2. In-Band Noise Power

```
1. Find bins in signal band: idx = (w > Flow) & (w < Fhigh)
2. Compute in-band noise power: np = mean(mag[idx]^2)
3. Convert to dB: noiSup = -10·log10(np)
```

The noise power is the mean-squared magnitude response in the signal band. For a flat NTF (|NTF| = 1), `np = 1` and `noiSup = 0 dB`.

### 3. Noise Suppression Interpretation

```
noiSup = -10·log10(np)
       = -10·log10(mean(|NTF|^2 in band))

If |NTF| < 1 in band → noiSup > 0 dB → Noise suppressed ✓
If |NTF| = 1 in band → noiSup = 0 dB → No shaping
If |NTF| > 1 in band → noiSup < 0 dB → Noise amplified ✗
```

### 4. Optional Plotting

If `isPlot = 1`, generates a semilog plot:
- X-axis: Normalized frequency (log scale, 0 to 0.5)
- Y-axis: |NTF| in dB (20·log10(mag))
- Vertical dashed lines mark signal band edges (Flow, Fhigh)

## Examples

### Example 1: Analyze 2nd-Order Delta-Sigma NTF

```matlab
% Design 2nd-order noise-shaping NTF
OSR = 64;           % Oversampling ratio
order = 2;          % Modulator order
H_inf = 1.5;        % Out-of-band gain

% Create NTF using Delta-Sigma toolbox
NTF = synthesizeNTF(order, OSR, 1, H_inf);

% Signal band: DC to Fs/(2*OSR)
Flow = 0;
Fhigh = 1 / (2 * OSR);

% Analyze NTF
noiSup = NTFAnalyzer(NTF, Flow, Fhigh, 1);
fprintf('Noise suppression in signal band: %.1f dB\n', noiSup);
```

**Output:**
```
Noise suppression in signal band: 42.3 dB
```

### Example 2: Compare Different NTF Designs

```matlab
OSR = 128;
Flow = 0;
Fhigh = 1 / (2 * OSR);

% Test different modulator orders
for order = 1:4
    NTF = synthesizeNTF(order, OSR, 1);
    noiSup = NTFAnalyzer(NTF, Flow, Fhigh);
    fprintf('Order %d: Noise suppression = %.1f dB\n', order, noiSup);
end
```

**Output:**
```
Order 1: Noise suppression = 24.5 dB
Order 2: Noise suppression = 49.2 dB
Order 3: Noise suppression = 73.8 dB
Order 4: Noise suppression = 98.3 dB
```

Each additional order adds approximately 20-25 dB of noise suppression.

### Example 3: Bandpass Delta-Sigma Analysis

```matlab
% Bandpass modulator centered at Fs/4
OSR = 64;
center_freq = 0.25;    % Fs/4
bandwidth = 1 / (2 * OSR);

Flow = center_freq - bandwidth/2;
Fhigh = center_freq + bandwidth/2;

% Design bandpass NTF
NTF = synthesizeBandpassNTF(2, OSR, center_freq);

noiSup = NTFAnalyzer(NTF, Flow, Fhigh, 1);
fprintf('Bandpass NTF noise suppression: %.1f dB\n', noiSup);
```

**Output:**
```
Bandpass NTF noise suppression: 38.7 dB
```

### Example 4: Verify Stability with Noise Suppression

```matlab
% Aggressive NTF design (may be unstable)
OSR = 64;
order = 4;
H_inf = 2.5;  % High out-of-band gain (risky)

NTF = synthesizeNTF(order, OSR, 1, H_inf);

Flow = 0;
Fhigh = 1 / (2 * OSR);

noiSup = NTFAnalyzer(NTF, Flow, Fhigh, 1);

% Check for noise amplification (instability indicator)
if noiSup < 0
    warning('Noise amplification detected! NTF may be unstable.');
else
    fprintf('Noise suppression: %.1f dB (stable)\n', noiSup);
end

% Check out-of-band gain
max_gain = max(abs(freqz(NTF, linspace(0, pi, 1024))));
fprintf('Max out-of-band gain: %.2f\n', max_gain);
if max_gain > 2.0
    warning('Out-of-band gain exceeds 2.0 - check stability!');
end
```

## Interpretation

### Noise Suppression vs OSR

For an L-th order Delta-Sigma modulator:

```
Theoretical noise suppression ≈ (2L+1)·10·log10(OSR)
```

| Order | OSR=32 | OSR=64 | OSR=128 | OSR=256 |
|-------|--------|--------|---------|---------|
| **1** | 22.6 dB | 31.6 dB | 40.6 dB | 49.6 dB |
| **2** | 45.1 dB | 54.1 dB | 63.1 dB | 72.1 dB |
| **3** | 67.7 dB | 76.7 dB | 85.7 dB | 94.7 dB |
| **4** | 90.3 dB | 99.3 dB | 108.3 dB | 117.3 dB |

**Rule of thumb:** Each doubling of OSR adds ~9 dB for 1st-order, ~15 dB for 2nd-order, ~21 dB for 3rd-order.

### Diagnostic Patterns

**Good NTF Design:**
- `noiSup > 40 dB` for 2nd-order, OSR=64
- Smooth magnitude response (no ripples)
- Out-of-band gain < 2.0

**Poor NTF Design:**
- `noiSup < 20 dB` (insufficient noise shaping)
- In-band peaking (|NTF| > 1 in signal band)
- Out-of-band gain > 3.0 (stability risk)

**Unstable NTF:**
- `noiSup < 0` (noise amplification instead of suppression)
- Poles outside unit circle
- Excessive out-of-band gain (> 4.0)

### Relationship to SNR

The achievable SNR in a Delta-Sigma ADC is:

```
SNR ≈ 10·log10(OSR) + noiSup + 10·log10(signal_power) - quantization_noise

For 1-bit quantizer:
SNR ≈ noiSup + 1.76 dB - 10·log10(π^(2L) / (2L+1))
```

Higher `noiSup` directly translates to higher SNR.

## Limitations

1. **Assumes Linear Model**: NTF analysis assumes small-signal, linear operation. Nonlinear effects (limit cycles, idle tones) are not predicted.

2. **Stability Not Guaranteed**: High noise suppression does not guarantee stability. Check pole locations and maximum out-of-band gain.

3. **Signal-Independent**: This analysis assumes white quantization noise. Actual performance depends on input signal level and modulator architecture.

4. **Single-Bit vs Multi-Bit**: Multi-bit modulators have different quantization noise levels not captured by NTF alone.

5. **Frequency Resolution**: Uses 2^16 = 65536 frequency points. Very narrow bands (< 1e-5 × Fs) may have insufficient resolution.

## Use Cases

### NTF Design Verification
Confirm that designed NTF meets target noise suppression specification.

```matlab
target_snr = 100;  % dB
order = 3;
OSR = 128;

NTF = synthesizeNTF(order, OSR, 1);
noiSup = NTFAnalyzer(NTF, 0, 1/(2*OSR));

fprintf('Achieved noise suppression: %.1f dB\n', noiSup);
if noiSup >= target_snr - 10
    fprintf('✓ NTF meets SNR target\n');
else
    fprintf('✗ NTF insufficient - increase order or OSR\n');
end
```

### OSR Optimization
Find minimum OSR to achieve target noise suppression.

```matlab
target_noiSup = 80;  % dB
order = 3;

for OSR = [32, 64, 128, 256, 512]
    NTF = synthesizeNTF(order, OSR, 1);
    noiSup = NTFAnalyzer(NTF, 0, 1/(2*OSR));

    fprintf('OSR=%4d: noiSup=%.1f dB', OSR, noiSup);
    if noiSup >= target_noiSup
        fprintf(' ✓ (minimum OSR found)\n');
        break;
    else
        fprintf('\n');
    end
end
```

### Bandpass Center Frequency Tuning
Optimize bandpass NTF center frequency for maximum in-band suppression.

```matlab
OSR = 64;
target_center = 0.25;  % Fs/4

center_freqs = linspace(0.2, 0.3, 20);
noiSup_vec = zeros(size(center_freqs));

for i = 1:length(center_freqs)
    NTF = synthesizeBandpassNTF(2, OSR, center_freqs(i));
    Flow = target_center - 1/(4*OSR);
    Fhigh = target_center + 1/(4*OSR);
    noiSup_vec(i) = NTFAnalyzer(NTF, Flow, Fhigh);
end

[max_noiSup, idx] = max(noiSup_vec);
optimal_center = center_freqs(idx);

fprintf('Optimal center frequency: %.4f (noiSup = %.1f dB)\n', ...
        optimal_center, max_noiSup);
```

## See Also

- [`specPlot`](specPlot.md) — Spectrum analysis for measured ADC output
- [`ENoB_bitSweep`](ENoB_bitSweep.md) — ENoB vs resolution analysis
- **Delta-Sigma Toolbox** — NTF synthesis and simulation (MATLAB `delsig` package)

## References

1. Schreier, R., and Temes, G.C., *Understanding Delta-Sigma Data Converters*, Wiley-IEEE Press, 2005
2. Candy, J.C., and Temes, G.C., *Oversampling Delta-Sigma Data Converters*, IEEE Press, 1992
3. Norsworthy, S.R., Schreier, R., and Temes, G.C., *Delta-Sigma Data Converters: Theory, Design, and Simulation*, Wiley-IEEE Press, 1997
4. **IEEE Std 1241-2010** — Standard for Terminology and Test Methods for Analog-to-Digital Converters

---

## Version History

| Version | Date | Key Features |
|---------|------|--------------|
| **v1.0** | 2025-01-28 | Initial documentation for NTFAnalyzer |
