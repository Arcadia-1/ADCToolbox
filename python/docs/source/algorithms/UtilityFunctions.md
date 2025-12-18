# Utility Functions

## Overview

This document describes helper functions used internally by ADCToolbox analysis tools.

---

## alias

### Purpose
Computes aliased frequency bin after Nyquist folding.

### Syntax
```matlab
bin = alias(J, N)
```

### Algorithm
```
if floor(J/N × 2) is even:
    bin = J - floor(J/N) × N
else:
    bin = N - J + floor(J/N) × N  % Nyquist fold
```

### Example
```matlab
% 3rd harmonic of bin 200 in 1024-point FFT
bin_h3 = alias(200 × 3, 1024);  % Returns 600

% Above Nyquist: wraps back
bin_high = alias(800, 1024);  % Returns 224 (folded)
```

### Use Cases
- Locate harmonics in FFT (used by `specPlot`, `specPlotPhase`)
- Handle frequency wrapping above Nyquist

---

## findBin

### Purpose
Finds coherent bin index for given frequency, ensuring `gcd(bin, N) = 1` (prime factorization for coherence).

### Syntax
```matlab
bin = findBin(Fs, Fin, N)
```

### Algorithm
```
bin_init = floor(Fin/Fs × N)
while gcd(bin, N) > 1:  % Not coprime
    bin += 1
```

### Example
```matlab
Fs = 1e9;  % 1 GHz sampling
Fin = 123.4e6;  % 123.4 MHz signal
N = 1024;  % FFT length

bin = findBin(Fs, Fin, N);  % Returns nearest coherent bin
```

### Use Cases
- Generate coherent test tones for ADC testing
- Avoid spectral leakage in FFT analysis

---

## findFin

### Purpose
Estimates input frequency using `sineFit`. Wrapper for frequency detection.

### Syntax
```matlab
fin = findFin(data)
fin = findFin(data, Fs)
```

### Algorithm
```
[~, freq, ~, ~, ~] = sineFit(data)
fin = freq × Fs
```

### Example
```matlab
data = adc_output;
fin = findFin(data, 1e9);  % Returns frequency in Hz
fprintf('Detected input: %.3f MHz\n', fin/1e6);
```

### Use Cases
- Auto-detect signal frequency before calibration
- Verify test tone frequency

---

## cap2weight

### Purpose
Calculates SAR ADC bit weights from capacitor network topology (DAC caps, bridge caps, parasitic caps).

### Syntax
```matlab
[weight, Co] = cap2weight(Cd, Cb, Cp)
```

### Inputs
- **`Cd`** — DAC bit capacitors [LSB ... MSB]
- **`Cb`** — Bridge capacitors [LSB ... MSB] (0 = no bridge)
- **`Cp`** — Parasitic capacitors [LSB ... MSB]

### Outputs
- **`weight`** — Bit weights [LSB ... MSB]
- **`Co`** — Output node capacitance

### Algorithm
Iteratively computes charge redistribution from LSB to MSB:
```
For each bit i (LSB to MSB):
    Cs = Cp(i) + Cd(i) + Cl  % Total switched cap
    weight(i) = Cd(i) / Cs    % Voltage division ratio
    Update Cl via bridge network
```

### Topology
```
MSB <---||--------||---< LSB
      Cb   |    |   Cl
          ---  ---
      Cp  ---  ---  Cd
           |    |
          gnd   Vi
```

### Example
```matlab
% 10-bit binary-weighted DAC
Cd = 2.^(0:9);  % [1, 2, 4, ..., 512] fF
Cb = zeros(1,10);  % No bridge caps
Cp = ones(1,10) × 0.1;  % 0.1 fF parasitic

[weight, Co] = cap2weight(Cd, Cb, Cp);
fprintf('Actual weights: %s\n', mat2str(weight, 4));
fprintf('Output cap: %.2f fF\n', Co);
```

### Use Cases
- Predict non-binary weight for SAR ADC with parasitics
- Design bridge-cap networks for redundancy
- Input to `FGCalSine` as `nomWeight` for rank patching

---

## bitInBand

### Purpose
Bandpass filter in frequency domain by keeping only specified frequency bands.

### Syntax
```matlab
dout = bitInBand(din, bands)
```

### Inputs
- **`din`** — Input signal (N×M matrix, each column filtered independently)
- **`bands`** — Frequency bands to keep (P×2 matrix)
  - Each row: `[f_low, f_high]` in normalized frequency (0-1)

### Algorithm
```
1. FFT: spec = fft(din)
2. Create mask = 0 everywhere
3. For each band [f_low, f_high]:
       Set mask[f_low×N : f_high×N] = 1
       Handle aliasing for negative frequencies
4. Apply: spec_filtered = spec .× mask
5. IFFT: dout = real(ifft(spec_filtered))
```

### Example
```matlab
% Keep only 0.1-0.2 and 0.3-0.4 normalized freq
bands = [0.1, 0.2; 0.3, 0.4];
filtered = bitInBand(data, bands);

% Remove DC and low-freq noise (high-pass)
bands = [0.01, 0.5];  % Keep 0.01-0.5 (exclude DC)
hp_filtered = bitInBand(data, bands);
```

### Use Cases
- Isolate signal+harmonics, remove out-of-band noise
- Create band-limited test signals
- Remove DC offset or low-frequency drift

---

## See Also

- [`FGCalSine`](FGCalSine.md) — Uses `cap2weight` for `nomWeight` estimation
- [`specPlot`](specPlot.md) — Uses `alias` for harmonic detection
- [`sineFit`](sineFit.md) — Called by `findFin` for frequency detection

---

## Version History

| Version | Date | Key Features |
|---------|------|--------------|
| **v1.0** | 2025-01-26 | Initial utility functions documentation |
