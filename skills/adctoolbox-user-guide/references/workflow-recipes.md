# ADCToolbox Workflow Recipes

Four self-contained, runnable scripts demonstrating common ADC characterization tasks.
Each recipe can be run headless: `MPLBACKEND=Agg python recipe.py`

---

## Recipe 1: Basic ADC Evaluation

Generate a test signal, compute spectrum, extract ENOB/SNDR/SFDR.

```python
import numpy as np
from adctoolbox import find_coherent_frequency, analyze_spectrum

# --- Setup ---
N = 8192
Fs = 800e6
fin_target = 100e6

# Step 1: Find coherent frequency (avoids spectral leakage)
Fin, bin_idx = find_coherent_frequency(Fs, fin_target, N)
print(f"Coherent Fin = {Fin/1e6:.4f} MHz (bin {bin_idx})")

# Step 2: Generate test signal (12-bit ADC, 0-to-1 range)
t = np.arange(N) / Fs
amplitude = 0.49
dc_offset = 0.5
signal = amplitude * np.sin(2 * np.pi * Fin * t) + dc_offset

# Step 3: Add some noise
noise_rms = 50e-6
signal_noisy = signal + np.random.randn(N) * noise_rms

# Step 4: Analyze spectrum
result = analyze_spectrum(signal_noisy, fs=Fs, create_plot=False)

# Step 5: Read results from dictionary
print(f"ENOB     = {result['enob']:.2f} bits")
print(f"SNDR     = {result['sndr_dbc']:.1f} dBc")
print(f"SFDR     = {result['sfdr_dbc']:.1f} dBc")
print(f"SNR      = {result['snr_dbc']:.1f} dBc")
print(f"THD      = {result['thd_dbc']:.1f} dBc")
print(f"NSD      = {result['nsd_dbfs_hz']:.1f} dBFS/Hz")
print(f"Sig Pwr  = {result['sig_pwr_dbfs']:.1f} dBFS")
```

---

## Recipe 2: Noise Characterization with Signal Generator

Use `ADC_Signal_Generator` to sweep thermal noise and compare NSD.

```python
import numpy as np
import matplotlib.pyplot as plt
from adctoolbox import find_coherent_frequency, analyze_spectrum, snr_to_nsd
from adctoolbox.siggen import ADC_Signal_Generator

# --- Setup ---
N = 8192
Fs = 800e6
Fin, _ = find_coherent_frequency(Fs, 100e6, N)

noise_levels = [10e-6, 50e-6, 200e-6, 1000e-6]
fig, axes = plt.subplots(1, len(noise_levels), figsize=(5 * len(noise_levels), 4))

for idx, noise_rms in enumerate(noise_levels):
    # Generate signal with thermal noise
    gen = ADC_Signal_Generator(N=N, Fs=Fs, Fin=Fin, A=0.49, DC=0.5)
    sig = gen.apply_thermal_noise(noise_rms=noise_rms)

    # Analyze spectrum (plot into subplot)
    result = analyze_spectrum(sig, fs=Fs, create_plot=True,
                              show_title=False, show_label=True, ax=axes[idx])
    axes[idx].set_title(f"Noise={noise_rms*1e6:.0f} µV")

    # Compare measured NSD with theoretical
    nsd_measured = result['nsd_dbfs_hz']
    nsd_theory = snr_to_nsd(result['snr_dbc'], Fs)
    print(f"Noise={noise_rms*1e6:.0f}µV: ENOB={result['enob']:.2f}, "
          f"NSD={nsd_measured:.1f} dBFS/Hz (theory: {nsd_theory:.1f})")

plt.tight_layout()
plt.savefig("noise_sweep.png", dpi=200, bbox_inches='tight')
plt.close()
```

---

## Recipe 3: Nonlinearity Investigation

Inject known k2/k3 distortion, then extract coefficients from the distorted signal.

```python
import numpy as np
from adctoolbox import find_coherent_frequency, analyze_spectrum, fit_sine_4param
from adctoolbox import analyze_decomposition_time, analyze_error_by_phase, fit_static_nonlin
from adctoolbox.siggen import ADC_Signal_Generator

# --- Setup ---
N = 8192
Fs = 800e6
Fin, _ = find_coherent_frequency(Fs, 100e6, N)

# Step 1: Generate signal with known nonlinearity
gen = ADC_Signal_Generator(N=N, Fs=Fs, Fin=Fin, A=0.49, DC=0.5)
sig = gen.apply_static_nonlinearity(k2=0.02, k3=0.01)

# Step 2: Spectrum analysis to see harmonics
result = analyze_spectrum(sig, fs=Fs, max_harmonic=7, create_plot=False)
print(f"THD = {result['thd_dbc']:.1f} dBc")

# Step 3: Harmonic decomposition (time domain)
decomp = analyze_decomposition_time(sig, harmonic=7, create_plot=False)
print(f"Harmonic magnitudes (dB): {decomp['magnitudes_db'][:5]}")

# Step 4: Phase-domain error analysis (AM/PM separation)
phase_result = analyze_error_by_phase(sig, create_plot=False)
print(f"AM noise = {phase_result['am_noise_rms_v']:.6f} V")
print(f"PM noise = {phase_result['pm_noise_rms_v']:.6f} V")

# Step 5: Extract nonlinearity coefficients
k2_est, k3_est, fitted_sine, fitted_tf = fit_static_nonlin(sig, order=3)
print(f"Extracted k2={k2_est:.4f} (injected 0.02)")
print(f"Extracted k3={k3_est:.4f} (injected 0.01)")
```

---

## Recipe 4: SAR Calibration Flow

Simulate a sub-radix SAR ADC, calibrate bit weights, compare before/after.

```python
import numpy as np
import matplotlib.pyplot as plt
from adctoolbox import (find_coherent_frequency, freq_to_bin,
                         calibrate_weight_sine, analyze_spectrum,
                         analyze_bit_activity, analyze_weight_radix)

# --- Setup ---
N = 8192
Fs = 800e6
Fin, _ = find_coherent_frequency(Fs, 100e6, N)
t = np.arange(N) / Fs
n_bits = 12

# Step 1: Generate ideal SAR bit outputs with sub-radix weights
# Simulate weight errors (real ADC has capacitor mismatch)
ideal_weights = 2.0 ** np.arange(n_bits - 1, -1, -1)
weight_errors = 1 + 0.005 * np.random.randn(n_bits)  # 0.5% mismatch
actual_weights = ideal_weights * weight_errors

# Generate analog signal and quantize to bits
signal = 0.49 * np.sin(2 * np.pi * Fin * t) + 0.5
signal_scaled = signal * (2**n_bits - 1)

# SAR quantization
bits = np.zeros((N, n_bits), dtype=int)
residue = signal_scaled.copy()
for b in range(n_bits):
    bits[:, b] = (residue >= actual_weights[b]).astype(int)
    residue -= bits[:, b] * actual_weights[b]

# Step 2: Uncalibrated reconstruction
uncal_signal = bits @ ideal_weights / (2**n_bits - 1)

# Step 3: Check bit activity (should be ~50% for each bit)
activity = analyze_bit_activity(bits, create_plot=False)
print(f"Bit activity: {activity}")

# Step 4: Calibrate weights
cal_result = calibrate_weight_sine(bits, freq=Fin/Fs)
cal_signal = cal_result['calibrated_signal']
cal_weights = cal_result['weight']

# Step 5: Compare before and after
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
r_before = analyze_spectrum(uncal_signal, fs=Fs, create_plot=True,
                            show_label=True, ax=axes[0])
axes[0].set_title(f"Before Cal: ENOB={r_before['enob']:.2f}")

r_after = analyze_spectrum(cal_signal, fs=Fs, create_plot=True,
                           show_label=True, ax=axes[1])
axes[1].set_title(f"After Cal: ENOB={r_after['enob']:.2f}")

plt.tight_layout()
plt.savefig("calibration_comparison.png", dpi=200, bbox_inches='tight')
plt.close()

print(f"ENOB improvement: {r_before['enob']:.2f} → {r_after['enob']:.2f}")

# Step 6: Visualize weight radix
radix = analyze_weight_radix(cal_weights, create_plot=False)
print(f"Calibrated radix: {radix}")
```
