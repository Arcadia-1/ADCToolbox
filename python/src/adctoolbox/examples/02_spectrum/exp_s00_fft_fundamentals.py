"""Demonstrate the raw principles of the Discrete Fourier Transform (FFT) without windows.
Specifically addressing how DC, typical frequencies, and Nyquist frequencies are calculated,
and answering whether they should be multiplied by 2 when forming a single-sided spectrum.

Concepts covered:
1. Two-sided FFT magnitude and symmetry.
2. Why sine waves split their amplitudes across positive and negative frequencies.
3. Why DC (Bin 0) and Nyquist (Bin N/2) do NOT split, and thus are NOT multiplied by 2
   in a single-sided amplitude spectrum.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setup output directory
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# 1. Define simulation parameters (Extremely small N for clarity)
N = 16
Fs = 16.0  # 16 Hz, so bin k corresponds exactly to k Hz
t = np.arange(N) / Fs

# 2. Define signal components
# - DC component: A_dc = 1.0 (Bin 0)
# - Mid component: A_mid = 2.0 at 2 Hz (Bin 2)
# - Nyq component: A_nyq = 1.5 at 8 Hz (Bin 8 = N/2)
A_dc = 1.0
A_mid = 2.0
A_nyq = 1.5

k_mid = 2
k_nyq = N // 2  # 8

# Generate the actual time-domain signal
# Note: A cosine at Nyquist is exactly: cos(pi * n) = [1, -1, 1, -1, ...]
signal = A_dc + A_mid * np.sin(2 * np.pi * k_mid * t) + A_nyq * np.cos(2 * np.pi * k_nyq * t)

# 3. Compute raw FFT
# The raw FFT equation is X[k] = sum(x[n] * exp(-j*2*pi*k*n/N))
fft_raw = np.fft.fft(signal)
fft_mag = np.abs(fft_raw)

# Calculate theoretical raw FFT bin magnitudes
# - Sine splits into 2 complex exponentials: magnitude becomes A_mid * N / 2
# - DC and Nyquist do NOT split: magnitude is A * N
expected_raw = np.zeros(N)
expected_raw[0] = A_dc * N
expected_raw[k_mid] = (A_mid * N) / 2
expected_raw[N - k_mid] = (A_mid * N) / 2
expected_raw[k_nyq] = A_nyq * N

print("=" * 70)
print("1. RAW TWO-SIDED FFT (0 to Fs)")
print("=" * 70)
print(f"Signal components: DC={A_dc}V, Mid({k_mid}Hz)={A_mid}V, Nyquist({k_nyq}Hz)={A_nyq}V")
print(f"{'Bin':<5} | {'Freq (Hz)':<10} | {'Raw Mag':<10} | {'Expected Mag (NxA or Nx(A/2))'}")
print("-" * 70)
for k in range(N):
    freq = k * (Fs / N)
    print(f"{k:<5} | {freq:<10.1f} | {fft_mag[k]:<10.2f} | {expected_raw[k]:.2f}")


# 4. Form Single-Sided Amplitude Spectrum (0 to Fs/2)
# To recover the TRUE voltage amplitude A of each sine wave:
# 1. Divide all raw bins by N
# 2. Multiply bins 1 to (N/2 - 1) by 2 (because those frequencies were split in the 2-sided FFT)
# 3. DO NOT multiply Bin 0 (DC) or Bin N/2 (Nyquist) by 2!

single_sided_mag = np.zeros(N // 2 + 1)
for k in range(N // 2 + 1):
    mag_normalized = fft_mag[k] / N
    if k == 0 or k == (N // 2):
        single_sided_mag[k] = mag_normalized  # Do NOT multiply by 2
    else:
        single_sided_mag[k] = mag_normalized * 2  # Recover split energy

print("\n" + "=" * 70)
print("2. RECONSTRUCTED SINGLE-SIDED VOLTAGE AMPLITUDE SPECTRUM (0 to Fs/2)")
print("=" * 70)
print("Rule:")
print("- Amplitude [k] = (Raw / N) * 2   if  0 < k < N/2")
print("- Amplitude [k] = (Raw / N)       if  k == 0 or k == N/2")
print(f"\n{'Bin':<5} | {'Freq (Hz)':<10} | {'Reconstructed Amplitude (V)'}")
print("-" * 70)
for k in range(N // 2 + 1):
    freq = k * (Fs / N)
    print(f"{k:<5} | {freq:<10.1f} | {single_sided_mag[k]:.3f} V")

# 5. Plot the result
fig, axes = plt.subplots(3, 1, figsize=(10, 10))

# Subplot 1: Time domain
axes[0].plot(t, signal, 'ko-', label='Sampled Signal')
axes[0].set_title('Time Domain Signal (N=16)')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Amplitude (V)')
axes[0].grid(True)
axes[0].legend()

# Subplot 2: Two-sided Fast Fourier Transform
k_all = np.arange(N)
freq_all = k_all * (Fs / N)
markerline, stemlines, baseline = axes[1].stem(freq_all, fft_mag, basefmt=" ")
plt.setp(stemlines, 'linewidth', 2)
plt.setp(markerline, 'markersize', 6)
axes[1].set_title(f"Raw Two-sided FFT Magnitude (0 to Fs)\nNotice Mid({k_mid}Hz) splits into 2Hz and {16-k_mid}Hz. DC and Nyquist ({k_nyq}Hz) do not.")
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel('Raw Magnitude')
axes[1].grid(True)
axes[1].set_xticks(freq_all)

# Subplot 3: Single-sided True Amplitude
k_half = np.arange(N // 2 + 1)
freq_half = k_half * (Fs / N)
markerline, stemlines, baseline = axes[2].stem(freq_half, single_sided_mag, basefmt=" ")
plt.setp(stemlines, 'linewidth', 2)
plt.setp(markerline, 'markersize', 6)
axes[2].set_title("Single-Sided True Voltage Amplitude (0 to Fs/2)\nMultiplied by 2 everywhere EXCEPT DC (0Hz) and Nyquist (8Hz)")
axes[2].set_xlabel('Frequency (Hz)')
axes[2].set_ylabel('True Amplitude (V)')
axes[2].grid(True)
axes[2].set_xticks(freq_half)

plt.tight_layout()
out_path = output_dir / "exp_s00_fft_fundamentals.png"
plt.savefig(out_path, dpi=120)
print(f"\n[Plot saved] -> {out_path}")
plt.close()
