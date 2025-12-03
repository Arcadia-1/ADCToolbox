"""Debug script to investigate noise floor"""

import numpy as np

N = 8192
Fs = 1e9
J = 323
Fin = J * Fs / N

t = np.arange(N) / Fs

# Generate clean signal
A_fundamental = 0.45
A_HD2 = 0.05
A_HD3 = 0.02
A_HD4 = 0.01

phi_fundamental = 0.5
phi_HD2 = 1.2
phi_HD3 = -0.8
phi_HD4 = 0.3

signal_clean = (A_fundamental * np.sin(2*np.pi*Fin*t + phi_fundamental) +
                A_HD2 * np.sin(2*np.pi*2*Fin*t + phi_HD2) +
                A_HD3 * np.sin(2*np.pi*3*Fin*t + phi_HD3) +
                A_HD4 * np.sin(2*np.pi*4*Fin*t + phi_HD4))

# Add DC offset
signal_clean = signal_clean + 0.5

# Add noise
noise = np.random.randn(N) * 1e-5
signal = signal_clean + noise

print(f"Input noise RMS: {np.std(noise):.3e}")
print(f"Input noise dB: {20*np.log10(np.std(noise)):.2f} dB")
print()

# Normalize (matching LMS mode)
maxSignal = np.max(signal) - np.min(signal)
sig_avg = signal - np.mean(signal)
sig_avg = sig_avg / maxSignal

print(f"maxSignal: {maxSignal:.6f}")
print(f"After normalization, signal std: {np.std(sig_avg):.6f}")
print()

# Find frequency
spec_temp = np.fft.fft(sig_avg, N)
spec_temp[0] = 0
spec_mag = np.abs(spec_temp[:N // 2])
bin_idx = np.argmax(spec_mag)

sig_e = np.log10(spec_mag[bin_idx] + 1e-20)
sig_l = np.log10(spec_mag[bin_idx - 1] + 1e-20)
sig_r = np.log10(spec_mag[bin_idx + 1] + 1e-20)
bin_r = bin_idx + (sig_r - sig_l) / (2 * sig_e - sig_l - sig_r) / 2

freq = bin_r / N

print(f"Bin index: {bin_idx} (expected {J})")
print(f"Refined bin: {bin_r:.6f}")
print(f"Detected freq: {freq:.10f}")
print(f"Expected freq: {Fin/Fs:.10f}")
print(f"Freq error: {abs(freq - Fin/Fs):.3e}")
print()

# Prepare for fitting
harmonic = 4
t_idx = np.arange(N)

# Test with EXACT frequency
print("=" * 60)
print("Testing with EXACT frequency:")
print("=" * 60)
freq_exact = Fin / Fs

# Build basis with exact frequency
SI_exact = np.zeros((N, harmonic))
SQ_exact = np.zeros((N, harmonic))
for ii in range(harmonic):
    SI_exact[:, ii] = np.cos(t_idx * freq_exact * (ii + 1) * 2 * np.pi)
    SQ_exact[:, ii] = np.sin(t_idx * freq_exact * (ii + 1) * 2 * np.pi)

A_exact = np.hstack([SI_exact, SQ_exact])
W_exact = np.linalg.lstsq(A_exact, sig_avg, rcond=None)[0]

signal_all_exact = A_exact @ W_exact
residual_exact = sig_avg - signal_all_exact
noise_power_exact = np.sqrt(np.mean(residual_exact**2))
noise_dB_exact = 20 * np.log10(noise_power_exact)

print(f"Residual RMS (exact freq): {noise_power_exact:.3e}")
print(f"Residual dB (exact freq): {noise_dB_exact:.2f} dB")
print()

print("=" * 60)
print("Testing with ESTIMATED frequency:")
print("=" * 60)

# Build basis with estimated frequency
SI = np.zeros((N, harmonic))
SQ = np.zeros((N, harmonic))
for ii in range(harmonic):
    SI[:, ii] = np.cos(t_idx * freq * (ii + 1) * 2 * np.pi)
    SQ[:, ii] = np.sin(t_idx * freq * (ii + 1) * 2 * np.pi)

# Least squares fit
A = np.hstack([SI, SQ])
W = np.linalg.lstsq(A, sig_avg, rcond=None)[0]

# Reconstruct
signal_all = A @ W

# Calculate residual
residual = sig_avg - signal_all
noise_power = np.sqrt(np.mean(residual**2))
noise_dB = 20 * np.log10(noise_power)

print(f"Residual RMS: {noise_power:.3e}")
print(f"Residual dB: {noise_dB:.2f} dB")
print()

# Compare to normalized input noise
noise_normalized = noise / maxSignal
noise_normalized_rms = np.std(noise_normalized)
print(f"Normalized input noise RMS: {noise_normalized_rms:.3e}")
print(f"Normalized input noise dB: {20*np.log10(noise_normalized_rms):.2f} dB")
print()

# Check magnitudes
print("Fitted harmonic magnitudes:")
for ii in range(harmonic):
    I_weight = W[ii]
    Q_weight = W[ii + harmonic]
    mag = np.sqrt(I_weight**2 + Q_weight**2) * maxSignal
    print(f"  H{ii+1}: {mag:.6f}")
