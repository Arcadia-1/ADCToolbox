"""Calculate INL/DNL from sine wave excitation (modular: separate compute and plot)"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, compute_inl_from_sine, plot_dnl_inl, amplitudes_to_snr, snr_to_nsd

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# Parameters
n_bits = 12
full_scale = 1.0
Fs = 800e6
N = 2**16  # Record length
Fin, Fin_bin = find_coherent_frequency(Fs, 123.456e6, N)

A = 0.49
DC = 0.5
noise_rms = 50e-6
hd2_dB, hd3_dB = -80, -66

# Compute HD coefficients
hd2_amp = 10**(hd2_dB/20)
hd3_amp = 10**(hd3_dB/20)
k2 = hd2_amp / (A / 2)
k3 = hd3_amp / (A**2 / 4)

snr_ref = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=noise_rms)
nsd_ref = snr_to_nsd(snr_ref, fs=Fs, osr=1)
print(f"[Sinewave] Fs=[{Fs/1e6:.2f} MHz], Fin=[{Fin/1e6:.2f} MHz], Bin/N=[{Fin_bin}/{N}], A=[{A:.3f} Vpeak]")
print(f"[Nonideal] Noise RMS=[{noise_rms*1e6:.2f} uVrms], Theoretical SNR=[{snr_ref:.2f} dB], Theoretical NSD=[{nsd_ref:.2f} dBFS/Hz]\n")

t = np.arange(N) / Fs
sinewave = A * np.sin(2 * np.pi * Fin * t)
signal_distorted = sinewave + k2 * sinewave**2 + k3 * sinewave**3 + DC + np.random.randn(N) * noise_rms

# --- MODULAR APPROACH: separate compute and plot ---
# Step 1: Compute INL/DNL (passing analog signal with full_scale)
inl, dnl, code = compute_inl_from_sine(signal_distorted, num_bits=n_bits, full_scale=full_scale, clip_percent=0.01)
print(f"[analyze_inl_from_sine] [INL: {np.min(inl):5.2f}, {np.max(inl):5.2f}] LSB, [DNL: {np.min(dnl):5.2f}, {np.max(dnl):5.2f}] LSB")

# Step 2: Create figure and plot DNL/INL
fig, ax = plt.subplots(figsize=(6, 6))
plot_dnl_inl(code, dnl, inl, num_bits=n_bits)

plt.tight_layout()
fig_path = output_dir / 'exp_a12_compute_inl_from_sine_modular.png'
fig.savefig(fig_path, dpi=150)
print(f"[INL/DNL] [INL: {np.min(inl):5.2f}, {np.max(inl):5.2f}] LSB, [DNL: {np.min(dnl):5.2f}, {np.max(dnl):5.2f}] LSB")
print(f"[Save fig] -> [{fig_path}]")
plt.close(fig)
