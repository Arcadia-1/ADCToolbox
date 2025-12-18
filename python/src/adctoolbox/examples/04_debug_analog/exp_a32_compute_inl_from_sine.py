"""Calculate INL/DNL from sine wave excitation"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import analyze_inl_from_sine

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# Parameters
n_bits = 12
full_scale = 1.0
Fs = 800e6
N = 2**16  # Record length
Fin = 10.1234567e6 # no need to be coherent

A = 0.49
DC = 0.5
noise_rms = 50e-6
hd2_dB, hd3_dB = -80, -66

# Compute HD coefficients
hd2_amp = 10**(hd2_dB/20)
hd3_amp = 10**(hd3_dB/20)
k2 = hd2_amp / (A / 2)
k3 = hd3_amp / (A**2 / 4)

print(f"[Sinewave] Fs=[{Fs/1e6:.2f} MHz], Fin=[{Fin/1e6:.2f} MHz], N=[{N}], A=[{A:.3f} Vpeak]")

t = np.arange(N) / Fs
sinewave = A * np.sin(2 * np.pi * Fin * t)
signal_distorted = sinewave + k2 * sinewave**2 + k3 * sinewave**3 + DC + np.random.randn(N) * noise_rms

fig, ax = plt.subplots(figsize=(6, 6))
result_inl = analyze_inl_from_sine(signal_distorted, num_bits=n_bits, full_scale=full_scale)

inl, dnl, code = result_inl['inl'], result_inl['dnl'], result_inl['code']
print(f"[analyze_inl_from_sine] [INL: {np.min(inl):5.2f}, {np.max(inl):5.2f}] LSB, [DNL: {np.min(dnl):5.2f}, {np.max(dnl):5.2f}] LSB")

plt.tight_layout()
fig_path = output_dir / 'exp_a32_compute_inl_from_sine.png'
fig.savefig(fig_path, dpi=150)
print(f"[Save fig] -> [{fig_path}]")
plt.close(fig)