"""Calculate INL/DNL from sine wave excitation"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, compute_inl_from_sine, analyze_spectrum, plot_dnl_inl

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# Parameters
n_bits = 10
full_scale = 1.0
fs = 800e6
fin_target = 80e6
N = 2**14  # Record length

# Nonidealities
A = 0.49
DC = 0.5
base_noise = 50e-6
hd2_dB, hd3_dB = -80, -66

# Compute HD coefficients
hd2_amp = 10**(hd2_dB/20)
hd3_amp = 10**(hd3_dB/20)
k2 = hd2_amp / (A / 2)
k3 = hd3_amp / (A**2 / 4)

print(f"[INL/DNL from Sine] [Fs = {fs/1e6:.0f} MHz, Fin = {fin_target/1e6:.0f} MHz, N = {N}]")
print(f"  [HD2 = {hd2_dB} dB, HD3 = {hd3_dB} dB, Noise = {base_noise*1e6:.1f} uV]\n")

# Generate test signal
fin, J = find_coherent_frequency(fs, fin_target, N)
t = np.arange(N) / fs
sinewave = A * np.sin(2 * np.pi * fin * t)
signal_distorted = sinewave + k2 * sinewave**2 + k3 * sinewave**3 + DC + np.random.randn(N) * base_noise

result = analyze_spectrum(signal_distorted, fs=fs, show_plot=False)

# Quantize to ADC codes
digital_output = np.round(signal_distorted * (2**n_bits) / full_scale).astype(int)
digital_output = np.clip(digital_output, 0, 2**n_bits - 1)

# Calculate INL and DNL
inl, dnl, code = compute_inl_from_sine(digital_output, num_bits=n_bits, clip_percent=0.01)

# Plot DNL and INL
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
plt.sca(ax)
plot_dnl_inl(code, dnl, inl, num_bits=n_bits)

fig.suptitle(f'INL/DNL from Sine Wave (Fs={fs/1e6:.0f} MHz, Fin={fin_target/1e6:.0f} MHz, N={N})',
             fontsize=14, fontweight='bold')
plt.subplots_adjust(left=0.1, right=0.95, top=0.88, bottom=0.1, hspace=0.4)

fig_path = output_dir / 'exp_a03_compute_inl_from_sine.png'
fig.savefig(fig_path, dpi=150)
print(f"[ENOB = {result['enob']:5.2f}] [INL: {np.min(inl):5.2f} to {np.max(inl):5.2f}] [DNL: {np.min(dnl):5.2f} to {np.max(dnl):5.2f}] LSB")
print(f"[Save fig] -> [{fig_path}]")
plt.close(fig)
