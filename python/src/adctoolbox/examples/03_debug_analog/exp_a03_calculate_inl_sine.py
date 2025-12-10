"""INL/DNL sweep with different record lengths"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, compute_inl_from_sine, analyze_spectrum

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# Parameters
n_bits = 10
full_scale = 1.0
fs = 800e6
fin_target = 80e6

# Nonidealities (same as exp_a01)
A = 0.49
DC = 0.5
base_noise = 50e-6
hd2_dB, hd3_dB = -80, -66

# Compute HD coefficients
hd2_amp = 10**(hd2_dB/20)
hd3_amp = 10**(hd3_dB/20)
k2 = hd2_amp / (A / 2)
k3 = hd3_amp / (A**2 / 4)

N_list = [2**i for i in range(10, 18, 2)]  # [2^8, 2^10, 2^12, 2^14, 2^16]
n_plots = len(N_list)
fig_width = n_plots * 4
fig, axes = plt.subplots(2, n_plots, figsize=(fig_width, 6))

print(f"[INL/DNL Sweep] [Fs = {fs/1e6:.0f} MHz, Fin = {fin_target/1e6:.0f} MHz]")
print(f"  [HD2 = {hd2_dB} dB, HD3 = {hd3_dB} dB, Noise = {base_noise*1e6:.1f} uV]\n")

for idx, N in enumerate(N_list):

    # Generate signal with distortion
    fin, J = find_coherent_frequency(fs, fin_target, N)
    t = np.arange(N) / fs
    sinewave = A * np.sin(2 * np.pi * fin * t)
    signal_distorted = sinewave + k2 * sinewave**2 + k3 * sinewave**3 + DC + np.random.randn(N) * base_noise

    result = analyze_spectrum(signal_distorted, fs=fs)
    enob = result['enob']

    # Quantize to ADC codes
    digital_output = np.round(signal_distorted * (2**n_bits) / full_scale).astype(int)
    digital_output = np.clip(digital_output, 0, 2**n_bits - 1)

    # Calculate INL and DNL
    inl, dnl, code = compute_inl_from_sine(digital_output, num_bits=n_bits, clip_percent=0.01)

    # Plot DNL (top row)
    axes[0, idx].plot(code, dnl, 'r-', linewidth=0.5)
    axes[0, idx].axhline(0, color='k', linestyle='--', linewidth=0.5)
    axes[0, idx].grid(True, alpha=0.3)
    axes[0, idx].set_xlabel('Code (LSB)')
    axes[0, idx].set_ylabel('DNL (LSB)')
    axes[0, idx].set_title(f'{n_bits}-bit ADC, N = 2^{int(np.log2(N))}\nDNL: [{np.min(dnl):5.2f}, {np.max(dnl):5.2f}] LSB', fontweight='bold')
    axes[0, idx].set_xlim([0, 2**n_bits])

    # Plot INL (bottom row)
    axes[1, idx].plot(code, inl, 'b-', linewidth=0.5)
    axes[1, idx].axhline(0, color='k', linestyle='--', linewidth=0.5)
    axes[1, idx].grid(True, alpha=0.3)
    axes[1, idx].set_xlabel('Code (LSB)')
    axes[1, idx].set_ylabel('INL (LSB)')
    axes[1, idx].set_title(f'INL: [{np.min(inl):5.2f}, {np.max(inl):5.2f}] LSB', fontweight='bold')
    axes[1, idx].set_xlim([0, 2**n_bits])

    print(f"  [N = 2^{int(np.log2(N)):2d} = {N:5d}] [ENOB = {enob:5.2f}] [INL: {np.min(inl):5.2f} to {np.max(inl):5.2f}] [DNL: {np.min(dnl):5.2f} to {np.max(dnl):5.2f}] LSB")

plt.tight_layout()
fig_path = output_dir / 'exp_a03_compute_inl_from_sine.png'
plt.savefig(fig_path, dpi=150)
print(f"\n[Save fig] -> [{fig_path}]")
plt.close()
