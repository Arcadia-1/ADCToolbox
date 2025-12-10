"""Harmonic decomposition: thermal noise vs static nonlinearity"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, decompose_harmonics

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N = 2**13
Fs = 800e6
Fin_target = 10e6
Fin, Fin_bin = find_coherent_frequency(Fs, Fin_target, N)
t = np.arange(N) / Fs
A = 0.49

sig_ideal = A * np.sin(2 * np.pi * Fin * np.arange(N) / Fs)
print(f"[Harmonic Decomposition] Fs={Fs/1e6:.1f} MHz, Fin={Fin/1e6:.6f} MHz, Bin={Fin_bin}, N_fft={N}")

# Case 1: Thermal noise only
noise_rms = 500e-6
signal_noise = sig_ideal + np.random.randn(N) * noise_rms

# Case 2: Static nonlinearity (k2 and k3) + base noise
k2 = 0.001
k3 = 0.005
base_noise_rms = 50e-6
signal_nonlin = sig_ideal + k2 * sig_ideal**2 + k3 * sig_ideal**3 + np.random.randn(N) * base_noise_rms

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle('Harmonic Decomposition', fontsize=16, fontweight='bold')

plt.sca(ax1)
decompose_harmonics(signal_noise, re_fin=Fin/Fs, order=10, disp=1)
ax1.set_title(f'Case 1: Thermal Noise ({noise_rms*1e6:.0f}uV RMS)', fontsize=14, fontweight='bold')

plt.sca(ax2)
decompose_harmonics(signal_nonlin, re_fin=Fin/Fs, order=10, disp=1)
ax2.set_title(f'Case 2: Static Nonlinearity (k2={k2:.3f}, k3={k3:.3f})', fontsize=14, fontweight='bold')

plt.tight_layout()
fig_path = output_dir / 'exp_a10_decompose_harmonics.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\n[Save fig] -> [{fig_path}]\n")
plt.close(fig)