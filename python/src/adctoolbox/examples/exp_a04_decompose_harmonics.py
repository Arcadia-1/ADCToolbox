"""Harmonic decomposition: thermal noise vs static nonlinearity"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_bin, decompose_harmonics

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N = 2**13
Fs = 800e6
Fin_target = 10e6
J = find_bin(Fs, Fin_target, N)
Fin = J * Fs / N
t = np.arange(N) / Fs
A = 0.49

print(f"[Harmonic Decomposition] [Fs = {Fs/1e6:.0f} MHz, Fin = {Fin/1e6:.1f} MHz, N = {N}]\n")

# Case 1: Thermal noise only
noise_rms = 500e-6
signal_noise = A * np.sin(2*np.pi*Fin*t) + np.random.randn(N) * noise_rms

# Case 2: Static nonlinearity (k2 and k3) + base noise
k2 = 0.001
k3 = 0.005
base_noise_rms = 50e-6
x_ideal = A * np.sin(2*np.pi*Fin*t)
signal_nonlin = x_ideal + k2 * x_ideal**2 + k3 * x_ideal**3 + np.random.randn(N) * base_noise_rms

# Create 1x2 subplot for side-by-side comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle('Harmonic Decomposition',
             fontsize=16, fontweight='bold')

# Case 1: Thermal Noise - use built-in decompose_harmonics plotting
plt.sca(ax1)
decompose_harmonics(signal_noise, re_fin=Fin/Fs, order=10, disp=1)
ax1.set_title(f'Case 1: Thermal Noise ({noise_rms*1e6:.0f}uV RMS)', fontsize=13, fontweight='bold')

# Case 2: Static Nonlinearity - use built-in decompose_harmonics plotting
plt.sca(ax2)
decompose_harmonics(signal_nonlin, re_fin=Fin/Fs, order=10, disp=1)
ax2.set_title(f'Case 2: Static Nonlinearity (k2={k2:.3f}, k3={k3:.3f})', fontsize=13, fontweight='bold')

plt.tight_layout()
fig_path = output_dir / 'exp_a04_decompose_harmonics.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"[Save fig] -> [{fig_path}]\n")
plt.close(fig)