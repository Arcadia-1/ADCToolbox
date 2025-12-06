"""Thompson decomposition: sinewave with noise, glitch, error bit alignment"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_bin, tom_decomp

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N = 2**13
Fs = 800e6
Fin_target = 80e6
J = find_bin(Fs, Fin_target, N)
Fin = J * Fs / N
t = np.arange(N) / Fs
A, DC = 0.49, 0.5

print(f"[Thompson Decomposition] [Fs = {Fs/1e6:.0f} MHz, Fin = {Fin/1e6:.1f} MHz, N = {N}]")

# Create signal with multiple error types
np.random.seed(42)

# Base signal with noise
noise_rms = 80e-6
signal = A * np.sin(2*np.pi*Fin*t) + DC + np.random.randn(N) * noise_rms

# Add harmonic distortion (3rd and 5th)
hd3_amp = 0.01 * A
hd5_amp = 0.005 * A
signal += hd3_amp * np.sin(3*2*np.pi*Fin*t) + hd5_amp * np.sin(5*2*np.pi*Fin*t)

# Add glitch (plan requirement)
glitch_idx = N // 3
glitch_amplitude = 0.05
signal[glitch_idx:glitch_idx+20] += glitch_amplitude

# Perform Thompson decomposition
fundamental_signal, total_error, harmonic_error, residual_error = tom_decomp(
    signal, re_fin=Fin/Fs, order=10, disp=0
)

# Calculate RMS values
rms_total = np.sqrt(np.mean(total_error**2))
rms_harmonic = np.sqrt(np.mean(harmonic_error**2))
rms_residual = np.sqrt(np.mean(residual_error**2))

print(f"\n[Decomposition Results]")
print(f"  Total Error RMS:     {rms_total*1e6:.2f} uV")
print(f"  Harmonic Error RMS:  {rms_harmonic*1e6:.2f} uV (correlated)")
print(f"  Residual Error RMS:  {rms_residual*1e6:.2f} uV (uncorrelated)")
print(f"  Harmonic/Total ratio: {rms_harmonic/rms_total*100:.1f}%")

# Create 2x2 visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Top-left: Original signal
axes[0, 0].plot(t*1e6, signal, 'b-', linewidth=0.8)
axes[0, 0].set_xlabel('Time (us)', fontsize=11)
axes[0, 0].set_ylabel('Signal (V)', fontsize=11)
axes[0, 0].set_title('Original Signal (with noise, harmonics, glitch)', fontsize=11, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xlim([0, t[-1]*1e6])

# Top-right: Fundamental reconstruction
axes[0, 1].plot(t*1e6, signal, 'b-', linewidth=0.5, alpha=0.3, label='Original')
axes[0, 1].plot(t*1e6, fundamental_signal, 'r-', linewidth=1.5, label='Fundamental')
axes[0, 1].set_xlabel('Time (us)', fontsize=11)
axes[0, 1].set_ylabel('Signal (V)', fontsize=11)
axes[0, 1].set_title('Fundamental Signal (DC + F1)', fontsize=11, fontweight='bold')
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xlim([0, t[-1]*1e6])

# Bottom-left: Total error
axes[1, 0].plot(t*1e6, total_error*1e6, 'k-', linewidth=0.8)
axes[1, 0].set_xlabel('Time (us)', fontsize=11)
axes[1, 0].set_ylabel('Error (uV)', fontsize=11)
axes[1, 0].set_title(f'Total Error (RMS = {rms_total*1e6:.2f} uV)', fontsize=11, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xlim([0, t[-1]*1e6])

# Bottom-right: Harmonic vs Residual errors
axes[1, 1].plot(t*1e6, harmonic_error*1e6, 'r-', linewidth=1, alpha=0.7, label=f'Harmonic ({rms_harmonic*1e6:.1f} uV)')
axes[1, 1].plot(t*1e6, residual_error*1e6, 'b-', linewidth=0.8, alpha=0.5, label=f'Residual ({rms_residual*1e6:.1f} uV)')
axes[1, 1].set_xlabel('Time (us)', fontsize=11)
axes[1, 1].set_ylabel('Error (uV)', fontsize=11)
axes[1, 1].set_title('Error Decomposition', fontsize=11, fontweight='bold')
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xlim([0, t[-1]*1e6])

plt.tight_layout()
fig_path = output_dir / f'exp_a13_tom_decomp_fin_{int(Fin/1e6)}M.png'
plt.savefig(fig_path, dpi=150)
plt.close()

print(f"\n[Save fig] -> [{fig_path}]")
