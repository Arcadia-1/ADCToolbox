"""Error PDF comparison: noise, jitter, harmonic distortion, kickback"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, amplitudes_to_snr
from adctoolbox.aout.analyze_error_pdf import plot_error_pdf

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# Parameters
N = 2**13
Fs = 800e6
Fin_target = 60e6
Fin, J = find_coherent_frequency(Fs, Fin_target, N)
t = np.arange(N) / Fs
A, DC = 0.49, 0.5
base_noise = 50e-6
B = 12  # ADC resolution in bits

print(f"[Error PDF Comparison] [Fs = {Fs/1e6:.0f} MHz, Fin = {Fin/1e6:.1f} MHz, N = {N}]")
print(f"[Signal Parameters] A={A:.3f} V, DC={DC:.3f} V\n")

# Signal 1: Thermal noise
noise_rms = 180e-6
signal_noise = A * np.sin(2*np.pi*Fin*t) + DC + np.random.randn(N) * noise_rms

snr_noise = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=noise_rms)
print(f"[Noise Signal] Noise RMS=[{noise_rms*1e6:.2f} uVrms], Theoretical SNR=[{snr_noise:.2f} dB]")

# Signal 2: Jitter
jitter_rms = 1e-12
phase_jitter = np.random.randn(N) * 2 * np.pi * Fin * jitter_rms
signal_jitter = A * np.sin(2*np.pi*Fin*t + phase_jitter) + DC + np.random.randn(N) * base_noise

# Signal 3: Harmonic distortion (via static nonlinearity)
hd2_dB, hd3_dB = -80, -66
hd2_amp = 10**(hd2_dB/20)  # Harmonic amplitude / Fundamental amplitude
hd3_amp = 10**(hd3_dB/20)

# Compute nonlinearity coefficients to achieve target HD levels
# HD2: coef2 * A^2 / 2 = hd2_amp * A  →  coef2 = hd2_amp / (A/2)
# HD3: coef3 * A^3 / 4 = hd3_amp * A  →  coef3 = hd3_amp / (A^2/4)
coef2 = hd2_amp / (A / 2)
coef3 = hd3_amp / (A**2 / 4)

# Generate distorted signal: y = x + coef2*x^2 + coef3*x^3
sinewave = A * np.sin(2*np.pi*Fin*t)
signal_harmonic = sinewave + coef2 * sinewave**2 + coef3 * sinewave**3 + DC + np.random.randn(N) * base_noise

snr_harmonic = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=base_noise)
print(f"[Harmonic Signal] Noise RMS=[{base_noise*1e6:.2f} uVrms], HD2={hd2_dB}dB, HD3={hd3_dB}dB, Theoretical SNR=[{snr_harmonic:.2f} dB]")
print()

# Signal 4: Kickback
kickback_strength = 0.009
t_ext = np.arange(N+1) / Fs
sig_clean_ext = A * np.sin(2*np.pi*Fin*t_ext) + DC + np.random.randn(N+1) * base_noise
msb_ext = np.floor(sig_clean_ext * 2**4) / 2**4
lsb_ext = np.floor((sig_clean_ext - msb_ext) * 2**12) / 2**12
msb_shifted = msb_ext[:-1]
msb = msb_ext[1:]
lsb = lsb_ext[1:]
signal_kickback = msb + lsb + kickback_strength * msb_shifted

# Create 2x2 subplot
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Thermal Noise
plt.sca(axes[0, 0])
err_lsb1, mu1, sigma1, KL1, x1, fx1, gauss1 = plot_error_pdf(signal_noise, resolution=B, full_scale=1, plot=True)
axes[0, 0].set_title(f'Thermal Noise: RMS = {noise_rms*1e6:.0f} uV', fontsize=11, fontweight='bold')
print(f"  [Thermal Noise      ] μ = {mu1:6.3f} LSB, σ = {sigma1:6.3f} LSB, KL = {KL1:.4f}")

# Plot 2: Jitter
plt.sca(axes[0, 1])
err_lsb2, mu2, sigma2, KL2, x2, fx2, gauss2 = plot_error_pdf(signal_jitter, resolution=B, full_scale=1, plot=True)
axes[0, 1].set_title(f'Jitter: {jitter_rms*1e15:.1f} fs', fontsize=11, fontweight='bold')
print(f"  [Jitter             ] μ = {mu2:6.3f} LSB, σ = {sigma2:6.3f} LSB, KL = {KL2:.4f}")

# Plot 3: Harmonic Distortion
plt.sca(axes[1, 0])
err_lsb3, mu3, sigma3, KL3, x3, fx3, gauss3 = plot_error_pdf(signal_harmonic, resolution=B, full_scale=1, plot=True)
axes[1, 0].set_title(f'Harmonic Distortion: HD2 = {hd2_dB} dB, HD3 = {hd3_dB} dB', fontsize=11, fontweight='bold')
print(f"  [Harmonic Distortion] μ = {mu3:6.3f} LSB, σ = {sigma3:6.3f} LSB, KL = {KL3:.4f}")

# Plot 4: Kickback
plt.sca(axes[1, 1])
err_lsb4, mu4, sigma4, KL4, x4, fx4, gauss4 = plot_error_pdf(signal_kickback, resolution=B, full_scale=1, plot=True)
axes[1, 1].set_title(f'Kickback: Strength = {kickback_strength}', fontsize=11, fontweight='bold')
print(f"  [Kickback           ] μ = {mu4:6.3f} LSB, σ = {sigma4:6.3f} LSB, KL = {KL4:.4f}")

fig.suptitle(f'Error PDF Comparison: 4 Non-idealities (Fs = {Fs/1e6:.0f} MHz, Fin = {Fin/1e6:.1f} MHz)',
             fontsize=13, fontweight='bold')
plt.tight_layout()

fig_path = output_dir / 'exp_a40_plot_error_pdf.png'
plt.savefig(fig_path, dpi=150)
print(f"\n[Save fig] -> [{fig_path}]")
plt.close()
