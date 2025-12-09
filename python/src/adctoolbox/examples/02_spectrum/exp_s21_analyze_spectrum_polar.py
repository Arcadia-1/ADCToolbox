"""
Polar phase spectrum analysis: visualize FFT bins as magnitude-phase vectors in polar coordinates.
Demonstrates harmonic distortion with positive and negative k3 coefficients showing how
HD2 and HD3 phases vary with nonlinearity polarity.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import calculate_coherent_freq, calculate_snr_from_amplitude, snr_to_nsd, analyze_spectrum_polar

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N = 2**13
Fs = 800e6
Fin_target = 80e6
Fin, J = calculate_coherent_freq(Fs, Fin_target, N)
t = np.arange(N) / Fs
A, DC = 0.49, 0.5
base_noise = 50e-6

# Harmonic distortion levels
hd2_dB, hd3_dB = -80, -66
hd2_amp = 10**(hd2_dB/20)  # Harmonic amplitude / Fundamental amplitude
hd3_amp = 10**(hd3_dB/20)

# Compute nonlinearity coefficients to achieve target HD levels
# For y = x + k2*x² + k3*x³:
# HD2: k2 * A^2 / 2 = hd2_amp * A  →  k2 = hd2_amp / (A/2)
# HD3: k3 * A^3 / 4 = hd3_amp * A  →  k3 = hd3_amp / (A²/4)
k2 = hd2_amp / (A / 2)
k3_pos = hd3_amp / (A**2 / 4)
k3_neg = -k3_pos

# Signal 1: Harmonic distortion with positive k3
sig_ideal = A * np.sin(2*np.pi*Fin*t)
signal_hd_pos_k3 = (sig_ideal + k2 * sig_ideal**2 + k3_pos * sig_ideal**3 + DC + np.random.randn(N) * base_noise)

# Signal 2: Harmonic distortion with negative k3
signal_hd_neg_k3 = (sig_ideal + k2 * sig_ideal**2 + k3_neg * sig_ideal**3 + DC + np.random.randn(N) * base_noise)

signals = [signal_hd_pos_k3, signal_hd_neg_k3]
titles = ['HD: Positive k3', 'HD: Negative k3']
params = [f'HD2={hd2_dB} dB, HD3={hd3_dB} dB\nk2={k2:.4e}, k3={k3_pos:.4e}',
          f'HD2={hd2_dB} dB, HD3={hd3_dB} dB\nk2={k2:.4e}, k3={k3_neg:.4e}']

snr_ref = calculate_snr_from_amplitude(sig_amplitude=A, noise_amplitude=base_noise)
nsd_ref = snr_to_nsd(snr_ref, fs=Fs, osr=1)
print(f"[Sinewave] Fs=[{Fs/1e6:.0f} MHz], Fin=[{Fin/1e6:.1f} MHz] (coherent, Bin {J}), N=[{N}], A=[{A:.3f} Vpeak]")
print(f"[Base Noise] RMS=[{base_noise*1e6:.2f} uVrms], Theoretical SNR=[{snr_ref:.2f} dB], Theoretical NSD=[{nsd_ref:.2f} dBFS/Hz]")
print(f"[Nonlinearity] HD2=[{hd2_dB} dB], HD3=[{hd3_dB} dB]\n")

# Calculate theoretical SNR for harmonic distortion
thd_power_theory = hd2_amp**2 + hd3_amp**2  # Total harmonic power
# For harmonic distortion signal, include base noise
total_noise_amp_hd = np.sqrt(thd_power_theory + (base_noise/A)**2)  # Normalized to signal amplitude
snr_thd_theory = -20 * np.log10(total_noise_amp_hd)  # SNR limited by THD + base noise

# Create 1x2 figure for comparison
fig = plt.figure(figsize=(14, 6))

# Store axes and their limits for restoration after tight_layout
axes_info = []
measured_snrs = []  # Store measured SNR for comparison

for i, (signal, title, param) in enumerate(zip(signals, titles, params)):
    # Create individual subplot with polar projection
    ax = fig.add_subplot(1, 2, i+1, projection='polar')

    # Analyze spectrum with polar phase visualization
    coherent_result = analyze_spectrum_polar(
        signal,
        fs=Fs,
        show_plot=False,
        ax=ax,
        fixed_radial_range=120
    )

    ax.set_title(f'{param}', pad=20, fontsize=14, fontweight='bold')

    # Store axis and its ylim for later restoration
    axes_info.append((ax, ax.get_ylim()))

    metrics = coherent_result.get('metrics', {})
    snr_measured = metrics.get('snr_db', 0)
    measured_snrs.append(snr_measured)
    print(f"[{title:20s}] SNR=[{snr_measured:6.2f} dB], Bin=[{coherent_result['bin_idx']}], Noise floor=[{coherent_result['minR_dB']:7.2f} dB]")

print(f"\n[Theoretical SNR] Harmonic Distortion=[{snr_thd_theory:.1f} dB]")

plt.tight_layout()

# Restore ylim after tight_layout (which resets polar axis limits)
for ax, ylim in axes_info:
    ax.set_ylim(ylim)

fig_path = output_dir / 'exp_s21_analyze_spectrum_polar.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\n[Save fig] -> [{fig_path}]")
plt.close()
