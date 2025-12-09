"""
Polar phase spectrum analysis: visualize FFT bins as magnitude-phase vectors in polar coordinates.
Demonstrates 4 ADC impairments: noise (random phase scatter), jitter (phase noise),
harmonic distortion (HD2/HD3 with distinct phases), and kickback (MSB-dependent distortion).
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import calculate_coherent_freq, calculate_snr_from_amplitude, snr_to_nsd
from adctoolbox.spectrum import analyze_spectrum_polar

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N = 2**13
Fs = 800e6
Fin_target = 80e6
Fin, J = calculate_coherent_freq(Fs, Fin_target, N)
t = np.arange(N) / Fs
A, DC = 0.49, 0.5
base_noise = 50e-6

# Signal 1: Noise
noise_rms = 180e-6
signal_noise = A * np.sin(2*np.pi*Fin*t) + DC + np.random.randn(N) * noise_rms

# Signal 2: Jitter
jitter_rms = 1.3e-12
phase_jitter = np.random.randn(N) * 2 * np.pi * Fin * jitter_rms
signal_jitter = A * np.sin(2*np.pi*Fin*t + phase_jitter) + DC + np.random.randn(N) * base_noise

# Signal 3: Harmonic distortion (via static nonlinearity)
hd2_dB, hd3_dB = -80, -66
hd2_amp = 10**(hd2_dB/20)  # Harmonic amplitude / Fundamental amplitude
hd3_amp = 10**(hd3_dB/20)

# Compute nonlinearity coefficients to achieve target HD levels
# For y = x + k2*x² + k3*x³:
# HD2: k2 * A^2 / 2 = hd2_amp * A  →  k2 = 2*hd2_amp/A
# HD3: k3 * A^3 / 4 = hd3_amp * A  →  k3 = 4*hd3_amp/A²
k2 = hd2_amp / (A / 2)
k3 = hd3_amp / (A**2 / 4)

# Generate signal through nonlinearity: y = x + k2*x² + k3*x³
sig_ideal = A * np.sin(2*np.pi*Fin*t)
# Using trigonometric identities:
# sin²(ωt) = (1 - cos(2ωt))/2
# sin³(ωt) = (3sin(ωt) - sin(3ωt))/4
signal_harmonic = (sig_ideal + k2 * sig_ideal**2 + k3 * sig_ideal**3 + DC + np.random.randn(N) * base_noise)

# Signal 4: Kickback
t_ext = np.arange(N+1) / Fs
sig_clean_ext = A * np.sin(2*np.pi*Fin*t_ext) + DC + np.random.randn(N+1) * base_noise
msb_ext = np.floor(sig_clean_ext * 2**4) / 2**4
lsb_ext = np.floor((sig_clean_ext - msb_ext) * 2**12) / 2**12
msb_shifted = msb_ext[:-1]
msb = msb_ext[1:]
lsb = lsb_ext[1:]
kickback_strength = 0.009
signal_kickback = msb + lsb + kickback_strength * msb_shifted

signals = [signal_noise, signal_jitter, signal_harmonic, signal_kickback]
titles = ['Noise', 'Jitter', 'Harmonic Distortion', 'Kickback']
params = [f'RMS = {noise_rms*1e3:.2f} mV',
          f'{jitter_rms*1e12:.1f} ps',
          f'HD2 = {hd2_dB} dB, HD3 = {hd3_dB} dB',
          f'strength = {kickback_strength}']

print(f"[Sinewave] Fs=[{Fs/1e6:.0f} MHz], Fin=[{Fin/1e6:.1f} MHz] (coherent, Bin {J}), N=[{N}], A=[{A:.3f} Vpeak]")
print(f"[Scenarios] Noise, Jitter, Harmonic Distortion, Kickback\n")

# Calculate theoretical SNR for each signal
snr_noise_theory = calculate_snr_from_amplitude(sig_amplitude=A, noise_amplitude=noise_rms)
snr_jitter_theory = -20 * np.log10(2 * np.pi * Fin * jitter_rms)  # Phase jitter SNR formula
thd_power_theory = hd2_amp**2 + hd3_amp**2  # Total harmonic power
# For harmonic distortion signal, include base noise
total_noise_amp_hd = np.sqrt(thd_power_theory + (base_noise/A)**2)  # Normalized to signal amplitude
snr_thd_theory = -20 * np.log10(total_noise_amp_hd)  # SNR limited by THD + base noise

# Create 2x2 figure for comparison
fig = plt.figure(figsize=(14, 10))

# Store axes and their limits for restoration after tight_layout
axes_info = []
measured_snrs = []  # Store measured SNR for comparison

for i, (signal, title, param) in enumerate(zip(signals, titles, params)):
    # Create individual subplot with polar projection
    ax = fig.add_subplot(2, 2, i+1, projection='polar')

    # Analyze spectrum with polar phase visualization
    # Pass title to avoid axis limit reset issues in polar subplots
    coherent_result, plot_data = analyze_spectrum_polar(
        signal,
        fs=Fs,
        harmonic=5,
        win_type='boxcar',
        title=f'{title}\n{param}',
        show_plot=False,
        ax=ax
    )

    # Store axis and its ylim for later restoration
    axes_info.append((ax, ax.get_ylim()))

    metrics = coherent_result.get('metrics', {})
    snr_measured = metrics.get('snr_db', 0)
    measured_snrs.append(snr_measured)
    print(f"[{title:20s}] SNR=[{snr_measured:6.2f} dB], Bin=[{coherent_result['bin_idx']}], Noise floor=[{coherent_result['minR_dB']:7.2f} dB]")

print(f"\n[Theoretical SNR] Noise=[{snr_noise_theory:.1f} dB], Jitter=[{snr_jitter_theory:.1f} dB], Harmonic Distortion=[{snr_thd_theory:.1f} dB]")

# Adjust legend position
handles, labels = ax.get_legend_handles_labels()
if handles:
    fig.legend(handles[:3], labels[:3], loc='upper right', fontsize=9, bbox_to_anchor=(0.98, 0.98))

plt.tight_layout()

# Restore ylim after tight_layout (which resets polar axis limits)
for ax, ylim in axes_info:
    ax.set_ylim(ylim)

fig_path = output_dir / 'exp_s21_analyze_spectrum_polar.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n[Save fig] -> [{fig_path}]")

# Also save individual plots for better visibility
for i, (signal, title) in enumerate(zip(signals, titles)):
    # Save individual plot
    individual_path = output_dir / f'exp_s21_polar_{title.lower().replace(" ", "_")}.png'
    analyze_spectrum_polar(
        signal,
        fs=Fs,
        harmonic=5,
        win_type='boxcar',
        title=f'Phase Spectrum - {title}',
        save_path=individual_path,
        show_plot=False
    )