"""
Polar phase spectrum analysis: MSB-dependent memory effect distortion.
Demonstrates memory effect effect where MSB transition from previous sample affects current sample,
creating a characteristic distortion pattern visible in the polar phase plot.
Shows 4 input frequencies × 2 memory effect strengths to visualize how spur phases vary with Fin/Fs.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, amplitudes_to_snr, snr_to_nsd, analyze_spectrum_polar

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N = 2**13
Fs = 800e6
A, DC = 0.49, 0.5
base_noise = 100e-6

# 4 different input frequencies
Fin_targets = [40e6, 80e6, 160e6, 280e6]
# 2 memory effect strengths
memory_effect_strengths = [0.01, 0.02]

snr_ref = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=base_noise)
nsd_ref = snr_to_nsd(snr_ref, fs=Fs, osr=1)
print(f"[Signal] Fs=[{Fs/1e6:.0f} MHz], N=[{N}], A=[{A:.3f} Vpeak]")
print(f"[Base Noise] RMS=[{base_noise*1e6:.2f} uVrms], Theoretical SNR=[{snr_ref:.2f} dB], Theoretical NSD=[{nsd_ref:.2f} dBFS/Hz]\n")

# Create 2x4 figure (2 rows for memory effect strengths, 4 columns for frequencies)
fig = plt.figure(figsize=(16, 8))

# Store axes and their limits for restoration after tight_layout
axes_info = []

plot_idx = 0
for row_idx, me_strength in enumerate(memory_effect_strengths):
    for col_idx, Fin_target in enumerate(Fin_targets):
        Fin, J = find_coherent_frequency(Fs, Fin_target, N)

        # Generate clean signal for this frequency
        t_ext = np.arange(N+1) / Fs
        sig_clean_ext = A * np.sin(2*np.pi*Fin*t_ext) + DC + np.random.randn(N+1) * base_noise
        msb_ext = np.floor(sig_clean_ext * 2**4) / 2**4
        lsb_ext = np.floor((sig_clean_ext - msb_ext) * 2**18) / 2**18
        msb_shifted = msb_ext[:-1]
        msb = msb_ext[1:]
        lsb = lsb_ext[1:]

        # Expected phase delay per sample
        phase_delay_deg = 360 * Fin / Fs

        # Generate memory effect signal
        signal_me = msb + lsb + me_strength * msb_shifted

        # Create subplot with polar projection
        ax = fig.add_subplot(2, 4, plot_idx + 1, projection='polar')
        plot_idx += 1

        # Analyze spectrum with polar phase visualization
        result = analyze_spectrum_polar(
            signal_me,
            fs=Fs,
            harmonic=5,
            win_type='boxcar',
            ax=ax,
            fixed_radial_range=120
        )

        # Calculate theoretical harmonic phases
        # HD2: follows simple 2φ relationship
        # HD3: follows 180° - 3φ (phase inverted due to memory effect mechanism)
        hd2_phase_theory = (2 * phase_delay_deg) % 360
        hd3_phase_theory = (180 - 3 * phase_delay_deg) % 360

        # Set title with theoretical phase information
        title = f'Fin={Fin/1e6:.0f}MHz (φ={phase_delay_deg:.1f}°), ME={me_strength}\nHD2∠{hd2_phase_theory:.1f}°, HD3∠{hd3_phase_theory:.1f}°'
        ax.set_title(title, pad=20, fontsize=10, fontweight='bold')

        # Store axis and its ylim for later restoration
        axes_info.append((ax, ax.get_ylim()))

        print(f"[Fin={Fin/1e6:5.0f}MHz, ME={me_strength}] sndr={result['sndr_db']:5.2f}dB, snr={result['snr_db']:5.2f}dB, thd={result['thd_db']:6.2f}dB")

plt.tight_layout()

# Restore ylim after tight_layout (which resets polar axis limits)
for ax, ylim in axes_info:
    ax.set_ylim(ylim)

fig_path = output_dir / 'exp_s11_polar_memory_effect.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n[Save fig] -> [{fig_path}]")
