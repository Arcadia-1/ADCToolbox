"""Error histogram vs sine phase: noise, jitter, harmonic distortion, kickback"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, amplitudes_to_snr, snr_to_nsd
from adctoolbox.aout import plot_error_hist_phase

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N = 2**13
Fs = 800e6
Fin_target = 80e6
Fin, J = find_coherent_frequency(Fs, Fin_target, N)
t = np.arange(N) / Fs
A, DC = 0.49, 0.5
base_noise = 50e-6

print(f"[Error Histogram vs Phase] [Fs = {Fs/1e6:.0f} MHz, Fin = {Fin/1e6:.1f} MHz, N = {N}]")
print(f"[Signal Parameters] A={A:.3f} V, DC={DC:.3f} V\n")

# Signal 1: Noise
noise_rms = 180e-6
signal_noise = A * np.sin(2*np.pi*Fin*t) + DC + np.random.randn(N) * noise_rms

snr_noise = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=noise_rms)
nsd_noise = snr_to_nsd(snr_noise, fs=Fs, osr=1)
print(f"[Noise Signal] Noise RMS=[{noise_rms*1e6:.2f} uVrms], Theoretical SNR=[{snr_noise:.2f} dB], Theoretical NSD=[{nsd_noise:.2f} dBFS/Hz]")

# Signal 2: Jitter
jitter_rms = 1.3e-12
phase_jitter = np.random.randn(N) * 2 * np.pi * Fin * jitter_rms
signal_jitter = A * np.sin(2*np.pi*Fin*t + phase_jitter) + DC + np.random.randn(N) * base_noise

# Signal 3: Harmonic distortion
hd2_dB, hd3_dB = -80, -73
hd2, hd3 = 10**(hd2_dB/20), 10**(hd3_dB/20)
signal_harmonic = A * np.sin(2*np.pi*Fin*t) + DC + hd2 * np.sin(2*2*np.pi*Fin*t) + hd3 * np.sin(3*2*np.pi*Fin*t) + np.random.randn(N) * base_noise

snr_harmonic = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=base_noise)
nsd_harmonic = snr_to_nsd(snr_harmonic, fs=Fs, osr=1)
print(f"[Harmonic Signal] Noise RMS=[{base_noise*1e6:.2f} uVrms], HD2={hd2_dB}dB, HD3={hd3_dB}dB, Theoretical SNR=[{snr_harmonic:.2f} dB], Theoretical NSD=[{nsd_harmonic:.2f} dBFS/Hz]")
print()

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

# Generate individual plots using built-in plotting
for signal, title, param in zip(signals, titles, params):
    plot_error_hist_phase(signal, bins=100, disp=1)
    plt.gcf().suptitle(f'{title}: {param}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    fig_name = title.lower().replace(' ', '_')
    fig_path = output_dir / f'exp_a13_err_hist_sine_phase_{fig_name}.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[{title:20s}] -> [{fig_path}]")

print(f"\n[Complete] All 4 error histogram (phase) plots saved")
