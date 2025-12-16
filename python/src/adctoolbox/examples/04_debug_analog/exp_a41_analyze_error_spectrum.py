"""Error spectrum: noise, jitter, harmonic distortion, kickback
AND individual phase/amplitude noise comparison"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, amplitudes_to_snr, snr_to_nsd
from adctoolbox.aout.fit_sine_4param import fit_sine_4param as fit_sine

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N = 2**13
Fs = 800e6
Fin_target = 80e6
Fin, J = find_coherent_frequency(Fs, Fin_target, N)
t = np.arange(N) / Fs
A, DC = 0.49, 0.5
base_noise = 50e-6

# Signal 1: Noise (flat spectrum)
noise_rms = 180e-6
signal_noise = A * np.sin(2*np.pi*Fin*t) + DC + np.random.randn(N) * noise_rms

# Signal 2: Jitter (noise floor rises at high frequencies)
jitter_rms = 1.3e-12
phase_jitter = np.random.randn(N) * 2 * np.pi * Fin * jitter_rms
signal_jitter = A * np.sin(2*np.pi*Fin*t + phase_jitter) + DC + np.random.randn(N) * base_noise

# Signal 3: Harmonic distortion (clear harmonics at 2F, 3F)
hd2_dB, hd3_dB = -80, -73
hd2, hd3 = 10**(hd2_dB/20), 10**(hd3_dB/20)
signal_harmonic = A * np.sin(2*np.pi*Fin*t) + DC + hd2 * np.sin(2*2*np.pi*Fin*t) + hd3 * np.sin(3*2*np.pi*Fin*t) + np.random.randn(N) * base_noise

# Signal 4: Kickback (code-dependent spurs)
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
titles = ['Noise (Flat Spectrum)', 'Jitter (HF Rise)', 'Harmonic Distortion (Harmonics)', 'Kickback (Spurs)']
params = [f'RMS = {noise_rms*1e3:.2f} mV',
          f'{jitter_rms*1e12:.1f} ps',
          f'HD2 = {hd2_dB} dB, HD3 = {hd3_dB} dB',
          f'strength = {kickback_strength}']

print(f"[Error Spectrum] [Fs = {Fs/1e6:.0f} MHz, Fin = {Fin/1e6:.1f} MHz, N = {N}]")
print(f"[Signal Parameters] A={A:.3f} V, DC={DC:.3f} V")

snr_noise = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=noise_rms)
nsd_noise = snr_to_nsd(snr_noise, fs=Fs, osr=1)
print(f"[Noise Signal] Noise RMS=[{noise_rms*1e6:.2f} uVrms], Theoretical SNR=[{snr_noise:.2f} dB], Theoretical NSD=[{nsd_noise:.2f} dBFS/Hz]")

snr_harmonic = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=base_noise)
nsd_harmonic = snr_to_nsd(snr_harmonic, fs=Fs, osr=1)
print(f"[Harmonic Signal] Noise RMS=[{base_noise*1e6:.2f} uVrms], HD2={hd2_dB}dB, HD3={hd3_dB}dB, Theoretical SNR=[{snr_harmonic:.2f} dB], Theoretical NSD=[{nsd_harmonic:.2f} dBFS/Hz]\n")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, (signal, title, param) in enumerate(zip(signals, titles, params)):
    # Fit sine and get error
    fit_result = fit_sine(signal, Fin/Fs)
    sig_fit = fit_result['fitted_signal']
    err = sig_fit - signal

    # Compute FFT of error
    window = np.hanning(N)
    spec = np.fft.fft(err * window)
    spec_mag = np.abs(spec[:N//2])
    spec_dB = 20 * np.log10(spec_mag / (np.max(spec_mag) + 1e-10))
    freq = np.arange(N//2) * Fs / N

    # Plot error spectrum
    axes[i].plot(freq/1e6, spec_dB, 'b-', linewidth=0.8)
    axes[i].set_xlabel('Frequency (MHz)', fontsize=11)
    axes[i].set_ylabel('Error Spectrum (dB)', fontsize=11)
    axes[i].set_title(f'{title}\n{param}', fontsize=11, fontweight='bold')
    axes[i].grid(True, alpha=0.3)
    axes[i].set_ylim([-80, 0])
    axes[i].set_xlim([0, Fs/2/1e6])

    # Mark harmonic locations for harmonic distortion
    if i == 2:
        for h in [2, 3]:
            axes[i].axvline((Fin*h)/1e6, color='r', linestyle='--', linewidth=1, alpha=0.5)

    err_rms = np.sqrt(np.mean(err**2))
    print(f"  {title:35s} - Error RMS: {err_rms*1e6:.2f} uV")

plt.tight_layout()
fig_path = output_dir / 'exp_a41_analyze_error_spectrum.png'
plt.savefig(fig_path, dpi=150)
plt.close()

print(f"\n[Save fig] -> [{fig_path}]")

# ============================================================================
# 3-Plot Comparison: Thermal Noise vs Phase Noise vs Amplitude Noise
# ============================================================================

print("\n" + "="*80)
print("3-PLOT COMPARISON: Thermal Noise vs Phase Noise vs Amplitude Noise")
print("="*80 + "\n")

# Set seed for reproducibility
np.random.seed(42)

# Define noise levels
target_thermal = 150e-6
target_pm_rad = 200e-6
target_am = 100e-6

# Generate phase signals
phase_clean = 2 * np.pi * Fin * t

# Case 1: Thermal Noise Only (additive white noise)
n_thermal = np.random.randn(N) * target_thermal
sig_thermal_only = A * np.sin(phase_clean) + DC + n_thermal

# Case 2: Phase Noise Only (phase jitter)
n_pm = np.random.randn(N) * target_pm_rad
sig_pm_only = A * np.sin(phase_clean + n_pm) + DC

# Case 3: Amplitude Noise Only (amplitude modulation)
n_am = np.random.randn(N) * target_am
sig_am_only = (A + n_am) * np.sin(phase_clean) + DC

signals_3 = [sig_thermal_only, sig_pm_only, sig_am_only]
titles_3 = [f'Thermal Noise Only\n({target_thermal*1e6:.0f} µV)',
            f'Phase Noise Only\n({target_pm_rad*1e6:.0f} µV)',
            f'Amplitude Noise Only\n({target_am*1e6:.0f} µV)']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, (signal, title) in enumerate(zip(signals_3, titles_3)):
    # Fit sine and get error
    fit_result = fit_sine(signal, Fin/Fs)
    sig_fit = fit_result['fitted_signal']
    err = sig_fit - signal

    # Compute FFT of error
    window = np.hanning(N)
    spec = np.fft.fft(err * window)
    spec_mag = np.abs(spec[:N//2])
    spec_dB = 20 * np.log10(spec_mag / (np.max(spec_mag) + 1e-10))
    freq = np.arange(N//2) * Fs / N

    # Plot error spectrum
    axes[i].plot(freq/1e6, spec_dB, 'b-', linewidth=1)
    axes[i].set_xlabel('Frequency (MHz)', fontsize=11)
    axes[i].set_ylabel('Error Spectrum (dB)', fontsize=11)
    axes[i].set_title(title, fontsize=12, fontweight='bold')
    axes[i].grid(True, alpha=0.3)
    axes[i].set_ylim([-80, 0])
    axes[i].set_xlim([0, Fs/2/1e6])

    err_rms = np.sqrt(np.mean(err**2))
    print(f"  {title.split(chr(10))[0]:30s} - Error RMS: {err_rms*1e6:.2f} uV")

plt.suptitle('Error Spectrum Comparison: Phase Noise vs Amplitude Noise vs Thermal Noise',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig_path = output_dir / 'exp_a41_analyze_error_spectrum_3plot.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n[Save fig] -> [{fig_path}]\n")
