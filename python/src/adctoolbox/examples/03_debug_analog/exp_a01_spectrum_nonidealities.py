"""Spectrum comparison: noise, jitter, harmonic distortion, kickback"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, analyze_spectrum, amplitudes_to_snr, snr_to_nsd
from adctoolbox.common import calculate_jitter_limit

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N = 2**13
Fs = 800e6
Fin_target = 80e6
Fin, J = find_coherent_frequency(Fs, Fin_target, N)
t = np.arange(N) / Fs
A, DC = 0.49, 0.5
base_noise = 50e-6

# Calculate actual signal power in dBFS (relative to full scale)
sig_pwr_dbfs = 20 * np.log10(A)

print(f"[Sinewave] Fs=[{Fs/1e6:.0f} MHz], Fin=[{Fin/1e6:.1f} MHz], Bin/N=[{J}/{N}], A=[{A:.3f} Vpeak], Signal Power=[{sig_pwr_dbfs:.2f} dBFS]\n")

# Signal 1: Noise
noise_rms = 180e-6
signal_noise = A * np.sin(2*np.pi*Fin*t) + DC + np.random.randn(N) * noise_rms
snr_ref_noise = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=noise_rms)
nsd_ref_noise = snr_to_nsd(snr_ref_noise, fs=Fs, signal_pwr_dbfs=sig_pwr_dbfs, osr=1)

# Signal 2: Jitter
jitter_rms = 1000e-15
phase_jitter = np.random.randn(N) * 2 * np.pi * Fin * jitter_rms
signal_jitter = A * np.sin(2*np.pi*Fin*t + phase_jitter) + DC + np.random.randn(N) * base_noise
snr_ref_jitter = calculate_jitter_limit(Fin, jitter_rms)
nsd_ref_jitter = snr_to_nsd(snr_ref_jitter, fs=Fs, signal_pwr_dbfs=sig_pwr_dbfs, osr=1)

# Signal 3: Harmonic distortion (via static nonlinearity)
hd2_dB, hd3_dB = -80, -66
hd2_amp = 10**(hd2_dB/20)  # Harmonic amplitude / Fundamental amplitude
hd3_amp = 10**(hd3_dB/20)

# Compute nonlinearity coefficients to achieve target HD levels
# HD2: k2 * A^2 / 2 = hd2_amp * A  →  k2 = hd2_amp / (A/2)
# HD3: k3 * A^3 / 4 = hd3_amp * A  →  k3 = hd3_amp / (A^2/4)
k2 = hd2_amp / (A / 2)
k3 = hd3_amp / (A**2 / 4)

# Generate distorted signal: y = x + k2*x^2 + k3*x^3
sinewave = A * np.sin(2*np.pi*Fin*t)
signal_harmonic = sinewave + k2 * sinewave**2 + k3 * sinewave**3 + DC + np.random.randn(N) * base_noise
snr_ref_harmonic = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=base_noise)
nsd_ref_harmonic = snr_to_nsd(snr_ref_harmonic, fs=Fs, signal_pwr_dbfs=sig_pwr_dbfs, osr=1)

# Signal 4: Kickback
kickback_strength = 0.009
t_ext = np.arange(N+1) / Fs  # Generate N+1 samples
sig_clean_ext = A * np.sin(2*np.pi*Fin*t_ext) + DC + np.random.randn(N+1) * base_noise
msb_ext = np.floor(sig_clean_ext * 2**4) / 2**4
lsb_ext = np.floor((sig_clean_ext - msb_ext) * 2**12) / 2**12
msb_shifted = msb_ext[:-1]  # First N samples (delayed MSB)
msb = msb_ext[1:]           # Last N samples (current MSB)
lsb = lsb_ext[1:]           # Last N samples (current LSB)
signal_kickback = msb + lsb + kickback_strength * msb_shifted
snr_ref_kickback = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=base_noise)
nsd_ref_kickback = snr_to_nsd(snr_ref_kickback, fs=Fs, signal_pwr_dbfs=sig_pwr_dbfs, osr=1)

# Organize test cases in a dictionary
test_cases = {
    'Noise': {
        'signal': signal_noise,
        'title': f'Noise: RMS={noise_rms*1e6:.0f} uV',
        'param_str': f'Noise RMS=[{noise_rms*1e6:.2f} uVrms]',
        'snr_ref': snr_ref_noise,
        'nsd_ref': nsd_ref_noise,
        'position': (0, 0),
    },
    'Jitter': {
        'signal': signal_jitter,
        'title': f'Jitter: {jitter_rms*1e15:.0f} fs',
        'param_str': f'Jitter RMS=[{jitter_rms*1e15:.2f} fs]',
        'snr_ref': snr_ref_jitter,
        'nsd_ref': nsd_ref_jitter,
        'position': (0, 1),
    },
    'Harmonic': {
        'signal': signal_harmonic,
        'title': f'Harmonic Distortion: HD2 = {hd2_dB} dB, HD3 = {hd3_dB} dB',
        'param_str': f'Base Noise=[{base_noise*1e6:.2f} uVrms]',
        'snr_ref': snr_ref_harmonic,
        'nsd_ref': nsd_ref_harmonic,
        'position': (1, 0),
    },
    'Kickback': {
        'signal': signal_kickback,
        'title': f'Kickback: strength = {kickback_strength}',
        'param_str': f'Base Noise=[{base_noise*1e6:.2f} uVrms]',
        'snr_ref': snr_ref_kickback,
        'nsd_ref': nsd_ref_kickback,
        'position': (1, 1),
    }
}

# Create figure and analyze all cases
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for name, case in test_cases.items():
    # Set current axis and analyze spectrum
    plt.sca(axes[case['position']])
    result = analyze_spectrum(case['signal'], fs=Fs)
    axes[case['position']].set_title(case['title'])

    # Print results (use name for label, pad to 8 chars for alignment)
    label = f"{name:8s}"
    print(f"[{label}] {case['param_str']}, Theoretical SNR=[{case['snr_ref']:.2f} dB], Theoretical NSD=[{case['nsd_ref']:.2f} dBFS/Hz]")
    print(f"[{label}] ENoB=[{result['enob']:5.2f} b], SNDR=[{result['sndr_db']:6.2f} dB], SFDR=[{result['sfdr_db']:6.2f} dB], SNR=[{result['snr_db']:6.2f} dB], NSD=[{result['nsd_dbfs_hz']:7.2f} dBFS/Hz]\n")

fig.suptitle(f'Spectrum Comparison: 4 Non-idealities (Fs = {Fs/1e6:.0f} MHz, Fin = {Fin/1e6:.1f} MHz)', fontsize=12, fontweight='bold')
plt.tight_layout()
fig_path = output_dir / 'exp_a01_analyze_spectrum_nonidealities.png'
print(f"[Save fig] -> [{fig_path}]\n")
plt.savefig(fig_path, dpi=150)
plt.close()
