"""
Basic demo: Spectrum analysis and figure saving with amplitude comparison.

This script demonstrates using the analyze_spectrum function for performing standard FFT analysis
and saving the figure directly to the output directory. Shows comparison between two signal amplitudes
on the same dBFS scale (fixed FSR).
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import analyze_spectrum, find_coherent_frequency, amplitudes_to_snr, snr_to_nsd

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N_fft = 2**13
Fs = 100e6
Fin_target = 12e6
Fin, Fin_bin = find_coherent_frequency(fs=Fs, fin_target=Fin_target, n_fft=N_fft)
noise_rms = 200e-6

# Fixed full-scale range for both signals (to compare on same dBFS scale)
adc_range = [-0.5, 0.5]  # ADC FSR = 1.0V
fsr_magnitude = adc_range[1] - adc_range[0]

# Create figure with 1 row, 2 columns
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

amplitudes = [0.5, 0.25]

for idx, A in enumerate(amplitudes):
    snr_ref = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=noise_rms)
    # Calculate signal power in dBFS (peak amplitude relative to FSR)
    sig_pwr_dbfs_theory = 20 * np.log10(A / (fsr_magnitude / 2))
    nsd_ref = snr_to_nsd(snr_ref, fs=Fs, osr=1, psignal_dbfs=sig_pwr_dbfs_theory)
    print(f"[Sinewave] Fs=[{Fs/1e6:.2f} MHz], Fin=[{Fin/1e6:.2f} MHz], Bin/N=[{Fin_bin}/{N_fft}], A=[{A:.3f} Vpeak]")
    print(f"[Nonideal] Noise RMS=[{noise_rms*1e6:.2f} uVrms], Theoretical SNR=[{snr_ref:.2f} dB], Theoretical NSD=[{nsd_ref:.2f} dBFS/Hz]\n")

    t = np.arange(N_fft) / Fs
    signal = A * np.sin(2*np.pi*Fin*t) + np.random.randn(N_fft) * noise_rms

    plt.sca(axes[idx])
    result = analyze_spectrum(signal, fs=Fs, max_scale_range=adc_range)

    print(f"[analyze_spectrum] ENoB=[{result['enob']:5.2f} b], SNDR=[{result['sndr_dbc']:6.2f} dB], SFDR=[{result['sfdr_dbc']:6.2f} dB], SNR=[{result['snr_dbc']:6.2f} dB], NSD=[{result['nsd_dbfs_hz']:7.2f} dBFS/Hz]")
    expected_power = 20 * np.log10(2*A / fsr_magnitude)
    print(f"[analyze_spectrum] Noise Floor=[{result['noise_floor_dbfs']:7.2f} dBFS], Signal Power=[{result['sig_pwr_dbfs']:6.2f} dBFS] (expected: {expected_power:.2f} dB for {2*A:.3f}V in {fsr_magnitude:.1f}V FSR), \n")

    # Set title and limits
    axes[idx].set_title(f'A = {A:.3f} Vpeak (SFDR = {result["sfdr_dbc"]:.2f} dB)')

fig_path = (output_dir / 'exp_s03_analyze_spectrum_savefig.png').resolve()
print(f"\n[Save fig] -> [{fig_path}]\n")
plt.tight_layout()
plt.savefig(fig_path, dpi=150)
plt.close()
