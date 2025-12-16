"""Generate signals with ADC non-idealities and analyze their spectra.

This example demonstrates the ADC_Signal_Generator class by creating signals
with various non-idealities and plotting their frequency spectra.
"""

import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, amplitudes_to_snr, snr_to_nsd, analyze_spectrum
from adctoolbox.siggen import ADC_Signal_Generator

# Setup output directory
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# Parameters
N = 2**13
Fs = 5000e6
Fin_target = 1000e6
Fin, J = find_coherent_frequency(Fs, Fin_target, N)
A, DC = 0.49, 0.5
base_noise = 50e-6

# Calculate reference SNR and NSD for base noise
snr_base = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=base_noise)
nsd_base = snr_to_nsd(snr_base, fs=Fs, osr=1)

print(f"[Sinewave] Fs=[{Fs/1e6:.2f} MHz], Fin=[{Fin/1e6:.2f} MHz], Bin/N=[{J}/{N}], A=[{A:.3f} Vpeak]")
print(f"[Nonideal] Noise RMS=[{base_noise*1e6:.2f} uVrms], Theoretical SNR=[{snr_base:.2f} dB], Theoretical NSD=[{nsd_base:.2f} dBFS/Hz]\n")

# Initialize signal generator
gen = ADC_Signal_Generator(N=N, Fs=Fs, Fin=Fin, A=A, DC=DC)

# Define all signal configurations
all_signals = [
    # Figure 1: Basic noise and jitter characterization
    {
        'title': 'Clean Sinewave (Ideal)',
        'method': lambda: gen.get_clean_signal(),
    },
    {
        'title': 'Thermal Noise: RMS=50 uV',
        'method': lambda: gen.apply_thermal_noise(noise_rms=50e-6),
    },
    {
        'title': 'Thermal Noise: RMS=100 uV',
        'method': lambda: gen.apply_thermal_noise(noise_rms=100e-6),
    },
    {
        'title': 'Quantization Noise: 8-bit',
        'method': lambda: gen.apply_quantization_noise(n_bits=8, quant_range=(DC - A/2, DC + A/2)),
    },
    {
        'title': 'Quantization Noise: 10-bit',
        'method': lambda: gen.apply_quantization_noise(n_bits=10, quant_range=(DC - A/2, DC + A/2)),
    },
    {
        'title': 'Quantization Noise: 12-bit',
        'method': lambda: gen.apply_quantization_noise(n_bits=12, quant_range=(DC - A/2, DC + A/2)),
    },
    {
        'title': 'Jitter: 100 fs',
        'method': lambda: gen.apply_jitter(jitter_rms=50e-15),
    },
    {
        'title': 'Jitter: 500 fs',
        'method': lambda: gen.apply_jitter(jitter_rms=100e-15),
    },
    {
        'title': 'Jitter: 1000 fs',
        'method': lambda: gen.apply_jitter(jitter_rms=200e-15),
    },
    {
        'title': 'Static Nonlinearity: HD2=-80, HD3=-66 dB',
        'method': lambda: gen.apply_static_nonlinearity(hd2_dB=-80, hd3_dB=-66),
    },
    {
        'title': 'Kickback: strength=0.009',
        'method': lambda: gen.apply_kickback(kickback_strength=0.009),
    },
    {
        'title': 'AM Noise: 1 MHz, 10%',
        'method': lambda: gen.apply_am_noise(am_noise_freq=1e6, am_noise_depth=0.1),
    },
    {
        'title': 'AM Tone: 500 kHz, 5%',
        'method': lambda: gen.apply_am_tone(am_tone_freq=500e3, am_tone_depth=0.05),
    },
    {
        'title': 'Gain Error: -1% (2-stage pipeline)',
        'method': lambda: gen.apply_gain_error(gain_error=0.99),
    },
    {
        'title': 'Gain Error: +1% (2-stage pipeline)',
        'method': lambda: gen.apply_gain_error(gain_error=1.01),
    },
    {
        'title': 'Clipping: level=0.2',
        'method': lambda: gen.apply_clipping(clip_level=0.02),
    },
    {
        'title': 'Clipping: level=0.2',
        'method': lambda: gen.apply_clipping(clip_level=0.05),
    },
    {
        'title': 'Drift: Random Walk',
        'method': lambda: gen.apply_drift(drift_scale=5e-5),
    },
    {
        'title': 'Glitch: Probability 0.015%',
        'method': lambda: gen.apply_glitch(glitch_prob=0.00015, glitch_amplitude=0.1),
    },
    {
        'title': 'Dynamic Nonlinearity: tau=40ps, k=0.15',
        'method': lambda: gen.apply_dynamic_nonlinearity(T_track=(1/Fs)*0.2, tau_nom=40e-12, coeff_k=0.15),
    },
    {
        'title': 'Reference Error: 2 MHz, 2%',
        'method': lambda: gen.apply_reference_error(ref_error_amplitude=0.02, ref_error_freq=2e6),
    },
]

# Generate signals and create figures automatically
print("Generating signals and computing spectra...")
print("=" * 60)

# Split signals into groups of 8 (2x4 layout)
signals_per_fig = 8
n_figs = (len(all_signals) + signals_per_fig - 1) // signals_per_fig

for fig_num in range(n_figs):
    start_idx = fig_num * signals_per_fig
    end_idx = min(start_idx + signals_per_fig, len(all_signals))
    signals_group = all_signals[start_idx:end_idx]

    # Create figure (2x4 subplots)
    n_cols = 4
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 7, n_rows * 6))
    axes = axes.flatten()

    print(f"\nFigure {fig_num + 1}: Signals {start_idx + 1} - {end_idx}")
    print("-" * 60)

    for local_idx, config in enumerate(signals_group):
        global_idx = start_idx + local_idx + 1
        signal = config['method']()

        # Set current axes for analyze_spectrum
        plt.sca(axes[local_idx])

        # Analyze spectrum and plot on current axes
        # Function automatically normalizes to signal peak (0 dB)
        result = analyze_spectrum(signal, fs=Fs, show_plot=True, show_title=False,
                                 show_label=True, ax=axes[local_idx])

        # Set title with non-ideality info
        axes[local_idx].set_title(config['title'], fontsize=10, fontweight='bold')
        axes[local_idx].set_ylim([-140, 0])

        # Print spectrum metrics
        print(f"{global_idx:2d}. {config['title']:40s} - "
              f"ENOB=[{result['enob']:6.2f} b], "
              f"SNDR=[{result['sndr_db']:6.2f} dB], "
              f"SFDR=[{result['sfdr_db']:6.2f} dB], "
              f"SNR=[{result['snr_db']:6.2f} dB], "
              f"NSD=[{result['nsd_dbfs_hz']:6.2f} dBFS/Hz]")

    # Hide unused axes if this is the last figure and it's not full
    for idx in range(len(signals_group), signals_per_fig):
        axes[idx].axis('off')

    # Finalize and save figure
    plt.suptitle(f'ADC Non-Idealities Spectrum Analysis (Part {fig_num + 1})',
                 fontsize=12, fontweight='bold', y=0.995)
    plt.tight_layout()

    fig_path = output_dir / f'exp_g02_generate_signals_spectra_{fig_num + 1}.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[Save figure] -> [{fig_path}]")

print("=" * 60)
print(f"Done! Generated {n_figs} figures with {len(all_signals)} signals total.")
