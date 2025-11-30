"""
Complete ADC Analysis Workflow

This example demonstrates a comprehensive end-to-end ADC characterization
workflow using multiple tools from the ADCToolbox package.

Workflow Steps:
1. Load/generate ADC digital output data
2. Perform foreground calibration
3. Run comprehensive performance analysis
4. Generate detailed analysis report

Tools used:
- fg_cal_sine: Calibrate ADC weights
- spec_plot: Spectrum analysis
- tom_decomp: Error decomposition
- err_hist_sine: Error histogram
- err_pdf: Error distribution
- err_auto_correlation: Error correlation
- sine_fit: Parameter extraction
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import all required tools
from adctoolbox.dout import fg_cal_sine, overflow_chk
from adctoolbox.aout import (spec_plot, spec_plot_phase, tom_decomp,
                             err_hist_sine, err_pdf, err_auto_correlation)
from adctoolbox.common import find_bin, sine_fit

# For loading packaged example data (optional)
from adctoolbox.examples.data import get_example_data_path

# Create output directory with timestamp
import os
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"../output/complete_analysis_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("Complete ADC Analysis Workflow")
print("=" * 80)
print(f"\nOutput directory: {output_dir}")

# NOTE: This workflow uses synthetic data to demonstrate the complete analysis.
# To analyze real digital output data included with the package, use:
#   data_file = get_example_data_path('dout_SAR_12b_weight_1.csv')
#   bits = np.loadtxt(data_file, delimiter=',', dtype=int)
#   # Then skip data generation and proceed to calibration
# See examples/dout/example_fg_cal_sine.py for a real-data example.

#%% Configuration
print("\n" + "=" * 80)
print("CONFIGURATION")
print("=" * 80)

config = {
    'N': 2**13,           # Number of samples
    'resolution': 12,     # ADC resolution (bits)
    'Fin_norm': 0.0789,   # Normalized frequency
    'cal_order': 5,       # Calibration polynomial order
    'num_harmonics': 10,  # Number of harmonics to analyze
    'weight_error': 0.02  # Capacitor mismatch (±2%)
}

for key, value in config.items():
    print(f"  {key:20s}: {value}")

#%% Step 1: Generate/Load ADC Data
print("\n" + "=" * 80)
print("STEP 1: Generate Test ADC Data")
print("=" * 80)

N = config['N']
resolution = config['resolution']
Fin_norm = config['Fin_norm']

# Generate coherent sinewave
J = find_bin(1, Fin_norm, N)
Fin = J / N
t = np.arange(N)

# Ideal signal with multiple imperfections
A = 0.499
DC = 0.5

# 1. Base sinewave
signal = A * np.sin(2 * np.pi * Fin * t) + DC

# 2. Add harmonic distortion
HD2_dB, HD3_dB, HD5_dB = -60, -70, -80
signal += 10**(HD2_dB/20) * A * np.sin(2 * 2 * np.pi * Fin * t)
signal += 10**(HD3_dB/20) * A * np.sin(3 * 2 * np.pi * Fin * t)
signal += 10**(HD5_dB/20) * A * np.sin(5 * 2 * np.pi * Fin * t)

# 3. Add noise
signal += np.random.randn(N) * 1e-4

print(f"\nGenerated test signal:")
print(f"  Samples: {N}")
print(f"  Frequency bin: {J}")
print(f"  Normalized frequency: {Fin:.6f}")
print(f"  HD2/HD3/HD5: {HD2_dB}/{HD3_dB}/{HD5_dB} dB")

# Quantize to digital codes
digital_codes = np.floor(signal * (2**resolution))
digital_codes = np.clip(digital_codes, 0, 2**resolution - 1).astype(int)

# Convert to binary matrix
bits = np.zeros((N, resolution), dtype=int)
for i in range(resolution):
    bits[:, i] = (digital_codes >> (resolution - 1 - i)) & 1

# Simulate capacitor mismatch
np.random.seed(42)
ideal_weights = 2.0 ** np.arange(resolution - 1, -1, -1)
actual_weights = ideal_weights * (1 + np.random.randn(resolution) * config['weight_error'])

uncalibrated_output = np.dot(bits, actual_weights)

print(f"  Weight errors: ±{config['weight_error']*100:.1f}%")
print(f"  Digital code range: [{digital_codes.min()}, {digital_codes.max()}]")

#%% Step 2: Analyze Uncalibrated Performance
print("\n" + "=" * 80)
print("STEP 2: Analyze Uncalibrated Performance")
print("=" * 80)

fig_uncal = plt.figure(figsize=(12, 8))
enob_uncal, sndr_uncal, sfdr_uncal, snr_uncal, thd_uncal, _, _, _ = spec_plot(
    uncalibrated_output,
    label=True,
    harmonic=config['num_harmonics'],
    osr=1,
    win_type='hamming'
)
plt.title('Uncalibrated ADC Spectrum', fontsize=14, fontweight='bold')
plt.savefig(os.path.join(output_dir, '1_uncalibrated_spectrum.png'), dpi=200)
plt.close()

print(f"\nUncalibrated metrics:")
print(f"  ENoB:  {enob_uncal:.2f} bits")
print(f"  SNDR:  {sndr_uncal:.2f} dB")
print(f"  SFDR:  {sfdr_uncal:.2f} dB")
print(f"  SNR:   {snr_uncal:.2f} dB")
print(f"  THD:   {thd_uncal:.2f} dB")

#%% Step 3: Calibration
print("\n" + "=" * 80)
print("STEP 3: Foreground Calibration")
print("=" * 80)

weight, offset, calibrated_output, ideal, err, freq_cal = fg_cal_sine(
    bits,
    freq=0,  # Auto-detect
    order=config['cal_order']
)

print(f"\nCalibration results:")
print(f"  Detected frequency: {freq_cal:.6f}")
print(f"  DC offset: {offset:.6f}")
print(f"  Error RMS: {np.std(err):.6f}")

# Visualize weight corrections
fig_weights, axes = plt.subplots(1, 2, figsize=(14, 5))

bit_labels = [f'B{resolution-1-i}' for i in range(resolution)]
weight_errors = (actual_weights / ideal_weights - 1) * 100
weight_corrections = (weight / actual_weights - 1) * 100

axes[0].bar(bit_labels, weight_errors, alpha=0.7, color='red')
axes[0].set_ylabel('Weight Error (%)')
axes[0].set_title('Original Weight Errors (Mismatch)')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=0, color='k', linestyle='--', linewidth=0.5)

axes[1].bar(bit_labels, weight_corrections, alpha=0.7, color='green')
axes[1].set_ylabel('Correction (%)')
axes[1].set_title('Calibration Corrections')
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '2_weight_corrections.png'), dpi=200)
plt.close()

#%% Step 4: Analyze Calibrated Performance
print("\n" + "=" * 80)
print("STEP 4: Analyze Calibrated Performance")
print("=" * 80)

fig_cal = plt.figure(figsize=(12, 8))
enob_cal, sndr_cal, sfdr_cal, snr_cal, thd_cal, _, _, _ = spec_plot(
    calibrated_output,
    label=True,
    harmonic=config['num_harmonics'],
    osr=1,
    win_type='hamming'
)
plt.title('Calibrated ADC Spectrum', fontsize=14, fontweight='bold')
plt.savefig(os.path.join(output_dir, '3_calibrated_spectrum.png'), dpi=200)
plt.close()

print(f"\nCalibrated metrics:")
print(f"  ENoB:  {enob_cal:.2f} bits  (+{enob_cal - enob_uncal:.2f})")
print(f"  SNDR:  {sndr_cal:.2f} dB   (+{sndr_cal - sndr_uncal:.2f} dB)")
print(f"  SFDR:  {sfdr_cal:.2f} dB   (+{sfdr_cal - sfdr_uncal:.2f} dB)")
print(f"  SNR:   {snr_cal:.2f} dB   (+{snr_cal - snr_uncal:.2f} dB)")
print(f"  THD:   {thd_cal:.2f} dB   (+{thd_cal - thd_uncal:.2f} dB)")

#%% Step 5: Detailed Error Analysis
print("\n" + "=" * 80)
print("STEP 5: Detailed Error Analysis")
print("=" * 80)

# 5a. Thompson Decomposition
print("\n  5a. Thompson decomposition (dependent/independent error)...")
fig_tom = plt.figure(figsize=(12, 10))
signal_tom, error_tom, indep_tom, dep_tom, phi_tom = tom_decomp(
    calibrated_output,
    freq_cal,
    num_harmonics=config['num_harmonics'],
    display=True
)
plt.suptitle('Error Decomposition (Thompson Method)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '4_error_decomposition.png'), dpi=200)
plt.close()

print(f"      Phase: {phi_tom:.4f} rad")
print(f"      Independent error RMS: {np.std(indep_tom):.6f}")
print(f"      Dependent error RMS: {np.std(dep_tom):.6f}")

# 5b. Phase-domain spectrum
print("\n  5b. Phase-domain spectrum analysis...")
fig_phase = plt.figure(figsize=(12, 8))
h_phase, spec_phase, phi_phase, bin_phase = spec_plot_phase(
    calibrated_output,
    harmonic=config['num_harmonics']
)
plt.title('Phase-Domain Spectrum (Polar)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '5_phase_spectrum.png'), dpi=200)
plt.close()

print(f"      Fundamental bin: {bin_phase}")

# 5c. Error histogram by phase
print("\n  5c. Error histogram by phase...")
fig_hist_phase = plt.figure(figsize=(12, 8))
emean_p, erms_p, phase_axis, anoi, pnoi, _, _ = err_hist_sine(
    calibrated_output,
    bins=99,
    fin=freq_cal,
    display=True,
    mode=0  # Phase mode
)
plt.suptitle('Error Histogram by Phase', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '6_error_hist_phase.png'), dpi=200)
plt.close()

print(f"      Amplitude noise: {anoi:.6f}")
print(f"      Phase noise: {pnoi:.6f} rad")

# 5d. Error histogram by code
print("\n  5d. Error histogram by code...")
fig_hist_code = plt.figure(figsize=(12, 8))
emean_c, erms_c, code_axis, _, _, _, _ = err_hist_sine(
    calibrated_output,
    bins=20,
    fin=freq_cal,
    display=True,
    mode=1  # Code mode
)
plt.suptitle('Error Histogram by Code', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '7_error_hist_code.png'), dpi=200)
plt.close()

# 5e. Error PDF
print("\n  5e. Error probability density function...")
err_data = calibrated_output - ideal
fig_pdf = plt.figure(figsize=(12, 8))
counts, mu, sigma, kl_div, x, fx, gauss_pdf = err_pdf(
    err_data,
    resolution=resolution,
    full_scale=1.0
)
plt.title('Error Probability Density Function', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '8_error_pdf.png'), dpi=200)
plt.close()

print(f"      Mean: {mu:.6f}")
print(f"      Std dev: {sigma:.6f}")
print(f"      KL divergence: {kl_div:.6f}")

# 5f. Error autocorrelation
print("\n  5f. Error autocorrelation...")
fig_acf = plt.figure(figsize=(12, 8))
acf, lags = err_auto_correlation(
    err_data,
    max_lag=200,
    normalize=True
)
plt.title('Error Autocorrelation', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '9_error_autocorrelation.png'), dpi=200)
plt.close()

#%% Step 6: Create Summary Report
print("\n" + "=" * 80)
print("STEP 6: Generate Summary Report")
print("=" * 80)

# Create comprehensive summary figure
fig_summary = plt.figure(figsize=(18, 12))

# Title
fig_summary.suptitle(f'ADC Analysis Summary Report\n{timestamp}',
                     fontsize=16, fontweight='bold')

# Create grid
gs = fig_summary.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Comparison spectrum
ax1 = fig_summary.add_subplot(gs[0, :])
freq_axis = np.fft.fftfreq(N)
spec_uncal = np.fft.fft(uncalibrated_output * np.hamming(N))
spec_cal = np.fft.fft(calibrated_output * np.hamming(N))
ax1.plot(freq_axis[:N//2], 20*np.log10(np.abs(spec_uncal[:N//2])),
         'b-', alpha=0.5, label='Uncalibrated')
ax1.plot(freq_axis[:N//2], 20*np.log10(np.abs(spec_cal[:N//2])),
         'r-', alpha=0.7, label='Calibrated')
ax1.set_xlabel('Normalized Frequency')
ax1.set_ylabel('Magnitude (dB)')
ax1.set_title('Spectrum Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2-4. Metrics table
ax2 = fig_summary.add_subplot(gs[1, 0])
ax2.axis('off')
metrics_text = f"""
PERFORMANCE METRICS

Uncalibrated:
  ENoB:  {enob_uncal:.2f} bits
  SNDR:  {sndr_uncal:.2f} dB
  SFDR:  {sfdr_uncal:.2f} dB
  SNR:   {snr_uncal:.2f} dB
  THD:   {thd_uncal:.2f} dB

Calibrated:
  ENoB:  {enob_cal:.2f} bits
  SNDR:  {sndr_cal:.2f} dB
  SFDR:  {sfdr_cal:.2f} dB
  SNR:   {snr_cal:.2f} dB
  THD:   {thd_cal:.2f} dB

Improvement:
  Δ ENoB: +{enob_cal - enob_uncal:.2f} bits
  Δ SNDR: +{sndr_cal - sndr_uncal:.2f} dB
"""
ax2.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
         verticalalignment='center')

# 5. Error decomposition
ax3 = fig_summary.add_subplot(gs[1, 1])
ax3.plot(indep_tom[:500], 'b-', alpha=0.7, label='Independent')
ax3.plot(dep_tom[:500], 'r-', alpha=0.7, label='Dependent')
ax3.set_xlabel('Sample Index')
ax3.set_ylabel('Error')
ax3.set_title('Error Decomposition')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 6. Error histogram
ax4 = fig_summary.add_subplot(gs[1, 2])
ax4.plot(phase_axis, emean_p, 'b-', linewidth=2)
ax4.fill_between(phase_axis, emean_p - erms_p, emean_p + erms_p,
                 alpha=0.3, color='blue')
ax4.set_xlabel('Phase (rad)')
ax4.set_ylabel('Error Mean ± RMS')
ax4.set_title('Error vs Phase')
ax4.grid(True, alpha=0.3)

# 7-9. Configuration and notes
ax5 = fig_summary.add_subplot(gs[2, :])
ax5.axis('off')
config_text = f"""
CONFIGURATION:  Samples: {N} | Resolution: {resolution} bits | Frequency: {Fin:.6f} | Cal. Order: {config['cal_order']}

ERROR ANALYSIS:  Indep. Error RMS: {np.std(indep_tom):.6f} | Dep. Error RMS: {np.std(dep_tom):.6f} | Amp. Noise: {anoi:.6f} | Phase Noise: {pnoi:.6f} rad

STATISTICS:  Error Mean: {mu:.6f} | Error Std: {sigma:.6f} | KL Divergence: {kl_div:.6f}
"""
ax5.text(0.5, 0.5, config_text, fontsize=10, family='monospace',
         horizontalalignment='center', verticalalignment='center')

plt.savefig(os.path.join(output_dir, '0_SUMMARY_REPORT.png'), dpi=200, bbox_inches='tight')
plt.close()

print(f"\nSummary report generated successfully!")

#%% Final Summary
print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print(f"\nOutput directory: {output_dir}")
print("\nGenerated files:")
print("  0_SUMMARY_REPORT.png          - Complete analysis summary")
print("  1_uncalibrated_spectrum.png   - Before calibration")
print("  2_weight_corrections.png      - Weight error analysis")
print("  3_calibrated_spectrum.png     - After calibration")
print("  4_error_decomposition.png     - Thompson decomposition")
print("  5_phase_spectrum.png          - Phase-domain analysis")
print("  6_error_hist_phase.png        - Error vs phase")
print("  7_error_hist_code.png         - Error vs code")
print("  8_error_pdf.png               - Error distribution")
print("  9_error_autocorrelation.png   - Error correlation")

print("\n" + "=" * 80)
print("Analysis workflow completed successfully!")
print("=" * 80)

def main():
    """Entry point for CLI command."""
    pass  # Script already executed at module level

if __name__ == "__main__":
    main()
