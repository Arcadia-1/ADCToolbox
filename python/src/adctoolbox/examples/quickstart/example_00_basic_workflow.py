"""
Quickstart: Basic ADC Analysis Workflow

This example provides a quick introduction to the ADCToolbox package,
demonstrating a complete ADC analysis workflow from data loading to
performance characterization.

Workflow:
1. Load or generate ADC data
2. Calibrate digital codes (if needed)
3. Analyze performance metrics
4. Visualize results
"""

import numpy as np
import matplotlib.pyplot as plt

# Import ADCToolbox modules
from adctoolbox.aout import spec_plot, tom_decomp, err_hist_sine
from adctoolbox.dout import fg_cal_sine
from adctoolbox.common import find_bin, sine_fit

# For loading packaged example data (optional)
from adctoolbox.examples.data import get_example_data_path

# Create output directory
import os
output_dir = "../output"
os.makedirs(output_dir, exist_ok=True)

print("=" * 70)
print("ADCToolbox Quickstart: Basic ADC Analysis Workflow")
print("=" * 70)

# NOTE: This quickstart uses synthetic data to demonstrate the complete workflow.
# To analyze real example data included with the package, use:
#   data_file = get_example_data_path('sinewave_jitter_400fs.csv')
#   signal = np.loadtxt(data_file, delimiter=',')
#   # Then skip to Step 3 for analysis
# See examples/aout/ and examples/common/ for simpler real-data examples.

#%% Step 1: Generate or Load ADC Data
print("\nStep 1: Generate Test ADC Data")
print("-" * 70)

# For this example, we'll generate synthetic ADC data
# In practice, you would load real ADC data from a file

N = 2**12  # 4096 samples
resolution = 10  # 10-bit ADC
Fin_norm = 0.1  # Normalized frequency (Fin/Fs)

# Generate ideal sinewave
J = find_bin(1, Fin_norm, N)
Fin = J / N
t = np.arange(N)

ideal_signal = 0.49 * np.sin(2 * np.pi * Fin * t) + 0.5

# Add some imperfections
# 1. Harmonic distortion
HD2 = 10**(-60/20) * 0.49  # -60dB HD2
HD3 = 10**(-70/20) * 0.49  # -70dB HD3
signal = (ideal_signal +
          HD2 * np.sin(2 * 2 * np.pi * Fin * t) +
          HD3 * np.sin(3 * 2 * np.pi * Fin * t))

# 2. Add noise
signal += np.random.randn(N) * 1e-4

print(f"  Generated {N} samples")
print(f"  ADC resolution: {resolution} bits")
print(f"  Input frequency: {Fin:.6f} (normalized)")
print(f"  Signal range: [{signal.min():.4f}, {signal.max():.4f}]")

#%% Step 2: Convert to Digital Codes (simulate ADC quantization)
print("\nStep 2: Simulate ADC Quantization")
print("-" * 70)

# Quantize to digital codes
digital_codes = np.floor(signal * (2**resolution))
digital_codes = np.clip(digital_codes, 0, 2**resolution - 1).astype(int)

# Convert to binary matrix (MSB to LSB)
bits = np.zeros((N, resolution), dtype=int)
for i in range(resolution):
    bits[:, i] = (digital_codes >> (resolution - 1 - i)) & 1

# Add weight errors (simulate capacitor mismatch)
np.random.seed(42)
ideal_weights = 2.0 ** np.arange(resolution - 1, -1, -1)
actual_weights = ideal_weights * (1 + np.random.randn(resolution) * 0.02)

# Reconstruct analog signal with weight errors
uncalibrated_output = np.dot(bits, actual_weights)

print(f"  Digital codes: {digital_codes.min()} to {digital_codes.max()}")
print(f"  Binary matrix shape: {bits.shape}")
print(f"  Added ±2% random weight errors")

#%% Step 3: Quick Performance Check (Uncalibrated)
print("\nStep 3: Analyze Uncalibrated Performance")
print("-" * 70)

# Quick spectrum analysis
enob_uncal, sndr_uncal, sfdr_uncal, _, _, _, _, _ = spec_plot(
    uncalibrated_output,
    label=True,
    harmonic=5
)
plt.title('Uncalibrated ADC Spectrum')
plt.savefig(os.path.join(output_dir, 'quickstart_uncalibrated.png'), dpi=150)
plt.close()

print(f"\nUncalibrated Performance:")
print(f"  ENoB:  {enob_uncal:.2f} bits")
print(f"  SNDR:  {sndr_uncal:.2f} dB")
print(f"  SFDR:  {sfdr_uncal:.2f} dB")

#%% Step 4: Calibrate ADC
print("\nStep 4: Calibrate ADC using Foreground Calibration")
print("-" * 70)

# Run foreground calibration
weight, offset, calibrated_output, ideal, err, freq_detected = fg_cal_sine(
    bits,
    freq=0,  # Auto-detect frequency
    order=5
)

print(f"  Detected frequency: {freq_detected:.6f}")
print(f"  DC offset: {offset:.6f}")
print(f"  Calibration error RMS: {np.std(err):.6f}")

#%% Step 5: Analyze Calibrated Performance
print("\nStep 5: Analyze Calibrated Performance")
print("-" * 70)

# Spectrum analysis after calibration
enob_cal, sndr_cal, sfdr_cal, _, _, _, _, _ = spec_plot(
    calibrated_output,
    label=True,
    harmonic=5
)
plt.title('Calibrated ADC Spectrum')
plt.savefig(os.path.join(output_dir, 'quickstart_calibrated.png'), dpi=150)
plt.close()

print(f"\nCalibrated Performance:")
print(f"  ENoB:  {enob_cal:.2f} bits")
print(f"  SNDR:  {sndr_cal:.2f} dB")
print(f"  SFDR:  {sfdr_cal:.2f} dB")

print(f"\nImprovement:")
print(f"  ENoB:  +{enob_cal - enob_uncal:.2f} bits")
print(f"  SNDR:  +{sndr_cal - sndr_uncal:.2f} dB")

#%% Step 6: Additional Analysis Tools
print("\nStep 6: Run Additional Analysis Tools")
print("-" * 70)

# 6a. Error decomposition
print("\n  6a. Time-domain error decomposition (Thompson method)...")
fig_tom = plt.figure(figsize=(12, 10))
signal_tom, error, indep, dep, phi = tom_decomp(
    calibrated_output,
    re_fin=freq_detected,
    order=10,
    disp=1
)
plt.suptitle('Error Decomposition Analysis')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'quickstart_error_decomp.png'), dpi=150)
plt.close()

print(f"     Phase offset: {phi:.4f} rad")
print(f"     Independent error RMS: {np.std(indep):.6f}")
print(f"     Dependent error RMS: {np.std(dep):.6f}")

# 6b. Error histogram by phase
print("\n  6b. Error histogram by phase...")
fig_hist = plt.figure(figsize=(12, 8))
emean, erms, phase_axis, anoi, pnoi, _, _ = err_hist_sine(
    calibrated_output,
    bins=99,
    fin=freq_detected,
    display=True,
    mode=0  # Phase mode
)
plt.suptitle('Error Histogram by Phase')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'quickstart_error_hist.png'), dpi=150)
plt.close()

print(f"     Amplitude noise: {anoi:.6f}")
print(f"     Phase noise: {pnoi:.6f} rad")

#%% Step 7: Create Summary Plot
print("\nStep 7: Create Summary Comparison Plot")
print("-" * 70)

fig_summary, axes = plt.subplots(1, 2, figsize=(16, 6))

# Before calibration
plt.sca(axes[0])
spec_plot(uncalibrated_output, label=True, harmonic=5)
axes[0].set_title(f'Before Calibration\nENoB: {enob_uncal:.2f} bits, SNDR: {sndr_uncal:.2f} dB',
                  fontsize=14, fontweight='bold')

# After calibration
plt.sca(axes[1])
spec_plot(calibrated_output, label=True, harmonic=5)
axes[1].set_title(f'After Calibration\nENoB: {enob_cal:.2f} bits, SNDR: {sndr_cal:.2f} dB',
                  fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'quickstart_summary.png'), dpi=150)
plt.close()

#%% Summary
print("\n" + "=" * 70)
print("Quickstart Complete!")
print("=" * 70)

print(f"\nGenerated files in '{output_dir}/':")
print("  - quickstart_uncalibrated.png    (Spectrum before calibration)")
print("  - quickstart_calibrated.png      (Spectrum after calibration)")
print("  - quickstart_error_decomp.png    (Error decomposition analysis)")
print("  - quickstart_error_hist.png      (Error histogram by phase)")
print("  - quickstart_summary.png         (Side-by-side comparison)")

print("\n" + "=" * 70)
print("Next Steps:")
print("=" * 70)
print("\n1. Explore individual tools:")
print("   - See examples/aout/ for analog analysis tools")
print("   - See examples/dout/ for digital calibration tools")
print("   - See examples/common/ for utility functions")

print("\n2. Load your own ADC data:")
print("   - Replace the data generation step with:")
print("     data = np.loadtxt('your_adc_data.csv')")

print("\n3. Run complete workflows:")
print("   - See examples/workflows/ for comprehensive analysis")

print("\n4. Customize parameters:")
print("   - Adjust window functions (hamming, hann, blackman)")
print("   - Change number of harmonics for analysis")
print("   - Modify calibration polynomial order")

print("\n" + "=" * 70)

def main():
    """Entry point for CLI command."""
    pass  # Script already executed at module level

if __name__ == "__main__":
    main()
