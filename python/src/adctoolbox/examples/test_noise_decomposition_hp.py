"""Test amplitude/phase noise decomposition - High Precision Fix"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# FIX 1: Increase N significantly to handle the huge difference between PM (1000uV) and AM (1uV)
N = 2**20  # ~1 Million points (was 2^14)
Fs = 800e6
Fin = 123/N * Fs
t = np.arange(N) / Fs
A, DC = 0.5, 0.0

# --- 1. SET KNOWN NOISE PARAMETERS ---
real_amp_noise_rms = 1e-6       # 1 uV
real_phase_noise_rms = 1000e-6  # 1000 urad
thermal_noise_floor = 1e-6      # 1 uV

print(f"Generating {N} samples...")
np.random.seed(42)

# Generate Noise
n_amp = np.random.randn(N) * real_amp_noise_rms
n_phase = np.random.randn(N) * real_phase_noise_rms
n_thermal = np.random.randn(N) * thermal_noise_floor

# Signal Model
phase_arg = 2*np.pi*Fin*t
signal_clean = A * np.sin(phase_arg) + DC
signal_noisy = (A + n_amp) * np.sin(phase_arg + n_phase) + DC + n_thermal

# Calculate Error
err = signal_noisy - signal_clean

# --- 2. DECOMPOSITION (Advanced: Raw Regression) ---
# Instead of binning, we perform regression on the raw data points.
# This eliminates bin-width artifacts and alignment errors.

print("Performing decomposition...")

# Calculate exact phase for every point 0..360
phase_vals = np.mod(phase_arg * 180 / np.pi, 360)
rad_vals = phase_vals * np.pi / 180

# Basis functions for every single point
asen_raw = np.sin(rad_vals)**2
psen_raw = np.cos(rad_vals)**2

# Solve Least Squares on 1 million points: err^2 = c1*sin^2 + c2*cos^2
# Note: Thermal noise (constant) will be distributed equally into c1 and c2
A_matrix = np.column_stack([asen_raw, psen_raw])
coeffs = np.linalg.lstsq(A_matrix, err**2, rcond=None)[0]

# Extract results
calc_anoi = np.sqrt(max(coeffs[0], 0))
calc_pnoi_rad = np.sqrt(max(coeffs[1], 0)) / A

# --- 3. PLOTTING (Visualization only) ---
# We still bin for plotting purposes, but not for calculation
bin_count = 100
phase_axis_edges = np.arange(bin_count) / bin_count * 360
phase_axis_centers = (np.arange(bin_count) + 0.5) / bin_count * 360 # FIX 2: Use Centers

# Quick binning for plot
esum_sq = np.zeros(bin_count)
enum = np.zeros(bin_count)
bin_indices = (phase_vals / 360 * bin_count).astype(int) % bin_count
np.add.at(esum_sq, bin_indices, err**2)
np.add.at(enum, bin_indices, 1)
with np.errstate(divide='ignore', invalid='ignore'):
    erms = np.sqrt(esum_sq / enum)

fig = plt.figure(figsize=(10, 8))
ax = plt.subplot(1, 1, 1)

# Plot measured RMS
ax.bar(phase_axis_centers, erms*1e6, width=360/bin_count, color='lightgray', label='Measured RMS Error')

# Plot fitted curves
rad_plot = phase_axis_centers * np.pi / 180
asen_plot = np.sin(rad_plot)**2
psen_plot = np.cos(rad_plot)**2
curve_am = np.sqrt(coeffs[0] * asen_plot)
curve_pm = np.sqrt(coeffs[1] * psen_plot)
curve_total = np.sqrt(coeffs[0]*asen_plot + coeffs[1]*psen_plot)

ax.plot(phase_axis_centers, curve_total*1e6, 'k--', linewidth=2, label='Total Fit')
ax.plot(phase_axis_centers, curve_am*1e6, 'b-', linewidth=2, label=f'Extracted AM ({calc_anoi*1e6:.1f} uV)')
ax.plot(phase_axis_centers, curve_pm*1e6, 'r-', linewidth=2, label=f'Extracted PM')

ax.set_xlabel('Phase (deg)')
ax.set_ylabel('RMS Noise (uV)')
ax.set_title(f'Noise Decomposition (N={N})')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 360)

plt.tight_layout()
fig_path = output_dir / 'precision_test.png'
plt.savefig(fig_path, dpi=150)
print(f"Saved to {fig_path}")

# --- Validation Print ---
print(f"\n[Validation N={N}]")
print(f"Amplitude Noise:")
print(f"  Input : {real_amp_noise_rms*1e6:.2f} uV")
print(f"  Calc  : {calc_anoi*1e6:.2f} uV")
print(f"Phase Noise:")
print(f"  Input : {real_phase_noise_rms*1e6:.2f} urad")
print(f"  Calc  : {calc_pnoi_rad*1e6:.2f} urad")