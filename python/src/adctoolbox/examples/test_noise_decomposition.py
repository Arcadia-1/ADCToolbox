"""Test amplitude/phase noise decomposition with known clean signal (FIXED)"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N = 2**14  # Increase samples for better statistics
Fs = 800e6
Fin = 123/N *Fs
t = np.arange(N) / Fs
A, DC = 0.5, 0.0

# --- 1. SET KNOWN NOISE PARAMETERS ---
# Intentionally add specific AM and PM noise to test the algorithm
real_amp_noise_rms = 1e-6  # 500 uV Amplitude Noise
real_phase_noise_rms = 1000e-6 # 2000 urad Phase Noise (scaled by A later)
thermal_noise_floor = 1e-6 # 100 uV Thermal Noise

np.random.seed(42)

# Generate Random Noise Vectors
n_amp = np.random.randn(N) * real_amp_noise_rms
n_phase = np.random.randn(N) * real_phase_noise_rms
n_thermal = np.random.randn(N) * thermal_noise_floor

# Create Signals
# Signal Model: (A + n_amp) * sin(wt + n_phase) + DC + n_thermal
phase_arg = 2*np.pi*Fin*t
signal_clean = A * np.sin(phase_arg) + DC
signal_noisy = (A + n_amp) * np.sin(phase_arg + n_phase) + DC + n_thermal

# Calculate Error (Residual)
err = signal_noisy - signal_clean

# --- 2. PHASE BINNING ---
bin_count = 100
# Calculate ideal phase 0..360 for each sample
phase_vals = np.mod(phase_arg * 180 / np.pi, 360)
phase_axis = np.arange(bin_count) / bin_count * 360

# Binning process
esum_sq = np.zeros(bin_count)
enum = np.zeros(bin_count)

# Vectorized binning (faster than loop)
bin_indices = (phase_vals / 360 * bin_count).astype(int) % bin_count
np.add.at(esum_sq, bin_indices, err**2)
np.add.at(enum, bin_indices, 1)

with np.errstate(divide='ignore', invalid='ignore'):
    erms = np.sqrt(esum_sq / enum)

# --- 3. DECOMPOSITION (FIXED PHYSICS) ---
# FIX 1: Amplitude noise scales with sin^2 (peaks), Phase noise with cos^2 (zero crossings)
# We normalize x-axis phase to radians for calculation
rad_axis = phase_axis / 180 * np.pi
asen = np.sin(rad_axis)**2  # Amplitude Sensitivity
psen = np.cos(rad_axis)**2  # Phase Sensitivity

valid_mask = ~np.isnan(erms) & (enum > 10) # Filter empty bins
erms_squared = erms[valid_mask]**2
asen_fit = asen[valid_mask]
psen_fit = psen[valid_mask]

# FIX 2: Solve erms² = A_noise² * asen + P_noise² * (A*psen)
# We REMOVE the constant term to avoid rank deficiency (sin^2 + cos^2 = 1).
# Thermal noise will be distributed roughly equally into both components.
A_matrix = np.column_stack([asen_fit, psen_fit])
coeffs = np.linalg.lstsq(A_matrix, erms_squared, rcond=None)[0]

# Extract results
# coeff[0] is Variance of Amplitude Noise
# coeff[1] is Variance of (Phase Noise * Signal Amplitude)
calc_anoi = np.sqrt(max(coeffs[0], 0))
calc_pnoi_rad = np.sqrt(max(coeffs[1], 0)) / A # Normalize by Amplitude to get radians

# Compute fitted curves for visualization
rms_fit_curve = np.sqrt(coeffs[0]*asen + coeffs[1]*psen)

# --- 4. PLOTTING ---
fig = plt.figure(figsize=(10, 8))
ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)

# Top: Data and Error vs Phase (1 full cycle, dual y-axis like MATLAB)
ax1_left = ax1
ax1_left.plot(phase_vals, signal_noisy, 'k.', markersize=2, label='data')
ax1_left.set_xlim([0, 360])
ax1_left.set_ylim([np.min(signal_noisy), np.max(signal_noisy)])
ax1_left.set_ylabel('data', color='k')
ax1_left.tick_params(axis='y', labelcolor='k')

ax1_right = ax1.twinx()
ax1_right.plot(phase_vals, err, 'r.', markersize=2, alpha=0.5)
ax1_right.set_xlim([0, 360])
ax1_right.set_ylim([np.min(err), np.max(err)])
ax1_right.set_ylabel('error', color='r')
ax1_right.tick_params(axis='y', labelcolor='r')

ax1.legend(['data', 'error'], loc='upper right')
ax1.set_xlabel('phase(deg)')
ax1.set_title(f'Error - Phase\nActual AM Noise: {real_amp_noise_rms*1e6:.0f}uV, Actual PM Noise: {real_phase_noise_rms*1e6:.0f}urad')
ax1.grid(True, alpha=0.3)

# Bottom: RMS Decomposition
ax2.bar(phase_axis, erms*1e6, width=360/bin_count, color='lightgray', label='Measured RMS Error')
ax2.plot(phase_axis, rms_fit_curve*1e6, 'k--', linewidth=2, label='Total Fit')

# Plot components
comp_am = np.sqrt(coeffs[0] * asen)
comp_pm = np.sqrt(coeffs[1] * psen)
ax2.plot(phase_axis, comp_am*1e6, 'b-', linewidth=2, label='Extracted AM Noise')
ax2.plot(phase_axis, comp_pm*1e6, 'r-', linewidth=2, label='Extracted PM Noise (scaled)')

ax2.set_xlabel('Phase (deg)')
ax2.set_ylabel('RMS Noise (uV)')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 360)

# Results Text
res_text = (
    f"--- RESULTS ---\n"
    f"Amplitude Noise:\n"
    f"  Input: {real_amp_noise_rms*1e6:.1f} uV\n"
    f"  Calc : {calc_anoi*1e6:.1f} uV\n"
    f"Phase Noise:\n"
    f"  Input: {real_phase_noise_rms*1e6:.1f} urad\n"
    f"  Calc : {calc_pnoi_rad*1e6:.1f} urad"
)
ax2.text(10, np.max(erms)*1e6*0.8, res_text, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
fig_path = output_dir / 'fixed_noise_decomposition.png'
plt.savefig(fig_path, dpi=150)
print(f"Saved to {fig_path}")
plt.close()

# Validation Print
print(f"[Validation]")
print(f"Amplitude Noise (Input vs Calc): {real_amp_noise_rms*1e6:.1f} vs {calc_anoi*1e6:.1f} uV")
print(f"Phase Noise     (Input vs Calc): {real_phase_noise_rms*1e6:.1f} vs {calc_pnoi_rad*1e6:.1f} urad")