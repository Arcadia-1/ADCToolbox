"""Phase-based error analysis: Raw data mode WITHOUT baseline

Demonstrates raw mode (high-precision, all samples) with baseline noise term
EXCLUDED from the AM/PM fitting model (forced to pure AM/PM only).
"""

import time

# --- 1. Timing: Imports ---
t_start = time.time()
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import analyze_error_by_phase

print(f"[Timing] Library Imports: {time.time() - t_start:.4f}s")

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# --- 2. Timing: Data Generation ---
t_gen = time.time()

# Setup parameters
N = 2**16
Fs = 800e6
Fin = 10.1234567e6
normalized_freq = Fin / Fs
t = np.arange(N) / Fs
A = 0.49
target_thermal = 50e-6
target_pm_rad = 0.05
target_am = 50e-6
print(f"[Config] Fs={Fs/1e6:.0f} MHz, Fin={Fin/1e6:.2f} MHz, N={N}")

# Set seed for reproducibility
np.random.seed(42)

# Case 1: Thermal Noise Only (White noise baseline)
phase_clean = 2 * np.pi * Fin * t
n_thermal = np.random.randn(N) * target_thermal
sig_thermal_only = A * np.sin(phase_clean) + n_thermal

# Case 2: Phase Jitter Only (Phase modulation)
n_pm = np.random.randn(N) * target_pm_rad
phase_jittered = phase_clean + n_pm
sig_pm_only = A * np.sin(phase_jittered)

# Case 3: Amplitude Modulation Only (Amplitude noise)
n_am = np.random.randn(N) * target_am
sig_am_only = (A + n_am) * np.sin(phase_clean)

print(f"[Timing] Data Generation: {time.time() - t_gen:.4f}s")

# --- 3. Timing: Analysis & Plotting ---
t_plot = time.time()

fig = plt.figure(figsize=(18, 12))
fig.suptitle('Phase Error Analysis (Raw Mode WITHOUT Baseline) - Thermal Noise vs Phase Jitter vs Amplitude Modulation', fontsize=16, fontweight='bold')

# Analyze 3 cases
# Case 1: Thermal Noise Only
ax1 = plt.subplot(3, 2, 1)
analyze_error_by_phase(sig_thermal_only, normalized_freq, data_mode="raw", include_baseline=False, ax=ax1)

# Case 2: Phase Jitter Only
ax2 = plt.subplot(3, 2, 2)
analyze_error_by_phase(sig_pm_only, normalized_freq, data_mode="raw", include_baseline=False, ax=ax2)

# Case 3: Amplitude Modulation Only
ax3 = plt.subplot(3, 2, 3)
analyze_error_by_phase(sig_am_only, normalized_freq, data_mode="raw", include_baseline=False, ax=ax3)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])

print(f"[Timing] Analysis & Plotting Setup: {time.time() - t_plot:.4f}s")

# --- 4. Timing: File Saving ---
t_save = time.time()

fig_path = (output_dir / 'exp_a14_error_phase_raw_without_baseline.png').resolve()
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"[Save fig] -> [{fig_path}]")
print(f"[Timing] Image Rendering & Saving: {time.time() - t_save:.4f}s")

# ============================================================================
# 3-Plot Comparison: Thermal Noise vs Phase Noise vs Amplitude Noise
# ============================================================================

print("\n" + "="*80)
print("3-PLOT COMPARISON: Thermal Noise Only vs Phase Noise Only vs Amplitude Noise Only")
print("="*80 + "\n")

# Create 3-plot comparison figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Set titles first
axes[0].set_title(f'Thermal Noise Only\n({target_thermal*1e6:.0f} µV)', fontsize=12, fontweight='bold', pad=10)
axes[1].set_title(f'Phase Noise Only\n({target_pm_rad*1e6:.0f} µV)', fontsize=12, fontweight='bold', pad=10)
axes[2].set_title(f'Amplitude Noise Only\n({target_am*1e6:.0f} µV)', fontsize=12, fontweight='bold', pad=10)

fig.suptitle('Phase Error Analysis Comparison: Thermal vs Phase vs Amplitude Noise (Raw Mode WITHOUT Baseline)',
             fontsize=14, fontweight='bold')

# Case 1: Thermal Noise Only
analyze_error_by_phase(sig_thermal_only, normalized_freq, data_mode="raw", include_baseline=False, ax=axes[0])

# Case 2: Phase Noise Only
analyze_error_by_phase(sig_pm_only, normalized_freq, data_mode="raw", include_baseline=False, ax=axes[1])

# Case 3: Amplitude Noise Only
analyze_error_by_phase(sig_am_only, normalized_freq, data_mode="raw", include_baseline=False, ax=axes[2])

plt.tight_layout()
fig_path_3plot = (output_dir / 'exp_a14_error_phase_raw_without_baseline_3plot.png').resolve()
plt.savefig(fig_path_3plot, dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"[Save fig] -> [{fig_path_3plot}]")

print(f"--- Total Runtime: {time.time() - t_start:.4f}s ---\n")
