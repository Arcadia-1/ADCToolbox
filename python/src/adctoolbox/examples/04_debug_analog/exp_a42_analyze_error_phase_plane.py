"""Residual Phase Plane Analysis: Magnify Harmonic Distortion 1000x

This experiment demonstrates residual phase plane (error vs amplitude) visualization
for various ADC anomalies. By fitting and removing the fundamental sine wave, tiny
harmonic distortions become visible that are invisible in regular phase planes.

Key Advantages over Standard Phase Plane:
- 1000x more sensitive to harmonic distortion (HD2/HD3)
- Reveals linearity issues at -80 dBc and below
- Shows characteristic shapes:
  * Parabola (U-shape): HD2 - gain asymmetry
  * S-curve: HD3 - compression/saturation
  * Sharp breaks: Clipping
  * Horizontal line: Good linearity (only noise)

Use Cases:
- Detecting subtle harmonic distortion invisible in time/frequency domain
- Visualizing error vs amplitude characteristics for different non-idealities
- INL characterization without histogram test
- Validating linearity in high-ENOB ADCs
"""

import sys
from pathlib import Path

# Add the example directory to sys.path to allow running from anywhere
sys.path.insert(0, str(Path(__file__).parent))

import time

# --- 1. Timing: Imports ---
t_start = time.time()
import matplotlib.pyplot as plt
from adctoolbox.aout import analyze_error_phase_plane
from nonideality_cases import get_batch_test_setup  # type: ignore

print(f"[Timing] Library Imports: {time.time() - t_start:.4f}s")

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# --- 2. Timing: Setup ---
t_setup = time.time()

# Get batch test setup
gen, CASES, params = get_batch_test_setup(hd2_target_dB=-80, hd3_target_dB=-70)

print(f"[Config] Fs={params['Fs']/1e6:.0f} MHz, Fin={params['Fin']/1e6:.1f} MHz, Bin={params['Fin_bin']}, N={params['N']}")
print(f"[Config] A={params['A']:.3f} V, DC={params['DC']:.3f} V, Resolution={params['B']} bits")
print(f"[Timing] Setup: {time.time() - t_setup:.4f}s\n")

# --- 3. Generate Signals and Plot ---
t_plot = time.time()

# Create 3x5 subplot grid
fig, axes = plt.subplots(3, 5, figsize=(25, 12))
axes = axes.flatten()

print("\n" + "=" * 80)
print(f"{'Case':<4} | {'Non-Ideality':<30} | {'RMS Error':<12} | {'Peak Error':<12}")
print("-" * 80)

for idx, case in enumerate(CASES):
    # Generate signal
    signal = case['func']()

    # Use the analyze_error_phase_plane function with ax parameter
    result = analyze_error_phase_plane(
        signal,
        fs=params['Fs'],
        ax=axes[idx],
        title=case['title'],
        create_plot=False,
        fit_polynomial_order=3
    )

    # Print metrics (convert from V to uV)
    rms_uV = result['residual'].std() * 1e6
    peak_uV = abs(result['residual']).max() * 1e6
    print(f"{idx+1:<4} | {case['title']:<30} | "
          f"{rms_uV:>10.1f} uV | "
          f"{peak_uV:>10.1f} uV")

print("=" * 80)
print(f"\n[Timing] Signal Generation & Plotting: {time.time() - t_plot:.4f}s")

# --- 4. Finalize and Save ---
t_save = time.time()

fig.suptitle(f'Residual Phase Plane Analysis: 15 ADC Non-idealities (Fs={params["Fs"]/1e6:.0f} MHz, Fin={params["Fin"]/1e6:.1f} MHz, {params["B"]}-bit)',
             fontsize=14, fontweight='bold')

plt.tight_layout()

fig_path = (output_dir / 'exp_a42_analyze_error_phase_plane.png').resolve()
print(f"[Save fig] -> [{fig_path}]")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"[Timing] Image Saving: {time.time() - t_save:.4f}s")

print(f"\n--- Total Runtime: {time.time() - t_start:.4f}s ---")

print("\n" + "=" * 80)
print("Key Observations:")
print("-" * 80)
print("* HD2/HD3 cases show characteristic parabola/S-curve shapes even at -80/-70 dBc")
print("* Clipping shows sharp breaks at signal extremes")
print("* Memory effect and RA gain errors create subtle curvature")
print("* Thermal/quantization noise appear as horizontal scatter (good linearity)")
print("* This method is 1000x more sensitive than regular phase planes for detecting harmonics")
print("=" * 80)
