"""Phase Plane Analysis: Sweep across 15 different ADC non-idealities

This experiment demonstrates phase plane (lag plot) visualization for various ADC
anomalies. Plots x[n] vs x[n+k] to reveal issues that may not be visible in time
or frequency domain.

Use Cases:
- Sparkle Codes: Random outliers appearing as scattered red points
- Hysteresis: Loop/figure-8 patterns in the trajectory
- Metastability: Hesitation or clustering near decision thresholds
- Quantization: Regular grid patterns
"""

import sys
from pathlib import Path

# Add the example directory to sys.path to allow running from anywhere
sys.path.insert(0, str(Path(__file__).parent))

import time

# --- 1. Timing: Imports ---
t_start = time.time()
import matplotlib.pyplot as plt
from adctoolbox.aout import analyze_phase_plane
from nonideality_cases import get_batch_test_setup  # type: ignore

print(f"[Timing] Library Imports: {time.time() - t_start:.4f}s")

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# --- 2. Timing: Setup ---
t_setup = time.time()

# Get batch test setup
gen, CASES, params = get_batch_test_setup()

print(f"[Config] Fs={params['Fs']/1e6:.0f} MHz, Fin={params['Fin']/1e6:.1f} MHz, Bin={params['Fin_bin']}, N={params['N']}")
print(f"[Config] A={params['A']:.3f} V, DC={params['DC']:.3f} V, Resolution={params['B']} bits")
print(f"[Timing] Setup: {time.time() - t_setup:.4f}s\n")

# --- 3. Get Standard Non-ideality Cases ---
t_gen = time.time()

print(f"[Timing] Case Definition: {time.time() - t_gen:.4f}s")

# --- 4. Generate Signals and Plot ---
t_plot = time.time()

# Create 3x5 subplot grid
fig, axes = plt.subplots(3, 5, figsize=(25, 12))
axes = axes.flatten()

print("\n" + "=" * 100)
print(f"{'Case':<4} | {'Non-Ideality':<30} | {'Lag':<6} | {'Outliers':<10}")
print("-" * 100)

for idx, case in enumerate(CASES):
    # Generate signal
    signal = case['func']()
    
    # Use the analyze_phase_plane function with ax parameter
    result = analyze_phase_plane(
        signal,
        lag='auto',
        fs=params['Fs'],
        detect_outliers=True,
        threshold=4.0,
        ax=axes[idx],
        title=case['title'],
        create_plot=False
    )
    
    # Print metrics
    print(f"{idx+1:<4} | {case['title']:<30} | {result['lag']:<6} | {len(result['outliers']):<10}")

print("=" * 100)
print(f"\n[Timing] Signal Generation & Plotting: {time.time() - t_plot:.4f}s")

# --- 5. Finalize and Save ---
t_save = time.time()

fig.suptitle(f'Phase Plane Analysis: 15 ADC Non-idealities (Fs={params["Fs"]/1e6:.0f} MHz, Fin={params["Fin"]/1e6:.1f} MHz, {params["B"]}-bit)',
             fontsize=14, fontweight='bold')

plt.tight_layout()

fig_path = (output_dir / 'exp_a41_analyze_phase_plane.png').resolve()
print(f"[Save fig] -> [{fig_path}]")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"[Timing] Image Saving: {time.time() - t_save:.4f}s")

print(f"\n--- Total Runtime: {time.time() - t_start:.4f}s ---")
