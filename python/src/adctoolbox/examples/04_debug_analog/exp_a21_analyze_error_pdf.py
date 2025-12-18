"""Error PDF Analysis: Sweep across 12 different ADC non-idealities

This experiment demonstrates error PDF analysis for various ADC impairments
including thermal noise, jitter, static/dynamic nonlinearities, and circuit effects.
Shows how different error mechanisms produce distinct PDF patterns.
"""

import sys
from pathlib import Path

# Add the example directory to sys.path to allow running from anywhere
sys.path.insert(0, str(Path(__file__).parent))

import time

# --- 1. Timing: Imports ---
t_start = time.time()
import matplotlib.pyplot as plt
from adctoolbox import analyze_error_pdf
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
print(f"{'Case':<4} | {'Non-Ideality':<30} | {'mu (LSB)':<10} | {'sigma (LSB)':<10} | {'KL Div':<10}")
print("-" * 100)

for idx, case in enumerate(CASES):
    # Generate signal
    signal = case['func']()

    # Analyze error PDF
    result = analyze_error_pdf(signal, resolution=params['B'], ax=axes[idx], title=case['title'])

    # Print metrics
    print(f"{idx+1:<4} | {case['title']:<30} | {result['mu']:>10.3f} | {result['sigma']:>10.3f} | {result['kl_divergence']:>10.4f}")

print("=" * 100)
print(f"\n[Timing] Signal Generation & Plotting: {time.time() - t_plot:.4f}s")

# --- 5. Finalize and Save ---
t_save = time.time()

fig.suptitle(f'Error PDF Analysis: 15 ADC Non-idealities (Fs={params["Fs"]/1e6:.0f} MHz, Fin={params["Fin"]/1e6:.1f} MHz, {params["B"]}-bit)',
             fontsize=14, fontweight='bold')
plt.tight_layout()

fig_path = (output_dir / 'exp_a21_analyze_error_pdf.png').resolve()
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"[Save fig] -> [{fig_path}]")
print(f"[Timing] Image Rendering & Saving: {time.time() - t_save:.4f}s")
print(f"\n--- Total Runtime: {time.time() - t_start:.4f}s ---\n")
