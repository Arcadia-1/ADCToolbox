"""
Batch AOUT Dashboard Generation: 15 ADC Non-idealities

This example generates 15 comprehensive 12-tool dashboards, one for each
standard ADC non-ideality case. Each dashboard provides a complete diagnostic
view combining spectrum, error analysis, phase plane, and statistical tools.

This is useful for:
- Creating a comprehensive test report for ADC characterization
- Comparing how different non-idealities manifest across all analysis domains
- Generating documentation and reference materials
"""

import sys
from pathlib import Path

# Add the example directory to sys.path to allow running from anywhere
sys.path.insert(0, str(Path(__file__).parent.parent / "04_debug_analog"))

import time

# --- 1. Timing: Imports ---
t_start = time.time()

# Set non-interactive backend before importing analysis libraries
import matplotlib
matplotlib.use('Agg')

from adctoolbox.toolset import generate_aout_dashboard
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

# --- 3. Generate Dashboards ---
t_batch = time.time()

print("\n" + "=" * 100)
print(f"{'#':<4} | {'Non-Ideality':<30} | {'Status':<40} | {'Time (s)':<10}")
print("-" * 100)

for idx, case in enumerate(CASES):
    t_case = time.time()

    # Generate signal
    signal = case['func']()

    # Create dashboard filename
    # Sanitize title for filename (replace spaces and special chars)
    safe_title = case['title'].replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
    fig_path = (output_dir / f'exp_t02_dashboard_{idx+1:02d}_{safe_title}.png').resolve()

    # Generate dashboard
    fig, axes = generate_aout_dashboard(
        signal=signal,
        fs=params['Fs'],
        freq=params['Fin'],
        resolution=params['B'],
        output_path=fig_path
    )

    elapsed = time.time() - t_case
    print(f"{idx+1:<4} | {case['title']:<30} | {str(fig_path.name):<40} | {elapsed:<10.3f}")

print("=" * 100)
print(f"\n[Timing] Batch Dashboard Generation: {time.time() - t_batch:.4f}s")

print(f"\n--- Total Runtime: {time.time() - t_start:.4f}s ---\n")

print("\n" + "=" * 100)
print("Dashboard Generation Complete!")
print("-" * 100)
print(f"Generated {len(CASES)} comprehensive 12-tool dashboards")
print(f"Each dashboard includes:")
print("  1. Spectrum                    7. Error PDF")
print("  2. Spectrum Polar              8. Error Autocorrelation")
print("  3. Error by Value              9. Error Spectrum")
print("  4. Error by Phase             10. Error Envelope Spectrum")
print("  5. Decomposition Time         11. Phase Plane")
print("  6. Decomposition Polar        12. Error Phase Plane")
print(f"\nOutput directory: {output_dir}")
print("=" * 100)
