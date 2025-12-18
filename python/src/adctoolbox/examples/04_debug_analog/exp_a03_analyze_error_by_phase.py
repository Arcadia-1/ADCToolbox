"""Phase-based error analysis with AM/PM decomposition.

Demonstrates dual-track parallel design:
- Raw fitting on all N samples for highest precision numerics
- Binned statistics for visualization with R² validation

9 Test Cases in 6 Figures (3 figures × 2 baseline modes):
- Figure 1 (Pure): Thermal only, AM only, PM only
- Figure 2 (Mixed): AM+Thermal, PM+Thermal, AM+PM+Thermal
- Figure 3 (AM+PM): Equal, PM dominates, AM dominates

Each set is generated twice:
- With base noise (include_base_noise=True)
- Without base noise (include_base_noise=False)
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

# --- 2. Setup & Signal Generation ---
t_gen = time.time()

# Base parameters
N = 2**16
Fs = 800e6
Fin = 10.1234567e6 # no need to be coherent
norm_freq = Fin / Fs
t = np.arange(N) / Fs
A = 0.49  # Signal amplitude
DC = 0.5   # Signal DC offset
phase_clean = 2 * np.pi * Fin * t

print(f"[Config] Fs={Fs/1e6:.0f} MHz, Fin={Fin/1e6:.2f} MHz, N={N}, A={A}")

# =============================================================================
# Define all 9 test cases (noise levels in Volts)
# =============================================================================
test_cases = [
    # --- Figure 1: Pure Cases ---
    {'am': 0,    'pm': 0,    'thermal': 50e-6, 'label': 'Thermal Only',
     'expected': {'am': 0, 'pm': 0, 'baseline': 50}},
    {'am': 50e-6, 'pm': 0,    'thermal': 0,     'label': 'AM Only',
     'expected': {'am': 50, 'pm': 0, 'baseline': 0}},
    {'am': 0,    'pm': 50e-6, 'thermal': 0,     'label': 'PM Only',
     'expected': {'am': 0, 'pm': 50, 'baseline': 0}},

    # --- Figure 2: Mixed Cases ---
    {'am': 50e-6, 'pm': 0,    'thermal': 30e-6, 'label': 'AM + Thermal',
     'expected': {'am': 50, 'pm': 0, 'baseline': 30}},
    {'am': 0,    'pm': 50e-6, 'thermal': 30e-6, 'label': 'PM + Thermal',
     'expected': {'am': 0, 'pm': 50, 'baseline': 30}},
    {'am': 50e-6, 'pm': 50e-6, 'thermal': 30e-6, 'label': 'AM + PM + Thermal',
     'expected': {'am': 0, 'pm': 0, 'baseline': 58}},  # sqrt(50^2+50^2+30^2) when AM=PM

    # --- Figure 3: AM+PM Cases (no thermal) ---
    {'am': 50e-6, 'pm': 50e-6, 'thermal': 0,     'label': 'AM + PM (equal)',
     'expected': {'am': 0, 'pm': 0, 'baseline': 50}},  # Equal cancels to flat
    {'am': 30e-6, 'pm': 50e-6, 'thermal': 0,     'label': 'AM(30) + PM(50)',
     'expected': {'am': 0, 'pm': 40, 'baseline': 30}},  # PM dominates
    {'am': 50e-6, 'pm': 30e-6, 'thermal': 0,     'label': 'AM(50) + PM(30)',
     'expected': {'am': 40, 'pm': 0, 'baseline': 30}},  # AM dominates
]

# Generate signals for all test cases
for case in test_cases:
    am_noise = np.random.randn(N) * case['am'] if case['am'] > 0 else 0
    pm_noise = np.random.randn(N) * case['pm'] / A if case['pm'] > 0 else 0
    th_noise = np.random.randn(N) * case['thermal'] if case['thermal'] > 0 else 0
    case['signal'] = (A + am_noise) * np.sin(phase_clean + pm_noise) + DC + th_noise

print(f"[Timing] Signal Generation: {time.time() - t_gen:.4f}s")

# --- 3. Analysis & Plotting ---
t_plot = time.time()
fig_titles = ['Pure Noise Cases', 'Mixed Noise Cases', 'Mixed Noise Cases (AM+PM)']
base_noise_modes = [True, False]

for base_noise_mode in base_noise_modes:
    base_noise_suffix = 'with_base_noise' if base_noise_mode else 'no_base_noise'
    print(f"\n{'='*80}")
    print(f"Mode: include_base_noise={base_noise_mode}")
    print(f"{'='*80}")

    for fig_idx in range(3):
        fig, axes = plt.subplots(1, 3, figsize=(18, 8))
        mode_str = 'With Base Noise' if base_noise_mode else 'Without Base Noise'
        fig.suptitle(f'Phase Error Analysis - {fig_titles[fig_idx]} ({mode_str})', fontsize=14, fontweight='bold')
        print(f"\n=== Figure {fig_idx + 1}: {fig_titles[fig_idx]} ({mode_str}) ===")

        for i in range(3):
            case = test_cases[fig_idx * 3 + i]

            plt.sca(axes[i])
            results = analyze_error_by_phase(case['signal'], n_bins=100, include_base_noise=base_noise_mode, title=case['label'])

            exp = case['expected']
            exp_total = np.sqrt((exp['am']**2)/2 + (exp['pm']**2)/2 + exp['baseline']**2)

            print(f"{case['label']:15s}")
            print(f"  [Expected  ] [AM={exp['am']:4.0f} uV] [PM={exp['pm']:4.0f} uV] [Base={exp['baseline']:4.0f} uV] [Total={exp_total:4.1f} uV]")
            print(f"  [Calculated] [AM={results['am_noise_rms_v']*1e6:4.1f} uV] [PM={results['pm_noise_rms_v']*1e6:4.1f} uV] [Base={results['base_noise_rms_v']*1e6:4.1f} uV] [Total={results['total_rms_v']*1e6:4.1f} uV] [R2={results['r_squared_binned']:.3f}]\n")

        plt.tight_layout()
        fig_path = output_dir / f'exp_a03_analyze_error_by_phase_{fig_idx + 1}_{base_noise_suffix}.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[Save fig] -> [{fig_path.resolve()}]")

print(f"\n[Timing] Analysis & Plotting: {time.time() - t_plot:.4f}s")
print(f"--- Total Runtime: {time.time() - t_start:.4f}s ---")
