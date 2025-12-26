"""Sweep static nonlinearity coefficients to analyze harmonic distortion.

Demonstrates INL-induced harmonic generation in ADC spectrum.
"""

import time
t_start = time.perf_counter()

import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, analyze_spectrum 
from adctoolbox.siggen import ADC_Signal_Generator 

t_import = time.perf_counter() - t_start
t_prep_start = time.perf_counter()

# Setup
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# Parameters
N = 2**16           
Fs = 1000e6         
Fin_target = 80e6  
Fin, _ = find_coherent_frequency(Fs, Fin_target, N)
A, DC = 0.5, 0.5    
base_noise = 50e-6  

# CORE: Define Target Magnitudes
HD2_TARGET_DB = -90.0
HD3_TARGET_DB = -80.0

# Helper Function: Calculate k magnitude
def calculate_k_mag(gen_instance, db_val, order):
    amp_ratio = 10 ** (db_val / 20.0)
    return (2**(order-1) * amp_ratio) / (gen_instance.A**(order-1))

# Initialize Generator and calculate base k values
gen = ADC_Signal_Generator(N=N, Fs=Fs, Fin=Fin, A=A, DC=DC)

K_HD2_MAG = calculate_k_mag(gen, HD2_TARGET_DB, 2)
K_HD3_MAG = calculate_k_mag(gen, HD3_TARGET_DB, 3)

# Define 4 Sweep Cases (Sign Combinations only)
cases = [
    {'title': 'k2(+), k3(+)',  'k2': K_HD2_MAG, 'k3': K_HD3_MAG},
    {'title': 'k2(-), k3(+)',  'k2': -K_HD2_MAG, 'k3': K_HD3_MAG},
    {'title': 'k2(+), k3(-)',  'k2': K_HD2_MAG, 'k3': -K_HD3_MAG},
    {'title': 'k2(-), k3(-)',  'k2': -K_HD2_MAG, 'k3': -K_HD3_MAG},
]

# Prepare Figure (1 row x 4 columns)
n_cols = 4
n_rows = 1
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 6))
axes = axes.flatten()

print("=" * 80)
print(f"{'Case Title':<20} | {'SFDR (dB)':<10} | {'THD (dB)':<10} | {'HD2 (Meas)':<12} | {'HD3 (Meas)':<12}")
print("-" * 80)

t_prep = time.perf_counter() - t_prep_start
t_loop_start = time.perf_counter()

t_tool_total = 0.0
# Run Sweep
for idx, config in enumerate(cases):
    
    sig_nonlinear = gen.apply_static_nonlinearity(
        input_signal=None, 
        k2=config['k2'], 
        k3=config['k3']
    )
    
    signal = gen.apply_thermal_noise(sig_nonlinear, noise_rms=base_noise)
    
    plt.sca(axes[idx])
    
    t_tool_start = time.perf_counter()
    result = analyze_spectrum(signal, fs=Fs)
    t_tool_total += time.perf_counter() - t_tool_start
    
    axes[idx].set_title(config['title'], fontsize=11, fontweight='bold')
        
    print(f"{config['title']:<20} | {result['sfdr_db']:<10.2f} | {result['thd_db']:<10.2f} | {result['hd2_db']:<12.2f} | {result['hd3_db']:<12.2f}")

t_loop = time.perf_counter() - t_loop_start
t_fig_start = time.perf_counter()

# Finalize
plt.suptitle(f'Static Nonlinearity Sweep: HD2/HD3 Sign Analysis\n|HD2|={HD2_TARGET_DB}dBc, |HD3|={HD3_TARGET_DB}dBc', 
             fontsize=14, fontweight='bold', y=0.99)
plt.tight_layout()

fig_path = output_dir / "exp_g05_sweep_nonlinear_sign_fixed.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n[Save figure] -> [{fig_path}]")

t_fig = time.perf_counter() - t_fig_start
t_total = time.perf_counter() - t_start

print(f"\n{'='*60}")
print(f"Timing Report:")
print(f"{'='*60}")
print(f"  Import time:      {t_import*1000:7.2f} ms")
print(f"  Preparation time: {t_prep*1000:7.2f} ms")
print(f"  Core tool time:   {t_tool_total*1000:7.2f} ms  (analyze_spectrum x4)")
print(f"  Main loop time:   {t_loop*1000:7.2f} ms")
print(f"  Figure time:      {t_fig*1000:7.2f} ms")
print(f"  Total runtime:    {t_total*1000:7.2f} ms")
print(f"{'='*60}\n")
