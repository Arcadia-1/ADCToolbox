"""Sweep static nonlinearity coefficients to analyze harmonic distortion.

Demonstrates INL-induced harmonic generation in ADC spectrum.
"""

import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, analyze_spectrum 
from adctoolbox.siggen import ADC_Signal_Generator 

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

# Run Sweep
for idx, config in enumerate(cases):
    
    sig_nonlinear = gen.apply_static_nonlinearity(
        input_signal=None, 
        k2=config['k2'], 
        k3=config['k3']
    )
    
    signal = gen.apply_thermal_noise(sig_nonlinear, noise_rms=base_noise)
    
    plt.sca(axes[idx])
    result = analyze_spectrum(signal, fs=Fs)
    axes[idx].set_title(config['title'], fontsize=11, fontweight='bold')
        
    print(f"{config['title']:<20} | {result['sfdr_dbc']:<10.2f} | {result['thd_dbc']:<10.2f} | {result['harmonics_dbc'][0]:<12.2f} | {result['harmonics_dbc'][1]:<12.2f}")

# Finalize
plt.suptitle(f'Static Nonlinearity Sweep: HD2/HD3 Sign Analysis\n|HD2|={HD2_TARGET_DB}dBc, |HD3|={HD3_TARGET_DB}dBc', 
             fontsize=14, fontweight='bold', y=0.99)
plt.tight_layout()

fig_path = output_dir / "exp_g05_sweep_nonlinear_sign_fixed.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n[Save figure] -> [{fig_path}]\n")
