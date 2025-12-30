"""Compare isolated nonlinearity effects on ADC spectrum.

Each subplot demonstrates one specific non-ideality type.
"""

import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, analyze_spectrum
from adctoolbox.siggen import ADC_Signal_Generator

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N = 2**13
Fs = 1000e6
Fin_target = 80e6
Fin, _ = find_coherent_frequency(Fs, Fin_target, N)

A, DC = 0.5, 0
base_noise = 10e-6

gen = ADC_Signal_Generator(N=N, Fs=Fs, Fin=Fin, A=A, DC=DC)

hd3_dB = -80
hd3_amp = 10**(hd3_dB/20)
k3 = hd3_amp / (A**2 / 4)

print(f"[Setup] Fs={Fs/1e6:.0f} MHz | N={N} | Fin={Fin/1e6:.2f} MHz")
print(f"[Setup] Target HD3 = {hd3_dB:.1f} dBc -> k3 = {k3:.4e}")

# Define Isolated Experiments
EXPERIMENTS = [
    {
        'title': f"Static HD3 Only ({hd3_dB} dBc)",
        'func': 'apply_static_nonlinearity',
        'param': {'k3': k3}
    },
    {
        'title': "Incomplete Settling (+0.005)",
        'func': 'apply_incomplete_sampling',
        'param': {'coeff_k': +0.09}
    },
    {
        'title': "Memory Effect (+0.005)",
        'func': 'apply_memory_effect',
        'param': {'memory_strength': +0.005}
    },
    {
        'title': "Memory Effect (-0.005)",
        'func': 'apply_memory_effect',
        'param': {'memory_strength': -0.005}
    },
    {
        'title': "RA Static Gain (+0.5%)",
        'func': 'apply_ra_gain_error',
        'param': {'relative_gain': 1.005}
    },
    {
        'title': "RA Static Gain (-0.5%)",
        'func': 'apply_ra_gain_error',
        'param': {'relative_gain': 0.995}
    },
    {
        'title': "RA Dynamic Gain (+0.5%)",
        'func': 'apply_ra_gain_error_dynamic',
        'param': {'coeff_3': 0.005}
    },
    {
        'title': "RA Dynamic Gain (-0.5%)",
               'func': 'apply_ra_gain_error_dynamic',
        'param': {'coeff_3': -0.005}
    },
]

# Prepare Figure
fig, axes = plt.subplots(2, 4, figsize=(24, 10))
axes = axes.flatten()

print("=" * 95)
print(f"{'Exp':<4} | {'Non-Ideality':<35} | {'SFDR (dB)':<10} | {'THD (dB)':<10}")
print("-" * 95)

# Run Experiments
for idx, exp in enumerate(EXPERIMENTS):

    # Apply single non-ideality
    if exp['func'] == 'apply_static_nonlinearity':
        signal = gen.apply_static_nonlinearity(input_signal=None, **exp['param'])

    elif exp['func'] in ['apply_ra_gain_error', 'apply_ra_gain_error_dynamic']:
        signal = getattr(gen, exp['func'])(
            input_signal=None,
            msb_bits=4,
            lsb_bits=12,
            **exp['param']
        )

    elif exp['func'] == 'apply_incomplete_sampling':
        signal = gen.apply_incomplete_sampling(
            input_signal=None,
            T_track=0.2 * (1 / Fs),
            **exp['param']
        )

    else:
        signal = getattr(gen, exp['func'])(input_signal=None, **exp['param'])

    # Add thermal noise
    signal = gen.apply_thermal_noise(signal, noise_rms=base_noise)

    # Spectrum analysis
    plt.sca(axes[idx])
    result = analyze_spectrum(signal, fs=Fs)

    axes[idx].set_title(exp['title'], fontsize=11, fontweight='bold')
    axes[idx].set_ylim([-140, 0])

    print(f"{idx+1:<4} | {exp['title']:<35} | "
          f"{result['sfdr_dbc']:<10.2f} | {result['thd_dbc']:<10.2f}")

# Finalize
plt.suptitle(
    f'Impact of Different Nonlinearities',
    fontsize=14,
    fontweight='bold',
    y=0.98
)

plt.tight_layout()
plt.subplots_adjust(top=0.90)

fig_path = output_dir / "exp_g06_sweep_dynamic_nonlinearity.png"
print(f"\n[Save figure] -> {fig_path}\n")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()
