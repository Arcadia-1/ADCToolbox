"""INL/DNL sweep with different record lengths"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, analyze_spectrum, analyze_inl_from_sine

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# Parameters
n_bits = 16
full_scale = 1.0
fs = 800e6
fin_target = 80e6

# Nonidealities (same as exp_a01)
A = 0.49
DC = 0.5
base_noise = 50e-6
hd2_dB, hd3_dB = -80, -66

# Compute HD coefficients
hd2_amp = 10**(hd2_dB/20)
hd3_amp = 10**(hd3_dB/20)
k2 = hd2_amp / (A / 2)
k3 = hd3_amp / (A**2 / 4)

N_list = [2**i for i in range(10, 24, 4)]  
n_plots = len(N_list)
fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 6))

print(f"[INL/DNL Sweep] [Fs = {fs/1e6:.0f} MHz, Fin = {fin_target/1e6:.0f} MHz]")
print(f"  [HD2 = {hd2_dB} dB, HD3 = {hd3_dB} dB, Noise = {base_noise*1e6:.1f} uV]\n")

for idx, N in enumerate(N_list):

    fin, J = find_coherent_frequency(fs, fin_target, N)
    t = np.arange(N) / fs
    sinewave = A * np.sin(2 * np.pi * fin * t)
    signal_distorted = sinewave + k2 * sinewave**2 + k3 * sinewave**3 + DC + np.random.randn(N) * base_noise

    result = analyze_spectrum(signal_distorted, fs=fs, create_plot=False)

    # Analyze INL/DNL and plot (quantization handled internally)
    plt.sca(axes[idx])
    result_inl = analyze_inl_from_sine(
        signal_distorted,
        num_bits=n_bits,
        full_scale=full_scale,
        clip_percent=0.01,
        col_title=f'N = 2^{int(np.log2(N))}'
    )
    inl, dnl, code = result_inl['inl'], result_inl['dnl'], result_inl['code']

    print(f"  [N = 2^{int(np.log2(N)):2d} = {N:5d}] [ENOB = {result['enob']:5.2f}] [INL: {np.min(inl):5.2f} to {np.max(inl):5.2f}] [DNL: {np.min(dnl):5.2f} to {np.max(dnl):5.2f}] LSB")
    
fig.suptitle(f'INL/DNL Sweep: Record Length Comparison (Fs={fs/1e6:.0f} MHz, Fin={fin_target/1e6:.0f} MHz)',
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig_path = output_dir / 'exp_a33_compute_inl_sweep_length.png'
fig.savefig(fig_path, dpi=150)
print(f"\n[Save fig] -> [{fig_path}]")
plt.close(fig)
