"""Extract static nonlinearity coefficients k2 and k3 from distorted sinewave"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, fit_static_nonlin

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N = 2**13
Fs = 800e6
Fin, Fin_bin = find_coherent_frequency(fs=Fs, fin_target=70e6, n_fft=N)
A = 0.5
base_noise = 500e-6

sig_ideal = A * np.sin(2 * np.pi * Fin * np.arange(N) / Fs)
print(f"[Nonlinearity Extraction] Fs={Fs/1e6:.1f} MHz, Fin={Fin/1e6:.6f} MHz, Bin={Fin_bin}, N_fft={N}")


# Scenarios: (k2, k3, title)
scenarios = [
    (0.00, 0.00, 'Ideal (No Distortion)'),
    (0.01, 0.00, '2nd Order Only (k2=1%)'),
    (0.00, 0.01, '3rd Order Only (k3=1%)'),
    (0.01, 0.01, 'Mixed (k2=1%, k3=1%)'),
]

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes = axes.flatten()

for idx, (k2_inject, k3_inject, title) in enumerate(scenarios):

    # Generate distorted signal: y = x + k2*x^2 + k3*x^3 + noise
    sig_distorted = sig_ideal + k2_inject * sig_ideal**2 + k3_inject * sig_ideal**3 + np.random.randn(N) * base_noise

    # Extract nonlinearity coefficients
    k2_extracted, k3_extracted, fitted_sine, fitted_transfer = fit_static_nonlin(sig_distorted, order=3)

    # Prepare plotting data
    # 1. Measured residual: deviation from the fundamental sine wave
    residual = sig_distorted - fitted_sine

    # 2. Fitted curve: use the smooth transfer curve directly
    transfer_x, transfer_y = fitted_transfer
    nonlinearity_curve = transfer_y - transfer_x

    # Plot
    ax = axes[idx]
    ax.plot(fitted_sine, residual, 'b.', ms=1, alpha=0.5, label='Measured')
    ax.plot(transfer_x, nonlinearity_curve, 'r-', lw=2, label='Fitted Model')

    ax.set_title(f"{title}\nExtracted: k2={k2_extracted:.4f}, k3={k3_extracted:.4f}",
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Input Amplitude (V)', fontsize=10)
    if idx == 0:
        ax.set_ylabel('Nonlinearity Error (V)', fontsize=10)

    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)

    print(f"[{title:24s}] Injected: k2={k2_inject:7.4f}, k3={k3_inject:7.4f}  |  Extracted: k2={k2_extracted:7.4f}, k3={k3_extracted:7.4f}")

plt.tight_layout()
fig_path = output_dir / 'exp_a03_fit_static_nonlin.png'
plt.savefig(fig_path, dpi=150)
print(f"\n[Saved] -> {fig_path}")
plt.close()
