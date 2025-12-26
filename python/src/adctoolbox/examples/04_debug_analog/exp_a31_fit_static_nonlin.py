"""Extract static nonlinearity coefficients k2 and k3 from distorted sinewave"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, fit_static_nonlin, amplitudes_to_snr, snr_to_nsd

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N = 2**13
Fs = 800e6
Fin, Fin_bin = find_coherent_frequency(fs=Fs, fin_target=70e6, n_fft=N)
A = 0.5
base_noise = 500e-6

snr_ref = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=base_noise)
nsd_ref = snr_to_nsd(snr_ref, fs=Fs, osr=1)

sig_ideal = A * np.sin(2 * np.pi * Fin * np.arange(N) / Fs)
print(f"[Sinewave] Fs=[{Fs/1e6:.2f} MHz], Fin=[{Fin/1e6:.2f} MHz], Bin/N=[{Fin_bin}/{N}], A=[{A:.3f} Vpeak]")
print(f"[Nonideal] Noise RMS=[{base_noise*1e6:.2f} uVrms], Theoretical SNR=[{snr_ref:.2f} dB], Theoretical NSD=[{nsd_ref:.2f} dBFS/Hz]")

# Scenarios: (k2_inject, k3_inject)
scenarios = [
    (0.00, 0.00),
    (0.01, 0.00),
    (0.00, 0.01),
    (0.01, 0.01),
]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for idx, (k2_inject, k3_inject) in enumerate(scenarios):
    sig_distorted = sig_ideal + k2_inject * sig_ideal**2 + k3_inject * sig_ideal**3 + np.random.randn(N) * base_noise
    
    k2_extracted, k3_extracted, fitted_sine, fitted_transfer = fit_static_nonlin(sig_distorted, order=3)

    residual = sig_distorted - fitted_sine
    transfer_x, transfer_y = fitted_transfer
    nonlinearity_curve = transfer_y - transfer_x

    # Row 0: Transfer function (data and fitted line with y=x reference)
    ax_top = axes[0, idx]
    ax_top.plot(fitted_sine, sig_distorted, 'b.', ms=1, alpha=0.5, label='Measured')
    lim = max(abs(fitted_sine).max(), abs(sig_distorted).max())
    ax_top.plot([-lim, lim], [-lim, lim], 'k--', lw=1, alpha=0.5, label='Ideal (y=x)')
    ax_top.plot(transfer_x, transfer_y, 'r-', lw=2, label='Fitted')
    ax_top.set_title(f"Transfer Curve\nInjected: k2={k2_inject:.4f}, k3={k3_inject:.4f}\nExtracted: k2={k2_extracted:.4f}, k3={k3_extracted:.4f}", fontsize=11, fontweight='bold')
    ax_top.set_xlabel('Input Amplitude (V)', fontsize=10)
    if idx == 0:
        ax_top.set_ylabel('Output Amplitude (V)', fontsize=10)
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc='upper left', fontsize=9)
    ax_top.set_aspect('equal', adjustable='box')

    # Row 1: Residual plot
    ax_bottom = axes[1, idx]
    ax_bottom.plot(fitted_sine, residual, 'b.', ms=1, alpha=0.5, label='Measured')
    ax_bottom.plot(transfer_x, nonlinearity_curve, 'r-', lw=2, label='Fitted Model')
    ax_bottom.set_title("Residue Error", fontsize=11, fontweight='bold')
    ax_bottom.set_xlabel('Input Amplitude (V)', fontsize=10)
    if idx == 0:
        ax_bottom.set_ylabel('Nonlinearity Error (V)', fontsize=10)
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.legend(loc='upper left', fontsize=9)

    print(f"[Injected: k2={k2_inject:7.4f}, k3={k3_inject:7.4f}] [Extracted: k2={k2_extracted:7.4f}, k3={k3_extracted:7.4f}]")

plt.tight_layout()
fig_path = output_dir / 'exp_a31_fit_static_nonlin.png'
plt.savefig(fig_path, dpi=150)
print(f"\n[Saved] -> {fig_path}")
plt.close()
