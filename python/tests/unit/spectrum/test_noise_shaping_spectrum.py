"""Unit test for 1st order noise shaping sinewave spectrum."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import analyze_spectrum, find_coherent_frequency


# Create output directory for test figures
output_dir = Path(__file__).parent / "test_output"
output_dir.mkdir(exist_ok=True)


def test_first_order_noise_shaping_spectrum():
    """Test spectrum of 1st and 2nd order noise shaping sinewave.

    1st order noise shaping: NTF(z) = 1 - z^-1
    2nd order noise shaping: NTF(z) = (1 - z^-1)^2 = 1 - 2z^-1 + z^-2

    Both push noise to higher frequencies, with 2nd order providing steeper roll-off.
    """
    # Signal parameters
    N = 2**16
    Fs = 100e6
    OSR = 64  # Oversampling ratio
    Fin_target = Fs / (2 * OSR) / 14  # Signal in lower quarter of Nyquist band
    A = 0.49

    # Find coherent frequency
    Fin, Fin_bin = find_coherent_frequency(fs=Fs, fin_target=Fin_target, n_fft=N)

    # Generate ideal sinewave
    t = np.arange(N) / Fs
    signal_ideal = A * np.sin(2 * np.pi * Fin * t)

    # Thermal noise (white, cannot be shaped) - keep small
    thermal_noise = np.random.randn(N) * 100e-6

    # Quantization noise (white noise as approximation, can be shaped) - make dominant
    quant_noise_white = np.random.randn(N) * 2000e-6

    # Apply 1st order noise shaping to quantization noise: NTF(z) = 1 - z^-1
    # noise_shaped[n] = noise[n] - noise[n-1]
    quant_noise_shaped_1st = np.zeros(N)
    quant_noise_shaped_1st[0] = quant_noise_white[0]
    for n in range(1, N):
        quant_noise_shaped_1st[n] = quant_noise_white[n] - quant_noise_white[n-1]

    # Apply 2nd order noise shaping to quantization noise: NTF(z) = (1 - z^-1)^2
    # noise_shaped[n] = noise[n] - 2*noise[n-1] + noise[n-2]
    quant_noise_shaped_2nd = np.zeros(N)
    quant_noise_shaped_2nd[0] = quant_noise_white[0]
    quant_noise_shaped_2nd[1] = quant_noise_white[1] - quant_noise_white[0]
    for n in range(2, N):
        quant_noise_shaped_2nd[n] = quant_noise_white[n] - 2*quant_noise_white[n-1] + quant_noise_white[n-2]

    # Combine signal with thermal noise (always present) + quantization noise (shaped or unshaped)
    signal_without_shaping = signal_ideal + thermal_noise + quant_noise_white
    signal_with_1st_order = signal_ideal + thermal_noise + quant_noise_shaped_1st
    signal_with_2nd_order = signal_ideal + thermal_noise + quant_noise_shaped_2nd

    # Generate 16 runs for spectrum averaging
    N_runs = 16
    signal_matrix_no_shaping = np.zeros((N_runs, N))
    signal_matrix_1st_order = np.zeros((N_runs, N))
    signal_matrix_2nd_order = np.zeros((N_runs, N))

    for run in range(N_runs):
        # Generate new noise for each run
        thermal_run = np.random.randn(N) * 100e-6
        quant_run = np.random.randn(N) * 2000e-6

        # Apply noise shaping to quantization noise
        quant_shaped_1st_run = np.zeros(N)
        quant_shaped_1st_run[0] = quant_run[0]
        for n in range(1, N):
            quant_shaped_1st_run[n] = quant_run[n] - quant_run[n-1]

        quant_shaped_2nd_run = np.zeros(N)
        quant_shaped_2nd_run[0] = quant_run[0]
        quant_shaped_2nd_run[1] = quant_run[1] - quant_run[0]
        for n in range(2, N):
            quant_shaped_2nd_run[n] = quant_run[n] - 2*quant_run[n-1] + quant_run[n-2]

        # Combine signal with noise
        signal_matrix_no_shaping[run, :] = signal_ideal + thermal_run + quant_run
        signal_matrix_1st_order[run, :] = signal_ideal + thermal_run + quant_shaped_1st_run
        signal_matrix_2nd_order[run, :] = signal_ideal + thermal_run + quant_shaped_2nd_run

    # Create comparison plot: 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    # Row 1: Single run
    # Left: Without noise shaping (white noise)
    plt.sca(axes[0, 0])
    result_white = analyze_spectrum(signal_without_shaping, fs=Fs, osr=OSR)
    axes[0, 0].set_title(f'Without Noise Shaping (Single Run)',
                     fontsize=12, fontweight='bold')

    # Middle: With 1st order noise shaping
    plt.sca(axes[0, 1])
    result_1st = analyze_spectrum(signal_with_1st_order, fs=Fs, osr=OSR)
    axes[0, 1].set_title(f'1st Order Noise Shaping (Single Run)',
                     fontsize=12, fontweight='bold')

    # Right: With 2nd order noise shaping
    plt.sca(axes[0, 2])
    result_2nd = analyze_spectrum(signal_with_2nd_order, fs=Fs, osr=OSR)
    axes[0, 2].set_title(f'2nd Order Noise Shaping (Single Run)',
                     fontsize=12, fontweight='bold')

    # Row 2: 16-run averaging
    # Left: Without noise shaping (averaged)
    plt.sca(axes[1, 0])
    result_white_avg = analyze_spectrum(signal_matrix_no_shaping, fs=Fs, osr=OSR)
    axes[1, 0].set_title(f'Without Noise Shaping ({N_runs}-run Averaging)',
                     fontsize=12, fontweight='bold')

    # Middle: With 1st order noise shaping (averaged)
    plt.sca(axes[1, 1])
    result_1st_avg = analyze_spectrum(signal_matrix_1st_order, fs=Fs, osr=OSR)
    axes[1, 1].set_title(f'1st Order Noise Shaping ({N_runs}-run Averaging)',
                     fontsize=12, fontweight='bold')

    # Right: With 2nd order noise shaping (averaged)
    plt.sca(axes[1, 2])
    result_2nd_avg = analyze_spectrum(signal_matrix_2nd_order, fs=Fs, osr=OSR)
    axes[1, 2].set_title(f'2nd Order Noise Shaping ({N_runs}-run Averaging)',
                     fontsize=12, fontweight='bold')

    # Save figure
    plt.tight_layout()
    fig_path = output_dir / 'test_noise_shaping_1st_2nd_order.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Verify file was created
    assert fig_path.exists(), f"Figure file not created: {fig_path}"
    assert fig_path.stat().st_size > 0, f"Figure file is empty: {fig_path}"

    # Calculate improvements
    snr_improvement_1st = result_1st['snr_dbc'] - result_white['snr_dbc']
    snr_improvement_2nd = result_2nd['snr_dbc'] - result_white['snr_dbc']
    snr_improvement_1st_avg = result_1st_avg['snr_dbc'] - result_white_avg['snr_dbc']
    snr_improvement_2nd_avg = result_2nd_avg['snr_dbc'] - result_white_avg['snr_dbc']

    print(f'\n[Single Run]')
    print(f'  White Noise SNR: {result_white["snr_dbc"]:.2f} dB')
    print(f'  1st Order SNR: {result_1st["snr_dbc"]:.2f} dB (Improvement: {snr_improvement_1st:.2f} dB)')
    print(f'  2nd Order SNR: {result_2nd["snr_dbc"]:.2f} dB (Improvement: {snr_improvement_2nd:.2f} dB)')
    print(f'\n[{N_runs}-run Averaging]')
    print(f'  White Noise SNR: {result_white_avg["snr_dbc"]:.2f} dB')
    print(f'  1st Order SNR: {result_1st_avg["snr_dbc"]:.2f} dB (Improvement: {snr_improvement_1st_avg:.2f} dB)')
    print(f'  2nd Order SNR: {result_2nd_avg["snr_dbc"]:.2f} dB (Improvement: {snr_improvement_2nd_avg:.2f} dB)')
    print(f'[Figure saved] -> {fig_path}')

    # Assert SNR improvements for single run
    # For OSR=32, expect good improvement
    assert snr_improvement_1st > 5.0, f"1st order SNR improvement too low: {snr_improvement_1st:.2f} dB (expected > 5 dB)"
    assert snr_improvement_2nd > snr_improvement_1st, f"2nd order should improve more than 1st order"
    assert snr_improvement_2nd > 10.0, f"2nd order SNR improvement too low: {snr_improvement_2nd:.2f} dB (expected > 10 dB)"

    # Assert SNR improvements for averaged (should be similar or better)
    assert snr_improvement_1st_avg > 5.0, f"1st order averaged SNR improvement too low: {snr_improvement_1st_avg:.2f} dB"
    assert snr_improvement_2nd_avg > snr_improvement_1st_avg, f"2nd order averaged should improve more than 1st order"
