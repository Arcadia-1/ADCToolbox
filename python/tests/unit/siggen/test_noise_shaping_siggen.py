"""Unit test for ADC_Signal_Generator noise shaping methods (0th to 4th order)."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import analyze_spectrum, find_coherent_frequency
from adctoolbox.siggen.nonidealities import ADC_Signal_Generator


# Create output directory for test figures
output_dir = Path(__file__).parent / "test_output"
output_dir.mkdir(exist_ok=True)


def _add_slope_indicator(ax, slope_db_per_decade, order):
    """Add a slope indicator triangle to the spectrum plot.

    Args:
        ax: matplotlib axes object
        slope_db_per_decade: Slope in dB/decade (20, 40, 60, 80, or 100)
        order: Noise shaping order (1, 2, 3, 4, or 5)
    """
    # Get current axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Position the triangle in the right side where the slope is visible
    # Start at 85% across the frequency axis (in log scale)
    log_x_start = xlim[0] + 0.5 * (xlim[1] - xlim[0])

    # Ensure x_start is within reasonable bounds
    if log_x_start > 8:  # Cap at 10^8 Hz to avoid overflow
        log_x_start = 8

    x_start = 10**log_x_start

    # Triangle width: one decade (but adjusted if near edge)
    log_x_end = min(log_x_start + 1, xlim[1])
    x_end = 10**log_x_end

    # y position: place in the middle-upper area to avoid interfering with noise floor
    y_range = ylim[1] - ylim[0]
    y_start = ylim[0] + 0.55 * y_range  # Start at 55% from bottom

    # Calculate y_end based on the slope (in dB per decade)
    # Scale the vertical height to fit nicely in the plot
    decade_span = (log_x_end - log_x_start)
    y_end = y_start + slope_db_per_decade * decade_span

    # Ensure y_end doesn't go outside plot bounds
    if y_end > ylim[1]:
        y_end = ylim[1] - 0.05 * y_range
        # Recalculate y_start to maintain correct slope
        y_start = y_end - slope_db_per_decade * decade_span

    # Draw the triangle (right angle triangle)
    ax.plot([x_start, x_end], [y_start, y_start], 'r-', linewidth=2.5, zorder=100)  # Horizontal line
    ax.plot([x_end, x_end], [y_start, y_end], 'r-', linewidth=2.5, zorder=100)      # Vertical line
    ax.plot([x_start, x_end], [y_start, y_end], 'r--', linewidth=2, zorder=100)     # Hypotenuse (dashed)
    print(f'Slope indicator located at x={x_start:.2e} Hz to x={x_end:.2e} Hz, y={y_start:.2f} dB to y={y_end:.2f} dB')

    # Add text label
    # label = f'{slope_db_per_decade} dB/dec'
    # # Position text to the left of the triangle
    # text_x = x_start * 0.5  # Left of triangle
    # text_y = y_start + (y_end - y_start) / 2
    # ax.text(text_x, text_y, label, fontsize=10, color='red',
    #         fontweight='bold', ha='center', va='center',
    #         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='red', alpha=0.9, linewidth=1.5),
    #         zorder=101)


def test_noise_shaping_0th_to_4th_order_spectrum():
    """Test spectrum of 0th (no shaping) to 5th order noise shaping.

    Order 0: No shaping (regular quantization)
    Order 1: NTF(z) = 1 - z^-1 (20 dB/decade)
    Order 2: NTF(z) = (1 - z^-1)^2 (40 dB/decade)
    Order 3: NTF(z) = (1 - z^-1)^3 (60 dB/decade)
    Order 4: NTF(z) = (1 - z^-1)^4 (80 dB/decade)
    Order 5: NTF(z) = (1 - z^-1)^5 (100 dB/decade)

    All push noise to higher frequencies, with higher orders providing steeper roll-off.
    Sweeps n_bits = 1, 2, 4, 6, 8, 10 and generates one figure per n_bits value.
    """
    # Signal parameters
    N = 2**13  # Reduced from 2**16 for faster computation
    Fs = 100e6
    OSR = 32  # Oversampling ratio
    Fin_target = Fs / (2 * OSR) / 14  # Signal in lower quarter of Nyquist band
    A = 0.4
    DC = 0.0  # Centered signal

    # Quantization parameters
    quant_range = (-0.5, 0.5)  # Bipolar range centered at 0

    # Noise parameters
    thermal_noise_rms = 100e-6  # 100 uV base noise

    # Find coherent frequency
    Fin, Fin_bin = find_coherent_frequency(fs=Fs, fin_target=Fin_target, n_fft=N)

    # Initialize signal generator
    gen = ADC_Signal_Generator(N=N, Fs=Fs, Fin=Fin, A=A, DC=DC)

    # Sweep n_bits: 1, 2, 4, 6, 8, 10 (6 values)
    n_bits_values = [10]
    for n_bits in n_bits_values:

        # Generate 16 runs for spectrum averaging
        N_runs = 16
        signal_matrix_order_0 = np.zeros((N_runs, N))
        signal_matrix_order_1 = np.zeros((N_runs, N))
        signal_matrix_order_2 = np.zeros((N_runs, N))
        signal_matrix_order_3 = np.zeros((N_runs, N))
        signal_matrix_order_4 = np.zeros((N_runs, N))
        signal_matrix_order_5 = np.zeros((N_runs, N))

        for run in range(N_runs):
            # Order 0: No noise shaping (just quantization + thermal noise)
            signal_order_0 = gen.apply_thermal_noise(noise_rms=thermal_noise_rms)
            signal_order_0 = gen.apply_quantization_noise(signal_order_0, n_bits=n_bits, quant_range=quant_range)
            signal_matrix_order_0[run, :] = signal_order_0

            # Order 1-5: Apply noise shaping
            signal_order_1 = gen.apply_thermal_noise(noise_rms=thermal_noise_rms)
            signal_order_1 = gen.apply_noise_shaping(signal_order_1, n_bits=n_bits, quant_range=quant_range, order=1)
            signal_matrix_order_1[run, :] = signal_order_1

            signal_order_2 = gen.apply_thermal_noise(noise_rms=thermal_noise_rms)
            signal_order_2 = gen.apply_noise_shaping(signal_order_2, n_bits=n_bits, quant_range=quant_range, order=2)
            signal_matrix_order_2[run, :] = signal_order_2

            signal_order_3 = gen.apply_thermal_noise(noise_rms=thermal_noise_rms)
            signal_order_3 = gen.apply_noise_shaping(signal_order_3, n_bits=n_bits, quant_range=quant_range, order=3)
            signal_matrix_order_3[run, :] = signal_order_3

            signal_order_4 = gen.apply_thermal_noise(noise_rms=thermal_noise_rms)
            signal_order_4 = gen.apply_noise_shaping(signal_order_4, n_bits=n_bits, quant_range=quant_range, order=4)
            signal_matrix_order_4[run, :] = signal_order_4

            signal_order_5 = gen.apply_thermal_noise(noise_rms=thermal_noise_rms)
            signal_order_5 = gen.apply_noise_shaping(signal_order_5, n_bits=n_bits, quant_range=quant_range, order=5)
            signal_matrix_order_5[run, :] = signal_order_5

        # Create comparison plot: 2x3 layout (6 plots for orders 0-5)
        fig, axes = plt.subplots(2, 3, figsize=(21, 10))

        # Analyze and plot each order
        results = {}

        # Order 0 (No shaping)
        plt.sca(axes[0, 0])
        result_0 = analyze_spectrum(signal_matrix_order_0, fs=Fs, osr=OSR)
        results[0] = result_0
        axes[0, 0].set_title(f'Order 0: No Shaping ({N_runs}-run avg)',
                             fontsize=12, fontweight='bold')
        # No slope indicator for order 0 (flat noise floor)

        # Order 1
        plt.sca(axes[0, 1])
        result_1 = analyze_spectrum(signal_matrix_order_1, fs=Fs, osr=OSR)
        results[1] = result_1
        axes[0, 1].set_title(f'Order 1: NTF(z) = 1 - z^-1 ({N_runs}-run avg)',
                             fontsize=12, fontweight='bold')
        _add_slope_indicator(axes[0, 1], slope_db_per_decade=20, order=1)

        # Order 2
        plt.sca(axes[0, 2])
        result_2 = analyze_spectrum(signal_matrix_order_2, fs=Fs, osr=OSR)
        results[2] = result_2
        axes[0, 2].set_title(f'Order 2: NTF(z) = (1 - z^-1)^2 ({N_runs}-run avg)',
                             fontsize=12, fontweight='bold')
        _add_slope_indicator(axes[0, 2], slope_db_per_decade=40, order=2)

        # Order 3
        plt.sca(axes[1, 0])
        result_3 = analyze_spectrum(signal_matrix_order_3, fs=Fs, osr=OSR)
        results[3] = result_3
        axes[1, 0].set_title(f'Order 3: NTF(z) = (1 - z^-1)^3 ({N_runs}-run avg)',
                             fontsize=12, fontweight='bold')
        _add_slope_indicator(axes[1, 0], slope_db_per_decade=60, order=3)

        # Order 4
        plt.sca(axes[1, 1])
        result_4 = analyze_spectrum(signal_matrix_order_4, fs=Fs, osr=OSR)
        results[4] = result_4
        axes[1, 1].set_title(f'Order 4: NTF(z) = (1 - z^-1)^4 ({N_runs}-run avg)',
                             fontsize=12, fontweight='bold')
        _add_slope_indicator(axes[1, 1], slope_db_per_decade=80, order=4)

        # Order 5
        plt.sca(axes[1, 2])
        result_5 = analyze_spectrum(signal_matrix_order_5, fs=Fs, osr=OSR)
        results[5] = result_5
        axes[1, 2].set_title(f'Order 5: NTF(z) = (1 - z^-1)^5 ({N_runs}-run avg)',
                             fontsize=12, fontweight='bold')
        _add_slope_indicator(axes[1, 2], slope_db_per_decade=100, order=5)

        # Add main title with n_bits information
        fig.suptitle(f'Noise Shaping Performance: {n_bits}-bit Quantizer (OSR={OSR})',
                     fontsize=16, fontweight='bold', y=0.995)

        # Save figure
        plt.tight_layout(rect=[0, 0, 1, 0.99])  # Leave space for suptitle
        fig_path = output_dir / f'test_noise_shaping_0th_to_5th_order_{n_bits}bit.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Verify file was created
        assert fig_path.exists(), f"Figure file not created: {fig_path}"
        assert fig_path.stat().st_size > 0, f"Figure file is empty: {fig_path}"

        # Print results
        print(f'\nNoise Shaping Performance (n_bits={n_bits}, OSR={OSR}):')
        print('-' * 60)
        for order in [0, 1, 2, 3, 4, 5]:
            snr = results[order]['snr_dbc']
            if order == 0:
                print(f'  Order {order} (No Shaping): {snr:6.2f} dB')
            else:
                improvement = snr - results[0]['snr_dbc']
                print(f'  Order {order}: {snr:6.2f} dB (Improvement: +{improvement:5.2f} dB)')

        print(f'\n[Figure saved] -> {fig_path}')

if __name__ == "__main__":
    test_noise_shaping_0th_to_4th_order_spectrum()
