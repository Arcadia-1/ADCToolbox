"""Generate DOUT analysis dashboard with 6 analysis plots in a 2x3 panel."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from adctoolbox.spectrum import analyze_spectrum
from adctoolbox.calibration import calibrate_weight_sine
from adctoolbox.dout import analyze_bit_activity
from adctoolbox.dout import analyze_overflow
from adctoolbox.dout import analyze_enob_sweep
from adctoolbox.dout import analyze_weight_radix

def generate_dout_dashboard(bits, freq=None, weights=None, output_path=None, show=False):
    """
    Generate comprehensive digital analysis dashboard with 6 subplots in a 2x3 panel.

    Parameters
    ----------
    bits : array_like
        Digital bits (N samples x B bits, MSB to LSB order)
    freq : float, optional
        Normalized frequency (0-0.5). If None, auto-detect from calibration
    weights : array_like, optional
        Nominal weights for bits (default: None, uses binary weights)
    output_path : str or Path, optional
        Path to save figure (default: None, don't save)
    show : bool, optional
        Whether to display figure (default: False)

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the dashboard
    axes : ndarray
        Array of axes objects (2x3 grid, flattened)
    """

    bits = np.asarray(bits)
    n_samples, n_bits = bits.shape

    # Use binary weights if not specified
    if weights is None:
        weights = 2.0 ** np.arange(n_bits - 1, -1, -1)
    else:
        weights = np.asarray(weights)

    # Create 2x3 panel
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    axes = axes.flatten()

    # Perform calibration once for all tools
    result = calibrate_weight_sine(bits, freq=freq)
    weights_calibrated = result['weight']
    freq_refined = result['refined_frequency']

    # Plot 1: Spectrum with nominal weights
    plt.sca(axes[0])
    signal_nominal = bits @ weights
    analyze_spectrum(signal_nominal, create_plot=True)
    axes[0].set_title('(1) Spectrum: Nominal Weights', fontsize=14, fontweight='bold')

    # Plot 2: Spectrum after calibration
    plt.sca(axes[1])
    signal_calibrated = result['calibrated_signal']
    analyze_spectrum(signal_calibrated, create_plot=True)
    axes[1].set_title('(2) Spectrum: Calibrated Weights', fontsize=14, fontweight='bold')

    # Plot 3: Bit Activity
    plt.sca(axes[2])
    analyze_bit_activity(bits, create_plot=True, ax=axes[2])
    axes[2].set_title('(3) Bit Activity', fontsize=14, fontweight='bold')

    # Plot 4: Overflow Check
    plt.sca(axes[3])
    analyze_overflow(bits, weights_calibrated, create_plot=True, ax=axes[3])
    axes[3].set_title('(4) Overflow Check', fontsize=14, fontweight='bold')

    # Plot 5: ENOB Sweep
    plt.sca(axes[4])
    analyze_enob_sweep(bits, freq=freq_refined, create_plot=True, ax=axes[4], verbose=False)
    axes[4].set_title('(5) ENOB Bit Sweep', fontsize=14, fontweight='bold')

    # Plot 6: Weight Radix
    plt.sca(axes[5])
    analyze_weight_radix(weights_calibrated, create_plot=True, ax=axes[5])
    axes[5].set_title('(6) Weight Radix', fontsize=14, fontweight='bold')

    # Overall title
    fig.suptitle('Comprehensive Digital ADC Analysis Dashboard (6 Tools)',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()

    # Save if requested
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[Dashboard saved] -> {output_path}")
        if not show:
            plt.close(fig)

    if show:
        plt.show()

    return fig, axes
