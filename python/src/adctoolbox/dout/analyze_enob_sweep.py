"""ENOB sweep analysis versus number of calibration bits.

Sweeps through bit counts to evaluate how calibration quality improves with more bits.
"""

import numpy as np
import matplotlib.pyplot as plt
from adctoolbox.calibration import calibrate_weight_sine
from adctoolbox.spectrum import analyze_spectrum

def analyze_enob_sweep(
    bits: np.ndarray,
    freq: float | None = None,
    harmonic_order: int = 1,
    osr: int = 1,
    win_type: str = 'hamming',
    create_plot: bool = True,
    ax=None,
    title: str | None = None,
    verbose: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sweep ENOB vs number of bits used for calibration.

    Incrementally adds bits (MSB to LSB) and measures ENOB after calibration
    to understand diminishing returns and optimal bit count.

    Parameters
    ----------
    bits : np.ndarray
        Binary matrix (N samples x M bits, MSB to LSB order)
    freq : float, optional
        Normalized frequency (0-0.5). If None, auto-detect from data
    harmonic_order : int, default=1
        Harmonic order for calibrate_weight_sine
    osr : int, default=1
        Oversampling ratio for spectrum analysis
    win_type : str, default='hamming'
        Window function: 'boxcar', 'hann', 'hamming'
    create_plot : bool, default=True
        If True, plot ENOB sweep curve
    ax : plt.Axes, optional
        Axes to plot on. If None, uses current axes (plt.gca())
    title : str, optional
        Title for the plot. If None, uses default title
    verbose : bool, default=False
        If True, print progress messages

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - enob_sweep: ENOB for each bit count (length M)
        - n_bits_vec: Bit counts from 1 to M

    Notes
    -----
    What to look for in the plot:
    - Increasing trend: More bits improve resolution
    - Plateau: Additional bits don't help (noise/distortion limited)
    - Decrease: Extra bits add noise/calibration errors
    """
    bits = np.asarray(bits)
    n_samples, m_bits = bits.shape

    # Calibrate once with all bits to get all weights
    result = calibrate_weight_sine(bits, freq=freq, harmonic_order=harmonic_order)
    weights_all = result['weight']
    freq = result['refined_frequency']

    enob_sweep = np.zeros(m_bits)
    n_bits_vec = np.arange(1, m_bits + 1)

    # Sweep through bit counts using subsets of calibrated weights
    for n_bits in range(1, m_bits + 1):
        # Use only first n_bits and their corresponding weights
        bits_subset = bits[:, :n_bits]
        weights_subset = weights_all[:n_bits]

        # Compute calibrated signal with subset of weights
        calibrated_signal = bits_subset @ weights_subset

        spectrum_result = analyze_spectrum(
            calibrated_signal, osr=osr, win_type=win_type, create_plot=False
        )
        enob_sweep[n_bits - 1] = spectrum_result['enob']

        if verbose:
            print(f"[{n_bits:2d} bits] ENOB = {enob_sweep[n_bits - 1]:.2f}")

    # Plotting
    if create_plot:
        if ax is None:
            ax = plt.gca()

        ax.plot(n_bits_vec, enob_sweep, 'o-k', linewidth=2, markersize=8, markerfacecolor='k')
        ax.grid(True)
        ax.set_xlabel('Number of Bits Used for Calibration')
        ax.set_ylabel('ENOB (bits)')

        # Set title if provided
        if title is not None:
            ax.set_title(title)
        else:
            ax.set_title('ENOB vs Number of Bits Used for Calibration')

        ax.set_xlim([0.5, m_bits + 0.5])
        ax.set_xticks(n_bits_vec)

        valid_enob = enob_sweep[~np.isnan(enob_sweep)]
        if len(valid_enob) > 0:
            ax.set_ylim([np.min(valid_enob) - 0.5, np.max(valid_enob) + 2])

        # Annotate with delta ENOB
        delta_enob = np.concatenate([[enob_sweep[0]], np.diff(enob_sweep)])

        if len(valid_enob) > 0:
            y_offset = (np.max(valid_enob) - np.min(valid_enob)) * 0.06
        else:
            y_offset = 0.1

        for i in range(m_bits):
            if not np.isnan(enob_sweep[i]) and not np.isnan(delta_enob[i]):
                if i == 0:
                    annotation_text = f'{delta_enob[i]:.2f}'
                    text_color = [0, 0, 0]
                else:
                    annotation_text = f'+{delta_enob[i]:.2f}'
                    normalized_delta = max(0, min(1, delta_enob[i]))
                    text_color = [1 - normalized_delta, 0, 0]

                ax.text(n_bits_vec[i], enob_sweep[i] + y_offset, annotation_text,
                        ha='center', va='bottom', fontsize=10, fontweight='bold',
                        color=text_color)

    return enob_sweep, n_bits_vec
