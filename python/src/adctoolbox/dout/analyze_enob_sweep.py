"""ENoB vs Number of Bits Used for Calibration."""

import numpy as np
import matplotlib.pyplot as plt
from .calibrate_weight_sine import calibrate_weight_sine
from ..spectrum import analyze_spectrum


def analyze_enob_sweep(bits, freq=0, order=5, osr=1, win_type='hamming', plot=True):
    """
    ENoB vs Number of Bits Used for Calibration.

    Parameters
    ----------
    bits : array_like
        Binary matrix (N samples x M bits, MSB to LSB)
    freq : float, optional
        Normalized frequency (0-0.5). If 0, auto-detect (default: 0)
    order : int, optional
        Polynomial order for FGCalSine (default: 5)
    osr : int, optional
        Oversampling ratio for specPlot (default: 1)
    win_type : str, optional
        Window type (default: 'hamming')
        Options: 'boxcar', 'hann', 'hamming'
    plot : bool, optional
        Create plot (default: True)

    Returns
    -------
    enob_sweep : ndarray
        ENoB for each number of bits (1D array of length M)
    n_bits_vec : ndarray
        Vector of bit counts (1 to M)

    Description
    -----------
    This function sweeps through different numbers of bits, calibrating
    with FGCalSine and measuring ENoB with specPlot.

    What to look for:
    - Increasing trend: More bits improve resolution
    - Plateaus: Additional bits not helping (noise/distortion limited)
    - Decreasing: Extra bits adding noise/errors
    """
    bits = np.asarray(bits)
    n_samples, m_bits = bits.shape

    # Auto-detect frequency if needed
    if freq == 0:
        _, _, _, _, _, freq = calibrate_weight_sine(bits, freq=0, order=order)

    enob_sweep = np.zeros(m_bits)
    n_bits_vec = np.arange(1, m_bits + 1)

    for n_bits in range(1, m_bits + 1):
        bits_subset = bits[:, :n_bits]

        try:
            weight_cal, _, post_cal_temp, _, _, _ = calibrate_weight_sine(bits_subset, freq=freq, order=order)

            result = analyze_spectrum(post_cal_temp, osr=osr, win_type=win_type, show_plot=False)
            enob_sweep[n_bits - 1] = result['enob']
        except Exception as e:
            enob_sweep[n_bits - 1] = np.nan
            print(f'FAILED for {n_bits} bits: {str(e)}')

    if plot:
        plt.plot(n_bits_vec, enob_sweep, 'o-k', linewidth=2, markersize=8, markerfacecolor='k')
        plt.grid(True)
        plt.xlabel('Number of Bits Used for Calibration')
        plt.ylabel('ENoB (bits)')
        plt.title('ENoB vs Number of Bits Used for Calibration')
        plt.xlim([0.5, m_bits + 0.5])
        plt.xticks(n_bits_vec)

        valid_enob = enob_sweep[~np.isnan(enob_sweep)]
        if len(valid_enob) > 0:
            plt.ylim([np.min(valid_enob) - 0.5, np.max(valid_enob) + 2])

        # Annotate with delta ENoB
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

                plt.text(n_bits_vec[i], enob_sweep[i] + y_offset, annotation_text,
                        ha='center', va='bottom', fontsize=10, fontweight='bold',
                        color=text_color)

    return enob_sweep, n_bits_vec
