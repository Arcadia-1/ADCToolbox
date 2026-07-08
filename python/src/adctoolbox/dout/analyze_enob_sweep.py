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
    calibration_mode: str = "recalibrate_each_subset",
    frequency_policy: str = "python",
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
    calibration_mode : {'recalibrate_each_subset',
        'prefix_of_full_calibration'}, default='recalibrate_each_subset'
        ENOB sweep calibration policy. ``'recalibrate_each_subset'`` matches
        MATLAB ``bitsweep``: estimate the frequency once when needed, then
        recalibrate each bit-prefix subset independently.
        ``'prefix_of_full_calibration'`` preserves the historical Python
        behavior: calibrate all bits once and sweep prefixes of the
        full-weight solution.
    frequency_policy : {'python', 'matlab'}, default='python'
        Coarse frequency estimator passed to ``calibrate_weight_sine`` when
        automatic frequency search is requested.
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
    The default ``'recalibrate_each_subset'`` mode answers the usual bit-depth
    sweep question: "if only the first n bits are available, how well can that
    n-bit subsystem be calibrated?"  Use ``'prefix_of_full_calibration'`` for
    prefix-ablation diagnostics: "after a full-bit calibration, how much
    performance remains if lower-bit terms are removed?"

    What to look for in the plot:
    - Increasing trend: More bits improve resolution
    - Plateau: Additional bits don't help (noise/distortion limited)
    - Decrease: Extra bits add noise/calibration errors
    """
    bits = np.asarray(bits)
    _, m_bits = bits.shape

    valid_modes = {"recalibrate_each_subset", "prefix_of_full_calibration"}
    if calibration_mode not in valid_modes:
        raise ValueError(
            f"Unknown calibration_mode {calibration_mode!r}. "
            f"Expected one of {sorted(valid_modes)}."
        )

    enob_sweep = np.zeros(m_bits)
    n_bits_vec = np.arange(1, m_bits + 1)

    if calibration_mode == "recalibrate_each_subset":
        enob_sweep = _sweep_recalibrating_each_subset(
            bits=bits,
            freq=freq,
            harmonic_order=harmonic_order,
            osr=osr,
            win_type=win_type,
            frequency_policy=frequency_policy,
            verbose=verbose,
        )
    else:
        enob_sweep = _sweep_prefix_of_full_calibration(
            bits=bits,
            freq=freq,
            harmonic_order=harmonic_order,
            osr=osr,
            win_type=win_type,
            frequency_policy=frequency_policy,
            verbose=verbose,
        )

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


def _sweep_recalibrating_each_subset(
    bits: np.ndarray,
    freq: float | None,
    harmonic_order: int,
    osr: int,
    win_type: str,
    frequency_policy: str,
    verbose: bool,
) -> np.ndarray:
    """MATLAB bitsweep-compatible mode: recalibrate every bit-prefix subset."""
    _, m_bits = bits.shape
    enob_sweep = np.zeros(m_bits)

    auto_frequency_requested = freq is None or np.all(np.asarray(freq) == 0)
    if auto_frequency_requested:
        full_result = calibrate_weight_sine(
            bits,
            freq=freq,
            harmonic_order=harmonic_order,
            frequency_policy=frequency_policy,
        )
        subset_freq = full_result["refined_frequency"]
    else:
        subset_freq = freq

    for n_bits in range(1, m_bits + 1):
        bits_subset = bits[:, :n_bits]

        try:
            result = calibrate_weight_sine(
                bits_subset,
                freq=subset_freq,
                force_search=False,
                harmonic_order=harmonic_order,
                frequency_policy=frequency_policy,
            )
            spectrum_result = analyze_spectrum(
                result["calibrated_signal"],
                osr=osr,
                win_type=win_type,
                create_plot=False,
            )
            enob_sweep[n_bits - 1] = spectrum_result["enob"]
        except Exception as exc:
            enob_sweep[n_bits - 1] = np.nan
            if verbose:
                print(f"[{n_bits:2d} bits] FAILED: {exc}")
            continue

        if verbose:
            print(f"[{n_bits:2d} bits] ENOB = {enob_sweep[n_bits - 1]:.2f}")

    return enob_sweep


def _sweep_prefix_of_full_calibration(
    bits: np.ndarray,
    freq: float | None,
    harmonic_order: int,
    osr: int,
    win_type: str,
    frequency_policy: str,
    verbose: bool,
) -> np.ndarray:
    """Historical Python mode: sweep prefixes of one full-bit calibration."""
    _, m_bits = bits.shape
    enob_sweep = np.zeros(m_bits)

    result = calibrate_weight_sine(
        bits,
        freq=freq,
        harmonic_order=harmonic_order,
        frequency_policy=frequency_policy,
    )
    weights_all = result["weight"]

    for n_bits in range(1, m_bits + 1):
        bits_subset = bits[:, :n_bits]
        weights_subset = weights_all[:n_bits]
        calibrated_signal = bits_subset @ weights_subset

        try:
            spectrum_result = analyze_spectrum(
                calibrated_signal,
                osr=osr,
                win_type=win_type,
                create_plot=False,
            )
            enob_sweep[n_bits - 1] = spectrum_result["enob"]
        except Exception as exc:
            enob_sweep[n_bits - 1] = np.nan
            if verbose:
                print(f"[{n_bits:2d} bits] FAILED: {exc}")
            continue

        if verbose:
            print(f"[{n_bits:2d} bits] ENOB = {enob_sweep[n_bits - 1]:.2f}")

    return enob_sweep
