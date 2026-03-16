"""Performance metrics vs oversampling ratio (OSR) sweep.

Analyzes how ADC performance (SNDR, SFDR, ENOB) varies with oversampling ratio
by separating ideal signal from error via sine fitting.
"""

import numpy as np
import matplotlib.pyplot as plt

from adctoolbox.fundamentals import fit_sine_4param, fold_frequency_to_nyquist


def sweep_performance_vs_osr(
    data: np.ndarray,
    osr: np.ndarray | None = None,
    harmonic: int = 5,
    create_plot: bool = True,
    ax: plt.Axes | None = None,
) -> dict:
    """
    Sweep ADC performance metrics versus oversampling ratio.

    Parameters
    ----------
    data : np.ndarray
        Input signal (1D), typically ADC output samples.
    osr : np.ndarray, optional
        OSR values to evaluate. Default: N/2 / (N/2, N/2-1, ..., 1).
    harmonic : int, default=5
        Number of harmonics to mark on plot.
    create_plot : bool, default=True
        If True, create performance plot(s).
    ax : plt.Axes, optional
        Axes for main performance plot. If None and create_plot, creates
        2-subplot figure (performance + slope).

    Returns
    -------
    dict
        'osr': OSR values
        'sndr': SNDR in dB at each OSR
        'sfdr': SFDR in dB at each OSR
        'enob': ENOB in bits at each OSR
    """
    data = np.asarray(data, dtype=float).ravel()
    n = len(data)

    # Default OSR: sweep from 1 to N/2
    if osr is None:
        n_bins = n // 2
        osr = (n / 2) / np.arange(n_bins, 0, -1)

    osr = np.asarray(osr, dtype=float)

    # Step 1: Sine fit to separate ideal signal from error
    fit_result = fit_sine_4param(data)
    sig_fit = fit_result['fitted_signal']
    freq = fit_result['frequency']
    amplitude = fit_result['amplitude']

    # Step 2: Error spectrum with Hann window
    err = data - sig_fit
    win = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / n))
    err_windowed = err * win / np.sqrt(np.mean(win ** 2))
    err_spec = np.abs(np.fft.fft(err_windowed)) ** 2 / n ** 2 * 4  # one-sided scaling
    err_spec = err_spec[:n // 2 + 1]

    # Signal power (constant)
    sig_power = amplitude ** 2 / 2

    # Step 3: Sweep OSR (sorted descending for incremental accumulation)
    n_osr = len(osr)
    sndr = np.zeros(n_osr)
    sfdr = np.zeros(n_osr)
    enob = np.zeros(n_osr)

    sort_idx = np.argsort(osr)[::-1]  # descending
    osr_sorted = osr[sort_idx]

    n_inband_prev = 0
    noi_power = 0.0
    spur_power = 0.0

    for ii in range(n_osr):
        n_inband = int(np.floor(n / 2 / osr_sorted[ii]))
        n_inband = max(1, min(n_inband, len(err_spec)))

        if n_inband > n_inband_prev:
            incremental = err_spec[n_inband_prev:n_inband]
            noi_power += np.sum(incremental)
            spur_power = max(spur_power, np.max(incremental))
        n_inband_prev = n_inband

        orig_idx = sort_idx[ii]
        sndr[orig_idx] = 10 * np.log10(sig_power / noi_power)
        sfdr[orig_idx] = 10 * np.log10(sig_power / spur_power)
        enob[orig_idx] = (sndr[orig_idx] - 1.76) / 6.02

    # Step 4: Plot
    if create_plot:
        make_slope = (ax is None)

        if ax is None:
            fig, axes_arr = plt.subplots(1, 2, figsize=(14, 5))
            ax_main = axes_arr[0]
            ax_slope = axes_arr[1]
        else:
            ax_main = ax

        # --- Main performance plot ---
        ax_main.semilogx(osr, sndr, 'b-', linewidth=1.5, label='SNDR (ENOB)')
        ax_main.semilogx(osr, sfdr, 'r-', linewidth=1.5, label='SFDR')
        ax_main.set_ylabel('SNDR / SFDR (dB)')
        ax_main.set_xlabel('OSR')
        ax_main.set_title('Performance vs OSR')
        ax_main.grid(True)
        ax_main.legend(loc='lower right')

        # Right y-axis for ENOB
        ax_enob = ax_main.twinx()
        sndr_lim = [min(np.min(sndr), np.min(sfdr)) - 5,
                    max(np.max(sndr), np.max(sfdr)) + 5]
        enob_lim = [(s - 1.76) / 6.02 for s in sndr_lim]
        ax_main.set_ylim(sndr_lim)
        ax_enob.set_ylim(enob_lim)
        ax_enob.set_ylabel('ENOB (bits)')

        # Mark fundamental and harmonics
        osr_sig = 1.0 / (2 * freq)
        y_lim = ax_main.get_ylim()
        if min(osr) <= osr_sig <= max(osr):
            ax_main.axvline(osr_sig, color='k', linewidth=1)
            ax_main.text(osr_sig, y_lim[0], 'Fund', fontsize=8,
                         ha='right', va='bottom', color='k')

        for h in range(2, harmonic + 1):
            f_h = fold_frequency_to_nyquist(freq * h, 1.0)
            osr_h = 1.0 / (2 * f_h)
            if min(osr) <= osr_h <= max(osr):
                ax_main.axvline(osr_h, color='k', linestyle='--', linewidth=0.5)
                ax_main.text(osr_h, y_lim[0], f'HD{h}', fontsize=8,
                             ha='right', va='bottom', color='k')

        # --- Slope subplot ---
        if make_slope and len(osr) >= 3:
            log_osr = np.log10(osr)
            n_pts = len(osr)
            smooth_win = max(5, round(n_pts / 10))
            smooth_win = min(smooth_win, (n_pts - 1) // 2)

            local_slope = np.zeros(n_pts)
            for i in range(n_pts):
                i_lo = max(0, i - smooth_win)
                i_hi = min(n_pts - 1, i + smooth_win)
                denom = log_osr[i_hi] - log_osr[i_lo]
                if abs(denom) > 1e-15:
                    local_slope[i] = (sndr[i_hi] - sndr[i_lo]) / denom

            ax_slope.semilogx(osr, local_slope, 'b-', linewidth=1.5)
            ax_slope.axhline(10, color='k', linestyle='--', linewidth=0.5)
            ax_slope.text(max(osr), 10, 'White Noise Limit', fontsize=8,
                          ha='right', va='bottom')
            ax_slope.set_ylabel('SNDR Slope (dB/decade)')
            ax_slope.set_xlabel('OSR')
            ax_slope.grid(True)

            slope_range = np.ptp(local_slope)
            ax_slope.set_ylim([np.min(local_slope) - 0.1 * slope_range - 5,
                               np.max(local_slope) + 0.1 * slope_range + 5])

            plt.tight_layout()

    return {'osr': osr, 'sndr': sndr, 'sfdr': sfdr, 'enob': enob}
