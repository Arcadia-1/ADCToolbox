"""Performance metrics vs oversampling ratio (OSR) sweep.

Analyzes how ADC performance (SNDR, SFDR, ENOB) varies with oversampling ratio
by separating ideal signal from error via sine fitting.
"""

import numpy as np
import matplotlib.pyplot as plt

from adctoolbox.fundamentals import fit_sine_4param, fold_frequency_to_nyquist
from adctoolbox.spectrum._bin_ranges import rfft_inband_bin_count


def _one_sided_rms_power_spectrum(data: np.ndarray) -> np.ndarray:
    """Return a one-sided spectrum whose sum equals time-domain RMS power."""
    n = len(data)
    spectrum = np.abs(np.fft.rfft(data)) ** 2 / n**2
    if n <= 1:
        return spectrum

    if n % 2 == 0:
        spectrum[1:-1] *= 2.0
    else:
        spectrum[1:] *= 2.0
    return spectrum


def sweep_performance_vs_osr(
    data: np.ndarray,
    osr: np.ndarray | None = None,
    harmonic: int = 5,
    create_plot: bool = True,
    ax: plt.Axes | None = None,
    logscale: bool = True,
    smooth: int | None = None,
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
    logscale : bool, default=True
        Use logarithmic OSR axis, matching MATLAB ``perfosr`` default.
    smooth : int, optional
        Half-width used for local SNDR slope estimation. Default matches the
        local MATLAB-style heuristic.

    Returns
    -------
    dict
        'osr': OSR values
        'sndr': SNDR in dB at each OSR
        'sfdr': Fast residual-spectrum single-bin SFDR estimate in dB at each OSR
        'enob': ENOB in bits at each OSR
    """
    data = np.asarray(data, dtype=float)
    if data.ndim > 1 and min(data.shape) > 1:
        # Concatenating several records would splice their phases together
        # and read as a broadband error (SNDR near 0 dB).
        raise ValueError(
            f"data must be a single 1-D record, got shape {data.shape}; "
            "analyze each record separately."
        )
    data = data.ravel()
    n = len(data)

    # Default OSR: sweep from 1 to N/2
    if osr is None:
        n_bins = rfft_inband_bin_count(n, osr=1) - 1
        osr = (n / 2) / np.arange(n_bins, 0, -1)

    osr = np.atleast_1d(np.asarray(osr, dtype=float))
    if np.any(~np.isfinite(osr)) or np.any(osr <= 0):
        raise ValueError("osr values must be finite and positive")

    # Step 1: Sine fit to separate ideal signal from error
    fit_result = fit_sine_4param(data)
    sig_fit = fit_result['fitted_signal']
    freq = fit_result['frequency']
    amplitude = fit_result['amplitude']

    # Step 2: Error spectrum with Hann window. Use one-sided RMS-power
    # scaling so sum(err_spec) matches the residual RMS power.
    err = data - sig_fit
    win = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / n))
    err_windowed = err * win / np.sqrt(np.mean(win ** 2))
    err_spec = _one_sided_rms_power_spectrum(err_windowed)

    # Mask residual DC leakage (bins 0..1) and fundamental sine-fit leakage
    # (center ± 3 bins) caused by unweighted time-domain fitting in strong
    # out-of-band noise (e.g. high-order delta-sigma modulators).
    err_spec[: min(2, len(err_spec))] = 0.0
    fund_bin = int(round(freq * n))
    if 0 <= fund_bin < len(err_spec):
        f_lo = max(0, fund_bin - 3)
        f_hi = min(len(err_spec), fund_bin + 4)
        err_spec[f_lo:f_hi] = 0.0

    # Integrate each candidate spur over its 3-bin Hann main lobe (center ± 1)
    # to eliminate the single-bin Hann ENBW bias (+1.76 dB coherent, +3.19 dB
    # half-bin) while remaining invariant across single-point and multi-point sweeps.
    spur_lobe_spec = np.convolve(err_spec, np.ones(3), mode='same')

    # Signal power (constant)
    sig_power = amplitude ** 2 / 2

    # Step 3: Sweep OSR using prefix sums/max so any single-point and
    # multi-point OSR queries agree bit-for-bit.
    n_osr = len(osr)
    sndr = np.zeros(n_osr)
    sfdr = np.zeros(n_osr)
    enob = np.zeros(n_osr)

    cum_noi = np.cumsum(err_spec)
    cum_spur = np.maximum.accumulate(spur_lobe_spec)

    for ii in range(n_osr):
        n_inband = rfft_inband_bin_count(n, osr[ii])
        noi_power = float(cum_noi[n_inband - 1])
        spur_power = float(cum_spur[n_inband - 1])

        sndr[ii] = 10 * np.log10(sig_power / noi_power) if noi_power > 0 else np.inf
        sfdr[ii] = 10 * np.log10(sig_power / spur_power) if spur_power > 0 else np.inf
        enob[ii] = (sndr[ii] - 1.76) / 6.02

    # Step 4: Plot
    if create_plot:
        make_slope = (ax is None)

        if ax is None:
            fig, axes_arr = plt.subplots(1, 2, figsize=(14, 5))
            ax_main = axes_arr[0]
            ax_slope = axes_arr[1]
        else:
            ax_main = ax

        # Sort ascending for clean line plots and monotonic local-slope differences
        asc_idx = np.argsort(osr)
        osr_asc = osr[asc_idx]
        sndr_asc = sndr[asc_idx]
        sfdr_asc = sfdr[asc_idx]

        # --- Main performance plot ---
        plot_fn = ax_main.semilogx if logscale else ax_main.plot
        plot_fn(osr_asc, sndr_asc, 'b-', linewidth=1.5, label='SNDR (ENOB)')
        plot_fn(osr_asc, sfdr_asc, 'r-', linewidth=1.5, label='SFDR')
        ax_main.set_ylabel('SNDR / SFDR (dB)')
        ax_main.set_xlabel('OSR')
        ax_main.set_title('Performance vs OSR')
        ax_main.grid(True)
        ax_main.legend(loc='lower right')

        # Right y-axis for ENOB
        ax_enob = ax_main.twinx()
        finite_vals = np.concatenate([sndr[np.isfinite(sndr)], sfdr[np.isfinite(sfdr)]])
        if finite_vals.size > 0:
            sndr_lim = [float(np.min(finite_vals)) - 5, float(np.max(finite_vals)) + 5]
        else:
            sndr_lim = [0.0, 100.0]
        enob_lim = [(s - 1.76) / 6.02 for s in sndr_lim]
        ax_main.set_ylim(sndr_lim)
        ax_enob.set_ylim(enob_lim)
        ax_enob.set_ylabel('ENOB (bits)')

        # Mark fundamental and harmonics (skip any that alias to DC)
        y_lim = ax_main.get_ylim()
        if freq > 0:
            osr_sig = 1.0 / (2 * freq)
            if min(osr) <= osr_sig <= max(osr):
                ax_main.axvline(osr_sig, color='k', linewidth=1)
                ax_main.text(osr_sig, y_lim[0], 'Fund', fontsize=8,
                             ha='right', va='bottom', color='k')

        for h in range(2, harmonic + 1):
            f_h = fold_frequency_to_nyquist(freq * h, 1.0)
            if f_h <= 0:
                continue
            osr_h = 1.0 / (2 * f_h)
            if min(osr) <= osr_h <= max(osr):
                ax_main.axvline(osr_h, color='k', linestyle='--', linewidth=0.5)
                ax_main.text(osr_h, y_lim[0], f'HD{h}', fontsize=8,
                             ha='right', va='bottom', color='k')

        # --- Slope subplot ---
        if make_slope and len(osr_asc) >= 3:
            log_osr = np.log10(osr_asc)
            n_pts = len(osr_asc)
            smooth_win = max(5, round(n_pts / 10)) if smooth is None else int(smooth)
            smooth_win = min(smooth_win, (n_pts - 1) // 2)

            local_slope = np.zeros(n_pts)
            for i in range(n_pts):
                i_lo = max(0, i - smooth_win)
                i_hi = min(n_pts - 1, i + smooth_win)
                denom = log_osr[i_hi] - log_osr[i_lo]
                if abs(denom) > 1e-15 and np.isfinite(sndr_asc[i_hi]) and np.isfinite(sndr_asc[i_lo]):
                    local_slope[i] = (sndr_asc[i_hi] - sndr_asc[i_lo]) / denom

            slope_plot_fn = ax_slope.semilogx if logscale else ax_slope.plot
            slope_plot_fn(osr_asc, local_slope, 'b-', linewidth=1.5)
            ax_slope.axhline(10, color='k', linestyle='--', linewidth=0.5)
            ax_slope.text(max(osr_asc), 10, 'White Noise Limit', fontsize=8,
                          ha='right', va='bottom')
            ax_slope.set_ylabel('SNDR Slope (dB/decade)')
            ax_slope.set_xlabel('OSR')
            ax_slope.grid(True)

            slope_range = np.ptp(local_slope)
            ax_slope.set_ylim([np.min(local_slope) - 0.1 * slope_range - 5,
                               np.max(local_slope) + 0.1 * slope_range + 5])

            plt.tight_layout()

    return {'osr': osr, 'sndr': sndr, 'sfdr': sfdr, 'enob': enob}
