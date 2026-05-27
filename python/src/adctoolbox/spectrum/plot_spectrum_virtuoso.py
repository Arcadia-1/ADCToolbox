"""
Cadence Virtuoso / ADE Explorer style spectrum plot.

Dark canvas, red stem bars per FFT bin, fine dotted grid.  Same set of
annotations as the analyzer-style plotter (fundamental marker, max-spur
diamond, harmonic squares, metrics text block, NSD line) but recolored
for the dark theme:

    - stems      : red          ('1 stem = 1 bin' Virtuoso convention)
    - fundamental: yellow dot + "Sig = X dB"
    - max spur   : yellow diamond + "MaxSpur"
    - harmonics  : cyan squares + cyan order numbers (2, 3, ...)
    - metric box : white text
    - NSD line   : yellow dashed
    - OSR line   : dim white dashed

Mirrors `plot_spectrum.py`'s structure so the two stay in lock-step
when the underlying metrics / plot_data layout evolves.
"""

import numpy as np
import matplotlib.pyplot as plt

from adctoolbox.spectrum.plot_spectrum import (
    _noise_floor_axis_min,
    _should_label_harmonic,
)


# Color palette — single place to retune the dark theme
_C_STEM   = '#ff3030'   # red, slightly brighter than 'r' for visibility on black
_C_FUND   = '#ffd000'   # warm yellow — fundamental + spur (loud-and-clear)
_C_HARM   = '#00e5ff'   # cyan — harmonics (distinct from fundamental)
_C_METRIC = 'white'     # metric text block
_C_NSD    = '#ffd000'   # NSD line — match fundamental color
_C_OSR    = '#999999'   # dim white — OSR cutoff line
_C_COH    = '#ffe680'   # light yellow — coherent gain note
_C_WARN   = '#ffa500'   # orange — collision warning


def plot_spectrum_virtuoso(compute_results, show_title=True, show_label=True,
                           plot_harmonics_up_to=3, ax=None, baseline_db=None):
    """
    Virtuoso/ADE-Explorer style spectrum plot with annotations.

    Parameters
    ----------
    compute_results : dict
        Output of `compute_spectrum`.
    show_title : bool
        Show auto-generated plot title.
    show_label : bool
        Add fundamental marker, spur marker, harmonics markers, and the
        metric text block.  Set False for a bare-stem look.
    plot_harmonics_up_to : int
        Highlight harmonics up to this order (HD2..HDk).
    ax : matplotlib.axes.Axes, optional
        Target axes.  If None, uses current axes.  Axes + figure facecolor
        will be set to black.
    baseline_db : float, optional
        Bottom of stem bars in dB.  None → adaptive (min(spec) - 5).
    """
    metrics    = compute_results['metrics']
    plot_data  = compute_results['plot_data']
    spec_db    = plot_data['power_spectrum_db_plot']
    freq       = plot_data['freq']
    fundamental_bin = plot_data['fundamental_bin']
    spur_bin_idx    = plot_data['spur_bin_idx']
    spur_db         = spec_db[spur_bin_idx]
    is_coherent     = plot_data.get('is_coherent', False)
    collided_harmonics = plot_data.get('collided_harmonics', [])
    harmonic_bins      = plot_data.get('harmonic_bins', [])
    harmonics_dbc      = metrics.get('harmonics_dbc', [])

    N    = compute_results['N']
    M    = compute_results['M']
    fs   = compute_results['fs']
    osr  = compute_results['osr']
    nf_line_level = metrics['noise_floor_dbfs'] - 10 * np.log10(N / (2 * osr))

    if ax is None:
        ax = plt.gca()
    fig = ax.figure

    # ---- Dark theme: black canvas, white annotations -----------------
    fig.set_facecolor('black')
    ax.set_facecolor('black')
    for spine in ax.spines.values():
        spine.set_color('white')
    ax.tick_params(axis='both', which='both', colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')

    # ---- Axis floor + stem bars --------------------------------------
    sndr_floor_level = metrics['sig_pwr_dbfs'] - metrics['sndr_dbc']
    minx = _noise_floor_axis_min(nf_line_level, fallback_level=sndr_floor_level)
    if baseline_db is None:
        baseline_db = minx
    else:
        minx = min(minx, baseline_db)
    ax.vlines(freq, baseline_db, spec_db, colors=_C_STEM, linewidth=0.8)

    # ---- Grid --------------------------------------------------------
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle=':', color='white', alpha=0.35, linewidth=0.6)
    ax.grid(True, which='minor', linestyle=':', color='white', alpha=0.15, linewidth=0.4)

    # ---- Y-axis follows the plotted NSD/bin line ---------------------
    x_min = fs / N
    x_max = fs / 2
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(minx, 0)

    # ---- Labels ------------------------------------------------------
    ax.set_xlabel('freq (Hz)')
    ax.set_ylabel('(dB)')
    if show_title:
        ax.set_title('Power Spectrum')

    if not show_label:
        return

    # ---- Fundamental marker + "Sig" text -----------------------------
    ax.plot(freq[fundamental_bin], spec_db[fundamental_bin],
            'o', color=_C_FUND, markersize=8)

    # ---- Harmonic markers --------------------------------------------
    if plot_harmonics_up_to > 0 and len(harmonic_bins) and len(harmonics_dbc):
        for idx in range(len(harmonics_dbc)):
            order = idx + 2                  # HD2, HD3, ...
            if order > plot_harmonics_up_to or order in collided_harmonics:
                continue
            bin_center = harmonic_bins[idx]
            if not _should_label_harmonic(spec_db[bin_center], nf_line_level):
                continue
            ax.plot(bin_center * fs / N, spec_db[bin_center],
                    's', color=_C_HARM, markersize=5)
            ax.text(bin_center * fs / N, spec_db[bin_center] + 3, str(order),
                    color=_C_HARM, fontsize=11, ha='center')

    # ---- Max-spur diamond + "MaxSpur" text ---------------------------
    ax.plot(spur_bin_idx / N * fs, spur_db,
            'd', color=_C_FUND, markersize=5)
    ax.text(spur_bin_idx / N * fs, spur_db + 10, 'MaxSpur',
            color=_C_FUND, fontsize=10, ha='center')

    # ---- Text-block positioning (mirrors plot_spectrum.py) -----------
    # Axes-relative metric text stays fixed if callers change y-limits.
    metric_x = 0.02 if osr > 1 or fundamental_bin / N >= 0.2 else 0.60
    metric_y_start = 0.94
    metric_y_step = 0.06

    def _fmt_freq(f):
        if f >= 1e9: return f'{f/1e9:.1f}G'
        if f >= 1e6: return f'{f/1e6:.1f}M'
        if f >= 1e3: return f'{f/1e3:.1f}K'
        return f'{f:.1f}'

    Fin = fundamental_bin / N * fs

    ax.text(freq[fundamental_bin], 0.98,
            f'Sig = {metrics["sig_pwr_dbfs"]:.2f} dB',
            transform=ax.get_xaxis_transform(),
            color=_C_FUND, fontsize=10, va='top')

    snr_text = f'{metrics["snr_dbc"]:.2f} dB' if np.isfinite(metrics["snr_dbc"]) else 'N/A'
    noise_floor_text = (
        f'{metrics["noise_floor_dbfs"]:.2f} dB'
        if np.isfinite(metrics["noise_floor_dbfs"])
        else 'N/A'
    )
    nsd_text = (
        f'{metrics["nsd_dbfs_hz"]:.2f} dBFS/Hz'
        if np.isfinite(metrics["nsd_dbfs_hz"])
        else 'N/A'
    )

    metric_lines = [
        f'Fin/fs = {_fmt_freq(Fin)} / {_fmt_freq(fs)} Hz',
        f'ENoB = {metrics["enob"]:.2f}',
        f'SNDR = {metrics["sndr_dbc"]:.2f} dB',
        f'SFDR = {metrics["sfdr_dbc"]:.2f} dB',
        f'THD = {metrics["thd_dbc"]:.2f} dB',
        f'SNR = {snr_text}',
        f'Noise Floor = {noise_floor_text}',
        f'NSD = {nsd_text}',
    ]
    # ---- NSD baseline line -------------------------------------------
    if not np.isfinite(nf_line_level):
        pass
    elif osr > 1:
        ax.semilogx([fs / N, fs / 2 / osr],
                    [nf_line_level, nf_line_level],
                    '--', color=_C_NSD, linewidth=1)
        metric_lines.append(f'OSR = {osr:.2f}')
        ax.plot([fs / 2 / osr, fs / 2 / osr], [0, -1000],
                '--', color=_C_OSR, linewidth=1)
    else:
        ax.plot([0, fs / 2], [nf_line_level, nf_line_level],
                '--', color=_C_NSD, linewidth=1)

    # ---- Optional gain / collision notes -----------------------------
    warning_lines = []
    if is_coherent and M > 1:
        coh_gain_db = 10 * np.log10(M)
        warning_lines.append((f'*Coherent Gain = {coh_gain_db:.2f} dB', _C_COH))
    if collided_harmonics:
        collision_str = ', '.join(f'HD{h}' for h in sorted(collided_harmonics))
        warning_lines.append((f'*Collided with fundamental: {collision_str}', _C_WARN))

    metric_rows = [(line, _C_METRIC) for line in metric_lines] + warning_lines
    for row, (line, color) in enumerate(metric_rows):
        ax.text(
            metric_x,
            metric_y_start - metric_y_step * row,
            line,
            transform=ax.transAxes,
            color=color,
            fontsize=10,
            ha='left',
            va='top',
        )
