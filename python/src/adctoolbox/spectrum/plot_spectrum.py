"""
Pure spectrum plotting functionality without calculations.

This module extracts the plotting logic from analyze_spectrum to create
a pure plotting function that can be used with pre-computed metrics.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_spectrum(analysis_results, show_title=True, show_label=True, plot_harmonics_up_to=3, ax=None):
    """
    Pure spectrum plotting using pre-computed analysis results.

    Parameters:
        analysis_results: Dictionary containing 'metrics' and 'plot_data' from compute_spectrum
        show_label: Add labels and annotations (True) or not (False)
        plot_harmonics_up_to: Number of harmonics to highlight
        show_title: Display auto-generated title (True) or not (False)
        ax: Optional matplotlib axes object
    """
    # Extract metrics and plot_data from analysis_results
    metrics = analysis_results['metrics']
    plot_data = analysis_results['plot_data']
    collided_harmonics = metrics.get('collided_harmonics', [])

    # Extract plot data
    spec_db = plot_data['spec_db']
    freq = plot_data['freq']
    bin_idx = plot_data['bin_idx']
    sig_bin_start = plot_data['sig_bin_start']
    sig_bin_end = plot_data['sig_bin_end']
    spur_bin_idx = plot_data['spur_bin_idx']
    spur_db = plot_data['spur_db']
    Nd2_inband = plot_data['Nd2_inband']
    N = plot_data['N']
    M = plot_data['M']
    fs = plot_data['fs']
    osr = plot_data['osr']
    nf_line_level = plot_data['nf_line_level']
    harmonics = plot_data['harmonics']
    is_coherent = plot_data.get('is_coherent', False)  # Default to False for backward compatibility

    # Extract metrics
    enob = metrics['enob']
    sndr_db = metrics['sndr_db']
    sfdr_db = metrics['sfdr_db']
    thd_db = metrics['thd_db']
    snr_db = metrics['snr_db']
    sig_pwr_dbfs = metrics['sig_pwr_dbfs']
    noise_floor_dbfs = metrics['noise_floor_dbfs']
    nsd_dbfs_hz = metrics['nsd_dbfs_hz']

    # Setup axes
    if ax is None:
        ax = plt.gca()

    # --- Plot spectrum ---
    # Always use ax.plot() - when osr>1, the semilogx call later will convert axes to log
    ax.plot(freq, spec_db)
    ax.grid(True, which='both', linestyle='--')

    if show_label:
        # Highlight fundamental - always use ax.plot(), axes scale handled by osr
        ax.plot(freq[sig_bin_start:sig_bin_end], spec_db[sig_bin_start:sig_bin_end], 'r-', linewidth=2.0)
        ax.plot(freq[bin_idx], spec_db[bin_idx], 'ro', linewidth=1.0, markersize=8)

        # Plot harmonics
        if plot_harmonics_up_to > 0:
            for harm in harmonics:
                if harm['harmonic_num'] <= plot_harmonics_up_to:
                    ax.plot(harm['freq'], harm['power_db'], 'rs', markersize=5)
                    ax.text(harm['freq'], harm['power_db'] + 5, str(harm['harmonic_num']),
                            fontname='Arial', fontsize=12, ha='center')

        # Plot max spurious
        ax.plot(spur_bin_idx / N * fs, spur_db, 'rd', markersize=5)
        ax.text(spur_bin_idx / N * fs, spur_db + 10, 'MaxSpur',
                fontname='Arial', fontsize=10, ha='center')

    # --- Set axis limits ---
    # Adaptive y-axis: start at -100 dB, extend if >5% of data is below each threshold
    minx = -100
    for threshold in [-100, -120, -140, -160, -180]:
        below_threshold = np.sum(spec_db[:Nd2_inband] < threshold)
        percentage = below_threshold / len(spec_db[:Nd2_inband]) * 100
        if percentage > 5.0:
            minx = threshold - 20  # Extend to next level
        else:
            break
    minx = max(minx, -200)  # Absolute floor
    ax.set_xlim(fs/N, fs/2)
    ax.set_ylim(minx, 0)

    # --- Add annotations ---
    if show_label:
        # OSR line
        ax.plot([fs/2/osr, fs/2/osr], [0, -1000], '--', color='gray', linewidth=1)

        # Text positioning
        if osr > 1:
            TX = 10**(np.log10(fs)*0.01 + np.log10(fs/N)*0.99)
        else:
            if bin_idx/N < 0.2:
                TX = fs * 0.3
            else:
                TX = fs * 0.01
        TYD = minx * 0.06

        # Format helpers
        def format_freq(f):
            if f >= 1e9: return f'{f/1e9:.1f}G'
            elif f >= 1e6: return f'{f/1e6:.1f}M'
            elif f >= 1e3: return f'{f/1e3:.1f}K'
            else: return f'{f:.1f}'

        txt_fs = format_freq(fs)
        Fin = bin_idx/N * fs

        if Fin >= 1e9: txt_fin = f'{Fin/1e9:.1f}G'
        elif Fin >= 1e6: txt_fin = f'{Fin/1e6:.1f}M'
        elif Fin >= 1e3: txt_fin = f'{Fin/1e3:.1f}K'
        elif Fin >= 1: txt_fin = f'{Fin/1e3:.1f}'  # Matches original logic
        else: txt_fin = f'{Fin:.3f}'

        # Annotation block
        ax.text(TX, TYD, f'Fin/fs = {txt_fin} / {txt_fs} Hz', fontsize=10)
        ax.text(TX, TYD*2, f'ENoB = {enob:.2f}', fontsize=10)
        ax.text(TX, TYD*3, f'SNDR = {sndr_db:.2f} dB', fontsize=10)
        ax.text(TX, TYD*4, f'SFDR = {sfdr_db:.2f} dB', fontsize=10)
        ax.text(TX, TYD*5, f'THD = {thd_db:.2f} dB', fontsize=10)
        ax.text(TX, TYD*6, f'SNR = {snr_db:.2f} dB', fontsize=10)
        ax.text(TX, TYD*7, f'Noise Floor = {noise_floor_dbfs:.2f} dB', fontsize=10)
        ax.text(TX, TYD*8, f'NSD = {nsd_dbfs_hz:.2f} dBFS/Hz', fontsize=10)

        # Noise floor baseline
        if osr > 1:
            ax.semilogx([fs/N, fs/2/osr], [nf_line_level, nf_line_level], 'r--', linewidth=1)
            ax.text(TX, TYD*9, f'OSR = {osr:.2f}', fontsize=10)
        else:
            ax.plot([0, fs/2], [nf_line_level, nf_line_level], 'r--', linewidth=1)

        # Add coherent integration gain note
        if is_coherent and M > 1:
            coh_gain_db = 10 * np.log10(M)
            if osr > 1:
                ax.text(TX, TYD*10, f'*Coherent Integration Gain = {coh_gain_db:.2f} dB', fontsize=10)
            else:
                ax.text(TX, TYD*9, f'*Coherent Integration Gain = {coh_gain_db:.2f} dB', fontsize=10)

        # Add collision warning if harmonics collided with fundamental
        if collided_harmonics:
            collision_str = ', '.join([f'HD{h}' for h in sorted(collided_harmonics)])
            text_y_offset = TYD*11 if (is_coherent and M > 1 and osr > 1) else (TYD*10 if (is_coherent and M > 1) or osr > 1 else TYD*9)
            ax.text(TX, text_y_offset, f'*Collided with fundamental: {collision_str}', fontsize=10, color='orange')

        # Signal annotation
        sig_y_pos = min(sig_pwr_dbfs, TYD/2)
        if osr > 1:
            ax.text(freq[bin_idx], sig_y_pos, f'Sig = {sig_pwr_dbfs:.2f} dB', fontsize=10)
        else:
            offset = -0.01 if bin_idx/N > 0.4 else 0.01
            ha_align = 'right' if bin_idx/N > 0.4 else 'left'
            ax.text((bin_idx/N + offset) * fs, sig_y_pos, f'Sig = {sig_pwr_dbfs:.2f} dB',
                    ha=ha_align, fontsize=10)

        ax.set_xlabel('Freq (Hz)', fontsize=10)
        ax.set_ylabel('dBFS', fontsize=10)

    # Title - auto-generate based on mode and number of runs
    if show_title:
        if is_coherent:
            if M > 1:
                ax.set_title(f'Coherent averaging (N_run = {M})', fontsize=12, fontweight='bold')
            else:
                ax.set_title('Coherent Spectrum', fontsize=12, fontweight='bold')
        else:
            if M > 1:
                ax.set_title(f'Power averaging (N_run = {M})', fontsize=12, fontweight='bold')
            else:
                ax.set_title('Power Spectrum', fontsize=12, fontweight='bold')