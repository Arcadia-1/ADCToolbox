"""
Pure spectrum plotting functionality without calculations.

This module extracts the plotting logic from analyze_spectrum to create
a pure plotting function that can be used with pre-computed metrics.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_spectrum(compute_results, show_title=True, show_label=True, plot_harmonics_up_to=3, ax=None):
    """
    Pure spectrum plotting using pre-computed analysis results.

    Parameters:
        compute_results: Dictionary containing 'metrics' and 'plot_data' from compute_spectrum
        show_label: Add labels and annotations (True) or not (False)
        plot_harmonics_up_to: Number of harmonics to highlight
        show_title: Display auto-generated title (True) or not (False)
        ax: Optional matplotlib axes object
    """
    # Extract metrics and plot_data from compute_results
    metrics = compute_results['metrics']
    plot_data = compute_results['plot_data']
    collided_harmonics = plot_data.get('collided_harmonics', [])

    # Extract plot data
    spec_db = plot_data['power_spectrum_db_plot']
    freq = plot_data['freq']
    fundamental_bin = plot_data['fundamental_bin']
    sig_bin_start = plot_data['sig_bin_start']
    sig_bin_end = plot_data['sig_bin_end']
    spur_bin_idx = plot_data['spur_bin_idx']
    spur_db = spec_db[spur_bin_idx]  # Calculate from power_spectrum_db_plot
    is_coherent = plot_data.get('is_coherent', False)

    # Extract metadata
    N = compute_results['N']
    M = compute_results['M']
    fs = compute_results['fs']
    osr = compute_results['osr']
    v_offset = plot_data['v_offset']
    Nd2_inband = len(freq) // osr
    # Noise floor line: use NSD (per-Hz) converted to per-bin level, then apply v_offset
    # NSD stays constant across OSR changes, while integrated noise_floor_dbfs varies
    # v_offset aligns the spectrum with sig_pwr_dbfs, so noise floor line must be shifted too
    nf_line_level = metrics['nsd_dbfs_hz'] + 10 * np.log10(fs / N) + v_offset

    # Build harmonics list from plot_data and metrics (for plotting)
    harmonic_bins = plot_data.get('harmonic_bins', [])
    harmonics_dbc = metrics.get('harmonics_dbc', [])
    harmonics = []
    if len(harmonic_bins) > 0 and len(harmonics_dbc) > 0:
        for harmonic_index in range(len(harmonics_dbc)):
            harmonic_order = harmonic_index + 2  # HD2=2, HD3=3, etc.

            # Skip if this harmonic collided with fundamental
            if harmonic_order in collided_harmonics:
                continue

            # Get harmonic bin position
            harmonic_bin_center = harmonic_bins[harmonic_index]

            # Get power in dB and calculate frequency
            harmonic_power_db = spec_db[harmonic_bin_center]
            harmonic_freq = harmonic_bin_center * fs / N

            harmonics.append({
                'harmonic_num': harmonic_order,
                'freq': harmonic_freq,
                'power_db': harmonic_power_db
            })

    # Extract metrics
    enob = metrics['enob']
    sndr_dbc = metrics['sndr_dbc']
    sfdr_dbc = metrics['sfdr_dbc']
    thd_dbc = metrics['thd_dbc']
    snr_dbc = metrics['snr_dbc']
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
        ax.plot(freq[fundamental_bin], spec_db[fundamental_bin], 'ro', linewidth=1.0, markersize=8)

        # Plot harmonics
        if plot_harmonics_up_to > 0:
            for harm in harmonics:
                if harm['harmonic_num'] <= plot_harmonics_up_to:
                    ax.plot(harm['freq'], harm['power_db'], 'rs', markersize=5)
                    ax.text(harm['freq'], harm['power_db'] + 3, str(harm['harmonic_num']),
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
            if fundamental_bin/N < 0.2:
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
        Fin = fundamental_bin/N * fs

        if Fin >= 1e9: txt_fin = f'{Fin/1e9:.1f}G'
        elif Fin >= 1e6: txt_fin = f'{Fin/1e6:.1f}M'
        elif Fin >= 1e3: txt_fin = f'{Fin/1e3:.1f}K'
        elif Fin >= 1: txt_fin = f'{Fin/1e3:.1f}'  # Matches original logic
        else: txt_fin = f'{Fin:.3f}'

        # Annotation block
        ax.text(TX, TYD, f'Fin/fs = {txt_fin} / {txt_fs} Hz', fontsize=10)
        ax.text(TX, TYD*2, f'ENoB = {enob:.2f}', fontsize=10)
        ax.text(TX, TYD*3, f'SNDR = {sndr_dbc:.2f} dB', fontsize=10)
        ax.text(TX, TYD*4, f'SFDR = {sfdr_dbc:.2f} dB', fontsize=10)
        ax.text(TX, TYD*5, f'THD = {thd_dbc:.2f} dB', fontsize=10)
        ax.text(TX, TYD*6, f'SNR = {snr_dbc:.2f} dB', fontsize=10)
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
                ax.text(TX, TYD*10, f'*Coherent Gain = {coh_gain_db:.2f} dB', fontsize=10)
            else:
                ax.text(TX, TYD*9, f'*Coherent Gain = {coh_gain_db:.2f} dB', fontsize=10)

        # Add collision warning if harmonics collided with fundamental
        if collided_harmonics:
            collision_str = ', '.join([f'HD{h}' for h in sorted(collided_harmonics)])
            text_y_offset = TYD*11 if (is_coherent and M > 1 and osr > 1) else (TYD*10 if (is_coherent and M > 1) or osr > 1 else TYD*9)
            ax.text(TX, text_y_offset, f'*Collided with fundamental: {collision_str}', fontsize=10, color='orange')

        # Signal annotation
        sig_y_pos = min(sig_pwr_dbfs, TYD/2)
        if osr > 1:
            ax.text(freq[fundamental_bin], sig_y_pos, f'Sig = {sig_pwr_dbfs:.2f} dB', fontsize=10)
        else:
            offset = -0.01 if fundamental_bin/N > 0.4 else 0.01
            ha_align = 'right' if fundamental_bin/N > 0.4 else 'left'
            ax.text((fundamental_bin/N + offset) * fs, sig_y_pos, f'Sig = {sig_pwr_dbfs:.2f} dB',
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