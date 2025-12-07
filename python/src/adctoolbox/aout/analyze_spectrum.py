"""
ADC spectrum analysis with ENOB, SNDR, SFDR, SNR, THD, Noise Floor, NSD calculations.

MATLAB counterpart: specPlot.m, plotspec.m
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows
from adctoolbox.common.calc_aliased_freq import calc_aliased_freq

def analyze_spectrum(data, fs=1.0, max_code=None, harmonic=3, win_type='hann',
              side_bin=1, log_sca=0, label=1, assumed_signal=np.nan, is_plot=1,
              n_thd=5, osr=1, co_avg=0, nf_method=0, ax=None):
    """
    Spectral analysis and plotting.

    Parameters:
        data: Input data (N,) or (M, N)
        fs: Sampling frequency
        ax: Optional matplotlib axes object. If None and is_plot=1, a new figure is created.

    Returns:
        dict: Dictionary with keys:
            - enob: Effective Number of Bits
            - sndr_db: Signal-to-Noise and Distortion Ratio (dB)
            - sfdr_db: Spurious-Free Dynamic Range (dB)
            - snr_db: Signal-to-Noise Ratio (dB)
            - thd_db: Total Harmonic Distortion (dB)
            - sig_pwr_dbfs: Signal power (dBFS)
            - noise_floor_db: Noise floor (dB)
            - nsd_dbfs_hz: Noise Spectral Density (dBFS/Hz)
    """
    # --- Parameter processing ---
    data = np.asarray(data)

    if data.ndim == 1:
        data = data.reshape(1, -1)
    elif data.ndim == 2:
        N_rows, M_cols = data.shape
        if M_cols == 1 and N_rows > 1:
            data = data.T 

    M, N = data.shape 

    if max_code is None:
        max_code = np.max(data) - np.min(data) if N > 0 else 1

    Nd2 = N // 2
    freq = np.arange(Nd2) / N * fs

    # --- Window function ---
    win_type_str = win_type.lower()
    if win_type_str in ('boxcar', 'rectangular'):
        win = windows.boxcar(N)
    elif win_type_str in ('hann', 'hanning'):
        win = windows.hann(N, sym=False)
    elif win_type_str == 'hamming':
        win = windows.hamming(N, sym=False)
    else:
        raise ValueError(f"Unsupported window type: '{win_type}'")

    Nd2_inband = N // 2 // osr

    # --- Spectrum calculation ---
    spec = np.zeros(N)
    ME = 0
    for iter in range(M):
        tdata = data[iter, :]
        if np.sqrt(np.mean(tdata**2)) == 0:
            continue
        
        tdata = tdata / max_code
        tdata = tdata - np.mean(tdata)
        tdata = tdata * win / np.sqrt(np.mean(win**2))
        spec = spec + np.abs(np.fft.fft(tdata))**2
        ME += 1

    spec = spec[:Nd2]
    spec[0] = 0  # Ignore DC
    if ME > 0:
        spec = spec / (N**2) * 16 / ME

    # --- Find fundamental power ---
    spec_inband_search = spec[:Nd2_inband]
    bin_ = np.argmax(spec_inband_search)

    # Parabolic interpolation for refined bin location (matches MATLAB)
    sig_e = np.log10(spec[bin_]) if spec[bin_] > 0 else -20
    sig_l = np.log10(spec[max(bin_ - 1, 0)]) if spec[max(bin_ - 1, 0)] > 0 else -20
    sig_r = np.log10(spec[min(bin_ + 1, Nd2 - 1)]) if spec[min(bin_ + 1, Nd2 - 1)] > 0 else -20
    denominator = 2 * sig_e - sig_l - sig_r
    if denominator != 0:
        bin_r = bin_ + (sig_r - sig_l) / denominator / 2
    else:
        bin_r = bin_

    start = max(bin_ - side_bin, 0)
    end = min(bin_ + side_bin + 1, Nd2_inband)
    sig = np.sum(spec[start:end])
    sig_pwr_dbfs = 10 * np.log10(sig) if sig > 0 else -999

    if not np.isnan(assumed_signal):
        sig = 10**(assumed_signal / 10)
        sig_pwr_dbfs = assumed_signal

    # --- Plotting Setup ---
    if is_plot:
        if ax is None:
            ax = plt.gca()
        
        # Plot spectrum
        spec_db = 10 * np.log10(spec.clip(1e-20))
        if log_sca == 0:
            ax.plot(freq, spec_db)
        else:
            ax.semilogx(freq, spec_db)

        ax.grid(True, which='both', linestyle='--')

        if label:
            # Highlight fundamental
            if log_sca == 0:
                ax.plot(freq[start:end], spec_db[start:end], 'r-', linewidth=0.5)
                ax.plot(freq[bin_], spec_db[bin_], 'ro', linewidth=0.5)
            else:
                ax.semilogx(freq[start:end], spec_db[start:end], 'r-', linewidth=0.5)

        if label and harmonic > 0:
            for i in range(2, harmonic + 1):
                b = int(calc_aliased_freq(int(round(bin_r * i)), N))
                if b < len(spec):
                    ax.plot(b / N * fs, spec_db[b], 'rs')
                    ax.text(b / N * fs, spec_db[b] + 5, str(i),
                            fontname='Arial', fontsize=12, ha='center')

    # --- Metrics Calculation ---
    sigs = spec[bin_]
    if not np.isnan(assumed_signal):
        sigs = 10**(assumed_signal / 10)

    # Remove fundamental
    spec_no_sig = np.copy(spec)
    spec_no_sig[start:end] = 0
    spec_no_sig[:side_bin] = 0

    spec_inband = spec_no_sig[:Nd2_inband]
    noi = np.sum(spec_inband)

    # SFDR
    spur = np.max(spec_inband)
    sbin = np.argmax(spec_inband)

    if is_plot and label:
        spur_db = 10 * np.log10(spur + 1e-20)
        ax.plot(sbin / N * fs, spur_db, 'rd')
        ax.text(sbin / N * fs, spur_db + 5, 'MaxSpur',
                fontname='Arial', fontsize=10, ha='center')

    sndr_db = 10 * np.log10(sig / noi) if noi > 0 else 999
    sfdr_db = 10 * np.log10(sigs / spur) if spur > 0 else 999
    enob = (sndr_db - 1.76) / 6.02

    # Noise Floor Calculation (Methods 0, 1, 2)
    if nf_method == 0: # Median-based
        spec_for_nf = spec_no_sig[:Nd2_inband]
        median_val = np.median(spec_for_nf)
        noi_for_snr = median_val / np.sqrt((1 - 2/(9*ME))**3) * Nd2_inband
    elif nf_method == 1: # Trimmed mean
        spec_for_nf = spec_no_sig[:Nd2_inband]
        spec_sorted = np.sort(spec_for_nf)
        idx_start = int(np.floor(Nd2_inband * 0.05))
        idx_end = int(np.floor(Nd2_inband * 0.95))
        noi_for_snr = np.mean(spec_sorted[idx_start:idx_end]) * Nd2_inband
    else: # Sum after removing harmonics
        spec_noise = np.copy(spec_no_sig)
        for i in range(2, n_thd + 1):
            b = int(calc_aliased_freq(int(round(bin_r * i)), N))
            if b < Nd2_inband:
                spec_noise[b] = 0
        noi_for_snr = np.sum(spec_noise[:Nd2_inband])

    # THD
    thd_pwr = 0
    for i in range(2, n_thd + 1):
        b = int(calc_aliased_freq(int(round(bin_r * i)), N))
        if b < Nd2_inband:
            thd_pwr += spec_no_sig[b]

    thd_db = 10 * np.log10(thd_pwr / sigs) if sigs > 0 else -999
    snr_db = 10 * np.log10(sig / noi_for_snr) if noi_for_snr > 0 else 999
    noise_floor_db = snr_db - sig_pwr_dbfs
    
    # NSD (Noise Spectral Density)
    nsd_dbfs_hz = noise_floor_db + 10*np.log10(fs/2/osr)

    # --- Plot Annotations ---
    if is_plot:
        # Dynamic Y-axis limit
        minx = min(max(np.median(spec_db[:Nd2_inband])-20, -200), -40)
        ax.set_xlim(fs/N, fs/2)
        ax.set_ylim(minx, 0)

        if label:
            # OSR line
            ax.plot([fs/2/osr, fs/2/osr], [0, -1000], '--', color='gray', linewidth=1)

            # Text positioning
            if osr > 1:
                TX = 10**(np.log10(fs)*0.01 + np.log10(fs/N)*0.99)
            else:
                if bin_/N < 0.2:
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
            Fin = bin_/N * fs
            
            if Fin >= 1e9: txt_fin = f'{Fin/1e9:.1f}G'
            elif Fin >= 1e6: txt_fin = f'{Fin/1e6:.1f}M'
            elif Fin >= 1e3: txt_fin = f'{Fin/1e3:.1f}K'
            elif Fin >= 1: txt_fin = f'{Fin/1e3:.1f}' # Matches original logic
            else: txt_fin = f'{Fin:.3f}'

            # Annotation block
            ax.text(TX, TYD, f'Fin/fs = {txt_fin} / {txt_fs} Hz', fontsize=10)
            ax.text(TX, TYD*2, f'ENoB = {enob:.2f}', fontsize=10)
            ax.text(TX, TYD*3, f'SNDR = {sndr_db:.2f} dB', fontsize=10)
            ax.text(TX, TYD*4, f'SFDR = {sfdr_db:.2f} dB', fontsize=10)
            ax.text(TX, TYD*5, f'THD = {thd_db:.2f} dB', fontsize=10)
            ax.text(TX, TYD*6, f'SNR = {snr_db:.2f} dB', fontsize=10)
            ax.text(TX, TYD*7, f'Noise Floor = {noise_floor_db:.2f} dB', fontsize=10)
            ax.text(TX, TYD*8, f'NSD = {nsd_dbfs_hz:.2f} dBFS/Hz', fontsize=10)

            # Noise floor baseline
            nf_level = -(noise_floor_db + 10*np.log10(N/2/osr))
            if osr > 1:
                ax.semilogx([fs/N, fs/2/osr], [nf_level, nf_level], 'r--', linewidth=1)
                ax.text(TX, TYD*9, f'osr = {osr:.2f}', fontsize=10)
            else:
                ax.plot([0, fs/2], [nf_level, nf_level], 'r--', linewidth=1)

            # Signal annotation
            sig_y_pos = min(sig_pwr_dbfs, TYD/2)
            if osr > 1:
                ax.text(freq[bin_], sig_y_pos, f'Sig = {sig_pwr_dbfs:.2f} dB', fontsize=10)
            else:
                offset = -0.01 if bin_/N > 0.4 else 0.01
                ha_align = 'right' if bin_/N > 0.4 else 'left'
                ax.text((bin_/N + offset) * fs, sig_y_pos, f'Sig = {sig_pwr_dbfs:.2f} dB', 
                        ha=ha_align, fontsize=10)

            ax.set_xlabel('Freq (Hz)', fontsize=10)
            ax.set_ylabel('dBFS', fontsize=10)

        # Title
        title_suffix = f'({M}x {"Jointed" if co_avg else "Averaged"})' if M > 1 else ''
        ax.set_title(f'Power Spectrum {title_suffix}', fontsize=12)

    metrics = {
        'enob': enob,
        'sndr_db': sndr_db,
        'sfdr_db': sfdr_db,
        'snr_db': snr_db,
        'thd_db': thd_db,
        'sig_pwr_dbfs': sig_pwr_dbfs,
        'noise_floor_db': noise_floor_db,
        'nsd_dbfs_hz': nsd_dbfs_hz
    }

    return metrics