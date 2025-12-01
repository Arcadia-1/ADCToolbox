import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows
from ..common.alias import alias

# Verified
def spec_plot(data, Fs=1.0, maxCode=None, harmonic=7, winType=1,
                 sideBin=1, logSca=0, label=1, assumedSignal=np.nan, isPlot=1,
                 nTHD=5, OSR=1, coAvg=0, NFMethod=0):
    """
    Precise Python port of specPlot.m

    Parameters:
        data: Input data, can be 1D (N,), 2D row (1,N), 2D column (N,1), or batch (M,N)
        Fs: Sampling frequency (default 1.0)
        maxCode: Maximum code range (default: max-min)
        harmonic: Number of harmonics to mark (default 7)
        winType: Window type, 0=boxcar, 1=hann (default 1 to match MATLAB)
        sideBin: Side bins for signal power (default 1)
        logSca: Use log scale for x-axis (default 0)
        label: Show labels (default 1)
        assumedSignal: Assumed signal power in dB (default NaN)
        isPlot: Generate plot (default 1)
        nTHD: Number of harmonics for THD calculation (default 5)
        OSR: Oversampling ratio (default 1)
        coAvg: Coherent averaging mode (default 0)
        NFMethod: Noise floor calculation method (default 0)
            0 = median-based estimation (assumes normal distribution)
            1 = trimmed mean (5%-95% range)
            2 = sum after removing harmonics
    """
    # --- Parameter processing ---
    data = np.asarray(data)
    fig = None

    # Handle different input shapes to match MATLAB behavior
    if data.ndim == 1:
        data = data.reshape(1, -1)  # (N,) -> (1, N)
    elif data.ndim == 2:
        N_rows, M_cols = data.shape
        if M_cols == 1 and N_rows > 1:
            data = data.T  # (N, 1) -> (1, N) - transpose column to row

    M, N = data.shape  # M = number of runs, N = samples per run

    if maxCode is None:
        maxCode = np.max(data) - np.min(data) if N > 0 else 1

    Nd2 = N // 2
    freq = np.arange(Nd2) / N * Fs

    # --- Window function selection (default: Hann to match MATLAB) ---
    if winType == 0:
        win = windows.boxcar(N)
    elif winType == 1:
        win = windows.hann(N, sym=False)  # 'periodic' in MATLAB -> sym=False
    else:
        win = windows.boxcar(N)

    # In-band limit for OSR
    Nd2_inband = N // 2 // OSR

    # --- Spectrum calculation ---
    spec = np.zeros(N)
    ME = 0
    for iter in range(M):
        tdata = data[iter, :]
        if np.sqrt(np.mean(tdata**2)) == 0:
            continue
        
        tdata = tdata / maxCode
        tdata = tdata - np.mean(tdata)
        tdata = tdata * win / np.sqrt(np.mean(win**2))
        spec = spec + np.abs(np.fft.fft(tdata))**2
        ME += 1

    spec = spec[:Nd2]
    spec[0] = 0  # Ignore DC
    if ME > 0:
        spec = spec / (N**2) * 16 / ME

    # --- Find fundamental power (use in-band spectrum like MATLAB) ---
    spec_inband_search = spec[:Nd2_inband]
    bin_ = np.argmax(spec_inband_search)

    start = max(bin_ - sideBin, 0)
    end = min(bin_ + sideBin + 1, Nd2_inband)
    sig = np.sum(spec[start:end])
    pwr = 10 * np.log10(sig) if sig > 0 else -999

    if not np.isnan(assumedSignal):
        sig = 10**(assumedSignal / 10)
        pwr = assumedSignal

    # --- Plotting (optional) ---
    h = None
    if isPlot:
        # Only create new figure if one doesn't exist
        if plt.get_fignums() == []:
            fig = plt.figure(figsize=(12, 8))

        if logSca == 0:
            h, = plt.plot(freq, 10 * np.log10(spec.clip(1e-20)))
        else:
            h, = plt.semilogx(freq, 10 * np.log10(spec.clip(1e-20)))

        plt.grid(True, which='both', linestyle='--')

        if label:
            # Highlight fundamental signal bins
            if logSca == 0:
                plt.plot(freq[start:end], 10 * np.log10(spec[start:end].clip(1e-20)), 'r-', linewidth=0.5)
                plt.plot(freq[bin_], 10 * np.log10(spec[bin_].clip(1e-20)), 'ro', linewidth=0.5)
            else:
                plt.semilogx(freq[start:end], 10 * np.log10(spec[start:end].clip(1e-20)), 'r-', linewidth=0.5)

        if label and harmonic > 0:
            for i in range(2, harmonic + 1):
                b = alias(bin_ * i, N)  # Python bin_ is 0-based = MATLAB's (bin-1)
                if b < len(spec):
                    plt.plot(b / N * Fs, 10 * np.log10(spec[b] + 1e-20), 'rs')
                    plt.text(b / N * Fs, 10 * np.log10(spec[b] + 1e-20) + 5, str(i),
                             fontname='Arial', fontsize=12, ha='center')

    # --- Performance metrics calculation ---
    # Save single-bin signal value for SFDR and THD (matches MATLAB 'sigs')
    sigs = spec[bin_]
    if not np.isnan(assumedSignal):
        sigs = 10**(assumedSignal / 10)

    # Remove fundamental
    spec_no_sig = np.copy(spec)
    spec_no_sig[start:end] = 0
    spec_no_sig[:sideBin] = 0  # Also zero first sideBin bins like MATLAB

    # Use in-band spectrum for calculations
    spec_inband = spec_no_sig[:Nd2_inband]
    noi = np.sum(spec_inband)

    # SFDR - use single bin spur value (not sum)
    spur = np.max(spec_inband)
    sbin = np.argmax(spec_inband)

    # Mark max spur on plot (before clearing spec_no_sig)
    if isPlot and label:
        plt.plot(sbin / N * Fs, 10 * np.log10(spur + 1e-20), 'rd')
        plt.text(sbin / N * Fs, 10 * np.log10(spur + 1e-20) + 5, 'MaxSpur',
                 fontname='Arial', fontsize=10, ha='center')

    SNDR = 10 * np.log10(sig / noi) if noi > 0 else 999
    SFDR = 10 * np.log10(sigs / spur) if spur > 0 else 999
    ENoB = (SNDR - 1.76) / 6.02

    # Calculate noise floor for SNR based on NFMethod
    # This must match MATLAB's specPlot.m lines 198-210
    if NFMethod == 0:
        # Method 0: Median-based estimation (assumes normal distribution)
        # MATLAB: noi = median(spec(1:floor(N_fft/2/OSR)))/sqrt((1-2/(9*N_run))^3) *floor(N_fft/2/OSR);
        spec_for_nf = spec_no_sig[:Nd2_inband]
        median_val = np.median(spec_for_nf)
        noi_for_snr = median_val / np.sqrt((1 - 2/(9*ME))**3) * Nd2_inband
    elif NFMethod == 1:
        # Method 1: Trimmed mean (5%-95% range)
        # MATLAB: spec_sort = sort(...); noi = mean(spec_sort(floor(N_fft/2/OSR*0.05):floor(N_fft/2/OSR*0.95)))*floor(N_fft/2/OSR);
        spec_for_nf = spec_no_sig[:Nd2_inband]
        spec_sorted = np.sort(spec_for_nf)
        idx_start = int(np.floor(Nd2_inband * 0.05))
        idx_end = int(np.floor(Nd2_inband * 0.95))
        noi_for_snr = np.mean(spec_sorted[idx_start:idx_end]) * Nd2_inband
    else:
        # Method 2: Sum after removing harmonics
        # MATLAB: for i = 2:nTHD ... spec_noise(b) = 0; end; noi = sum(spec_noise(...));
        spec_noise = np.copy(spec_no_sig)
        for i in range(2, nTHD + 1):
            b = alias(bin_ * i, N)  # Python bin_ is 0-based
            if b < Nd2_inband:
                spec_noise[b] = 0
        noi_for_snr = np.sum(spec_noise[:Nd2_inband])

    # Calculate THD
    thd = 0
    for i in range(2, nTHD + 1):
        b = alias(bin_ * i, N)  # Python bin_ is 0-based = MATLAB's (bin-1)
        if b < Nd2_inband:
            thd += spec_no_sig[b]

    THD = 10 * np.log10(thd / sigs) if sigs > 0 else -999
    SNR = 10 * np.log10(sig / noi_for_snr) if noi_for_snr > 0 else 999
    NF = SNR - pwr

    # --- Plot annotations ---
    if isPlot:
        # Calculate axis limits
        minx = min(max(np.median(10*np.log10(spec_inband+1e-20))-20, -200), -40)
        plt.xlim(Fs/N, Fs/2)
        plt.ylim(minx, 0)

        if label:
            # OSR boundary line (vertical dashed line at Fs/2/OSR)
            plt.plot([Fs/2/OSR, Fs/2/OSR], [0, -1000], '--', color='gray', linewidth=1)

            # Determine text position based on fundamental bin location
            if OSR > 1:
                TX = 10**(np.log10(Fs)*0.01 + np.log10(Fs/N)*0.99)
            else:
                if bin_/N < 0.2:
                    TX = Fs * 0.3
                else:
                    TX = Fs * 0.01

            TYD = minx * 0.06

            # Format frequency text helper
            def format_freq(f):
                if f >= 1e9:
                    return f'{f/1e9:.1f}G'
                elif f >= 1e6:
                    return f'{f/1e6:.1f}M'
                elif f >= 1e3:
                    return f'{f/1e3:.1f}K'
                elif f >= 1:
                    return f'{f:.1f}'
                else:
                    return f'{f:.3f}'

            txt_fs = format_freq(Fs)
            Fin = bin_/N * Fs

            # Special case for Fin formatting (matches MATLAB line 259)
            if Fin >= 1e9:
                txt_fin = f'{Fin/1e9:.1f}G'
            elif Fin >= 1e6:
                txt_fin = f'{Fin/1e6:.1f}M'
            elif Fin >= 1e3:
                txt_fin = f'{Fin/1e3:.1f}K'
            elif Fin >= 1:
                txt_fin = f'{Fin/1e3:.1f}'  # Note: /1e3 for values >= 1 Hz
            else:
                txt_fin = f'{bin_/N*Fs:.3f}'

            # NSD calculation
            NSD = NF + 10*np.log10(Fs/2/OSR)

            # Metric annotations (left side)
            plt.text(TX, TYD, f'Fin/Fs = {txt_fin} / {txt_fs} Hz', fontsize=10)
            plt.text(TX, TYD*2, f'ENoB = {ENoB:.2f}', fontsize=10)
            plt.text(TX, TYD*3, f'SNDR = {SNDR:.2f} dB', fontsize=10)
            plt.text(TX, TYD*4, f'SFDR = {SFDR:.2f} dB', fontsize=10)
            plt.text(TX, TYD*5, f'THD = {THD:.2f} dB', fontsize=10)
            plt.text(TX, TYD*6, f'SNR = {SNR:.2f} dB', fontsize=10)
            plt.text(TX, TYD*7, f'Noise Floor = {NF:.2f} dB', fontsize=10)
            plt.text(TX, TYD*8, f'NSD = {NSD:.2f} dBFS/Hz', fontsize=10)

            # Noise floor baseline (red dashed horizontal line)
            nf_level = -(NF + 10*np.log10(N/2/OSR))
            if OSR > 1:
                plt.semilogx([Fs/N, Fs/2/OSR], [nf_level, nf_level], 'r--', linewidth=1)
                plt.text(TX, TYD*9, f'OSR = {OSR:.2f}', fontsize=10)
            else:
                plt.plot([0, Fs/2], [nf_level, nf_level], 'r--', linewidth=1)

            # Signal annotation (near fundamental peak)
            if OSR > 1:
                plt.text(freq[bin_], min(pwr, TYD/2), f'Sig = {pwr:.2f} dB', fontsize=10)
            else:
                if bin_/N > 0.4:
                    plt.text((bin_/N - 0.01) * Fs, min(pwr, TYD/2), f'Sig = {pwr:.2f} dB',
                             ha='right', fontsize=10)
                else:
                    plt.text((bin_/N + 0.01) * Fs, min(pwr, TYD/2), f'Sig = {pwr:.2f} dB', fontsize=10)

            plt.xlabel('Freq (Hz)', fontsize=10)
            plt.ylabel('dBFS', fontsize=10)

        # Title
        if M > 1:
            if coAvg:
                plt.title(f'Power Spectrum ({M}x Jointed)', fontsize=12)
            else:
                plt.title(f'Power Spectrum ({M}x Averanged)', fontsize=12)
        else:
            plt.title('Power Spectrum', fontsize=12)

    if not isPlot:
        h = None

    # Close figure to prevent memory leak
    if isPlot and fig is not None:
        plt.close(fig)

    return ENoB, SNDR, SFDR, SNR, THD, pwr, NF, h