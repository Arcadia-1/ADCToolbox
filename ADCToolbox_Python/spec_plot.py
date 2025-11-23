import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows
from .alias import alias

#已验证
def spec_plot(data, Fs=1.0, maxCode=None, harmonic=7, winType=1,
                 sideBin=1, logSca=0, label=1, assumedSignal=np.nan, isPlot=1,
                 nTHD=5, OSR=1):
    """
    specPlot.m 的精确 Python 移植版本

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
    """
    # --- 参数处理 ---
    data = np.asarray(data)

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

    # --- 窗函数选择 (default: Hann to match MATLAB) ---
    if winType == 0:
        win = windows.boxcar(N)
    elif winType == 1:
        win = windows.hann(N, sym=False)  # 'periodic' in MATLAB -> sym=False
    else:
        win = windows.boxcar(N)

    # In-band limit for OSR
    Nd2_inband = N // 2 // OSR

    # --- 频谱计算 ---
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
    spec[0] = 0 # 忽略直流
    if ME > 0:
        spec = spec / (N**2) * 16 / ME

    # --- 查找基波功率 (use in-band spectrum like MATLAB) ---
    spec_inband_search = spec[:Nd2_inband]
    bin_ = np.argmax(spec_inband_search)

    start = max(bin_ - sideBin, 0)
    end = min(bin_ + sideBin + 1, Nd2_inband)
    sig = np.sum(spec[start:end])
    pwr = 10 * np.log10(sig) if sig > 0 else -999

    if not np.isnan(assumedSignal):
        sig = 10**(assumedSignal / 10)
        pwr = assumedSignal

    # --- 绘图 (可选) ---
    h = None
    if isPlot:
        plt.figure(figsize=(12, 8))
        if logSca == 0:
            h, = plt.plot(freq, 10 * np.log10(spec.clip(1e-20)))
        else:
            h, = plt.semilogx(freq, 10 * np.log10(spec.clip(1e-20)))

        plt.grid(True, which='both', linestyle='--')

        if label:
            if logSca == 0:
                plt.plot(freq[start:end], 10 * np.log10(spec[start:end].clip(1e-20)), 'r-', linewidth=1.5)
            else:
                plt.semilogx(freq[start:end], 10 * np.log10(spec[start:end].clip(1e-20)), 'r-', linewidth=1.5)

        if harmonic > 0:
            for i in range(2, harmonic + 1):
                b = alias(bin_ * i, N)  # Python bin_ is 0-based = MATLAB's (bin-1)
                if b < len(spec):
                    plt.plot(b / N * Fs, 10 * np.log10(spec[b] + 1e-20), 'rs')
                    plt.text(b / N * Fs, 10 * np.log10(spec[b] + 1e-20) + 5, str(i),
                             fontname='Arial', fontsize=12, ha='center')

    # --- 性能指标计算 ---
    # Save single-bin signal value for SFDR and THD (matches MATLAB 'sigs')
    sigs = spec[bin_]
    if not np.isnan(assumedSignal):
        sigs = 10**(assumedSignal / 10)

    # 移除基波
    spec_no_sig = np.copy(spec)
    spec_no_sig[start:end] = 0
    spec_no_sig[:sideBin] = 0  # Also zero first sideBin bins like MATLAB

    # Use in-band spectrum for calculations
    spec_inband = spec_no_sig[:Nd2_inband]
    noi = np.sum(spec_inband)

    # SFDR - use single bin spur value (not sum)
    spur = np.max(spec_inband)
    sbin = np.argmax(spec_inband)

    SNDR = 10 * np.log10(sig / noi) if noi > 0 else 999
    SFDR = 10 * np.log10(sigs / spur) if spur > 0 else 999
    ENoB = (SNDR - 1.76) / 6.02

    # THD 和 SNR - match MATLAB nTHD parameter
    thd = 0
    spec_no_harm = np.copy(spec_no_sig)
    for i in range(2, nTHD + 1):
        b = alias(bin_ * i, N)  # Python bin_ is 0-based = MATLAB's (bin-1)
        if b < Nd2_inband:
            thd += spec_no_harm[b]
            spec_no_harm[b] = 0

    noi_for_snr = np.sum(spec_no_harm[:Nd2_inband])
    THD = 10 * np.log10(thd / sigs) if sigs > 0 else -999
    SNR = 10 * np.log10(sig / noi_for_snr) if noi_for_snr > 0 else 999
    NF = SNR - pwr

    # --- 绘图标注 ---
    if isPlot:
        # y_max = pwr if pwr > 0 else 0
        # axis_limits = [Fs/N, Fs/2, -120, y_max + 10]
        # plt.axis(axis_limits)
        plt.ylim(-120, (pwr if pwr > 0 else 0) + 10)
        plt.xlim(Fs/N, Fs/2)


        if label:
            ax = plt.gca()
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            
            text_str = '\n'.join([
                f'ENoB = {ENoB:.2f} bits',
                f'SNDR = {SNDR:.2f} dB',
                f'SFDR = {SFDR:.2f} dB',
                f'THD  = {THD:.2f} dB',
                f'SNR  = {SNR:.2f} dB',
                f'Noise Floor = {NF:.2f} dBFS/bin'
            ])
            ax.text(0.2, 0.99, text_str, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
            
            fund_text = f'Fund = {pwr:.2f} dBFS @ {freq[bin_]/1e3:.2f} kHz'
            plt.text(freq[bin_], pwr, fund_text, ha='center', va='bottom', fontsize=9)
            
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('dBFS')
            plt.title('Output Spectrum')

    if not isPlot:
        h = None
        
    return ENoB, SNDR, SFDR, SNR, THD, pwr, NF, h