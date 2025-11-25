"""
Two-Tone Spectrum Analyzer

Analyze intermodulation distortion with two-tone input.
Measures IMD2, IMD3 and other spectral metrics.

Ported from MATLAB: specPlot2Tone.m
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from ..common.alias import alias


def spec_plot_2tone(
    data: np.ndarray,
    fs: float = 1.0,
    max_code: Optional[float] = None,
    harmonic: int = 7,
    win_type: str = 'hann',
    side_bin: int = 1,
    is_plot: bool = True,
    save_path: Optional[str] = None
) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
    """
    Two-tone spectrum analysis with IMD calculation.

    Args:
        data: ADC output data, shape (M, N) for M runs or (N,) for single run
        fs: Sampling frequency (Hz)
        max_code: Maximum code range (default: max-min of data)
        harmonic: Number of harmonics to mark (default 7)
        win_type: Window type ('hann', 'blackman', 'hamming')
        side_bin: Side bins to include in signal power (default 1)
        is_plot: Whether to plot (default True)
        save_path: Path to save figure (optional)

    Returns:
        Tuple of:
            - ENoB: Effective number of bits
            - SNDR: Signal to noise and distortion ratio (dB)
            - SFDR: Spurious free dynamic range (dB)
            - SNR: Signal to noise ratio (dB)
            - THD: Total harmonic distortion (dB)
            - pwr1: Power of first tone (dBFS)
            - pwr2: Power of second tone (dBFS)
            - NF: Noise floor (dB)
            - IMD2: 2nd order intermodulation distortion (dB)
            - IMD3: 3rd order intermodulation distortion (dB)
    """
    data = np.asarray(data)

    # Handle 1D or 2D input
    if data.ndim == 1:
        data = data.reshape(1, -1)

    m_runs, n = data.shape

    if max_code is None:
        max_code = np.max(data) - np.min(data)

    nd2 = n // 2
    freq = np.arange(nd2) / n * fs

    # Create window
    if win_type == 'hann':
        win = np.hanning(n)
    elif win_type == 'blackman':
        win = np.blackman(n)
    elif win_type == 'hamming':
        win = np.hamming(n)
    else:
        win = np.hanning(n)

    # Average spectrum over multiple runs
    spec = np.zeros(n)
    me = 0

    for i in range(m_runs):
        tdata = data[i, :]
        if np.std(tdata) == 0:
            continue

        tdata = tdata / max_code
        tdata = tdata - np.mean(tdata)
        tdata = tdata * win / np.sqrt(np.mean(win**2))

        spec += np.abs(np.fft.fft(tdata))**2
        me += 1

    if me == 0:
        raise ValueError("No valid data runs")

    spec = spec[:nd2]
    spec[:side_bin] = 0
    spec = spec / (n**2) * 16 / me

    # Find two tones
    bin1 = np.argmax(spec)
    t_spec = spec.copy()
    t_spec[bin1] = 0
    bin2 = np.argmax(t_spec)

    # Ensure bin1 < bin2
    if bin1 > bin2:
        bin1, bin2 = bin2, bin1

    # Calculate signal powers
    sig1 = np.sum(spec[max(bin1 - side_bin, 0):min(bin1 + side_bin + 1, nd2)])
    sig2 = np.sum(spec[max(bin2 - side_bin, 0):min(bin2 + side_bin + 1, nd2)])
    pwr1 = 10 * np.log10(sig1 + 1e-20)
    pwr2 = 10 * np.log10(sig2 + 1e-20)

    # Remove signal bins for noise calculation
    spec_noise = spec.copy()
    spec_noise[max(bin1 - side_bin, 0):min(bin1 + side_bin + 1, nd2)] = 0
    spec_noise[max(bin2 - side_bin, 0):min(bin2 + side_bin + 1, nd2)] = 0

    noi = np.sum(spec_noise)

    # Find max spur
    sbin = np.argmax(spec_noise)
    spur = np.sum(spec_noise[max(sbin - side_bin, 0):min(sbin + side_bin + 1, nd2)])

    # Calculate metrics
    SNDR = 10 * np.log10((sig1 + sig2) / noi)
    SFDR = 10 * np.log10((sig1 + sig2) / (spur + 1e-20))
    ENoB = (SNDR - 1.76) / 6.02

    # IMD2: f1+f2 and f2-f1
    b_imd2_sum = alias(bin1 + bin2, n)  # Python bins are 0-based
    b_imd2_diff = alias(bin2 - bin1, n)
    spur21 = np.sum(spec[max(b_imd2_sum, 0):min(b_imd2_sum + 3, nd2)])
    spur22 = np.sum(spec[max(b_imd2_diff, 0):min(b_imd2_diff + 3, nd2)])
    IMD2 = 10 * np.log10((sig1 + sig2) / (spur21 + spur22 + 1e-20))

    # IMD3: 2f1+f2, f1+2f2, 2f1-f2, 2f2-f1
    b31 = alias(2 * bin1 + bin2, n)
    b32 = alias(bin1 + 2 * bin2, n)
    b33 = alias(2 * bin1 - bin2, n)
    b34 = alias(2 * bin2 - bin1, n)

    spur31 = np.sum(spec[max(b31, 0):min(b31 + 3, nd2)])
    spur32 = np.sum(spec[max(b32, 0):min(b32 + 3, nd2)])
    spur33 = np.sum(spec[max(b33, 0):min(b33 + 3, nd2)])
    spur34 = np.sum(spec[max(b34, 0):min(b34 + 3, nd2)])
    IMD3 = 10 * np.log10((sig1 + sig2) / (spur31 + spur32 + spur33 + spur34 + 1e-20))

    # THD calculation (interleaved harmonics)
    thd = 0
    spec_thd = spec_noise.copy()
    for i in range(2, n // 100 + 1):
        b = alias(bin2 + (bin2 - bin1) * (i - 1), n)  # Python bins are 0-based
        thd += np.sum(spec_thd[max(b, 0):min(b + 3, nd2)])
        spec_thd[max(b, 0):min(b + 3, nd2)] = 0

        b = alias(bin1 - (bin2 - bin1) * (i - 1), n)
        if b >= 0:
            thd += np.sum(spec_thd[max(b, 0):min(b + 3, nd2)])
            spec_thd[max(b, 0):min(b + 3, nd2)] = 0

    noi_final = np.sum(spec_thd)
    THD = 10 * np.log10(thd / (sig1 + sig2 + 1e-20))
    SNR = 10 * np.log10((sig1 + sig2) / (noi_final + 1e-20))
    NF = SNR - 10 * np.log10(sig1 + sig2)

    # Plot
    if is_plot:
        fig, ax = plt.subplots(figsize=(10, 6))

        spec_db = 10 * np.log10(spec + 1e-20)
        ax.plot(freq, spec_db, 'b-', linewidth=0.5)

        # Mark tones
        ax.plot(freq[bin1], spec_db[bin1], 'ro', markersize=8)
        ax.plot(freq[bin2], spec_db[bin2], 'ro', markersize=8)

        # Mark harmonics and IMD products
        if harmonic > 0:
            for i in range(2, harmonic + 1):
                for j in range(i + 1):
                    # i-j, j combinations
                    b = alias(bin1 * j + bin2 * (i - j), n)  # Python bins are 0-based
                    if 0 < b < nd2:
                        ax.plot(b / n * fs, spec_db[b], 'rs', markersize=4)
                        ax.text(b / n * fs, spec_db[b] + 3, str(i),
                               fontsize=8, ha='center')

        # Add metrics text
        mins = np.min(spec_db[spec_db > -200])
        ax.text(fs / n * 2, mins * 0.05, f'Fs = {fs/1e6:.1f} MHz' if fs > 1e6 else f'Fs = {fs/1e3:.1f} kHz')
        ax.text(fs / n * 2, mins * 0.10, f'ENoB = {ENoB:.2f}')
        ax.text(fs / n * 2, mins * 0.15, f'SNDR = {SNDR:.2f} dB')
        ax.text(fs / n * 2, mins * 0.20, f'SFDR = {SFDR:.2f} dB')
        ax.text(fs / n * 2, mins * 0.25, f'SNR = {SNR:.2f} dB')
        ax.text(fs / n * 2, mins * 0.30, f'NF = {NF:.2f} dB')
        ax.text(fs / n * 2, mins * 0.35, f'IMD2 = {IMD2:.2f} dB')
        ax.text(fs / n * 2, mins * 0.40, f'IMD3 = {IMD3:.2f} dB')

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('dBFS')
        ax.set_title('Two-Tone Output Spectrum')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([fs / n, fs / 2])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[specPlot2Tone] Figure saved to: {save_path}")

        plt.close()

    return ENoB, SNDR, SFDR, SNR, THD, pwr1, pwr2, NF, IMD2, IMD3


if __name__ == "__main__":
    print("=" * 60)
    print("Testing specPlot2Tone.py")
    print("=" * 60)

    # Generate two-tone test signal
    n = 4096
    fs = 1e6
    f1 = fs / n * 101  # Prime bins for coherent sampling
    f2 = fs / n * 131

    t = np.arange(n) / fs
    # Two tones with some IMD (simulated nonlinearity)
    signal = 0.4 * np.sin(2 * np.pi * f1 * t) + 0.4 * np.sin(2 * np.pi * f2 * t)

    # Add some nonlinearity (generates IMD)
    signal = signal + 0.01 * signal**2 + 0.005 * signal**3

    # Add noise
    signal += np.random.randn(n) * 0.001

    # Quantize (12-bit)
    signal = np.round(signal * 2048) / 2048

    print(f"\n[Test] Two-tone: f1={f1/1e3:.1f}kHz, f2={f2/1e3:.1f}kHz")

    results = spec_plot_2tone(
        signal, fs,
        is_plot=True,
        save_path='../output_data/test_2tone_spectrum.png'
    )

    ENoB, SNDR, SFDR, SNR, THD, pwr1, pwr2, NF, IMD2, IMD3 = results

    print(f"\n[Results]")
    print(f"  ENoB = {ENoB:.2f} bits")
    print(f"  SNDR = {SNDR:.2f} dB")
    print(f"  SFDR = {SFDR:.2f} dB")
    print(f"  SNR = {SNR:.2f} dB")
    print(f"  THD = {THD:.2f} dB")
    print(f"  Tone 1 power = {pwr1:.2f} dBFS")
    print(f"  Tone 2 power = {pwr2:.2f} dBFS")
    print(f"  Noise Floor = {NF:.2f} dB")
    print(f"  IMD2 = {IMD2:.2f} dB")
    print(f"  IMD3 = {IMD3:.2f} dB")

    print("\n" + "=" * 60)
