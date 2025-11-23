"""
Error histogram analysis for ADC output.

- errHistPhase: Error vs phase, for detecting jitter
- errHistCode: Error vs code, for detecting INL/DNL
"""

import numpy as np
import matplotlib.pyplot as plt


def _get_error(data, fin):
    """Get error via Thompson decomposition."""
    N = len(data)

    if fin == 0:
        try:
            from .findFin import findFin
            fin = findFin(data, 1)
        except ImportError:
            spec = np.abs(np.fft.fft(data))
            spec[0] = 0
            fin = np.argmax(spec[:N//2]) / N

    try:
        from .tomDecomp import tomDecomp
        _, error, _, _, phi = tomDecomp(data, fin, order=1, disp=0)
    except ImportError:
        t = np.arange(N)
        SI, SQ = np.cos(t * fin * 2 * np.pi), np.sin(t * fin * 2 * np.pi)
        WI, WQ = np.mean(SI * data) * 2, np.mean(SQ * data) * 2
        error = data - (np.mean(data) + SI * WI + SQ * WQ)
        phi = -np.arctan2(WQ, WI)

    return fin, error, phi


def _calc_histogram(x, error, bin_count, x_min, x_max):
    """
    Calculate mean and RMS error per bin.

    Matches MATLAB errHistSine.m phase mode implementation:
    - phase_code = (0:bin-1)/bin*360;  (NOT bin centers!)
    - b = mod(round(phi_list(ii)/360*bin),bin)+1;
    """
    bin_wid = (x_max - x_min) / bin_count
    # Match MATLAB: phase_code = (0:bin-1)/bin*360 for phase domain
    phase_positions = np.arange(bin_count) / bin_count * (x_max - x_min) + x_min

    enum = np.zeros(bin_count)
    esum = np.zeros(bin_count)
    erms = np.zeros(bin_count)

    # Match MATLAB binning: b = mod(round(phi_list(ii)/360*bin),bin)+1
    for i, (xi, ei) in enumerate(zip(x, error)):
        # MATLAB: mod(round(phi_list(ii)/360*bin),bin)+1
        # Python equivalent (0-indexed):
        b = int(np.round((xi - x_min) / (x_max - x_min) * bin_count)) % bin_count
        esum[b] += ei
        enum[b] += 1

    enum = np.maximum(enum, 1)
    emean = esum / enum

    for i, (xi, ei) in enumerate(zip(x, error)):
        b = int(np.round((xi - x_min) / (x_max - x_min) * bin_count)) % bin_count
        erms[b] += (ei - emean[b])**2

    return emean, np.sqrt(erms / enum), phase_positions, bin_wid


def errHistPhase(data, bin_count=360, fin=0, disp=0, erange=None, save_path=None):
    """Phase domain error histogram. Detects jitter (peaks at 0/180 deg)."""
    data = np.asarray(data).flatten()
    fin, error, phi = _get_error(data, fin)

    phi_list = (phi / np.pi * 180 + np.arange(len(data)) * fin * 360) % 360
    emean, erms, phase, _ = _calc_histogram(phi_list, error, bin_count, 0, 360)

    edata = error[(phi_list >= erange[0]) & (phi_list <= erange[1])] if erange else None

    if disp or save_path:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        ax1.plot(phi_list, error, 'r.', markersize=2, alpha=0.8)
        ax1.plot(phase, emean, 'b-', linewidth=2)
        ax1.set_xlim([0, 360])
        ax1.set_xlabel('Phase (deg)')
        ax1.set_ylabel('Error')
        ax1.grid(True, alpha=0.3)

        ax2.bar(phase, erms, width=360/bin_count*0.8, color='skyblue')
        ax2.set_xlim([0, 360])
        ax2.set_xlabel('Phase (deg)')
        ax2.set_ylabel('RMS Error')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        if disp:
            plt.show()
        plt.close()

    return emean, erms, phase, edata


def errHistCode(data, bin_count=100, fin=0, disp=0, erange=None, save_path=None):
    """Code domain error histogram. Detects INL/DNL patterns."""
    data = np.asarray(data).flatten()
    fin, error, _ = _get_error(data, fin)

    dat_min, dat_max = np.min(data), np.max(data)
    emean, erms, codes, bin_wid = _calc_histogram(data, error, bin_count, dat_min, dat_max)

    edata = error[(data >= erange[0]) & (data <= erange[1])] if erange else None

    if disp or save_path:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        ax1.plot(data, error, 'r.', markersize=2, alpha=0.8)
        ax1.plot(codes, emean, 'b-', linewidth=2)
        ax1.set_xlim([dat_min, dat_max])
        ax1.set_xlabel('Code')
        ax1.set_ylabel('Error')
        ax1.grid(True, alpha=0.3)

        ax2.bar(codes, erms, width=bin_wid*0.8, color='skyblue')
        ax2.set_xlim([dat_min, dat_max])
        ax2.set_xlabel('Code')
        ax2.set_ylabel('RMS Error')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        if disp:
            plt.show()
        plt.close()

    return emean, erms, codes, edata


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "tests", "output")
    os.makedirs(output_dir, exist_ok=True)

    # Test data with jitter
    N = 4096
    J = 101  # cycles in N samples
    jitter_std = 0.002  # phase jitter

    # Jitter added to phase argument
    data = np.sin(np.arange(N) * J * 2 * np.pi / N + np.random.randn(N) * jitter_std) * 0.49 + 0.5 + np.random.randn(N) * 0.00001

    print(f"Test with phase jitter std = {jitter_std}")
    errHistPhase(data, fin=J/N, save_path=os.path.join(output_dir, "errHistPhase_test.png"))
    errHistCode(data, fin=J/N, save_path=os.path.join(output_dir, "errHistCode_test.png"))
    print(f"Saved to {output_dir}")
