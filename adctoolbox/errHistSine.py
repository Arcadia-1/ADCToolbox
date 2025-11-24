"""
Error histogram analysis for ADC output.

Matches MATLAB errHistSine.m exactly for jitter detection.
"""

import numpy as np
import matplotlib.pyplot as plt


def errHistSine(data, bin=100, fin=0, disp=1, mode=0, erange=None):
    """
    Error histogram analysis - matches MATLAB errHistSine.m exactly.

    Args:
        data: ADC output data (1D array)
        bin: Number of bins (default: 100)
        fin: Normalized frequency (0-1), 0 = auto detect (default: 0)
        disp: Display plots (1=yes, 0=no) (default: 1)
        mode: 0=phase domain, >=1=code domain (default: 0)
        erange: Error range filter [min, max] (default: None)

    Returns:
        emean: Mean error per bin
        erms: RMS error per bin
        phase_code: Phase positions (deg) or code positions
        anoi: Amplitude noise (reference noise)
        pnoi: Phase noise (phase jitter in radians)
        err: Raw error signal
        xx: x-axis values corresponding to raw error
    """
    # Ensure data is row vector
    data = np.asarray(data).flatten()
    N = len(data)

    # Sine fit to get ideal signal and error
    if fin == 0:
        from .sineFit import sine_fit
        data_fit, fin, mag, dc, phi = sine_fit(data)
    else:
        from .sineFit import sine_fit
        data_fit, _, mag, dc, phi = sine_fit(data, fin)

    # MATLAB line 38: err = data_fit - data
    err = data_fit - data

    if mode >= 1:
        # Code mode (not used for jitter detection)
        xx = data
        dat_min = np.min(data)
        dat_max = np.max(data)
        bin_wid = (dat_max - dat_min) / bin
        phase_code = dat_min + np.arange(1, bin+1) * bin_wid - bin_wid/2

        enum = np.zeros(bin)
        esum = np.zeros(bin)
        erms = np.zeros(bin)

        for ii in range(N):
            b = min(int(np.floor((data[ii] - dat_min) / bin_wid)), bin-1)
            esum[b] += err[ii]
            enum[b] += 1

        enum = np.maximum(enum, 1)
        emean = esum / enum

        for ii in range(N):
            b = min(int(np.floor((data[ii] - dat_min) / bin_wid)), bin-1)
            erms[b] += (err[ii] - emean[b])**2

        erms = np.sqrt(erms / enum)

        anoi = np.nan
        pnoi = np.nan

        if erange is not None:
            eid = (xx >= erange[0]) & (xx <= erange[1])
            xx = xx[eid]
            err = err[eid]

        # Plotting for code mode
        if disp:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

            ax1.plot(data, err, 'r.', markersize=2)
            ax1.plot(phase_code, emean, 'b-', linewidth=2)
            ax1.set_xlim([dat_min, dat_max])
            ax1.set_ylim([np.min(err), np.max(err)])
            ax1.set_ylabel('error')
            ax1.set_xlabel('code')
            ax1.grid(True, alpha=0.3)

            if erange is not None:
                ax1.plot(xx, err, 'm.', markersize=2)

            ax2.bar(phase_code, erms, width=bin_wid*0.8, color='skyblue')
            ax2.set_xlim([dat_min, dat_max])
            ax2.set_ylim([0, np.max(erms)*1.1])
            ax2.set_xlabel('code')
            ax2.set_ylabel('RMS error')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

    else:
        # Phase mode (MATLAB lines 93-188) - THIS IS CRITICAL FOR JITTER DETECTION

        # MATLAB line 94: xx = mod(phi/pi*180 + (0:length(data)-1)*fin*360,360);
        xx = np.mod(phi/np.pi*180 + np.arange(N)*fin*360, 360)

        # MATLAB line 95: phase_code = (0:bin-1)/bin*360;
        phase_code = np.arange(bin) / bin * 360

        enum = np.zeros(bin)
        esum = np.zeros(bin)
        erms = np.zeros(bin)

        # MATLAB lines 101-105: binning
        for ii in range(N):
            # MATLAB line 102: b = mod(round(xx(ii)/360*bin),bin)+1;
            b = int(np.mod(np.round(xx[ii]/360*bin), bin))
            esum[b] += err[ii]
            enum[b] += 1

        # MATLAB line 106: emean = esum./enum (allows NaN for empty bins)
        with np.errstate(divide='ignore', invalid='ignore'):
            emean = esum / enum  # Will create NaN for empty bins (enum=0)

        # MATLAB lines 107-111: RMS calculation
        for ii in range(N):
            b = int(np.mod(np.round(xx[ii]/360*bin), bin))
            erms[b] += (err[ii] - emean[b])**2

        # MATLAB line 111: erms = sqrt(erms./enum)
        with np.errstate(divide='ignore', invalid='ignore'):
            erms = np.sqrt(erms / enum)  # Will create NaN for empty bins

        # ========== CRITICAL: Amplitude/Phase Noise Decomposition ==========
        # MATLAB lines 114-147

        # MATLAB line 114: asen = abs(cos(phase_code/360*2*pi)).^2;
        asen = np.abs(np.cos(phase_code/360*2*np.pi))**2

        # MATLAB line 115: psen = abs(sin(phase_code/360*2*pi)).^2;
        psen = np.abs(np.sin(phase_code/360*2*np.pi))**2

        # MATLAB line 117: tmp = linsolve([asen',psen',ones(bin,1)], erms'.^2);
        # Filter out NaN values (empty bins) before least squares fit
        valid_mask = ~np.isnan(erms)
        A = np.column_stack([asen[valid_mask], psen[valid_mask], np.ones(np.sum(valid_mask))])
        b_vec = erms[valid_mask]**2

        # Use non-negative least squares to ensure non-negative variance components
        # This avoids numerical issues with rank-deficient systems (asen + psen = 1)
        try:
            from scipy.optimize import nnls
            tmp, _ = nnls(A, b_vec)
        except ImportError:
            # Fallback to pseudoinverse if scipy not available
            tmp = np.linalg.pinv(A) @ b_vec

        # MATLAB line 119-121
        anoi = np.sqrt(tmp[0]) if tmp[0] >= 0 else 0
        pnoi = np.sqrt(tmp[1]) / mag if tmp[1] >= 0 else 0
        ermsbl = tmp[2]

        # Handle edge cases where fit failed
        if not np.isfinite(anoi):
            anoi = 0
        if not np.isfinite(pnoi):
            pnoi = 0

        # Filter error range if specified
        if erange is not None:
            eid = (xx >= erange[0]) & (xx <= erange[1])
            xx = xx[eid]
            err = err[eid]

        # Plotting for phase mode
        if disp:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

            # Top subplot: data and error vs phase
            ax1_left = ax1
            ax1_left.plot(xx, data, 'k.', markersize=2, label='data')
            ax1_left.set_xlim([0, 360])
            ax1_left.set_ylim([np.min(data), np.max(data)])
            ax1_left.set_ylabel('data', color='k')
            ax1_left.tick_params(axis='y', labelcolor='k')

            ax1_right = ax1.twinx()
            ax1_right.plot(xx, err, 'r.', markersize=2, alpha=0.5)
            ax1_right.plot(phase_code, emean, 'b-', linewidth=2, label='error')
            ax1_right.set_xlim([0, 360])
            ax1_right.set_ylim([np.min(err), np.max(err)])
            ax1_right.set_ylabel('error', color='r')
            ax1_right.tick_params(axis='y', labelcolor='r')

            ax1.legend(['data', 'error'], loc='upper right')
            ax1.set_xlabel('phase(deg)')
            ax1.grid(True, alpha=0.3)

            if erange is not None:
                ax1_right.plot(xx, err, 'm.', markersize=2)

            # Bottom subplot: RMS error with fitted curves
            ax2.bar(phase_code, erms, width=360/bin*0.8, color='skyblue', alpha=0.7)
            ax2.plot(phase_code, np.sqrt(anoi**2 * asen + ermsbl), 'b-', linewidth=2)
            ax2.plot(phase_code, np.sqrt(pnoi**2 * psen * mag**2 + ermsbl), 'r-', linewidth=2)
            ax2.set_xlim([0, 360])
            ax2.set_ylim([0, np.max(erms)*1.2])

            # Add text labels (MATLAB lines 183-184)
            ax2.text(10, np.max(erms)*1.15,
                    f'Normalized Amplitude Noise RMS = {anoi/mag:.2e}',
                    color='b', fontsize=10)
            ax2.text(10, np.max(erms)*1.05,
                    f'Phase Noise RMS = {pnoi:.2e} rad',
                    color='r', fontsize=10)

            ax2.set_xlabel('phase(deg)')
            ax2.set_ylabel('RMS error')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

    return emean, erms, phase_code, anoi, pnoi, err, xx


if __name__ == "__main__":
    import os

    # Test with jitter
    N = 2**14
    Fs = 10e9
    Fin = 200e6
    J = int(np.round(Fin / Fs * N))
    fin_norm = J / N
    Fin_actual = J / N * Fs

    # Generate jittered signal
    Tj = 100e-15  # 100 fs jitter
    phase_noise_rms = 2 * np.pi * Fin_actual * Tj

    Ts = 1 / Fs
    theta = 2 * np.pi * Fin_actual * np.arange(N) * Ts
    phase_jitter = np.random.randn(N) * phase_noise_rms

    data = np.sin(theta + phase_jitter) * 0.49 + 0.5 + np.random.randn(N) * 0.00001

    print(f"[Test] [Set jitter] = {Tj*1e15:.2f} fs")

    # Test errHistSine
    emean, erms, phase_code, anoi, pnoi, err, xx = errHistSine(data, bin=99, fin=fin_norm, disp=0)

    # Convert pnoi to jitter (MATLAB line 47)
    jitter_calc = pnoi / (2 * np.pi * Fin_actual)

    print(f"[Test] [Calculated jitter] = {jitter_calc*1e15:.2f} fs")
    print(f"[Test] [anoi] = {anoi:.6e}")
    print(f"[Test] [pnoi] = {pnoi:.6e} rad")
    print(f"[Test] [Ratio] = {jitter_calc/Tj:.2f}")
