"""
Error histogram analysis for ADC output.

Matches MATLAB errHistSine.m exactly for jitter detection.
"""

import numpy as np
import matplotlib.pyplot as plt


def err_hist_sine(data, bin=100, fin=0, disp=1, mode=0, erange=None, polyorder=0):
    """
    Error histogram analysis - matches MATLAB errHistSine.m exactly.

    Args:
        data: ADC output data (1D array)
        bin: Number of bins (default: 100)
        fin: Normalized frequency (0-1), 0 = auto detect (default: 0)
        disp: Display plots (1=yes, 0=no) (default: 1)
        mode: 0=phase domain, >=1=code domain (default: 0)
        erange: Error range filter [min, max] (default: None)
        polyorder: Polynomial order for transfer function fit in code mode (default: 0)

    Returns:
        emean: Mean error per bin
        erms: RMS error per bin
        phase_code: Phase positions (deg) or code positions
        anoi: Amplitude noise (reference noise)
        pnoi: Phase noise (phase jitter in radians)
        err: Raw error signal
        xx: x-axis values corresponding to raw error
        polycoeff: Polynomial coefficients (code mode only)
        k1_static: Linear coefficient (code mode with polyorder>0)
        k2_static: Quadratic coefficient (code mode with polyorder>=2)
        k3_static: Cubic coefficient (code mode with polyorder>=3)
    """
    fig = None
    # Ensure data is row vector
    data = np.asarray(data).flatten()
    N = len(data)

    # Sine fit to get ideal signal and error
    from ..common.sine_fit import sine_fit
    if fin == 0:
        data_fit, fin, mag, dc, phi = sine_fit(data)
    else:
        data_fit, _, mag, dc, phi = sine_fit(data, fin)

    # MATLAB line 38: err = data_fit - data
    err = data_fit - data

    if mode >= 1:
        # Code mode - for INL/DNL and static nonlinearity analysis
        xx = data
        dat_min = np.min(data)
        dat_max = np.max(data)
        bin_wid = (dat_max - dat_min) / bin
        phase_code = dat_min + np.arange(1, bin+1) * bin_wid - bin_wid/2

        enum = np.zeros(bin)
        esum = np.zeros(bin)
        erms = np.zeros(bin)

        # MATLAB lines 55-59: binning for mean
        for ii in range(N):
            b = min(int(np.floor((data[ii] - dat_min) / bin_wid)), bin-1)
            esum[b] += err[ii]
            enum[b] += 1

        # MATLAB line 60
        emean = esum / enum

        # MATLAB lines 61-64: binning for RMS (total RMS from sine fit)
        for ii in range(N):
            b = min(int(np.floor((data[ii] - dat_min) / bin_wid)), bin-1)
            erms[b] += err[ii]**2  # Total RMS, not relative to mean

        # MATLAB line 65
        erms = np.sqrt(erms / enum)

        anoi = np.nan
        pnoi = np.nan

        # Static nonlinearity extraction using transfer function fit
        # MATLAB lines 70-99
        polycoeff = []
        k1_static = np.nan
        k2_static = np.nan
        k3_static = np.nan

        if polyorder > 0:
            # Extract transfer function: y = k1*x + k2*x^2 + k3*x^3 + ...
            # where x = ideal input, y = actual output
            x_ideal = data_fit - np.mean(data_fit)  # Zero-mean ideal input
            y_actual = data - np.mean(data)         # Zero-mean actual output

            # Normalize for numerical stability
            x_max = np.max(np.abs(x_ideal))
            x_norm = x_ideal / x_max

            # Fit polynomial to transfer function
            if len(x_norm) > polyorder + 1:
                polycoeff = np.polyfit(x_norm, y_actual, polyorder)

                # Extract denormalized coefficients
                k1_static = polycoeff[-2] / x_max
                if polyorder >= 2:
                    k2_static = polycoeff[-3] / (x_max**2)
                if polyorder >= 3:
                    k3_static = polycoeff[-4] / (x_max**3)

        if erange is not None:
            eid = (xx >= erange[0]) & (xx <= erange[1])
            xx = xx[eid]
            err = err[eid]

        # Plotting for code mode
        if disp:
            fig = plt.figure(figsize=(10, 8))
            ax1 = plt.subplot(2, 1, 1)
            ax2 = plt.subplot(2, 1, 2)

            ax1.plot(data, err, 'r.', markersize=2, label='Raw error')
            ax1.plot(phase_code, emean, 'b-', linewidth=2, label='Mean error')

            # Add fitted transfer function error curve if polyorder > 0
            if len(polycoeff) > 0 and polyorder > 0:
                x_fit_plot = np.linspace(np.min(x_ideal), np.max(x_ideal), 200)
                y_fit_plot = np.polyval(polycoeff, x_fit_plot / x_max)
                # Convert back to code domain for plotting
                code_fit_plot = x_fit_plot + np.mean(data)
                err_fit_plot = x_fit_plot - y_fit_plot  # err = ideal - actual
                ax1.plot(code_fit_plot, err_fit_plot, 'g-', linewidth=2, label='Fitted error curve')

            ax1.set_xlim([dat_min, dat_max])
            ax1.set_ylim([np.min(err), np.max(err)])
            ax1.set_ylabel('error')
            ax1.set_xlabel('code')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)

            if erange is not None:
                ax1.plot(xx, err, 'm.', markersize=2)

            ax2.bar(phase_code, erms, width=bin_wid*0.8, color='skyblue')
            ax2.set_xlim([dat_min, dat_max])
            ax2.set_ylim([0, np.max(erms)*1.1])
            ax2.set_xlabel('code')
            ax2.set_ylabel('RMS error')
            ax2.grid(True, alpha=0.3)

            # Add text annotation with extracted coefficients
            if len(polycoeff) > 0 and polyorder > 0:
                ax2.text(dat_min + (dat_max - dat_min) * 0.02, np.max(erms) * 1.05,
                        f'k1={k1_static:.6f}, k2={k2_static:.6f}, k3={k3_static:.6f}',
                        fontsize=14, verticalalignment='top')

            plt.tight_layout()

    else:
        # Phase mode (MATLAB lines 149-246) - THIS IS CRITICAL FOR JITTER DETECTION

        # MATLAB line 150: xx = mod(phi/pi*180 + (0:length(data)-1)*fin*360,360);
        xx = np.mod(phi/np.pi*180 + np.arange(N)*fin*360, 360)

        # MATLAB line 151: phase_code = (0:bin-1)/bin*360;
        phase_code = np.arange(bin) / bin * 360

        enum = np.zeros(bin)
        esum = np.zeros(bin)
        erms = np.zeros(bin)

        # Not applicable in phase mode
        polycoeff = []
        k1_static = np.nan
        k2_static = np.nan
        k3_static = np.nan

        # MATLAB lines 160-164: binning
        for ii in range(N):
            # MATLAB line 161: b = mod(round(xx(ii)/360*bin),bin)+1;
            b = int(np.mod(np.round(xx[ii]/360*bin), bin))
            esum[b] += err[ii]
            enum[b] += 1

        # MATLAB line 165: emean = esum./enum (allows NaN for empty bins)
        with np.errstate(divide='ignore', invalid='ignore'):
            emean = esum / enum  # Will create NaN for empty bins (enum=0)

        # MATLAB lines 166-169: RMS calculation (total RMS from sine fit)
        for ii in range(N):
            b = int(np.mod(np.round(xx[ii]/360*bin), bin))
            erms[b] += err[ii]**2  # Total RMS, not relative to mean

        # MATLAB line 170: erms = sqrt(erms./enum)
        with np.errstate(divide='ignore', invalid='ignore'):
            erms = np.sqrt(erms / enum)  # Will create NaN for empty bins

        # ========== CRITICAL: Amplitude/Phase Noise Decomposition ==========
        # MATLAB lines 173-206: Robust fallback logic

        # MATLAB line 173: asen = abs(cos(phase_code/360*2*pi)).^2;
        asen = np.abs(np.cos(phase_code/360*2*np.pi))**2

        # MATLAB line 174: psen = abs(sin(phase_code/360*2*pi)).^2;
        psen = np.abs(np.sin(phase_code/360*2*np.pi))**2

        # Filter out NaN values (empty bins) before least squares fit
        valid_mask = ~np.isnan(erms)
        erms_squared = erms[valid_mask]**2

        # MATLAB line 176: Try full fit first [asen', psen', ones]
        # Use scipy's lstsq with 'gelsd' (SVD) driver to match MATLAB's behavior for rank-deficient systems
        try:
            from scipy import linalg as sp_linalg
            A_full = np.column_stack([asen[valid_mask], psen[valid_mask], np.ones(np.sum(valid_mask))])
            tmp, residuals, rank, s = sp_linalg.lstsq(A_full, erms_squared, lapack_driver='gelsd')
        except ImportError:
            # Fallback to numpy if scipy not available
            A_full = np.column_stack([asen[valid_mask], psen[valid_mask], np.ones(np.sum(valid_mask))])
            tmp = np.linalg.lstsq(A_full, erms_squared, rcond=None)[0]

        # MATLAB lines 178-179
        anoi = np.sqrt(tmp[0]) if tmp[0] >= 0 and np.isreal(tmp[0]) else -1
        pnoi = np.sqrt(tmp[1]) / mag if tmp[1] >= 0 and np.isreal(tmp[1]) else -1
        ermsbl = tmp[2]

        # MATLAB lines 182-193: If anoi fails, try phase-only fit
        if anoi < 0 or np.imag(anoi) != 0:
            A_phase = np.column_stack([psen[valid_mask], np.ones(np.sum(valid_mask))])
            try:
                tmp = sp_linalg.lstsq(A_phase, erms_squared, lapack_driver='gelsd')[0]
            except:
                tmp = np.linalg.lstsq(A_phase, erms_squared, rcond=None)[0]
            anoi = 0
            pnoi = np.sqrt(tmp[0]) / mag if tmp[0] >= 0 and np.isreal(tmp[0]) else -1
            ermsbl = tmp[1]

            # If phase-only also fails, fallback to mean baseline
            if pnoi < 0 or np.imag(pnoi) != 0:
                anoi = 0
                pnoi = 0
                ermsbl = np.mean(erms_squared)

        # MATLAB lines 195-206: If pnoi fails, try amplitude-only fit
        if pnoi < 0 or np.imag(pnoi) != 0:
            A_amp = np.column_stack([asen[valid_mask], np.ones(np.sum(valid_mask))])
            try:
                tmp = sp_linalg.lstsq(A_amp, erms_squared, lapack_driver='gelsd')[0]
            except:
                tmp = np.linalg.lstsq(A_amp, erms_squared, rcond=None)[0]
            pnoi = 0
            anoi = np.sqrt(tmp[0]) if tmp[0] >= 0 and np.isreal(tmp[0]) else -1
            ermsbl = tmp[1]

            # If amplitude-only also fails, fallback to mean baseline
            if anoi < 0 or np.imag(anoi) != 0:
                anoi = 0
                pnoi = 0
                ermsbl = np.mean(erms_squared)

        # Ensure real values
        anoi = float(np.real(anoi))
        pnoi = float(np.real(pnoi))

        # Filter error range if specified
        if erange is not None:
            eid = (xx >= erange[0]) & (xx <= erange[1])
            xx = xx[eid]
            err = err[eid]

        # Plotting for phase mode
        if disp:
            fig = plt.figure(figsize=(10, 8))
            ax1 = plt.subplot(2, 1, 1)
            ax2 = plt.subplot(2, 1, 2)

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

            # Compute fitted curves with proper handling of negative values
            amp_fit = anoi**2 * asen + ermsbl
            phase_fit = pnoi**2 * psen * mag**2 + ermsbl

            # Only plot where values are non-negative
            ax2.plot(phase_code, np.sqrt(np.maximum(amp_fit, 0)), 'b-', linewidth=2)
            ax2.plot(phase_code, np.sqrt(np.maximum(phase_fit, 0)), 'r-', linewidth=2)
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

            # Note: Figure is left open for caller to save/close
            # (tests need to save the figure before closing)

    return emean, erms, phase_code, anoi, pnoi, err, xx, polycoeff, k1_static, k2_static, k3_static


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
