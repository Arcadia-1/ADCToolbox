"""
Error histogram in phase domain for ADC output.

Critical for jitter detection and phase noise analysis.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_error_hist_phase(data, bins=100, freq=0, disp=1, error_range=None):
    """
    Error histogram in phase domain - for jitter detection.

    Parameters:
        data: ADC output data (1D array)
        bins: Number of bins (default: 100)
        freq: Normalized frequency (0-1), 0 = auto detect (default: 0)
        disp: Display plots (1=yes, 0=no) (default: 1)
        error_range: Error range filter [min, max] (default: None)

    Returns:
        error_mean: Mean error per bin
        error_rms: RMS error per bin
        phase_bins: Phase positions (degrees, 0-360)
        amplitude_noise: Amplitude noise (reference noise)
        phase_noise: Phase noise (phase jitter in radians)
        error: Raw error signal
        phase: Phase values (degrees) corresponding to raw error

    Notes:
        This function bins errors by phase to separate:
        - Amplitude noise (constant across phase)
        - Phase noise/jitter (varies with sin(phase))
    """
    # Ensure data is row vector
    data = np.asarray(data).flatten()
    N = len(data)

    # Sine fit to get ideal signal and error
    from ..common.sine_fit import sine_fit
    if freq == 0:
        data_fit, freq, mag, dc, phi = sine_fit(data)
    else:
        data_fit, _, mag, dc, phi = sine_fit(data, freq)

    # Error = ideal - actual
    error = data_fit - data

    # Phase mode - bin by phase
    phase = np.mod(phi/np.pi*180 + np.arange(N)*freq*360, 360)
    phase_bins = np.arange(bins) / bins * 360

    bin_count = np.zeros(bins)
    error_sum = np.zeros(bins)
    error_rms = np.zeros(bins)

    # Binning
    for ii in range(N):
        b = int(np.mod(np.round(phase[ii]/360*bins), bins))
        error_sum[b] += error[ii]
        bin_count[b] += 1

    # Mean error (allows NaN for empty bins)
    with np.errstate(divide='ignore', invalid='ignore'):
        error_mean = error_sum / bin_count

    # RMS calculation (total RMS from sine fit)
    for ii in range(N):
        b = int(np.mod(np.round(phase[ii]/360*bins), bins))
        error_rms[b] += error[ii]**2

    with np.errstate(divide='ignore', invalid='ignore'):
        error_rms = np.sqrt(error_rms / bin_count)

    # ========== Amplitude/Phase Noise Decomposition ==========
    # Sensitivity curves
    amplitude_sensitivity = np.abs(np.cos(phase_bins/360*2*np.pi))**2
    phase_sensitivity = np.abs(np.sin(phase_bins/360*2*np.pi))**2

    # Filter out NaN values (empty bins) before least squares fit
    valid_mask = ~np.isnan(error_rms)
    error_rms_squared = error_rms[valid_mask]**2

    # Try full fit first [amplitude, phase, baseline]
    try:
        from scipy import linalg as sp_linalg
        A_full = np.column_stack([amplitude_sensitivity[valid_mask],
                                  phase_sensitivity[valid_mask],
                                  np.ones(np.sum(valid_mask))])
        tmp, residuals, rank, s = sp_linalg.lstsq(A_full, error_rms_squared, lapack_driver='gelsd')
    except ImportError:
        # Fallback to numpy if scipy not available
        A_full = np.column_stack([amplitude_sensitivity[valid_mask],
                                  phase_sensitivity[valid_mask],
                                  np.ones(np.sum(valid_mask))])
        tmp = np.linalg.lstsq(A_full, error_rms_squared, rcond=None)[0]

    amplitude_noise = np.sqrt(tmp[0]) if tmp[0] >= 0 and np.isreal(tmp[0]) else -1
    phase_noise = np.sqrt(tmp[1]) / mag if tmp[1] >= 0 and np.isreal(tmp[1]) else -1
    baseline = tmp[2]

    # If amplitude noise fails, try phase-only fit
    if amplitude_noise < 0 or np.imag(amplitude_noise) != 0:
        A_phase = np.column_stack([phase_sensitivity[valid_mask], np.ones(np.sum(valid_mask))])
        try:
            tmp = sp_linalg.lstsq(A_phase, error_rms_squared, lapack_driver='gelsd')[0]
        except:
            tmp = np.linalg.lstsq(A_phase, error_rms_squared, rcond=None)[0]
        amplitude_noise = 0
        phase_noise = np.sqrt(tmp[0]) / mag if tmp[0] >= 0 and np.isreal(tmp[0]) else -1
        baseline = tmp[1]

        # If phase-only also fails, fallback to mean baseline
        if phase_noise < 0 or np.imag(phase_noise) != 0:
            amplitude_noise = 0
            phase_noise = 0
            baseline = np.mean(error_rms_squared)

    # If phase noise fails, try amplitude-only fit
    if phase_noise < 0 or np.imag(phase_noise) != 0:
        A_amp = np.column_stack([amplitude_sensitivity[valid_mask], np.ones(np.sum(valid_mask))])
        try:
            tmp = sp_linalg.lstsq(A_amp, error_rms_squared, lapack_driver='gelsd')[0]
        except:
            tmp = np.linalg.lstsq(A_amp, error_rms_squared, rcond=None)[0]
        phase_noise = 0
        amplitude_noise = np.sqrt(tmp[0]) if tmp[0] >= 0 and np.isreal(tmp[0]) else -1
        baseline = tmp[1]

        # If amplitude-only also fails, fallback to mean baseline
        if amplitude_noise < 0 or np.imag(amplitude_noise) != 0:
            amplitude_noise = 0
            phase_noise = 0
            baseline = np.mean(error_rms_squared)

    # Ensure real values
    amplitude_noise = float(np.real(amplitude_noise))
    phase_noise = float(np.real(phase_noise))

    # Filter error range if specified
    if error_range is not None:
        eid = (phase >= error_range[0]) & (phase <= error_range[1])
        phase = phase[eid]
        error = error[eid]

    # Plotting
    if disp:
        fig = plt.figure(figsize=(10, 8))
        ax1 = plt.subplot(2, 1, 1)
        ax2 = plt.subplot(2, 1, 2)

        # Top subplot: data and error vs phase
        ax1_left = ax1
        ax1_left.plot(phase, data, 'k.', markersize=2, label='data')
        ax1_left.set_xlim([0, 360])
        ax1_left.set_ylim([np.min(data), np.max(data)])
        ax1_left.set_ylabel('Data', color='k')
        ax1_left.tick_params(axis='y', labelcolor='k')

        ax1_right = ax1.twinx()
        ax1_right.plot(phase, error, 'r.', markersize=2, alpha=0.5)
        ax1_right.plot(phase_bins, error_mean, 'b-', linewidth=2, label='error')
        ax1_right.set_xlim([0, 360])
        ax1_right.set_ylim([np.min(error), np.max(error)])
        ax1_right.set_ylabel('Error', color='r')
        ax1_right.tick_params(axis='y', labelcolor='r')

        ax1.legend(['data', 'error'], loc='upper right')
        ax1.set_xlabel('Phase (deg)')
        ax1.grid(True, alpha=0.3)

        if error_range is not None:
            ax1_right.plot(phase, error, 'm.', markersize=2)

        # Bottom subplot: RMS error with fitted curves
        ax2.bar(phase_bins, error_rms, width=360/bins*0.8, color='skyblue', alpha=0.7)

        # Compute fitted curves with proper handling of negative values
        amp_fit = amplitude_noise**2 * amplitude_sensitivity + baseline
        phase_fit = phase_noise**2 * phase_sensitivity * mag**2 + baseline

        # Only plot where values are non-negative
        ax2.plot(phase_bins, np.sqrt(np.maximum(amp_fit, 0)), 'b-', linewidth=2)
        ax2.plot(phase_bins, np.sqrt(np.maximum(phase_fit, 0)), 'r-', linewidth=2)
        ax2.set_xlim([0, 360])
        ax2.set_ylim([0, np.max(error_rms)*1.2])

        # Add text labels
        ax2.text(10, np.max(error_rms)*1.15,
                f'Normalized Amplitude Noise RMS = {amplitude_noise/mag:.2e}',
                color='b', fontsize=10)
        ax2.text(10, np.max(error_rms)*1.05,
                f'Phase Noise RMS = {phase_noise:.2e} rad',
                color='r', fontsize=10)

        ax2.set_xlabel('Phase (deg)')
        ax2.set_ylabel('RMS Error')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

    return error_mean, error_rms, phase_bins, amplitude_noise, phase_noise, error, phase
