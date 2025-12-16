"""
Plot phase error analysis results (RMS curves, AM/PM decomposition).

Visualization function for displaying phase-binned error analysis results,
including RMS curves and AM/PM noise separation.

MATLAB counterpart: errsin.m (phase mode)
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_rearranged_error_by_phase(results: dict, disp=1, plot_mode="binned"):
    """
    Plot phase error analysis results (RMS curves, AM/PM).

    Creates a comprehensive visualization showing:
    - Top panel: Signal and error vs phase (dual y-axis)
    - Bottom panel: RMS error vs phase with AM/PM fitted curves (binned mode)
                   or raw error scatter (raw mode)

    Parameters
    ----------
    results : dict
        Dictionary from rearrange_error_by_phase(). Must contain:
        - 'error': Raw error signal
        - 'phase': Phase values for each sample (radians)
        - 'fitted_signal': Fitted sine signal
        - 'erms': RMS error per phase bin
        - 'emean': Mean error per phase bin
        - 'phase_bins': Phase bin centers (radians)
        - 'am_param': AM parameter
        - 'pm_param': PM parameter
        - 'baseline': Baseline noise
        - 'fundamental_amplitude': Fitted amplitude

    disp : int, optional
        Display plots (1=yes, 0=no) (default: 1)

    plot_mode : str, optional
        Plot visualization mode:
        - "binned": Show binned RMS bars with AM/PM curves (default)
        - "raw": Show all raw error points as scatter plot

    Notes
    -----
    The top panel uses dual y-axis:
    - Left axis: Signal data
    - Right axis: Error signal

    The bottom panel shows the RMS error decomposition:
    Binned mode:
    - Blue curve: AM component (cos² dependence)
    - Red curve: PM component (sin² dependence)
    - Bars: Actual binned RMS values

    Raw mode:
    - Scatter: All raw error points

    The fitted model is:
        RMS²(φ) = am_param² * cos²(φ) + pm_param² * A² * sin²(φ) + baseline

    Examples
    --------
    >>> from adctoolbox.aout import rearrange_error_by_phase
    >>> sig = np.sin(2*np.pi*0.1*np.arange(1000)) + 0.01*np.random.randn(1000)
    >>> results = rearrange_error_by_phase(sig, 0.1, mode="binned")
    >>> plot_rearranged_error_by_phase(results, plot_mode="binned")
    """
    if not disp:
        return

    # Extract data from results
    error = results.get('error', np.array([]))
    phase = results.get('phase', np.array([]))
    fitted_signal = results.get('fitted_signal', np.array([]))
    erms = results['erms']
    emean = results['emean']
    phase_bins = results['phase_bins']
    am_param = results['am_param']
    pm_param = results['pm_param']
    baseline = results['baseline']
    fundamental_amplitude = results.get('fundamental_amplitude', 1.0)

    # Convert phase to degrees for plotting
    phase_bins_deg = phase_bins * 180 / np.pi
    phase_deg = phase * 180 / np.pi

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # --- Top Panel: Signal and Error vs Phase (dual y-axis) ---
    ax1_left = ax1
    ax1_right = ax1.twinx()

    # Left axis: signal data
    if len(phase_deg) > 0 and len(fitted_signal) > 0:
        ax1_left.plot(phase_deg, fitted_signal, 'k.', markersize=3)
        ax1_left.set_xlim([0, 360])
        ax1_left.set_ylim([np.min(fitted_signal), np.max(fitted_signal)])
        ax1_left.set_ylabel('Data', color='k')
        ax1_left.tick_params(axis='y', labelcolor='k')

    # Right axis: error
    if len(phase_deg) > 0 and len(error) > 0:
        ax1_right.plot(phase_deg, error, 'r.', markersize=3, label='Error')
        ax1_right.plot(phase_bins_deg, emean, 'b-', linewidth=2, label='Mean error')
        ax1_right.set_xlim([0, 360])
        ax1_right.set_ylim([np.min(error), np.max(error)])
        ax1_right.set_ylabel('Error', color='r')
        ax1_right.tick_params(axis='y', labelcolor='r')

    ax1.set_xlabel('Phase (deg)')
    ax1.set_title('Phase Error Analysis: Signal and Error')
    ax1.grid(True, alpha=0.3)
    if len(phase_deg) > 0:
        ax1_right.legend(loc='upper right', fontsize=9)

    # --- Bottom Panel: RMS Error vs Phase with AM/PM curves (Binned) OR Raw scatter (Raw) ---
    if plot_mode == "raw":
        # Raw mode: show all error points as scatter
        ax2.scatter(phase_deg, error, alpha=0.3, s=1, label='Raw error', color='r')
        ax2.set_xlim([0, 360])
        ax2.set_ylim([np.min(error), np.max(error)])
        ax2.set_ylabel('Error')
        ax2.set_title('Error vs Phase (Raw Data - All Points)')
    else:
        # Binned mode: show binned RMS bars with AM/PM curves
        bin_width = 360 / len(phase_bins)

        # Bar plot for actual RMS
        ax2.bar(phase_bins_deg, erms, width=bin_width*0.8, color='skyblue', alpha=0.7)

        # Compute fitted curves using sensitivity patterns
        asen = np.cos(phase_bins * 2 * np.pi / 360)**2  # AM sensitivity
        psen = np.sin(phase_bins * 2 * np.pi / 360)**2  # PM sensitivity

        am_curve = np.sqrt(am_param**2 * asen + baseline)
        pm_curve = np.sqrt(pm_param**2 * fundamental_amplitude**2 * psen + baseline)

        ax2.plot(phase_bins_deg, am_curve, 'b-', linewidth=2, label='AM Component')
        ax2.plot(phase_bins_deg, pm_curve, 'r-', linewidth=2, label='PM Component')

        ax2.set_xlim([0, 360])
        ax2.set_ylim([0, np.max(erms) * 1.2])
        ax2.set_ylabel('RMS Error')
        ax2.set_title('RMS Error vs Phase (AM/PM Decomposition)')

    ax2.set_xlabel('Phase (deg)')
    ax2.grid(True, alpha=0.3)

    # Add parameter annotations (MATLAB style) - only for binned mode
    if plot_mode == "binned":
        max_rms = np.max(erms)
        text_y1 = max_rms * 1.15
        text_y2 = max_rms * 1.05

        # Normalize AM parameter by amplitude for display
        am_normalized = am_param / fundamental_amplitude if fundamental_amplitude > 1e-10 else am_param

        ax2.text(10, text_y1,
                f'Normalized Amplitude Noise RMS = {am_normalized:.2e}',
                color='b', fontsize=10, fontweight='bold')
        ax2.text(10, text_y2,
                f'Phase Noise RMS = {pm_param:.2e} rad',
                color='r', fontsize=10, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
