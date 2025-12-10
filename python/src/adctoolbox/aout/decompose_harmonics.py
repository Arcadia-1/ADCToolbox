"""
Decompose signal into fundamental and harmonic errors.

Separates ADC output into fundamental signal, harmonic distortion, and other noise.
"""

import numpy as np
import matplotlib.pyplot as plt


def decompose_harmonics(data, normalized_freq=None, order=10, show_plot=True):
    """
    Decompose signal into fundamental and harmonic errors.

    Parameters
    ----------
    data : array_like
        ADC output data, 1D numpy array
    normalized_freq : float, optional
        Normalized frequency (f_in / f_sample), auto-detect if None
    order : int, default=10
        Harmonic order for fitting (fits fundamental + harmonics 2 through order)
    show_plot : bool, default=True
        Whether to display result plot

    Returns
    -------
    fundamental_signal : ndarray
        Fundamental sinewave component (including DC)
    total_error : ndarray
        Total error (data - fundamental_signal)
    harmonic_error : ndarray
        Harmonic distortions (2nd through nth harmonics)
    other_error : ndarray
        All other errors (data - all harmonics)

    Notes
    -----
    The decomposition uses the following model:

    - fundamental_signal = DC + weight_i*cos(ωt) + weight_q*sin(ωt)
    - signal_all = DC + Σ[weight_i_k*cos(kωt) + weight_q_k*sin(kωt)]
    - total_error = data - fundamental_signal
    - harmonic_error = signal_all - fundamental_signal
    - other_error = data - signal_all
    """

    # Prepare data
    data = np.asarray(data).flatten()
    n_samples = len(data)
    t = np.arange(n_samples)

    # Auto-detect frequency if not provided
    if normalized_freq is None or np.isnan(normalized_freq):
        try:
            from findFin import findFin
            normalized_freq = findFin(data)
        except ImportError:
            # FFT-based frequency detection
            spec = np.abs(np.fft.fft(data))
            spec[0] = 0
            normalized_freq = np.argmax(spec[:n_samples//2]) / n_samples
            print(f"Warning: findFin not found, using FFT detection: freq={normalized_freq:.6f}")

    # Compute fundamental (I/Q) components
    phase = t * normalized_freq * 2 * np.pi
    cos_basis = np.cos(phase)
    sin_basis = np.sin(phase)

    dc_offset = np.mean(data)
    weight_i = np.mean(cos_basis * data) * 2
    weight_q = np.mean(sin_basis * data) * 2
    fundamental_signal = dc_offset + cos_basis * weight_i + sin_basis * weight_q

    # Build harmonic basis matrix (vectorized)
    cos_matrix = np.array([np.cos(phase * (k + 1)) for k in range(order)]).T
    sin_matrix = np.array([np.sin(phase * (k + 1)) for k in range(order)]).T
    basis_matrix = np.column_stack([cos_matrix, sin_matrix])

    # Least squares fit for harmonics
    weights, *_ = np.linalg.lstsq(basis_matrix, data, rcond=None)
    signal_all = dc_offset + basis_matrix @ weights

    # Decompose errors
    total_error = data - fundamental_signal
    harmonic_error = signal_all - fundamental_signal
    other_error = data - signal_all

    # Visualization
    if show_plot:
        ax1 = plt.gca()
        ax1.plot(data, 'kx', label='data', markersize=3, alpha=0.5)
        ax1.plot(fundamental_signal, '-', color=[0.5, 0.5, 0.5], label='fundamental_signal', linewidth=1.5)

        # Display range: first 3 periods or at least 100 points
        xlim_max = min(max(int(3 / normalized_freq), 100), n_samples)
        ax1.set_xlim([0, xlim_max])
        data_min, data_max = np.min(data), np.max(data)
        ax1.set_ylim([data_min * 1.1, data_max * 1.1])
        ax1.set_ylabel('Signal', color='k')
        ax1.tick_params(axis='y', labelcolor='k')

        # Right Y-axis for errors
        ax2 = ax1.twinx()

        # Calculate RMS and percentages
        rms = np.sqrt(np.mean(np.array([harmonic_error, other_error, total_error])**2, axis=1))
        rms_harmonic, rms_other, rms_total = rms
        pct_harmonic = (rms_harmonic / rms_total)**2 * 100
        pct_other = (rms_other / rms_total)**2 * 100

        # Select unit based on magnitude
        scale, unit = (1e6, 'uV') if rms_total < 1e-3 else (1e3, 'mV') if rms_total < 1 else (1, 'V')

        ax2.plot(harmonic_error, 'r-', label=f'harmonics ({rms_harmonic*scale:.1f}{unit}, {pct_harmonic:.1f}%)', linewidth=1.5)
        ax2.plot(other_error, 'b-', label=f'other errors ({rms_other*scale:.1f}{unit}, {pct_other:.1f}%)', linewidth=1)

        error_min, error_max = np.min(total_error), np.max(total_error)
        ax2.set_ylim([error_min * 1.1, error_max * 1.1])
        ax2.set_ylabel('Error', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        ax1.set_xlabel('Samples')
        ax1.set_title(f'Decompose Harmonics (freq={normalized_freq:.6f}, order={order})')

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        ax1.grid(True, alpha=0.3)

    return fundamental_signal, total_error, harmonic_error, other_error