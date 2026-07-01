"""Generate AOUT analysis dashboard with 12 analysis plots in a 3x4 panel."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from adctoolbox.spectrum import analyze_spectrum
from adctoolbox.spectrum import analyze_spectrum_polar
from adctoolbox.aout import analyze_error_by_value
from adctoolbox.aout import analyze_error_by_phase
from adctoolbox.aout import analyze_decomposition_time
from adctoolbox.aout import analyze_decomposition_polar
from adctoolbox.aout import analyze_error_spectrum
from adctoolbox.aout import analyze_error_envelope_spectrum
from adctoolbox.aout import analyze_error_autocorr
from adctoolbox.aout import analyze_error_pdf
from adctoolbox.aout import analyze_phase_plane
from adctoolbox.aout import analyze_error_phase_plane

def generate_aout_dashboard(signal, fs=1.0, freq=None, output_path=None, resolution=12, show=False):
    """
    Generate comprehensive analysis dashboard with 12 subplots in a 3x4 panel.

    Parameters
    ----------
    signal : array_like
        Input signal (ADC output or analog signal)
    fs : float, optional
        Sampling frequency (default: 1.0 for normalized frequency)
    freq : float, optional
        Signal frequency in Hz (default: None, auto-estimate)
        Will be converted to normalized frequency where needed
    output_path : str or Path, optional
        Path to save figure (default: None, don't save)
    resolution : int, optional
        ADC resolution in bits (default: 12)
    show : bool, optional
        Whether to display figure (default: False)

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the dashboard
    axes : ndarray
        Array of axes objects (3x4 grid, flattened)
    """

    signal = np.asarray(signal).flatten()

    # Calculate normalized frequency if freq is provided
    norm_freq = freq / fs if freq is not None else None
    fit_kwargs = {"max_iterations": 0} if norm_freq is not None else {}

    # Create 3x4 panel
    fig, axes = plt.subplots(3, 4, figsize=(32, 18))
    axes = axes.flatten()

    # Recreate polar axes for plots that need them (plot 2 and plot 6)
    fig.delaxes(axes[1])
    axes[1] = fig.add_subplot(3, 4, 2, projection='polar')
    fig.delaxes(axes[5])
    axes[5] = fig.add_subplot(3, 4, 6, projection='polar')

    # Plot 1: analyze_spectrum
    plt.sca(axes[0])
    analyze_spectrum(signal, fs=fs)
    axes[0].set_title('(1) Spectrum', fontsize=12, fontweight='bold')

    # Plot 2: analyze_spectrum_polar
    plt.sca(axes[1])
    analyze_spectrum_polar(signal, fs=fs)
    axes[1].set_title('(2) Spectrum Polar', fontsize=12, fontweight='bold', pad=20)

    # Plot 3: analyze_error_by_value
    analyze_error_by_value(
        signal,
        norm_freq=norm_freq,
        ax=axes[2],
        title='(3) Error by Value',
        **fit_kwargs,
    )

    # Plot 4: analyze_error_by_phase
    analyze_error_by_phase(
        signal,
        norm_freq=norm_freq,
        ax=axes[3],
        title='(4) Error by Phase',
        **fit_kwargs,
    )

    # Plot 5: analyze_decomposition_time
    analyze_decomposition_time(
        signal,
        ax=axes[4],
        title='(5) Decomposition Time',
        frequency=norm_freq,
        **fit_kwargs,
    )

    # Plot 6: analyze_decomposition_polar
    analyze_decomposition_polar(
        signal,
        ax=axes[5],
        title='(6) Decomposition Polar',
        frequency=norm_freq,
        **fit_kwargs,
    )

    # Plot 7: analyze_error_pdf
    analyze_error_pdf(
        signal,
        resolution=resolution,
        frequency=norm_freq,
        ax=axes[6],
        title='(7) Error PDF',
        **fit_kwargs,
    )

    # Plot 8: analyze_error_autocorr
    analyze_error_autocorr(
        signal,
        frequency=norm_freq,
        ax=axes[7],
        title='(8) Error Autocorrelation',
    )

    # Plot 9: analyze_error_spectrum
    analyze_error_spectrum(
        signal,
        fs=fs,
        frequency=norm_freq,
        ax=axes[8],
        title='(9) Error Spectrum',
        **fit_kwargs,
    )

    # Plot 10: analyze_error_envelope_spectrum
    analyze_error_envelope_spectrum(
        signal,
        fs=fs,
        frequency=norm_freq,
        ax=axes[9],
        title='(10) Error Envelope Spectrum',
        **fit_kwargs,
    )

    # Plot 11: analyze_phase_plane
    plt.sca(axes[10])
    analyze_phase_plane(signal, fs=fs, ax=axes[10], title='(11) Phase Plane')

    # Plot 12: analyze_error_phase_plane
    plt.sca(axes[11])
    analyze_error_phase_plane(
        signal,
        fs=fs,
        ax=axes[11],
        title='(12) Error Phase Plane',
    )

    # Overall title
    fig.suptitle('Comprehensive ADC Analysis Dashboard (12 Tools)',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()

    # Save if requested
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[Dashboard saved] -> {output_path}")
        plt.close(fig)

    return fig, axes
