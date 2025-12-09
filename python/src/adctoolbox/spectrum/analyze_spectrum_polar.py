"""Analyze spectrum with polar phase visualization (FFT coherent mode).

This module provides a high-level wrapper that combines FFT coherent spectrum
calculation with polar phase visualization.

Part of the modular ADC analysis architecture.
Matches MATLAB plotphase.m FFT mode functionality.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any

from .calculate_spectrum_data import calculate_spectrum_data
from .plot_spectrum_polar import plot_spectrum_polar


def analyze_spectrum_polar(
    data: np.ndarray,
    max_code: Optional[float] = None,
    harmonic: int = 5,
    osr: int = 1,
    cutoff_freq: float = 0,
    fs: float = 1.0,
    win_type: str = 'boxcar',
    n_fft: Optional[int] = None,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
    ax: Optional[plt.Axes] = None,
    fixed_radial_range: Optional[float] = None
) -> Dict[str, Any]:
    """Analyze spectrum with polar phase visualization (FFT coherent mode).

    This wrapper function combines FFT coherent spectrum calculation
    with polar phase plot visualization, following the modular architecture pattern.
    Matches MATLAB plotphase.m FFT mode behavior.

    Parameters
    ----------
    data : np.ndarray
        Input ADC data, shape (N,) for single run or (M, N) for M runs
    max_code : float, optional
        Maximum code level for normalization. If None, uses (max - min)
    harmonic : int, optional
        Number of harmonics to mark on polar plot (default: 5)
    osr : int, optional
        Oversampling ratio (default: 1)
    cutoff_freq : float, optional
        High-pass cutoff frequency in Hz for removing low-frequency noise (default: 0)
    fs : float, optional
        Sampling frequency in Hz (default: 1.0)
    win_type : str, optional
        Window function type: 'boxcar', 'hann', 'hamming', etc. (default: 'boxcar')
    n_fft : int, optional
        FFT length. If None, uses data length
    title : str, optional
        Custom title for the plot
    save_path : str or Path, optional
        Path to save the figure. If None, figure is not saved
    show_plot : bool, optional
        Whether to create and display the plot (default: True)
    ax : matplotlib.axes.Axes, optional
        Pre-existing polar axes to plot on. If None, creates new figure
    fixed_radial_range : float, optional
        Fixed radial range in dB (e.g., 120 for 0 to -120 dB range).
        If None, auto-scales based on noise floor.

    Returns
    -------
    coherent_result : dict
        Dictionary containing coherent spectrum results from calculate_coherent_spectrum():
        - 'complex_spec_coherent': Phase-aligned complex spectrum
        - 'minR_dB': Noise floor for plot scaling
        - 'bin_idx': Fundamental bin index
        - 'bin_r': Refined bin position
        - 'n_fft': FFT length
        - 'metrics': Performance metrics (ENOB, SNR, SNDR, etc.)
        - 'hd2_phase_deg': HD2 phase in degrees
        - 'hd3_phase_deg': HD3 phase in degrees

    Examples
    --------
    >>> import numpy as np
    >>> from adctoolbox.aout import analyze_spectrum_polar
    >>>
    >>> # Generate test signal with multiple runs
    >>> N = 2**13
    >>> n_runs = 10
    >>> t = np.arange(N)
    >>> signal = np.array([0.5*np.sin(2*np.pi*0.1*t) + 0.01*np.random.randn(N)
    ...                    for _ in range(n_runs)])
    >>>
    >>> # Analyze with polar plot (FFT coherent mode)
    >>> coherent_result, plot_data = analyze_spectrum_polar(
    ...     signal, harmonic=5, fs=800e6, save_path='polar_fft.png'
    ... )
    >>>
    >>> print(f"Fundamental bin: {coherent_result['bin_idx']}")
    >>> print(f"Noise floor: {coherent_result['minR_dB']:.1f} dB")

    Notes
    -----
    Modular architecture:
    1. calculate_coherent_spectrum() - Pure calculation (FFT with phase alignment)
    2. plot_polar_phase() - Pure visualization (polar plot with spectrum dots)
    3. analyze_spectrum_polar() - Wrapper combining both

    FFT Coherent Mode Features:
    - Phase alignment across multiple runs for improved SNR
    - All frequency bins shown as black dots
    - Fundamental normalized to 0° (pointing up)
    - Harmonics marked with blue squares
    - Parabolic interpolation for sub-bin frequency accuracy
    """

    # Step 1: Calculate coherent spectrum (pure computation)
    result = calculate_spectrum_data(
        data=data,
        max_scale_range=max_code,
        osr=osr,
        cutoff_freq=cutoff_freq,
        fs=fs,
        win_type=win_type,
        complex_spectrum=True
    )

    # Step 2: Prepare plot data (keys must match plot_spectrum_polar requirements)
    plot_data = {
        'complex_spec_coherent': result['complex_spec_coherent'],
        'minR_dB': result['minR_dB'],
        'bin_idx': result['bin_idx'],
        'n_fft': result['N'],  # Use lowercase n_fft for plot_spectrum_polar
        'metrics': result.get('metrics', {}),  # Pass metrics for annotation
        'hd2_phase_deg': result.get('hd2_phase_deg', 0),  # Pass HD2 phase
        'hd3_phase_deg': result.get('hd3_phase_deg', 0),  # Pass HD3 phase
    }

    # Also store the result dict for coherent_result return value
    coherent_result = {
        'complex_spec_coherent': result['complex_spec_coherent'],
        'minR_dB': result['minR_dB'],
        'bin_idx': result['bin_idx'],
        'bin_r': result.get('bin_r', result['bin_idx']),
        'n_fft': result['N'],
        'metrics': result.get('metrics', {}),
        'hd2_phase_deg': result.get('hd2_phase_deg', 0),
        'hd3_phase_deg': result.get('hd3_phase_deg', 0)
    }

    # Step 3: Plot if requested (pure visualization)
    # If ax is provided, always plot (for subplots). Otherwise check show_plot/save_path
    if ax is not None or show_plot or save_path:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        else:
            fig = ax.get_figure()
            # Verify axes has polar projection
            if not hasattr(ax, 'set_theta_zero_location'):
                raise ValueError("Axes must have polar projection")

        plot_spectrum_polar(plot_data, harmonic=harmonic, ax=ax, title=title, fixed_radial_range=fixed_radial_range)

        # Save figure if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        # Show plot if requested
        if show_plot and ax is None:
            plt.show()

    return coherent_result
