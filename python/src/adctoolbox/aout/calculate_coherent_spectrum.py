"""Calculate coherent spectrum for phase analysis.

This module computes the phase-aligned complex spectrum from multiple FFT runs,
providing the core data for polar phase plotting.

Key responsibilities:
- Phase alignment of multiple FFT runs
- Coherent averaging to improve SNR
- Preparation of complex spectrum data for visualization

MATLAB counterpart: Part of plotphase.m (FFT mode)
"""

import numpy as np
from typing import Dict, Optional, Tuple
from ._prepare_fft_input import _prepare_fft_input
from ._find_fundamental import _find_fundamental
from ._find_harmonic_bins import _find_harmonic_bins


def calculate_coherent_spectrum(
    data: np.ndarray,
    max_code: Optional[float] = None,
    osr: int = 1,
    cutoff_freq: float = 0,
    fs: float = 1.0,
    win_type: str = 'boxcar',
    n_fft: Optional[int] = None
) -> Dict:
    """Calculate phase-aligned coherent spectrum.

    Performs coherent averaging of multiple FFT runs with phase alignment.
    The output is a complex spectrum ready for polar phase visualization.

    Parameters
    ----------
    data : np.ndarray
        ADC output data, shape (M, N) for M runs or (N,) for single run
    max_code : float, optional
        Maximum code level for normalization. If None, uses peak-to-peak
    osr : int, optional
        Oversampling ratio, default is 1
    cutoff_freq : float, optional
        High-pass cutoff to remove flicker noise (Hz), default is 0
    fs : float, optional
        Sampling frequency (Hz), default is 1.0
    win_type : str, optional
        Window function type, default is 'boxcar'
    n_fft : int, optional
        FFT length, default is length of data

    Returns
    -------
    dict
        Dictionary containing:
        - 'complex_spec_coherent': Complex spectrum with phase alignment
        - 'minR_dB': Noise floor level in dB for plot scaling
        - 'bin_idx': Integer index of fundamental frequency
        - 'bin_r': Refined fundamental bin position
        - 'n_fft': FFT length used
        - 'spec_mag_db': Magnitude spectrum in dB
        - 'phase': Phase spectrum in radians
        - 'harmonic_bins': Array of harmonic bin positions
        - 'n_runs': Number of valid runs processed

    Notes
    -----
    - Aligns phase of fundamental to 0 degrees across all runs
    - Applies coherent averaging (complex sum) to preserve phase information
    - Calculates noise floor reference for polar plot scaling
    - Handles aliasing for harmonic positions
    """
    # Prepare input data
    data_processed, max_code_used, n_samples = _prepare_fft_input(
        data, max_code=max_code, win_type=win_type, n_fft=n_fft
    )

    # Set n_fft if not provided
    if n_fft is None:
        n_fft = n_samples

    # Get dimensions
    n_runs, n = data_processed.shape
    n = min(n, n_fft)  # Truncate if needed

    # Initialize accumulated spectrum
    spec_coherent = np.zeros(n_fft, dtype=complex)
    valid_runs = 0

    # Process each run
    for run_idx in range(n_runs):
        # Get current run data
        run_data = data_processed[run_idx, :n_fft]

        # Skip if signal is too weak
        if np.std(run_data) < 1e-10:
            continue

        # Compute FFT
        fft_data = np.fft.fft(run_data)
        fft_data[0] = 0  # Remove DC

        # Find fundamental in this run
        search_range = n_fft // 2 // osr
        spectrum_search = np.abs(fft_data[:search_range])

        # Skip if DC bin is the maximum (invalid signal)
        if len(spectrum_search) > 0 and np.argmax(spectrum_search) == 0:
            continue

        # Find fundamental bin
        bin_idx, bin_r = _find_fundamental(
            fft_data, n_fft, osr=osr, method='magnitude'
        )

        # Get fundamental phase
        if bin_idx < len(fft_data):
            fundamental_phase = np.angle(fft_data[bin_idx])
        else:
            continue

        # Calculate phase rotation to align fundamental to 0 degrees
        phase_rotation = np.exp(-1j * fundamental_phase)

        # Apply phase alignment
        # For each harmonic and non-harmonic bin, apply appropriate rotation
        fft_aligned = np.zeros_like(fft_data)
        harmonic_markers = np.zeros(n_fft, dtype=bool)

        # Process harmonics (phase rotates with harmonic number)
        for h in range(1, n_fft + 1):
            # Calculate which bin this harmonic maps to
            j = bin_idx * h  # Frequency index
            phase_for_harmonic = (h - 1) * fundamental_phase  # Phase accumulates

            # Handle aliasing (match MATLAB logic)
            if (j // (n_fft // 2)) % 2 == 0:
                # Even Nyquist zone: no reflection
                b = j % n_fft
                if not harmonic_markers[b]:
                    fft_aligned[b] = fft_data[b] * np.exp(-1j * phase_for_harmonic)
                    harmonic_markers[b] = True
            else:
                # Odd Nyquist zone: reflected
                b = n_fft - (j % n_fft)
                if b < n_fft and not harmonic_markers[b]:
                    fft_aligned[b] = fft_data[b] * np.exp(1j * phase_for_harmonic)  # Conj rotation for reflection
                    harmonic_markers[b] = True

        # Process non-harmonic bins (fractional phase rotation)
        for k in range(n_fft):
            if not harmonic_markers[k] and bin_idx > 0:
                # Fractional phase based on frequency ratio
                phase_fraction = (k / bin_idx) * fundamental_phase
                fft_aligned[k] = fft_data[k] * np.exp(-1j * phase_fraction)

        # Add to coherent sum
        spec_coherent += fft_aligned
        valid_runs += 1

    # Check if we processed any valid runs
    if valid_runs == 0:
        raise ValueError("No valid data runs processed")

    # Average the coherent spectrum
    spec_coherent = spec_coherent / valid_runs

    # Limit to in-band portion
    nd2 = n_fft // 2 // osr
    spec_inband = spec_coherent[:nd2]

    # Apply high-pass cutoff if specified (remove flicker noise)
    if cutoff_freq > 0 and fs > 0:
        n_cutoff = int(cutoff_freq / fs * n_fft)
        if n_cutoff > 0 and n_cutoff < len(spec_inband):
            spec_inband[:n_cutoff] = 0

    # Calculate magnitude in dB (for noise floor reference)
    # Use coherent averaging formula
    mag_linear = np.abs(spec_inband)
    mag_db = 20 * np.log10(mag_linear + 1e-20)

    # Add coherent averaging gain
    mag_db = mag_db + 20 * np.log10(valid_runs)

    # Normalize to full scale
    mag_db = mag_db + 20 * np.log10(max_code_used / 2)

    # Calculate noise floor (minR_dB) for plot scaling
    # Use 1st percentile of magnitude (robust to outliers)
    mag_sorted = np.sort(mag_db)
    minR_idx = max(1, int(len(mag_sorted) * 0.01))  # At least 1 element
    minR_dB = mag_sorted[minR_idx - 1]

    # Ensure minimum noise floor
    if np.isinf(minR_dB) or minR_dB < -150:
        minR_dB = -150

    # Find fundamental in coherent spectrum
    bin_idx_final, bin_r_final = _find_fundamental(
        spec_inband, n_fft, osr=osr, method='log'
    )

    # Calculate harmonic positions
    harmonic_bins = _find_harmonic_bins(
        bin_r_final, harmonic=10, n_fft=n_fft
    )

    # Prepare phase information
    phase_aligned = np.angle(spec_inband)

    # Return results
    return {
        'complex_spec_coherent': spec_inband,
        'minR_dB': minR_dB,
        'bin_idx': bin_idx_final,
        'bin_r': bin_r_final,
        'n_fft': n_fft,
        'spec_mag_db': mag_db,
        'phase': phase_aligned,
        'harmonic_bins': harmonic_bins,
        'n_runs': valid_runs,
        'max_code': max_code_used
    }


def prepare_polar_plot_data(coherent_result: Dict, harmonic: int = 5) -> Dict:
    """Prepare plot data for polar phase visualization.

    Converts the coherent spectrum result to the format expected by
    plot_polar_phase function.

    Parameters
    ----------
    coherent_result : dict
        Output from calculate_coherent_spectrum
    harmonic : int, optional
        Number of harmonics to include, default is 5

    Returns
    -------
    dict
        Plot data dictionary with keys:
        - 'complex_spec_coherent': Complex spectrum
        - 'minR_dB': Noise floor in dB
        - 'bin_idx': Fundamental bin index
        - 'N_fft': FFT length
        - 'harmonic_bins': Harmonic positions (truncated to requested number)
    """
    # Extract required fields
    return {
        'complex_spec_coherent': coherent_result['complex_spec_coherent'],
        'minR_dB': coherent_result['minR_dB'],
        'bin_idx': coherent_result['bin_idx'],
        'N_fft': coherent_result['n_fft'],
        'harmonic_bins': coherent_result['harmonic_bins'][:harmonic]
    }