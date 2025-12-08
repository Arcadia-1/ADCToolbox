"""Calculate spectrum data for ADC analysis - unified calculation engine.

This module provides a pure computation function that calculates FFT spectrum
data for ADC analysis. It can operate in two modes:
1. Power spectrum mode (complex_spectrum=False) - for traditional spectrum plots
2. Complex spectrum mode (complex_spectrum=True) - for polar phase plots with coherent averaging

This is the core calculation engine that drives both plot_spectrum() and
plot_spectrum_polar() visualization functions.

"""

import numpy as np
from typing import Dict, Optional, Union
from ._prepare_fft_input import _prepare_fft_input
from ._find_fundamental import _find_fundamental
from ._find_harmonic_bins import _find_harmonic_bins
from ._align_spectrum_phase import _align_spectrum_phase



def calculate_spectrum_data(
    data: np.ndarray,
    fs: float,
    complex_spectrum: bool = False,
    max_code: Optional[float] = None,
    n_thd: int = 5,
    osr: int = 1,
    win_type: str = 'hann',
    side_bin: int = 1,
    cutoff_freq: float = 0,
    n_fft: Optional[int] = None,
    calc_nsd: bool = True,
    calc_metrics: bool = True,
    hd2_harmonic: bool = False,
    assumed_signal: Optional[float] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """Calculate spectrum data for ADC analysis.

    This is the unified calculation engine for FFT spectrum analysis. It can
    operate in two modes based on the complex_spectrum parameter.

    Parameters
    ----------
    data : np.ndarray
        Input ADC data, shape (N,) for single run or (M, N) for M runs
    fs : float
        Sampling frequency in Hz
    complex_spectrum : bool, optional
        Core mode switch. If True, performs phase alignment and returns complex spectrum.
        If False, returns only power spectrum data. Default: False
    max_code : float, optional
        Full scale range (max-min). If None, uses (max(data) - min(data))
    n_thd : int, optional
        Number of harmonics to include in THD calculation. Default: 5
    osr : int, optional
        Oversampling ratio. Default: 1
    win_type : str, optional
        Window function type: 'boxcar', 'hann', 'hamming', etc. Default: 'hann'
    side_bin : int, optional
        Number of side bins to exclude around signal for noise calculation. Default: 1
    cutoff_freq : float, optional
        High-pass cutoff frequency in Hz for removing low-frequency noise. Default: 0
    n_fft : int, optional
        FFT length. If None, uses data length
    calc_nsd : bool, optional
        Whether to calculate NSD metric. Default: True
    calc_metrics : bool, optional
        Whether to calculate performance metrics (SNDR, THD, etc.). Default: True
    hd2_harmonic : bool, optional
        Whether to include HD2 in harmonic calculations. Default: False
    assumed_signal : float, optional
        Assumed signal amplitude for metrics calculation. Default: None (auto-detect)

    Returns
    -------
    dict
        Dictionary containing three main sections:

        I. Performance Metrics (if calc_metrics=True):
        {
            'enob': Effective Number of Bits,
            'sndr_db': Signal-to-Noise-and-Distortion Ratio,
            'sfdr_db': Spurious-Free Dynamic Range,
            'snr_db': Signal-to-Noise Ratio,
            'thd_db': Total Harmonic Distortion,
            'sig_pwr_dbfs': Signal Power in dBFS,
            'noise_floor_db': Noise Floor in dBFS,
            'nsd_dbfs_hz': Noise Spectral Density in dBFS/Hz
        }

        II. Amplitude Spectrum Data (for plot_spectrum):
        {
            'freq': Frequency axis in Hz,
            'spec_mag_db': Magnitude spectrum in dBFS,
            'bin_r': Refined fundamental bin position,
            'noise_floor_db': Noise floor level
        }

        III. Complex Data (for plot_spectrum_polar, only if complex_spectrum=True):
        {
            'complex_spec_coherent': Phase-aligned complex spectrum,
            'minR_dB': Noise floor for polar plot scaling,
            'bin_idx': Fundamental bin index,
            'n_fft': FFT length
        }

    Notes
    -----
    Modes of operation:
    1. Power spectrum mode (complex_spectrum=False):
       - Calculates traditional power spectrum
       - Returns amplitude data for plot_spectrum
       - Calculates performance metrics

    2. Complex spectrum mode (complex_spectrum=True):
       - Performs phase alignment across multiple runs
       - Returns complex spectrum for polar plotting
       - Includes coherent averaging for improved SNR
       - Both amplitude and complex data are returned

    The function strictly follows the Single Responsibility Principle:
    - Pure computation only (no plotting)
    - All data prepared for downstream visualization functions
    """

    # ============= Common preprocessing =============

    # Prepare input data using shared helper
    data_processed, max_code_used, n_samples = _prepare_fft_input(
        data=data,
        max_code=max_code,
        win_type=win_type,
        n_fft=n_fft
    )

    if n_fft is None:
        n_fft = n_samples

    # Get number of runs
    n_runs = data_processed.shape[0]

    # Initialize results dictionary
    results = {}

    # ============= Mode-specific processing =============

    if complex_spectrum:
        # ============= Complex Spectrum Mode =============
        # Phase alignment and coherent averaging
        spec_coherent = np.zeros(n_fft, dtype=complex)
        n_valid_runs = 0

        for run_idx in range(n_runs):
            # Get current run data
            run_data = data_processed[run_idx, :n_fft]

            # Skip if signal is too weak
            if np.max(np.abs(run_data)) < 1e-10:
                continue

            # Perform FFT
            fft_data = np.fft.fft(run_data)

            # Find fundamental frequency
            bin_idx, bin_r = _find_fundamental(fft_data, n_fft, osr, method='magnitude')

            # Skip if fundamental is at DC
            if bin_idx == 0:
                continue

            # Apply phase alignment using helper function
            fft_aligned = _align_spectrum_phase(fft_data, bin_idx, bin_r, n_fft)

            # Accumulate
            spec_coherent += fft_aligned
            n_valid_runs += 1

        # Normalize by number of valid runs
        if n_valid_runs > 0:
            spec_coherent /= n_valid_runs

        # Take only first half (real signals)
        spec_coherent = spec_coherent[:n_fft // 2]

        # Remove low frequencies if cutoff specified
        if cutoff_freq > 0:
            n_cutoff = int(cutoff_freq / fs * n_fft)
            spec_coherent[:n_cutoff] = 0

        # Calculate noise floor (1st percentile)
        mag_db = 20 * np.log10(np.abs(spec_coherent) + 1e-20)

        # For fair comparison with traditional mode, use median method
        # Remove signal region first
        coherent_mag_temp = np.abs(spec_coherent)
        signal_mask = np.zeros_like(coherent_mag_temp, dtype=bool)
        coherent_bin_idx_temp = np.argmax(coherent_mag_temp[:n_fft // 2 // osr])
        if 1 <= coherent_bin_idx_temp < len(signal_mask):
            signal_mask[max(coherent_bin_idx_temp-2, 0):min(coherent_bin_idx_temp+3, len(signal_mask))] = True

        # Use median of non-signal region for noise floor
        noise_region = mag_db[~signal_mask]
        if len(noise_region) > 0:
            noise_floor_db = np.median(noise_region)
        else:
            noise_floor_db = np.percentile(mag_db, 1)

        if np.isinf(noise_floor_db):
            noise_floor_db = -100

        # Store complex spectrum data for polar plotting
        results['complex_spec_coherent'] = spec_coherent
        results['minR_dB'] = noise_floor_db

        # Find fundamental in coherent spectrum (using original magnitude)
        coherent_mag_temp = np.abs(spec_coherent)
        coherent_bin_idx = np.argmax(coherent_mag_temp[:n_fft // 2 // osr])
        results['bin_idx'] = coherent_bin_idx
        results['n_fft'] = n_fft

        # For complex mode, also provide amplitude data
        freq = np.arange(n_fft // 2) * fs / n_fft

        # Get magnitude spectrum
        coherent_mag = np.abs(spec_coherent)

        # For metrics calculation, normalize power spectrum to FSR (Full Scale Range)
        # This ensures signal_power represents power in dBFS scale
        peak_mag = np.max(coherent_mag)
        normalized_mag = coherent_mag / (peak_mag + 1e-20)
        spectrum_power_unnormalized = normalized_mag ** 2  # Normalized power for metrics

        # Normalize magnitude spectrum to 0 dBFS for display
        spec_mag_db = 20 * np.log10(normalized_mag + 1e-20)

    else:
        # ============= Power Spectrum Mode =============
        # Traditional power spectrum calculation

        # Average FFT across runs (power averaging)
        spectrum_sum = np.zeros(n_fft)

        for run_idx in range(n_runs):
            run_data = data_processed[run_idx, :n_fft]
            fft_data = np.fft.fft(run_data)
            spectrum_sum += np.abs(fft_data) ** 2

        # Average and take square root
        spectrum_avg = np.sqrt(spectrum_sum / n_runs)

        # Take only first half (real signals)
        spectrum_avg = spectrum_avg[:n_fft // 2]

        # Remove low frequencies if cutoff specified
        if cutoff_freq > 0:
            n_cutoff = int(cutoff_freq / fs * n_fft)
            spectrum_avg[:n_cutoff] = 0

        # Convert to dBFS
        spec_mag_db = 20 * np.log10(spectrum_avg + 1e-20)

        # Frequency axis
        freq = np.arange(n_fft // 2) * fs / n_fft

    # ============= Common amplitude data =============
    # Find fundamental frequency first
    # Use the appropriate power spectrum for metrics (un-normalized for complex mode)
    if complex_spectrum:
        spectrum_power = spectrum_power_unnormalized
    else:
        spectrum_power = 10 ** (spec_mag_db / 10)

    bin_idx, bin_r = _find_fundamental(spectrum_power, n_fft, osr, method='power')

    # Store amplitude spectrum data for plot_spectrum
    results['freq'] = freq
    results['spec_mag_db'] = spec_mag_db

    # Also provide with the names expected by plot_spectrum
    results['spec_db'] = spec_mag_db
    results['bin_idx'] = bin_r
    results['sig_bin_start'] = 0
    results['sig_bin_end'] = len(freq)
    results['bin_r'] = bin_r

    # Calculate noise floor
    n_search = n_fft // 2 // osr
    spectrum_search = spectrum_power[:n_search]

    # Remove signal and side bins
    if 1 <= bin_idx < len(spectrum_search) - side_bin:
        spectrum_search[bin_idx-side_bin:bin_idx+side_bin+1] = 0

    # Calculate noise floor from remaining bins (1st percentile)
    noise_power = np.percentile(spectrum_search[spectrum_search > 0], 1)
    noise_floor_db = 10 * np.log10(noise_power + 1e-20)
    results['noise_floor_db'] = noise_floor_db

    # ============= Calculate performance metrics =============
    if calc_metrics:
        # Calculate signal power
        if 1 <= bin_idx < len(spectrum_power):
            signal_power = spectrum_power[bin_idx]
        else:
            signal_power = np.max(spectrum_power)

        # Ensure signal power is reasonable (but don't override valid values!)
        if signal_power < 1e-15:
            signal_power = 1e-15

        sig_pwr_dbfs = 10 * np.log10(signal_power)

        # Find harmonic bins for THD calculation
        harmonic_bins = _find_harmonic_bins(bin_r, n_thd, n_fft)

        # Calculate THD
        thd_power = 0
        for h_idx in range(1, n_thd):
            h_bin = int(round(harmonic_bins[h_idx]))
            if h_bin < len(spectrum_power):
                thd_power += spectrum_power[h_bin]

        thd_db = 10 * np.log10(thd_power + 1e-20)

        # Calculate SNDR
        # Calculate noise by summing bins excluding signal and harmonics
        noise_spectrum = spectrum_power.copy()
        # Exclude DC
        noise_spectrum[0] = 0
        # Remove signal
        if 1 <= bin_idx < len(noise_spectrum):
            noise_spectrum[max(bin_idx-side_bin, 0):min(bin_idx+side_bin+1, len(noise_spectrum))] = 0
        # Remove harmonics
        for h_idx in range(1, n_thd):
            h_bin = int(round(harmonic_bins[h_idx]))
            if 1 <= h_bin < len(noise_spectrum):
                h_start = max(h_bin - side_bin, 0)
                h_end = min(h_bin + side_bin + 1, len(noise_spectrum))
                noise_spectrum[h_start:h_end] = 0

        noise_power = np.sum(noise_spectrum)

        # For very low noise (coherent averaging with many runs), noise_power might be very small but not zero
        # Add minimum threshold to avoid division issues
        noise_power = max(noise_power, 1e-15)
        thd_power_safe = max(thd_power, 1e-15)

        # Calculate SNDR and SNR
        sndr_db = 10 * np.log10(signal_power / (noise_power + thd_power_safe))
        snr_db = 10 * np.log10(signal_power / noise_power)

        # Calculate SFDR
        spectrum_copy = spectrum_power.copy()
        if 1 <= bin_idx < len(spectrum_copy):
            spectrum_copy[bin_idx-side_bin:bin_idx+side_bin+1] = 0
        sfdr_bin = np.argmax(spectrum_copy)
        spur_power = spectrum_copy[sfdr_bin]

        sfdr_db = sig_pwr_dbfs - 10 * np.log10(spur_power + 1e-20)

        # Calculate ENOB
        # SINAD = (Signal + Noise + Distortion) / (Noise + Distortion)
        sinad_power = signal_power / (noise_power + thd_power)
        sinad_db = 10 * np.log10(sinad_power)
        enob = (sinad_db - 1.76) / 6.02

        # Calculate NSD
        if calc_nsd and osr > 1:
            noise_bandwidth = fs / (2 * osr)
            nsd_dbfs_hz = noise_floor_db - 10 * np.log10(noise_bandwidth / 1)
        else:
            nsd_dbfs_hz = np.nan

        # Store metrics
        results['metrics'] = {
            'enob': enob,
            'sndr_db': sndr_db,
            'sfdr_db': sfdr_db,
            'snr_db': snr_db,
            'thd_db': thd_db,
            'sig_pwr_dbfs': sig_pwr_dbfs,
            'noise_floor_db': noise_floor_db,
            'nsd_dbfs_hz': nsd_dbfs_hz,
            'bin_idx': bin_idx,
            'bin_r': bin_r,
            'harmonic_bins': harmonic_bins
        }

    # ============= Add plot_data specific fields =============
    # Calculate additional fields needed by plot_spectrum
    n_search = n_fft // 2 // osr

    # Find spurious bin (max power in search range, excluding signal)
    spectrum_search_copy = spectrum_power[:n_search].copy()
    if 1 <= bin_idx < len(spectrum_search_copy) - side_bin:
        spectrum_search_copy[bin_idx-side_bin:bin_idx+side_bin+1] = 0
    spur_bin_idx = np.argmax(spectrum_search_copy)
    spur_power = spectrum_power[spur_bin_idx] if spur_bin_idx < len(spectrum_power) else 1e-20
    spur_db = 10 * np.log10(spur_power + 1e-20)

    # Add to results
    results['spur_bin_idx'] = spur_bin_idx
    results['spur_db'] = spur_db
    results['Nd2_inband'] = n_search
    results['N'] = n_fft
    results['M'] = n_runs
    results['fs'] = fs
    results['osr'] = osr
    results['nf_line_level'] = -(noise_floor_db + 10*np.log10(n_fft/2/osr))
    results['harmonics'] = []

    return results
