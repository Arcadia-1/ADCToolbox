"""Calculate spectrum data for ADC analysis - unified calculation engine."""

import numpy as np
from typing import Dict, Optional, Union
from ._prepare_fft_input import _prepare_fft_input
from ._find_fundamental import _find_fundamental
from ._find_harmonic_bins import _find_harmonic_bins
from ._align_spectrum_phase import _align_spectrum_phase
from ._exclude_bins import _exclude_bins_from_spectrum


def calculate_spectrum_data(
    data: np.ndarray,
    fs: float = 1.0,
    max_scale_range: Optional[float] = None,
    win_type: str = 'hann',
    side_bin: int = 1,
    osr: int = 1,
    n_thd: int = 5,
    nf_method: int = 2,
    assumed_sig_pwr_dbfs: Optional[float] = None,
    complex_spectrum: bool = False,
    cutoff_freq: float = 0
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """Calculate spectrum data for ADC analysis.

    Parameters
    ----------
    data : np.ndarray
        Input ADC data, shape (N,) or (M, N)
    fs : float
        Sampling frequency in Hz
    max_scale_range : float, optional
        Full scale range. If None, uses (max - min)
    win_type : str
        Window type: 'boxcar', 'hann', 'hamming', etc.
    side_bin : int
        Side bins to exclude around signal
    osr : int
        Oversampling ratio
    n_thd : int
        Number of harmonics for THD
    nf_method : int
        Noise floor method: 0=median, 1=trimmed mean, 2=exclude harmonics
    assumed_sig_pwr_dbfs : float, optional
        Override signal power (dBFS)
    complex_spectrum : bool
        If True, returns phase-aligned complex spectrum
    cutoff_freq : float
        High-pass cutoff frequency (Hz)

    Returns
    -------
    dict
        Contains 'metrics' and 'plot_data' dictionaries
    """
    # Preprocessing
    data_processed = _prepare_fft_input(data, max_scale_range, win_type)
    M, N = data_processed.shape
    n_half = N // 2
    n_search_inband = n_half // osr
    results = {}

    # Mode-specific FFT processing
    if complex_spectrum:
        # Complex spectrum: coherent averaging with phase alignment
        spec_coherent = np.zeros(N, dtype=complex)
        n_valid_runs = 0

        for run_idx in range(M):
            run_data = data_processed[run_idx, :N]
            if np.max(np.abs(run_data)) < 1e-10:
                continue

            # Compute FFT
            fft_data = np.fft.fft(run_data)
            fft_data[0] = 0  # Remove DC (MATLAB plotspec.m:282)

            # Find fundamental bin (MATLAB plotspec.m:284)
            fft_mag = np.abs(fft_data[:n_search_inband])
            bin_idx = np.argmax(fft_mag)

            # Guard against DC bin (MATLAB plotspec.m:286-289)
            if bin_idx <= 0:
                continue

            # Parabolic interpolation PER-RUN (MATLAB plotphase.m:144-152)
            # This is done BEFORE phase alignment
            if bin_idx > 0 and bin_idx < n_search_inband - 1:
                sig_e = np.log10(max(fft_mag[bin_idx], 1e-20))
                sig_l = np.log10(max(fft_mag[bin_idx - 1], 1e-20))
                sig_r = np.log10(max(fft_mag[bin_idx + 1], 1e-20))

                # Parabolic interpolation formula (MATLAB plotphase.m:149)
                delta = (sig_r - sig_l) / (2 * sig_e - sig_l - sig_r) / 2
                bin_r = bin_idx + delta

                if np.isnan(bin_r) or np.isinf(bin_r):
                    bin_r = float(bin_idx)
            else:
                bin_r = float(bin_idx)

            # Phase alignment (MATLAB plotspec.m:292-322)
            fft_aligned = _align_spectrum_phase(fft_data, bin_idx, bin_r, N)
            spec_coherent += fft_aligned
            n_valid_runs += 1

        # Apply coherent scaling: MATLAB plotspec.m:337
        # spec = abs(spec).^2/(N_fft^2)*16/ME^2
        # Keep complex spectrum (no division yet)
        spec_coherent_full = spec_coherent[:n_half]
        if cutoff_freq > 0:
            spec_coherent_full[:int(cutoff_freq / fs * N)] = 0

        # Convert to power spectrum with proper scaling
        # MATLAB: spec = abs(spec).^2/(N_fft^2)*16/ME^2
        # Factor of 16 comes from window power normalization
        if n_valid_runs > 0:
            spectrum_power_coherent = (np.abs(spec_coherent_full) ** 2) / (N ** 2) * 16 / (n_valid_runs ** 2)
        else:
            spectrum_power_coherent = (np.abs(spec_coherent_full) ** 2) / (N ** 2) * 16

        # Calculate noise floor for complex mode (used for polar plot)
        # MATLAB plotphase.m:195-200
        # Use amplitude (20*log10) to match plot_spectrum_polar.py line 87
        mag_db = 20 * np.log10(np.abs(spec_coherent_full) + 1e-20)

        # Use 1st percentile of entire spectrum (MATLAB: spec_sort(ceil(length(spec_sort)*0.01)))
        mag_db_sorted = np.sort(mag_db)
        percentile_idx = int(np.ceil(len(mag_db_sorted) * 0.01))
        percentile_idx = max(0, min(percentile_idx, len(mag_db_sorted) - 1))
        noise_floor_db = mag_db_sorted[percentile_idx]

        # Default to -100 if infinite (MATLAB: if(isinf(minR)) minR = -100; end)
        noise_floor_db = -100 if np.isinf(noise_floor_db) else noise_floor_db

        # Store complex spectrum for polar plot (normalized to noise floor)
        results.update({
            'complex_spec_coherent': spec_coherent_full,
            'minR_dB': noise_floor_db,
            'bin_idx': np.argmax(spectrum_power_coherent[:n_search_inband]),
            'N': N
        })

        # Use power spectrum for metrics calculation
        spectrum_power = spectrum_power_coherent
        spec_mag_db = 10 * np.log10(spectrum_power + 1e-20)

    else:
        # Power spectrum: traditional power averaging
        spectrum_sum = np.zeros(N)
        for run_idx in range(M):
            fft_data = np.fft.fft(data_processed[run_idx, :N])
            spectrum_sum += np.abs(fft_data) ** 2

        spectrum_sum[0] = 0  # Remove DC
        spectrum_power = spectrum_sum[:n_half] / (N ** 2) * 16 / M

        if cutoff_freq > 0:
            spectrum_power[:int(cutoff_freq / fs * N)] = 0

        spec_mag_db = 10 * np.log10(spectrum_power + 1e-20)

    # Common post-processing
    freq = np.arange(n_half) * fs / N
    bin_idx, bin_r = _find_fundamental(spectrum_power, N, osr, method='power')

    results.update({
        'freq': freq,
        'spec_mag_db': spec_mag_db,
        'spec_db': spec_mag_db,
        'bin_idx': bin_idx,
        'sig_bin_start': max(bin_idx - side_bin, 0),
        'sig_bin_end': min(bin_idx + side_bin + 1, len(freq)),
        'bin_r': bin_r
    })

    # Temporary noise floor for NSD (will be updated after SNR calculation)
    spectrum_search = spectrum_power[:n_search_inband].copy()
    if 1 <= bin_idx < len(spectrum_search) - side_bin:
        spectrum_search[bin_idx-side_bin:bin_idx+side_bin+1] = 0
    noise_power_percentile = np.percentile(spectrum_search[spectrum_search > 0], 1)
    temp_noise_floor_db = 10 * np.log10(noise_power_percentile + 1e-20)

    # ============= Calculate metrics =============
    # Signal power
    sig_start = max(bin_idx - side_bin, 0)
    sig_end = min(bin_idx + side_bin + 1, min(n_search_inband, len(spectrum_power)))
    signal_power = max(np.sum(spectrum_power[sig_start:sig_end]), 1e-15)
    sig_pwr_dbfs = 10 * np.log10(signal_power)

    # Override with assumed signal if provided
    if assumed_sig_pwr_dbfs is not None and not np.isnan(assumed_sig_pwr_dbfs):
        signal_power = 10 ** (assumed_sig_pwr_dbfs / 10)
        sig_pwr_dbfs = assumed_sig_pwr_dbfs

    # THD power (include side bins)
    harmonic_bins = _find_harmonic_bins(bin_r, n_thd, N)
    thd_power = 0
    for h_idx in range(1, n_thd):
        h_bin = int(round(harmonic_bins[h_idx]))
        if h_bin < len(spectrum_power):
            h_start = max(h_bin - side_bin, 0)
            h_end = min(h_bin + side_bin + 1, len(spectrum_power))
            thd_power += np.sum(spectrum_power[h_start:h_end])
    thd_power = max(thd_power, 1e-15)

    # Noise power (method-dependent)
    if nf_method == 0:
        # Median-based (robust to spurs)
        noise_power = np.median(spectrum_power[:n_search_inband]) / np.sqrt((1 - 2/(9*M))**3) * n_search_inband
    elif nf_method == 1:
        # Trimmed mean (removes top/bottom 5%)
        spec_sorted = np.sort(spectrum_power[:n_search_inband])
        start_idx = int(n_search_inband * 0.05)
        end_idx = int(n_search_inband * 0.95)
        noise_power = np.mean(spec_sorted[start_idx:end_idx]) * n_search_inband
    else:
        # Exclude harmonics (most accurate)
        noise_spectrum = _exclude_bins_from_spectrum(spectrum_power, bin_idx, harmonic_bins, side_bin, n_search_inband)
        noise_power = np.sum(noise_spectrum)
    noise_power = max(noise_power, 1e-15)

    # Calculate metrics
    sndr_db = 10 * np.log10(signal_power / (noise_power + thd_power))
    snr_db = 10 * np.log10(signal_power / noise_power)
    thd_db = 10 * np.log10(thd_power)
    enob = (sndr_db - 1.76) / 6.02

    # SFDR
    spectrum_copy = spectrum_power.copy()
    if 1 <= bin_idx < len(spectrum_copy):
        spectrum_copy[bin_idx-side_bin:bin_idx+side_bin+1] = 0
    spur_power = spectrum_copy[np.argmax(spectrum_copy)]
    sfdr_db = sig_pwr_dbfs - 10 * np.log10(spur_power + 1e-20)

    # Noise floor (MATLAB: NF = SNR - pwr)
    noise_floor_db = snr_db - sig_pwr_dbfs

    # NSD (Noise Spectral Density) - negative sign because NF represents depth below FS
    nsd_dbfs_hz = -(noise_floor_db + 10 * np.log10(fs / (2 * osr)))

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

    # ============= Plot data =============
    spectrum_search_copy = spectrum_power[:n_search_inband].copy()
    if 1 <= bin_idx < len(spectrum_search_copy) - side_bin:
        spectrum_search_copy[bin_idx-side_bin:bin_idx+side_bin+1] = 0
    spur_bin_idx = np.argmax(spectrum_search_copy)
    spur_db = 10 * np.log10(spectrum_power[spur_bin_idx] + 1e-20) if spur_bin_idx < len(spectrum_power) else -200

    results['plot_data'] = {
        'spec_db': spec_mag_db,
        'freq': freq,
        'bin_idx': bin_idx,
        'sig_bin_start': sig_start,
        'sig_bin_end': sig_end,
        'spur_bin_idx': spur_bin_idx,
        'spur_db': spur_db,
        'Nd2_inband': n_search_inband,
        'N': N,
        'M': M,
        'fs': fs,
        'osr': osr,
        'nf_line_level': -(noise_floor_db + 10*np.log10(n_search_inband)),
        'harmonics': [],
        'is_coherent': complex_spectrum  # Flag to indicate coherent vs power averaging
    }

    return results
