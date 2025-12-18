"""Calculate spectrum data for ADC analysis - unified calculation engine."""

import numpy as np
from typing import Dict, Optional, Union
from adctoolbox.spectrum._prepare_fft_input import _prepare_fft_input
from adctoolbox.spectrum._find_fundamental import _find_fundamental
from adctoolbox.spectrum._find_harmonic_bins import _find_harmonic_bins
from adctoolbox.spectrum._align_spectrum_phase import _align_spectrum_phase
from adctoolbox.spectrum._exclude_bins import _exclude_bins_from_spectrum


def compute_spectrum(
    data: np.ndarray,
    fs: float = 1.0,
    max_scale_range: Optional[float] = None,
    win_type: str = 'hann',
    side_bin: int = 1,
    osr: int = 1,
    n_thd: int = 5,
    nf_method: int = 2,
    assumed_sig_pwr_dbfs: Optional[float] = None,
    coherent_averaging: bool = False,
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
    coherent_averaging : bool
        If True, performs coherent averaging with phase alignment
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

    # Power correction factor for proper dBFS scaling
    # Factor of 16 = 2 (single-sided) * 2 (window power norm) * 2 (peak-to-RMS) * 2 (dBFS ref)
    # This ensures: full-scale sine (peak=1) -> power = 0 dBFS
    power_correction = 16.0

    # Mode-specific FFT processing
    if coherent_averaging:
        # Complex spectrum: coherent averaging with phase alignment
        spec_coherent = np.zeros(N, dtype=complex)
        n_valid_runs = 0
        original_fundamental_phase = None  # Store original phase before alignment

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

            # Store original fundamental phase from first valid run
            if original_fundamental_phase is None:
                original_fundamental_phase = np.angle(fft_data[bin_idx])

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
        # Since window is power-normalized, all windows use same correction factor
        if n_valid_runs > 0:
            spectrum_power_coherent = (np.abs(spec_coherent_full) ** 2) / (N ** 2) * power_correction / (n_valid_runs ** 2)
            # Normalize complex spectrum for polar plot (amplitude domain)
            spec_coherent_normalized = spec_coherent_full / N / n_valid_runs * np.sqrt(power_correction)
        else:
            spectrum_power_coherent = (np.abs(spec_coherent_full) ** 2) / (N ** 2) * power_correction
            spec_coherent_normalized = spec_coherent_full / N * np.sqrt(power_correction)

        # Calculate noise floor for complex mode (used for polar plot)
        # MATLAB plotphase.m:195-200
        # Use amplitude (20*log10) to match plot_spectrum_polar.py line 87
        mag_db = 20 * np.log10(np.abs(spec_coherent_normalized) + 1e-20)

        # Use 1st percentile of entire spectrum (MATLAB: spec_sort(ceil(length(spec_sort)*0.01)))
        mag_db_sorted = np.sort(mag_db)
        percentile_idx = int(np.ceil(len(mag_db_sorted) * 0.01))
        percentile_idx = max(0, min(percentile_idx, len(mag_db_sorted) - 1))
        noise_floor_dbfs = mag_db_sorted[percentile_idx]

        # Default to -100 if infinite (MATLAB: if(isinf(minR)) minR = -100; end)
        noise_floor_dbfs = -100 if np.isinf(noise_floor_dbfs) else noise_floor_dbfs

        # Calculate harmonic bins for phase extraction (need bin_r for this)
        temp_bin_idx = np.argmax(spectrum_power_coherent[:n_search_inband])
        temp_bin_r = _find_fundamental(spectrum_power_coherent, N, osr, method='power')[1]
        temp_harmonic_bins = _find_harmonic_bins(temp_bin_r, n_thd, N)

        # Store complex spectrum for polar plot (properly normalized)
        # Calculate HD2 and HD3 phases RELATIVE to fundamental
        # After _align_spectrum_phase(), the fundamental is at 0° and harmonics are rotated by h*φ₁
        # For memoryless nonlinearity: HD2 should be at 0°, HD3 should be at 0° or 180°
        # The aligned phase directly represents the relative phase (memory effect)
        hd2_phase_deg = 0
        hd3_phase_deg = 0

        if len(temp_harmonic_bins) > 1:
            hd2_bin = int(round(temp_harmonic_bins[1]))
            if hd2_bin < len(spec_coherent_normalized):
                # After alignment: HD2 phase = original_HD2_phase - 2*fundamental_phase
                # This is already the relative phase we want
                hd2_phase_rad = np.angle(spec_coherent_normalized[hd2_bin])
                hd2_phase_deg = np.degrees(hd2_phase_rad)

        if len(temp_harmonic_bins) > 2:
            hd3_bin = int(round(temp_harmonic_bins[2]))
            if hd3_bin < len(spec_coherent_normalized):
                # After alignment: HD3 phase = original_HD3_phase - 3*fundamental_phase
                # This is already the relative phase we want
                hd3_phase_rad = np.angle(spec_coherent_normalized[hd3_bin])
                hd3_phase_deg = np.degrees(hd3_phase_rad)

        results.update({
            'complex_spec_coherent': spec_coherent_normalized,
            'minR_dB': noise_floor_dbfs,
            'bin_idx': temp_bin_idx,
            'N': N,
            'hd2_phase_deg': hd2_phase_deg,
            'hd3_phase_deg': hd3_phase_deg
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
        spectrum_power = spectrum_sum[:n_half] / (N ** 2) * power_correction / M

        if cutoff_freq > 0:
            spectrum_power[:int(cutoff_freq / fs * N)] = 0

        spec_mag_db = 10 * np.log10(spectrum_power + 1e-20)

    # Common post-processing
    freq = np.arange(n_half) * fs / N
    bin_idx, bin_r = _find_fundamental(spectrum_power, N, osr, method='power')
    n_search_inband = n_half // osr

    # Use spectrum without normalization - dBFS reference is full scale (FSR = 0 dB)
    # For FSR=1.0V: Full scale=0 dBFS, Signal(0.5V)=-6 dBFS, etc.
    # This ensures all markers and values are consistent

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
    temp_noise_floor_dbfs = 10 * np.log10(noise_power_percentile + 1e-20)

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
    hd2_power = 0
    hd3_power = 0

    # Track which bins we've already counted to avoid double-counting when harmonics alias to same frequency
    counted_bins = set()
    collided_harmonics = []  # Track which harmonics collide with fundamental

    for h_idx in range(1, n_thd):
        h_bin = int(round(harmonic_bins[h_idx]))

        # Skip if this harmonic aliases to the fundamental (happens when h*f_in wraps to f_in)
        # Collision occurs when the side bins overlap: abs(h_bin - bin_idx) <= 2*side_bin
        collision_threshold = 2 * side_bin
        if abs(h_bin - bin_idx) <= collision_threshold:
            collided_harmonics.append(h_idx + 1)  # Store harmonic number (HD2 = h_idx+1)
            continue

        # Skip if this harmonic aliases to DC (bin 0) - DC is excluded from spectrum analysis
        if h_bin <= side_bin:
            continue

        # Skip if we've already counted this bin (happens when multiple harmonics alias to same frequency)
        if h_bin in counted_bins:
            continue

        if h_bin < len(spectrum_power):
            h_start = max(h_bin - side_bin, 0)
            h_end = min(h_bin + side_bin + 1, len(spectrum_power))
            h_power = np.sum(spectrum_power[h_start:h_end])
            thd_power += h_power

            # Mark these bins as counted
            for b in range(h_start, h_end):
                counted_bins.add(b)

            # Store HD2 and HD3 individually
            if h_idx == 1:
                hd2_power = h_power
            elif h_idx == 2:
                hd3_power = h_power
    thd_power = max(thd_power, 1e-15)
    hd2_power = max(hd2_power, 1e-15)
    hd3_power = max(hd3_power, 1e-15)

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
    hd2_db = 10 * np.log10(hd2_power)
    hd3_db = 10 * np.log10(hd3_power)
    enob = (sndr_db - 1.76) / 6.02

    # SFDR (Limited to in-band search when OSR > 1)
    # For OSR > 1, only search for spurs within signal band [0, Fs/2/OSR]
    # This matches MATLAB plotspec behavior for oversampled ADCs (Delta-Sigma)
    spectrum_copy = spectrum_power[:n_search_inband].copy()

    # Exclude signal and sidebins from search
    sig_start_inband = max(bin_idx - side_bin, 0)
    sig_end_inband = min(bin_idx + side_bin + 1, n_search_inband)
    if sig_start_inband < sig_end_inband:
        spectrum_copy[sig_start_inband:sig_end_inband] = 0

    # Find maximum spur within in-band range
    if len(spectrum_copy) > 0:
        spur_bin_idx = np.argmax(spectrum_copy)

        # Calculate spur's summed power (including side bins)
        spur_start = max(spur_bin_idx - side_bin, 0)
        spur_end = min(spur_bin_idx + side_bin + 1, n_search_inband)

        if spur_start < spur_end:
            spur_power_summed = np.sum(spectrum_power[spur_start:spur_end])
        else:
            spur_power_summed = 1e-20

        spur_power = spur_power_summed  # Use summed power instead of peak bin
    else:
        spur_power = 1e-20
    sfdr_db = sig_pwr_dbfs - 10 * np.log10(spur_power + 1e-20)

    # Noise floor
    noise_floor_dbfs = sig_pwr_dbfs - snr_db

    # NSD (Noise Spectral Density)
    nsd_dbfs_hz = noise_floor_dbfs - 10 * np.log10(fs / (2 * osr))

    results['metrics'] = {
        'enob': enob,
        'sndr_db': sndr_db,
        'sfdr_db': sfdr_db,
        'snr_db': snr_db,
        'thd_db': thd_db,
        'hd2_db': hd2_db,
        'hd3_db': hd3_db,
        'sig_pwr_dbfs': sig_pwr_dbfs,
        'noise_floor_dbfs': noise_floor_dbfs,
        'nsd_dbfs_hz': nsd_dbfs_hz,
        'bin_idx': bin_idx,
        'bin_r': bin_r,
        'harmonic_bins': harmonic_bins,
        'collided_harmonics': collided_harmonics  # List of harmonic numbers that collide with fundamental
    }

    # ============= Plot data =============
    spectrum_search_copy = spectrum_power[:n_search_inband].copy()
    if 1 <= bin_idx < len(spectrum_search_copy) - side_bin:
        spectrum_search_copy[bin_idx-side_bin:bin_idx+side_bin+1] = 0
    spur_bin_idx = np.argmax(spectrum_search_copy)
    
    # Get spur_db from center bin only (for plot consistency)
    # SFDR metric still uses summed power, but plot shows single bin value
    if spur_bin_idx < len(spectrum_power):
        spur_db = 10 * np.log10(spectrum_power[spur_bin_idx] + 1e-20)
    else:
        spur_db = -200

    # Build harmonics list for plotting
    harmonics_list = []
    collision_threshold = 2 * side_bin
    for h_idx in range(1, n_thd):
        h_bin = int(round(harmonic_bins[h_idx]))

        # Skip if this harmonic aliases to the fundamental (collision makes plotting meaningless)
        # Use same threshold as THD calculation: abs(h_bin - bin_idx) <= 2*side_bin
        if abs(h_bin - bin_idx) <= collision_threshold:
            continue

        # Skip if this harmonic aliases to DC (bin 0) - DC is excluded from spectrum analysis
        if h_bin <= side_bin:
            continue

        if h_bin < len(spectrum_power):
            h_start = max(h_bin - side_bin, 0)
            h_end = min(h_bin + side_bin + 1, len(spectrum_power))
            h_power = np.sum(spectrum_power[h_start:h_end])
            h_power_db = 10 * np.log10(h_power + 1e-20)
            h_freq = h_bin * fs / N
            harmonics_list.append({
                'harmonic_num': h_idx + 1,  # HD2 is harmonic 2, HD3 is harmonic 3, etc.
                'freq': h_freq,
                'power_db': h_power_db
            })

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
        'nf_line_level': noise_floor_dbfs - 10*np.log10(n_search_inband),
        'harmonics': harmonics_list,
        'is_coherent': coherent_averaging  # Flag to indicate coherent vs power averaging
    }

    return results
