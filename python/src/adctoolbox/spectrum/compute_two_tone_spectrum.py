"""Calculate two-tone spectrum data - pure calculation module.

This module provides the core calculation engine for two-tone IMD analysis,
following the modular architecture pattern used throughout the spectrum package.
"""

import numpy as np

from adctoolbox.fundamentals.frequency import fold_bin_to_nyquist
from adctoolbox.spectrum._prepare_fft_input import _prepare_fft_input
from adctoolbox.spectrum._align_spectrum_phase import _align_spectrum_phase

def compute_two_tone_spectrum(
    data: np.ndarray,
    fs: float = 1.0,
    max_scale_range: float | None = None,
    win_type: str = 'hann',
    side_bin: int = 1,
    harmonic: int = 7,
    coherent_averaging: bool = False
) -> dict[str, any]:
    """
    Calculate two-tone spectrum data with IMD analysis.

    Pure calculation function - no plotting or side effects.
    Follows the modular architecture pattern.

    Parameters
    ----------
    data : np.ndarray
        ADC output data, shape (M, N) for M runs or (N,) for single run
    fs : float, optional
        Sampling frequency (Hz), default: 1.0
    max_scale_range : float, optional
        Maximum code range, default: max-min of data
    win_type : str, optional
        Window type: 'hann', 'blackman', 'hamming', 'boxcar', default: 'hann'
    side_bin : int, optional
        Side bins to include in signal power, default: 1
    harmonic : int, optional
        Number of harmonic orders to calculate for IMD products, default: 7
    coherent_averaging : bool, optional
        If True, performs coherent averaging with phase alignment, default: False

    Returns
    -------
    dict
        Dictionary containing:
        - 'metrics': Performance metrics (enob, sndr, sfdr, snr, thd, etc.)
        - 'plot_data': Data for plotting (freq, spec_db, bin1, bin2, etc.)
        - 'imd_bins': IMD product bin locations
    """
    # Preprocessing using shared helper
    data_processed = _prepare_fft_input(data, max_scale_range)
    M, N = data_processed.shape
    n_half = N // 2

    freq = np.arange(n_half) / N * fs

    if coherent_averaging:
        # Step 1: First find reliable bin locations using power averaging
        spectrum_sum_prelim = np.zeros(N)
        n_valid_prelim = 0

        for i in range(M):
            run_data = data_processed[i, :]
            if np.max(np.abs(run_data)) < 1e-10:
                continue
            spectrum_sum_prelim += np.abs(np.fft.fft(run_data))**2
            n_valid_prelim += 1

        if n_valid_prelim == 0:
            raise ValueError("No valid data runs")

        # Compute magnitude spectrum for bin finding and parabolic interpolation
        spectrum_mag_prelim = np.sqrt(spectrum_sum_prelim / n_valid_prelim) / N * 2
        spectrum_mag_prelim = spectrum_mag_prelim[:n_half]
        spectrum_mag_prelim[:side_bin] = 0

        # Find bins from magnitude-averaged spectrum (reliable bin detection)
        bin1 = np.argmax(spectrum_mag_prelim)
        spectrum_temp_prelim = spectrum_mag_prelim.copy()
        spectrum_temp_prelim[bin1] = 0
        bin2 = np.argmax(spectrum_temp_prelim)

        # Ensure bin1 < bin2
        if bin1 > bin2:
            bin1, bin2 = bin2, bin1

        # Compute parabolic interpolation using log10(magnitude) (same as single-tone)
        if bin1 > 0 and bin1 < n_half - 1:
            sig_e = np.log10(max(spectrum_mag_prelim[bin1], 1e-20))
            sig_l = np.log10(max(spectrum_mag_prelim[bin1 - 1], 1e-20))
            sig_r = np.log10(max(spectrum_mag_prelim[bin1 + 1], 1e-20))

            delta = (sig_r - sig_l) / (2 * sig_e - sig_l - sig_r) / 2
            bin_r1 = bin1 + delta

            if np.isnan(bin_r1) or np.isinf(bin_r1):
                bin_r1 = float(bin1)
        else:
            bin_r1 = float(bin1)

        # Step 2: Coherent averaging using fixed bins and fixed bin_r1
        spec_coherent = np.zeros(N, dtype=complex)
        n_valid_runs = 0

        for i in range(M):
            run_data = data_processed[i, :]
            if np.max(np.abs(run_data)) < 1e-10:
                continue

            # Compute FFT
            fft_data = np.fft.fft(run_data)
            fft_data[0] = 0  # Remove DC

            # Align phase to F1 using fixed bin_r1 from power-averaged spectrum
            fft_aligned = _align_spectrum_phase(fft_data, bin1, bin_r1, N)

            # Accumulate complex spectrum
            spec_coherent += fft_aligned
            n_valid_runs += 1

        # Normalize and convert to power spectrum
        spec_coherent /= n_valid_runs
        # Window is already power-normalized in _prepare_fft_input
        # Factor of 4 = 2 (single-sided) * 2 (dBFS reference: full-scale sine power = 0.5)
        spectrum_power = (np.abs(spec_coherent[:n_half])**2) * 4 / (N**2)
        spectrum_power[:side_bin] = 0  # Remove DC and low-frequency bins

        # Re-find bins from final coherent-averaged spectrum for accurate measurement
        bin1 = np.argmax(spectrum_power)
        spectrum_temp = spectrum_power.copy()
        spectrum_temp[bin1] = 0
        bin2 = np.argmax(spectrum_temp)

        # Ensure bin1 < bin2
        if bin1 > bin2:
            bin1, bin2 = bin2, bin1

    else:
        # Average spectrum over multiple runs (power averaging)
        spectrum_sum = np.zeros(N)
        n_valid_runs = 0

        for i in range(M):
            run_data = data_processed[i, :]
            if np.max(np.abs(run_data)) < 1e-10:
                continue

            # Accumulate power spectrum
            spectrum_sum += np.abs(np.fft.fft(run_data))**2
            n_valid_runs += 1

        if n_valid_runs == 0:
            raise ValueError("No valid data runs")

        # Normalize spectrum (single-sided power spectrum)
        # Window is already power-normalized in _prepare_fft_input
        # Factor of 4 = 2 (single-sided) * 2 (dBFS reference: full-scale sine power = 0.5)
        spectrum_sum = spectrum_sum[:n_half]
        spectrum_sum[:side_bin] = 0  # Remove DC and low-frequency bins
        spectrum_power = spectrum_sum / (N**2) * 4 / n_valid_runs

        # Find two tones from power-averaged spectrum
        bin1 = np.argmax(spectrum_power)
        spectrum_temp = spectrum_power.copy()
        spectrum_temp[bin1] = 0
        bin2 = np.argmax(spectrum_temp)

        # Ensure bin1 < bin2
        if bin1 > bin2:
            bin1, bin2 = bin2, bin1

    # ========== Calculate signal powers ==========
    sig1_start = max(bin1 - side_bin, 0)
    sig1_end = min(bin1 + side_bin + 1, n_half)
    sig2_start = max(bin2 - side_bin, 0)
    sig2_end = min(bin2 + side_bin + 1, n_half)

    sig1_power = np.sum(spectrum_power[sig1_start:sig1_end])
    sig2_power = np.sum(spectrum_power[sig2_start:sig2_end])

    pwr1_dbfs = 10 * np.log10(sig1_power + 1e-20)
    pwr2_dbfs = 10 * np.log10(sig2_power + 1e-20)
    total_signal_power = sig1_power + sig2_power

    # ========== Remove signal bins for noise calculation ==========
    spectrum_noise = spectrum_power.copy()
    spectrum_noise[sig1_start:sig1_end] = 0
    spectrum_noise[sig2_start:sig2_end] = 0

    noise_power = np.sum(spectrum_noise)

    # ========== Find max spur ==========
    spur_bin = np.argmax(spectrum_noise)
    spur_start = max(spur_bin - side_bin, 0)
    spur_end = min(spur_bin + side_bin + 1, n_half)
    spur_power = np.sum(spectrum_noise[spur_start:spur_end])

    # ========== Calculate IMD2 products ==========
    # IMD2: f1+f2 and f2-f1
    bin_imd2_sum = fold_bin_to_nyquist(bin1 + bin2, N)
    bin_imd2_diff = fold_bin_to_nyquist(bin2 - bin1, N)

    imd2_sum_power = np.sum(spectrum_power[max(bin_imd2_sum, 0):min(bin_imd2_sum + 3, n_half)])
    imd2_diff_power = np.sum(spectrum_power[max(bin_imd2_diff, 0):min(bin_imd2_diff + 3, n_half)])
    imd2_total_power = imd2_sum_power + imd2_diff_power

    # ========== Calculate IMD3 products ==========
    # IMD3: 2f1+f2, f1+2f2, 2f1-f2, 2f2-f1
    bin_imd3_2f1_plus_f2 = fold_bin_to_nyquist(2 * bin1 + bin2, N)
    bin_imd3_f1_plus_2f2 = fold_bin_to_nyquist(bin1 + 2 * bin2, N)
    bin_imd3_2f1_minus_f2 = fold_bin_to_nyquist(2 * bin1 - bin2, N)
    bin_imd3_2f2_minus_f1 = fold_bin_to_nyquist(2 * bin2 - bin1, N)

    imd3_power_1 = np.sum(spectrum_power[max(bin_imd3_2f1_plus_f2, 0):min(bin_imd3_2f1_plus_f2 + 3, n_half)])
    imd3_power_2 = np.sum(spectrum_power[max(bin_imd3_f1_plus_2f2, 0):min(bin_imd3_f1_plus_2f2 + 3, n_half)])
    imd3_power_3 = np.sum(spectrum_power[max(bin_imd3_2f1_minus_f2, 0):min(bin_imd3_2f1_minus_f2 + 3, n_half)])
    imd3_power_4 = np.sum(spectrum_power[max(bin_imd3_2f2_minus_f1, 0):min(bin_imd3_2f2_minus_f1 + 3, n_half)])
    imd3_total_power = imd3_power_1 + imd3_power_2 + imd3_power_3 + imd3_power_4

    # ========== Calculate all harmonic/IMD product bins for plotting ==========
    harmonic_products = []
    collision_threshold = 2 * side_bin

    for i in range(2, harmonic + 1):
        for jj in range(i + 1):
            # Positive combination: jj*f1 + (i-jj)*f2
            b = fold_bin_to_nyquist(bin1 * jj + bin2 * (i - jj), N)

            # Skip if this harmonic/IMD product collides with fundamental tones
            # Use same threshold as single-tone analysis
            if abs(b - bin1) <= collision_threshold or abs(b - bin2) <= collision_threshold:
                continue

            # Skip if this harmonic aliases to DC (bin 0)
            if b <= side_bin:
                continue

            if 0 < b < n_half:
                harmonic_products.append({
                    'bin': b,
                    'freq': freq[b],
                    'power_db': 10 * np.log10(spectrum_power[b] + 1e-20),
                    'order': i
                })

            # Negative combination 1: -jj*f1 + (i-jj)*f2
            if -bin1 * jj + bin2 * (i - jj) > 0:
                b = fold_bin_to_nyquist(-bin1 * jj + bin2 * (i - jj), N)

                # Skip if collision with fundamentals
                if abs(b - bin1) <= collision_threshold or abs(b - bin2) <= collision_threshold:
                    continue

                # Skip if aliases to DC
                if b <= side_bin:
                    continue

                if 0 < b < n_half:
                    harmonic_products.append({
                        'bin': b,
                        'freq': freq[b],
                        'power_db': 10 * np.log10(spectrum_power[b] + 1e-20),
                        'order': i
                    })

            # Negative combination 2: jj*f1 - (i-jj)*f2
            if bin1 * jj - bin2 * (i - jj) > 0:
                b = fold_bin_to_nyquist(bin1 * jj - bin2 * (i - jj), N)

                # Skip if collision with fundamentals
                if abs(b - bin1) <= collision_threshold or abs(b - bin2) <= collision_threshold:
                    continue

                # Skip if aliases to DC
                if b <= side_bin:
                    continue

                if 0 < b < n_half:
                    harmonic_products.append({
                        'bin': b,
                        'freq': freq[b],
                        'power_db': 10 * np.log10(spectrum_power[b] + 1e-20),
                        'order': i
                    })

    # ========== Calculate THD and remove all distortion products for noise calculation ==========
    thd_power = 0
    spectrum_thd = spectrum_noise.copy()

    # Remove all identified IMD products from noise floor calculation
    for product in harmonic_products:
        b = product['bin']
        b_start = max(b - 1, 0)
        b_end = min(b + 2, n_half)
        thd_power += np.sum(spectrum_thd[b_start:b_end])
        spectrum_thd[b_start:b_end] = 0

    noise_power_final = np.sum(spectrum_thd)

    # ========== Calculate metrics ==========
    sndr_dbc = 10 * np.log10(total_signal_power / (noise_power + 1e-20))
    sfdr_dbc = 10 * np.log10(total_signal_power / (spur_power + 1e-20))
    snr_dbc = 10 * np.log10(total_signal_power / (noise_power_final + 1e-20))
    thd_dbc = 10 * np.log10(thd_power / (total_signal_power + 1e-20))
    enob = (sndr_dbc - 1.76) / 6.02
    imd2_dbc = 10 * np.log10(total_signal_power / (imd2_total_power + 1e-20))
    imd3_dbc = 10 * np.log10(total_signal_power / (imd3_total_power + 1e-20))
    sig_pwr_dbfs = 10 * np.log10(total_signal_power)
    noise_floor_db = sig_pwr_dbfs - snr_dbc
    nsd_dbfs_hz = noise_floor_db - 10 * np.log10(fs / 2)  # Noise spectral density

    # ========== Prepare return data ==========
    metrics = {
        'enob': enob,
        'sndr_dbc': sndr_dbc,
        'sfdr_dbc': sfdr_dbc,
        'snr_dbc': snr_dbc,
        'thd_dbc': thd_dbc,
        'signal_power_1_dbfs': pwr1_dbfs,
        'signal_power_2_dbfs': pwr2_dbfs,
        'noise_floor_db': noise_floor_db,
        'nsd_dbfs_hz': nsd_dbfs_hz,
        'imd2_dbc': imd2_dbc,
        'imd3_dbc': imd3_dbc
    }

    plot_data = {
        'freq': freq,
        'spec_db': 10 * np.log10(spectrum_power + 1e-20),
        'spectrum_power': spectrum_power,
        'bin1': bin1,
        'bin2': bin2,
        'freq1': freq[bin1],
        'freq2': freq[bin2],
        'N': N,
        'M': n_valid_runs,
        'fs': fs,
        'harmonic_products': harmonic_products,
        'coherent_averaging': coherent_averaging
    }

    imd_bins = {
        'imd2_sum': bin_imd2_sum,
        'imd2_diff': bin_imd2_diff,
        'imd3_2f1_plus_f2': bin_imd3_2f1_plus_f2,
        'imd3_f1_plus_2f2': bin_imd3_f1_plus_2f2,
        'imd3_2f1_minus_f2': bin_imd3_2f1_minus_f2,
        'imd3_2f2_minus_f1': bin_imd3_2f2_minus_f1
    }

    return {
        'metrics': metrics,
        'plot_data': plot_data,
        'imd_bins': imd_bins
    }
