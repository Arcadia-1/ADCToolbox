"""Calculate spectrum data for ADC analysis - unified calculation engine."""

import numpy as np

from adctoolbox.spectrum._prepare_fft_input import _prepare_fft_input
from adctoolbox.spectrum._locate_fundamental import _locate_fundamental

from adctoolbox.spectrum._harmonics import _locate_harmonic_bins
from adctoolbox.spectrum._harmonics import _calculate_harmonic_power
from adctoolbox.spectrum._harmonics import _extract_highest_spur

from adctoolbox.spectrum._spectrum_averaging import _power_average
from adctoolbox.spectrum._spectrum_averaging import _coherent_average

from adctoolbox.spectrum._window import _create_window
from adctoolbox.spectrum._window import _calculate_power_correction
from adctoolbox.spectrum._window import _get_default_side_bin

from adctoolbox.spectrum._estimate_noise_power import _estimate_noise_power

def compute_spectrum(
    data: np.ndarray,
    fs: float = 1.0,
    max_scale_range: float | list[float] | tuple[float | None | list[float]] = None,
    win_type: str = 'hann',
    side_bin: int | None = None,
    osr: int = 1,
    max_harmonic: int = 5,
    nf_method: int = 2,
    assumed_sig_pwr_dbfs: float | None = None,
    coherent_averaging: bool = False,
    cutoff_freq: float = 0,
    verbose: int = 0
) -> dict[str, np.ndarray | float | dict]:
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
    side_bin : int, optional
        Side bins to exclude around signal. If None, automatically determined based on:
        - Coherent signal (error < 0.01): ceil(enbw)
        - Non-coherent signal: ceil(2*enbw) + 1
        where enbw is the window's equivalent noise bandwidth factor
    osr : int
        Oversampling ratio
    max_harmonic : int
        Maximum harmonic number for THD (default: 5 means harmonics 2-5)
    nf_method : int
        Noise floor method: 0=median, 1=trimmed mean, 2=exclude harmonics
    assumed_sig_pwr_dbfs : float, optional
        Override signal power (dBFS)
    coherent_averaging : bool
        If True, performs coherent averaging with phase alignment
    cutoff_freq : float
        High-pass cutoff frequency (Hz)
    verbose : int
        Verbosity level

    Returns
    -------
    dict
        Contains 'metrics' and 'plot_data' dictionaries
    """
    # Preprocessing: data validation, DC removal, normalization
    data_normalized = _prepare_fft_input(data, max_scale_range)
    M, N = data_normalized.shape
    n_inband = N // 2 // osr

    # Window function handling
    window_vector, window_gain, equiv_noise_bw_factor = _create_window(win_type, N)
    data_windowed = data_normalized * window_vector

    # Spectrum averaging: coherent or power (returns raw FFT results)
    if coherent_averaging:
        power_spectrum, complex_spectrum = _coherent_average(data_windowed, osr)
    else:
        power_spectrum, complex_spectrum = _power_average(data_windowed)

    # Apply power correction to spectrum
    power_correction = _calculate_power_correction(window_gain)
    power_spectrum *= power_correction / equiv_noise_bw_factor
    if complex_spectrum is not None:
        # Complex spectrum (voltage) correction: sqrt(power_correction)
        # Note: ENBW correction NOT applied to complex_spectrum (it represents signal voltage)
        complex_spectrum = complex_spectrum * np.sqrt(power_correction)

    power_spectrum_db = 10 * np.log10(power_spectrum + 1e-20)
    freq = np.arange(len(power_spectrum)) * (fs / N)

    # Cut the spectrum below cutoff_freq (remove low-frequency content)
    if cutoff_freq > 0:
        cutoff_bin = min(int(np.ceil(cutoff_freq / fs * N)), len(power_spectrum))
        if cutoff_freq >= (fs / 2):
            import warnings
            warnings.warn(f"cutoff_freq [{cutoff_freq} Hz] exceeds Nyquist. Spectrum will be empty.")
        if verbose >= 2:
            print(f"Applying cutoff frequency: [{cutoff_freq} Hz], cut bin [0:{cutoff_bin}]")
        power_spectrum[:cutoff_bin] = 1e-20  # Avoid log(0)
        power_spectrum_db = 10 * np.log10(power_spectrum)
        if complex_spectrum is not None:
            complex_spectrum[:cutoff_bin] = 0j

    # Find fundamental (integer bin and refined fractional bin)
    fundamental_bin, fundamental_bin_fractional = _locate_fundamental(power_spectrum, n_inband)

    # Automatically determine side_bin based on coherence if not specified
    if side_bin is None:
        # Check if signal is strictly coherent (fractional bin very close to integer bin)
        coherence_error = abs(fundamental_bin_fractional - fundamental_bin)
        is_coherent = coherence_error < 0.01

        # Get default side_bin from mapping based on window type and coherence
        side_bin = _get_default_side_bin(win_type, is_coherent)

    # Calculate fundamental bin range ONCE (to be reused by multiple functions)
    sig_bin_start = max(fundamental_bin - side_bin, 0)
    sig_bin_end = min(fundamental_bin + side_bin + 1, n_inband)

    # Calculate signal power directly using pre-calculated range
    # ENBW compensation: Signal (coherent tone) should not be divided by ENBW
    signal_power = np.sum(power_spectrum[sig_bin_start:sig_bin_end])
    sig_pwr_dbfs = 10 * np.log10(signal_power)

    # Calculate noise + distortion power
    # Use total power without ENBW compensation (for correct noise calculation)
    total_power_raw = np.sum(power_spectrum[:n_inband])
    noise_distortion_power = total_power_raw - signal_power

    # Override with assumed signal if provided
    if assumed_sig_pwr_dbfs is not None and not np.isnan(assumed_sig_pwr_dbfs):
        signal_power = 10 ** (assumed_sig_pwr_dbfs / 10)
        sig_pwr_dbfs = assumed_sig_pwr_dbfs

    # Calculate SNDR using assumed signal power but actual noise+distortion
    sndr_dbc = 10 * np.log10(signal_power / (noise_distortion_power + 1e-20))
    enob = (sndr_dbc - 1.76) / 6.02

    # Calculate harmonic power (THD, HD2, HD3, ...)
    harmonic_bins = _locate_harmonic_bins(fundamental_bin_fractional, max_harmonic, N)
    thd_power, harmonic_powers, collided_harmonics = _calculate_harmonic_power(
        power_spectrum=power_spectrum,
        fundamental_bin=fundamental_bin,
        harmonic_bins=harmonic_bins,
        side_bin=side_bin,
        max_harmonic=max_harmonic
    )

    # THD and harmonic levels (HD2, HD3, ...) in dBc
    harmonics_dbc = 10 * np.log10(harmonic_powers / signal_power)
    thd_dbc = 10 * np.log10(thd_power / signal_power)

    # SFDR (Limited to in-band search when OSR > 1)
    spur_bin_idx, spur_power = _extract_highest_spur(power_spectrum, side_bin, n_inband, sig_bin_start, sig_bin_end)
    sfdr_dbc = 10 * np.log10(signal_power / spur_power)

    # Estimate noise power using specified method
    noise_power = _estimate_noise_power(
        spectrum_power=power_spectrum,
        nf_method=nf_method,
        n_inband=n_inband,
        M=M,
        bin_idx=fundamental_bin,
        harmonic_bins=harmonic_bins,
        side_bin=side_bin
    )

    # Calculate noise-related metrics (SNR, noise floor, NSD)
    snr_dbc = 10 * np.log10(signal_power / noise_power)
    noise_floor_dbfs = sig_pwr_dbfs - snr_dbc
    # NSD: power spectrum already normalized by ENBW at line 89, so don't subtract ENBW again
    nsd_dbfs_hz = noise_floor_dbfs - 10 * np.log10(fs / (2 * osr))

    
    # Compensate power spectrum for plotting (recover peak amplitude for visualization)
    v_offset = sig_pwr_dbfs - power_spectrum_db[fundamental_bin]
    power_spectrum_db_plot = power_spectrum_db + v_offset

    # Build unified results dictionary
    results = {
        'N': N,
        'M': M,
        'fs': fs,
        'osr': osr,
        'metrics': {
            'enob': enob,
            'sndr_dbc': sndr_dbc,
            'sfdr_dbc': sfdr_dbc,
            'snr_dbc': snr_dbc,
            'sig_pwr_dbfs': sig_pwr_dbfs,
            'noise_floor_dbfs': noise_floor_dbfs,
            'nsd_dbfs_hz': nsd_dbfs_hz,
            'thd_dbc': thd_dbc,
            'harmonics_dbc': harmonics_dbc,
        },
        'plot_data': {
            'freq': freq,
            'power_spectrum_db_plot': power_spectrum_db_plot,
            'complex_spectrum': complex_spectrum,
            'fundamental_bin': fundamental_bin,
            'fundamental_bin_fractional': fundamental_bin_fractional,
            'sig_bin_start': sig_bin_start,
            'sig_bin_end': sig_bin_end,
            'spur_bin_idx': spur_bin_idx,
            'is_coherent': coherent_averaging,
            'harmonic_bins': harmonic_bins,
            'collided_harmonics': collided_harmonics,
            'v_offset': v_offset
        }
    }
    return results