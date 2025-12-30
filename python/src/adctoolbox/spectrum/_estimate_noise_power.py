"""Noise power estimation for spectrum analysis.

This module provides functions for estimating noise power from spectrum data
using different methods (median, trimmed mean, or harmonic exclusion).

This is an internal helper module, not intended for direct use by end users.
"""

import numpy as np
from adctoolbox.spectrum._exclude_bins import _exclude_bins_from_spectrum


def _estimate_noise_power(
    spectrum_power: np.ndarray,
    nf_method: int,
    n_inband: int,
    M: int,
    bin_idx: int,
    harmonic_bins: np.ndarray,
    side_bin: int
) -> float:
    """Estimate noise power from spectrum using specified method.

    Parameters
    ----------
    spectrum_power : np.ndarray
        Power spectrum array (half-sided)
    nf_method : int
        Noise floor method:
        - 0: Median-based (robust to spurs)
        - 1: Trimmed mean (removes top/bottom 5%)
        - 2: Exclude harmonics (most accurate)
    n_inband : int
        Number of bins to search in-band
    M : int
        Number of averaged runs
    bin_idx : int
        Fundamental bin index
    harmonic_bins : np.ndarray
        Array of harmonic bin positions
    side_bin : int
        Side bins to exclude around signal and harmonics

    Returns
    -------
    noise_power : float
        Estimated noise power (linear scale, not dB)

    Notes
    -----
    Different methods trade off robustness vs accuracy:
    - Method 0 (median): Most robust to spurs, but may overestimate noise
    - Method 1 (trimmed mean): Balanced approach
    - Method 2 (exclude harmonics): Most accurate when harmonics are known
    """
    if nf_method == 0:
        # Median-based (robust to spurs)
        # Apply correction factor for median estimator with M runs
        noise_power = np.median(spectrum_power[:n_inband]) / np.sqrt((1 - 2/(9*M))**3) * n_inband
    elif nf_method == 1:
        # Trimmed mean (removes top/bottom 5%)
        spec_sorted = np.sort(spectrum_power[:n_inband])
        start_idx = int(n_inband * 0.05)
        end_idx = int(n_inband * 0.95)
        noise_power = np.mean(spec_sorted[start_idx:end_idx]) * n_inband
    else:
        # Exclude harmonics (most accurate)
        noise_spectrum = _exclude_bins_from_spectrum(
            spectrum_power, bin_idx, harmonic_bins, side_bin, n_inband
        )
        noise_power = np.sum(noise_spectrum)

    # Ensure minimum value to avoid log(0)
    noise_power = max(noise_power, 1e-15)

    return noise_power


def _calculate_noise_metrics(
    signal_power: float,
    noise_power: float,
    sig_pwr_dbfs: float,
    fs: float,
    osr: int,
    enbw: float
) -> tuple[float, float, float]:
    """Calculate noise-related metrics (SNR, noise floor, NSD).

    Parameters
    ----------
    signal_power : float
        Signal power (linear scale)
    noise_power : float
        Noise power (linear scale)
    sig_pwr_dbfs : float
        Signal power in dBFS
    fs : float
        Sampling frequency (Hz)
    osr : int
        Oversampling ratio
    enbw : float
        Equivalent Noise Bandwidth of the window

    Returns
    -------
    snr_db : float
        Signal-to-Noise Ratio in dB
    noise_floor_dbfs : float
        Noise floor in dBFS
    nsd_dbfs_hz : float
        Noise Spectral Density in dBFS/Hz

    Notes
    -----
    Noise floor is calculated as: sig_pwr_dbfs - SNR
    NSD accounts for bandwidth and window ENBW:
        NSD = noise_floor - 10*log10(BW) - 10*log10(ENBW)
    """
    # SNR in dB
    snr_db = 10 * np.log10(signal_power / noise_power)

    # Noise floor in dBFS
    noise_floor_dbfs = sig_pwr_dbfs - snr_db

    # Noise Spectral Density (NSD) in dBFS/Hz
    # Account for in-band bandwidth (fs/2/osr) and window ENBW
    nsd_dbfs_hz = noise_floor_dbfs - 10 * np.log10(fs / (2 * osr)) - 10 * np.log10(enbw)

    return snr_db, noise_floor_dbfs, nsd_dbfs_hz
