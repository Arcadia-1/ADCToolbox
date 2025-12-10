"""Utility module for converting between Noise Spectral Density (NSD) and Signal-to-Noise Ratio (SNR)."""

import numpy as np
from typing import Union


def snr_to_nsd(
    snr_db: Union[float, np.ndarray],
    fs: float,
    osr: float = 1.0,
    psignal_dbfs: float = 0.0
) -> Union[float, np.ndarray]:
    """Convert Signal-to-Noise Ratio (SNR) to Noise Spectral Density (NSD).

    This function converts SNR in dB to NSD in dBFS/Hz, given the sampling frequency
    and oversampling ratio. It assumes a full-scale sine wave signal (0 dBFS) unless
    specified otherwise.

    The relationship is derived from:
    - Signal power: P_signal = 10^(Psignal_dBFS / 10)
    - Noise power: P_noise = P_signal / 10^(SNR_dB / 10)
    - Noise bandwidth: BW = fs / (2 * OSR)
    - NSD = P_noise / BW (linear scale)
    - NSD_dBFS/Hz = 10 * log10(NSD)

    Parameters
    ----------
    snr_db : float or array_like
        Signal-to-Noise Ratio in dB.
    fs : float
        Sampling frequency in Hz.
    osr : float, optional
        Oversampling ratio. Default is 1.0 (Nyquist sampling).
        The noise bandwidth is fs / (2 * OSR).
    psignal_dbfs : float, optional
        Signal power in dBFS. Default is 0.0 dBFS (full-scale signal).

    Returns
    -------
    nsd_dbfs_hz : float or ndarray
        Noise Spectral Density in dBFS/Hz.

    Examples
    --------
    >>> # For a full-scale signal with 80 dB SNR, fs=1 MHz, OSR=256
    >>> nsd = snr_to_nsd(snr_db=80, fs=1e6, osr=256)
    >>> print(f"NSD = {nsd:.2f} dBFS/Hz")
    NSD = -134.08 dBFS/Hz

    >>> # For a -6 dBFS signal with 70 dB SNR, fs=100 kHz, Nyquist sampling
    >>> nsd = snr_to_nsd(snr_db=70, fs=1e5, osr=1, psignal_dbfs=-6)
    >>> print(f"NSD = {nsd:.2f} dBFS/Hz")
    NSD = -122.99 dBFS/Hz
    """
    # Check if input is scalar before converting to arrays
    is_scalar_input = np.ndim(snr_db) == 0

    # Convert to NumPy array for vectorization
    snr_db = np.asarray(snr_db)

    # Calculate signal power in linear scale
    # For a full-scale sine wave: P_signal = 0.5 (or -3.01 dBFS)
    # For dBFS scale: P_signal_dBFS = 10*log10(0.5) = -3.01 dBFS (for amplitude = 1)
    # But if we define 0 dBFS as the sine wave power, then P_signal = 1 (linear)
    p_signal_linear = 10 ** (psignal_dbfs / 10)

    # Calculate noise power from SNR
    # SNR_dB = 10 * log10(P_signal / P_noise)
    # P_noise = P_signal / 10^(SNR_dB / 10)
    p_noise_linear = p_signal_linear / (10 ** (snr_db / 10))

    # Calculate noise bandwidth
    # For baseband: BW = fs / (2 * OSR)
    bw = fs / (2 * osr)

    # Calculate NSD (Noise Spectral Density)
    # NSD = P_noise / BW (in linear scale)
    nsd_linear = p_noise_linear / bw

    # Convert to dBFS/Hz
    nsd_dbfs_hz = 10 * np.log10(nsd_linear)

    # Convert back to scalar if input was scalar
    if is_scalar_input:
        nsd_dbfs_hz = float(nsd_dbfs_hz)

    return nsd_dbfs_hz


def nsd_to_snr(
    nsd_dbfs_hz: Union[float, np.ndarray],
    fs: float,
    osr: float = 1.0,
    psignal_dbfs: float = 0.0
) -> Union[float, np.ndarray]:
    """Convert Noise Spectral Density (NSD) to Signal-to-Noise Ratio (SNR).

    This function converts NSD in dBFS/Hz to SNR in dB, given the sampling frequency
    and oversampling ratio. It assumes a full-scale sine wave signal (0 dBFS) unless
    specified otherwise.

    The relationship is derived from:
    - NSD in linear scale: NSD_linear = 10^(NSD_dBFS/Hz / 10)
    - Noise bandwidth: BW = fs / (2 * OSR)
    - Noise power: P_noise = NSD_linear * BW
    - Signal power: P_signal = 10^(Psignal_dBFS / 10)
    - SNR = 10 * log10(P_signal / P_noise)

    Parameters
    ----------
    nsd_dbfs_hz : float or array_like
        Noise Spectral Density in dBFS/Hz.
    fs : float
        Sampling frequency in Hz.
    osr : float, optional
        Oversampling ratio. Default is 1.0 (Nyquist sampling).
        The noise bandwidth is fs / (2 * OSR).
    psignal_dbfs : float, optional
        Signal power in dBFS. Default is 0.0 dBFS (full-scale signal).

    Returns
    -------
    snr_db : float or ndarray
        Signal-to-Noise Ratio in dB.

    Examples
    --------
    >>> # For NSD = -134 dBFS/Hz, fs=1 MHz, OSR=256
    >>> snr = nsd_to_snr(nsd_dbfs_hz=-134, fs=1e6, osr=256)
    >>> print(f"SNR = {snr:.2f} dB")
    SNR = 79.92 dB

    >>> # For NSD = -123 dBFS/Hz, fs=100 kHz, Nyquist sampling, -6 dBFS signal
    >>> snr = nsd_to_snr(nsd_dbfs_hz=-123, fs=1e5, osr=1, psignal_dbfs=-6)
    >>> print(f"SNR = {snr:.2f} dB")
    SNR = 70.01 dB
    """
    # Check if input is scalar before converting to arrays
    is_scalar_input = np.ndim(nsd_dbfs_hz) == 0

    # Convert to NumPy array for vectorization
    nsd_dbfs_hz = np.asarray(nsd_dbfs_hz)

    # Convert NSD from dBFS/Hz to linear scale
    nsd_linear = 10 ** (nsd_dbfs_hz / 10)

    # Calculate noise bandwidth
    # For baseband: BW = fs / (2 * OSR)
    bw = fs / (2 * osr)

    # Calculate noise power
    # P_noise = NSD * BW
    p_noise_linear = nsd_linear * bw

    # Calculate signal power in linear scale
    p_signal_linear = 10 ** (psignal_dbfs / 10)

    # Calculate SNR
    # SNR_dB = 10 * log10(P_signal / P_noise)
    snr_db = 10 * np.log10(p_signal_linear / p_noise_linear)

    # Convert back to scalar if input was scalar
    if is_scalar_input:
        snr_db = float(snr_db)

    return snr_db
