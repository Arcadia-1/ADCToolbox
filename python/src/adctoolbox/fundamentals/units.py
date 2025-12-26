"""
Unit conversions for ADC testing.
"""

import numpy as np

def db_to_mag(db):
    """Convert dB to magnitude ratio: 10^(x/20)"""
    return 10**(db / 20)

def mag_to_db(mag):
    """Convert magnitude ratio to dB: 20*log10(x)"""
    return 20 * np.log10(mag)

def db_to_power(db):
    """Convert dB to power ratio: 10^(x/10)"""
    return 10**(db / 10)

def power_to_db(power):
    """Convert power ratio to dB: 10*log10(x)"""
    return 10 * np.log10(power)

def lsb_to_volts(lsb_count, vref, n_bits):
    """Convert LSB count to voltage"""
    return lsb_count * (vref / 2**n_bits)

def volts_to_lsb(volts, vref, n_bits):
    """Convert voltage to LSB count"""
    return volts / (vref / 2**n_bits)

def bin_to_freq(bin_idx, fs, n_fft):
    """Convert FFT bin index to frequency (Hz)"""
    return bin_idx * fs / n_fft

def freq_to_bin(freq, fs, n_fft):
    """Convert frequency (Hz) to nearest FFT bin index"""
    return int(np.round(freq * n_fft / fs))

def snr_to_enob(snr_db):
    """Convert SNR/SNDR (dB) to ENOB (bits): (SNR - 1.76) / 6.02"""
    return (snr_db - 1.76) / 6.02

def enob_to_snr(enob):
    """Convert ENOB (bits) to ideal SNR (dB): ENOB * 6.02 + 1.76"""
    return enob * 6.02 + 1.76

def snr_to_nsd(snr_db, fs, signal_pwr_dbfs=0, osr=1):
    """Convert SNR (dB) to Noise Spectral Density (dBFS/Hz).

    Parameters
    ----------
    snr_db : float or array_like
        Signal-to-Noise Ratio in dB.
    fs : float
        Sampling frequency in Hz.
    signal_pwr_dbfs : float, optional
        Signal power in dBFS. Default is 0 dBFS (full-scale signal).
    osr : float, optional
        Oversampling ratio. Default is 1.0 (Nyquist sampling).
        The noise bandwidth is fs / (2 * osr).

    Returns
    -------
    nsd_dbfs_hz : float or ndarray
        Noise Spectral Density in dBFS/Hz.

    Examples
    --------
    >>> # Full-scale signal, 80 dB SNR, 1 MHz sampling, OSR=256
    >>> nsd = snr_to_nsd(80, 1e6, signal_pwr_dbfs=0, osr=256)
    >>> print(f"NSD = {nsd:.2f} dBFS/Hz")
    NSD = -134.08 dBFS/Hz
    """
    # Noise bandwidth: BW = fs / (2 * osr)
    bw = fs / (2 * osr)
    return signal_pwr_dbfs - snr_db - 10 * np.log10(bw)

def nsd_to_snr(nsd_dbfs_hz, fs, signal_pwr_dbfs=0, osr=1):
    """Convert NSD (dBFS/Hz) to SNR (dB).

    Parameters
    ----------
    nsd_dbfs_hz : float or array_like
        Noise Spectral Density in dBFS/Hz.
    fs : float
        Sampling frequency in Hz.
    signal_pwr_dbfs : float, optional
        Signal power in dBFS. Default is 0 dBFS (full-scale signal).
    osr : float, optional
        Oversampling ratio. Default is 1.0 (Nyquist sampling).
        The noise bandwidth is fs / (2 * osr).

    Returns
    -------
    snr_db : float or ndarray
        Signal-to-Noise Ratio in dB.

    Examples
    --------
    >>> # NSD = -134 dBFS/Hz, 1 MHz sampling, OSR=256
    >>> snr = nsd_to_snr(-134, 1e6, signal_pwr_dbfs=0, osr=256)
    >>> print(f"SNR = {snr:.2f} dB")
    SNR = 79.92 dB
    """
    # Noise bandwidth: BW = fs / (2 * osr)
    bw = fs / (2 * osr)
    noise_total_db = nsd_dbfs_hz + 10 * np.log10(bw)
    return signal_pwr_dbfs - noise_total_db

def dbm_to_vrms(dbm, z_load=50):
    """Convert dBm to Vrms (assuming load impedance z_load)"""
    power_watts = db_to_power(dbm) / 1000  # dBm to Watts
    return np.sqrt(power_watts * z_load)

def vrms_to_dbm(vrms, z_load=50):
    """Convert Vrms to dBm"""
    power_watts = vrms**2 / z_load
    return power_to_db(power_watts * 1000)  # Watts to dBm

def dbm_to_mw(dbm):
    """Convert dBm to mW: mW = 10^(dBm/10)"""
    return db_to_power(dbm)

def mw_to_dbm(mw):
    """Convert mW to dBm: dBm = 10*log10(mW)"""
    return power_to_db(mw)

def sine_amplitude_to_power(amplitude, z_load=50):
    """
    Convert sine wave peak amplitude to power.

    For sine wave: Vrms = A / sqrt(2)
    Power = Vrms^2 / Z = A^2 / (2*Z)
    """
    vrms = amplitude / np.sqrt(2)
    return vrms**2 / z_load
