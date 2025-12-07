"""
Unit conversions for ADC testing.

MATLAB counterpart: db2mag.m, mag2db.m, db2pow.m, pow2db.m
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


def snr_to_nsd(snr_db, fs, signal_pwr_dbfs=0):
    """Convert SNR (dB) to Noise Spectral Density (dBFS/Hz)"""
    return signal_pwr_dbfs - snr_db - 10 * np.log10(fs / 2)


def nsd_to_snr(nsd_dbfs_hz, fs, signal_pwr_dbfs=0):
    """Convert NSD (dBFS/Hz) to SNR (dB)"""
    noise_total_db = nsd_dbfs_hz + 10 * np.log10(fs / 2)
    return signal_pwr_dbfs - noise_total_db


def dbm_to_mv(dbm, impedance=50):
    """
    Convert dBm to mV (RMS voltage).

    Formula: V_mV = 10^((dBm + 10*log10(Z/1000) + 30)/20)
    For 50 ohm: V_mV = 10^((dBm + 46.99)/20)

    Args:
        dbm: Power in dBm
        impedance: Load impedance in ohms (default: 50)

    Returns:
        Voltage in mV (RMS)
    """
    return 10**((dbm + 10*np.log10(impedance/1000) + 30) / 20)


def mv_to_dbm(mv, impedance=50):
    """
    Convert mV (RMS voltage) to dBm.

    Formula: dBm = 20*log10(V_mV) - 10*log10(Z/1000) - 30
    For 50 ohm: dBm = 20*log10(V_mV) - 46.99

    Args:
        mv: Voltage in mV (RMS)
        impedance: Load impedance in ohms (default: 50)

    Returns:
        Power in dBm
    """
    return 20*np.log10(mv) - 10*np.log10(impedance/1000) - 30
