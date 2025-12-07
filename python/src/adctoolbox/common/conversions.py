"""
Unit Conversion Functions for ADC Testing.

Provides conversions between different units commonly used in ADC characterization:
- Amplitude/Power: dB ↔ linear magnitude/power ratios
- ADC codes: Voltage ↔ LSB counts
- Frequency domain: FFT bins ↔ Frequency
- Performance metrics: SNR ↔ ENOB
"""

import numpy as np


# --- Amplitude/Power Conversions ---

def db_to_mag(db):
    """
    Convert dB to magnitude ratio (Voltage/Current).

    Formula: 10^(x/20)

    Args:
        db: Decibel value

    Returns:
        Magnitude ratio (linear scale)
    """
    return 10**(db / 20)


def mag_to_db(mag):
    """
    Convert magnitude ratio to dB.

    Formula: 20*log10(x)

    Args:
        mag: Magnitude ratio (voltage or current)

    Returns:
        Decibel value
    """
    return 20 * np.log10(mag)


def db_to_power(db):
    """
    Convert dB to power ratio.

    Formula: 10^(x/10)

    Args:
        db: Decibel value

    Returns:
        Power ratio (linear scale)
    """
    return 10**(db / 10)


def power_to_db(power):
    """
    Convert power ratio to dB.

    Formula: 10*log10(x)

    Args:
        power: Power ratio

    Returns:
        Decibel value
    """
    return 10 * np.log10(power)


# --- ADC Specific Conversions (Voltage ↔ Code) ---

def lsb_to_volts(lsb_count, vref, resolution):
    """
    Convert LSB count to Voltage.

    Formula: lsb_size = vref / 2^N

    Args:
        lsb_count: Number of LSBs
        vref: Reference voltage (Vpp)
        resolution: ADC resolution in bits

    Returns:
        Voltage value
    """
    return lsb_count * (vref / (2**resolution))


def volts_to_lsb(volts, vref, resolution):
    """
    Convert Voltage to LSB count (float).

    Args:
        volts: Voltage value
        vref: Reference voltage (Vpp)
        resolution: ADC resolution in bits

    Returns:
        LSB count (floating point)
    """
    return volts / (vref / (2**resolution))


# --- Frequency Domain Conversions ---

def bin_to_freq(bin_idx, fs, n_fft):
    """
    Convert FFT bin index to Frequency (Hz).

    Args:
        bin_idx: FFT bin index
        fs: Sampling frequency (Hz)
        n_fft: FFT size

    Returns:
        Frequency in Hz
    """
    return bin_idx * fs / n_fft


def freq_to_bin(freq, fs, n_fft):
    """
    Convert Frequency (Hz) to nearest FFT bin index.

    Args:
        freq: Frequency in Hz
        fs: Sampling frequency (Hz)
        n_fft: FFT size

    Returns:
        Nearest bin index (integer)
    """
    return int(np.round(freq * n_fft / fs))


# --- Performance Metric Conversions ---

def snr_to_enob(snr_db):
    """
    Convert SNR/SNDR (dB) to ENOB (bits).

    Formula: (SNR - 1.76) / 6.02

    Args:
        snr_db: SNR or SNDR in dB

    Returns:
        ENOB in bits
    """
    return (snr_db - 1.76) / 6.02


def enob_to_snr(enob):
    """
    Convert ENOB (bits) to ideal SNR (dB).

    Formula: ENOB * 6.02 + 1.76

    Args:
        enob: Effective number of bits

    Returns:
        Ideal SNR in dB
    """
    return enob * 6.02 + 1.76


def snr_to_nsd(snr, fs, signal_pwr_dbfs=0):
    """
    Convert SNR (dB) to Noise Spectral Density (dBFS/Hz).

    Formula: NSD = Signal_dBFS - SNR - 10*log10(Fs/2)

    Assumes white noise distributed uniformly over the Nyquist zone.

    Args:
        snr: Signal-to-Noise Ratio in dB
        fs: Sampling frequency (Hz)
        signal_pwr_dbfs: Signal power in dBFS (default: 0 for full-scale)

    Returns:
        Noise Spectral Density in dBFS/Hz
    """
    return signal_pwr_dbfs - snr - 10 * np.log10(fs / 2)


def nsd_to_snr(nsd, fs, signal_pwr_dbfs=0):
    """
    Convert NSD (dBFS/Hz) to SNR (dB).

    Formula: SNR = Signal_dBFS - (NSD + 10*log10(Fs/2))

    Args:
        nsd: Noise Spectral Density in dBFS/Hz
        fs: Sampling frequency (Hz)
        signal_pwr_dbfs: Signal power in dBFS (default: 0 for full-scale)

    Returns:
        Signal-to-Noise Ratio in dB
    """
    noise_total_db = nsd + 10 * np.log10(fs / 2)
    return signal_pwr_dbfs - noise_total_db
