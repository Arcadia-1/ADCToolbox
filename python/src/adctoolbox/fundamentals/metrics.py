"""
Figure of Merit (FoM) Calculations for ADC Performance.

Provides standard FoM calculations and theoretical performance limits:
- Walden FoM: Standard for medium-resolution ADCs
- Schreier FoM: Standard for high-resolution/Sigma-Delta ADCs
- Thermal noise limit: kT/C noise floor
- Jitter limit: Aperture jitter SNR limit
"""

import numpy as np

# --- Comprehensive Performance Metrics (Figure of Merit) ---

def calculate_walden_fom(power, fs, enob):
    """
    Calculate Walden Figure of Merit (FoM_w).

    Standard metric for medium-resolution ADCs.
    Lower is better.

    Formula: Power / (2^ENOB * Fs)

    Args:
        power: Power consumption (W)
        fs: Sampling frequency (Hz)
        enob: Effective number of bits

    Returns:
        FoM_w in J/conv-step (Joules per conversion step)
    """
    return power / (2**enob * fs)

def calculate_schreier_fom(power, sndr_db, bw):
    """
    Calculate Schreier Figure of Merit (FoM_s).

    Standard metric for high-resolution / Sigma-Delta ADCs.
    Higher is better.

    Formula: SNDR + 10*log10(BW / Power)

    Args:
        power: Power consumption (W)
        sndr_db: Signal-to-Noise and Distortion Ratio (dB)
        bw: Signal bandwidth (Hz)

    Returns:
        FoM_s in dB
    """
    return sndr_db + 10 * np.log10(bw / power)

# --- Theoretical Performance Limits ---

def calculate_thermal_noise_limit(cap_pf, v_fs=1.0):
    """
    Calculate maximum achievable SNR limited by kT/C noise.

    Thermal noise sets the fundamental limit for switched-capacitor
    circuits and sample-and-hold amplifiers.

    Args:
        cap_pf: Sampling capacitance in pF
        v_fs: Full-scale voltage (Vpp), default 1.0V

    Returns:
        Maximum SNR in dB

    Theory:
        - Noise power = kT/C
        - Signal power (sine) = (Vfs/2)^2 / 2 = Vfs^2 / 8
        - SNR = 10*log10(P_signal / P_noise)
    """
    k = 1.38e-23  # Boltzmann constant (J/K)
    T = 300       # Temperature (K)
    C = cap_pf * 1e-12  # Convert pF to F

    # Thermal noise power
    p_noise = k * T / C

    # Signal power (full-scale sine wave)
    p_sig = (v_fs**2) / 8

    return 10 * np.log10(p_sig / p_noise)

def calculate_jitter_limit(freq, jitter_rms_sec):
    """
    Calculate maximum achievable SNR limited by aperture jitter.

    Sampling jitter creates phase noise that limits SNR, especially
    at high input frequencies.

    Formula: SNR = -20 * log10(2 * pi * fin * tj)

    Args:
        freq: Input frequency (Hz)
        jitter_rms_sec: RMS jitter in seconds

    Returns:
        Maximum SNR in dB
    """
    return -20 * np.log10(2 * np.pi * freq * jitter_rms_sec)
