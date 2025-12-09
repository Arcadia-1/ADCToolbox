"""A utility module for calculating theoretical Signal-to-Noise Ratio (SNR)."""

import numpy as np
from typing import Union, Tuple


def calculate_snr_from_amplitude(
    sig_amplitude: Union[float, np.ndarray],
    noise_amplitude: Union[float, np.ndarray],
    return_power: bool = False
) -> Union[float, np.ndarray, Tuple[Union[float, np.ndarray], ...]]:
    """Calculate Signal-to-Noise Ratio (SNR) in dB from sine wave peak amplitude and noise RMS.

    This function computes SNR, assuming the signal is a pure sine wave and the noise
    is Gaussian (White Noise).

    SNR is calculated based on the power ratio: SNR (dB) = 10 * log10(P_sig / P_noise).

    Parameters
    ----------
    sig_amplitude : float or array_like
        Sine wave peak amplitude (A), in Volts (V).
    noise_amplitude : float or array_like
        Noise RMS amplitude (σ), in Volts (V).
    return_power : bool, optional
        If True, returns a tuple containing (snr_db, sig_power, noise_power).
        Default is False, returning only snr_db.

    Returns
    -------
    snr_db : float or ndarray
        The calculated SNR in dB. Returns np.inf if noise_amplitude is zero.
    (snr_db, sig_power, noise_power) : tuple (if return_power=True)
        The SNR in dB, Signal Power (V^2), and Noise Power (V^2), respectively.
    """
    # Check if inputs are scalar before converting to arrays
    is_scalar_input = (np.ndim(sig_amplitude) == 0 and np.ndim(noise_amplitude) == 0)

    # Convert inputs to NumPy arrays to enable high-performance vectorization
    sig_amplitude = np.asarray(sig_amplitude)
    noise_amplitude = np.asarray(noise_amplitude)

    # 1. Calculate Signal RMS and Power
    # For a sine wave: RMS = Peak / sqrt(2)
    sig_rms = sig_amplitude / np.sqrt(2)
    # Power is proportional to RMS squared (assuming 1 Ohm resistance: P = V_rms^2)
    sig_power = sig_rms ** 2

    # 2. Calculate Noise Power
    # For Gaussian noise: Power = RMS^2 = σ^2
    noise_power = noise_amplitude ** 2

    # 3. Calculate SNR (dB) using the power ratio: 10 * log10(P_sig / P_noise)
    # Use np.errstate to handle division by zero gracefully (noise_amplitude = 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        # Calculate amplitude ratio: RMS_sig / RMS_noise
        ratio = sig_rms / noise_amplitude
        # Replace any inf/-inf with positive inf for zero noise
        ratio = np.where(noise_amplitude == 0, np.inf, ratio)

    # Convert amplitude ratio to SNR in dB: 20 * log10(Ratio)
    snr_db = 20 * np.log10(ratio)

    # Convert results back to standard Python float if inputs were scalar
    if is_scalar_input:
        snr_db = float(snr_db)
        sig_power = float(sig_power)
        noise_power = float(noise_power)

    # Return results based on the return_power flag
    if return_power:
        return snr_db, sig_power, noise_power
    else:
        return snr_db