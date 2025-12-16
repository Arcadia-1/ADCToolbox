"""Calculate LMS harmonic decomposition for ADC analysis.

This module provides pure computation functionality for extracting harmonic
components using least-squares fitting (LMS), strictly adhering to the
Single Responsibility Principle. No visualization is performed.

Matches MATLAB plotphase.m LMS mode calculation logic.
"""

import numpy as np
from typing import Optional, Dict, Any
from .fit_sine_harmonics import fit_sine_harmonics


def compute_harmonic_decomposition(
    data: np.ndarray,
    max_code: Optional[float] = None,
    harmonic: int = 5,
    fs: float = 1.0
) -> Dict[str, Any]:
    """Calculate harmonic decomposition using least-squares fitting.

    This is a pure calculation function that extracts harmonic components
    from ADC data using least-squares fitting (matching MATLAB plotphase LMS mode).

    Parameters
    ----------
    data : np.ndarray
        Input ADC data, shape (N,) for single run or (M, N) for M runs
        For multi-run data, runs are averaged before decomposition
    max_code : float, optional
        Maximum code level for normalization. If None, uses (max(data) - min(data))
    harmonic : int, optional
        Number of harmonics to extract (default: 5)
    fs : float, optional
        Sampling frequency in Hz (default: 1.0)

    Returns
    -------
    dict
        Dictionary containing decomposition results:

        - 'harm_mag': np.ndarray, shape (harmonic,)
            Magnitude of each harmonic (1 to harmonic)
        - 'harm_phase': np.ndarray, shape (harmonic,)
            Phase of each harmonic in radians (relative to fundamental)
        - 'harm_dB': np.ndarray, shape (harmonic,)
            Magnitude in dB relative to full scale
        - 'noise_power': float
            RMS power of residual noise
        - 'noise_dB': float
            Noise floor in dB relative to full scale
        - 'fundamental_freq': float
            Detected fundamental frequency (normalized, 0 to 1)
        - 'residual': np.ndarray, shape (N,)
            Residual signal after removing all harmonics
        - 'signal_reconstructed': np.ndarray, shape (N,)
            Reconstructed signal with all harmonics
        - 'fundamental_signal': np.ndarray, shape (N,)
            Reconstructed fundamental component only
        - 'harmonic_signal': np.ndarray, shape (N,)
            Reconstructed harmonic components (2nd to nth)
        - 'n_samples': int
            Number of samples
        - 'fs': float
            Sampling frequency used

    Notes
    -----
    Algorithm (matches MATLAB plotphase.m LMS mode):
    1. Average multiple runs if provided
    2. Normalize signal to full scale
    3. Find fundamental frequency using FFT peak detection
    4. Build sine/cosine basis for harmonics: cos(k*ω*t), sin(k*ω*t)
    5. Solve least squares: W = (A^T A)^(-1) A^T * signal
    6. Extract magnitude and phase for each harmonic
    7. Rotate phases relative to fundamental
    8. Calculate residual and noise floor

    Phase Convention:
    - Phases are relative to fundamental (fundamental rotated to reference)
    - Each harmonic phase = phase_of_harmonic - (phase_of_fundamental * harmonic_order)
    - Wrapped to [-π, π]
    """

    # Convert to numpy array
    data = np.asarray(data)

    # Handle different input shapes
    if data.ndim == 0:
        data = data.reshape(1, 1)
    elif data.ndim == 1:
        data = data.reshape(1, -1)
    elif data.ndim == 2:
        # Ensure shape is (M, N) not (N, M)
        M, N = data.shape
        if N == 1 and M > 1:
            data = data.T
    else:
        raise ValueError(f"Input data must be 1D or 2D, got {data.ndim}D")

    # Average all runs (matches MATLAB: sig_avg = mean(sig, 1))
    sig_avg = np.mean(data, axis=0)
    n_samples = len(sig_avg)

    # Determine max_code for normalization
    if max_code is None:
        max_code = np.max(sig_avg) - np.min(sig_avg)

    # Normalize signal (matches MATLAB normalization)
    sig_avg = sig_avg - np.mean(sig_avg)  # Remove DC
    sig_normalized = sig_avg / max_code

    # Find fundamental frequency using FFT (matches MATLAB sinfit call)
    fundamental_freq = _find_fundamental_freq(sig_normalized)

    # Use fit_sine_harmonics as the core math kernel for least-squares fitting
    # Note: fit_sine_harmonics expects order to be the highest harmonic (1-based),
    # so we pass order=harmonic directly
    W, signal_reconstructed, basis_matrix, phase = fit_sine_harmonics(
        sig_normalized,
        freq=fundamental_freq,
        order=harmonic,
        include_dc=False  # DC already removed in normalization
    )

    # Reconstruct fundamental only (1st harmonic)
    # W layout: [cos(H1), sin(H1), cos(H2), sin(H2), ..., cos(Hn), sin(Hn)]
    fundamental_signal = basis_matrix[:, 0] * W[0] + basis_matrix[:, 1] * W[harmonic]

    # Reconstruct harmonics only (2nd through nth)
    harmonic_signal = np.zeros(n_samples)
    for ii in range(1, harmonic):
        harmonic_signal += basis_matrix[:, 2*ii] * W[ii] + basis_matrix[:, 2*ii + 1] * W[ii + harmonic]

    # Calculate residual (noise)
    residual = sig_normalized - signal_reconstructed
    noise_power = np.sqrt(np.mean(residual**2)) * 2 * np.sqrt(2)
    noise_dB = 20 * np.log10(noise_power)

    # Extract magnitude and phase for each harmonic from coefficients
    harm_mag = np.zeros(harmonic)
    harm_phase = np.zeros(harmonic)

    for ii in range(harmonic):
        I_weight = W[ii]
        Q_weight = W[ii + harmonic]
        harm_mag[ii] = np.sqrt(I_weight**2 + Q_weight**2) * 2
        harm_phase[ii] = np.arctan2(Q_weight, I_weight)

    # Phase rotation: make phases relative to fundamental
    # Each harmonic's phase = phase_of_harmonic - (phase_of_fundamental * harmonic_order)
    fundamental_phase = harm_phase[0]
    for ii in range(harmonic):
        harm_phase[ii] = harm_phase[ii] - fundamental_phase * (ii + 1)

    # Wrap phases to [-pi, pi]
    harm_phase = np.mod(harm_phase + np.pi, 2 * np.pi) - np.pi

    # Convert to dB (relative to full scale)
    harm_dB = 20 * np.log10(harm_mag + 1e-20)

    return {
        'harm_mag': harm_mag,
        'harm_phase': harm_phase,
        'harm_dB': harm_dB,
        'noise_power': noise_power,
        'noise_dB': noise_dB,
        'fundamental_freq': fundamental_freq,
        'residual': residual,
        'signal_reconstructed': signal_reconstructed,
        'fundamental_signal': fundamental_signal,
        'harmonic_signal': harmonic_signal,
        'n_samples': n_samples,
        'fs': fs,
    }


def _find_fundamental_freq(signal: np.ndarray) -> float:
    """Find fundamental frequency using FFT peak detection.

    This matches MATLAB's sinfit function behavior for frequency detection.

    Parameters
    ----------
    signal : np.ndarray
        Normalized signal

    Returns
    -------
    float
        Normalized fundamental frequency (0 to 1)
    """
    n_samples = len(signal)

    # Perform FFT
    spectrum = np.fft.fft(signal)
    spectrum[0] = 0  # Remove DC

    # Find peak in positive frequencies
    half_spectrum = np.abs(spectrum[:n_samples // 2])
    bin_idx = np.argmax(half_spectrum)

    # Parabolic interpolation for sub-bin accuracy
    if bin_idx > 0 and bin_idx < len(half_spectrum) - 1:
        y_m1 = np.log10(max(half_spectrum[bin_idx - 1], 1e-20))
        y_0 = np.log10(max(half_spectrum[bin_idx], 1e-20))
        y_p1 = np.log10(max(half_spectrum[bin_idx + 1], 1e-20))

        delta = (y_p1 - y_m1) / (2 * (2 * y_0 - y_m1 - y_p1))
        if not np.isnan(delta):
            bin_r = bin_idx + delta
        else:
            bin_r = bin_idx
    else:
        bin_r = bin_idx

    # Convert to normalized frequency
    fundamental_freq = bin_r / n_samples

    return fundamental_freq
