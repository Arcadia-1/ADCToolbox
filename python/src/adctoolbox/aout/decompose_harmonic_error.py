"""Decompose ADC error into harmonic distortion and other errors.

This module provides pure computation functionality for extracting harmonic
components using least-squares fitting (LMS), strictly adhering to the
Single Responsibility Principle. No visualization is performed.
"""

import numpy as np
from typing import Any
from adctoolbox.aout._fit_sine_harmonics import _fit_sine_harmonics
from adctoolbox.fundamentals.fit_sine_4param import fit_sine_4param

def decompose_harmonic_error(
    signal: np.ndarray,
    n_harmonics: int = 5
) -> dict[str, Any]:
    """Decompose ADC error into harmonic distortion and other errors using least-squares fitting.

    This is a pure calculation function that extracts harmonic components
    from ADC signal using least-squares fitting. Works directly in the signal's
    original units without normalization.

    Parameters
    ----------
    signal : np.ndarray
        Input ADC signal, shape (N,) for single run or (M, N) for M runs
        For multi-run data, runs are averaged before decomposition
    n_harmonics : int, optional
        Number of harmonics to extract (default: 5)

    Returns
    -------
    dict
        Dictionary containing decomposition results:

        - 'magnitudes': np.ndarray, shape (n_harmonics,)
            Magnitude of each harmonic (1 to n_harmonics)
        - 'phases': np.ndarray, shape (n_harmonics,)
            Phase of each harmonic in radians (relative to fundamental)
        - 'magnitudes_db': np.ndarray, shape (n_harmonics,)
            Magnitude in dB relative to full scale
        - 'residual_rms': float
            RMS power of residual noise
        - 'noise_db': float
            Noise floor in dB relative to full scale
        - 'fundamental_freq': float
            Detected fundamental frequency (normalized, 0 to 1)
        - 'noise_residual': np.ndarray, shape (N,)
            Residual signal after removing all harmonics (centered at 0, no DC)
        - 'reconstructed_signal': np.ndarray, shape (N,)
            Reconstructed signal with all harmonics (includes DC offset)
        - 'fundamental_signal': np.ndarray, shape (N,)
            Reconstructed fundamental component only (includes DC offset)
        - 'harmonic_signal': np.ndarray, shape (N,)
            Reconstructed harmonic components 2nd to nth (centered at 0, no DC)

    Notes
    -----
    Algorithm:
    1. Average multiple runs if provided
    2. Normalize signal to full scale
    3. Find fundamental frequency using 4-parameter sine fit
    4. Build sine/cosine basis for harmonics: cos(k*ω*t), sin(k*ω*t)
    5. Solve least squares: W = (A^T A)^(-1) A^T * signal
    6. Extract magnitude and phase for each harmonic (vectorized)
    7. Rotate phases relative to fundamental (vectorized)
    8. Calculate residual and noise floor

    Phase Convention:
    - Phases are relative to fundamental
    - Each harmonic phase = phase_of_harmonic - (phase_of_fundamental * harmonic_order)
    - Wrapped to [-π, π]
    """

    # Convert to numpy array and ensure 2D shape (M runs x N samples)
    signal = np.atleast_2d(signal)
    if signal.shape[1] == 1:  # Handle (N, 1) shape
        signal = signal.T

    # Average all runs
    sig_avg = np.mean(signal, axis=0)

    # Remove DC component
    dc_offset = np.mean(sig_avg)
    sig_zero_mean = sig_avg - dc_offset

    # Find fundamental frequency using fit_sine_4param
    fit_result = fit_sine_4param(sig_zero_mean, frequency_estimate=None, max_iterations=1)
    fundamental_freq = fit_result['frequency']

    # Use fit_sine_harmonics as the core math kernel for least-squares fitting
    # Include DC term to capture any residual numerical bias and avoid offset in residual
    include_dc = True
    W, signal_reconstructed, basis_matrix, phase = _fit_sine_harmonics(
        sig_zero_mean,
        freq=fundamental_freq,
        order=n_harmonics,
        include_dc=include_dc
    )

    # Reconstruct fundamental only (1st harmonic)
    # Handle both include_dc=True and include_dc=False cases
    if include_dc:
        # W layout: [DC, cos(H1), sin(H1), cos(H2), sin(H2), ..., cos(Hn), sin(Hn)]
        fundamental_signal = basis_matrix[:, 0] * W[0] + basis_matrix[:, 1] * W[1] + basis_matrix[:, 2] * W[2]
    else:
        # W layout: [cos(H1), sin(H1), cos(H2), sin(H2), ..., cos(Hn), sin(Hn)]
        fundamental_signal = basis_matrix[:, 0] * W[0] + basis_matrix[:, 1] * W[1]

    # Reconstruct harmonics by difference
    # This ensures: signal = fundamental + harmonic + residual exactly
    harmonic_signal = signal_reconstructed - fundamental_signal

    # Calculate residual (noise)
    noise_residual = sig_zero_mean - signal_reconstructed

    # Add DC back to reconstructed signals to match original signal scale
    fundamental_signal = fundamental_signal + dc_offset
    reconstructed_signal = signal_reconstructed + dc_offset
    # Note: harmonic_signal and noise_residual remain centered at 0 (no DC added)

    # Calculate noise metrics in original signal units
    residual_rms = np.sqrt(np.mean(noise_residual**2)) * 2 * np.sqrt(2)
    signal_range = np.max(sig_avg) - np.min(sig_avg)
    noise_db = 20 * np.log10(residual_rms / signal_range)

    # Extract magnitude and phase for each harmonic from coefficients (vectorized)
    # Use complex number representation: I + jQ
    if include_dc:
        # W layout: [DC, cos(H1), sin(H1), cos(H2), sin(H2), ..., cos(Hn), sin(Hn)]
        # Coefficients are interleaved: cos at odd indices, sin at even indices (after DC)
        cos_coeffs = W[1:2*n_harmonics:2]  # W[1], W[3], W[5], ..., W[2*n_harmonics-1]
        sin_coeffs = W[2:2*n_harmonics+1:2]  # W[2], W[4], W[6], ..., W[2*n_harmonics]
    else:
        # W layout: [cos(H1), sin(H1), cos(H2), sin(H2), ..., cos(Hn), sin(Hn)]
        # Coefficients are interleaved: cos at even indices, sin at odd indices
        cos_coeffs = W[0:2*n_harmonics:2]  # W[0], W[2], W[4], ..., W[2*n_harmonics-2]
        sin_coeffs = W[1:2*n_harmonics:2]  # W[1], W[3], W[5], ..., W[2*n_harmonics-1]

    coeffs = cos_coeffs + 1j * sin_coeffs
    magnitudes = np.abs(coeffs) * 2
    phases = np.angle(coeffs)

    # Phase rotation: make phases relative to fundamental (vectorized)
    # Each harmonic's phase = phase_of_harmonic - (phase_of_fundamental * harmonic_order)
    fundamental_phase = phases[0]
    harmonic_order = np.arange(1, n_harmonics + 1)
    phases = phases - fundamental_phase * harmonic_order

    # Wrap phases to [-pi, pi]
    phases = (phases + np.pi) % (2 * np.pi) - np.pi

    # Convert to dB (relative to signal range)
    magnitudes_db = 20 * np.log10(magnitudes / signal_range + 1e-20)

    return {
        'magnitudes': magnitudes,
        'phases': phases,
        'magnitudes_db': magnitudes_db,
        'residual_rms': residual_rms,
        'noise_db': noise_db,
        'fundamental_freq': fundamental_freq,
        'noise_residual': noise_residual,
        'reconstructed_signal': reconstructed_signal,
        'fundamental_signal': fundamental_signal,
        'harmonic_signal': harmonic_signal,
    }
