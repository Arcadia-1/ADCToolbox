"""
Compute error binned by code values (for INL/DNL analysis).

Core computation kernel for computing error statistics binned by ADC code
value (amplitude). This is useful for analyzing static nonlinearity and
code-dependent errors.
"""

import numpy as np
from typing import Dict
from .decompose_harmonics import fit_sinewave_components


def compute_error_by_code(
    signal: np.ndarray,
    normalized_freq: float,
    num_bits: int = None,
    clip_percent: float = 0.01
) -> Dict[str, np.ndarray]:
    """
    Compute error binned by code values (for INL/DNL analysis).

    This function fits a fundamental sinewave to the input signal, computes
    the residual error, and bins the error by ADC code value (amplitude).
    This reveals code-dependent errors such as INL and DNL.

    Parameters
    ----------
    signal : np.ndarray
        Input signal, 1D numpy array.
    normalized_freq : float
        Normalized frequency (f/fs), range 0-0.5.
    num_bits : int, optional
        Number of bits for ADC resolution. If None, inferred from signal range.
    clip_percent : float, default=0.01
        Percentage of codes to clip from edges (excludes near-rail codes).

    Returns
    -------
    dict : Dictionary containing:
        - 'emean_by_code' : np.ndarray
            Mean error per code bin.
        - 'erms_by_code' : np.ndarray
            RMS error per code bin.
        - 'code_bins' : np.ndarray
            Code bin centers (ADC code values).
        - 'bin_counts' : np.ndarray
            Number of samples in each code bin.
        - 'fitted_signal' : np.ndarray
            Reconstructed fundamental signal from fitting.
        - 'error' : np.ndarray
            Residual error (signal - fitted_signal).
        - 'codes' : np.ndarray
            Quantized code values for each sample.
        - 'fundamental_amplitude' : float
            Amplitude of fitted fundamental component.
        - 'dc_offset' : float
            DC offset of fitted signal.
        - 'num_bits' : int
            Number of bits (inferred or provided).
        - 'code_min' : int
            Minimum code value after clipping.
        - 'code_max' : int
            Maximum code value after clipping.

    Notes
    -----
    The code-binned approach quantizes the signal to ADC codes, then computes
    error statistics (mean and RMS) for each code value:

        emean(code) = mean(error | signal ≈ code)
        erms(code) = RMS(error | signal ≈ code)

    This is similar to the histogram method used for INL/DNL calculation,
    but operates on error values rather than hit counts.

    Auto-detection logic for num_bits:
    - If input is integer type → Treated as ADC Codes
    - If input is float and range > 2.0 → Treated as Codes
    - If input is float and range <= 2.0 → Treated as Voltage (quantized)

    Examples
    --------
    >>> sig = np.sin(2*np.pi*0.1*np.arange(1000))
    >>> result = compute_error_by_code(sig, normalized_freq=0.1, num_bits=10)
    >>> print(f"Code range: {result['code_min']} to {result['code_max']}")
    >>> print(f"Mean error shape: {result['emean_by_code'].shape}")
    """
    # Validate inputs
    signal = np.asarray(signal).flatten()
    n_samples = len(signal)

    if not (0 < normalized_freq < 0.5):
        raise ValueError(f"normalized_freq must be in range (0, 0.5), got {normalized_freq}")

    # Step 1: Fit fundamental sinewave (DC + cos + sin)
    W, fitted_signal, basis_matrix, phase = fit_sinewave_components(
        signal, freq=normalized_freq, order=1, include_dc=True
    )

    # Extract DC and fundamental amplitude
    dc_offset = W[0]
    cos_coeff = W[1]
    sin_coeff = W[2]
    fundamental_amplitude = np.sqrt(cos_coeff**2 + sin_coeff**2)

    # Step 2: Compute residual error
    error = signal - fitted_signal

    # Step 3: Quantize signal to ADC codes
    signal_min = np.min(signal)
    signal_max = np.max(signal)
    signal_range = signal_max - signal_min

    # Detect if input is voltage or codes
    is_voltage = False
    if np.issubdtype(signal.dtype, np.floating):
        if signal_range <= 2.0:
            is_voltage = True

    if is_voltage:
        # Voltage input: quantize to codes
        if num_bits is None:
            num_bits = 10  # Default to 10 bits

        adc_full_scale = 2**num_bits

        # Normalize signal to [0, 1]
        if signal_min < -0.1:
            # Bipolar: -1 to 1
            signal_normalized = (signal + 1.0) / 2.0
        else:
            # Unipolar: 0 to 1
            signal_normalized = (signal - signal_min) / (signal_range + 1e-10)

        # Quantize to codes
        codes = np.round(signal_normalized * (adc_full_scale - 1)).astype(int)
        codes = np.clip(codes, 0, adc_full_scale - 1)
    else:
        # Code input
        codes = np.round(signal).astype(int)

        if num_bits is None:
            # Infer from signal range
            if signal_range == 0:
                signal_range = 1
            num_bits = int(np.ceil(np.log2(signal_range + 1)))

    # Step 4: Apply clipping to exclude edge codes
    code_min_orig = np.min(codes)
    code_max_orig = np.max(codes)
    code_range = code_max_orig - code_min_orig

    exclusion_amount = int(np.round(clip_percent * code_range))
    code_min = code_min_orig + exclusion_amount
    code_max = code_max_orig - exclusion_amount
    code_min = min(code_min, code_max)  # Ensure valid range

    # Clip codes to exclusion range
    codes_clipped = np.clip(codes, code_min, code_max)

    # Step 5: Bin error by code value
    code_bins = np.arange(code_min, code_max + 1)
    n_bins = len(code_bins)

    # Initialize binning arrays
    bin_counts_arr = np.zeros(n_bins)
    error_sum = np.zeros(n_bins)
    error_sq_sum = np.zeros(n_bins)

    # Accumulate statistics per code
    for i in range(n_samples):
        code_val = codes_clipped[i]
        bin_idx = code_val - code_min
        if 0 <= bin_idx < n_bins:
            bin_counts_arr[bin_idx] += 1
            error_sum[bin_idx] += error[i]
            error_sq_sum[bin_idx] += error[i]**2

    # Compute mean and RMS per code
    with np.errstate(divide='ignore', invalid='ignore'):
        emean_by_code = np.where(bin_counts_arr > 0, error_sum / bin_counts_arr, np.nan)
        erms_by_code = np.where(
            bin_counts_arr > 0,
            np.sqrt(error_sq_sum / bin_counts_arr),
            np.nan
        )

    # Return comprehensive results
    return {
        'emean_by_code': emean_by_code,
        'erms_by_code': erms_by_code,
        'code_bins': code_bins,
        'bin_counts': bin_counts_arr,
        'fitted_signal': fitted_signal,
        'error': error,
        'codes': codes_clipped,
        'fundamental_amplitude': float(fundamental_amplitude),
        'dc_offset': float(dc_offset),
        'num_bits': int(num_bits),
        'code_min': int(code_min),
        'code_max': int(code_max),
        'normalized_freq': float(normalized_freq),
    }
