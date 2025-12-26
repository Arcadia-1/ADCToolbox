"""
Calculate ADC INL/DNL using histogram method.

Extracts INL and DNL from sinewave test data using histogram-based method.

MATLAB counterpart: inlsin.m
"""

import numpy as np

def compute_inl_from_sine(
    data: np.ndarray,
    num_bits: int | None = None,
    full_scale: float | None = None,
    clip_percent: float = 0.01
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate ADC INL/DNL.

    Auto-detection logic:
    1. If input is Integer type -> Treated as ADC Codes.
    2. If input is Float type:
       - Range > 2.0  -> Treated as ADC Codes (floating point representation of codes).
       - Range <= 2.0 -> Treated as Normalized Voltage (auto-quantized to num_bits).

    Parameters
    ----------
    data : array_like
        Input signal (Codes or Voltage).
    num_bits : int, optional
        - If Input is Voltage: Target quantization resolution (default 10 if None).
        - If Input is Codes: ADC resolution (inferred from data range if None).
    full_scale : float, optional
        Full scale voltage range for quantization. If provided with float input,
        used for quantization: codes = round(data * 2^num_bits / full_scale).
        If None, assumes normalized input (0-1 or -1 to 1).
    clip_percent : float, default=0.01
        Percentage of codes to clip from edges.
    """
    data = np.asarray(data).flatten()

    data_min = np.min(data)
    data_max = np.max(data)
    span = data_max - data_min

    # --- 1. Auto-Detect Mode (Voltage vs Code) ---
    # Default assumption: It's codes unless proven otherwise
    is_voltage = False

    if np.issubdtype(data.dtype, np.floating):
        # If float and range is small (e.g. 0.0 to 1.0), it's likely voltage
        if span <= 2.0:
            is_voltage = True

    # --- 2. Process Data ---
    if is_voltage:
        # Case A: Analog Voltage Input (Quantize to Codes)
        if num_bits is None:
            num_bits = 10

        adc_full_scale = 2**num_bits

        # Quantize based on full_scale or auto-detect
        if full_scale is not None:
            # User provided full_scale: data in volts, scale to 0..2^n_bits-1
            codes_input = np.round(data / full_scale * adc_full_scale).astype(int)
        else:
            # Auto-detect: assume normalized voltage
            # Handle Bipolar (-1 to 1) vs Unipolar (0 to 1)
            if data_min < -0.1:
                # Map -1..1 to 0..FS
                data = (data + 1.0) / 2.0 * adc_full_scale
            else:
                # Map 0..1 to 0..FS
                data = data * adc_full_scale

            # Quantize to Code
            codes_input = np.round(data).astype(int)

        # Clip to valid code range
        codes_input = np.clip(codes_input, 0, adc_full_scale - 1)

    else:
        # Case B: Code Input (Integer or Float-Codes)
        codes_input = np.round(data).astype(int)

        if num_bits is None:
            # Infer resolution from the codes themselves
            # e.g. max code 1020 -> log2(1020) ~ 9.99 -> 10 bits
            if span == 0: span = 1 # Avoid log2(0)
            num_bits = int(np.ceil(np.log2(span)))
            # print(f"[Auto] Detected code input. inferred {num_bits}-bit range.")

    # Update range after processing
    c_min_orig = np.min(codes_input)
    c_max_orig = np.max(codes_input)

    # --- 3. Apply exclusion (MATLAB lines 96-102) ---
    # Calculate exclusion amount
    exclusion_amount = int(np.round(clip_percent * (c_max_orig - c_min_orig)))
    c_max = c_max_orig - exclusion_amount
    c_min = min(c_min_orig + exclusion_amount, c_max)

    # Clip data to exclusion range (CRITICAL: matches MATLAB line 102)
    codes_clipped = np.clip(codes_input, c_min, c_max)

    # Create code range and histogram (MATLAB lines 101, 106)
    code_axis = np.arange(c_min, c_max + 1)
    bins = np.arange(c_min - 0.5, c_max + 1.5, 1)
    counts, _ = np.histogram(codes_clipped, bins=bins)
    
    # Calculate CDF and apply cosine transform (same as MATLAB)
    # MATLAB: cumulative_distribution = -cos(pi * cumsum(histogram_counts) / sum(histogram_counts))
    cumulative_distribution = -np.cos(np.pi * np.cumsum(counts) / np.sum(counts))

    # DNL Calculation (MATLAB inlsin.m line 110)
    # MATLAB: dnl = cumulative_distribution(2:end-1) - cumulative_distribution(1:end-2)
    # This takes differences: [2]-[1], [3]-[2], ..., [end-1]-[end-2]
    # In Python indexing: [1]-[0], [2]-[1], ..., [-2]-[-3]
    # Which is simply np.diff(cumulative_distribution[0:-1])
    dnl = np.diff(cumulative_distribution[0:-1])

    # Align code axis (MATLAB line 111: code = code(1:end-2))
    # Drop last 2 codes
    valid_codes = code_axis[0:-2]

    # Normalize DNL to LSB units (MATLAB lines 119-122)
    # MATLAB: dnl = dnl ./ sum(dnl)
    # MATLAB: dnl = dnl * (max_data - min_data + 1) - 1
    code_range = len(code_axis)  # This is max_data - min_data + 1
    dnl = dnl / np.sum(dnl)  # Normalize to sum to 1
    dnl = dnl * code_range - 1  # Scale to code range
    dnl = dnl - np.mean(dnl)  # Remove DC offset (gain error)
    dnl = np.maximum(dnl, -1)  # Clip missing codes to -1 LSB

    # INL Calculation (MATLAB line 125)
    inl = np.cumsum(dnl)
    
    return inl, dnl, valid_codes