"""
Data Validation Utilities.

Provides rigorous checks for ADC data integrity (NaN, Inf, dimensions, etc.).
"""

import numpy as np
import warnings

def _validate_basic_integrity(data, name="Data"):
    """
    Internal helper: Check basic array properties (numeric, real, finite).
    
    Returns:
        np.ndarray: The array converted to numpy format.
    """
    data = np.asarray(data)

    # 1. Empty Check
    if data.size == 0:
        raise ValueError(f"{name} is empty.")

    # 2. Numeric Check
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError(f"{name} must be numeric, got dtype: {data.dtype}")

    # 3. Complex Check
    if np.iscomplexobj(data):
        raise ValueError(f"{name} must be real-valued, got complex numbers.")

    # 4. NaN/Inf Check (Fastest way)
    if not np.isfinite(data).all():
        if np.any(np.isnan(data)):
            raise ValueError(f"{name} contains NaN values.")
        if np.any(np.isinf(data)):
            raise ValueError(f"{name} contains Infinite values.")

    return data

def validate_aout_data(aout_data, min_samples=100):
    """
    Validate analog output data format.

    Checks: Numeric, Real, Finite, Sufficient Length, Signal Variation.
    """
    # Step 1: Basic checks (Reused)
    data = _validate_basic_integrity(aout_data, name="Analog Data")

    # Step 2: Dimension Handling
    # Standardize to (N_samples,) or (N_samples, M_channels)
    if data.ndim > 2:
        raise ValueError(f"Analog data must be 1D or 2D, got {data.ndim}D.")
    
    # Determine sample count based on shape (assuming N x M or just N)
    n_samples = data.shape[0] if data.ndim == 1 else max(data.shape)

    if n_samples < min_samples:
        raise ValueError(f"Insufficient analog samples ({n_samples}), need at least {min_samples}.")

    # Step 3: Physics/Signal Checks
    data_range = np.ptp(data) # Peak-to-peak (max - min)
    
    if data_range == 0:
        raise ValueError("Analog data is constant (flatline). Check connections.")
        
    if data_range < 1e-10:
        warnings.warn(f"Analog signal amplitude is extremely low ({data_range:.2e}). Is the input connected?")

def validate_dout_data(bits, min_samples=100):
    """
    Validate digital output (bits) data format.

    Checks: Binary (0/1), Dimensions, Stuck Bits.
    Expects: (N_samples, N_bits) matrix.
    """
    # Step 1: Basic checks (Reused)
    data = _validate_basic_integrity(bits, name="Digital Data")

    # Step 2: Dimension Check
    if data.ndim != 2:
        raise ValueError(f"Digital data must be 2D matrix (N samples x B bits), got {data.shape}.")

    n_samples, n_bits = data.shape

    # Heuristic: If bits > samples, user likely transposed the matrix
    if n_bits > n_samples:
        warnings.warn(
            f"More bits ({n_bits}) than samples ({n_samples}). "
            "Did you transpose the matrix? Expecting (N_samples, N_bits)."
        )

    if n_samples < min_samples:
        raise ValueError(f"Insufficient digital samples ({n_samples}), need at least {min_samples}.")

    if n_bits < 1:
        raise ValueError("No bits found (columns).")

    # Step 3: Binary Value Check (0 or 1 only)
    # Optimized check: Check if values are equal to themselves cast as bool (0 or 1)
    # Or use unique (slower but clearer error)
    unique_vals = np.unique(data)
    # Allow bool, int, or float as long as values are 0.0 or 1.0
    if not np.all(np.isin(unique_vals, [0, 1])):
        raise ValueError(f"Digital data must contain only 0 and 1. Found: {unique_vals}")

    # Step 4: Stuck Bit Check (High value warning)
    # Sum along time axis
    bit_sums = np.sum(data, axis=0)
    
    stuck_low = np.where(bit_sums == 0)[0]
    stuck_high = np.where(bit_sums == n_samples)[0]

    if len(stuck_low) > 0:
        warnings.warn(f"Bits stuck at 0 (Indices): {stuck_low}")
    
    if len(stuck_high) > 0:
        warnings.warn(f"Bits stuck at 1 (Indices): {stuck_high}")