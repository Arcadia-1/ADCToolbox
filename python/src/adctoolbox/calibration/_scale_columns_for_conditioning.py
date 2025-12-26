
import numpy as np

def _scale_columns_for_conditioning(
    bits: np.ndarray,
    verbose: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scale bit columns by powers of 10 for numerical conditioning.
    """
    # 1. Identify dynamic range
    col_extremes = np.vstack([np.max(bits, axis=0), np.min(bits, axis=0)])
    max_vals = np.max(np.abs(col_extremes), axis=0)

    # 2. Compute order of magnitude
    bit_scales = np.floor(np.log10(max_vals + 1e-15))

    # Identify near-zero columns
    near_zero_bits = (max_vals <= 1e-15)
    bit_scales[near_zero_bits] = 0

    if verbose >= 2 and np.any(near_zero_bits):
        idx = np.where(near_zero_bits)[0]
        print(f"[DEBUG] near-zero bit columns detected at indices: {idx.tolist()}")

    # 3. Apply scaling
    bits_effective = bits * (10.0 ** (-bit_scales))

    if verbose >= 2:
        print(f"[DEBUG] Bit column scales (powers of 10): {bit_scales.astype(int).tolist()}")

    return bits_effective, bit_scales

def _recover_columns_for_conditioning(
    coeffs: np.ndarray, 
    bit_width_effective: int, 
    norm_factor: float, 
    bit_scales: np.ndarray
) -> np.ndarray:
    """
    Inverse operation of _scale_columns_for_conditioning.
    
    This function converts the weights from the mathematically conditioned space 
    back to the physical weight space.
    
    Parameters
    ----------
    coeffs : ndarray
        The full coefficient vector returned by the solver.
    bit_width_effective : int
        Number of effective bit columns (determines the slice of weights).
    norm_factor : float
        The normalization factor (fundamental amplitude) to scale weights 
        relative to the sinewave magnitude.
    bit_scales : ndarray
        The power-of-10 scales used during the forward conditioning step.
        
    Returns
    -------
    w_physical_eff : ndarray
        The recovered weights for the effective columns in physical units.
    """
    
    # 1. Extract the raw weights from the solution vector
    w_raw = coeffs[:bit_width_effective]
    
    # 2. Reverse the normalization (scale by sinewave amplitude)
    w_normalized = w_raw / norm_factor
    
    # 3. Reverse the numerical conditioning (scale by 10^-bit_scales)
    # Since the bits were multiplied by 10^-S, the solver's weights 
    # effectively absorbed 10^S. We multiply by 10^-S to recover.
    w_physical_eff = w_normalized * (10.0 ** (-bit_scales))
    
    return w_physical_eff
