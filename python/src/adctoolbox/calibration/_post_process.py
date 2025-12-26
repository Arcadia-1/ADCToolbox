"""
Assemble and normalize calibration results into physical quantities.

This module handles the final stage of ADC calibration: converting the
least-squares solution vector into physical weights, reconstructing signals,
and computing performance metrics (SNDR, ENOB).
"""

import numpy as np

def _post_process(
    weights_final: np.ndarray,
    solution_vector: np.ndarray,
    norm_factor: float,
    basis_choice: int,
    bit_segments: list,
    bit_width_effective: int,
    segment_lengths: np.ndarray,
    harmonic_order: int,
    cos_basis: np.ndarray,
    sin_basis: np.ndarray,
    freq_array: np.ndarray
) -> dict:
    """
    Unified result assembly for single and multi-dataset calibration.
    """
    num_datasets = len(bit_segments)
    num_harm_total = num_datasets * harmonic_order
    is_single = (num_datasets == 1)

    # Copy weights for local modification (polarity enforcement)
    weights = weights_final.copy()

    dc_offset = -solution_vector[bit_width_effective] / norm_factor

    # 3. Reconstruct per-dataset quantities
    harm_start = bit_width_effective + num_datasets
    
    calibrated_signals = []
    reference_sines = []
    residual_errors = []
    snr_list = []
    enob_list = []

    row_start = 0
    for k in range(num_datasets):
        n_k = segment_lengths[k]
        row_end = row_start + n_k
        
        # Signal reconstruction
        sig_k = weights @ bit_segments[k].T
        
        # Ideal reference reconstruction
        ref_k = _reconstruct_sine_k(
            solution_vector, k, harmonic_order, num_harm_total,
            harm_start, basis_choice, 
            cos_basis[row_start:row_end, k*harmonic_order:(k+1)*harmonic_order],
            sin_basis[row_start:row_end, k*harmonic_order:(k+1)*harmonic_order],
            norm_factor
        )
        
        # Performance metrics
        err_k = sig_k - dc_offset - ref_k
        p_sig = np.mean(ref_k**2)
        p_noise = np.mean(err_k**2)
        sndr_k = 10 * np.log10(p_sig / p_noise) if p_noise > 0 else 200.0 # Cap max SNR
        enob_k = (sndr_k - 1.76) / 6.02

        calibrated_signals.append(sig_k)
        reference_sines.append(ref_k)
        residual_errors.append(err_k)
        snr_list.append(sndr_k)
        enob_list.append(enob_k)
        
        row_start = row_end

    # 4. Polarity Correction
    polarity = -1.0 if np.sum(weights) < 0 else 1.0
    
    weights *= polarity
    dc_offset *= polarity
    
    # Using list comprehensions for compact correction
    calibrated_signals = [s * polarity for s in calibrated_signals]
    reference_sines = [r * polarity for r in reference_sines]
    residual_errors = [e * polarity for e in residual_errors]

    # 5. Format final output dictionary
    return {
        'weight': weights,
        'offset': dc_offset,
        'calibrated_signal': calibrated_signals,
        'ideal': reference_sines,
        'error': residual_errors,
        'refined_frequency': freq_array[0] if is_single else freq_array,
        'snr_db': snr_list[0] if is_single else snr_list,
        'enob': enob_list[0] if is_single else enob_list,
    }

def _reconstruct_sine_k(
    solution_vector: np.ndarray,
    k: int,
    harmonic_order: int,
    num_harm_total: int,
    harm_start: int,
    basis_choice: int,
    cos_basis_k: np.ndarray,
    sin_basis_k: np.ndarray,
    norm_factor: float
) -> np.ndarray:
    """
    Reconstruct the ideal reference sinewave for the k-th dataset.
    Handles index mapping based on basis_choice (0: Cos=1, 1: Sin=1).
    """
    # The size of the first half of harmonic columns (where one fund is unity)
    unity_side_size = num_harm_total - 1 

    if basis_choice == 0:  # Cosine is unity
        # Cosine coeffs: first one is implicit 1.0, others are in solution_vector
        # Indices for dataset k: harm_start + k*(H-1) : harm_start + (k+1)*(H-1)
        h_idx = harm_start + k * (harmonic_order - 1)
        cos_coeffs_k = np.concatenate([[1.0], solution_vector[h_idx : h_idx + harmonic_order - 1]])
        
        # Sine coeffs: all are in solution_vector, after the unity_side_size block
        sin_idx = harm_start + unity_side_size + k * harmonic_order
        sin_coeffs_k = solution_vector[sin_idx : sin_idx + harmonic_order]
    else:  # Sine is unity
        # Sine coeffs: first one is implicit 1.0
        h_idx = harm_start + k * (harmonic_order - 1)
        sin_coeffs_k = np.concatenate([[1.0], solution_vector[h_idx : h_idx + harmonic_order - 1]])
        
        # Cosine coeffs
        cos_idx = harm_start + unity_side_size + k * harmonic_order
        cos_coeffs_k = solution_vector[cos_idx : cos_idx + harmonic_order]

    # Reference = -(Basis_Cos * Coeffs_Cos + Basis_Sin * Coeffs_Sin) / Norm
    return -(cos_basis_k @ cos_coeffs_k + sin_basis_k @ sin_coeffs_k) / norm_factor
