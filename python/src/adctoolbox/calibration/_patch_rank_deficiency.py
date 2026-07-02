"""Helper functions for handling rank deficiency in calibration."""
import numpy as np

def _patch_rank_deficiency(
    bits_stacked: np.ndarray,
    nominal_weights: np.ndarray,
    verbose: int = 0
) -> dict:
    """
    Unified patching for both single and multi-dataset scenarios.

    Resolves linear dependencies in the bit matrix by merging redundant columns.

    Parameters
    ----------
    bits_stacked : ndarray
        Stacked bit matrix (N_total x M)
    nominal_weights : ndarray
        Nominal bit weights (M,)
    verbose : int
        Verbosity level

    Returns
    -------
    dict
        - bits_effective : ndarray
            Patched bit matrix (N_total x M_effective)
        - bit_to_col_map : ndarray
            Maps original bit index to effective column index
        - bit_weight_ratios : ndarray
            Weight ratios for merging redundant bits
        - bit_width_effective : int
            Number of effective columns after patching
    """
    n_samples_total, bit_width = bits_stacked.shape

    # 1. Check if rank deficiency exists (Fast path)
    matrix_to_check = np.column_stack([bits_stacked, np.ones(n_samples_total)])
    # Full rank if rank >= M + 1 (including DC offset)
    if np.linalg.matrix_rank(matrix_to_check) >= bit_width + 1:
        return {
            "bits_effective": bits_stacked,
            "bit_to_col_map": np.arange(bit_width),
            "bit_weight_ratios": np.ones(bit_width),
            "bit_width_effective": bit_width,
            "rank_patch_applied": False,
            "dropped_constant_bits": np.array([], dtype=int),
            "unmapped_bits": np.array([], dtype=int),
        }

    if verbose >= 1:
        print('[INFO] Rank deficiency detected!')

    # 2. Iterative Patching Logic
    bits_effective = np.empty((n_samples_total, 0))
    eff_col_to_bit = []  # Reverse link to map effective columns back to original indices
    bit_to_col_map = np.full(bit_width, -1, dtype=int) # -1 indicates dropped/merged bits
    bit_weight_ratios = np.zeros(bit_width)
    constant_bits = []

    for bit_idx in range(bit_width):
        col = bits_stacked[:, bit_idx]

        # Case A: Constant column = dead bit, drop it
        # bit_to_col_map stays -1, bit_weight_ratios stays 0, eff_col_to_bit unchanged
        if np.ptp(col) < 1e-15:
            constant_bits.append(bit_idx)
            if verbose >= 1:
                print(f"[DEBUG] Bit [{bit_idx}] is constant (dead bit). Dropping this column.")
            continue

        # Case B: New column is linearly independent, keep it
        # bit_to_col_map = new index, weight ratio = 1.0, eff_col_to_bit = updated
        current_matrix = np.column_stack([np.ones(n_samples_total), bits_effective])
        if np.linalg.matrix_rank(np.column_stack([current_matrix, col])) > np.linalg.matrix_rank(current_matrix):
            bits_effective = np.column_stack([bits_effective, col])
            idx_eff = bits_effective.shape[1] - 1
            eff_col_to_bit.append(bit_idx)
            bit_to_col_map[bit_idx] = idx_eff
            bit_weight_ratios[bit_idx] = 1.0
            if verbose >= 2:
                print(f"[DEBUG] Bit [{bit_idx}] is independent. Keeping as new effective column [{idx_eff}].")

        # Case C: New column is linearly dependent, merge it
        # bit_to_col_map = existing index, weight ratio = nominal ratio, eff_col_to_bit = unchanged
        else:
            for col_idx, founding_bit_idx in enumerate(eff_col_to_bit):
                col_centered = col - np.mean(col)
                eff_centered = bits_effective[:, col_idx] - np.mean(bits_effective[:, col_idx])

                # Pearson correlation coefficient
                norm = np.sqrt(np.mean(col_centered**2)) * np.sqrt(np.mean(eff_centered**2))
                if norm < 1e-15:
                    correlation = 0.0
                else:
                    correlation = np.mean(col_centered * eff_centered) / norm

                if abs(abs(correlation) - 1.0) < 1e-3:
                    bit_to_col_map[bit_idx] = col_idx
                    bit_weight_ratios[bit_idx] = nominal_weights[bit_idx] / nominal_weights[founding_bit_idx]
                    # Merge contribution into effective column
                    bits_effective[:, col_idx] += col * bit_weight_ratios[bit_idx]
                    break
            if verbose >= 1:
                print(f"[DEBUG] Bit [{bit_idx}] is dependent. Merging into effective column [{bit_to_col_map[bit_idx]}] "
                      f"with weight ratio {bit_weight_ratios[bit_idx]:.4f}.")

    dropped_constant_bits = np.asarray(constant_bits, dtype=int)
    unmapped_bits = np.flatnonzero(bit_to_col_map < 0)
    if bits_effective.shape[1] == 0:
        raise ValueError(
            "No effective bit columns remain after rank-deficiency patching. "
            "All bit columns are constant in this capture, so sine-based weight "
            "calibration is not identifiable. Increase input amplitude, check "
            "bit activity, or provide a capture with sufficient code coverage."
        )

    return {
        "bits_effective": bits_effective,
        "bit_to_col_map": bit_to_col_map,
        "bit_weight_ratios": bit_weight_ratios,
        "bit_width_effective": bits_effective.shape[1],
        "rank_patch_applied": True,
        "dropped_constant_bits": dropped_constant_bits,
        "unmapped_bits": unmapped_bits,
    }

def _recover_rank_deficiency(
    w_effective: np.ndarray, 
    bit_to_col_map: np.ndarray, 
    bit_weight_ratios: np.ndarray
) -> np.ndarray:
    """
    Inverse operation of _patch_rank_deficiency.
    
    This function maps the weights solved in the 'effective bit space' back to the 
    'original bit space' (e.g., from 10 patched columns back to 12 physical bits).
    
    Parameters
    ----------
    w_effective : ndarray
        Weights extracted from the solver, corresponding to the patched columns.
    bit_to_col_map : ndarray
        Mapping array where each element indicates which effective column 
        an original bit belongs to. Value -1 indicates a discarded constant bit.
    bit_weight_ratios : ndarray
        Scaling ratios used during patching to handle merged correlated bits.
        
    Returns
    -------
    weights_recovered : ndarray
        The physical weights mapped back to the original bit dimensions.
    """
    w_effective = np.asarray(w_effective)
    bit_to_col_map = np.asarray(bit_to_col_map, dtype=int)
    bit_weight_ratios = np.asarray(bit_weight_ratios, dtype=float)

    if w_effective.size == 0:
        raise ValueError(
            "No effective bit weights are available for rank-deficiency recovery. "
            "The calibration problem is not identifiable when no effective bit "
            "columns remain."
        )

    if np.any(bit_to_col_map >= w_effective.size):
        raise ValueError(
            "rank-deficiency recovery map refers to an effective weight column "
            "that does not exist."
        )

    # 1. Broadly map effective weights back to original bit positions.
    # We use np.maximum(..., 0) to prevent -1 (discarded bits) from causing 
    # IndexError; these positions will be corrected in the final step.
    weights_recovered = w_effective[np.maximum(bit_to_col_map, 0)]
    
    # 2. Apply the weight ratios to compensate for merged columns.
    # If bit B was merged into bit A with a ratio of 0.5, its physical weight 
    # should be 0.5 * weight_A.
    weights_recovered = weights_recovered * bit_weight_ratios
    
    # 3. Force weights of discarded constant columns to zero.
    # Bits marked with -1 in bit_to_col_map had no AC information and 
    # contribute nothing to the dynamic signal reconstruction.
    weights_recovered[bit_to_col_map < 0] = 0.0
    
    return weights_recovered
