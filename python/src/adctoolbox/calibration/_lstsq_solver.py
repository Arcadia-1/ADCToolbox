"""Core least-squares solver for calibration at fixed frequency."""

import numpy as np

from scipy.linalg import lstsq

def _build_harmonic_basis(
    n_samples: int,
    freq: float,
    harmonic_order: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct cosine and sine harmonic basis matrices for least-squares fitting.

    This function generates a set of orthogonal basis vectors representing the 
    fundamental frequency and its higher-order harmonics. These matrices are 
    typically used as components of the design matrix $A$ to isolate harmonic 
    distortions from the signal of interest.

    Mathematical Definition:
        For the $n$-th sample and $k$-th harmonic:
        cos_basis[n, k-1] = cos(2 * pi * freq * n * k)
        sin_basis[n, k-1] = sin(2 * pi * freq * n * k)

    Parameters
    ----------
    n_samples : int
        Total number of samples ($N$).
    freq : float
        Normalized input frequency ($f_{in} / f_s$).
    harmonic_order : int
        Total number of harmonics to include ($H$). 
        A value of 1 includes only the fundamental frequency.

    Returns
    -------
    cos_basis : np.ndarray
        Cosine basis matrix of shape (n_samples, harmonic_order).
    sin_basis : np.ndarray
        Sine basis matrix of shape (n_samples, harmonic_order).

    Notes
    -----
    The phase matrix is computed using the outer product of the time vector 
    and the harmonic order vector to leverage NumPy's vectorized performance.
    """
    t = np.arange(n_samples)
    harmonics = np.arange(1, harmonic_order + 1)
    phase = 2.0 * np.pi * freq * np.outer(t, harmonics)
    return np.cos(phase), np.sin(phase)

def _dual_basis_lstsq(
    A_common: np.ndarray,
    cos_basis: np.ndarray,
    sin_basis: np.ndarray,
    offset_matrix: np.ndarray,
    deriv_matrix: np.ndarray | None = None
) -> tuple[np.ndarray, int, float]:
    """
    Core dual-basis solver: tries both cosine=1 and sine=1 assumptions.
    Returns the solution with the minimum residual.
    """
    # Combine extra columns (Offsets and potentially Frequency Derivatives)
    extra_cols = [offset_matrix]
    if deriv_matrix is not None:
        extra_cols.append(deriv_matrix)

    # Assumption 1: Cosine fundamental is unity
    # Layout: [Weights, Offsets, (Derivs), Cos_harmonics[1:], Sin_harmonics[:]]
    A1 = np.column_stack([A_common, *extra_cols, cos_basis[:, 1:], sin_basis])
    b1 = -cos_basis[:, 0]
    coeffs1, _, _, _ = lstsq(A1, b1)
    err1 = np.linalg.norm(A1 @ coeffs1 - b1)

    # Assumption 2: Sine fundamental is unity
    # Layout: [Weights, Offsets, (Derivs), Sin_harmonics[1:], Cos_harmonics[:]]
    A2 = np.column_stack([A_common, *extra_cols, sin_basis[:, 1:], cos_basis])
    b2 = -sin_basis[:, 0]
    coeffs2, _, _, _ = lstsq(A2, b2)
    err2 = np.linalg.norm(A2 @ coeffs2 - b2)
    
    print(f"\nDEBUG: err1 (Cos assumption) = {err1:.2e}")
    print(f"DEBUG: err2 (Sin assumption) = {err2:.2e}")

    if err1 < err2:
        return coeffs1, 0, float(err1)
    else:
        return coeffs2, 1, float(err2)

    
def _solve_weights_with_known_freq(
    bits_list: list[np.ndarray],
    freq_array: np.ndarray,
    harmonic_order: int
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """
    Static solve at known frequencies (unified for single/multi datasets).

    Builds per-dataset harmonic bases and solves for shared weights across
    all datasets using dual-basis least-squares. For single dataset, pass
    `bits_list=[bits]` and `freq_array=np.array([freq])`.

    Parameters
    ----------
    bits_list : list of ndarray
        List of patched bits arrays, one per dataset. Each has shape (n_samples_k, bit_width).
    freq_array : ndarray
        Array of frequencies, one per dataset
    harmonic_order : int
        Number of harmonics per dataset

    Returns
    -------
    coeffs : ndarray
        Solution vector containing [weights; DC_1; DC_2; ...; harmonics]
    basis_choice : int
        Selected basis (0: cosine=1, 1: sine=1)
    cos_basis : ndarray
        Cosine basis matrix (n_samples_total, num_datasets * harmonic_order)
    sin_basis : ndarray
        Sine basis matrix (n_samples_total, num_datasets * harmonic_order)
    """
    num_datasets = len(bits_list)
    segment_lengths = np.array([bits.shape[0] for bits in bits_list])
    n_samples_total = sum(segment_lengths)
    num_harmonic_cols = num_datasets * harmonic_order

    # Stack all bits vertically
    bits_effective_stacked = np.vstack(bits_list)

    # Build per-dataset harmonic bases with zero-padding
    cos_basis_list = []
    sin_basis_list = []
    
    for k in range(num_datasets):
        # Build basis for dataset k
        n_samples_k = segment_lengths[k]
        cos_basis_k, sin_basis_k = _build_harmonic_basis(n_samples_k, freq_array[k], harmonic_order)

        # Pad with zeros for other datasets' harmonics
        cos_basis_full = np.zeros((n_samples_k, num_harmonic_cols))
        sin_basis_full = np.zeros((n_samples_k, num_harmonic_cols))
        cos_basis_full[:, k*harmonic_order:(k+1)*harmonic_order] = cos_basis_k
        sin_basis_full[:, k*harmonic_order:(k+1)*harmonic_order] = sin_basis_k

        cos_basis_list.append(cos_basis_full)
        sin_basis_list.append(sin_basis_full)

    # Stack bases vertically
    cos_basis = np.vstack(cos_basis_list)
    sin_basis = np.vstack(sin_basis_list)

    # Build offset matrix for DC terms
    offset_matrix = np.zeros((n_samples_total, num_datasets))
    row_start = 0

    for ds_idx, n_samples_k in enumerate(segment_lengths):
        row_end = row_start + n_samples_k
        offset_matrix[row_start : row_end, ds_idx] = 1.0
        row_start = row_end

    # Solve using dual-basis least-squares
    coeffs, basis_choice, _ = _dual_basis_lstsq(bits_effective_stacked, cos_basis, sin_basis, offset_matrix)

    return coeffs, basis_choice, cos_basis, sin_basis

def _solve_weights_searching_freq(
    bits_list: list[np.ndarray],
    freq_array_init: np.ndarray,
    harmonic_order: int,
    learning_rate: float = 0.5,
    reltol: float = 1e-12,
    max_iter: int = 100,
    verbose: int = 0
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    """
    Iterative frequency refinement using gradient descent (unified for single/multi datasets).

    This function performs multi-dimensional frequency refinement while maintaining
    shared bit weights across all datasets. For single dataset, pass `bits_list=[bits]`
    and `freq_array_init=np.array([freq])`.

    The process iteratively tries "which set of shared weights minimizes the total
    residual across all datasets":
    1. Solves for shared weights at current frequencies
    2. Computes per-dataset gradients of residual w.r.t. frequency
    3. Updates all frequencies jointly to minimize total residual
    4. Repeats until convergence

    Parameters
    ----------
    bits_list : list of ndarray
        List of patched bits arrays, one per dataset
    freq_array_init : ndarray
        Initial frequency estimates per dataset
    harmonic_order : int
        Number of harmonics per dataset
    learning_rate : float
        Adaptive learning rate for frequency updates
    reltol : float
        Relative error tolerance
    max_iter : int
        Maximum iterations
    verbose : int
        Print progress messages

    Returns
    -------
    freq_array : ndarray
        Refined frequencies per dataset
    coeffs : ndarray
        Final solution vector (excludes frequency correction columns)
    basis_choice : int
        Selected basis (0=cosine unity, 1=sine unity)
    cos_basis : ndarray
        The final cosine basis matrix used for the solve
    sin_basis : ndarray
        The final sine basis matrix used for the solve
    """

    num_datasets = len(bits_list)
    segment_lengths = np.array([bits.shape[0] for bits in bits_list])
    bit_width = bits_list[0].shape[1]
    n_samples_total = sum(segment_lengths)

    freq_array = freq_array_init.copy()
    delta_f_array = np.zeros(num_datasets)
    coeffs = None
    basis_choice = None

    # Stack all bits vertically
    bits_effective_stacked = np.vstack(bits_list)

    # Build derivative columns for each dataset        
    offset_matrix = np.zeros((n_samples_total, num_datasets))
  
    # Extract harmonic coefficients for this dataset
    harmonic_start = bit_width + num_datasets  # After weights and single DC term
    unity_harmonics_size = num_datasets * harmonic_order - 1  # Total harmonics minus fundamental
    
    row_start = 0
    for ds_idx, n_samples_k in enumerate(segment_lengths):
        row_end = row_start + n_samples_k
        offset_matrix[row_start : row_end, ds_idx] = 1.0
        row_start = row_end

    for ii in range(max_iter):
        # Update all frequencies
        freq_array = freq_array + delta_f_array

        # Solve for current coefficients
        coeffs_temp, basis_choice_temp, cos_basis, sin_basis = _solve_weights_with_known_freq(
            bits_list, freq_array, harmonic_order
        )

        # Extract coefficients based on which basis is unity
        # Layout when basis_choice=0: [weights, DC, cos_harmonics[1:], sin_harmonics[:]]
        # Layout when basis_choice=1: [weights, DC, sin_harmonics[1:], cos_harmonics[:]]
        derivative_cols = []
        row_start = 0
        harmonic_orders = np.arange(1, harmonic_order + 1)
        for k in range(num_datasets):
            row_end = row_start + segment_lengths[k]            
            n_samples_k = segment_lengths[k]

            # For dataset k, find its harmonics in the coefficient vector
            if basis_choice_temp == 0:
                # Cosine is unity: [weights, DC, cos[1:], sin[:]]
                # cos[0] is implicit 1, cos[1:] are in coeffs
                # For dataset k, cos harmonics are at columns k*harmonic_order : (k+1)*harmonic_order
                h_idx = harmonic_start + k * (harmonic_order - 1)
                cos_coeffs_k = np.concatenate([[1.0], coeffs_temp[h_idx : h_idx + harmonic_order - 1]])

                # Sin coefficients start after cos
                sin_idx = harmonic_start + unity_harmonics_size + k * harmonic_order
                sin_coeffs_k = coeffs_temp[sin_idx : sin_idx + harmonic_order]
            else:
                # Sine is unity: [weights, DC, sin[1:], cos[:]]
                sin_idx = harmonic_start + (k * harmonic_order - 1)
                sin_coeffs_k = coeffs_temp[sin_idx : sin_idx + harmonic_order]

                # Cos coefficients start after sin
                cos_idx = harmonic_start + unity_harmonics_size + k * harmonic_order
                cos_coeffs_k = coeffs_temp[cos_idx : cos_idx + harmonic_order]

            # Build coefficient matrices
            cos_coeffs_weighted = cos_coeffs_k * harmonic_orders
            sin_coeffs_weighted = sin_coeffs_k * harmonic_orders

            # Compute time and phase grids
            time_grid = np.arange(n_samples_k)[:, np.newaxis]
            phase_grid = 2.0 * np.pi * freq_array[k] * time_grid * harmonic_orders

            # Partial derivatives w.r.t. frequency for this dataset
            cos_basis_deriv = -2 * np.pi * cos_coeffs_weighted * time_grid * np.sin(phase_grid)
            sin_basis_deriv = 2 * np.pi * sin_coeffs_weighted * time_grid * np.cos(phase_grid)

            # Sum derivatives for this dataset
            deriv_col_k = np.sum(cos_basis_deriv + sin_basis_deriv, axis=1) / n_samples_k

            # Build full column (zeros for other datasets)
            deriv_col_full = np.zeros(n_samples_total)
            deriv_col_full[row_start:row_end] = deriv_col_k

            derivative_cols.append(deriv_col_full)
            row_start = row_end

        # Stack all derivative columns horizontally
        deriv_matrix = np.column_stack(derivative_cols)

        # Solve using dual-basis least-squares
        coeffs, basis_choice, _ = _dual_basis_lstsq(bits_effective_stacked, cos_basis, sin_basis, offset_matrix, deriv_matrix)

        # Extract frequency corrections (last num_datasets elements)
        delta_f_raw = coeffs[-num_datasets:]
        delta_f_array = delta_f_raw * learning_rate / segment_lengths

        # Compute relative error using first harmonic magnitude for normalization
        fundamental_other_idx = bit_width + num_datasets + (num_datasets * harmonic_order - 1)
        other_fundamental_val = coeffs[fundamental_other_idx]
        w0 = np.sqrt(1 + other_fundamental_val**2)
        relerr = np.sqrt(np.mean((deriv_matrix @ (delta_f_raw * learning_rate))**2)) / w0

        if verbose >= 1:
            header = f"[{ii+1:3d}] Iteration - RelErr: [{relerr:.2e}]"
            print(header)
            
            for k, (f, df) in enumerate(zip(freq_array, delta_f_array)):
                print(f"      Dataset [{k+1:02d}] -> Freq: [{f:12.10f}] | Delta: [{df:10.2e}]")
            
            if ii < max_iter - 1:
                print("      " + "-" * 55)

        # Stop if relative error is below tolerance
        if relerr < reltol:
            break

    return freq_array, coeffs[:-num_datasets], basis_choice, cos_basis, sin_basis
