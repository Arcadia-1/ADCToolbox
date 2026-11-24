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


def _build_harmonic_basis_matrix(
    n_samples: int,
    freq: float,
    harmonic_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible alias for ``_build_harmonic_basis``."""
    return _build_harmonic_basis(n_samples, freq, harmonic_order)

def _dual_basis_lstsq(
    A_common: np.ndarray,
    cos_basis: np.ndarray,
    sin_basis: np.ndarray,
    offset_matrix: np.ndarray,
    deriv_matrix: np.ndarray | None = None,
    verbose: int = 0,
    fit_filter=None,
) -> tuple[np.ndarray, int, float]:
    """
    Core dual-basis solver: tries both cosine=1 and sine=1 assumptions.
    Returns the solution with the minimum residual.
    """
    if fit_filter is not None:
        A_common = fit_filter.apply(A_common)
        cos_basis = fit_filter.apply(cos_basis)
        sin_basis = fit_filter.apply(sin_basis)
        offset_matrix = fit_filter.apply(offset_matrix)
        if deriv_matrix is not None:
            deriv_matrix = fit_filter.apply(deriv_matrix)

    def _solve_system(columns: list[np.ndarray], rhs: np.ndarray, n_protected_prefix: int, n_protected_suffix: int):
        A_full = np.column_stack(columns)
        n_cols = A_full.shape[1]
        active = np.ones(n_cols, dtype=bool)
        # Drop identically-zero harmonic columns (e.g. out-of-band harmonics
        # masked when osr > 1) so lstsq remains full-rank and well-conditioned.
        harm_end = n_cols - n_protected_suffix
        if harm_end > n_protected_prefix:
            harm_block = A_full[:, n_protected_prefix:harm_end]
            active[n_protected_prefix:harm_end] = np.any(harm_block != 0.0, axis=0)

        A_active = A_full[:, active]
        if fit_filter is not None and A_active.shape[0] <= A_active.shape[1]:
            raise ValueError("Too few samples remain after calibration filtering for the fit.")

        coeffs_active, _, _, _ = lstsq(A_active, rhs)
        coeffs_full = np.zeros(n_cols, dtype=float)
        coeffs_full[active] = coeffs_active
        err = float(np.linalg.norm(A_active @ coeffs_active - rhs))
        return coeffs_full, err

    n_prefix = A_common.shape[1] + offset_matrix.shape[1]
    n_suffix = 0 if deriv_matrix is None else deriv_matrix.shape[1]

    # Assumption 1: Cosine fundamental is unity
    # Layout: [Weights, Offsets, Cos_harmonics[1:], Sin_harmonics[:], (Derivs)]
    columns1 = [A_common, offset_matrix, cos_basis[:, 1:], sin_basis]
    if deriv_matrix is not None:
        columns1.append(deriv_matrix)
    coeffs1, err1 = _solve_system(columns1, -cos_basis[:, 0], n_prefix, n_suffix)

    # Assumption 2: Sine fundamental is unity
    # Layout: [Weights, Offsets, Sin_harmonics[1:], Cos_harmonics[:], (Derivs)]
    columns2 = [A_common, offset_matrix, sin_basis[:, 1:], cos_basis]
    if deriv_matrix is not None:
        columns2.append(deriv_matrix)
    coeffs2, err2 = _solve_system(columns2, -sin_basis[:, 0], n_prefix, n_suffix)
    
    if verbose >= 2:
        print(f"\nDEBUG: err1 (Cos assumption) = {err1:.2e}")
        print(f"DEBUG: err2 (Sin assumption) = {err2:.2e}")

    if err1 < err2:
        return coeffs1, 0, float(err1)
    else:
        return coeffs2, 1, float(err2)


def _unpack_harmonic_coefficients(
    coeffs: np.ndarray,
    bit_width: int,
    num_datasets: int,
    harmonic_order: int,
    basis_choice: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decode the per-dataset harmonic coefficients of a dual-basis solution.

    ``_dual_basis_lstsq`` fixes a single coefficient to one: the fundamental
    cosine (basis_choice=0) or sine (basis_choice=1) of the first dataset.
    Every other dataset's fundamental is an ordinary free column. The global
    layout is::

        [weights (bit_width), DC (one per dataset),
         unity-side harmonics except the fixed term (num_datasets*H - 1),
         other-side harmonics (num_datasets*H), (frequency derivatives)]

    and each harmonic block is ordered dataset-major (dataset k, harmonic h at
    position k*H + h), matching the columns of the harmonic basis matrices.

    Returns
    -------
    cos_coeffs, sin_coeffs : ndarray
        Arrays of shape (num_datasets, harmonic_order). Row k holds dataset
        k's coefficients for harmonics 1..harmonic_order, with the fixed unity
        coefficient restored only in row 0.
    """
    num_harm_total = num_datasets * harmonic_order
    harm_start = bit_width + num_datasets
    other_start = harm_start + num_harm_total - 1

    unity_side = np.concatenate([[1.0], coeffs[harm_start:other_start]])
    other_side = coeffs[other_start:other_start + num_harm_total]

    if basis_choice == 0:
        cos_coeffs, sin_coeffs = unity_side, other_side
    else:
        cos_coeffs, sin_coeffs = other_side, unity_side

    shape = (num_datasets, harmonic_order)
    return cos_coeffs.reshape(shape), sin_coeffs.reshape(shape)


def _solve_weights_with_known_freq(
    bits_list: list[np.ndarray],
    freq_array: np.ndarray,
    harmonic_order: int,
    verbose: int = 0,
    fit_filter=None,
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
    if fit_filter is not None:
        fit_filter.validate_frequencies(freq_array)
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
        if fit_filter is not None and harmonic_order > 1:
            # Exclude harmonics at or above the lowpass cutoff (0.5/osr):
            # filtering attenuates stopband columns to ~1e-5..1e-9, which would
            # let lstsq assign huge unconstrained amplitudes to stopband noise.
            harmonics = np.arange(1, harmonic_order + 1)
            out_of_band = (harmonics >= 2) & (harmonics * freq_array[k] >= fit_filter.cutoff)
            if np.any(out_of_band):
                cos_basis_k[:, out_of_band] = 0.0
                sin_basis_k[:, out_of_band] = 0.0

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
    coeffs, basis_choice, _ = _dual_basis_lstsq(
        bits_effective_stacked,
        cos_basis,
        sin_basis,
        offset_matrix,
        verbose=verbose,
        fit_filter=fit_filter,
    )

    return coeffs, basis_choice, cos_basis, sin_basis

def _solve_weights_searching_freq(
    bits_list: list[np.ndarray],
    freq_array_init: np.ndarray,
    harmonic_order: int,
    learning_rate: float = 0.5,
    reltol: float = 1e-12,
    max_iter: int = 100,
    verbose: int = 0,
    fit_filter=None,
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
            bits_list, freq_array, harmonic_order, verbose=verbose,
            fit_filter=fit_filter,
        )

        # Per-dataset harmonic coefficients; only dataset 0's fundamental is
        # the fixed unity term (see _unpack_harmonic_coefficients).
        cos_coeffs, sin_coeffs = _unpack_harmonic_coefficients(
            coeffs_temp, bit_width, num_datasets, harmonic_order, basis_choice_temp
        )
        derivative_cols = []
        row_start = 0
        harmonic_orders = np.arange(1, harmonic_order + 1)
        for k in range(num_datasets):
            row_end = row_start + segment_lengths[k]            
            n_samples_k = segment_lengths[k]

            # Build coefficient matrices
            cos_coeffs_weighted = cos_coeffs[k] * harmonic_orders
            sin_coeffs_weighted = sin_coeffs[k] * harmonic_orders

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
        coeffs, basis_choice, _ = _dual_basis_lstsq(
            bits_effective_stacked,
            cos_basis,
            sin_basis,
            offset_matrix,
            deriv_matrix,
            verbose=verbose,
            fit_filter=fit_filter,
        )

        # Extract frequency corrections (last num_datasets elements).
        # deriv_matrix was formed using cos_coeffs/sin_coeffs in basis_choice_temp's
        # gauge (where one fundamental component of dataset 0 is 1). If the
        # augmented dual-basis solve switched to the other quadrature gauge, the
        # whole linear equation was divided by gauge_scale (the newly chosen unity
        # component in the old gauge), so delta_f_raw = (N * delta_f) / gauge_scale.
        # Multiplying by gauge_scale restores the true signed N * delta_f.
        gauge_scale = cos_coeffs[0, 0] if basis_choice == 0 else sin_coeffs[0, 0]
        delta_f_raw = coeffs[-num_datasets:]
        delta_f_scaled = delta_f_raw * gauge_scale
        delta_f_array = delta_f_scaled * learning_rate / segment_lengths
        if fit_filter is not None:
            delta_f_array = fit_filter.bound_frequencies(freq_array + delta_f_array) - freq_array
        else:
            eps = np.finfo(float).eps
            delta_f_array = np.clip(freq_array + delta_f_array, eps, 0.5 - eps) - freq_array

        # Compute gauge-invariant relative error using dataset-0 fundamental norm
        w0_temp = float(np.hypot(cos_coeffs[0, 0], sin_coeffs[0, 0]))
        correction = deriv_matrix @ (delta_f_array * segment_lengths)
        if fit_filter is not None:
            correction = fit_filter.apply(correction)
        relerr = np.sqrt(np.mean(correction**2)) / w0_temp if w0_temp > 0 else 0.0

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

    # Apply the final frequency step and perform a static refit without
    # derivative columns so returned weights/reference are the exact
    # least-squares solution at the returned refined frequencies (also handles
    # max_iter=0 and early termination consistently for both osr==1 and osr>1).
    if fit_filter is not None:
        freq_array = fit_filter.bound_frequencies(freq_array + delta_f_array)
    else:
        eps = np.finfo(float).eps
        freq_array = np.clip(freq_array + delta_f_array, eps, 0.5 - eps)
    coeffs, basis_choice, cos_basis, sin_basis = _solve_weights_with_known_freq(
        bits_list, freq_array, harmonic_order, verbose=verbose, fit_filter=fit_filter
    )
    return freq_array, coeffs, basis_choice, cos_basis, sin_basis
