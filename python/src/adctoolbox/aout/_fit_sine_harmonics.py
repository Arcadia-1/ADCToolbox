"""
Fit sinewave components using least-squares method.

Core mathematical kernel for decomposing signals into DC, fundamental,
and harmonic components using least-squares fitting.
"""

import numpy as np

from adctoolbox.fundamentals.frequency import fold_frequency_to_nyquist


def _fit_sine_harmonics(
    sig: np.ndarray,
    freq: float,
    order: int = 1,
    include_dc: bool = True,
    fs: float = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit sine harmonics using least-squares method.

    Constructs a design matrix with DC, fundamental, and harmonic basis
    functions, then solves the least-squares problem to extract component
    amplitudes and phases.

    Parameters
    ----------
    sig : np.ndarray
        Input signal, 1D numpy array.
    freq : float
        Input frequency. If fs is provided, freq is interpreted as physical
        frequency (Hz), otherwise as normalized frequency (f/fs).
    order : int, default=1
        Harmonic order to fit. order=1 fits only fundamental, order=N fits
        fundamental through Nth harmonic.
    include_dc : bool, default=True
        Include DC component in the fitting.
    fs : float, optional
        Sampling frequency (Hz). If provided, freq is interpreted as physical
        frequency. If None, freq is interpreted as normalized frequency (0-0.5).

    Returns
    -------
    W : np.ndarray
        Fitted coefficients vector. If include_dc=True, shape is (2*order+1,):
        [DC, Cos(H1), Sin(H1), Cos(H2), Sin(H2), ..., Cos(Hn), Sin(Hn)]
        If include_dc=False, shape is (2*order,): [Cos(H1), Sin(H1), ...]
    fitted_signal : np.ndarray
        Reconstructed signal using fitted coefficients, shape matches input sig.
    basis_matrix : np.ndarray
        Design matrix A. Shape is (n_samples, 2*order) or (n_samples, 2*order+1)
        depending on include_dc. Columns are [DC?, Cos(H1), Sin(H1), ...].
    time_vector : np.ndarray
        Phase vector used for basis construction, shape (n_samples,).
        Computed as 2*pi*freq*t (or 2*pi*normalized_freq*t).

    Notes
    -----
    The function uses numpy.linalg.lstsq to solve:
        min ||sig - A @ W||^2
    where A is the design matrix and W contains all coefficients.

    Examples
    --------
    >>> sig = np.sin(2*np.pi*0.1*np.arange(1000))  # Normalized freq 0.1
    >>> W, sig_fit, A, phase = _fit_sine_harmonics(sig, freq=0.1, order=1)
    >>> # W[0] ≈ DC offset, W[1] ≈ cos amplitude, W[2] ≈ sin amplitude
    """
    # Prepare input
    sig = np.asarray(sig).flatten()
    n_samples = len(sig)
    t = np.arange(n_samples)

    # Compute phase vector
    if fs is not None:
        # freq is physical frequency (Hz)
        normalized_freq = freq / fs
    else:
        # freq is already normalized (0-0.5)
        normalized_freq = freq

    phase = 2 * np.pi * normalized_freq * t

    # Build design matrix (vectorized)
    # Start with DC if requested
    if include_dc:
        dc_basis = np.ones((n_samples, 1))
    else:
        dc_basis = None

    # Build harmonic basis: cos and sin for each harmonic order, tracking
    # active (non-aliased) columns so rank-deficient aliasing collisions
    # (e.g. f=0.25 where H3/H5 alias onto H1, H4 onto DC, H2 sin==0) do not
    # split fundamental energy into fake harmonics.
    harmonic_basis_list = []
    active_cols = [True] if include_dc else []
    seen_freqs = [0.0] if include_dc else []

    for h in range(1, order + 1):
        cos_h = np.cos(h * phase)
        sin_h = np.sin(h * phase)
        harmonic_basis_list.append(cos_h)
        harmonic_basis_list.append(sin_h)

        f_fold = float(fold_frequency_to_nyquist(h * normalized_freq, 1.0))
        collides = any(abs(f_fold - f_prev) < 1e-9 for f_prev in seen_freqs)
        if collides:
            active_cols.extend([False, False])
        else:
            seen_freqs.append(f_fold)
            # At exact DC or Nyquist, sin(h * phase) == 0 identically.
            sin_active = (f_fold >= 1e-9) and (abs(f_fold - 0.5) >= 1e-9)
            active_cols.extend([True, sin_active])

    harmonic_basis = np.column_stack(harmonic_basis_list)

    # Combine DC and harmonic basis
    if include_dc:
        basis_matrix = np.column_stack([dc_basis, harmonic_basis])
    else:
        basis_matrix = harmonic_basis

    active_mask = np.asarray(active_cols, dtype=bool)
    W = np.zeros(basis_matrix.shape[1], dtype=float)
    if np.any(active_mask):
        W_active, _, _, _ = np.linalg.lstsq(basis_matrix[:, active_mask], sig, rcond=None)
        W[active_mask] = W_active

    # Reconstruct fitted signal
    fitted_signal = basis_matrix @ W

    return W, fitted_signal, basis_matrix, phase
