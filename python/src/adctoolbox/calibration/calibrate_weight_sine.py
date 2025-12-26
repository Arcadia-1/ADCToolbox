"""
Foreground Calibration using Sinewave Input

Main wrapper function that uses modular helper functions for:
- Input preparation and validation
- Rank deficiency patching
- Least-squares solving with frequency refinement
- Result assembly and normalization
"""

import numpy as np

from adctoolbox.calibration._prepare_input import _prepare_input

from adctoolbox.calibration._patch_rank_deficiency import _patch_rank_deficiency
from adctoolbox.calibration._patch_rank_deficiency import _recover_rank_deficiency

from adctoolbox.calibration._scale_columns_for_conditioning import _scale_columns_for_conditioning
from adctoolbox.calibration._scale_columns_for_conditioning import _recover_columns_for_conditioning

from adctoolbox.calibration._estimate_frequencies import _estimate_frequencies

from adctoolbox.calibration._lstsq_solver import _solve_weights_with_known_freq
from adctoolbox.calibration._lstsq_solver import _solve_weights_searching_freq

from adctoolbox.calibration._post_process import _post_process

def calibrate_weight_sine(
    bits: np.ndarray | list[np.ndarray],
    freq: float | np.ndarray | None = None,
    force_search: bool = False,
    nominal_weights: np.ndarray | None = None,
    harmonic_order: int = 1,
    learning_rate: float = 0.5,
    reltol: float = 1e-12,
    max_iter: int = 100,
    verbose: int = 0
) -> dict:
    """
    FGCalSine — Foreground calibration using a sinewave input

    This function estimates per-bit weights and a DC offset for an ADC by
    fitting the weighted sum of raw bit columns to a sine series at a given
    (or estimated) normalized frequency Fin/Fs. It optionally performs a
    coarse and fine frequency search to refine the input tone frequency.

    Implementation uses a unified pipeline where single-dataset calibration
    is treated as a special case of multi-dataset calibration (N=1).

    Parameters
    ----------
    bits : ndarray or list of ndarrays
        Binary data as matrix (N rows by M cols, N is data points, M is bitwidth).
        Each row is one sample; each column is a bit/segment.
        Can also be a list of arrays for multi-dataset calibration.
    freq : float, array-like, or None, optional
        Normalized frequency Fin/Fs. Default is None (triggers auto frequency search).
        - None: Automatic frequency search
        - float: Single frequency for all datasets
        - array-like: Per-dataset frequencies for multi-dataset mode
    force_search : bool, optional
        Force fine frequency search even when frequency is provided.
        Default is False. Set to True to refine provided frequencies.
    nominal_weights : array-like, optional
        Nominal bit weights (only effective when rank is deficient).
        Default is 2^(M-1) down to 2^0.
    harmonic_order : int, optional
        Number of harmonic terms to exclude in calibration.
        Default is 1 (fundamental only, no harmonic exclusion).
        Higher values exclude more harmonics from the error term.
    learning_rate : float, optional
        Adaptive learning rate for frequency updates (0..1), default is 0.5.
    reltol : float, optional
        Relative error tolerance for convergence, default is 1e-12.
    max_iter : int, optional
        Maximum iterations for fine frequency search, default is 100.
    verbose : int, optional
        Print frequency search progress (1) or not (0), default is 0.

    Returns
    -------
    dict
        Dictionary with keys:
        - weight : ndarray
            The calibrated weights, normalized by the magnitude of sinewave.
        - offset : float
            The calibrated DC offset, normalized by the magnitude of sinewave.
        - calibrated_signal : ndarray or list of ndarrays
            The signal after calibration (single array for single dataset,
            list of arrays for multi-dataset).
        - ideal : ndarray or list of ndarrays
            The best fitted sinewave (single array for single dataset,
            list of arrays for multi-dataset).
        - error : ndarray or list of ndarrays
            The residue errors after calibration, excluding distortion
            (single array for single dataset, list of arrays for multi-dataset).
        - refined_frequency : float or ndarray
            The refined frequency from calibration (float for single dataset,
            array for multi-dataset).
    """

    # 1. Normalize input to unified format
    clean_input = _prepare_input(bits, nominal_weights, verbose)
    bits_stacked = clean_input["bits_stacked"]
    bits_segments = clean_input["bits_segments"]
    segment_lengths = clean_input["segment_lengths"]
    nominal_weights = clean_input["nominal_weights"]

    
    # 2. Patch rank deficiency globally
    patched_input = _patch_rank_deficiency(bits_stacked, nominal_weights, verbose)
    bits_stacked_effective = patched_input["bits_effective"]
    bit_to_col_map = patched_input["bit_to_col_map"]
    bit_weight_ratios = patched_input["bit_weight_ratios"]
    bit_width_effective = patched_input["bit_width_effective"]

    # Scale columns for numerical conditioning
    bits_stacked_effective_scaled, bit_scales = _scale_columns_for_conditioning(bits_stacked_effective, verbose)

    # Estimate or validate frequencies
    freq_array = _estimate_frequencies(bits_stacked, segment_lengths, freq, verbose)

    bits_segments_scaled = []
    curr = 0
    for length in segment_lengths:
        bits_segments_scaled.append(bits_stacked_effective_scaled[curr : curr + length])
        curr += length

    if force_search or np.any(freq_array == 0):
        # Iterative frequency search (unified for single and multi-dataset)
        freq_array, coeffs, basis_choice, cos_basis, sin_basis = _solve_weights_searching_freq(
            bits_segments_scaled, freq_array, harmonic_order,
            learning_rate, reltol, max_iter, verbose=verbose
        )
    else:
        # Static solve at known frequencies (unified for single and multi-dataset)
        coeffs, basis_choice, cos_basis, sin_basis = _solve_weights_with_known_freq(
            bits_segments_scaled, freq_array, harmonic_order
        )

    num_datasets = len(bits_segments)
    num_harm_total = num_datasets * harmonic_order
    idx_quadrature = bit_width_effective + num_harm_total
    norm_factor = np.sqrt(1.0 + coeffs[idx_quadrature]**2)

    w_phys_effective = _recover_columns_for_conditioning(
        coeffs=coeffs,
        bit_width_effective=bit_width_effective,
        norm_factor=norm_factor,
        bit_scales=bit_scales
    )

    weights_final = _recover_rank_deficiency(
        w_effective=w_phys_effective,
        bit_to_col_map=bit_to_col_map,
        bit_weight_ratios=bit_weight_ratios
    )

    # 8. Assemble results (Unified for single and multi-dataset)
    results = _post_process(
        weights_final=weights_final,
        solution_vector=coeffs,
        norm_factor=norm_factor,
        basis_choice=basis_choice,
        bit_segments=bits_segments,
        bit_width_effective=bit_width_effective,
        segment_lengths=segment_lengths,
        harmonic_order=harmonic_order,
        cos_basis=cos_basis,
        sin_basis=sin_basis,
        freq_array=freq_array
    )

    return results
