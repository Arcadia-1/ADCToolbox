"""
Foreground Calibration using Sinewave Input

Main wrapper function that uses modular helper functions for:
- Input preparation and validation
- Rank deficiency patching
- Least-squares solving with frequency refinement
- Result assembly and normalization
"""

import warnings

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
    force_search: bool | None = None,
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
    (or estimated) normalized frequency Fin/Fs. Harmonic terms above the
    fundamental are fitted reference/nuisance terms: they can prevent source
    or test-chain harmonics from contaminating weight estimation, but they do
    not remove those harmonics from ``calibrated_signal``. It optionally
    performs a coarse and fine frequency search to refine the input tone
    frequency.

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
        Use None for automatic frequency search, a float for one frequency
        shared by all datasets, or an array-like value for per-dataset
        frequencies in multi-dataset mode.
    force_search : bool or None, optional
        Frequency fine-search policy. Default is None, which refines
        automatically estimated frequencies while keeping explicitly provided
        frequencies fixed. Set True to refine provided frequencies too, or
        False to disable fine search unless a zero frequency placeholder
        remains.
    nominal_weights : array-like, optional
        Nominal bit weights (only effective when rank is deficient).
        Default is 2^(M-1) down to 2^0.
    harmonic_order : int, optional
        Number of harmonic terms included in the fitted reference. Default is
        1 (fundamental only). Values greater than 1 include H2/H3/... as
        nuisance terms in ``ideal`` and exclude them from ``error``. This is
        useful for source/test-chain harmonic nuisance modeling; it should not
        be interpreted as proof that ADC harmonic distortion has been removed
        from ``calibrated_signal``.
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
        Calibration result containing ``weight``, ``offset``,
        ``calibrated_signal``, ``ideal``, ``error``, and
        ``refined_frequency``. ``ideal`` includes fitted harmonics up to
        ``harmonic_order``; ``error`` is the residual after subtracting that
        fitted reference. The returned ``snr_db`` and ``enob`` are calibration
        fitted-residual metrics, not FFT dynamic SNDR/ENOB when
        ``harmonic_order`` is greater than 1. Use spectrum analysis on
        ``calibrated_signal`` for ADC dynamic SNDR/THD/HDx. The ``rank_patch``
        entry reports any dropped or merged rank-deficient bit columns.
        Array-valued entries are returned as a single array for single-dataset
        input or as a list of arrays for multi-dataset input.

        The calibrated waveform fields use ``scale_convention ==
        "solver_unit_sine"``: the least-squares solve fixes the fitted
        fundamental sine magnitude to one. Before interpreting dBFS or NSD
        against a physical ADC full-scale, rescale the result with
        ``scale_calibration_output`` and pass an explicit full-scale range to
        the spectrum analyzer.
    """

    # 0. Frequency-unit guard: freq must be NORMALIZED Fin/Fs in [0, 0.5].
    # Silent-fail (all-zero weights) used to happen when callers passed Fin in Hz.
    if freq is not None:
        _freq_check = np.atleast_1d(np.asarray(freq, dtype=float))
        if np.any(_freq_check > 0.5):
            raise ValueError(
                f"freq must be normalized Fin/Fs (Nyquist range [0, 0.5]); got {freq}. "
                f"If you have Fin in Hz, pass freq=Fin/Fs instead."
            )

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
    rank_patch_applied = patched_input["rank_patch_applied"]
    dropped_constant_bits = patched_input["dropped_constant_bits"]
    unmapped_bits = patched_input["unmapped_bits"]

    if unmapped_bits.size > 0:
        warnings.warn(
            "Some bit columns were constant or otherwise unobservable in this "
            "capture and have no recoverable AC information. Returned weights "
            "for these bits are set to 0 for this fitted model; this does not "
            "imply their physical ADC weights are zero.",
            UserWarning,
            stacklevel=2,
        )

    # Scale columns for numerical conditioning
    bits_stacked_effective_scaled, bit_scales = _scale_columns_for_conditioning(bits_stacked_effective, verbose)

    auto_frequency_requested = freq is None or np.all(np.asarray(freq) == 0)

    # Estimate or validate frequencies
    freq_array = _estimate_frequencies(bits_stacked, segment_lengths, freq, verbose)

    bits_segments_scaled = []
    curr = 0
    for length in segment_lengths:
        bits_segments_scaled.append(bits_stacked_effective_scaled[curr : curr + length])
        curr += length

    run_frequency_search = (
        (auto_frequency_requested if force_search is None else force_search)
        or np.any(freq_array == 0)
    )

    if run_frequency_search:
        # Iterative frequency search (unified for single and multi-dataset)
        freq_array, coeffs, basis_choice, cos_basis, sin_basis = _solve_weights_searching_freq(
            bits_segments_scaled, freq_array, harmonic_order,
            learning_rate, reltol, max_iter, verbose=verbose
        )
    else:
        # Static solve at known frequencies (unified for single and multi-dataset)
        coeffs, basis_choice, cos_basis, sin_basis = _solve_weights_with_known_freq(
            bits_segments_scaled, freq_array, harmonic_order, verbose=verbose
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

    results["rank_patch"] = {
        "applied": rank_patch_applied,
        "bit_width_effective": bit_width_effective,
        "bit_to_col_map": bit_to_col_map.copy(),
        "bit_weight_ratios": bit_weight_ratios.copy(),
        "dropped_constant_bits": dropped_constant_bits.copy(),
        "unmapped_bits": unmapped_bits.copy(),
    }

    return results
