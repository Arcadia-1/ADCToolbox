"""
Helper functions for initial frequency estimation in ADC calibration.

Known Bugs:
- For very short N (sample size) or certain frequency points, the coarse estimation will fail.
- Known phenomenon:
  - Frequency 0.2 may be detected as 0.4 (2nd harmonic)
  - Frequency 0.5 (Nyquist) is definitely failed
"""

from adctoolbox.fundamentals.frequency import estimate_frequency
from adctoolbox.spectrum.compute_spectrum import compute_spectrum
import numpy as np
from collections import Counter

def _estimate_frequencies(
    bits_stacked: np.ndarray,
    segment_lengths: np.ndarray,
    freq_init: float | np.ndarray | None = 0,
    verbose: int = 0,
    frequency_policy: str = "python",
    nominal_weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Standardize or estimate the starting frequency for each dataset segment.

    ``frequency_policy="python"`` preserves the historical Python coarse
    estimator based on bit-toggle ordering and spectral quality.  Use
    ``frequency_policy="matlab"`` to emulate MATLAB ``wcalsin(freq=0)``:
    estimate a frequency from nominally reconstructed prefixes of the
    rank-patched bit matrix, then take their median.
    """
    num_datasets = len(segment_lengths)
    if frequency_policy not in {"python", "matlab"}:
        raise ValueError(
            "frequency_policy must be 'python' or 'matlab'; "
            f"got {frequency_policy!r}."
        )
    
    # --- Part 1: Handle User-Provided Frequencies ---
    # If freq_init is not 0/None, we validate and return
    if freq_init is not None and not np.all(np.asarray(freq_init) == 0):
        if np.isscalar(freq_init):
            return np.full(num_datasets, float(freq_init))
        
        freq_array = np.asarray(freq_init, dtype=float)
        if len(freq_array) != num_datasets:
            raise ValueError(
                f"Frequency array length ({len(freq_array)}) must match "
                f"number of datasets ({num_datasets})."
            )
        return freq_array

    if frequency_policy == "matlab":
        return _estimate_frequencies_matlab(
            bits_stacked=bits_stacked,
            segment_lengths=segment_lengths,
            nominal_weights=nominal_weights,
            verbose=verbose,
        )

    freq_array = np.zeros(num_datasets)
    row_offsets = np.insert(np.cumsum(segment_lengths), 0, 0)

    for k in range(num_datasets):
        start, end = row_offsets[k], row_offsets[k+1]
        current_segment = bits_stacked[start:end, :]

        toggles = np.sum(np.diff(current_segment, axis=0) != 0, axis=0)
        sorted_indices = np.argsort(toggles)
        
        n_use = min(6, current_segment.shape[1])
        assumed_weights = 2 ** np.arange(n_use - 1, -1, -1)

        if verbose == 2:
            print(f"\n[current dataset {k+1}/{num_datasets}] Samples: {segment_lengths[k]}")
            print(f"  - Bit toggle counts: {toggles}")
            print(f"  - Sorted bit indices (low to high toggles): {sorted_indices}")

        indices_A = sorted_indices[:n_use]
        sig_A = (current_segment[:, indices_A] @ assumed_weights).astype(float)
        
        indices_B = sorted_indices[::-1][:n_use]
        sig_B = (current_segment[:, indices_B] @ assumed_weights).astype(float)

        result_A = compute_spectrum(sig_A)
        result_B = compute_spectrum(sig_B)

        bit_bins = []
        for iter_bit in range(current_segment.shape[1]):
            spec = np.abs(np.fft.rfft(current_segment[:, iter_bit].astype(float) - 0.5))
            bit_bins.append(np.argmax(spec)) 
        
        most_common_bin = Counter(bit_bins).most_common(1)[0][0]
        f_anchor = most_common_bin / segment_lengths[k]

        if verbose == 2:
            print(f"  - Bit {iter_bit}: Most common bin: {most_common_bin}, Anchor freq: {f_anchor:.6f}")

        if result_A['metrics']["sndr_dbc"] >= result_B['metrics']["sndr_dbc"]:
            winner_sig = sig_A
            mode_str = "A"
        else:
            winner_sig = sig_B
            mode_str = "B"

        f_est = estimate_frequency(winner_sig)

        freq_array[k] = f_est
        if verbose == 2:
            SNDR_A = result_A['metrics']["sndr_dbc"]
            SNDR_B = result_B['metrics']["sndr_dbc"]
            Sig_power_A = result_A['metrics']["sig_pwr_dbfs"]
            Sig_power_B = result_B['metrics']["sig_pwr_dbfs"]

            print(f"  - Dataset [{k+1}/{num_datasets}]: Mode={mode_str}, SNDR_A={SNDR_A:.1f}, SNDR_B={SNDR_B:.1f}, Freq={f_est:.6f}")
            print(f"  - Dataset [{k+1}/{num_datasets}]: Mode={mode_str}, Sig_Power_A={Sig_power_A:.1f}, Sig_Power_B={Sig_power_B:.1f}, Freq={f_est:.6f}")

    return freq_array


def _estimate_frequencies_matlab(
    bits_stacked: np.ndarray,
    segment_lengths: np.ndarray,
    nominal_weights: np.ndarray | None,
    verbose: int = 0,
) -> np.ndarray:
    """
    MATLAB-compatible coarse frequency estimator for ``wcalsin(freq=0)``.

    MATLAB estimates several candidate frequencies from the first
    ``min(bit_width, 5)`` nominally reconstructed column prefixes and uses
    their median as the starting point for fine search.
    """
    if nominal_weights is None:
        raise ValueError(
            "nominal_weights are required when frequency_policy='matlab'."
        )

    nominal_weights = np.asarray(nominal_weights, dtype=float)
    if nominal_weights.ndim != 1:
        raise ValueError("nominal_weights must be a one-dimensional array.")
    if nominal_weights.size != bits_stacked.shape[1]:
        raise ValueError(
            "nominal_weights length must match the bit matrix column count "
            f"for frequency_policy='matlab'; got {nominal_weights.size} "
            f"weights for {bits_stacked.shape[1]} columns."
        )

    num_datasets = len(segment_lengths)
    freq_array = np.zeros(num_datasets)
    row_offsets = np.insert(np.cumsum(segment_lengths), 0, 0)

    for k in range(num_datasets):
        start, end = row_offsets[k], row_offsets[k + 1]
        current_segment = bits_stacked[start:end, :]
        n_use = min(current_segment.shape[1], 5)

        candidates = np.zeros(n_use)
        for i in range(1, n_use + 1):
            reconstructed = current_segment[:, :i] @ nominal_weights[:i]
            candidates[i - 1] = estimate_frequency(reconstructed)

        freq_array[k] = float(np.median(candidates))

        if verbose == 2:
            print(
                f"\n[current dataset {k + 1}/{num_datasets}] "
                f"Samples: {segment_lengths[k]}"
            )
            print(f"  - MATLAB policy candidate frequencies: {candidates}")
            print(f"  - MATLAB policy median frequency: {freq_array[k]:.12f}")

    return freq_array
