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
    verbose: int = 0
) -> np.ndarray:
    """
    Standardize or estimate the starting frequency for each dataset segment.
    
    This function uses a variance-based bit selection strategy:
    1. Identify the most active bit (MSB candidate) via variance.
    2. Perform RFFT on each segment of that bit to find the coarse peak.
    """
    num_datasets = len(segment_lengths)
    
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

        if result_A['metrics']["sndr_db"] >= result_B['metrics']["sndr_db"]:
            winner_sig = sig_A
            mode_str = "A"
        else:
            winner_sig = sig_B
            mode_str = "B"

        f_est = estimate_frequency(winner_sig)

        freq_array[k] = f_est
        if verbose == 2:
            SNDR_A = result_A['metrics']["sndr_db"]
            SNDR_B = result_B['metrics']["sndr_db"]
            Sig_power_A = result_A['metrics']["sig_pwr_dbfs"]
            Sig_power_B = result_B['metrics']["sig_pwr_dbfs"]

            print(f"  - Dataset [{k+1}/{num_datasets}]: Mode={mode_str}, SNDR_A={SNDR_A:.1f}, SNDR_B={SNDR_B:.1f}, Freq={f_est:.6f}")
            print(f"  - Dataset [{k+1}/{num_datasets}]: Mode={mode_str}, Sig_Power_A={Sig_power_A:.1f}, Sig_Power_B={Sig_power_B:.1f}, Freq={f_est:.6f}")

    return freq_array