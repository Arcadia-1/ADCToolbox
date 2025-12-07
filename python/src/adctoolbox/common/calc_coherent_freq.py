"""
Coherent Frequency Calculator.

Calculates the nearest coherent sampling frequency to avoid spectral leakage.
Ensures the number of cycles is an integer and coprime with the FFT size.
"""

import math


def calc_coherent_freq(fs, fin_target, n_fft, force_odd=True, search_radius=20):
    """
    Calculate the precise coherent input frequency and bin index.

    This function searches for the optimal integer number of cycles (M)
    closest to the target frequency that satisfies coherent sampling conditions
    (M is coprime with N, and optionally odd).

    Args:
        fs (float): Sampling frequency (Hz).
        fin_target (float): Desired target input frequency (Hz).
        n_fft (int): FFT size / Number of samples.
        force_odd (bool): If True, restricts bin M to odd numbers (recommended).
        search_radius (int): Range of bins to search around the target (default 20).

    Returns:
        tuple: (fin_actual, bin_idx)
            - fin_actual (float): The exact frequency to set on the signal generator.
            - bin_idx (int): The integer bin index (M).

    Raises:
        ValueError: If no valid coprime bin is found within the search radius.
    """
    # 1. Calculate the ideal (fractional) bin location
    target_bin_float = fin_target / fs * n_fft
    
    # 2. Define search center (nearest integer)
    center_int = int(round(target_bin_float))
    
    candidates = []
    
    # 3. Search neighborhood for the best candidate
    for i in range(-search_radius, search_radius + 1):
        bin_candidate = center_int + i
        
        # Validity checks: Must be positive and within Nyquist zone
        if bin_candidate <= 0 or bin_candidate >= n_fft // 2:
            continue
            
        # Condition A: Force Odd (Standard practice to hit both positive/negative peaks)
        if force_odd and (bin_candidate % 2 == 0):
            continue
            
        # Condition B: Coprime Check (GCD == 1) to prevent repeating codes
        if math.gcd(bin_candidate, n_fft) == 1:
            # Score by distance to target
            dist = abs(bin_candidate - target_bin_float)
            candidates.append((dist, bin_candidate))
    
    if not candidates:
        raise ValueError(
            f"No valid prime/odd bin found near {target_bin_float:.2f} cycles "
            f"(Fin={fin_target/1e6:.2f}MHz). Try increasing search_radius."
        )
    
    # 4. Pick the winner (smallest distance)
    candidates.sort(key=lambda x: x[0])
    best_bin = candidates[0][1]
    
    # 5. Calculate precise physical frequency
    fin_actual = best_bin * fs / n_fft
    
    return fin_actual, best_bin