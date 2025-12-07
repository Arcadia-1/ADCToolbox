"""
Coherent Frequency Calculator.

Calculates the nearest coherent sampling frequency to avoid spectral leakage.
Ensures the number of cycles is an integer and coprime with the FFT size.
"""

import math


def calc_coherent_freq(fs, fin_target, n_fft, force_odd=True, search_radius=200):
    """
    Calculate the precise coherent input frequency and bin index.
    
    Supports Undersampling (Fin > Fs/2).
    """
    # 1. Calculate the ideal (fractional) total cycles
    target_bin_float = fin_target / fs * n_fft
    
    # 2. Define search center (nearest integer)
    center_int = int(round(target_bin_float))
    
    candidates = []
    
    # 3. Search neighborhood for the best candidate
    for i in range(-search_radius, search_radius + 1):
        bin_candidate = center_int + i
        
        # Validity checks: Only check if positive. 
        # REMOVED: "or bin_candidate >= n_fft // 2" to allow undersampling/high freq.
        if bin_candidate <= 0:
            continue
            
        # Condition A: Force Odd (Standard practice)
        if force_odd and (bin_candidate % 2 == 0):
            continue
            
        # Condition B: Coprime Check (GCD == 1)
        # Even if M > N, if they are coprime, the aliased bin will be unique.
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