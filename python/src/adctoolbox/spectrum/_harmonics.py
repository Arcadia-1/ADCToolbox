"""Harmonic analysis helper functions for spectrum computation.

This module provides functions for:
- Finding harmonic bin positions with aliasing
- Calculating harmonic power (THD, HD2, HD3)
- Calculating harmonic phases relative to fundamental
- Calculating SFDR (Spurious-Free Dynamic Range)

This is an internal helper module, not intended for direct use by end users.
"""

import numpy as np
from adctoolbox.fundamentals.frequency import fold_bin_to_nyquist


def _locate_harmonic_bins(
    fundamental_bin: float,
    max_harmonic: int,
    n_fft: int
) -> np.ndarray:
    """Locate harmonic bin positions with aliasing, using fold_bin_to_nyquist()
    Harmonic h appears at bin = h * fundamental_bin (with aliasing)

    Parameters
    ----------
    fundamental_bin : float
        Fundamental bin position (can be fractional from interpolation)
    max_harmonic : int
        Maximum harmonic number to find (e.g., 5 means find harmonics 2-5)
    n_fft : int
        FFT length

    Returns
    -------
    harmonic_bins : np.ndarray
        Integer array of harmonic bin positions (0-based), length = max_harmonic - 1
        Contains H2, H3, ..., H(max_harmonic), rounded to nearest integer

    """
    harmonic_bins = np.zeros(max_harmonic - 1, dtype=int)

    for h in range(2, max_harmonic + 1):
        # Calculate harmonic bin (can be fractional)
        harmonic_bin = fundamental_bin * h

        # Apply aliasing using fold_bin_to_nyquist
        aliased_bin = fold_bin_to_nyquist(harmonic_bin, n_fft)

        harmonic_bins[h - 2] = round(aliased_bin)

    return harmonic_bins


def _calculate_harmonic_power(
    power_spectrum: np.ndarray,
    fundamental_bin: int,
    harmonic_bins: np.ndarray,
    side_bin: int,
    max_harmonic: int
) -> tuple[float, np.ndarray, list]:
    """Calculate harmonic power (THD and individual harmonics) from spectrum.

    Extracts power from harmonic bins while handling:
    - Aliasing collisions (when harmonics fold back to fundamental)
    - DC collisions (when harmonics fold to bin 0)
    - Double-counting prevention (multiple harmonics aliasing to same bin)

    Parameters
    ----------
    power_spectrum : np.ndarray
        Power spectrum array (half-sided)
    fundamental_bin : int
        Fundamental bin index (integer)
    harmonic_bins : np.ndarray
        Array of harmonic bin positions from _locate_harmonic_bins
    side_bin : int
        Side bins to include around each harmonic
    max_harmonic : int
        Maximum harmonic number for THD calculation (e.g., 5 means THD includes harmonics 2-5)

    Returns
    -------
    thd_power : float
        Total harmonic distortion power (sum of harmonics 2 through max_harmonic)
    harmonic_powers : np.ndarray
        Power of each harmonic (2 through max_harmonic), shape (max_harmonic-1,)
        harmonic_powers[0] = HD2, harmonic_powers[1] = HD3, etc.
    collided_harmonics : list
        List of harmonic numbers that collided with fundamental
        (e.g., [2, 4] means HD2 and HD4 aliased to fundamental bin)

    Notes
    -----
    Two-stage calculation:
    1. Build index map: scan all harmonics and collect valid bin indices (auto-deduplication via set)
    2. Unified summation: sum power from the deduplicated bin list

    Collision detection:
    - Fundamental collision: abs(h_bin - fundamental_bin) <= 2*side_bin
    - DC collision: h_bin <= side_bin
    - These harmonics are excluded from THD calculation

    Power extraction:
    - Individual harmonic powers are extracted before deduplication
    - THD uses deduplicated bin indices to prevent double-counting
    - Minimum power floor of 1e-15 prevents log(0) errors
    """
    # --- Stage 1: Build index map ---
    thd_bins_to_sum = set()  # Bin indices for THD calculation (auto-deduplicates)
    collided_harmonics = []
    harmonic_powers = np.full(max_harmonic - 1, 1e-15)  # Powers for harmonics 2 to max_harmonic

    for harmonic_index in range(max_harmonic - 1):
        harmonic_order = harmonic_index + 2  # Harmonic number (HD2 = 2, HD3 = 3, etc.)
        harmonic_bin_center = harmonic_bins[harmonic_index]  # Already int from _locate_harmonic_bins

        # Collision detection: fundamental or DC
        collision_threshold = 2 * side_bin
        is_fundamental_collision = abs(harmonic_bin_center - fundamental_bin) <= collision_threshold
        is_dc_collision = harmonic_bin_center <= side_bin

        if is_fundamental_collision:
            collided_harmonics.append(harmonic_order)
            continue
        if is_dc_collision:
            continue

        # Determine bin range for this harmonic
        harmonic_start_index = max(harmonic_bin_center - side_bin, 0)
        harmonic_end_index = min(harmonic_bin_center + side_bin + 1, len(power_spectrum))

        # Extract individual harmonic power (independent of deduplication)
        current_harmonic_power = np.sum(power_spectrum[harmonic_start_index:harmonic_end_index])
        harmonic_powers[harmonic_index] = max(current_harmonic_power, 1e-15)

        # Add bin indices to THD set (automatic deduplication for overlapping ranges)
        thd_bins_to_sum.update(range(harmonic_start_index, harmonic_end_index))

    # --- Stage 2: Unified summation ---
    if thd_bins_to_sum:
        # Use fancy indexing to sum power from deduplicated bins
        thd_power = np.sum(power_spectrum[list(thd_bins_to_sum)])
    else:
        thd_power = 1e-15

    # Apply minimum floor
    thd_power = max(thd_power, 1e-15)

    return thd_power, harmonic_powers, collided_harmonics


def _extract_highest_spur(
    spectrum_power: np.ndarray,
    side_bin: int,
    n_search_inband: int,
    sig_bin_start: int,
    sig_bin_end: int
) -> tuple[int, float, float]:
    """Extract the highest spur from in-band spectrum (excluding fundamental).

    Parameters
    ----------
    spectrum_power : np.ndarray
        Power spectrum array (half-sided)
    side_bin : int
        Side bins to include around spur center
    n_search_inband : int
        In-band search limit (for OSR > 1)
    sig_bin_start : int
        Pre-calculated start index of fundamental region (inclusive)
    sig_bin_end : int
        Pre-calculated end index of fundamental region (exclusive)

    Returns
    -------
    spur_bin_idx : int
        Bin index of the maximum spur
    spur_power : float
        Spur power (linear scale, summed over center ± side_bin)
    spur_db : float
        Spur power in dB (single bin, for plotting)
    """
    # Copy in-band spectrum for search
    spectrum_copy = spectrum_power[:n_search_inband].copy()

    # Exclude signal using pre-calculated range
    if sig_bin_start < sig_bin_end:
        spectrum_copy[sig_bin_start:sig_bin_end] = 0

    # Find maximum spur within in-band range
    spur_bin_idx = np.argmax(spectrum_copy)

    # Calculate spur's summed power (including side bins) for SFDR metric
    spur_start = max(spur_bin_idx - side_bin, 0)
    spur_end = min(spur_bin_idx + side_bin + 1, n_search_inband)
    spur_power = np.sum(spectrum_power[spur_start:spur_end])

    return spur_bin_idx, spur_power

def _calculate_harmonic_phases(
    spec_coherent_normalized: np.ndarray,
    harmonic_bins: np.ndarray
) -> tuple[float, float]:
    """Calculate HD2 and HD3 phases relative to fundamental.

    After phase alignment in coherent averaging, the fundamental is rotated to 0°
    and harmonics are rotated by h*φ₁. This function extracts the relative phases
    of HD2 and HD3 after alignment.

    Parameters
    ----------
    spec_coherent_normalized : np.ndarray
        Phase-aligned complex spectrum (normalized for amplitude)
    harmonic_bins : np.ndarray
        Array of harmonic bin positions from _locate_harmonic_bins

    Returns
    -------
    hd2_phase_deg : float
        HD2 phase in degrees (relative to fundamental after alignment)
    hd3_phase_deg : float
        HD3 phase in degrees (relative to fundamental after alignment)

    Notes
    -----
    After _align_spectrum_phase():
    - Fundamental is at 0°
    - HD2 phase = original_HD2_phase - 2*fundamental_phase (relative phase)
    - HD3 phase = original_HD3_phase - 3*fundamental_phase (relative phase)

    For memoryless nonlinearity:
    - HD2 should be at 0°
    - HD3 should be at 0° or 180°

    Deviations indicate memory effects in the system.
    """
    hd2_phase_deg = 0.0
    hd3_phase_deg = 0.0

    # Extract HD2 phase (harmonic 2, index 1)
    if len(harmonic_bins) > 1:
        hd2_bin = harmonic_bins[1]
        if hd2_bin < len(spec_coherent_normalized):
            hd2_phase_rad = np.angle(spec_coherent_normalized[hd2_bin])
            hd2_phase_deg = np.degrees(hd2_phase_rad)

    # Extract HD3 phase (harmonic 3, index 2)
    if len(harmonic_bins) > 2:
        hd3_bin = harmonic_bins[2]
        if hd3_bin < len(spec_coherent_normalized):
            hd3_phase_rad = np.angle(spec_coherent_normalized[hd3_bin])
            hd3_phase_deg = np.degrees(hd3_phase_rad)

    return hd2_phase_deg, hd3_phase_deg

