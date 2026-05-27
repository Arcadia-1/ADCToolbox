"""Noise power estimation for spectrum analysis (MATLAB plotspec-aligned)."""

import warnings

import numpy as np

from adctoolbox.spectrum._exclude_bins import _exclude_bins_from_spectrum


def _matlab_trimmed_indices(n_inband: int) -> tuple[int, int]:
    """plotspec.m: spec_sort(max(1,floor(5%)):max(1,floor(95%))) (1-based inclusive)."""
    lo_1based = max(1, int(np.floor(n_inband * 0.05)))
    hi_1based = max(1, int(np.floor(n_inband * 0.95)))
    start = lo_1based - 1
    end_exclusive = min(max(start + 1, hi_1based), n_inband)
    return start, end_exclusive


def _zeroed_inband_spectrum(
    spectrum_power: np.ndarray,
    n_inband: int,
    fundamental_bin: int,
    side_bin: int,
) -> np.ndarray:
    """In-band spectrum with DC..sideBin and fundamental main lobe cleared (plotspec.m)."""
    spec = spectrum_power[:n_inband].copy()
    spec[: min(side_bin + 1, n_inband)] = 0.0
    lo = max(fundamental_bin - side_bin, 0)
    hi = min(fundamental_bin + side_bin + 1, n_inband)
    if lo < hi:
        spec[lo:hi] = 0.0
    return spec


def _estimate_noise_power(
    spectrum_power: np.ndarray,
    nf_method: int,
    n_inband: int,
    M: int,
    bin_idx: int,
    harmonic_bins: np.ndarray,
    side_bin: int,
    *,
    return_parts: bool = False,
) -> float | tuple[float, dict[str, float]]:
    """Estimate noise power (linear). Methods 0-3 match MATLAB plotspec NFMethod.

    0=auto, 1=median, 2=trimmed mean, 3=exclude harmonics; 4=legacy wide exclude.
    """
    if nf_method == 4:
        noise_spectrum = _exclude_bins_from_spectrum(
            spectrum_power, bin_idx, harmonic_bins, side_bin, n_inband
        )
        candidate_bins = set(range(n_inband))
        candidate_bins.difference_update(range(0, min(side_bin + 1, n_inband)))
        if 1 <= bin_idx < n_inband:
            candidate_bins.difference_update(
                range(max(bin_idx - side_bin, 0), min(bin_idx + side_bin + 1, n_inband))
            )
        for h_bin in harmonic_bins:
            if 1 <= h_bin < n_inband:
                candidate_bins.difference_update(
                    range(max(h_bin - side_bin, 0), min(h_bin + side_bin + 1, n_inband))
                )

        if not candidate_bins:
            warnings.warn(
                "No noise bins remain after excluding DC, fundamental, and harmonics; "
                "SNR, noise_floor_dbfs, and nsd_dbfs_hz are undefined.",
                RuntimeWarning,
                stacklevel=2,
            )
            if return_parts:
                return np.nan, {}
            return np.nan

        noise_power = float(np.sum(noise_spectrum))
        noise_power = max(noise_power, 1e-15) if np.isfinite(noise_power) else np.nan
        if return_parts:
            return noise_power, {"legacy_wide_exclude": noise_power}
        return noise_power

    spec = _zeroed_inband_spectrum(spectrum_power, n_inband, bin_idx, side_bin)
    spec_for_floor = spec[np.abs(spec) > 1e-20]
    if spec_for_floor.size == 0:
        spec_for_floor = spec

    def _median_noise() -> float:
        mn = 0.72 if M == 1 else (1 - 2 / (9 * M)) ** 3
        return float(np.median(spec_for_floor) / mn * n_inband)

    def _trimmed_noise() -> float:
        spec_sorted = np.sort(spec_for_floor)
        start, end = _matlab_trimmed_indices(spec_for_floor.size)
        return float(np.mean(spec_sorted[start:end]) * n_inband)

    def _exclude_noise() -> float:
        spec_noise = spec.copy()
        for h_bin in harmonic_bins:
            if 0 <= h_bin < n_inband:
                spec_noise[h_bin] = 0.0
        return float(np.sum(spec_noise))

    parts = {
        "median": _median_noise(),
        "trimmed": _trimmed_noise(),
        "exclude": _exclude_noise(),
    }

    if nf_method == 0:
        vals = [parts["median"], parts["trimmed"], parts["exclude"]]
        min_val = min(vals)
        if min_val > 0 and max(vals) / min_val > 1.25:
            warnings.warn(
                f"Noise floor estimation methods differ by "
                f"{10 * np.log10(max(vals) / min_val):.1f} dB.",
                RuntimeWarning,
                stacklevel=2,
            )
        noise_power = float(np.median(vals))
    elif nf_method == 1:
        noise_power = parts["median"]
    elif nf_method == 2:
        noise_power = parts["trimmed"]
    elif nf_method == 3:
        noise_power = parts["exclude"]
    else:
        raise ValueError(f"nf_method must be 0-4, got {nf_method}")

    noise_power = max(float(noise_power), 1e-15)
    if return_parts:
        return noise_power, parts
    return noise_power


def _calculate_noise_metrics(
    signal_power: float,
    noise_power: float,
    sig_pwr_dbfs: float,
    fs: float,
    osr: int,
    enbw: float,
) -> tuple[float, float, float]:
    """Calculate SNR, noise floor, NSD (NSD omits extra ENBW term; matches plotspec.m)."""
    snr_db = 10 * np.log10(signal_power / noise_power)
    noise_floor_dbfs = sig_pwr_dbfs - snr_db
    nsd_dbfs_hz = noise_floor_dbfs - 10 * np.log10(fs / (2 * osr))
    return snr_db, noise_floor_dbfs, nsd_dbfs_hz
