"""Noise power estimation for spectrum analysis (MATLAB plotspec-aligned)."""

import warnings

import numpy as np
from scipy.special import gammaincinv

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
    """In-band spectrum with MATLAB plotspec low bins and fundamental lobe cleared."""
    spec = spectrum_power[:n_inband].copy()
    spec[: min(side_bin, n_inband)] = 0.0
    lo = max(fundamental_bin - side_bin, 0)
    hi = min(fundamental_bin + side_bin + 1, n_inband)
    if lo < hi:
        spec[lo:hi] = 0.0
    return spec


def _noise_regions(
    n_inband: int,
    passband_start: int,
    fundamental_bin: int,
    harmonic_bins: np.ndarray,
    side_bin: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Boolean masks over the in-band bins for the noise estimates.

    passband: bins from ``passband_start`` (the cutoff) to the band edge.
    masked: the DC region and the fundamental lobe, where noise cannot be
        separated from DC leakage or from the signal.
    harmonic: harmonic lobes outside the masked bins, where noise cannot be
        separated from distortion. A harmonic whose center lies in the DC
        region is treated as a DC collision and gets no lobe, as in THD.
    """
    idx = np.arange(n_inband)
    passband = idx >= passband_start
    masked = np.zeros(n_inband, dtype=bool)
    masked[: min(side_bin, n_inband)] = True
    lo = max(fundamental_bin - side_bin, 0)
    hi = min(fundamental_bin + side_bin + 1, n_inband)
    if lo < hi:
        masked[lo:hi] = True
    harmonic = np.zeros(n_inband, dtype=bool)
    for h_bin in harmonic_bins:
        h_bin = int(h_bin)
        if h_bin <= side_bin:
            continue
        lo = max(h_bin - side_bin, 0)
        hi = min(h_bin + side_bin + 1, n_inband)
        if lo < hi:
            harmonic[lo:hi] = True
    harmonic &= ~masked
    return passband, masked, harmonic


def _median_to_mean_ratio(n_records: int, coherent_averaging: bool) -> float:
    """Median/mean ratio of a white-noise bin in an averaged power spectrum.

    One complex FFT bin of white Gaussian noise has exponentially distributed
    power. Power averaging of ``n_records`` captures takes the mean of that
    many independent exponentials, a Gamma(n_records) / n_records variable.
    Coherent averaging adds complex bins, which stay complex Gaussian, so the
    averaged bin power is exponential (median/mean = ln 2) for any number of
    captures.
    """
    n_exponentials = 1 if coherent_averaging else max(int(n_records), 1)
    return float(gammaincinv(n_exponentials, 0.5) / n_exponentials)


def _estimate_noise_power(
    spectrum_power: np.ndarray,
    nf_method: int,
    n_inband: int,
    M: int,
    bin_idx: int,
    harmonic_bins: np.ndarray,
    side_bin: int,
    *,
    coherent_averaging: bool = False,
    passband_start: int = 0,
    return_parts: bool = False,
) -> float | tuple[float, dict[str, float]]:
    """Estimate noise power (linear). Methods 0-3 match MATLAB plotspec NFMethod.

    0=auto, 1=median, 2=trimmed mean, 3=exclude harmonics; 4=legacy wide exclude.
    ``M`` is the number of captures averaged into ``spectrum_power`` and
    ``coherent_averaging`` states how they were averaged; together they set
    the median-to-mean ratio of the median method.

    Methods 0-3 estimate the noise in the passband, the in-band bins from
    ``passband_start`` (the cutoff bin) up. Bins below it are not noise.
    Inside the passband the DC region, the fundamental lobe and the harmonic
    lobes hold noise that cannot be separated from DC leakage, the signal or
    distortion:

    - 1 and 2 take the median or the trimmed mean of the passband bins outside
      the DC region and the fundamental lobe, and scale it to the whole
      passband. This assumes a flat floor and, for the median, a known bin
      power distribution.
    - 3 sums the passband bins outside all three regions (``exclude_direct``)
      and adds the harmonic lobes at the mean power of those bins
      (``exclude_extrapolated``). This assumes the noise under the harmonic
      lobes has that mean level. The estimate never exceeds the passband
      noise plus distortion outside the DC region and the fundamental lobe.
      When no bin is left to average, the noise cannot be separated from
      the signal and distortion, and the estimate is NaN (as for method 4).
    - 0 is the median of 1, 2 and the direct part of 3.
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

    if nf_method not in (0, 1, 2, 3):
        raise ValueError(f"nf_method must be 0-4, got {nf_method}")

    spec = _zeroed_inband_spectrum(spectrum_power, n_inband, bin_idx, side_bin)
    passband, masked, harmonic = _noise_regions(
        n_inband, passband_start, bin_idx, harmonic_bins, side_bin
    )
    n_passband = int(np.count_nonzero(passband))
    if n_passband == 0:
        warnings.warn(
            "No in-band bins remain above the cutoff frequency; SNR, "
            "noise_floor_dbfs, and nsd_dbfs_hz are undefined.",
            RuntimeWarning,
            stacklevel=2,
        )
        undefined = dict.fromkeys(
            ("median", "trimmed", "exclude", "exclude_direct", "exclude_extrapolated"), np.nan
        )
        return (np.nan, undefined) if return_parts else np.nan

    spec_passband = spec[passband]
    spec_for_floor = spec_passband[np.abs(spec_passband) > 1e-20]
    if spec_for_floor.size == 0:
        spec_for_floor = spec_passband

    def _median_noise() -> float:
        median_to_mean = _median_to_mean_ratio(M, coherent_averaging)
        return float(np.median(spec_for_floor) / median_to_mean * n_passband)

    def _trimmed_noise() -> float:
        spec_sorted = np.sort(spec_for_floor)
        start, end = _matlab_trimmed_indices(spec_for_floor.size)
        return float(np.mean(spec_sorted[start:end]) * n_passband)

    remaining = passband & ~masked & ~harmonic
    exclude_direct = float(np.sum(np.where(remaining, spec, 0.0)))
    n_remaining = int(np.count_nonzero(remaining))
    noise_and_distortion = float(np.sum(np.where(passband & ~masked, spec, 0.0)))
    if n_remaining == 0:
        if nf_method == 3:
            warnings.warn(
                "No passband bins remain outside the DC region, the fundamental lobe "
                "and the harmonic lobes; SNR, noise_floor_dbfs, and nsd_dbfs_hz are "
                "undefined for nf_method=3.",
                RuntimeWarning,
                stacklevel=2,
            )
        exclude = np.nan
    else:
        n_harmonic = int(np.count_nonzero(harmonic & passband))
        exclude = min(
            exclude_direct + exclude_direct / n_remaining * n_harmonic,
            noise_and_distortion,
        )

    parts = {
        "median": _median_noise(),
        "trimmed": _trimmed_noise(),
        "exclude": exclude,
        "exclude_direct": exclude_direct,
        "exclude_extrapolated": exclude - exclude_direct,
    }

    if nf_method == 0:
        vals = [parts["median"], parts["trimmed"], parts["exclude_direct"]]
        noise_power = float(np.median(vals))
    elif nf_method == 1:
        noise_power = parts["median"]
    elif nf_method == 2:
        noise_power = parts["trimmed"]
    else:
        noise_power = parts["exclude"]
        if not np.isfinite(noise_power):
            return (np.nan, parts) if return_parts else np.nan

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
