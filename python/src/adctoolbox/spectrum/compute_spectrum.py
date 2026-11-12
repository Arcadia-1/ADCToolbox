"""Calculate spectrum data for ADC analysis - unified calculation engine (plotspec-aligned)."""

import numpy as np

from adctoolbox.spectrum._bin_ranges import rfft_inband_bin_count
from adctoolbox.spectrum._prepare_fft_input import _prepare_fft_input
from adctoolbox.spectrum._locate_fundamental import _locate_fundamental
from adctoolbox.spectrum._harmonics import _locate_harmonic_bins
from adctoolbox.spectrum._harmonics import _calculate_harmonic_power
from adctoolbox.spectrum._harmonics import _extract_highest_spur
from adctoolbox.spectrum._spectrum_averaging import _power_average, _coherent_average
from adctoolbox.spectrum._window import (
    _calculate_power_correction,
    _create_window,
    _get_auto_side_bin_fallback,
    _get_default_side_bin,
)
from adctoolbox.spectrum._side_bin_auto import (
    _detect_side_bin_auto_with_reference,
    _find_hidden_components,
    _warn_hidden_components,
)
from adctoolbox.spectrum._estimate_noise_power import _estimate_noise_power


def compute_spectrum(
    data: np.ndarray,
    fs: float = 1.0,
    max_scale_range: float | list[float] | tuple[float | None | list[float]] = None,
    win_type: str = "hann",
    side_bin: int | None = None,
    osr: int = 1,
    max_harmonic: int = 5,
    nf_method: int = 0,
    assumed_sig_pwr_dbfs: float | None = None,
    coherent_averaging: bool = False,
    cutoff_freq: float = 0,
    verbose: int = 0,
) -> dict[str, np.ndarray | float | dict]:
    """Calculate spectrum metrics aligned with MATLAB plotspec.m.

    Parameters
    ----------
    data
        Input ADC data, shape (N,) or (M, N)
    fs
        Sampling frequency in Hz
    max_scale_range
        Full scale range for normalization
    win_type
        Window type: 'boxcar', 'hann', 'hamming', etc.
    side_bin
        Side bins around fundamental. None or "auto" triggers auto detection,
        which follows the ideal tone's window leakage down to the noise floor.
        With non-coherent sampling that window can span tens of bins; a
        UserWarning is raised when it also contains a spur or second tone
        whose power would be counted as signal. Numeric values are truncated
        to a non-negative integer.
    osr
        Oversampling ratio
    max_harmonic
        Maximum harmonic order for THD (5 => H2..H5)
    nf_method
        Noise estimate behind ``snr_dbc``, ``noise_floor_dbfs`` and
        ``nsd_dbfs_hz``: 0=auto (default, the median of 1, 2 and the direct
        part of 3), 1=median, 2=trimmed mean, 3=exclude harmonics,
        4=legacy wide harmonic exclusion.

        1 and 2 scale a robust per-bin level to the whole passband; they
        assume a flat floor of random noise and ignore spurs. 3 sums all
        passband bins outside the DC region, the fundamental lobe and the
        harmonic lobes, so spurs count as noise, and fills the harmonic lobes
        at the mean level of those bins; the fill is reported separately in
        ``plot_data["noise_parts"]["exclude_extrapolated"]``. If the lobes
        leave no such bin, method 3 is NaN.
    assumed_sig_pwr_dbfs
        Override signal power in dBFS
    coherent_averaging
        Use coherent FFT averaging when True
    cutoff_freq
        High-pass cutoff in Hz. Bins below ``ceil(cutoff_freq / fs * N)`` are
        left out of SNDR, ENOB, SNR, the noise floor and NSD, and NSD divides
        the noise by the passband width. If no in-band bin remains, those
        metrics are NaN.
    verbose
        Verbosity level
    """
    if isinstance(side_bin, str):
        if side_bin != "auto":
            raise TypeError(
                f"side_bin must be an integer, None or 'auto'; got {side_bin!r}"
            )
        side_bin = None

    data_normalized = _prepare_fft_input(data, max_scale_range)
    M, N = data_normalized.shape
    n_inband = rfft_inband_bin_count(N, osr)

    window_vector, window_gain, equiv_noise_bw_factor = _create_window(win_type, N)
    data_windowed = data_normalized * window_vector

    if coherent_averaging:
        power_spectrum, complex_spectrum, n_averaged = _coherent_average(
            data_windowed, osr, lobe_half_width=_get_default_side_bin(win_type),
        )
    else:
        # Leave out all-zero records, as the coherent path and MATLAB plotspec
        # do. Averaging them in would scale every power down by the fraction
        # of empty records.
        nonzero_runs = np.max(np.abs(data_windowed), axis=1) >= 1e-10
        if np.any(nonzero_runs):
            data_windowed = data_windowed[nonzero_runs]
        power_spectrum, complex_spectrum = _power_average(data_windowed)
        n_averaged = data_windowed.shape[0]

    power_correction = _calculate_power_correction(window_gain, equiv_noise_bw_factor)
    power_spectrum = power_spectrum * power_correction
    if complex_spectrum is not None:
        complex_spectrum = complex_spectrum * np.sqrt(power_correction)

    power_spectrum_db = 10 * np.log10(power_spectrum + 1e-20)
    freq = np.arange(len(power_spectrum)) * (fs / N)

    passband_start = 0
    if cutoff_freq > 0:
        cutoff_bin = min(int(np.ceil(cutoff_freq / fs * N)), len(power_spectrum))
        passband_start = cutoff_bin
        if cutoff_freq >= (fs / 2):
            import warnings

            warnings.warn(f"cutoff_freq [{cutoff_freq} Hz] exceeds Nyquist.")
        if verbose >= 2:
            print(f"Applying cutoff frequency: [{cutoff_freq} Hz], cut bin [0:{cutoff_bin}]")
        power_spectrum[:cutoff_bin] = 1e-20
        power_spectrum_db = 10 * np.log10(power_spectrum)
        if complex_spectrum is not None:
            complex_spectrum[:cutoff_bin] = 0j

    fundamental_bin, fundamental_bin_fractional = _locate_fundamental(power_spectrum, n_inband)

    if side_bin is None:
        minimum_side_bin = _get_default_side_bin(win_type)
        side_bin, ideal_spec, noise_floor_per_bin = _detect_side_bin_auto_with_reference(
            power_spectrum,
            fundamental_bin,
            fundamental_bin_fractional,
            n_inband,
            N,
            window_vector,
            power_correction,
            fallback_side_bin=_get_auto_side_bin_fallback(win_type),
            minimum_side_bin=minimum_side_bin,
        )
        hidden_bins = _find_hidden_components(
            power_spectrum,
            fundamental_bin,
            side_bin,
            ideal_spec,
            noise_floor_per_bin,
            core_half_width=minimum_side_bin + 1,
            n_inband=n_inband,
        )
        _warn_hidden_components(
            "compute_spectrum", power_spectrum, fundamental_bin, side_bin,
            hidden_bins, stacklevel=3,
        )
    else:
        side_bin = int(max(side_bin, 0))

    sig_bin_start = max(fundamental_bin - side_bin, 0)
    sig_bin_end = min(fundamental_bin + side_bin + 1, n_inband)

    sig_linear = float(np.sum(power_spectrum[sig_bin_start:sig_bin_end]))
    measured_sig_linear = sig_linear
    sig_pwr_linear = sig_linear
    sig_pwr_dbfs = 10 * np.log10(max(sig_pwr_linear, 1e-30))

    if assumed_sig_pwr_dbfs is not None and not np.isnan(assumed_sig_pwr_dbfs):
        sig_pwr_linear = 10 ** (assumed_sig_pwr_dbfs / 10)
        sig_pwr_dbfs = assumed_sig_pwr_dbfs
        sig_linear = sig_pwr_linear

    harmonic_bins = _locate_harmonic_bins(fundamental_bin_fractional, max_harmonic, N)

    thd_power, harmonic_powers, collided_harmonics = _calculate_harmonic_power(
        power_spectrum=power_spectrum,
        fundamental_bin=fundamental_bin,
        harmonic_bins=harmonic_bins,
        side_bin=side_bin,
        max_harmonic=max_harmonic,
        n_inband=n_inband,
    )
    spur_bin_idx, spur_power = _extract_highest_spur(
        power_spectrum,
        side_bin,
        n_inband,
        sig_bin_start,
        sig_bin_end,
        fundamental_bin,
        measured_sig_linear,
    )
    if sig_linear > 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            harmonics_dbc = 10 * np.log10(harmonic_powers / sig_linear)
            thd_dbc = 10 * np.log10(thd_power / sig_linear)
            sfdr_dbc = np.inf if spur_power <= 0 else 10 * np.log10(sig_linear / spur_power)
    else:
        harmonics_dbc = np.full_like(harmonic_powers, np.nan, dtype=float)
        thd_dbc = np.nan
        sfdr_dbc = np.nan

    spec_sndr = power_spectrum.copy()
    spec_sndr[: min(side_bin, len(spec_sndr))] = 0.0
    spec_sndr[:passband_start] = 0.0
    lo = max(fundamental_bin - side_bin, 0)
    hi = min(fundamental_bin + side_bin + 1, n_inband)
    if lo < hi:
        spec_sndr[lo:hi] = 0.0
    noi_sndr = float(np.sum(spec_sndr[:n_inband]))
    if passband_start < n_inband:
        sndr_dbc = 10 * np.log10(sig_linear / (noi_sndr + 1e-20))
    else:
        sndr_dbc = np.nan
    enob = (sndr_dbc - 1.76) / 6.02

    noise_result = _estimate_noise_power(
        spectrum_power=power_spectrum,
        nf_method=nf_method,
        n_inband=n_inband,
        M=n_averaged,
        bin_idx=fundamental_bin,
        harmonic_bins=harmonic_bins,
        side_bin=side_bin,
        coherent_averaging=coherent_averaging,
        passband_start=passband_start,
        return_parts=True,
    )
    noise_power, noise_parts = noise_result

    if np.isfinite(noise_power):
        snr_dbc = 10 * np.log10(sig_linear / noise_power)
        noise_floor_dbfs = sig_pwr_dbfs - snr_dbc
    else:
        snr_dbc = np.nan
        noise_floor_dbfs = np.nan

    # Width of the passband the noise is integrated over. The full band spans
    # N/(2*osr) bins with the DC bin counting half, so a cutoff at bin
    # passband_start removes passband_start - 1/2 bins.
    removed_bins = max(passband_start - 0.5, 0)
    noise_bandwidth_bins = N / (2 * osr) - removed_bins
    noise_bandwidth_hz = fs / (2 * osr) - removed_bins * fs / N
    if noise_bandwidth_bins <= 0:
        noise_bandwidth_bins = np.nan
        noise_bandwidth_hz = np.nan

    nsd_dbfs_hz = (
        noise_floor_dbfs - 10 * np.log10(noise_bandwidth_hz)
        if np.isfinite(noise_floor_dbfs)
        else np.nan
    )

    # Plot array: raw 10*log10(spec) like plotspec.m (no display offset to 0 dBFS)
    v_offset = sig_pwr_dbfs - power_spectrum_db[fundamental_bin]
    power_spectrum_db_plot = power_spectrum_db

    return {
        "N": N,
        "M": n_averaged,
        "fs": fs,
        "osr": osr,
        "metrics": {
            "enob": enob,
            "sndr_dbc": sndr_dbc,
            "sfdr_dbc": sfdr_dbc,
            "snr_dbc": snr_dbc,
            "sig_pwr_dbfs": sig_pwr_dbfs,
            "noise_floor_dbfs": noise_floor_dbfs,
            "nsd_dbfs_hz": nsd_dbfs_hz,
            "thd_dbc": thd_dbc,
            "harmonics_dbc": harmonics_dbc,
        },
        "plot_data": {
            "freq": freq,
            "power_spectrum_db_plot": power_spectrum_db_plot,
            "complex_spectrum": complex_spectrum,
            "fundamental_bin": fundamental_bin,
            "fundamental_bin_fractional": fundamental_bin_fractional,
            "sig_bin_start": sig_bin_start,
            "sig_bin_end": sig_bin_end,
            "side_bin": side_bin,
            "spur_bin_idx": spur_bin_idx,
            "is_coherent": coherent_averaging,
            "harmonic_bins": harmonic_bins,
            "collided_harmonics": collided_harmonics,
            "v_offset": v_offset,
            "noise_parts": noise_parts,
            "noise_bandwidth_bins": noise_bandwidth_bins,
        },
    }
