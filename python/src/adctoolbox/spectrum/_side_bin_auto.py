"""Auto-detect coherent main-lobe width (MATLAB plotspec sideBin='auto')."""

import warnings

import numpy as np

from adctoolbox.spectrum._spectrum_averaging import _power_average

# A bin inside the auto signal window is reported as a separate component when
# its measured power exceeds the ideal-tone leakage plus the noise floor by this
# factor (10 dB).
_HIDDEN_COMPONENT_MARGIN = 10.0


def _detect_side_bin_auto(
    power_spectrum: np.ndarray,
    fundamental_bin: int,
    fundamental_bin_fractional: float,
    n_inband: int,
    n_fft: int,
    window_vector: np.ndarray,
    power_correction: float,
    fallback_side_bin: int = 1,
    minimum_side_bin: int = 0,
) -> int:
    """Match ``plotspec.m`` sideBin auto: ideal leakage vs median noise floor."""
    side_bin, _, _ = _detect_side_bin_auto_with_reference(
        power_spectrum,
        fundamental_bin,
        fundamental_bin_fractional,
        n_inband,
        n_fft,
        window_vector,
        power_correction,
        fallback_side_bin=fallback_side_bin,
        minimum_side_bin=minimum_side_bin,
    )
    return side_bin


def _detect_side_bin_auto_with_reference(
    power_spectrum: np.ndarray,
    fundamental_bin: int,
    fundamental_bin_fractional: float,
    n_inband: int,
    n_fft: int,
    window_vector: np.ndarray,
    power_correction: float,
    fallback_side_bin: int = 1,
    minimum_side_bin: int = 0,
) -> tuple[int, np.ndarray, float]:
    """Return the auto side_bin plus the ideal-tone leakage and noise floor it used."""
    t = np.arange(n_fft, dtype=float)
    ideal = np.sin(2 * np.pi * fundamental_bin_fractional * t / n_fft)
    ideal_spec, _ = _power_average((ideal * window_vector).reshape(1, -1))
    ideal_spec *= power_correction

    peak = max(power_spectrum[fundamental_bin], 1e-30)
    ideal_spec *= peak / max(ideal_spec[fundamental_bin], 1e-30)

    inband = power_spectrum[:n_inband]
    positive = inband[inband > 1e-20]
    if len(positive) >= 3:
        noise_floor_per_bin = float(np.median(positive))
    else:
        noise_floor_per_bin = float(np.median(inband))

    numerical_floor = peak * 1e-24
    noise_floor_per_bin = max(noise_floor_per_bin, numerical_floor)

    # The crossing search needs a bin on both sides, so it stops at the
    # nearer of DC and the band edge. The window's coherent main-lobe width is
    # a floor that this distance must not reduce: a lobe cut short there would
    # have its remaining bins counted as distortion. The signal mask itself is
    # clipped to the band by the caller.
    max_sidebin = min(fundamental_bin, n_inband - 1 - fundamental_bin)
    minimum_side_bin = int(max(minimum_side_bin, 0))
    for sb in range(1, max_sidebin + 1):
        left = fundamental_bin - sb
        right = fundamental_bin + sb
        left_below = left >= 0 and ideal_spec[left] <= noise_floor_per_bin
        right_below = right < n_inband and ideal_spec[right] <= noise_floor_per_bin
        if left_below and right_below:
            return int(max(sb - 1, minimum_side_bin)), ideal_spec, noise_floor_per_bin

    fallback_side_bin = int(max(fallback_side_bin, 0))
    side_bin = int(max(min(fallback_side_bin, max_sidebin), minimum_side_bin))
    return side_bin, ideal_spec, noise_floor_per_bin


def _find_hidden_components(
    power_spectrum: np.ndarray,
    fundamental_bin: int,
    side_bin: int,
    ideal_spec: np.ndarray,
    noise_floor_per_bin: float,
    core_half_width: int,
    n_inband: int,
    margin: float = _HIDDEN_COMPONENT_MARGIN,
) -> np.ndarray:
    """Bins inside the signal window that the ideal tone's leakage cannot explain.

    The auto side_bin follows the ideal tone's leakage down to the noise floor.
    With non-coherent sampling that window can span tens of bins, and a spur or
    second tone inside it is summed into the signal power instead of being
    reported as a spur. Bins within ``core_half_width`` of the fundamental are
    skipped because the leakage template is least accurate on the main lobe,
    and so are the DC bins, where input mean removal leaves a windowed step
    that the ideal tone does not model. The measured-to-expected ratio is
    normalized by its median over the window, so a template that is uniformly
    off (for example a rectangular window with a coarse fractional-bin
    estimate) is not mistaken for a separate component.
    """
    start = max(fundamental_bin - side_bin, 0)
    end = min(fundamental_bin + side_bin + 1, n_inband)
    bins = np.arange(start, end)
    bins = bins[(np.abs(bins - fundamental_bin) > core_half_width) & (bins > core_half_width)]
    if bins.size == 0:
        return bins
    expected = ideal_spec[bins] + noise_floor_per_bin
    ratio = power_spectrum[bins] / expected
    template_scale = max(1.0, float(np.median(ratio)))
    return bins[ratio > margin * template_scale]


def _warn_hidden_components(
    caller: str,
    power_spectrum: np.ndarray,
    fundamental_bin: int,
    side_bin: int,
    hidden_bins: np.ndarray,
    stacklevel: int,
) -> None:
    """Warn that the auto signal window absorbs power the fundamental cannot explain."""
    if hidden_bins.size == 0:
        return
    strongest = int(hidden_bins[np.argmax(power_spectrum[hidden_bins])])
    peak = max(float(power_spectrum[fundamental_bin]), 1e-30)
    level_dbc = 10 * np.log10(max(float(power_spectrum[strongest]), 1e-30) / peak)
    warnings.warn(
        f"{caller}: the auto side_bin ({side_bin}) spans bins "
        f"{max(fundamental_bin - side_bin, 0)}-{fundamental_bin + side_bin} around the "
        f"fundamental at bin {fundamental_bin}, and {hidden_bins.size} bin(s) in that "
        f"window exceed the fundamental's leakage by more than 10 dB (strongest: "
        f"bin {strongest}, {level_dbc:.1f} dB relative to the fundamental peak bin). "
        "That power is counted as signal, so SFDR and SNDR can read optimistic. "
        "Use coherent sampling, a lower-leakage window such as "
        "win_type='blackmanharris', or pass an explicit side_bin.",
        UserWarning,
        stacklevel=stacklevel,
    )
