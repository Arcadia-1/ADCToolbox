"""Spectrum averaging methods: power and coherent averaging."""

import numpy as np
from adctoolbox.spectrum._align_spectrum_phase import _align_spectrum_phase
from adctoolbox.spectrum._locate_fundamental import _locate_fundamental


def _power_average(
    data_processed: np.ndarray
) -> tuple[np.ndarray, np.ndarray | None]:
    """Compute power-averaged spectrum with vectorized FFT.

    Computes FFT for all runs simultaneously and averages power spectra.
    Returns raw FFT power without correction (correction applied in compute_spectrum).

    Parameters
    ----------
    data_processed
        Windowed data, shape (M runs, N samples)

    Returns
    -------
    spectrum_power
        Raw power spectrum, shape (N//2 + 1,) - Includes DC and Nyquist
        Normalized by N² and with DC/Nyquist boundary correction applied
    None
        Placeholder for coherent spectrum (not used in power averaging)
    """
    M, N = data_processed.shape

    fft_matrix = np.fft.rfft(data_processed, axis=1)
    spectrum_power = np.mean(np.abs(fft_matrix)**2, axis=0)
    spectrum_power = spectrum_power / (N**2)

    # DC and Nyquist boundary correction (single-sided spectrum)
    spectrum_power[0] /= 2.0
    if N % 2 == 0:
        spectrum_power[-1] /= 2.0

    return spectrum_power, None


def _coherent_average(
    data_processed: np.ndarray,
    osr: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute coherent-averaged spectrum with phase alignment.

    For each run: finds fundamental, performs parabolic interpolation, aligns
    phase to first run, then coherently averages. Skips runs with zero data.
    Returns raw FFT results without correction (correction applied in compute_spectrum).

    Parameters
    ----------
    data_processed
        Windowed data, shape (M runs, N samples)
    osr
        Oversampling ratio for fundamental search range

    Returns
    -------
    spectrum_power
        Raw power spectrum, shape (N//2 + 1,)
        Normalized by N and n_valid_runs, with DC/Nyquist boundary correction
    spectrum_complex
        Raw complex spectrum for polar plots, shape (N//2 + 1,)
        Normalized by N and n_valid_runs, with DC/Nyquist boundary correction
    """
    M, N = data_processed.shape
    n_output = N // 2 + 1
    spec_coherent_sum = np.zeros(N, dtype=complex)
    n_valid_runs = 0
    original_fundamental_phase = None  # Store original phase before alignment

    for run_idx in range(M):
        run_data = data_processed[run_idx, :N]
        if np.max(np.abs(run_data)) < 1e-10:
            continue

        fft_data = np.fft.fft(run_data)

        # Get power spectrum for fundamental search
        fft_power = np.abs(fft_data[:N//2+1])**2
        n_inband = N // 2 // osr

        fundamental_bin, fundamental_bin_frantional = _locate_fundamental(fft_power, n_inband)

        # Guard against DC bin (MATLAB plotspec.m:286-289)
        if fundamental_bin <= 0:
            continue

        # Store original fundamental phase from first valid run
        if original_fundamental_phase is None:
            original_fundamental_phase = np.angle(fft_data[fundamental_bin])

        # Phase alignment (MATLAB plotspec.m:292-322)
        fft_aligned = _align_spectrum_phase(fft_data, fundamental_bin, fundamental_bin_frantional, N)
        spec_coherent_sum += fft_aligned
        n_valid_runs += 1

    # Normalization factor
    n_den = n_valid_runs if n_valid_runs > 0 else 1

    # Compute normalized complex voltage spectrum (raw, no power correction yet)
    spectrum_complex = spec_coherent_sum[:n_output] / (N * n_den)

    # Compute power spectrum (|V|^2)
    spectrum_power = np.abs(spectrum_complex)**2

    # DC and Nyquist boundary correction (single-sided spectrum)
    spectrum_power[0] /= 2.0
    spectrum_complex[0] /= np.sqrt(2.0)  # Maintain P = |V|^2 relationship

    if N % 2 == 0:
        spectrum_power[-1] /= 2.0
        spectrum_complex[-1] /= np.sqrt(2.0)

    return spectrum_power, spectrum_complex
