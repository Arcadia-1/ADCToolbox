"""
Spectrum Phase Analyzer

This module provides tools for analyzing the phase spectrum of ADC data,
including coherent phase alignment and polar plotting.

Structure:
- FFT mode: Uses calculate_coherent_spectrum + plot_polar_phase
- LMS mode: Direct least-squares harmonic fitting (unchanged)

MATLAB counterpart: plotphase.m
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict
from adctoolbox.common.fit_sine import fit_sine
from adctoolbox.common.calculate_aliased_freq import calculate_aliased_freq

# Import the new modular components
from .calculate_coherent_spectrum import calculate_coherent_spectrum, prepare_polar_plot_data
from .plot_polar_phase import plot_polar_phase

# Import shared helpers (for potential future use in calculate_spectrum_metrics)
from ._prepare_fft_input import _prepare_fft_input
from ._find_fundamental import _find_fundamental


def alias(bin_idx: int, n: int) -> int:
    """
    Handle bin aliasing for FFT analysis.

    Maps any bin index to the valid range [0, n/2] for real signals.

    Args:
        bin_idx: Bin index (can be negative or > n)
        n: Total number of FFT bins

    Returns:
        Aliased bin index in range [0, n/2]
    """
    # First wrap to [0, n) range
    bin_idx = bin_idx % n

    # For real signals, bins > n/2 are mirrored
    if bin_idx > n // 2:
        bin_idx = n - bin_idx

    return bin_idx


def analyze_phase_spectrum(
    data: np.ndarray,
    n_fft: Optional[int] = None,
    harmonic: int = 5,
    osr: int = 1,
    mode: str = 'FFT',
    fs: float = 1.0,
    cutoff_freq: float = 0,
    max_code: Optional[float] = None,
    win_type: str = 'boxcar',
    save_path: Optional[str] = None,
    show_plot: bool = False
) -> dict:
    """
    Spectrum phase analysis with polar plot.

    Supports two modes:
    - FFT: Full spectrum visualization with coherent phase alignment
    - LMS: Least-squares harmonic fitting with numerical outputs

    Args:
        data: ADC output data, shape (M, N) for M runs or (N,) for single run
        n_fft: FFT size (default: length of data)
        harmonic: Number of harmonics to mark (default 5)
        osr: Oversampling ratio (default 1)
        mode: 'FFT' for full spectrum or 'LMS' for harmonic fitting (default 'FFT')
        fs: Sampling frequency in Hz (default 1.0)
        cutoff_freq: High-pass cutoff frequency in Hz (default 0)
        max_code: Maximum code level for normalization (default: auto-detect)
        win_type: Window function type (default: 'boxcar')
        save_path: Path to save figure (optional)
        show_plot: Whether to display plot (default False)

    Returns:
        dict with keys:
            FFT mode:
            - spec: Complex spectrum (phase-aligned)
            - bin: Fundamental frequency bin
            - harmonics: List of harmonic info
            - coherent_result: Full result from calculate_coherent_spectrum

            LMS mode:
            - harm_phase: Harmonic phases array
            - harm_mag: Harmonic magnitudes array
            - freq: Normalized fundamental frequency
            - noise_dB: Noise floor in dB
    """
    data = np.asarray(data)

    # Handle 1D or 2D input
    if data.ndim == 1:
        data = data.reshape(1, -1)

    # Validate mode parameter
    mode = mode.upper()
    if mode not in ['FFT', 'LMS']:
        raise ValueError(f"Mode must be 'FFT' or 'LMS', got '{mode}'")

    # ========== FFT Mode: Use new modular structure ==========
    if mode == 'FFT':
        # Calculate coherent spectrum
        coherent_result = calculate_coherent_spectrum(
            data=data,
            max_code=max_code,
            osr=osr,
            cutoff_freq=cutoff_freq,
            fs=fs,
            win_type=win_type,
            n_fft=n_fft
        )

        # Prepare plot data
        plot_data = prepare_polar_plot_data(coherent_result, harmonic=harmonic)

        # Create polar plot using the new plot_polar_phase function
        if save_path or show_plot:
            # Create figure if saving or showing
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

            # Plot using the pure visualization function
            plot_polar_phase(plot_data, harmonic=harmonic, ax=ax)

            # Add title
            ax.set_title('Phase Spectrum Analysis (FFT Mode)', pad=20, fontsize=12, fontweight='bold')

            # Save/show if requested
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"[analyze_phase_spectrum] Figure saved to: {save_path}")

            if show_plot:
                plt.show()
            else:
                plt.close(fig)

        # Prepare harmonic info for backward compatibility
        harmonics_info = []
        for h in range(1, min(harmonic + 1, len(plot_data['harmonic_bins']) + 1)):
            if h-1 < len(plot_data['harmonic_bins']):
                bin_h = int(plot_data['harmonic_bins'][h-1])
                if bin_h < len(coherent_result['spec_mag_db']):
                    harmonics_info.append({
                        'harmonic': h,
                        'bin': bin_h,
                        'magnitude': coherent_result['spec_mag_db'][bin_h],
                        'phase': coherent_result['phase'][bin_h]
                    })

        # Return results matching previous format
        return {
            'spec': coherent_result['complex_spec_coherent'],  # Phase-weighted spectrum
            'bin': coherent_result['bin_idx'],  # Fundamental bin index
            'harmonics': harmonics_info,
            'coherent_result': coherent_result,  # Full result for advanced users
            # LMS mode outputs (empty for FFT mode)
            'harm_phase': np.array([]),
            'harm_mag': np.array([]),
            'freq': np.nan,
            'noise_dB': np.nan
        }

    # ========== LMS Mode: Least Squares Harmonic Fitting (unchanged) ==========
    else:  # mode == 'LMS'
        n_runs, n = data.shape
        if n_fft is None:
            n_fft = n

        # Average all runs
        sig_avg = np.mean(data, axis=0)

        # Normalize signal (match MATLAB: subtract mean, divide by peak-to-peak)
        if max_code is None:
            max_code = np.max(sig_avg) - np.min(sig_avg)
        sig_avg = sig_avg - np.mean(sig_avg)
        sig_avg = sig_avg / max_code

        # Find fundamental frequency using sinfit (match MATLAB line 251)
        _, freq_norm, _, _, _ = fit_sine(sig_avg)
        freq = freq_norm  # Already normalized frequency

        # Build sine/cosine basis for harmonics
        t = np.arange(n_fft)
        SI = np.zeros((n_fft, harmonic))
        SQ = np.zeros((n_fft, harmonic))
        for ii in range(harmonic):
            SI[:, ii] = np.cos(t * freq * (ii + 1) * 2 * np.pi)
            SQ[:, ii] = np.sin(t * freq * (ii + 1) * 2 * np.pi)

        # Least squares fit
        A = np.hstack([SI, SQ])
        W = np.linalg.lstsq(A, sig_avg, rcond=None)[0]

        # Reconstruct signal with all harmonics
        signal_all = A @ W

        # Calculate residual (noise) - match MATLAB line 270
        residual = sig_avg - signal_all
        noise_power = np.sqrt(np.mean(residual**2)) * 2 * np.sqrt(2)  # rms(residual)*2*sqrt(2)
        noise_dB = 20 * np.log10(noise_power)

        # Extract magnitude and phase for each harmonic
        harm_mag = np.zeros(harmonic)
        harm_phase = np.zeros(harmonic)
        for ii in range(harmonic):
            I_weight = W[ii]
            Q_weight = W[ii + harmonic]
            # Match MATLAB line 279: multiply by *2
            harm_mag[ii] = np.sqrt(I_weight**2 + Q_weight**2) * 2
            harm_phase[ii] = np.arctan2(Q_weight, I_weight)

        # Phase rotation: make phases relative to fundamental
        fundamental_phase = harm_phase[0]
        for ii in range(harmonic):
            harm_phase[ii] = harm_phase[ii] - fundamental_phase * (ii + 1)

        # Wrap phases to [-pi, pi]
        harm_phase = np.mod(harm_phase + np.pi, 2 * np.pi) - np.pi

        # Convert to dB for plotting
        harm_dB = 20 * np.log10(harm_mag)

        # Set maxR and minR for plot scaling
        maxR = np.ceil(np.max(harm_dB) / 10) * 10
        minR = min(np.min(harm_dB), noise_dB) - 10
        minR = max(minR, -200)  # Don't go below -200 dB
        minR = np.floor(minR / 10) * 10

        # Create polar plot for LMS mode if requested
        if save_path or show_plot:
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

            # Plot harmonics as radial lines
            angles = harm_phase
            radii = harm_dB - minR  # Offset by minR for display

            for ii in range(harmonic):
                color = 'r' if ii == 0 else 'b'
                linewidth = 2 if ii == 0 else 2.5
                ax.plot([0, angles[ii]], [0, radii[ii]], color=color, linewidth=linewidth)
                if ii > 0:
                    ax.annotate(str(ii + 1), (angles[ii], radii[ii] + 2),
                               fontsize=10, ha='center')

            # Mark fundamental with circle
            ax.plot(angles[0], radii[0], 'ro', markersize=10)

            # Plot noise floor as dashed circle
            noise_radius = noise_dB - minR
            if noise_radius > 0:
                theta_circle = np.linspace(0, 2 * np.pi, 100)
                ax.plot(theta_circle, np.ones_like(theta_circle) * noise_radius,
                       'g--', linewidth=1.5, label=f'Noise: {noise_dB:.1f} dB')
                ax.legend(loc='upper right')

            # Configure polar plot
            ax.set_theta_direction(-1)  # Clockwise
            ax.set_theta_zero_location('N')  # 0 degrees at top

            # Custom tick labels showing dB values
            tick_spacing = 10
            tick = np.arange(0, -minR + 1, tick_spacing)
            tickl = np.arange(minR, 1, tick_spacing)

            # Ensure same length
            min_len = min(len(tick), len(tickl))
            tick = tick[:min_len]
            tickl = tickl[:min_len]

            ax.set_rticks(tick)
            ax.set_yticklabels([f'{int(t):.0f}' for t in tickl])
            ax.set_rlim(0, -minR)

            ax.set_title('Phase Spectrum Analysis (LMS Mode)', pad=20, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"[analyze_phase_spectrum] Figure saved to: {save_path}")

            if show_plot:
                plt.show()
            else:
                plt.close(fig)

        # Return LMS mode outputs
        return {
            'harm_phase': harm_phase,
            'harm_mag': harm_mag,
            'freq': freq,
            'noise_dB': noise_dB,
            'bin': None,  # Not used in LMS mode
            'spec': None,  # Not used in LMS mode
            'harmonics': []  # Not used in LMS mode
        }


if __name__ == "__main__":
    print("=" * 60)
    print("Testing analyze_phase_spectrum.py")
    print("=" * 60)

    # Generate test signal with harmonics
    n = 4096
    fs = 1e6
    f_in = fs / n * 101  # Prime bin

    t = np.arange(n) / fs

    # Multiple runs with consistent phase relationship
    data = []
    for i in range(16):
        phase = np.random.randn() * 0.1  # Small phase variation
        signal = 0.45 * np.sin(2 * np.pi * f_in * t + phase)
        # Add harmonics
        signal += 0.02 * np.sin(2 * np.pi * 2 * f_in * t + 2 * phase)
        signal += 0.01 * np.sin(2 * np.pi * 3 * f_in * t + 3 * phase)
        signal += 0.005 * np.sin(2 * np.pi * 4 * f_in * t + 4 * phase)
        # Add noise
        signal += np.random.randn(n) * 0.001
        # DC offset
        signal += 0.5
        data.append(signal)

    data = np.array(data)

    print(f"\n[Test] {len(data)} runs, N={n}, fin={f_in/1e3:.1f}kHz")

    # Test FFT mode
    print("\n--- Testing FFT Mode ---")
    result_fft = analyze_phase_spectrum(
        data,
        harmonic=5,
        mode='FFT',
        save_path='../output_data/test_spectrum_phase_fft.png',
        show_plot=False
    )

    print(f"Fundamental bin: {result_fft['bin']}")
    if 'coherent_result' in result_fft:
        print(f"Valid runs processed: {result_fft['coherent_result']['n_runs']}")
        print(f"Noise floor: {result_fft['coherent_result']['minR_dB']:.1f} dB")

    print(f"\n  Harmonics:")
    for h in result_fft['harmonics'][:5]:
        print(f"    H{h['harmonic']}: bin={h['bin']}, mag={h['magnitude']:.1f}dB, phase={np.degrees(h['phase']):.1f}deg")

    # Test LMS mode
    print("\n--- Testing LMS Mode ---")
    result_lms = analyze_phase_spectrum(
        data,
        harmonic=5,
        mode='LMS',
        save_path='../output_data/test_spectrum_phase_lms.png',
        show_plot=False
    )

    print(f"Fundamental freq: {result_lms['freq']:.6f}")
    print(f"Noise floor: {result_lms['noise_dB']:.1f} dB")
    print(f"\n  Harmonics:")
    for i in range(min(5, len(result_lms['harm_mag']))):
        print(f"    H{i+1}: mag={20*np.log10(result_lms['harm_mag'][i]):.1f}dB, phase={np.degrees(result_lms['harm_phase'][i]):.1f}deg")

    print("\n" + "=" * 60)