"""
Pure polar spectrum plotting functionality without calculations.

This module extracts the plotting logic to create a pure plotting function
that can be used with pre-computed coherent spectrum results.
"""

import numpy as np
import matplotlib.pyplot as plt
from adctoolbox.spectrum._harmonics import _locate_harmonic_bins
from adctoolbox.spectrum._harmonics import _calculate_harmonic_phases

def plot_spectrum_polar(analysis_results, show_metrics=True, harmonic=5, fixed_radial_range=None, ax=None):
    """
    Pure polar spectrum plotting using pre-computed coherent spectrum results.

    Parameters:
        analysis_results: Dictionary containing output from compute_spectrum(coherent_averaging=True)
        show_metrics: Display metrics annotations (True) or not (False)
        harmonic: Number of harmonics to mark on the plot
        fixed_radial_range: Fixed radial range in dB. If None, auto-scales.
        ax: Optional matplotlib polar axes object
    """
    # Extract data from compute_spectrum output
    plot_data = analysis_results['plot_data']
    metrics = analysis_results.get('metrics', {})

    spec = plot_data['complex_spectrum']
    bin_idx = plot_data['fundamental_bin']
    fundamental_bin_fractional = plot_data['fundamental_bin_fractional']
    harmonic_bins = plot_data['harmonic_bins']
    collided_harmonics = plot_data.get('collided_harmonics', [])
    N_fft = analysis_results['N']

    # Calculate noise floor for polar plot (1st percentile)
    mag_db = 20 * np.log10(np.abs(spec) + 1e-20)
    minR_dB = np.percentile(mag_db, 1)
    minR_dB = -100 if np.isinf(minR_dB) else minR_dB

    # Calculate harmonic phases
    hd2_phase_deg, hd3_phase_deg = _calculate_harmonic_phases(spec, harmonic_bins)

    # Setup axes
    if ax is None:
        ax = plt.gca()

    # Verify axes has polar projection
    if not hasattr(ax, 'set_theta_zero_location'):
        raise ValueError("Axes must have polar projection")

    # Calculate magnitude and phase
    phi = spec / (np.abs(spec) + 1e-20)
    mag_dB = 20 * np.log10(np.abs(spec) + 1e-20)

    # Normalize to noise floor
    if fixed_radial_range is not None:
        reference_dB = -fixed_radial_range
        mag_dB = np.maximum(mag_dB, reference_dB)
        radius = mag_dB - reference_dB
    else:
        mag_dB = np.maximum(mag_dB, minR_dB)
        radius = mag_dB - minR_dB

    spec_polar = phi * radius
    phase = np.angle(spec_polar)
    mag = np.abs(spec_polar)

    # Configure polar axes
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # Set radial axis
    if fixed_radial_range is not None:
        max_radius = fixed_radial_range
        minR_dB_rounded = np.round(-fixed_radial_range / 10) * 10
    else:
        max_radius = -minR_dB
        minR_dB_rounded = np.round(minR_dB / 10) * 10

    tick_values = np.arange(0, max_radius + 1, 10)
    ax.set_ylim([0, max_radius])
    ax.set_rticks(tick_values)
    tick_labels = [str(int(minR_dB_rounded + val)) for val in tick_values]
    ax.set_yticklabels(tick_labels, fontsize=10)

    # Plot spectrum
    ax.scatter(phase, mag, s=1, c='k', alpha=0.5)
    ax.tick_params(axis='x', labelsize=10)
    ax.grid(True, alpha=0.3)

    # Mark fundamental
    if bin_idx < len(spec_polar):
        ax.plot(phase[bin_idx], mag[bin_idx], 'bo', markersize=5,
                markerfacecolor='blue', markeredgewidth=1.5, zorder=10)
        ax.plot([0, phase[bin_idx]], [0, mag[bin_idx]], 'b-', linewidth=2, zorder=10)

    # Mark harmonics
    for h in range(2, harmonic + 1):
        # Skip if this harmonic collides with fundamental (collision makes plotting meaningless)
        if h in collided_harmonics:
            continue

        harmonic_bin = (bin_idx * h) % N_fft
        if harmonic_bin > N_fft // 2:
            harmonic_bin = N_fft - harmonic_bin

        # Skip if this harmonic aliases to DC (bin 0)
        if harmonic_bin == 0:
            continue

        if harmonic_bin < len(spec_polar):
            ax.plot(phase[harmonic_bin], mag[harmonic_bin], 'bs',
                   markersize=5, markerfacecolor='none', markeredgewidth=1.5)
            ax.plot([0, phase[harmonic_bin]], [0, mag[harmonic_bin]], 'b-', linewidth=2)

            label_radius = min(mag[harmonic_bin] * 1.08, max_radius * 0.98)
            ax.text(phase[harmonic_bin], label_radius, str(h),
                   fontsize=10, ha='center', va='center')

    # Add metrics annotation
    if show_metrics and metrics:
        # Extract HD2 and HD3 from harmonic_powers array
        harmonic_powers = metrics.get('harmonic_powers', [])
        hd2_db = harmonic_powers[0] if len(harmonic_powers) > 0 else -150
        hd3_db = harmonic_powers[1] if len(harmonic_powers) > 1 else -150

        hd2_str = f"HD2 = {hd2_db:7.2f} dB ∠{hd2_phase_deg:6.1f}°"
        hd3_str = f"HD3 = {hd3_db:7.2f} dB ∠{hd3_phase_deg:6.1f}°"

        # Build collision warning if present
        collision_warning = ""
        if collided_harmonics:
            collision_str = ', '.join([f'HD{h}' for h in sorted(collided_harmonics)])
            collision_warning = f"\n*Collided: {collision_str}"

        metrics_text = (
            f"SNR = {metrics.get('snr_dbc', 0):7.2f} dB\n"
            f"THD = {metrics.get('thd_dbc', 0):7.2f} dB\n"
            f"{hd2_str}\n"
            f"{hd3_str}"
            f"{collision_warning}"
        )
        ax.text(0.02, 0.02, metrics_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='bottom', family='monospace',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

    ax.set_ylim([0, max_radius])