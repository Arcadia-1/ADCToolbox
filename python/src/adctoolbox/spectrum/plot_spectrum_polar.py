"""
Pure polar spectrum plotting functionality without calculations.

This module extracts the plotting logic to create a pure plotting function
that can be used with pre-computed coherent spectrum results.
"""

import numpy as np
import matplotlib.pyplot as plt


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
    spec = analysis_results['complex_spec_coherent']
    minR_dB = analysis_results['minR_dB']
    bin_idx = analysis_results['bin_idx']
    N_fft = analysis_results['N']
    metrics = analysis_results.get('metrics', {})
    collided_harmonics = analysis_results.get('collided_harmonics', [])

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
        hd2_str = f"HD2 = {metrics['hd2_db']:7.2f} dB ∠{analysis_results['hd2_phase_deg']:6.1f}°"
        hd3_str = f"HD3 = {metrics['hd3_db']:7.2f} dB ∠{analysis_results['hd3_phase_deg']:6.1f}°"

        # Build collision warning if present
        collision_warning = ""
        if collided_harmonics:
            collision_str = ', '.join([f'HD{h}' for h in sorted(collided_harmonics)])
            collision_warning = f"\n*Collided: {collision_str}"

        metrics_text = (
            f"SNR = {metrics['snr_db']:7.2f} dB\n"
            f"THD = {metrics['thd_db']:7.2f} dB\n"
            f"{hd2_str}\n"
            f"{hd3_str}"
            f"{collision_warning}"
        )
        ax.text(0.02, 0.02, metrics_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='bottom', family='monospace',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

    ax.set_ylim([0, max_radius])