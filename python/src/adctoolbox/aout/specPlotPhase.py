"""
Spectrum Phase Analyzer

Polar plot of spectrum with phase information.
Uses coherent phase alignment across multiple measurements.

Ported from MATLAB: specPlotPhase.m
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from ..common.alias import alias


def spec_plot_phase(
    data: np.ndarray,
    n_fft: Optional[int] = None,
    harmonic: int = 5,
    osr: int = 1,
    save_path: Optional[str] = None,
    show_plot: bool = False
) -> dict:
    """
    Spectrum phase analysis with polar plot.

    Aligns phase across multiple measurements for coherent averaging,
    then displays magnitude and phase in polar coordinates.

    Args:
        data: ADC output data, shape (M, N) for M runs or (N,) for single run
        n_fft: FFT size (default: length of data)
        harmonic: Number of harmonics to mark (default 5)
        osr: Oversampling ratio (default 1)
        save_path: Path to save figure (optional)
        show_plot: Whether to display plot (default False)

    Returns:
        dict with keys:
            - spec: Complex spectrum (phase-aligned)
            - freq_bin: Fundamental frequency bin
            - harmonics: List of (bin, magnitude, phase) for each harmonic
    """
    data = np.asarray(data)

    # Handle 1D or 2D input
    if data.ndim == 1:
        data = data.reshape(1, -1)

    n_runs, n = data.shape

    if n_fft is None:
        n_fft = n

    nd2 = n_fft // 2 // osr

    # Coherent averaging with phase alignment
    spec = np.zeros(n_fft, dtype=complex)
    me = 0

    for i in range(n_runs):
        tdata = data[i, :]
        if np.std(tdata) == 0:
            continue

        # Normalize
        tdata = tdata / (np.max(tdata) - np.min(tdata))
        tdata = tdata - np.mean(tdata)

        # FFT
        tspec = np.fft.fft(tdata)
        tspec[0] = 0

        # Find fundamental
        bin_fund = np.argmax(np.abs(tspec[:n_fft // 2 // osr]))
        phi = tspec[bin_fund] / np.abs(tspec[bin_fund])

        # Phase alignment for harmonics
        # Match MATLAB: process all N_fft harmonics with aliasing
        phasor = np.conj(phi)
        marker = np.zeros(n_fft, dtype=bool)

        for h in range(1, n_fft + 1):
            j = bin_fund * h  # 0-based frequency bin

            # Check which Nyquist zone and apply aliasing
            if (j // (n_fft // 2)) % 2 == 0:
                # Even zone: no reflection
                b = j % n_fft
                if not marker[b]:
                    tspec[b] = tspec[b] * phasor
                    marker[b] = True
            else:
                # Odd zone: reflected
                b = n_fft - (j % n_fft)
                if b < n_fft and not marker[b]:
                    tspec[b] = tspec[b] * np.conj(phasor)
                    marker[b] = True

            phasor = phasor * np.conj(phi)

        # Non-harmonic phase shift
        # Match MATLAB behavior (note: MATLAB has bug using outer loop var)
        for h in range(n_fft):
            if not marker[h] and bin_fund > 0:
                tspec[h] = tspec[h] * (np.conj(phi) ** (h / bin_fund))

        spec += tspec
        me += 1

    if me == 0:
        raise ValueError("No valid data runs")

    # Take in-band portion
    spec = spec[:nd2]

    # Calculate magnitude in dB
    # Note: Use conjugate to match MATLAB convention
    phi_spec = np.conj(spec) / (np.abs(spec) + 1e-20)
    mag_db = 10 * np.log10(np.abs(spec)**2 / (n_fft**2) * 16 / me**2 + 1e-20)

    # Normalize magnitude for polar plot
    # Match MATLAB: spec_sort(ceil(length(spec_sort)*0.01))
    mag_sorted = np.sort(mag_db)
    min_r_idx = int(np.ceil(len(mag_sorted) * 0.01))
    min_r = mag_sorted[min_r_idx - 1] if min_r_idx > 0 else mag_sorted[0]  # -1 for 0-based indexing
    if np.isinf(min_r):
        min_r = -100
    mag_norm = np.maximum(mag_db, min_r) - min_r

    # Find fundamental bin
    bin_fund = np.argmax(mag_norm)

    # Complex spectrum for polar plot
    spec_polar = phi_spec * mag_norm

    # Collect harmonic info
    harmonics_info = []
    for h in range(1, harmonic + 1):
        b = alias(bin_fund * h, n_fft)  # Python bin_fund is 0-based
        if b < nd2:
            harmonics_info.append({
                'harmonic': h,
                'bin': b,
                'magnitude': mag_db[b],
                'phase': np.angle(np.conj(spec[b]))
            })

    # Plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')

    # Plot all points
    angles = np.angle(spec_polar)
    radii = np.abs(spec_polar)
    ax.scatter(angles, radii, c='k', s=2, alpha=0.5)

    # Mark fundamental
    ax.plot(angles[bin_fund], radii[bin_fund], 'ro', markersize=10)
    ax.plot([0, angles[bin_fund]], [0, radii[bin_fund]], 'r-', linewidth=2)

    # Mark harmonics
    for h in range(2, harmonic + 1):
        b = alias(bin_fund * h, n_fft)  # Python bin_fund is 0-based
        if b < nd2:
            ax.plot([0, angles[b]], [0, radii[b]], 'b-', linewidth=2.5)
            ax.annotate(str(h), (angles[b], radii[b] + 2),
                       fontsize=10, ha='center')

    # Configure polar plot
    ax.set_theta_direction(-1)  # Clockwise
    ax.set_theta_zero_location('N')  # 0 degrees at top

    # Custom tick labels showing dB values (match MATLAB exactly)
    # MATLAB: tick = [-minR:-10:0]; pax.RTick = tick(end:-1:1);
    #         tickl = (0:-10:minR); pax.RTickLabel = tickl(end:-1:1);
    # minR is negative (e.g., -120), so -minR is positive (e.g., 120)
    tick = np.arange(-min_r, -1, -10)  # e.g., [120, 110, 100, ..., 10]
    tick = tick[::-1]  # Reverse: [10, 20, 30, ..., 120]
    # Prepend 0 if not already there
    if len(tick) == 0 or tick[0] != 0:
        tick = np.concatenate([[0], tick])

    tickl = np.arange(0, min_r, -10)  # e.g., [0, -10, -20, ..., -120]
    tickl = tickl[::-1]  # Reverse: [-120, -110, ..., -10, 0]

    # Ensure tick and tickl have same length
    if len(tick) != len(tickl):
        min_len = min(len(tick), len(tickl))
        tick = tick[:min_len]
        tickl = tickl[:min_len]

    ax.set_rticks(tick)
    ax.set_yticklabels([f'{int(t):.0f}' for t in tickl])
    ax.set_rlim(0, -min_r)

    ax.set_title('Spectrum Phase', pad=20, fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[specPlotPhase] Figure saved to: {save_path}")

    if show_plot:
        plt.show()

    # Close figure to prevent memory leak (after showing/saving)
    if fig is not None:
        plt.close(fig)

    # Prepare outputs matching MATLAB version
    freq_bins = np.arange(nd2) / n_fft

    return {
        'spec': spec_polar,  # Phase-weighted spectrum (complex)
        'phi': phi_spec,     # Normalized phase (complex)
        'bin': bin_fund,     # Fundamental bin index
        'freq_bins': freq_bins,  # Normalized frequency bins
        'harmonics': harmonics_info,
        'mag_db': mag_db,
        'phase': np.angle(np.conj(spec))
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Testing specPlotPhase.py")
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

    result = spec_plot_phase(
        data,
        harmonic=5,
        save_path='../output_data/test_spectrum_phase.png',
        show_plot=False
    )

    print(f"\n[Results]")
    print(f"  Fundamental bin: {result['bin']}")
    print(f"\n  Harmonics:")
    for h in result['harmonics']:
        print(f"    H{h['harmonic']}: bin={h['bin']}, mag={h['magnitude']:.1f}dB, phase={np.degrees(h['phase']):.1f}deg")

    print("\n" + "=" * 60)
