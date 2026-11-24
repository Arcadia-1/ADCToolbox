"""Phase plane analysis: visualize ADC data in phase space to detect anomalies."""

import numpy as np
import matplotlib.pyplot as plt

def analyze_phase_plane(data, lag='auto', fs=1.0, detect_outliers=True, threshold=4.0, ax=None, title=None, create_plot: bool = True):
    """
    PhaseScope: Visualize ADC data in phase space to detect anomalies.

    Plots x[n] vs x[n+k] to reveal hysteresis, sparkle codes, and quantization issues.

    Parameters
    ----------
    data : array_like
        1D array of ADC codes or voltage data.
    lag : int or 'auto'
        Delay samples (k).
        - 'auto': Calculates optimal k for a circular plot (approx 90 deg phase shift).
        - int: Manual delay (e.g., 1 for adjacent code check).
    fs : float
        Sampling frequency (only used for 'auto' lag calculation text).
    detect_outliers : bool
        If True, highlights points that deviate significantly from the main trajectory.
    threshold : float
        Sigma threshold for outlier detection (default 4.0).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    title : str, optional
        Custom title for the plot.
    create_plot : bool
        Whether to render the plot immediately.

    Returns
    -------
    dict
        'lag': The integer lag used.
        'outliers': Indices of detected outlier points.
    """
    data = np.asarray(data)
    N = len(data)
    
    # --- 1. Auto-Lag Calculation (Deep Search Mode) ---
    if lag == 'auto':
        # Remove DC
        data_ac = data - np.mean(data)

        # FFT to find dominant frequency
        # Hanning window helps with spectral leakage for non-coherent sampling
        fft_mag = np.abs(np.fft.rfft(data_ac * np.hanning(N)))
        peak_idx = np.argmax(fft_mag[1:]) + 1
        norm_freq = peak_idx / N  # fin / fs

        if norm_freq <= 0:
            k = 1
        else:
            # Deep Search for Near-Nyquist Sampling:
            # For signals close to Fs/2, adjacent samples have ~180° phase difference.
            # The small phase slip accumulates - e.g., 7.6° slip → 90° after ~12 samples.
            #
            # Dynamic search limit based on signal frequency:
            # - For low-frequency oversampled signals, we need to search further
            # - Target is ~1/4 period (90 degrees), with margin for estimation error
            # - Prevent excessive search that could pick up jitter/drift artifacts

            # For tones near DC or near Nyquist, the phase slip per sample is
            # min(f, 0.5 - f); reaching 90 deg requires k ≈ 0.25 / slip.
            phase_slip = min(norm_freq, abs(0.5 - norm_freq))
            estimated_period = 1.0 / max(phase_slip, 1e-6)
            search_limit = min(N // 2, max(int(estimated_period * 0.6) + 20, 20))
            best_k = 1
            best_metric = -1

            for candidate_k in range(1, search_limit):
                phase_shift = 2 * np.pi * norm_freq * candidate_k
                # We want sin(phase) close to 1 or -1 (i.e., 90 or 270 degrees)
                metric = np.abs(np.sin(phase_shift))

                # Very small distance penalty: only prefer smaller k if circularity is equal
                # Reduced from 0.01 to 0.0001 to allow algorithm to confidently choose k=12
                # when it's actually optimal (rather than settling for k=1 with worse circularity)
                score = metric - (candidate_k * 0.0001)

                if score > best_metric:
                    best_metric = score
                    best_k = candidate_k

            k = best_k
    else:
        k = int(lag)

    if k <= 0 or k >= N:
        raise ValueError(f"Lag {k} must be in [1, {N - 1}] for data length {N}.")

    # --- 2. Construct Phase Space Vectors ---
    # x axis: current sample, y axis: delayed sample
    x_vec = data[:-k]
    y_vec = data[k:]

    # --- 3. Outlier Detection (Sparkle Hunter) ---
    outlier_mask = np.zeros_like(x_vec, dtype=bool)
    sample_outlier_mask = np.zeros(N, dtype=bool)

    if detect_outliers:
        # Center data to origin
        xc = x_vec - np.mean(x_vec)
        yc = y_vec - np.mean(y_vec)

        # Calculate radius from center
        radius = np.sqrt(xc**2 + yc**2)

        # Robust statistics using MAD (Median Absolute Deviation)
        # MAD is immune to outliers, unlike std which is pulled by large sparkle codes
        r_median = np.median(radius)
        deviation = np.abs(radius - r_median)
        mad = np.median(deviation)

        # Convert MAD to sigma-equivalent (1.4826 is the consistency constant for normal distribution)
        r_sigma_robust = mad * 1.4826

        # Prevent division by zero for perfect signals
        if r_sigma_robust < 1e-12:
            r_sigma_robust = 1e-12

        # Flag points far from the "donut" ring
        outlier_mask = deviation > (threshold * r_sigma_robust)

        # A single glitch at sample m corrupts BOTH pair (m-k, m) and pair (m, m+k).
        # Map pair flags back to true sample indices so each isolated glitch is
        # reported once (at m) rather than twice (at m and m+k).
        n_pairs = len(outlier_mask)
        if n_pairs > 0:
            # Interior samples [k, N-k): appear as y in pair m-k and x in pair m
            if N > 2 * k:
                sample_outlier_mask[k : N - k] = outlier_mask[: N - 2 * k] & outlier_mask[k:]
            # Leading samples [0, min(k, n_pairs)): appear only as x in pair m
            lead = min(k, n_pairs)
            sample_outlier_mask[:lead] = outlier_mask[:lead] & ~np.concatenate(
                (outlier_mask[k : k + lead], np.zeros(max(0, lead - max(0, n_pairs - k)), dtype=bool))
            )
            # Trailing samples [max(k, N-k), N): appear only as y in pair m-k
            trail_start = max(k, N - k)
            trail_len = N - trail_start
            if trail_len > 0:
                prev_pair = np.zeros(trail_len, dtype=bool)
                for idx_t in range(trail_len):
                    m_idx = trail_start + idx_t
                    if m_idx - 2 * k >= 0:
                        prev_pair[idx_t] = outlier_mask[m_idx - 2 * k]
                sample_outlier_mask[trail_start:] = outlier_mask[trail_start - k :] & ~prev_pair

    outlier_indices = np.flatnonzero(sample_outlier_mask)

    # --- 4. Visualization ---
    if create_plot:
        # Use current axes if available, otherwise use pyplot's implicit axes
        if ax is None:
            ax = plt.gca()

        # Plot Main Trajectory (Normal points)
        # For large datasets (>20k points), use hexbin for performance and density visualization
        # For small datasets, use scatter for clarity
        if len(x_vec) > 20000:
            ax.hexbin(x_vec[~outlier_mask], y_vec[~outlier_mask],
                      gridsize=80, cmap='Blues', mincnt=1, alpha=0.6, linewidths=0.2)
        else:
            ax.scatter(x_vec[~outlier_mask], y_vec[~outlier_mask],
                       s=2, c='blue', alpha=0.3, label='Trajectory', edgecolors='none')

        # Plot Outliers (Red markers) - always show these regardless of data size
        if np.any(outlier_mask):
            ax.scatter(x_vec[outlier_mask], y_vec[outlier_mask],
                       s=20, c='red', marker='x', label='Anomalies', linewidth=1.5, zorder=10)

            # Annotate the first few outliers
            for i in range(min(3, len(outlier_indices))):
                idx = outlier_indices[i]
                ax.annotate(f'idx:{idx}', (x_vec[outlier_mask][i], y_vec[outlier_mask][i]),
                            fontsize=8, xytext=(5, 5), textcoords='offset points')

        # Formatting
        if title is not None:
            ax.set_title(title, fontsize=12, fontweight='bold')
        else:
            ax.set_title(f"Phase Plane Analysis (Lag k={k})", fontsize=12, fontweight='bold')

        ax.set_xlabel("x[n]", fontsize=10)
        ax.set_ylabel(f"x[n+{k}]", fontsize=10)
        ax.grid(True, which='both', alpha=0.3)
        ax.axis('equal')  # Critical for circular geometry

        # Add stats text box - compact version for all cases
        stats_text = f'Lag: {k}\nOutliers: {len(outlier_indices)}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                 verticalalignment='top', fontsize=8,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='gray'))

    return {
        "lag": k,
        "outliers": outlier_indices
    }
