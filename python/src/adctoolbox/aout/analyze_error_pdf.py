"""Error probability density function (PDF) analysis with KDE and Gaussian comparison.

Computes error PDF, fits Gaussian, and calculates KL divergence for goodness-of-fit.
"""

import numpy as np
import matplotlib.pyplot as plt
from adctoolbox.fundamentals.fit_sine_4param import fit_sine_4param
from adctoolbox.aout._fit_diagnostics import extract_fit_diagnostics

def analyze_error_pdf(signal, resolution=12, full_scale=None, frequency=None, create_plot: bool = True,
                       ax=None, title: str | None = None, max_iterations: int = 100,
                       tolerance: float = 1e-9, return_fit: bool = False):
    """
    Compute and optionally plot error probability density function using KDE.

    This function automatically fits an ideal sine to the signal,
    computes the error, and analyzes its probability distribution.

    Parameters
    ----------
    signal : np.ndarray
        ADC output signal (1D array)
    resolution : int, default=12
        ADC resolution in bits
    full_scale : float, optional
        Full-scale range. If None, inferred from signal range (max - min)
    frequency : float, optional
        Normalized frequency (0-0.5). If None, auto-detected
    create_plot : bool, default=True
        If True, plot the PDF on current axes
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, uses current axes (plt.gca())
    title : str, optional
        Title for the plot. If None, no title is set
    max_iterations : int, default=100
        Maximum frequency-refinement iterations passed to fit_sine_4param;
        refinement stops early once it converges.
    tolerance : float, default=1e-9
        Frequency-refinement convergence threshold passed to fit_sine_4param.
    return_fit : bool, default=False
        If True, include scalar sine-fit diagnostics under result['fit'].

    Returns
    -------
    result : dict
        Dictionary containing PDF analysis results:
        - 'err_lsb': Error in LSB units (1D array)
        - 'mu': Mean of error distribution (LSB)
        - 'sigma': Standard deviation of error distribution (LSB)
        - 'kl_divergence': KL divergence from Gaussian distribution
        - 'x': Sample points for PDF
        - 'pdf': KDE-estimated PDF values
        - 'gauss_pdf': Fitted Gaussian PDF values
        - 'fit': Optional sine-fit diagnostics when return_fit=True

    Notes
    -----
    - Error = signal - ideal_sine (fitted using fit_sine_4param)
    - Uses Kernel Density Estimation (KDE) with Silverman's bandwidth rule
    - KL divergence measures how different the actual error PDF is from Gaussian
    """

    # Fit ideal sine to extract reference
    fit_kwargs = {"max_iterations": max_iterations, "tolerance": tolerance}
    if frequency is None:
        fit_result = fit_sine_4param(signal, **fit_kwargs)
    else:
        fit_result = fit_sine_4param(signal, frequency_estimate=frequency, **fit_kwargs)

    sig_ideal = fit_result['fitted_signal']

    # Compute error
    err_data = signal - sig_ideal

    # Infer full_scale from signal if not provided
    if full_scale is None:
        full_scale = np.max(signal) - np.min(signal)

    # Convert error to LSB units
    lsb = full_scale / (2**resolution)
    err_lsb = np.asarray(err_data).flatten() / lsb
    N = len(err_lsb)

    # Gaussian fit
    mu = float(np.mean(err_lsb))
    sigma = float(np.std(err_lsb, ddof=1))

    # Silverman's rule for bandwidth (guard degenerate zero-variance residuals)
    h = max(1.06 * sigma * N**(-1 / 5), np.finfo(float).eps)

    # Determine x-axis range covering both KDE tails (+/- 4h) and Gaussian (+/- 5*sigma)
    max_abs_noise = float(np.max(np.abs(err_lsb)))
    half_span = max(max_abs_noise + 4.0 * h, abs(mu) + 5.0 * sigma)
    if sigma < 0.05 and max_abs_noise < 0.1:
        # Narrow error distribution: center grid on the actual distribution so
        # the quadrature grid resolves both KDE and Gaussian peaks accurately.
        narrow_span = max(5.0 * sigma, 4.0 * h, max_abs_noise + 4.0 * h, 1e-12)
        x = np.linspace(mu - narrow_span, mu + narrow_span, 200)
    else:
        xlim_range = max(0.5, half_span)
        x = np.linspace(-xlim_range, xlim_range, 200)

    # KDE computation (vectorized)
    u = (x[:, None] - err_lsb[None, :]) / h
    fx = np.mean(np.exp(-0.5 * u**2), axis=1) / (h * np.sqrt(2 * np.pi))

    if sigma > 0:
        gauss_pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    else:
        gauss_pdf = np.zeros_like(x)

    # Normalize discrete densities on the quadrature grid so KL >= 0 always
    # holds by Gibbs' inequality (prevents negative KL from tail truncation).
    p = fx + np.finfo(float).eps
    q = gauss_pdf + np.finfo(float).eps
    p_norm = p / np.sum(p)
    q_norm = q / np.sum(q)
    kl_divergence = float(max(0.0, np.sum(p_norm * np.log(p_norm / q_norm))))

    # Plot if requested
    if create_plot:
        # Use provided axes or get current axes
        if ax is None:
            ax = plt.gca()

        ax.plot(x, fx, 'b-', linewidth=2, label='Actual PDF (KDE)')
        ax.plot(x, gauss_pdf, 'r--', linewidth=2, label='Gaussian Fit')
        ax.set_xlabel('Error (LSB)')
        ax.set_ylabel('Probability Density')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)

        # Add statistics text box
        stats_text = f'μ = {mu:.3f} LSB\nσ = {sigma:.3f} LSB\nKL = {kl_divergence:.4f}'
        ax.text(0.98, 0.98, stats_text,
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Set title if provided
        if title is not None:
            ax.set_title(title, fontsize=12)

    result = {
        'err_lsb': err_lsb,
        'mu': mu,
        'sigma': sigma,
        'kl_divergence': kl_divergence,
        'x': x,
        'pdf': fx,
        'gauss_pdf': gauss_pdf
    }
    if return_fit:
        result['fit'] = extract_fit_diagnostics(fit_result)

    return result
