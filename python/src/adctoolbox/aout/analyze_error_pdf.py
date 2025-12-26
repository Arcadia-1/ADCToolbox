"""Error probability density function (PDF) analysis with KDE and Gaussian comparison.

Computes error PDF, fits Gaussian, and calculates KL divergence for goodness-of-fit.
"""

import numpy as np
import matplotlib.pyplot as plt
from adctoolbox.fundamentals.fit_sine_4param import fit_sine_4param

def analyze_error_pdf(signal, resolution=12, full_scale=None, frequency=None, create_plot: bool = True,
                       ax=None, title: str | None = None):
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

    Notes
    -----
    - Error = signal - ideal_sine (fitted using fit_sine_4param)
    - Uses Kernel Density Estimation (KDE) with Silverman's bandwidth rule
    - KL divergence measures how different the actual error PDF is from Gaussian
    """

    # Fit ideal sine to extract reference
    if frequency is None:
        fit_result = fit_sine_4param(signal)
    else:
        fit_result = fit_sine_4param(signal, frequency_estimate=frequency)

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

    # Silverman's rule for bandwidth
    h = 1.06 * np.std(err_lsb, ddof=1) * N**(-1/5)

    # Determine x-axis range
    max_abs_noise = np.max(np.abs(err_lsb))
    xlim_range = max(0.5, max_abs_noise)
    x = np.linspace(-xlim_range, xlim_range, 200)
    fx = np.zeros_like(x)

    # KDE computation
    for i in range(len(x)):
        u = (x[i] - err_lsb) / h
        fx[i] = np.mean(np.exp(-0.5 * u**2)) / (h * np.sqrt(2*np.pi))

    # Gaussian fit
    mu = np.mean(err_lsb)
    sigma = np.std(err_lsb, ddof=1)
    gauss_pdf = (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-(x - mu)**2 / (2*sigma**2))

    # KL divergence calculation
    dx = x[1] - x[0]
    p = fx + np.finfo(float).eps
    q = gauss_pdf + np.finfo(float).eps
    kl_divergence = np.sum(p * np.log(p / q)) * dx

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
            ax.set_title(title, fontsize=10, fontweight='bold')

    return {
        'err_lsb': err_lsb,
        'mu': mu,
        'sigma': sigma,
        'kl_divergence': kl_divergence,
        'x': x,
        'pdf': fx,
        'gauss_pdf': gauss_pdf
    }
