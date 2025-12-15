"""
Error probability density function (PDF) analysis with KDE and Gaussian comparison.

Computes error PDF, fits Gaussian, and calculates KL divergence for goodness-of-fit.

MATLAB counterpart: errpdf.m
"""

import numpy as np
import matplotlib.pyplot as plt
from adctoolbox.aout.fit_sine_4param import fit_sine_4param as fit_sine


def plot_error_pdf(sig_distorted, resolution=12, full_scale=1, freq=0, plot=False):
    """
    Compute and optionally plot error probability density function using KDE.

    This function automatically fits an ideal sine to the distorted signal,
    computes the error, and analyzes its probability distribution.

    Parameters:
        sig_distorted: Distorted ADC output signal (1D array)
        resolution: ADC resolution in bits (default: 12)
        full_scale: Full-scale range (default: 1)
        freq: Normalized frequency (0-0.5), 0 for auto-detection (default: 0)
        plot: If True, plot the PDF on current axes (default: False)

    Returns:
        err_lsb: Error in LSB units (1D array)
        mu: Mean of error distribution (LSB)
        sigma: Standard deviation of error distribution (LSB)
        KL_divergence: KL divergence from Gaussian distribution
        x: Sample points for PDF
        fx: KDE-estimated PDF values
        gauss_pdf: Fitted Gaussian PDF values

    Transfer Function Model:
        error = sig_distorted - sig_ideal
        where sig_ideal is fitted using sine_fit

    Usage Examples:
        # Basic usage (no plot)
        err_lsb, mu, sigma, KL, x, fx, gauss = err_pdf(signal, resolution=12)

        # With automatic plotting on current axes
        plt.sca(ax)
        err_lsb, mu, sigma, KL, x, fx, gauss = err_pdf(signal, resolution=12, plot=True)
    """

    # Fit ideal sine to extract reference
    from adctoolbox.aout.fit_sine_4param import fit_sine_4param as fit_sine
    if freq == 0:
        fit_result = fit_sine(sig_distorted)
        sig_ideal = fit_result['fitted_signal']
    else:
        fit_result = fit_sine(sig_distorted, freq)
        sig_ideal = fit_result['fitted_signal']

    # Compute error
    err_data = sig_distorted - sig_ideal

    # Convert error to LSB units
    lsb = full_scale / (2**resolution)
    err_lsb = np.asarray(err_data).flatten() / lsb
    n = err_lsb
    N = len(n)

    # Silverman's rule for bandwidth
    h = 1.06 * np.std(n, ddof=1) * N**(-1/5)

    # Determine x-axis range
    max_abs_noise = np.max(np.abs(n))
    xlim_range = max(0.5, max_abs_noise)
    x = np.linspace(-xlim_range, xlim_range, 200)
    fx = np.zeros_like(x)

    # KDE computation (manual implementation for consistency with MATLAB)
    for i in range(len(x)):
        u = (x[i] - n) / h
        fx[i] = np.mean(np.exp(-0.5 * u**2)) / (h * np.sqrt(2*np.pi))

    # Gaussian fit
    mu = np.mean(n)
    sigma = np.std(n, ddof=1)
    gauss_pdf = (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-(x - mu)**2 / (2*sigma**2))

    # KL divergence calculation
    dx = x[1] - x[0]
    p = fx + np.finfo(float).eps
    q = gauss_pdf + np.finfo(float).eps
    KL_divergence = np.sum(p * np.log(p / q)) * dx

    # Plot if requested
    if plot:
        plt.plot(x, fx, 'b-', linewidth=2, label='Actual PDF (KDE)')
        plt.plot(x, gauss_pdf, 'r--', linewidth=2, label='Gaussian Fit')
        plt.xlabel('Error (LSB)', fontsize=11)
        plt.ylabel('Probability Density', fontsize=11)
        plt.title('Error Probability Density Function', fontsize=12, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        # Add statistics text box
        stats_text = f'μ = {mu:.3f} LSB\nσ = {sigma:.3f} LSB\nKL = {KL_divergence:.4f}'
        ax = plt.gca()
        ax.text(0.98, 0.98, stats_text,
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    return err_lsb, mu, sigma, KL_divergence, x, fx, gauss_pdf
