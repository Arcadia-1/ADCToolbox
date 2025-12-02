import numpy as np
import matplotlib.pyplot as plt


def err_pdf(err_data, resolution=12, full_scale=1):
    """
    Compute error probability density function using Kernel Density Estimation (KDE).

    Parameters:
        err_data: Error signal (1D array)
        resolution: ADC resolution in bits (default: 12)
        full_scale: Full-scale range (default: 1)

    Returns:
        noise_lsb: Error in LSB units (1D array)
        mu: Mean of error distribution
        sigma: Standard deviation of error distribution
        KL_divergence: KL divergence from Gaussian distribution
        x: Sample points for PDF
        fx: KDE-estimated PDF values
        gauss_pdf: Fitted Gaussian PDF values
    """
    # Convert error to LSB units
    lsb = full_scale / (2**resolution)
    noise_lsb = np.asarray(err_data).flatten() / lsb
    n = noise_lsb
    N = len(n)

    # Silverman's rule for bandwidth
    h = 1.06 * np.std(n) * N**(-1/5)

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
    sigma = np.std(n)
    gauss_pdf = (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-(x - mu)**2 / (2*sigma**2))

    # KL divergence calculation
    dx = x[1] - x[0]
    p = fx + np.finfo(float).eps
    q = gauss_pdf + np.finfo(float).eps
    KL_divergence = np.sum(p * np.log(p / q)) * dx

    return noise_lsb, mu, sigma, KL_divergence, x, fx, gauss_pdf
