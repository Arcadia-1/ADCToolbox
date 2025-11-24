import numpy as np
import matplotlib.pyplot as plt


def errPDF(err_data, Resolution=12, FullScale=1):
    """
    Compute error probability density function using Kernel Density Estimation (KDE).

    Parameters:
        err_data: Error signal (1D array)
        Resolution: ADC resolution in bits (default: 12)
        FullScale: Full-scale range (default: 1)

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
    lsb = FullScale / (2**Resolution)
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

    # Plotting with larger fonts to match MATLAB
    plt.plot(x, fx, linewidth=2, label='KDE Estimate')
    label_str = f"Gaussian Fit (KL = {KL_divergence:.4f}, μ={mu:.2f}, σ={sigma:.2f})"
    plt.plot(x, gauss_pdf, '--r', linewidth=2, label=label_str)
    plt.xlabel("Noise (LSB)", fontsize=18)
    plt.ylabel("PDF", fontsize=18)
    plt.legend(loc="upper left", fontsize=18)
    plt.grid(True)

    # Set y-axis limits
    y_max = max(np.max(fx), np.max(gauss_pdf)) * 1.3
    plt.ylim([0, y_max])
    plt.gca().tick_params(labelsize=18)

    return noise_lsb, mu, sigma, KL_divergence, x, fx, gauss_pdf
