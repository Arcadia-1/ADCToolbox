import numpy as np
import matplotlib.pyplot as plt

from adctoolbox.common import fit_sine
from adctoolbox.aout import plot_error_pdf
from tests._utils import save_variable, save_fig
from tests.unit._runner import run_unit_test_batch
from tests import config

plt.rcParams['font.size'] = 14
plt.rcParams['axes.grid'] = True

def _process_plot_error_pdf(raw_data, sub_folder, dataset_name, figures_folder, test_name):
    """
    Callback function to process a single file:
    1. Calculate error data using fit_sine
    2. Run plot_error_pdf analysis
    3. Save variables and plot

    NOTE: Both Python plot_error_pdf and MATLAB errpdf now take raw signal
    and compute error internally (MATLAB was updated to match Python).
    """
    # Run plot_error_pdf with raw data (Python's design)
    # Now matches MATLAB interface (both take raw signal)
    noise_lsb, mu, sigma, KL_divergence, x, fx, gauss_pdf = plot_error_pdf(
        raw_data,
        resolution=12,  # Match MATLAB resolution
        full_scale=np.max(raw_data) - np.min(raw_data)
    )

    # Create plot
    plt.figure(figsize=(12, 8))
    plt.plot(x, fx, 'b-', linewidth=2, label='KDE')
    plt.plot(x, gauss_pdf, 'r--', linewidth=2, label='Gaussian Fit')
    plt.xlabel('Error (LSB)', fontsize=14)
    plt.ylabel('Probability Density', fontsize=14)
    plt.title(f'errPDF: {dataset_name}', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)

    # Save plot
    figure_name = f"{test_name}_{dataset_name}_python.png"
    save_fig(figures_folder, figure_name, dpi=150)

    # Save variables
    save_variable(sub_folder, mu, 'mu')
    save_variable(sub_folder, sigma, 'sigma')
    save_variable(sub_folder, KL_divergence, 'KL_divergence')
    save_variable(sub_folder, x, 'x')
    save_variable(sub_folder, fx, 'fx')
    save_variable(sub_folder, gauss_pdf, 'gauss_pdf')

def test_plot_error_pdf(project_root):
    """
    Batch runner for error PDF analysis.
    """
    run_unit_test_batch(
        project_root=project_root,
        input_subpath=config.AOUT['input_path'],
        test_module_name="test_plot_error_pdf",
        file_pattern=config.AOUT['file_pattern'],
        process_callback=_process_plot_error_pdf
    )
