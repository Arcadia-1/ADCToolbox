import matplotlib.pyplot as plt

from adctoolbox.common import fit_sine
from adctoolbox.aout import plot_error_autocorr
from tests._utils import save_variable, save_fig
from tests.unit._runner import run_unit_test_batch
from tests import config

plt.rcParams['font.size'] = 14
plt.rcParams['axes.grid'] = True

def _process_plot_error_autocorr(raw_data, sub_folder, dataset_name, figures_folder, test_name):
    """
    Callback function to process a single file:
    1. Calculate error data using fit_sine
    2. Run error autocorrelation analysis
    3. Save variables and plot
    """
    # Compute error data using sineFit
    fitted_signal, frequency, amplitude, dc_offset, phase = fit_sine(raw_data)
    err_data = raw_data - fitted_signal

    # Run errAutoCorrelation
    acf, lags = plot_error_autocorr(err_data, max_lag=200, normalize=False)

    # Create plot
    fig = plt.figure(figsize=(8, 6))
    plt.plot(lags, acf, linewidth=2)
    plt.grid(True)
    plt.xlabel("Lag (samples)", fontsize=14)
    plt.ylabel("Autocorrelation", fontsize=14)
    plt.title(test_name)
    plt.gca().tick_params(labelsize=14)

    # Save plot and variables
    figure_name = f"{test_name}_{dataset_name}_python.png"
    save_fig(figures_folder, figure_name, dpi=150)
    save_variable(sub_folder, lags, 'lags')
    save_variable(sub_folder, acf, 'acf')

def test_plot_error_autocorr(project_root):
    """
    Batch runner for error autocorrelation analysis.
    """
    run_unit_test_batch(
        project_root=project_root,
        input_subpath=config.AOUT['input_path'], test_module_name="test_plot_error_autocorr", file_pattern=config.AOUT['file_pattern'],
        process_callback=_process_plot_error_autocorr
    )
