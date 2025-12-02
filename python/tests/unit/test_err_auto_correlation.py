import matplotlib.pyplot as plt

from adctoolbox.common import sine_fit
from adctoolbox.aout import err_auto_correlation
from tests._utils import save_variable, save_fig
from tests.unit._runner import run_unit_test_batch

plt.rcParams['font.size'] = 14
plt.rcParams['axes.grid'] = True

def _process_err_auto_correlation(raw_data, sub_folder, dataset_name, figures_folder, test_name):
    """
    Callback function to process a single file:
    1. Calculate error data using sine_fit
    2. Run error autocorrelation analysis
    3. Save variables and plot
    """
    # Compute error data using sineFit
    data_fit, freq_est, mag, dc, phi = sine_fit(raw_data)
    err_data = raw_data - data_fit

    # Run errAutoCorrelation
    acf, lags = err_auto_correlation(err_data, max_lag=200, normalize=False)

    # Create plot
    fig = plt.figure(figsize=(8, 6))
    plt.plot(lags, acf, linewidth=2)
    plt.grid(True)
    plt.xlabel("Lag (samples)", fontsize=14)
    plt.ylabel("Autocorrelation", fontsize=14)
    plt.title(test_name)
    plt.gca().tick_params(labelsize=14)

    # Save plot and variables
    figure_name = f"{dataset_name}_{test_name}_python.png"
    save_fig(figures_folder, figure_name, dpi=150)
    save_variable(sub_folder, lags, 'lags')
    save_variable(sub_folder, acf, 'acf')

def test_err_auto_correlation(project_root):
    """
    Batch runner for error autocorrelation analysis.
    """
    run_unit_test_batch(
        project_root=project_root,
        input_subpath="dataset",
        test_module_name="test_err_auto_correlation",
        file_pattern="sinewave_*.csv",
        process_callback=_process_err_auto_correlation
    )
