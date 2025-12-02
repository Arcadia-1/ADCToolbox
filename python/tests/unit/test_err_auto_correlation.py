import matplotlib.pyplot as plt

from adctoolbox.common import sine_fit
from adctoolbox.aout import err_auto_correlation
from tests._utils import save_variable, save_fig
from tests.unit._runner import run_unit_test_batch

plt.rcParams['font.size'] = 14
plt.rcParams['axes.grid'] = True

def _process_err_auto_correlation(raw_data, sub_folder, dataset_name):
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
    plt.figure(figsize=(12, 8))
    acf, lags = err_auto_correlation(err_data, MaxLag=200)
    plt.title(f'errAutoCorrelation: {dataset_name}')

    # Save plot and variables
    save_fig(sub_folder, 'errACF_python.png', dpi=150)
    save_variable(sub_folder, lags, 'lags')
    save_variable(sub_folder, acf, 'acf')

def test_err_auto_correlation(project_root):
    """
    Batch runner for error autocorrelation analysis.
    """
    run_unit_test_batch(
        project_root=project_root,
        input_subpath="dataset/aout/sinewave",
        test_module_name="test_errAutoCorrelation",
        file_pattern="sinewave_*.csv",
        output_subpath="test_output",
        process_callback=_process_err_auto_correlation
    )
