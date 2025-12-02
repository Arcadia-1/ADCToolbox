import matplotlib.pyplot as plt

from adctoolbox.common import sine_fit
from adctoolbox.aout import spec_plot
from tests._utils import save_fig
from tests.unit._runner import run_unit_test_batch

plt.rcParams['font.size'] = 14
plt.rcParams['axes.grid'] = True

def _process_err_spectrum(raw_data, sub_folder, dataset_name, figures_folder, test_name):
    """
    Callback function to process a single file:
    1. Calculate error data using sine_fit
    2. Plot error spectrum
    3. Save plot
    """
    # Compute error data using sineFit
    data_fit, freq_est, mag, dc, phi = sine_fit(raw_data)
    err_data = raw_data - data_fit

    # Run spec_plot on error data (label=0 means no labeling)
    plt.figure(figsize=(12, 8))
    spec_plot(err_data, label=0)
    plt.title(f'errSpectrum: {dataset_name}')

    # Save plot
    figure_name = f"{dataset_name}_{test_name}_python.png"
    save_fig(figures_folder, figure_name, dpi=150)

def test_err_spectrum(project_root):
    """
    Batch runner for error spectrum analysis.
    """
    run_unit_test_batch(
        project_root=project_root,
        input_subpath="dataset",
        test_module_name="test_err_spectrum",
        file_pattern="sinewave_*.csv",        process_callback=_process_err_spectrum
    )
