import matplotlib.pyplot as plt

from adctoolbox.common import sine_fit
from adctoolbox.aout import spec_plot
from tests._utils import save_fig
from tests.unit._runner import run_unit_test_batch
from tests import config

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
    fitted_signal, frequency, amplitude, dc_offset, phase = sine_fit(raw_data)
    err_data = raw_data - fitted_signal

    # Run spec_plot on error data (label=0 means no labeling)
    plt.figure(figsize=(12, 8))
    spec_plot(err_data, label=0)
    plt.title(f'errSpectrum: {dataset_name}')

    # Save plot
    figure_name = f"{test_name}_{dataset_name}_python.png"
    save_fig(figures_folder, figure_name, dpi=150)

def test_err_spectrum(project_root):
    """
    Batch runner for error spectrum analysis.
    """
    run_unit_test_batch(
        project_root=project_root,
        input_subpath=config.AOUT['input_path'], test_module_name="test_err_spectrum", file_pattern=config.AOUT['file_pattern'],        process_callback=_process_err_spectrum
    )
