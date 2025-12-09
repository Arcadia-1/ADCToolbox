import matplotlib.pyplot as plt

from adctoolbox.common import fit_sine
from adctoolbox.aout import analyze_spectrum
from tests._utils import save_fig, save_variable
from tests.unit._runner import run_unit_test_batch
from tests import config

plt.rcParams['font.size'] = 14
plt.rcParams['axes.grid'] = True

def _process_err_spectrum(raw_data, sub_folder, dataset_name, figures_folder, test_name):
    """
    Callback function to process a single file:
    1. Calculate error data using fit_sine
    2. Plot error spectrum
    3. Save variables
    4. Save plot
    """
    # Compute error data using sineFit
    fit_result = fit_sine(raw_data)
    fitted_signal = fit_result['fitted_signal']
    err_data = raw_data - fitted_signal

    # Run analyze_spectrum on error data (show_label=False means no labeling)
    plt.figure(figsize=(12, 8))
    analyze_spectrum(err_data, show_label=False)
    plt.title(f'errSpectrum: {dataset_name}')

    # Save variables
    save_variable(sub_folder, err_data, 'err_data')

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
