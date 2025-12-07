import matplotlib.pyplot as plt

from adctoolbox.common import estimate_frequency
from adctoolbox.aout import decompose_harmonics
from tests._utils import save_variable, save_fig
from tests.unit._runner import run_unit_test_batch
from tests import config

plt.rcParams['font.size'] = 14
plt.rcParams['axes.grid'] = True

def _process_decompose_harmonics(raw_data, sub_folder, dataset_name, figures_folder, test_name):
    """
    Callback function to process a single file:
    1. Run Thompson decomposition
    2. Save variables (using Pythonic names, auto-mapped to MATLAB names)
    3. Save plot
    """
    # Find input frequency
    re_fin = estimate_frequency(raw_data)

    # Run decompose_harmonics (creates a figure when disp=1)
    fundamental_signal, total_error, harmonic_error, residual_error = decompose_harmonics(raw_data, re_fin, 10, 1)

    # Get the figure and add title
    fig = plt.gcf()
    fig.suptitle(f'tomDecomp: {dataset_name}', fontsize=14)

    # Save plot
    figure_name = f"{test_name}_{dataset_name}_python.png"
    save_fig(figures_folder, figure_name, dpi=150, close_fig=False)

    # Save variables - auto-mapped to MATLAB names
    save_variable(sub_folder, fundamental_signal, 'fundamental_signal')    # → sine_python.csv
    save_variable(sub_folder, total_error, 'total_error')                  # → error_python.csv
    save_variable(sub_folder, harmonic_error, 'harmonic_error')            # → harmic_python.csv
    save_variable(sub_folder, residual_error, 'residual_error')            # → others_python.csv

    # Close figure at the end
    plt.close(fig)

def test_decompose_harmonics(project_root):
    """
    Batch runner for Thompson decomposition.
    """
    run_unit_test_batch(
        project_root=project_root,
        input_subpath=config.AOUT['input_path'], test_module_name="test_decompose_harmonics", file_pattern=config.AOUT['file_pattern'],        process_callback=_process_decompose_harmonics
    )
