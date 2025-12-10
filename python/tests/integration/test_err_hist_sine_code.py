import matplotlib.pyplot as plt

from adctoolbox.common import estimate_frequency
from adctoolbox.aout import plot_error_hist_code
from tests._utils import save_variable, save_fig
from tests.unit._runner import run_unit_test_batch
from tests import config

plt.rcParams['font.size'] = 14
plt.rcParams['axes.grid'] = True

def _process_plot_error_hist_code(raw_data, sub_folder, dataset_name, figures_folder, test_name):
    """
    Callback function to process a single file:
    1. Find fundamental frequency
    2. Run plot_error_hist_code
    3. Save variables
    4. Save plot
    """
    # 1. Find fundamental frequency
    freq = estimate_frequency(raw_data, fs=1)

    # 2. Error Histogram Analysis (Code Mode)
    emean_code, erms_code, code_axis, error, codes = plot_error_hist_code(
        raw_data,
        bins=256,
        freq=freq,
        disp=1
    )

    # Get the figure that err_hist_sine created and add title
    fig = plt.gcf()
    fig.suptitle(f'Error Histogram (Code): {dataset_name}', fontsize=14)

    # 3. Save Figure (before saving variables to ensure figure is current)
    figure_name = f"{test_name}_{dataset_name}_python.png"
    save_fig(figures_folder, figure_name, close_fig=False)

    # 4. Save Variables (only the 3 that are in the reference output)
    save_variable(sub_folder, code_axis, 'code_axis')
    save_variable(sub_folder, emean_code, 'emean_code')
    save_variable(sub_folder, erms_code, 'erms_code')

    # Close figure at the end
    plt.close(fig)

def test_plot_error_hist_code(project_root):
    """
    Batch runner for plot_error_hist_code (Single Channel Version).
    """
    run_unit_test_batch(
        project_root=project_root,
        input_subpath=config.AOUT['input_path'], test_module_name="test_plot_error_hist_code", file_pattern=config.AOUT['file_pattern'],        process_callback=_process_plot_error_hist_code
    )
