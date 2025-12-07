import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from adctoolbox.common import estimate_frequency
from adctoolbox.aout import plot_error_hist_phase
from tests._utils import save_variable, save_fig
from tests.unit._runner import run_unit_test_batch
from tests import config

plt.rcParams['font.size'] = 14
plt.rcParams['axes.grid'] = True

def _process_plot_error_hist_phase(raw_data, sub_folder, dataset_name, figures_folder, test_name):
    """
    Callback function to process a single file:
    1. Find fundamental frequency
    2. Run plot_error_hist_phase
    3. Save variables
    4. Save plot
    """
    # 1. Find fundamental frequency
    freq = estimate_frequency(raw_data, fs=1)

    # 2. Error Histogram Analysis (Phase Mode)
    emean, erms, phase_code, anoi, pnoi, _, _ = plot_error_hist_phase(
        raw_data,
        bin=360,
        fin=freq,
        disp=1,
        mode=0
    )

    # Get the figure that err_hist_sine created and add title
    fig = plt.gcf()
    fig.suptitle(f'Error Histogram (Phase): {dataset_name}', fontsize=14)

    # 3. Save Figure (before saving variables to ensure figure is current)
    figure_name = f"{test_name}_{dataset_name}_python.png"
    save_fig(figures_folder, figure_name, close_fig=False)

    # 4. Save Variables
    save_variable(sub_folder, anoi, 'anoi')
    save_variable(sub_folder, pnoi, 'pnoi')
    save_variable(sub_folder, phase_code, 'phase_code')
    save_variable(sub_folder, emean, 'emean')
    save_variable(sub_folder, erms, 'erms')

    # Close figure at the end
    plt.close(fig)

def test_plot_error_hist_phase(project_root):
    """
    Batch runner for plot_error_hist_phase (Single Channel Version).
    """
    run_unit_test_batch(
        project_root=project_root,
        input_subpath=config.AOUT['input_path'], test_module_name="test_plot_error_hist_phase", file_pattern=config.AOUT['file_pattern'],        process_callback=_process_plot_error_hist_phase
    )
