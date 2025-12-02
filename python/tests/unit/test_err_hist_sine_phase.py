import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from adctoolbox.common import find_fin
from adctoolbox.aout import err_hist_sine
from tests._utils import save_variable, save_fig
from tests.unit._runner import run_unit_test_batch
from tests import config

plt.rcParams['font.size'] = 14
plt.rcParams['axes.grid'] = True

def _process_err_hist_sine_phase(raw_data, sub_folder, dataset_name, figures_folder, test_name):
    """
    Callback function to process a single file:
    1. Find fundamental frequency
    2. Run err_hist_sine in phase mode
    3. Save variables
    4. Save plot
    """
    # 1. Find fundamental frequency
    freq = find_fin(raw_data, fs=1)

    # 2. Error Histogram Analysis (Phase Mode)
    emean, erms, phase_code, anoi, pnoi, _, _ = err_hist_sine(
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
    figure_name = f"{dataset_name}_{test_name}_python.png"
    save_fig(figures_folder, figure_name, close_fig=False)

    # 4. Save Variables
    save_variable(sub_folder, anoi, 'anoi')
    save_variable(sub_folder, pnoi, 'pnoi')
    save_variable(sub_folder, phase_code, 'phase_code')
    save_variable(sub_folder, emean, 'emean')
    save_variable(sub_folder, erms, 'erms')

    # Close figure at the end
    plt.close(fig)

def test_err_hist_sine_phase(project_root):
    """
    Batch runner for err_hist_sine phase mode (Single Channel Version).
    """
    run_unit_test_batch(
        project_root=project_root,
        input_subpath=config.AOUT['input_path'], test_module_name="test_err_hist_sine_phase", file_pattern=config.AOUT['file_pattern'],        process_callback=_process_err_hist_sine_phase
    )
