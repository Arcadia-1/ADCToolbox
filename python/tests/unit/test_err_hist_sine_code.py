import matplotlib.pyplot as plt

from adctoolbox.common import find_fin
from adctoolbox.aout import err_hist_sine
from adctoolbox.aout import fit_static_nol
from tests._utils import save_variable, save_fig
from tests.unit._runner import run_unit_test_batch
from tests import config

plt.rcParams['font.size'] = 14
plt.rcParams['axes.grid'] = True

def _process_err_hist_sine_code(raw_data, sub_folder, dataset_name, figures_folder, test_name):
    """
    Callback function to process a single file:
    1. Find fundamental frequency
    2. Run err_hist_sine in code mode
    3. Extract static nonlinearity coefficients
    4. Save variables
    5. Save plot
    """
    # 1. Find fundamental frequency
    freq = find_fin(raw_data, fs=1)

    # 2. Error Histogram Analysis (Code Mode)
    emean_code, erms_code, code_axis, _, _, _, _ = err_hist_sine(
        raw_data,
        bin=256,
        fin=freq,
        disp=1,
        mode=1
    )

    # 3. Extract static nonlinearity coefficients using fit_static_nol
    k1, k2, k3, polycoeff, fit_curve = fit_static_nol(raw_data, order=3, freq=freq)

    # Get the figure that err_hist_sine created and add title
    fig = plt.gcf()
    fig.suptitle(f'Error Histogram (Code): {dataset_name}', fontsize=14)

    # 4. Save Figure (before saving variables to ensure figure is current)
    figure_name = f"{dataset_name}_{test_name}_python.png"
    save_fig(figures_folder, figure_name, close_fig=False)

    # 5. Save Variables
    save_variable(sub_folder, code_axis, 'code_axis')
    save_variable(sub_folder, emean_code, 'emean_code')
    save_variable(sub_folder, erms_code, 'erms_code')
    save_variable(sub_folder, k1, 'k1')
    save_variable(sub_folder, k2, 'k2')
    save_variable(sub_folder, k3, 'k3')

    # Close figure at the end
    plt.close(fig)

def test_err_hist_sine_code(project_root):
    """
    Batch runner for err_hist_sine code mode (Single Channel Version).
    """
    run_unit_test_batch(
        project_root=project_root,
        input_subpath=config.AOUT['input_path'], test_module_name="test_err_hist_sine_code", file_pattern=config.AOUT['file_pattern'],        process_callback=_process_err_hist_sine_code
    )
