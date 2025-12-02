import matplotlib.pyplot as plt

from adctoolbox.common import find_fin
from adctoolbox.aout import err_hist_sine
from tests._utils import save_variable, save_fig
from tests.unit._runner import run_unit_test_batch

plt.rcParams['font.size'] = 14
plt.rcParams['axes.grid'] = True

def _process_err_hist_sine_code(raw_data, sub_folder, dataset_name):
    """
    Callback function to process a single file:
    1. Find fundamental frequency
    2. Run err_hist_sine in code mode
    3. Save variables
    4. Save plot
    """
    # 1. Find fundamental frequency
    freq = find_fin(raw_data, Fs=1)

    # 2. Error Histogram Analysis (Code Mode)
    emean_code, erms_code, code_axis, _, _, _, _, polycoeff, k1, k2, k3 = err_hist_sine(
        raw_data,
        bin=256,
        fin=freq,
        disp=1,
        mode=1,
        polyorder=3
    )
    plt.gcf().suptitle(f'Error Histogram (Code): {dataset_name}')

    # 3. Save Variables
    save_variable(sub_folder, code_axis, 'code_axis')
    save_variable(sub_folder, emean_code, 'emean_code')
    save_variable(sub_folder, erms_code, 'erms_code')
    save_variable(sub_folder, k1, 'k1')
    save_variable(sub_folder, k2, 'k2')
    save_variable(sub_folder, k3, 'k3')

    # 4. Save Figure
    save_fig(sub_folder, 'errHistSine_code_python.png')
    plt.close()

def test_err_hist_sine_code(project_root):
    """
    Batch runner for err_hist_sine code mode (Single Channel Version).
    """
    run_unit_test_batch(
        project_root=project_root,
        input_subpath="dataset/aout/sinewave",
        test_module_name="test_err_hist_sine_code",
        file_pattern="sinewave_*.csv",
        output_subpath="test_output",
        process_callback=_process_err_hist_sine_code
    )
