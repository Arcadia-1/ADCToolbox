import matplotlib.pyplot as plt

from adctoolbox.common import find_fin
from adctoolbox.aout import err_hist_sine
from tests._utils import save_variable, save_fig
from tests.unit._runner import run_unit_test_batch

plt.rcParams['font.size'] = 14
plt.rcParams['axes.grid'] = True

def _process_err_hist_sine_phase(raw_data, sub_folder, dataset_name):
    """
    Callback function to process a single file:
    1. Find fundamental frequency
    2. Run err_hist_sine in phase mode
    3. Save variables
    4. Save plot
    """
    # 1. Find fundamental frequency
    freq = find_fin(raw_data, Fs=1)

    # 2. Error Histogram Analysis (Phase Mode)
    emean, erms, phase_code, anoi, pnoi, _, _, _, _, _, _ = err_hist_sine(
        raw_data,
        bin=360,
        fin=freq,
        disp=1,
        mode=0
    )
    plt.gcf().suptitle(f'Error Histogram (Phase): {dataset_name}')

    # 3. Save Variables
    save_variable(sub_folder, anoi, 'anoi')
    save_variable(sub_folder, pnoi, 'pnoi')
    save_variable(sub_folder, phase_code, 'phase_code')
    save_variable(sub_folder, emean, 'emean')
    save_variable(sub_folder, erms, 'erms')

    # 4. Save Figure
    save_fig(sub_folder, 'errHistSine_phase_python.png')
    plt.close()

def test_err_hist_sine_phase(project_root):
    """
    Batch runner for err_hist_sine phase mode (Single Channel Version).
    """
    run_unit_test_batch(
        project_root=project_root,
        input_subpath="dataset/aout/sinewave",
        test_module_name="test_err_hist_sine_phase",
        file_pattern="sinewave_*.csv",
        output_subpath="test_output",
        process_callback=_process_err_hist_sine_phase
    )
