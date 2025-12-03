import numpy as np
import matplotlib.pyplot as plt

from adctoolbox.dout import fg_cal_sine, overflow_chk
from tests._utils import save_variable, save_fig
from tests.unit._runner import run_unit_test_batch
from tests import config

plt.rcParams['font.size'] = 14
plt.rcParams['axes.grid'] = True

def _process_overflow_chk(raw_data, sub_folder, dataset_name, figures_folder, test_name):
    """
    Callback function to process a single file:
    1. Run foreground calibration to get calibrated weights
    2. Run overflow_chk to get overflow statistics
    3. Create visualization plot
    4. Save variables and plot
    """
    # Run FGCalSine to get calibrated weights
    weight, _, _, _, _, _ = fg_cal_sine(
        raw_data,
        freq=0,
        order=5
    )

    # Run overflow_chk and get overflow statistics
    range_min, range_max, ovf_percent_zero, ovf_percent_one = overflow_chk(
        raw_data, weight, ofb=None, disp=False
    )

    # Create visualization plot
    fig = plt.figure(figsize=(10, 6))
    overflow_chk(raw_data, weight, ofb=None, disp=True)
    plt.title(f'Overflow Check: {dataset_name}')

    # Save outputs
    figure_name = f"{test_name}_{dataset_name}_python.png"
    save_fig(figures_folder, figure_name, dpi=100)
    plt.close(fig)

    # Save variables
    save_variable(sub_folder, range_min, 'range_min')
    save_variable(sub_folder, range_max, 'range_max')
    save_variable(sub_folder, ovf_percent_zero, 'ovf_percent_zero')
    save_variable(sub_folder, ovf_percent_one, 'ovf_percent_one')

def test_overflow_chk(project_root):
    """
    Batch runner for overflow_chk function.
    Tests overflow detection by analyzing bit segment residue distributions.
    """
    run_unit_test_batch(
        project_root=project_root,
        input_subpath=config.DOUT['input_path'],
        test_module_name="test_overflow_chk",
        file_pattern=config.DOUT['file_pattern'],
        process_callback=_process_overflow_chk,
        flatten=False  # Digital output data is 2D (N samples x M bits)
    )
