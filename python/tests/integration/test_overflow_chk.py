import numpy as np
import matplotlib.pyplot as plt

from adctoolbox.dout import calibrate_weight_sine, check_overflow
from tests._utils import save_variable, save_fig
from tests.unit._runner import run_unit_test_batch
from tests import config

plt.rcParams['font.size'] = 14
plt.rcParams['axes.grid'] = True

def _process_check_overflow(raw_data, sub_folder, dataset_name, figures_folder, test_name):
    """
    Callback function to process a single file:
    1. Run foreground calibration to get calibrated weights
    2. Run check_overflow to get overflow statistics
    3. Create visualization plot
    4. Save variables and plot
    """
    # Run calibrate_weight_sine to get calibrated weights
    weight, _, _, _, _, _ = calibrate_weight_sine(
        raw_data,
        freq=0,
        order=5
    )

    # Run check_overflow and get overflow statistics
    range_min, range_max, ovf_percent_zero, ovf_percent_one = check_overflow(
        raw_data, weight, ofb=None, disp=False
    )

    # Create visualization plot
    fig = plt.figure(figsize=(10, 6))
    check_overflow(raw_data, weight, ofb=None, disp=True)
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

def test_check_overflow(project_root):
    """
    Batch runner for check_overflow function.
    Tests overflow detection by analyzing bit segment residue distributions.
    """
    run_unit_test_batch(
        project_root=project_root,
        input_subpath=config.DOUT['input_path'],
        test_module_name="test_check_overflow",
        file_pattern=config.DOUT['file_pattern'],
        process_callback=_process_check_overflow,
        flatten=False  # Digital output data is 2D (N samples x M bits)
    )
