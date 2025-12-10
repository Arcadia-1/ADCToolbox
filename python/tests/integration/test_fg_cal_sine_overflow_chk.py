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
    1. Run foreground calibration to get weights
    2. Run overflow check analysis
    3. Save data_decom variable
    4. Save plot
    """
    # Run calibrate_weight_sine to get calibrated weights
    weights_cal = calibrate_weight_sine(raw_data)[0]  # Only need weights

    # Run check_overflow
    fig = plt.figure(figsize=(10, 6))
    plt.ioff()  # Turn off interactive mode
    data_decom = check_overflow(raw_data, weights_cal)

    plt.title(f'check_overflow: {dataset_name}')
    # Save plot
    figure_name = f"{test_name}_{dataset_name}_python.png"
    save_fig(figures_folder, figure_name, dpi=150)

    # Save data_decom variable
    save_variable(sub_folder, data_decom, 'data_decom')

def test_calibrate_weight_sine_check_overflow(project_root):
    """
    Batch runner for overflow check analysis.
    """
    run_unit_test_batch(
        project_root=project_root,
        input_subpath=config.DOUT['input_path'], test_module_name="test_calibrate_weight_sine_check_overflow", file_pattern=config.DOUT['file_pattern'],        process_callback=_process_check_overflow,
        flatten=False  # Digital output data is 2D (N samples x M bits)
    )
