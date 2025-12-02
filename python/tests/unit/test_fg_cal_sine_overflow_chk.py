import matplotlib.pyplot as plt

from adctoolbox.dout import fg_cal_sine, overflow_chk
from tests._utils import save_variable, save_fig
from tests.unit._runner import run_unit_test_batch

plt.rcParams['font.size'] = 14
plt.rcParams['axes.grid'] = True

def _process_overflow_chk(raw_data, sub_folder, dataset_name, figures_folder, test_name):
    """
    Callback function to process a single file:
    1. Run foreground calibration to get weights
    2. Run overflow check analysis
    3. Save data_decom variable
    4. Save plot
    """
    # Run FGCalSine to get calibrated weights
    weights_cal = fg_cal_sine(raw_data)[0]  # Only need weights

    # Run overflow_chk
    fig = plt.figure(figsize=(10, 6))
    plt.ioff()  # Turn off interactive mode
    data_decom = overflow_chk(raw_data, weights_cal)

    plt.title(f'overflow_chk: {dataset_name}')
    # Save plot
    figure_name = f"{dataset_name}_{test_name}_python.png"
    save_fig(figures_folder, figure_name, dpi=150)

    # Save data_decom variable
    save_variable(sub_folder, data_decom, 'data_decom')

def test_fg_cal_sine_overflow_chk(project_root):
    """
    Batch runner for overflow check analysis.
    """
    run_unit_test_batch(
        project_root=project_root,
        input_subpath="dataset/dout",
        test_module_name="test_fg_cal_sine_overflow_chk",
        file_pattern="dout_*.csv",
        output_subpath="test_output",
        process_callback=_process_overflow_chk,
        flatten=False  # Digital output data is 2D (N samples x M bits)
    )
