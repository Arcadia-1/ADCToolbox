import matplotlib.pyplot as plt

from adctoolbox.dout import calibrate_weight_sine, plot_weight_radix
from tests._utils import save_variable, save_fig
from tests.unit._runner import run_unit_test_batch
from tests import config

plt.rcParams['font.size'] = 14
plt.rcParams['axes.grid'] = True

def _process_plot_weight_radix(raw_data, sub_folder, dataset_name, figures_folder, test_name):
    """
    Callback function to process a single file:
    1. Run foreground calibration to get weights
    2. Run weight scaling analysis
    3. Save radix and weight_cal variables
    4. Save plot
    """
    # Run calibrate_weight_sine to get calibrated weights
    weight_cal, offset, k_static, residual, cost, freq_cal = calibrate_weight_sine(
        raw_data, freq=0, order=5)

    # Run plot_weight_radix tool
    fig = plt.figure(figsize=(8, 6))
    radix = plot_weight_radix(weight_cal)
    plt.gca().tick_params(labelsize=16)

    # Save figure
    figure_name = f"{test_name}_{dataset_name}_python.png"
    save_fig(figures_folder, figure_name, dpi=150)

    # Save variables
    save_variable(sub_folder, radix, 'radix')
    save_variable(sub_folder, weight_cal, 'weight_cal')

def test_plot_weight_radix(project_root):
    """
    Batch runner for weight scaling analysis.
    """
    run_unit_test_batch(
        project_root=project_root,
        input_subpath=config.DOUT['input_path'], test_module_name="test_plot_weight_radix", file_pattern=config.DOUT['file_pattern'],        process_callback=_process_plot_weight_radix,
        flatten=False  # Digital output data is 2D (N samples x M bits)
    )
