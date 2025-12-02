import matplotlib.pyplot as plt

from adctoolbox.dout import fg_cal_sine
from adctoolbox.dout.weight_scaling import weight_scaling
from tests._utils import save_variable, save_fig
from tests.unit._runner import run_unit_test_batch

plt.rcParams['font.size'] = 14
plt.rcParams['axes.grid'] = True

def _process_weight_scaling(raw_data, sub_folder, dataset_name, figures_folder, test_name):
    """
    Callback function to process a single file:
    1. Run foreground calibration to get weights
    2. Run weight scaling analysis
    3. Save radix and weight_cal variables
    4. Save plot
    """
    # Run FGCalSine to get calibrated weights
    weight_cal, offset, k_static, residual, cost, freq_cal = fg_cal_sine(
        raw_data, freq=0, order=5)

    # Run weightScaling tool
    fig = plt.figure(figsize=(8, 6))
    radix = weight_scaling(weight_cal)
    plt.gca().tick_params(labelsize=16)

    # Save figure
    figure_name = f"{dataset_name}_{test_name}_python.png"
    save_fig(figures_folder, figure_name, dpi=150)

    # Save variables
    save_variable(sub_folder, radix, 'radix')
    save_variable(sub_folder, weight_cal, 'weight_cal')

def test_weight_scaling(project_root):
    """
    Batch runner for weight scaling analysis.
    """
    run_unit_test_batch(
        project_root=project_root,
        input_subpath="dataset",
        test_module_name="test_weight_scaling",
        file_pattern="dout_*.csv",        process_callback=_process_weight_scaling,
        flatten=False  # Digital output data is 2D (N samples x M bits)
    )
