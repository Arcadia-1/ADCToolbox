import matplotlib.pyplot as plt

from adctoolbox.dout.bit_activity import bit_activity
from tests._utils import save_variable, save_fig
from tests.unit._runner import run_unit_test_batch
from tests import config

plt.rcParams['font.size'] = 16
plt.rcParams['axes.grid'] = True

def _process_bit_activity(raw_data, sub_folder, dataset_name, figures_folder, test_name):
    """
    Callback function to process a single file:
    1. Run bit activity analysis
    2. Save bit usage variable
    3. Save plot
    """
    # Create figure and run bit_activity
    fig = plt.figure(figsize=(10, 7.5))
    bit_usage = bit_activity(raw_data)
    plt.gca().tick_params(labelsize=16)
    plt.title(f'Bit activity: {dataset_name}')

    figure_name = f"{test_name}_{dataset_name}_python.png"
    save_fig(figures_folder, figure_name, dpi=150)

    # Save bit_usage data
    save_variable(sub_folder, bit_usage, 'bit_usage')

def test_bit_activity(project_root):
    """
    Batch runner for bit activity analysis.
    """
    run_unit_test_batch(
        project_root=project_root,
        input_subpath=config.DOUT['input_path'], test_module_name="test_bit_activity", file_pattern=config.DOUT['file_pattern'],        process_callback=_process_bit_activity,
        flatten=False  # Digital output data is 2D (N samples x M bits)
    )
