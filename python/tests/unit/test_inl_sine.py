import numpy as np
import matplotlib.pyplot as plt

from adctoolbox.aout import inl_sine
from tests._utils import save_fig
from tests.unit._runner import run_unit_test_batch

plt.rcParams['font.size'] = 14
plt.rcParams['axes.grid'] = True

# Resolution list to scan
RESOLUTION_LIST = [12]

def _process_inl_sine(raw_data, sub_folder, dataset_name, figures_folder, test_name):
    """
    Callback function to process a single file:
    1. Scale data by resolution
    2. Calculate INL/DNL
    3. Generate and save plot
    """
    for Resolution in RESOLUTION_LIST:
        # 1. Scale data by 2^Resolution
        scaled_data = raw_data * (2 ** Resolution)
        expected_max = 2 ** Resolution

        # 2. Calculate INL/DNL
        INL, DNL, code = inl_sine(scaled_data)

        # Calculate ranges for plot titles
        max_inl = np.max(INL)
        min_inl = np.min(INL)
        max_dnl = np.max(DNL)
        min_dnl = np.min(DNL)

        # 3. Generate Plot
        fig = plt.figure(figsize=(10, 8))

        # Top subplot: INL
        plt.subplot(2, 1, 1)
        plt.scatter(code, INL, s=8, alpha=0.6)
        plt.xlabel('Code')
        plt.ylabel('INL (LSB)')
        plt.grid(True)
        plt.title(f'INL = [{min_inl:.2f}, {max_inl:+.2f}] LSB')
        ylim_min = min(min_inl, -1)
        ylim_max = max(max_inl, 1)
        plt.ylim([ylim_min, ylim_max])
        plt.xlim([0, expected_max])

        # Bottom subplot: DNL
        plt.subplot(2, 1, 2)
        plt.scatter(code, DNL, s=8, alpha=0.6)
        plt.xlabel('Code')
        plt.ylabel('DNL (LSB)')
        plt.grid(True)
        plt.title(f'DNL = [{min_dnl:.2f}, {max_dnl:.2f}] LSB')
        ylim_min = min(min_dnl, -1)
        ylim_max = max(max_dnl, 1)
        plt.ylim([ylim_min, ylim_max])
        plt.xlim([0, expected_max])

        plt.tight_layout()

        # 4. Save Figure
        figure_name = f"{dataset_name}_{test_name}_python.png"
        save_fig(figures_folder, figure_name, dpi=150)
        plt.close(fig)

def test_inl_sine(project_root):
    """
    Batch runner for inl_sine (Single Channel Version).
    """
    run_unit_test_batch(
        project_root=project_root,
        input_subpath="dataset/aout/sinewave",
        test_module_name="test_inl_sine",
        file_pattern="sinewave_*.csv",
        output_subpath="test_output",
        process_callback=_process_inl_sine
    )
