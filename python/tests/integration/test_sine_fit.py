import numpy as np
import matplotlib.pyplot as plt

from adctoolbox.common import fit_sine
from tests._utils import save_variable, save_fig
from tests.unit._runner import run_unit_test_batch
from tests import config

plt.rcParams['font.size'] = 14
plt.rcParams['axes.grid'] = True

def _process_fit_sine(raw_data, sub_folder, dataset_name, figures_folder, test_name):
    """
    Callback function to process a single file:
    1. Perform sine fitting
    2. Save variables
    3. Generate and save plot
    """
    # 1. Sine Fitting - using new Pythonic names
    fitted_signal, frequency, amplitude, dc_offset, phase = fit_sine(raw_data)

    # 2. Save Variables - Pythonic names auto-mapped to MATLAB names
    save_variable(sub_folder, frequency, 'frequency')        # → freq_python.csv
    save_variable(sub_folder, amplitude, 'amplitude')        # → mag_python.csv
    save_variable(sub_folder, dc_offset, 'dc_offset')        # → dc_python.csv
    save_variable(sub_folder, phase, 'phase')                # → phi_python.csv
    save_variable(sub_folder, fitted_signal, 'fitted_signal')  # → fitout_python.csv

    # 3. Plotting Logic
    period_samples = int(round(1.0 / frequency)) if frequency > 0 else len(raw_data)
    n_plot = min(max(period_samples, 20), len(raw_data))

    fig = plt.figure(figsize=(8, 6))

    t_data = np.arange(n_plot)
    plt.plot(t_data, raw_data[:n_plot], 'bo-', linewidth=2, markersize=6, label='Original')

    t_dense = np.linspace(0, n_plot - 1, n_plot * 50)
    fitted_sine = amplitude * np.cos(2 * np.pi * frequency * t_dense + phase) + dc_offset
    plt.plot(t_dense, fitted_sine, 'r--', linewidth=2, label='Fitted Sine')

    plt.title(f'Sine Fit: {dataset_name}')
    plt.ylim([np.min(fitted_sine) - 0.1, np.max(fitted_sine) + 0.2])
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper left')
    plt.tight_layout()

    figure_name = f"{test_name}_{dataset_name}_python.png"
    save_fig(figures_folder, figure_name)
    plt.close(fig)

def test_fit_sine(project_root):
    """
    Batch runner for fit_sine (Single Channel Version).
    """
    run_unit_test_batch(
        project_root=project_root,
        input_subpath=config.AOUT['input_path'], test_module_name="test_fit_sine", file_pattern=config.AOUT['file_pattern'],        process_callback=_process_fit_sine
    )