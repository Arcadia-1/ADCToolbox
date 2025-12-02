import numpy as np
import matplotlib.pyplot as plt

from adctoolbox.common import sine_fit
from tests._utils import save_variable, save_fig
from tests.unit._runner import run_unit_test_batch
from tests import config

plt.rcParams['font.size'] = 14
plt.rcParams['axes.grid'] = True

def _process_sine_fit(raw_data, sub_folder, dataset_name, figures_folder, test_name):
    """
    Callback function to process a single file:
    1. Perform sine fitting
    2. Save variables
    3. Generate and save plot
    """
    # 1. Sine Fitting
    data_fit, freq, mag, dc, phi = sine_fit(raw_data)

    # 2. Save Variables
    save_variable(sub_folder, freq, 'freq')
    save_variable(sub_folder, mag, 'mag')
    save_variable(sub_folder, dc, 'dc')
    save_variable(sub_folder, phi, 'phi')
    save_variable(sub_folder, data_fit, 'data_fit')

    # 3. Plotting Logic
    period_samples = int(round(1.0 / freq)) if freq > 0 else len(raw_data)
    n_plot = min(max(period_samples, 20), len(raw_data))

    fig = plt.figure(figsize=(8, 6))

    t_data = np.arange(n_plot)
    plt.plot(t_data, raw_data[:n_plot], 'bo-', linewidth=2, markersize=6, label='Original')

    t_dense = np.linspace(0, n_plot - 1, n_plot * 50)
    fitted_sine = mag * np.cos(2 * np.pi * freq * t_dense + phi) + dc
    plt.plot(t_dense, fitted_sine, 'r--', linewidth=2, label='Fitted Sine')

    plt.title(f'Sine Fit: {dataset_name}')
    plt.ylim([np.min(fitted_sine) - 0.1, np.max(fitted_sine) + 0.2])
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper left')
    plt.tight_layout()

    figure_name = f"{dataset_name}_{test_name}_python.png"
    save_fig(figures_folder, figure_name)
    plt.close(fig)

def test_sine_fit(project_root):
    """
    Batch runner for sine_fit (Single Channel Version).
    """
    run_unit_test_batch(
        project_root=project_root,
        input_subpath=config.AOUT['input_path'], test_module_name="test_sine_fit", file_pattern=config.AOUT['file_pattern'],        process_callback=_process_sine_fit
    )