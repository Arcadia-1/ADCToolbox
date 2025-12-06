import matplotlib.pyplot as plt
import numpy as np

from adctoolbox.aout import extract_static_nonlin
from tests._utils import save_variable, save_fig
from tests.unit._runner import run_unit_test_batch
from tests import config

plt.rcParams['font.size'] = 14
plt.rcParams['axes.grid'] = True

def _process_fit_static_nol(raw_data, sub_folder, dataset_name, figures_folder, test_name):
    """
    Callback function to process a single file:
    1. Extract static nonlinearity coefficients
    2. Compute residual error
    3. Create visualization
    4. Save variables and plot
    """
    # 1. Extract static nonlinearity coefficients
    order = 3
    k2, k3, fitted_sine, fitted_transfer = extract_static_nonlin(raw_data, order)

    print(f'  [Static non-linearity: k2={k2:.6f}, k3={k3:.6f}]')

    # 2. Prepare plotting data (match exp_a07 example)
    # Measured residual: deviation from the fundamental sine wave
    residual = raw_data - fitted_sine

    # Fitted curve: use the smooth transfer curve directly
    transfer_x, transfer_y = fitted_transfer
    nonlinearity_curve = transfer_y - transfer_x

    # 3. Create visualization of nonlinearity error
    fig = plt.figure(figsize=(10, 7.5))

    plt.plot(fitted_sine, residual, 'b.', markersize=1, alpha=0.5, label='Measured')
    plt.plot(transfer_x, nonlinearity_curve, 'r-', linewidth=2, label='Fitted Model')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Input Amplitude (V)', fontsize=14)
    plt.ylabel('Nonlinearity Error (V)', fontsize=14)
    plt.title(f'Static Nonlinearity: k2={k2:.6f}, k3={k3:.6f}',
              fontsize=14)
    plt.legend(loc='upper left', fontsize=12)

    # Add dataset name as subtitle
    fig.suptitle(f'Static Nonlinearity Extraction: {dataset_name}', fontsize=16, y=0.98)
    plt.tight_layout()

    # 4. Save outputs
    figure_name = f"{test_name}_{dataset_name}_python.png"
    save_fig(figures_folder, figure_name, dpi=150, close_fig=False)

    # Save variables
    save_variable(sub_folder, k2, 'k2')
    save_variable(sub_folder, k3, 'k3')
    save_variable(sub_folder, fitted_sine, 'fitted_sine')
    save_variable(sub_folder, transfer_x, 'fitted_transfer_x')
    save_variable(sub_folder, transfer_y, 'fitted_transfer_y')

    # Close figure at the end
    plt.close(fig)

def test_fit_static_nol(project_root):
    """
    Batch runner for extract_static_nonlin function.
    Tests static nonlinearity extraction from ADC transfer function.
    """
    run_unit_test_batch(
        project_root=project_root,
        input_subpath=config.AOUT['input_path'],
        test_module_name="test_fit_static_nol",
        file_pattern=config.AOUT['file_pattern'],
        process_callback=_process_fit_static_nol
    )
