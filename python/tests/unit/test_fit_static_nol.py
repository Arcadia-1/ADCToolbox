import matplotlib.pyplot as plt
import numpy as np

from adctoolbox.common import sine_fit
from adctoolbox.aout import fit_static_nol
from tests._utils import save_variable, save_fig
from tests.unit._runner import run_unit_test_batch
from tests import config

plt.rcParams['font.size'] = 14
plt.rcParams['axes.grid'] = True

def _process_fit_static_nol(raw_data, sub_folder, dataset_name, figures_folder, test_name):
    """
    Callback function to process a single file:
    1. Get ideal sine fit
    2. Extract static nonlinearity coefficients
    3. Compute residual error
    4. Create visualization
    5. Save variables and plot
    """
    # 1. Get ideal fit for plotting
    sig_fit, freq, mag, dc, phi = sine_fit(raw_data)

    # 2. Extract static nonlinearity coefficients
    order = 3
    k1, k2, k3, polycoeff, fit_curve = fit_static_nol(raw_data, order=order, freq=freq)

    print(f'  [Static non-linearity: k1={k1:.6f}, k2={k2:.6f}, k3={k3:.6f}]')

    # 3. Compute residual error
    x_ideal = sig_fit - dc
    y_actual = raw_data - np.mean(raw_data)
    y_fit = fit_curve - dc

    residual = y_actual - y_fit
    residual_rms = np.sqrt(np.mean(residual**2))

    # 4. Create visualization of transfer function
    fig = plt.figure(figsize=(10, 7.5))

    plt.plot(x_ideal, residual, 'b.', markersize=3, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=1.5)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Ideal Input (zero-mean)', fontsize=14)
    plt.ylabel('Residual Error', fontsize=14)
    plt.title(f'Fit Residual: k1={k1:.6f}, k2={k2:.6f}, k3={k3:.6f} (RMS={residual_rms:.2e})',
              fontsize=14)

    # Add dataset name as subtitle
    fig.suptitle(f'Static Nonlinearity Fit: {dataset_name}', fontsize=16, y=0.98)
    plt.tight_layout()

    # 5. Save outputs
    figure_name = f"{dataset_name}_{test_name}_python.png"
    save_fig(figures_folder, figure_name, dpi=150, close_fig=False)

    # Save variables
    save_variable(sub_folder, k1, 'k1')
    save_variable(sub_folder, k2, 'k2')
    save_variable(sub_folder, k3, 'k3')
    save_variable(sub_folder, polycoeff, 'polycoeff')
    save_variable(sub_folder, fit_curve, 'fit_curve')
    save_variable(sub_folder, residual, 'residual')

    # Close figure at the end
    plt.close(fig)

def test_fit_static_nol(project_root):
    """
    Batch runner for fit_static_nol function.
    Tests static nonlinearity extraction from ADC transfer function.
    """
    run_unit_test_batch(
        project_root=project_root,
        input_subpath=config.AOUT['input_path'],
        test_module_name="test_fit_static_nol",
        file_pattern=config.AOUT['file_pattern'],
        process_callback=_process_fit_static_nol
    )
