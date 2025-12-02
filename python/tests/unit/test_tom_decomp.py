import matplotlib.pyplot as plt

from adctoolbox.common import find_fin
from adctoolbox.aout import tom_decomp
from tests._utils import save_variable, save_fig
from tests.unit._runner import run_unit_test_batch

plt.rcParams['font.size'] = 14
plt.rcParams['axes.grid'] = True

def _process_tom_decomp(raw_data, sub_folder, dataset_name, figures_folder, test_name):
    """
    Callback function to process a single file:
    1. Run Thompson decomposition
    2. Save variables (signal, error, indep, dep, phi, rms metrics)
    3. Save plot
    """
    # Find input frequency
    re_fin = find_fin(raw_data)

    # Run tomDecomp
    signal, error, indep, dep, phi = tom_decomp(raw_data, re_fin, 10, 1)

    # Current figure is the decomposition plot
    plt.gcf().suptitle(f'tomDecomp: {dataset_name}')

    # Save plot
    figure_name = f"{dataset_name}_{test_name}_python.png"
    save_fig(figures_folder, figure_name, dpi=150)

    # Calculate metrics
    rms_error = (error**2).mean()**0.5
    rms_indep = (indep**2).mean()**0.5
    rms_dep = (dep**2).mean()**0.5

    # Save variables
    save_variable(sub_folder, signal, 'signal')
    save_variable(sub_folder, error, 'error')
    save_variable(sub_folder, indep, 'indep')
    save_variable(sub_folder, dep, 'dep')
    save_variable(sub_folder, phi, 'phi')
    save_variable(sub_folder, rms_error, 'rms_error')
    save_variable(sub_folder, rms_indep, 'rms_indep')
    save_variable(sub_folder, rms_dep, 'rms_dep')

def test_tom_decomp(project_root):
    """
    Batch runner for Thompson decomposition.
    """
    run_unit_test_batch(
        project_root=project_root,
        input_subpath="dataset",
        test_module_name="test_tom_decomp",
        file_pattern="sinewave_*.csv",        process_callback=_process_tom_decomp
    )
