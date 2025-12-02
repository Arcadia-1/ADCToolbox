import matplotlib.pyplot as plt

from adctoolbox.common import find_fin
from adctoolbox.aout import tom_decomp
from tests._utils import save_variable, save_fig
from tests.unit._runner import run_unit_test_batch
from tests import config

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

    # Run tomDecomp (creates a figure when disp=1)
    signal, error, indep, dep, phi = tom_decomp(raw_data, re_fin, 10, 1)

    # Get the figure and add title
    fig = plt.gcf()
    fig.suptitle(f'tomDecomp: {dataset_name}', fontsize=14)

    # Save plot
    figure_name = f"{dataset_name}_{test_name}_python.png"
    save_fig(figures_folder, figure_name, dpi=150, close_fig=False)

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

    # Close figure at the end
    plt.close(fig)

def test_tom_decomp(project_root):
    """
    Batch runner for Thompson decomposition.
    """
    run_unit_test_batch(
        project_root=project_root,
        input_subpath=config.AOUT['input_path'], test_module_name="test_tom_decomp", file_pattern=config.AOUT['file_pattern'],        process_callback=_process_tom_decomp
    )
