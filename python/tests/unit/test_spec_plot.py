import matplotlib.pyplot as plt

from adctoolbox.aout import spec_plot
from tests._utils import save_variable, save_fig
from tests.unit._runner import run_unit_test_batch

plt.rcParams['font.size'] = 14
plt.rcParams['axes.grid'] = True

def _process_spec_plot(raw_data, sub_folder, dataset_name):
    """
    Callback function to process a single file:
    1. Perform spectral analysis
    2. Save variables
    3. Generate and save plot
    """
    # 1. Spectral Analysis
    fig = plt.figure(figsize=(12, 8))
    ENoB, SNDR, SFDR, SNR, THD, pwr, NF, _ = spec_plot(
        raw_data,
        label=1,
        harmonic=5,
        OSR=1,
        NFMethod=0
    )
    plt.title(f'Spectrum: {dataset_name}')

    # 2. Save Variables
    save_variable(sub_folder, ENoB, 'ENoB')
    save_variable(sub_folder, SNDR, 'SNDR')
    save_variable(sub_folder, SFDR, 'SFDR')
    save_variable(sub_folder, SNR, 'SNR')
    save_variable(sub_folder, THD, 'THD')
    save_variable(sub_folder, pwr, 'pwr')
    save_variable(sub_folder, NF, 'NF')

    # 3. Save Figure
    save_fig(sub_folder, 'spectrum_python.png', dpi=100)
    plt.close(fig)

def test_spec_plot(project_root):
    """
    Batch runner for spec_plot (Single Channel Version).
    """
    run_unit_test_batch(
        project_root=project_root,
        input_subpath="dataset/aout/sinewave",
        test_module_name="test_spec_plot",
        file_pattern="sinewave_*.csv",
        output_subpath="test_output",
        process_callback=_process_spec_plot
    )