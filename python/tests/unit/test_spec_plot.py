import matplotlib.pyplot as plt

from adctoolbox.aout import spec_plot
from tests._utils import save_variable, save_fig
from tests.unit._runner import run_unit_test_batch

plt.rcParams['font.size'] = 14
plt.rcParams['axes.grid'] = True

def _process_spec_plot(raw_data, sub_folder, dataset_name, figures_folder, test_name):
    # 1. Spectral Analysis
    fig = plt.figure(figsize=(8, 6))
    ENoB, SNDR, SFDR, SNR, THD, pwr, NF, _ = spec_plot(
        raw_data,
        label=1,
        harmonic=5,
        osr=1,
        nf_method=0
    )
    plt.title("Spectrum")

    # 2. Save Variables
    save_variable(sub_folder, ENoB, 'ENoB')
    save_variable(sub_folder, SNDR, 'SNDR')
    save_variable(sub_folder, SFDR, 'SFDR')
    save_variable(sub_folder, SNR, 'SNR')
    save_variable(sub_folder, THD, 'THD')
    save_variable(sub_folder, pwr, 'pwr')
    save_variable(sub_folder, NF, 'NF')

    # 3. Save Figure
    figure_name = f"{dataset_name}_{test_name}_python.png"
    save_fig(figures_folder, figure_name, dpi=100)
    plt.close(fig)

def test_spec_plot(project_root):
    """
    Batch runner for spec_plot (Single Channel Version).
    """
    run_unit_test_batch(
        project_root=project_root,
        input_subpath="dataset",
        test_module_name="test_spec_plot",
        file_pattern="sinewave_*.csv",        process_callback=_process_spec_plot
    )