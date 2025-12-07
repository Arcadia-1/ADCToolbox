import matplotlib.pyplot as plt

from adctoolbox.aout import spec_plot
from tests._utils import save_variable, save_fig
from tests.unit._runner import run_unit_test_batch
from tests import config

plt.rcParams['font.size'] = 14
plt.rcParams['axes.grid'] = True

def _process_spec_plot(raw_data, sub_folder, dataset_name, figures_folder, test_name):
    # 1. Spectral Analysis - using Pythonic names
    fig = plt.figure(figsize=(8, 6))
    enob, sndr, sfdr, snr, thd, signal_power, noise_floor, noise_spectral_density = spec_plot(
        raw_data,
        label=1,
        harmonic=5,
        osr=1,
        nf_method=0
    )
    plt.title("Spectrum")

    # 2. Save Variables - auto-mapped to MATLAB names
    save_variable(sub_folder, enob, 'enob')                                        # → enob_python.csv
    save_variable(sub_folder, sndr, 'sndr')                                        # → sndr_python.csv
    save_variable(sub_folder, sfdr, 'sfdr')                                        # → sfdr_python.csv
    save_variable(sub_folder, snr, 'snr')                                          # → snr_python.csv
    save_variable(sub_folder, thd, 'thd')                                          # → thd_python.csv
    save_variable(sub_folder, signal_power, 'signal_power')                        # → sigpwr_python.csv
    save_variable(sub_folder, noise_floor, 'noise_floor')                          # → noi_python.csv
    save_variable(sub_folder, noise_spectral_density, 'noise_spectral_density')    # → nsd_python.csv

    # 3. Save Figure
    figure_name = f"{test_name}_{dataset_name}_python.png"
    save_fig(figures_folder, figure_name, dpi=100)
    plt.close(fig)

def test_spec_plot(project_root):
    """
    Batch runner for spec_plot (Single Channel Version).
    """
    run_unit_test_batch(
        project_root=project_root,
        input_subpath=config.AOUT['input_path'], test_module_name="test_spec_plot", file_pattern=config.AOUT['file_pattern'],        process_callback=_process_spec_plot
    )