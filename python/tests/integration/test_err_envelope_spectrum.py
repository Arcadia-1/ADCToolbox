import matplotlib.pyplot as plt

from adctoolbox.common import fit_sine
from adctoolbox.aout import plot_envelope_spectrum
from tests._utils import save_fig, save_variable
from tests.unit._runner import run_unit_test_batch
from tests import config

plt.rcParams['font.size'] = 14
plt.rcParams['axes.grid'] = True

def _process_plot_envelope_spectrum(raw_data, sub_folder, dataset_name, figures_folder, test_name):
    """
    Callback function to process a single file:
    1. Calculate error data using fit_sine
    2. Run error envelope spectrum analysis
    3. Save variables
    4. Save plot
    """
    # Compute error data using sineFit
    fit_result = fit_sine(raw_data)
    fitted_signal = fit_result['fitted_signal']
    err_data = raw_data - fitted_signal

    # Run plot_envelope_spectrum
    plt.figure(figsize=(12, 8))
    result = plot_envelope_spectrum(err_data, fs=1)
    enob = result['enob']
    sndr = result['sndr_db']
    sfdr = result['sfdr_db']
    snr = result['snr_db']
    thd = result['thd_db']
    signal_power = result['sig_pwr_dbfs']
    noise_floor = result['noise_floor_db']
    plt.title(f'errEnvelopeSpectrum: {dataset_name}')

    # Save variables with MATLAB-compatible names
    # Note: err_envelope_spectrum uses different naming than spec_plot
    save_variable(sub_folder, enob, 'ENoB')        # Capital ENoB
    save_variable(sub_folder, sndr, 'SNDR')        # Capital SNDR
    save_variable(sub_folder, sfdr, 'SFDR')        # Capital SFDR
    save_variable(sub_folder, snr, 'SNR')          # Capital SNR
    save_variable(sub_folder, thd, 'THD')          # Capital THD
    save_variable(sub_folder, signal_power, 'pwr') # pwr instead of sigpwr
    save_variable(sub_folder, noise_floor, 'NF')   # NF instead of noi

    # Save plot
    figure_name = f"{test_name}_{dataset_name}_python.png"
    save_fig(figures_folder, figure_name, dpi=150)

def test_plot_envelope_spectrum(project_root):
    """
    Batch runner for error envelope spectrum analysis.
    """
    run_unit_test_batch(
        project_root=project_root,
        input_subpath=config.AOUT['input_path'], test_module_name="test_plot_envelope_spectrum", file_pattern=config.AOUT['file_pattern'],        process_callback=_process_plot_envelope_spectrum
    )
