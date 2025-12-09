import numpy as np
import matplotlib.pyplot as plt

from adctoolbox.dout import calibrate_weight_sine
from adctoolbox.aout import analyze_spectrum
from tests._utils import save_variable, save_fig
from tests.unit._runner import run_unit_test_batch
from tests import config

plt.rcParams['font.size'] = 14
plt.rcParams['axes.grid'] = True

def _process_calibrate_weight_sine(raw_data, sub_folder, dataset_name, figures_folder, test_name):
    """
    Callback function to process a single file:
    1. Calculate pre-calibration signal using nominal binary weights
    2. Run foreground calibration
    3. Plot and save spectrum before calibration
    4. Plot and save spectrum after calibration
    5. Save calibrated weights, offset, frequency, waveforms, and ENoB metrics
    """
    N, M = raw_data.shape

    # Calculate nominal binary weights
    nomWeight = 2.0 ** np.arange(M - 1, -1, -1)

    # Pre-calibration: Convert using nominal weights
    preCal = raw_data @ nomWeight

    # Run calibrate_weight_sine
    weight, offset, postCal, ideal, err, freqCal = calibrate_weight_sine(
        raw_data,
        freq=0,
        order=5
    )

    # Spectrum plot BEFORE calibration (using nominal weights)
    fig = plt.figure(figsize=(12, 8))
    result_pre = analyze_spectrum(
        preCal,
        show_label=True,
        n_thd=5,
        osr=1,
        nf_method=0
    )
    ENoB_pre = result_pre['enob']
    SNDR_pre = result_pre['sndr_db']
    SFDR_pre = result_pre['sfdr_db']
    SNR_pre = result_pre['snr_db']
    THD_pre = result_pre['thd_db']
    pwr_pre = result_pre['sig_pwr_dbfs']
    NF_pre = result_pre['noise_floor_db']
    NSD_pre = result_pre['nsd_dbfs_hz']
    plt.title(f'Spectrum Before Calibration: {dataset_name}')
    figure_name_preCal = f"{test_name}_{dataset_name}_preCal_python.png"
    save_fig(figures_folder, figure_name_preCal, dpi=100)
    plt.close(fig)

    # Spectrum plot AFTER calibration
    fig = plt.figure(figsize=(12, 8))
    result_post = analyze_spectrum(
        postCal,
        show_label=True,
        n_thd=5,
        osr=1,
        nf_method=0
    )
    ENoB_post = result_post['enob']
    SNDR_post = result_post['sndr_db']
    SFDR_post = result_post['sfdr_db']
    SNR_post = result_post['snr_db']
    THD_post = result_post['thd_db']
    pwr_post = result_post['sig_pwr_dbfs']
    NF_post = result_post['noise_floor_db']
    NSD_post = result_post['nsd_dbfs_hz']
    plt.title(f'Spectrum After Calibration: {dataset_name}')
    figure_name_postCal = f"{test_name}_{dataset_name}_postCal_python.png"
    save_fig(figures_folder, figure_name_postCal, dpi=100)
    plt.close(fig)

    # Save variables
    save_variable(sub_folder, weight, 'weight')
    save_variable(sub_folder, offset, 'offset')
    save_variable(sub_folder, postCal, 'postCal')
    save_variable(sub_folder, ideal, 'ideal')
    save_variable(sub_folder, err, 'err')
    save_variable(sub_folder, freqCal, 'freqCal')

    save_variable(sub_folder, ENoB_pre, 'ENoB_pre')
    save_variable(sub_folder, ENoB_post, 'ENoB_post')

def test_calibrate_weight_sine(project_root):
    """
    Batch runner for foreground calibration sine test.
    """
    run_unit_test_batch(
        project_root=project_root,
        input_subpath=config.DOUT['input_path'], test_module_name="test_calibrate_weight_sine", file_pattern=config.DOUT['file_pattern'],        process_callback=_process_calibrate_weight_sine,
        flatten=False  # Digital output data is 2D (N samples x M bits)
    )
