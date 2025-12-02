import numpy as np
import matplotlib.pyplot as plt

from adctoolbox.dout import fg_cal_sine
from adctoolbox.aout import spec_plot
from tests._utils import save_variable, save_fig
from tests.unit._runner import run_unit_test_batch

plt.rcParams['font.size'] = 14
plt.rcParams['axes.grid'] = True

def _process_fg_cal_sine(raw_data, sub_folder, dataset_name, figures_folder, test_name):
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

    # Run FGCalSine
    weight, offset, postCal, ideal, err, freqCal = fg_cal_sine(
        raw_data,
        freq=0,
        order=5
    )

    # Spectrum plot BEFORE calibration (using nominal weights)
    fig = plt.figure(figsize=(12, 8))
    ENoB_pre, SNDR_pre, SFDR_pre, SNR_pre, THD_pre, pwr_pre, NF_pre, _ = spec_plot(
        preCal,
        label=1,
        harmonic=5,
        OSR=1,
        NFMethod=0
    )
    plt.title(f'Spectrum Before Calibration: {dataset_name}')
    figure_name_preCal = f"{dataset_name}_{test_name}_preCal_python.png"
    save_fig(figures_folder, figure_name_preCal, dpi=100)
    plt.close(fig)

    # Spectrum plot AFTER calibration
    fig = plt.figure(figsize=(12, 8))
    ENoB_post, SNDR_post, SFDR_post, SNR_post, THD_post, pwr_post, NF_post, _ = spec_plot(
        postCal,
        label=1,
        harmonic=5,
        OSR=1,
        NFMethod=0
    )
    plt.title(f'Spectrum After Calibration: {dataset_name}')
    figure_name_postCal = f"{dataset_name}_{test_name}_postCal_python.png"
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

def test_fg_cal_sine(project_root):
    """
    Batch runner for foreground calibration sine test.
    """
    run_unit_test_batch(
        project_root=project_root,
        input_subpath="dataset/dout",
        test_module_name="test_fg_cal_sine",
        file_pattern="dout_*.csv",
        output_subpath="test_output",
        process_callback=_process_fg_cal_sine,
        flatten=False  # Digital output data is 2D (N samples x M bits)
    )
