"""
Error histogram in code domain for INL/DNL and static nonlinearity analysis.

Bins errors by ADC code value to reveal static transfer function characteristics.

MATLAB counterpart: errHistSine.m (code mode)
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_error_hist_code(data, bins=100, freq=0, disp=1, error_range=None):
    """
    Error histogram in code domain - for static nonlinearity analysis.

    Parameters:
        data: ADC output data (1D array)
        bins: Number of bins (default: 100)
        freq: Normalized frequency (0-1), 0 = auto detect (default: 0)
        disp: Display plots (1=yes, 0=no) (default: 1)
        error_range: Error range filter [min, max] (default: None)

    Returns:
        error_mean: Mean error per bin
        error_rms: RMS error per bin
        code_bins: Code positions (bin centers)
        error: Raw error signal
        codes: Code values corresponding to raw error

    Notes:
        This function bins errors by ADC code value to reveal static
        transfer function characteristics like INL/DNL.
    """
    # Ensure data is row vector
    data = np.asarray(data).flatten()
    N = len(data)

    # Sine fit to get ideal signal and error
    from adctoolbox.aout.fit_sine_4param import fit_sine_4param
    if freq == 0:
        fit_result = fit_sine_4param(data)
        data_fit = fit_result['fitted_signal']
        freq = fit_result['frequency']
    else:
        fit_result = fit_sine_4param(data, freq)
        data_fit = fit_result['fitted_signal']

    # Error = ideal - actual
    error = data_fit - data

    # Code mode - bin by ADC code value
    codes = data
    code_min = np.min(data)
    code_max = np.max(data)
    bin_width = (code_max - code_min) / bins
    code_bins = code_min + np.arange(1, bins+1) * bin_width - bin_width/2

    bin_count = np.zeros(bins)
    error_sum = np.zeros(bins)
    error_rms = np.zeros(bins)

    # Binning for mean
    for ii in range(N):
        b = min(int(np.floor((data[ii] - code_min) / bin_width)), bins-1)
        error_sum[b] += error[ii]
        bin_count[b] += 1

    error_mean = error_sum / bin_count

    # Binning for RMS (total RMS from sine fit)
    for ii in range(N):
        b = min(int(np.floor((data[ii] - code_min) / bin_width)), bins-1)
        error_rms[b] += error[ii]**2

    error_rms = np.sqrt(error_rms / bin_count)

    # Filter error range if specified
    if error_range is not None:
        eid = (codes >= error_range[0]) & (codes <= error_range[1])
        codes = codes[eid]
        error = error[eid]

    # Plotting
    if disp:
        fig = plt.figure(figsize=(10, 8))
        ax1 = plt.subplot(2, 1, 1)
        ax2 = plt.subplot(2, 1, 2)

        ax1.plot(data, error, 'r.', markersize=2, label='Raw error')
        ax1.plot(code_bins, error_mean, 'b-', linewidth=2, label='Mean error')

        ax1.set_xlim([code_min, code_max])
        ax1.set_ylim([np.min(error), np.max(error)])
        ax1.set_ylabel('Error')
        ax1.set_xlabel('Code')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        if error_range is not None:
            ax1.plot(codes, error, 'm.', markersize=2)

        ax2.bar(code_bins, error_rms, width=bin_width*0.8, color='skyblue')
        ax2.set_xlim([code_min, code_max])
        ax2.set_ylim([0, np.max(error_rms)*1.1])
        ax2.set_xlabel('Code')
        ax2.set_ylabel('RMS Error')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

    return error_mean, error_rms, code_bins, error, codes
