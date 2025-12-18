"""Run 9 analog analysis tools on calibrated ADC data."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox.common.validate import validate_aout_data
from adctoolbox.aout.decompose_harmonics import decompose_harmonics
from adctoolbox.spectrum import analyze_spectrum, analyze_phase_spectrum
from adctoolbox.aout.rearrange_error_by_value import rearrange_error_by_value
from adctoolbox.aout.plot_rearranged_error_by_value import plot_rearranged_error_by_value
from adctoolbox.aout.plot_error_hist_phase import plot_error_hist_phase
from adctoolbox.aout.plot_error_pdf import plot_error_pdf
from adctoolbox.aout.plot_error_autocorr import plot_error_autocorr
from adctoolbox.aout.plot_envelope_spectrum import plot_envelope_spectrum
from adctoolbox.aout.fit_sine_4param import fit_sine_4param as fit_sine
from adctoolbox.common.estimate_frequency import estimate_frequency


def generate_aout_dashboard(aout_data, output_dir, visible=False, resolution=11, prefix='aout'):
    """
    Run 9 analog analysis tools on calibrated ADC data.

    Parameters
    ----------
    aout_data : array_like
        Analog output signal (1D vector)
    output_dir : str or Path
        Directory to save output figures
    visible : bool, optional
        Show figures (default: False)
    resolution : int, optional
        ADC resolution in bits (default: 11)
    prefix : str, optional
        Filename prefix (default: 'aout')

    Returns
    -------
    status : dict
        Dictionary with fields:
        - success : bool (True if all tools completed)
        - tools_completed : list of 9 success flags
        - errors : list of error messages
        - panel_path : path to panel figure
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    status = {
        'success': False,
        'tools_completed': [0] * 9,
        'errors': [],
        'panel_path': ''
    }

    # Validate input data
    print('[Validation]', end='')
    try:
        validate_aout_data(aout_data)
        print(' OK')
    except Exception as e:
        print(f' FAIL {str(e)}')
        raise ValueError(f'Input validation failed: {str(e)}')

    # Handle multirun data (take first row if 2D)
    aout_data = np.asarray(aout_data)
    if aout_data.ndim > 1:
        aout_data = aout_data[0, :]

    freq_cal = estimate_frequency(aout_data)
    full_scale = np.max(aout_data) - np.min(aout_data)

    # Tool 1: Harmonic Decomposition
    print('[1/9][Harmonic Decomposition]', end='')
    try:
        fundamental, total_error, harmonic_error, other_error = decompose_harmonics(aout_data, freq_cal, order=10, show_plot=True)
        plt.gca().tick_params(labelsize=14)
        png_path = output_dir / f'{prefix}_1_harmonicDecomp.png'
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()
        status['tools_completed'][0] = 1
        print(f' OK -> [{png_path}]')
    except Exception as e:
        print(f' FAIL {str(e)}')
        status['errors'].append(f'Tool 1: {str(e)}')

    # Tool 2: specPlot
    print('[2/9][specPlot]', end='')
    try:
        fig = plt.figure(figsize=(10, 7.5))
        result = analyze_spectrum(
            aout_data, label=1, harmonic=5, osr=1, win_type='boxcar')
        plt.title('specPlot: Frequency Spectrum')
        plt.gca().tick_params(labelsize=14)
        png_path = output_dir / f'{prefix}_2_specPlot.png'
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        status['tools_completed'][1] = 1
        print(f' OK -> [{png_path}]')
    except Exception as e:
        print(f' FAIL {str(e)}')
        status['errors'].append(f'Tool 2: {str(e)}')

    # Tool 3: specPlotPhase
    print('[3/9][specPlotPhase]', end='')
    try:
        png_path = output_dir / f'{prefix}_3_specPlotPhase.png'
        result = analyze_phase_spectrum(aout_data, harmonic=10, show_plot=False, save_path=str(png_path))
        status['tools_completed'][2] = 1
        print(f' OK -> [{png_path}]')
    except Exception as e:
        print(f' FAIL {str(e)}')
        status['errors'].append(f'Tool 3: {str(e)}')

    # Compute error data using fit_sine
    try:
        data_fit, freq_est, mag, dc, phi = fit_sine(aout_data)
        err_data = aout_data - data_fit
    except:
        err_data = aout_data - np.mean(aout_data)

    # Tool 4: Error Histogram (value mode)
    print('[4/9][Error Histogram (value)]', end='')
    try:
        results = rearrange_error_by_value(aout_data, normalized_freq=freq_cal, num_bits=resolution, num_bins=20)
        fig = plt.figure(figsize=(12, 8))
        plot_rearranged_error_by_value(results)
        png_path = output_dir / f'{prefix}_4_errHistSine_code.png'
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        status['tools_completed'][3] = 1
        print(f' OK -> [{png_path}]')
    except Exception as e:
        print(f' FAIL {str(e)}')
        status['errors'].append(f'Tool 4: {str(e)}')

    # Tool 5: Error Histogram (phase mode)
    print('[5/9][Error Histogram (phase)]', end='')
    try:
        error_mean, error_rms, phase_bins, amplitude_noise, phase_noise, error, phase = plot_error_hist_phase(
            aout_data, bins=99, freq=freq_cal, disp=1)
        png_path = output_dir / f'{prefix}_5_errHistSine_phase.png'
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()
        status['tools_completed'][4] = 1
        print(f' OK -> [{png_path}]')
    except Exception as e:
        print(f' FAIL {str(e)}')
        status['errors'].append(f'Tool 5: {str(e)}')

    # Tool 6: errPDF
    print('[6/9][errPDF]', end='')
    try:
        fig = plt.figure(figsize=(10, 7.5))
        _, mu, sigma, kl_div, x, fx, gauss_pdf = plot_error_pdf(
            err_data, resolution=resolution, full_scale=full_scale)
        plt.title('errPDF: Error PDF')
        plt.gca().tick_params(labelsize=14)
        png_path = output_dir / f'{prefix}_6_errPDF.png'
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        status['tools_completed'][5] = 1
        print(f' OK -> [{png_path}]')
    except Exception as e:
        print(f' FAIL {str(e)}')
        status['errors'].append(f'Tool 6: {str(e)}')

    # Tool 7: errAutoCorrelation
    print('[7/9][errAutoCorrelation]', end='')
    try:
        fig = plt.figure(figsize=(10, 7.5))
        acf, lags = plot_error_autocorr(err_data, max_lag=200, normalize=True)
        plt.title('errAutoCorrelation: Error Autocorrelation')
        plt.gca().tick_params(labelsize=14)
        png_path = output_dir / f'{prefix}_7_errAutoCorrelation.png'
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        status['tools_completed'][6] = 1
        print(f' OK -> [{png_path}]')
    except Exception as e:
        print(f' FAIL {str(e)}')
        status['errors'].append(f'Tool 7: {str(e)}')

    # Tool 8: Error Spectrum
    print('[8/9][errSpectrum]', end='')
    try:
        fig = plt.figure(figsize=(10, 7.5))
        result = analyze_spectrum(err_data, label=0)
        plt.title('errSpectrum: Error Spectrum')
        plt.gca().tick_params(labelsize=14)
        png_path = output_dir / f'{prefix}_8_errSpectrum.png'
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        status['tools_completed'][7] = 1
        print(f' OK -> [{png_path}]')
    except Exception as e:
        print(f' FAIL {str(e)}')
        status['errors'].append(f'Tool 8: {str(e)}')

    # Tool 9: errEnvelopeSpectrum
    print('[9/9][errEnvelopeSpectrum]', end='')
    try:
        fig = plt.figure(figsize=(10, 7.5))
        result = plot_envelope_spectrum(err_data, fs=1)
        plt.title('errEnvelopeSpectrum: Error Envelope Spectrum')
        plt.gca().tick_params(labelsize=14)
        png_path = output_dir / f'{prefix}_9_errEnvelopeSpectrum.png'
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        status['tools_completed'][8] = 1
        print(f' OK -> [{png_path}]')
    except Exception as e:
        print(f' FAIL {str(e)}')
        status['errors'].append(f'Tool 9: {str(e)}')

    # Create Panel Overview (3x3 grid)
    print('[Panel]', end='')
    try:
        plot_files = [
            output_dir / f'{prefix}_1_tomDecomp.png',
            output_dir / f'{prefix}_2_specPlot.png',
            output_dir / f'{prefix}_3_specPlotPhase.png',
            output_dir / f'{prefix}_4_errHistSine_code.png',
            output_dir / f'{prefix}_5_errHistSine_phase.png',
            output_dir / f'{prefix}_6_errPDF.png',
            output_dir / f'{prefix}_7_errAutoCorrelation.png',
            output_dir / f'{prefix}_8_errSpectrum.png',
            output_dir / f'{prefix}_9_errEnvelopeSpectrum.png',
        ]

        plot_labels = [
            '(1) tomDecomp',
            '(2) specPlot',
            '(3) specPlotPhase',
            '(4) errHistSine (code)',
            '(5) errHistSine (phase)',
            '(6) errPDF',
            '(7) errAutoCorrelation',
            '(8) errSpectrum',
            '(9) errEnvelopeSpectrum',
        ]

        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        axes = axes.flatten()

        for p, (img_path, label) in enumerate(zip(plot_files, plot_labels)):
            ax = axes[p]
            if img_path.exists():
                img = plt.imread(img_path)
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(label, fontsize=12)
            else:
                ax.text(0.5, 0.5, f'Missing:\n{label}',
                        ha='center', va='center', fontsize=10, color='red')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                ax.set_title(label, fontsize=12, color='red')

        fig.suptitle('AOUT Toolset Overview', fontsize=16, fontweight='bold')
        plt.tight_layout()

        panel_path = output_dir / f'PANEL_{prefix.upper()}.png'
        plt.savefig(panel_path, dpi=150, bbox_inches='tight')
        if not visible:
            plt.close(fig)
        status['panel_path'] = str(panel_path)
        print(f' OK -> [{panel_path}]')
    except Exception as e:
        print(f' FAIL {str(e)}')
        status['errors'].append(f'Panel: {str(e)}')

    # Final status
    n_success = sum(status['tools_completed'])
    print(f'=== Toolset complete: {n_success}/9 tools succeeded ===\n')
    status['success'] = (n_success == 9)

    return status
