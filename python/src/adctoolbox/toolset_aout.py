"""Run 9 analog analysis tools on calibrated ADC data."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from .validate_aout_data import validate_aout_data
from .aout.tom_decomp import tom_decomp
from .aout.spec_plot import spec_plot
from .aout.spec_plot_phase import spec_plot_phase
from .aout.err_hist_sine import err_hist_sine
from .aout.err_pdf import err_pdf
from .aout.err_auto_correlation import err_auto_correlation
from .aout.err_envelope_spectrum import err_envelope_spectrum
from .common.sine_fit import sine_fit
from .common.find_fin import find_fin


def toolset_aout(aout_data, output_dir, visible=False, resolution=11, prefix='aout'):
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

    freq_cal = find_fin(aout_data)
    full_scale = np.max(aout_data) - np.min(aout_data)

    # Tool 1: tomDecomp
    print('[1/9][tomDecomp]', end='')
    try:
        fig = plt.figure(figsize=(10, 7.5))
        signal, error, indep, dep, phi = tom_decomp(aout_data, freq_cal, order=10, disp=1)
        fig.suptitle('tomDecomp: Time-domain Error Decomposition')
        plt.gca().tick_params(labelsize=14)
        png_path = output_dir / f'{prefix}_1_tomDecomp.png'
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        if not visible:
            plt.close(fig)
        status['tools_completed'][0] = 1
        print(f' OK -> [{png_path}]')
    except Exception as e:
        print(f' FAIL {str(e)}')
        status['errors'].append(f'Tool 1: {str(e)}')

    # Tool 2: specPlot
    print('[2/9][specPlot]', end='')
    try:
        fig = plt.figure(figsize=(10, 7.5))
        enob, sndr, sfdr, snr, thd, pwr, nf, h = spec_plot(
            aout_data, label=1, harmonic=5, OSR=1, winType=4)
        plt.title('specPlot: Frequency Spectrum')
        plt.gca().tick_params(labelsize=14)
        png_path = output_dir / f'{prefix}_2_specPlot.png'
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        if not visible:
            plt.close(fig)
        status['tools_completed'][1] = 1
        print(f' OK -> [{png_path}]')
    except Exception as e:
        print(f' FAIL {str(e)}')
        status['errors'].append(f'Tool 2: {str(e)}')

    # Tool 3: specPlotPhase
    print('[3/9][specPlotPhase]', end='')
    try:
        fig = plt.figure(figsize=(10, 7.5))
        result = spec_plot_phase(aout_data, harmonic=10, show_plot=True)
        plt.title('specPlotPhase: Phase-domain Error')
        plt.gca().tick_params(labelsize=14)
        png_path = output_dir / f'{prefix}_3_specPlotPhase.png'
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        if not visible:
            plt.close(fig)
        status['tools_completed'][2] = 1
        print(f' OK -> [{png_path}]')
    except Exception as e:
        print(f' FAIL {str(e)}')
        status['errors'].append(f'Tool 3: {str(e)}')

    # Compute error data using sine_fit
    try:
        data_fit, freq_est, mag, dc, phi = sine_fit(aout_data)
        err_data = aout_data - data_fit
    except:
        err_data = aout_data - np.mean(aout_data)

    # Tool 4: errHistSine (code mode)
    print('[4/9][errHistSine (code)]', end='')
    try:
        fig = plt.figure(figsize=(10, 7.5))
        emean_code, erms_code, code_axis, _, _, _, _ = err_hist_sine(
            aout_data, bin=20, fin=freq_cal, disp=1, mode=1)
        fig.suptitle('errHistSine (code): Error Histogram by Code')
        plt.gca().tick_params(labelsize=14)
        png_path = output_dir / f'{prefix}_4_errHistSine_code.png'
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        if not visible:
            plt.close(fig)
        status['tools_completed'][3] = 1
        print(f' OK -> [{png_path}]')
    except Exception as e:
        print(f' FAIL {str(e)}')
        status['errors'].append(f'Tool 4: {str(e)}')

    # Tool 5: errHistSine (phase mode)
    print('[5/9][errHistSine (phase)]', end='')
    try:
        fig = plt.figure(figsize=(10, 7.5))
        emean, erms, phase_code, anoi, pnoi, _, _ = err_hist_sine(
            aout_data, bin=99, fin=freq_cal, disp=1, mode=0)
        fig.suptitle('errHistSine (phase): Error Histogram by Phase')
        plt.gca().tick_params(labelsize=14)
        png_path = output_dir / f'{prefix}_5_errHistSine_phase.png'
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        if not visible:
            plt.close(fig)
        status['tools_completed'][4] = 1
        print(f' OK -> [{png_path}]')
    except Exception as e:
        print(f' FAIL {str(e)}')
        status['errors'].append(f'Tool 5: {str(e)}')

    # Tool 6: errPDF
    print('[6/9][errPDF]', end='')
    try:
        fig = plt.figure(figsize=(10, 7.5))
        _, mu, sigma, kl_div, x, fx, gauss_pdf = err_pdf(
            err_data, resolution=resolution, full_scale=full_scale)
        plt.title('errPDF: Error PDF')
        plt.gca().tick_params(labelsize=14)
        png_path = output_dir / f'{prefix}_6_errPDF.png'
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        if not visible:
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
        acf, lags = err_auto_correlation(err_data, max_lag=200, normalize=True)
        plt.title('errAutoCorrelation: Error Autocorrelation')
        plt.gca().tick_params(labelsize=14)
        png_path = output_dir / f'{prefix}_7_errAutoCorrelation.png'
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        if not visible:
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
        _, _, _, _, _, _, _, h = spec_plot(err_data, label=0)
        plt.title('errSpectrum: Error Spectrum')
        plt.gca().tick_params(labelsize=14)
        png_path = output_dir / f'{prefix}_8_errSpectrum.png'
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        if not visible:
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
        err_envelope_spectrum(err_data, fs=1)
        plt.title('errEnvelopeSpectrum: Error Envelope Spectrum')
        plt.gca().tick_params(labelsize=14)
        png_path = output_dir / f'{prefix}_9_errEnvelopeSpectrum.png'
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        if not visible:
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
