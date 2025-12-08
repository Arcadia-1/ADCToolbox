"""Run digital analysis tools on ADC digital output."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ..common.validate import validate_dout_data
from ..spectrum import analyze_spectrum
from .calibrate_weight_sine import calibrate_weight_sine
from .check_overflow import check_overflow
from .check_bit_activity import check_bit_activity
from .plot_weight_radix import plot_weight_radix
from .analyze_enob_sweep import analyze_enob_sweep


def generate_dout_dashboard(bits, output_dir, visible=False, order=5, prefix='dout'):
    """
    Run 6 digital analysis tools on ADC digital output.

    Parameters
    ----------
    bits : array_like
        Digital bits (N samples x B bits, MSB to LSB)
    output_dir : str or Path
        Directory to save output figures
    visible : bool, optional
        Show figures (default: False)
    order : int, optional
        Polynomial order for calibration (default: 5)
    prefix : str, optional
        Filename prefix (default: 'dout')

    Returns
    -------
    status : dict
        Dictionary with fields:
        - success : bool (True if all tools completed)
        - tools_completed : list of success flags
        - errors : list of error messages
        - panel_path : path to panel figure
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    status = {
        'success': False,
        'tools_completed': [0] * 6,
        'errors': [],
        'panel_path': ''
    }

    print('\n=== Running DOUT Toolset (6 Tools) ===')

    # Validate input data
    print('[Validation]', end='')
    try:
        validate_dout_data(bits)
        print(' OK')
    except Exception as e:
        print(f' FAIL {str(e)}')
        raise ValueError(f'Input validation failed: {str(e)}')

    bits = np.asarray(bits)
    n_bits = bits.shape[1]
    print(f'Resolution: {n_bits} bits')

    nominal_weights = 2.0 ** np.arange(n_bits - 1, -1, -1)

    # Tool 1: Digital Spectrum with Nominal Weights
    print('[1/3][Spectrum (Nominal)]', end='')
    try:
        digital_codes_nominal = bits @ nominal_weights
        fig = plt.figure(figsize=(10, 7.5))
        result_nom = analyze_spectrum(
            digital_codes_nominal, label=1, harmonic=5, osr=1, win_type='boxcar')
        plt.title('Digital Spectrum: Nominal Weights')
        plt.gca().tick_params(labelsize=16)
        png_path = output_dir / f'{prefix}_1_spectrum_nominal.png'
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        status['tools_completed'][0] = 1
        print(f' OK -> [{png_path}]')
    except Exception as e:
        print(f' FAIL {str(e)}')
        status['errors'].append(f'Tool 1: {str(e)}')

    # Tool 2: Digital Spectrum with Calibrated Weights
    print('[2/3][Spectrum (Calibrated)]', end='')
    try:
        weight_cal, offset, k_static, residual, cost, freq_cal = calibrate_weight_sine(
            bits, freq=0, order=order)
        digital_codes_calibrated = bits @ weight_cal
        fig = plt.figure(figsize=(10, 7.5))
        result_cal = analyze_spectrum(
            digital_codes_calibrated, label=1, harmonic=5, osr=1, win_type='boxcar')
        plt.title('Digital Spectrum: Calibrated Weights')
        plt.gca().tick_params(labelsize=16)
        png_path = output_dir / f'{prefix}_2_spectrum_calibrated.png'
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        status['tools_completed'][1] = 1
        improvement = result_cal['enob'] - result_nom['enob']
        print(f' OK (+{improvement:.2f} ENoB) -> [{png_path}]')
    except Exception as e:
        print(f' FAIL {str(e)}')
        status['errors'].append(f'Tool 2: {str(e)}')

    # Tool 3: Overflow Check
    print('[3/3][Overflow Check]', end='')
    try:
        if 'weight_cal' not in locals():
            weight_cal, _, _, _, _, _ = calibrate_weight_sine(bits, freq=0, order=order)
        data_decom = check_overflow(bits, weight_cal, disp=True)
        png_path = output_dir / f'{prefix}_3_overflowChk.png'
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()
        status['tools_completed'][2] = 1
        print(f' OK → [{png_path}]')
    except Exception as e:
        print(f' FAIL {str(e)}')
        status['errors'].append(f'Tool 3: {str(e)}')

    # Tool 4: Bit Activity
    print('[4/6][Bit Activity]', end='')
    try:
        fig = plt.figure(figsize=(10, 7.5))
        bit_usage = check_bit_activity(bits)
        plt.title('Bit Activity')
        plt.gca().tick_params(labelsize=16)
        png_path = output_dir / f'{prefix}_4_bitActivity.png'
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        status['tools_completed'][3] = 1
        print(f' OK → [{png_path}]')
    except Exception as e:
        print(f' FAIL {str(e)}')
        status['errors'].append(f'Tool 4: {str(e)}')

    # Tool 5: Weight Scaling
    print('[5/6][Weight Scaling]', end='')
    try:
        fig = plt.figure(figsize=(8, 6))
        radix = plot_weight_radix(weight_cal)
        plt.gca().tick_params(labelsize=16)
        png_path = output_dir / f'{prefix}_5_weightScaling.png'
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        status['tools_completed'][4] = 1
        print(f' OK → [{png_path}]')
    except Exception as e:
        print(f' FAIL {str(e)}')
        status['errors'].append(f'Tool 5: {str(e)}')

    # Tool 6: ENOB Bit Sweep
    print('[6/6][ENOB Bit Sweep]', end='')
    try:
        fig = plt.figure(figsize=(10, 7.5))
        enob_sweep, n_bits_vec = analyze_enob_sweep(
            bits, freq=0, order=order, harmonic=5, osr=1, win_type='hamming', plot=True)
        plt.gca().tick_params(labelsize=16)
        png_path = output_dir / f'{prefix}_6_enobBitSweep.png'
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        status['tools_completed'][5] = 1
        print(f' OK → [{png_path}]')
    except Exception as e:
        print(f' FAIL {str(e)}')
        status['errors'].append(f'Tool 6: {str(e)}')

    # Create Panel Overview (2x3 grid)
    print('[Panel]', end='')
    try:
        plot_files = [
            output_dir / f'{prefix}_1_spectrum_nominal.png',
            output_dir / f'{prefix}_2_spectrum_calibrated.png',
            output_dir / f'{prefix}_3_overflowChk.png',
            output_dir / f'{prefix}_4_bitActivity.png',
            output_dir / f'{prefix}_5_weightScaling.png',
            output_dir / f'{prefix}_6_enobBitSweep.png',
        ]

        plot_labels = [
            '(1) Nominal Weights',
            '(2) Calibrated Weights',
            '(3) Overflow Check',
            '(4) Bit Activity',
            '(5) Weight Scaling',
            '(6) ENOB Bit Sweep',
        ]

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
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

        fig.suptitle('DOUT Toolset Overview', fontsize=16, fontweight='bold')
        plt.tight_layout()

        panel_path = output_dir / f'PANEL_{prefix.upper()}.png'
        plt.savefig(panel_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        status['panel_path'] = str(panel_path)
        print(f' OK → [{panel_path}]')
    except Exception as e:
        print(f' FAIL {str(e)}')
        status['errors'].append(f'Panel: {str(e)}')

    # Final status
    n_success = sum(status['tools_completed'])
    print(f'=== Toolset complete: {n_success}/6 tools succeeded ===\n')
    status['success'] = (n_success == 6)

    return status
