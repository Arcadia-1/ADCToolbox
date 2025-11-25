"""
Test Error Analysis Functions - Python version of test_error_analysis.m

This script analyzes various ADC error types using:
- spec_plot: Spectrum analysis
- errPDF: Error probability density function with KDE
- errAutoCorrelation: Error autocorrelation function
- errEnvelopeSpectrum: Envelope spectrum via Hilbert transform

Usage:
    Run: python test_error_analysis.py
    Auto-discovers all sinewave_*.csv files in test_data/
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from adctoolbox.common import sine_fit
from adctoolbox.aout import spec_plot, errPDF, errAutoCorrelation, errEnvelopeSpectrum

# Get project root directory
project_root = Path(__file__).parent.parent.parent.parent

# Configuration
SAMPLING_FREQ = 1e9  # 1 GHz
ACF_MAX_LAG = 300


def analyze_data_file(data_file_path, Fs=SAMPLING_FREQ, MaxLag=ACF_MAX_LAG):
    """
    Perform complete error analysis on a single data file.

    Parameters
    ----------
    data_file_path : Path
        Path to the CSV data file
    Fs : float
        Sampling frequency in Hz
    MaxLag : int
        Maximum lag for autocorrelation
    """
    # Extract dataset name
    dataset_name = data_file_path.stem

    # Create output directory
    output_dir = project_root / "test_output" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Processing: {data_file_path.name}")
    print(f"{'='*70}")

    # Load data
    try:
        data = np.loadtxt(data_file_path, delimiter=',')
        print(f"[Loaded] {len(data)} samples from {data_file_path.name}")
    except Exception as e:
        print(f"[ERROR] Failed to load {data_file_path.name}: {e}")
        return

    # Fit sine wave and extract error
    try:
        data_fit, freq_est, mag, dc, phi = sine_fit(data)
        err_data = data - data_fit
        print(f"[sineFit] freq={freq_est:.6f}, mag={mag:.4f}, dc={dc:.4f}, phi={phi:.4f}")
    except Exception as e:
        print(f"[ERROR] Sine fit failed: {e}")
        return

    # ========================================================================
    # Test 1: Spectrum Plot
    # ========================================================================
    try:
        plt.figure(figsize=(12, 8))
        spec_plot(data, label=1, Fs=Fs)

        output_file = output_dir / f"spectrum_of_{dataset_name}_python.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] Spectrum plot saved: {output_file.name}")
    except Exception as e:
        print(f"[ERROR] Spectrum plot failed: {e}")

    # ========================================================================
    # Test 2: Error PDF
    # ========================================================================
    try:
        plt.figure(figsize=(12, 8))
        noise_lsb, mu, sigma, KL_divergence, x, fx, gauss_pdf = errPDF(err_data)

        output_file = output_dir / f"errPDF_of_{dataset_name}_python.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] Error PDF saved: {output_file.name}")
        print(f"    KL_divergence={KL_divergence:.4f}, μ={mu:.2f}, σ={sigma:.2f}")
    except Exception as e:
        print(f"[ERROR] Error PDF failed: {e}")

    # ========================================================================
    # Test 3: Error Autocorrelation
    # ========================================================================
    try:
        plt.figure(figsize=(12, 8))
        acf, lags = errAutoCorrelation(err_data, MaxLag=MaxLag)

        output_file = output_dir / f"errACF_of_{dataset_name}_python.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] Error ACF saved: {output_file.name}")
    except Exception as e:
        print(f"[ERROR] Error autocorrelation failed: {e}")

    # ========================================================================
    # Test 4: Error Spectrum
    # ========================================================================
    try:
        plt.figure(figsize=(12, 8))
        spec_plot(err_data, label=0, Fs=Fs)
        plt.title("Error Spectrum")

        output_file = output_dir / f"errSpectrum_of_{dataset_name}_python.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] Error spectrum saved: {output_file.name}")
    except Exception as e:
        print(f"[ERROR] Error spectrum failed: {e}")

    # ========================================================================
    # Test 5: Error Envelope Spectrum
    # ========================================================================
    try:
        plt.figure(figsize=(12, 8))
        errEnvelopeSpectrum(err_data, Fs=Fs)

        output_file = output_dir / f"errEnvelopeSpectrum_of_{dataset_name}_python.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] Error envelope spectrum saved: {output_file.name}")
    except Exception as e:
        print(f"[ERROR] Error envelope spectrum failed: {e}")

    print(f"[Done] All plots saved to: {output_dir}")


def main():
    """Test error analysis functions on multiple datasets."""

    input_dir = Path("test_data")
    output_dir = Path("test_output")

    # Test datasets - leave empty to auto-search
    files_list = [
        # Uncomment to test specific files:
        # "sinewave_glitch_0P100.csv",
    ]

    # Auto-search if list is empty
    if not files_list:
        search_patterns = ['sinewave_*.csv']
        files_list = []
        for pattern in search_patterns:
            files_list.extend([f for f in input_dir.glob(pattern)])
        print(f"Auto-discovered {len(files_list)} files matching patterns: {', '.join(search_patterns)}")
    else:
        files_list = [input_dir / f for f in files_list]

    if not files_list:
        raise ValueError(f"No test files found in {input_dir}")

    output_dir.mkdir(exist_ok=True)

    # Test Loop
    print("=== test_error_analysis.py ===")
    print(f"Testing error analysis functions with {len(files_list)} datasets...")
    print(f"Sampling frequency: {SAMPLING_FREQ/1e9:.1f} GHz")
    print(f"ACF max lag: {ACF_MAX_LAG} samples\n")

    for k, data_file_path in enumerate(files_list, 1):
        if not data_file_path.is_file():
            print(f"[{k}/{len(files_list)}] {data_file_path.name} - NOT FOUND, skipping\n")
            continue

        print(f"[{k}/{len(files_list)}] {data_file_path.name}")
        analyze_data_file(data_file_path, Fs=SAMPLING_FREQ, MaxLag=ACF_MAX_LAG)

    print(f"\n{'='*70}")
    print(f"[Complete] Processed {len(files_list)} file(s)")
    print(f"{'='*70}\n")

    # Close any remaining figures
    plt.close('all')


if __name__ == "__main__":
    main()
