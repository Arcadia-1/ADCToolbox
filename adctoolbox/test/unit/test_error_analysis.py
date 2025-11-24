"""
Test Error Analysis Functions - Python version of test_error_analysis.m

This script analyzes various ADC error types using:
- spec_plot: Spectrum analysis
- errPDF: Error probability density function with KDE
- errAutoCorrelation: Error autocorrelation function
- errEnvelopeSpectrum: Envelope spectrum via Hilbert transform

Usage:
    1. Enable/disable data files in DATA_FILES list below
    2. Run: python test_error_analysis.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add project root to sys.path if needed (for direct script execution)
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from ADCToolbox_Python.sineFit import sine_fit
from ADCToolbox_Python.spec_plot import spec_plot
from ADCToolbox_Python.errPDF import errPDF
from ADCToolbox_Python.errAutoCorrelation import errAutoCorrelation
from ADCToolbox_Python.errEnvelopeSpectrum import errEnvelopeSpectrum


# ============================================================================
# DATA FILE SELECTION - Enable/disable by commenting/uncommenting
# ============================================================================
DATA_FILES = [
    # Glitch errors
    "sinewave_glitch_0P100.csv",         # Enabled by default
    "sinewave_glitch_0P050.csv",
    "sinewave_glitch_0P010.csv",
    "sinewave_glitch_0P001.csv",

    # Gain errors
    "sinewave_gain_error_0P98.csv",
    "sinewave_gain_error_0P99.csv",
    "sinewave_gain_error_1P01.csv",
    "sinewave_gain_error_1P02.csv",

    # Clipping errors
    "sinewave_clipping_0P055.csv",
    "sinewave_clipping_0P060.csv",
    "sinewave_clipping_0P070.csv",

    # Jitter errors
    "sinewave_jitter_0P001.csv",
    "sinewave_jitter_0P002.csv",
    "sinewave_jitter_0P0002.csv",

    # Drift errors
    "sinewave_drift_0P050.csv",
    "sinewave_drift_0P100.csv",
    "sinewave_drift_0P200.csv",

    # Amplitude modulation
    "sinewave_amplitude_modulation_0P001.csv",
    "sinewave_amplitude_modulation_0P005.csv",

    # Amplitude noise
    "sinewave_amplitude_noise_0P001.csv",
    "sinewave_amplitude_noise_0P005.csv",
    "sinewave_amplitude_noise_0P010.csv",
]

# Sampling frequency (Hz) - adjust based on your data
SAMPLING_FREQ = 1e9  # 1 GHz

# ACF max lag
ACF_MAX_LAG = 300


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================
def analyze_data_file(data_filename, Fs=SAMPLING_FREQ, MaxLag=ACF_MAX_LAG):
    """
    Perform complete error analysis on a single data file.

    Parameters
    ----------
    data_filename : str
        Filename (without path) of the CSV data file
    Fs : float
        Sampling frequency in Hz
    MaxLag : int
        Maximum lag for autocorrelation
    """
    # Extract base name (without .csv extension)
    base_name = os.path.splitext(data_filename)[0]

    # Construct full paths using string operations
    input_dir = os.path.join(_project_root, "ADCToolbox_example_data")
    output_dir = os.path.join(_project_root, "ADCToolbox_example_output", base_name)
    data_file_path = os.path.join(input_dir, data_filename)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Processing: {data_filename}")
    print(f"{'='*70}")

    # Load data
    try:
        data = np.loadtxt(data_file_path, delimiter=',')
        print(f"[Loaded] {len(data)} samples from {data_filename}")
    except Exception as e:
        print(f"[ERROR] Failed to load {data_filename}: {e}")
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

        output_file = os.path.join(output_dir, f"spectrum_of_{base_name}_python.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] Spectrum plot saved: {os.path.basename(output_file)}")
    except Exception as e:
        print(f"[ERROR] Spectrum plot failed: {e}")

    # ========================================================================
    # Test 2: Error PDF
    # ========================================================================
    try:
        plt.figure(figsize=(12, 8))
        noise_lsb, mu, sigma, KL_divergence, x, fx, gauss_pdf = errPDF(err_data)

        output_file = os.path.join(output_dir, f"errPDF_of_{base_name}_python.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] Error PDF saved: {os.path.basename(output_file)}")
        print(f"    KL_divergence={KL_divergence:.4f}, μ={mu:.2f}, σ={sigma:.2f}")
    except Exception as e:
        print(f"[ERROR] Error PDF failed: {e}")

    # ========================================================================
    # Test 3: Error Autocorrelation
    # ========================================================================
    try:
        plt.figure(figsize=(12, 8))
        acf, lags = errAutoCorrelation(err_data, MaxLag=MaxLag)

        output_file = os.path.join(output_dir, f"errACF_of_{base_name}_python.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] Error ACF saved: {os.path.basename(output_file)}")
    except Exception as e:
        print(f"[ERROR] Error autocorrelation failed: {e}")

    # ========================================================================
    # Test 4: Error Spectrum
    # ========================================================================
    try:
        plt.figure(figsize=(12, 8))
        spec_plot(err_data, label=0, Fs=Fs)
        plt.title("Error Spectrum")

        output_file = os.path.join(output_dir, f"errSpectrum_of_{base_name}_python.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] Error spectrum saved: {os.path.basename(output_file)}")
    except Exception as e:
        print(f"[ERROR] Error spectrum failed: {e}")

    # ========================================================================
    # Test 5: Error Envelope Spectrum
    # ========================================================================
    try:
        plt.figure(figsize=(12, 8))
        errEnvelopeSpectrum(err_data, Fs=Fs)

        output_file = os.path.join(output_dir, f"errEnvelopeSpectrum_of_{base_name}_python.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] Error envelope spectrum saved: {os.path.basename(output_file)}")
    except Exception as e:
        print(f"[ERROR] Error envelope spectrum failed: {e}")

    print(f"[Done] All plots saved to: {output_dir}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def main():
    """Process all enabled data files."""

    if not DATA_FILES:
        print("[WARNING] No data files enabled in DATA_FILES list!")
        print("Please edit the script and uncomment desired data files.")
        return

    print(f"\n{'='*70}")
    print(f"ADC Error Analysis Test")
    print(f"{'='*70}")
    print(f"Number of files to process: {len(DATA_FILES)}")
    print(f"Sampling frequency: {SAMPLING_FREQ/1e9:.1f} GHz")
    print(f"ACF max lag: {ACF_MAX_LAG} samples")

    # Process each enabled file
    for data_file in DATA_FILES:
        analyze_data_file(data_file, Fs=SAMPLING_FREQ, MaxLag=ACF_MAX_LAG)

    print(f"\n{'='*70}")
    print(f"[Complete] Processed {len(DATA_FILES)} file(s)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
