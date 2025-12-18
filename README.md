# ADCToolbox

Comprehensive toolbox for ADC characterization, calibration, and performance analysis.

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://arcadia-1.github.io/ADCToolbox/)
[![PyPI version](https://badge.fury.io/py/adctoolbox.svg)](https://badge.fury.io/py/adctoolbox)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Documentation

📚 **[Full Documentation](https://arcadia-1.github.io/ADCToolbox/)** - Complete API reference, algorithm guides, and tutorials

- **[Installation Guide](https://arcadia-1.github.io/ADCToolbox/installation.html)** - Getting started
- **[Quick Start](https://arcadia-1.github.io/ADCToolbox/quickstart.html)** - First steps with examples
- **[Algorithm Reference](https://arcadia-1.github.io/ADCToolbox/algorithms/index.html)** - 15 detailed algorithm guides
- **[API Documentation](https://arcadia-1.github.io/ADCToolbox/api/index.html)** - Function signatures and parameters
- **[Changelog](https://arcadia-1.github.io/ADCToolbox/changelog.html)** - Version history

## Features

- **45 Ready-to-Run Examples**: Basic (2) + Spectrum Analysis (14) + Signal Generation (6) + Analog Debug (13) + Digital Debug (5) + Metrics (5)
- **Spectrum Analysis**: FFT-based analysis with ENOB, SNR, SFDR, THD, windowing, averaging, and polar visualization
- **Signal Generation**: Thermal noise, jitter, quantization, static/dynamic nonlinearity, and interference modeling
- **Analog Error Analysis**: Time-domain, frequency-domain, and statistical error characterization (PDF, autocorrelation, envelope spectrum)
- **Digital Calibration**: Bit-weighted ADC calibration and redundancy analysis
- **Production-Ready**: Modern Python package with comprehensive documentation and examples

## Installation

```bash
# Install
pip install adctoolbox

# Upgrade if already installed
pip install --upgrade adctoolbox

# Check version
python -c "import adctoolbox; print(adctoolbox.__version__)"
```

## Quick Start

**Get all 45 examples in one command:**

```bash
cd /path/to/your/workspace
adctoolbox-get-examples
```

This creates an `adctoolbox_examples/` directory with all examples organized by category.

**Run an example:**

```bash
cd adctoolbox_examples/02_spectrum
python exp_s01_analyze_spectrum_simplest.py
```

**Use in your code:**

```python
from adctoolbox import analyze_spectrum

# Analyze signal spectrum
result = analyze_spectrum(signal, fs=800e6, show_plot=True)
print(f"ENOB: {result['enob']:.2f} bits, SNDR: {result['sndr_db']:.2f} dB")
```

See [Usage Examples](#usage-examples) section below for detailed code examples.

## Example Categories

### 01_basic - Fundamentals (2 examples)
Environment verification and coherent sampling basics.

| Example | Description |
|---------|-------------|
| `exp_b01_environment_check.py` | Verify installation and plot test signal |
| `exp_b02_coherent_vs_non_coherent.py` | Demonstrate coherent vs non-coherent sampling impact on ENOB |

### 02_spectrum - FFT-Based Analysis (14 examples)
Spectrum analysis with windowing, averaging, polar plots, and two-tone testing.

| Example | Description |
|---------|-------------|
| `exp_s01_analyze_spectrum_simplest.py` | Simplest spectrum analysis example |
| `exp_s02_analyze_spectrum_interactive.py` | Interactive spectrum visualization |
| `exp_s03_analyze_spectrum_savefig.py` | Save spectrum plots with amplitude comparison |
| `exp_s04_sweep_dynamic_range.py` | Dynamic range measurement (SNR vs amplitude) |
| `exp_s05_annotating_spur.py` | Annotate and identify spurs in spectrum |
| `exp_s06_sweeping_fft_and_osr.py` | FFT length and OSR comparison (resolution vs SNR) |
| `exp_s07_spectrum_averaging.py` | Coherent averaging demonstration |
| `exp_s08_windowing_deep_dive.py` | Window function comparison (8 windows × 3 scenarios) |
| `exp_s10_polar_noise_and_harmonics.py` | Polar phase spectrum: noise vs harmonics |
| `exp_s11_polar_memory_effect.py` | Memory effect analysis via polar spectrum |
| `exp_s12_polar_coherent_averaging.py` | Coherent averaging with polar plots |
| `exp_s21_analyze_two_tone_spectrum.py` | Two-tone spectrum analysis (IMD2/IMD3) |
| `exp_s22_two_tone_imd_comparison.py` | IMD product comparison across frequencies |
| `exp_s23_two_tone_spectrum_averaging.py` | Power vs coherent averaging for two-tone |

### 03_generate_signals - Non-Ideality Modeling (6 examples)
Generate ADC signals with various impairments for testing and validation.

| Example | Description |
|---------|-------------|
| `exp_g01_generate_signal_demo.py` | Thermal noise demonstration (4 noise levels) |
| `exp_g03_sweep_quant_bits.py` | Quantization noise vs bit resolution (2-16 bits) |
| `exp_g04_sweep_jitter_fin.py` | Jitter vs input frequency sweep |
| `exp_g05_sweep_static_nonlin.py` | Static nonlinearity (HD2/HD3 sign combinations) |
| `exp_g06_sweep_dynamic_nonlin.py` | Dynamic effects (settling, memory, RA gain) |
| `exp_g07_sweep_interferences.py` | Interference types (glitch, AM, clipping, drift) |

### 04_debug_analog - Error Characterization (13 examples)
Time-domain, frequency-domain, and statistical error analysis on analog waveforms.

| Example | Description |
|---------|-------------|
| `exp_a01_fit_sine_4param.py` | 4-parameter sine fitting (DC, amplitude, frequency, phase) |
| `exp_a02_analyze_error_by_value.py` | Error histograms binned by ADC code |
| `exp_a03_analyze_error_by_phase.py` | Decompose error into AM/PM noise components |
| `exp_a04_jitter_calculation.py` | Jitter measurement and validation |
| `exp_a11_decompose_harmonics.py` | Time-domain harmonic decomposition |
| `exp_a12_decompose_harmonics_polar.py` | Polar plot harmonic decomposition |
| `exp_a21_analyze_error_pdf.py` | Error probability distribution (15 non-idealities) |
| `exp_a22_analyze_error_spectrum.py` | Error spectrum analysis |
| `exp_a23_analyze_error_autocorrelation.py` | Temporal correlation in error signal |
| `exp_a24_analyze_error_envelope_spectrum.py` | Error envelope spectrum (AM patterns) |
| `exp_a25_spectra.py` | Spectrum comparison across non-idealities |
| `exp_a31_fit_static_nonlin.py` | Extract k2/k3 nonlinearity coefficients |
| `exp_a32_inl_from_sine_sweep_length.py` | INL/DNL vs record length (N = 2^10 to 2^16) |

### 05_debug_digital - Calibration & Redundancy (5 examples)
Digital code analysis for pipeline and SAR ADCs with calibration algorithms.

| Example | Description |
|---------|-------------|
| `exp_d01_bit_activity.py` | Bit flip activity visualization |
| `exp_d02_cal_weight_sine.py` | Foreground weight calibration using sine wave |
| `exp_d03_redundancy_comparison.py` | Architecture comparison (1.5-bit vs 2-bit stages) |
| `exp_d04_weight_scaling.py` | Digital weight scaling analysis |
| `exp_d05_sweep_bit_enob.py` | ENOB vs bit resolution sweep |

### 06_calculate_metric - Utility Functions (5 examples)
Helper functions for unit conversions and metric calculations.

| Example | Description |
|---------|-------------|
| `exp_b01_aliasing_nyquist_zones.py` | Aliasing and Nyquist zone demonstration |
| `exp_b02_unit_conversions.py` | dB, magnitude, SNR, ENOB conversions |
| `exp_b03_calculate_fom.py` | Figure of Merit (FoM) calculations |
| `exp_b05_amplitudes_to_snr.py` | Amplitude to SNR conversion |
| `exp_b06_convert_nsd_snr.py` | Noise Spectral Density and SNR conversion |

## Usage Examples

<details>
<summary><b>Spectrum Analysis</b></summary>

```python
from adctoolbox import analyze_spectrum, analyze_two_tone_spectrum

# Single-tone analysis
result = analyze_spectrum(signal, fs=800e6, harmonic=5, show_plot=True)
print(f"ENOB: {result['enob']:.2f} bits, SNDR: {result['sndr_db']:.2f} dB")

# Two-tone analysis (IMD)
result = analyze_two_tone_spectrum(signal, fs=1000e6, show_plot=True)
print(f"IMD2: {result['imd2_db']:.2f} dB, IMD3: {result['imd3_db']:.2f} dB")
```
</details>

<details>
<summary><b>Error Analysis (Auto-fits sine internally)</b></summary>

```python
from adctoolbox import (
    analyze_error_pdf,
    analyze_error_spectrum,
    analyze_error_autocorr,
    analyze_error_envelope_spectrum
)

# Error PDF
result = analyze_error_pdf(signal, resolution=12, show_plot=True)
print(f"Std: {result['sigma']:.2f} LSB, KL div: {result['kl_divergence']:.4f}")

# Error autocorrelation
result = analyze_error_autocorr(signal, max_lag=100, show_plot=True)

# Error spectrum
result = analyze_error_spectrum(signal, fs=800e6, show_plot=True)

# Error envelope spectrum (AM detection)
result = analyze_error_envelope_spectrum(signal, fs=800e6, show_plot=True)
```
</details>

<details>
<summary><b>Sine Fitting & Harmonic Decomposition</b></summary>

```python
from adctoolbox import fit_sine_4param, analyze_decomposition_time

# 4-parameter sine fitting
result = fit_sine_4param(signal, frequency_estimate=0.1)
print(f"Freq: {result['frequency']:.6f}, Amp: {result['amplitude']:.4f}")

# Harmonic decomposition
result = analyze_decomposition_time(signal, fs=800e6, harmonic=5, show_plot=True)
```
</details>

<details>
<summary><b>INL/DNL Extraction</b></summary>

```python
from adctoolbox import analyze_inl_from_sine

result = analyze_inl_from_sine(signal, resolution=12, show_plot=True)
print(f"INL: [{result['inl'].min():.2f}, {result['inl'].max():.2f}] LSB")
print(f"DNL: [{result['dnl'].min():.2f}, {result['dnl'].max():.2f}] LSB")
```
</details>

<details>
<summary><b>Digital Calibration</b></summary>

```python
from adctoolbox import calibrate_weight_sine, calibrate_weight_two_tone

# Weight calibration
result = calibrate_weight_sine(digital_codes, order=5)
print(f"SNR: {result['snr_db']:.2f} dB, THD: {result['thd_db']:.2f} dB")

# Two-tone calibration
result = calibrate_weight_two_tone(digital_codes, order=5)
```
</details>

<details>
<summary><b>Complete Example with Signal Generation</b></summary>

```python
import numpy as np
from adctoolbox import analyze_spectrum, find_coherent_frequency, amplitudes_to_snr

# Setup
N, Fs, Fin_target = 2**13, 800e6, 80e6
Fin, Fin_bin = find_coherent_frequency(Fs, Fin_target, N)

# Generate signal
t = np.arange(N) / Fs
A, DC, noise_rms = 0.49, 0.5, 100e-6
signal = A * np.sin(2*np.pi*Fin*t) + DC + np.random.randn(N) * noise_rms

# Analyze
result = analyze_spectrum(signal, fs=Fs, harmonic=5, show_plot=True)
snr_theory = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=noise_rms)

print(f"Measured SNR: {result['snr_db']:.2f} dB")
print(f"Theoretical SNR: {snr_theory:.2f} dB")
```
</details>

## Requirements

- Python >= 3.8
- NumPy >= 1.20.0
- Matplotlib >= 3.3.0
- SciPy >= 1.6.0

## Citation

If you use this toolbox in your research, please cite:

```bibtex
@software{adctoolbox2025,
  author = {Zhang, Zhishuai and Lu, Jie},
  title = {ADCToolbox: Comprehensive ADC Characterization and Analysis Toolkit},
  year = {2025},
  url = {https://github.com/Arcadia-1/ADCToolbox}
}
```

## Authors

- **Zhishuai Zhang**
- **Lu Jie**

## License

MIT License



