# ADCToolbox

Comprehensive toolbox for ADC characterization, calibration, and performance analysis.

## Features

- **21 Ready-to-Run Examples**: Basic (4) + Analog Analysis (14) + Digital Analysis (5)
- **Analog Output Analysis**: 9 diagnostic tools for time-domain, frequency-domain, and statistical error analysis
- **Digital Output Analysis**: 6 tools for bit-weighted ADCs with automatic calibration
- **Dual Implementation**: Full MATLAB and Python implementations with numerical parity validated
- **Production-Ready**: CI/CD enabled, 100% test coverage, fully documented

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

### Step 1: Copy Examples to Your Workspace

Open your terminal (or command prompt on Windows) and navigate to where you want the examples:

```bash
cd /path/to/your/workspace

# Run the command-line tool (installed with pip)
adctoolbox-get-examples
```

This command creates an `adctoolbox_examples/` directory with all 21 examples in your current location.

### Step 2: Run Examples

```bash
cd adctoolbox_examples

# Basic examples (b01-b04)
python exp_b01_plot_sine.py              # Sine wave visualization
python exp_b02_spectrum.py               # FFT spectrum analysis
python exp_b03_sine_fit.py               # Fit sine to noisy data
python exp_b04_aliasing.py               # Nyquist zones demo

# Analog output analysis (a01-a14)
python exp_a01_spec_plot_nonidealities.py  # 4 non-idealities comparison
python exp_a02_spec_plot_jitter.py         # Jitter across Nyquist zones
python exp_a03_err_pdf.py                  # Error PDF comparison
# ... see examples/README.md for all 14 analog examples

# Digital output analysis (d01-d05)
python exp_d01_bit_activity.py           # Pipeline bit activity
python exp_d02_fg_cal_sine.py            # Foreground calibration
python exp_d03_redundancy_comparison.py  # Architecture comparison
# ... see examples/README.md for all 5 digital examples
```

All outputs save to `./output/` directory.

### Step 3: Use in Your Code

**Option A: Run complete toolsets (like MATLAB)**

```python
import numpy as np
from adctoolbox.aout.toolset_aout import toolset_aout
from adctoolbox.dout.toolset_dout import toolset_dout

# Analog output analysis (9 tools)
aout_data = np.loadtxt('sinewave.csv')
status = toolset_aout(aout_data, 'output/test1', visible=False)
# Creates 9 diagnostic plots + 1 panel overview

# Digital output analysis (3 tools)
bits = np.loadtxt('sar_bits.csv', delimiter=',')
status = toolset_dout(bits, 'output/test1', visible=False)
# Creates 3 diagnostic plots + 1 panel overview
```

**Option B: Use individual functions**

```python
import numpy as np
from adctoolbox import spec_plot, sine_fit, find_bin

# Generate test signal
N = 2**13
Fs = 800e6
Fin = 80e6
t = np.arange(N) / Fs
signal = 0.49 * np.sin(2*np.pi*Fin*t) + 0.5

# Analyze spectrum
enob, sndr, sfdr, snr, thd, pwr, nf, nsd = spec_plot(signal, fs=Fs, harmonic=5)
print(f"ENOB: {enob:.2f} bits, SNDR: {sndr:.2f} dB")

# Fit sine wave
fit_signal, freq, mag, dc, phi = sine_fit(signal)
print(f"Fitted frequency: {freq*Fs/1e6:.2f} MHz")

# Find coherent bin
J = find_bin(Fs, Fin, N)
print(f"Coherent bin: {J}")
```

## Example Categories

### Basic Examples (b01-b04)
Foundation tools for signal generation, visualization, and basic analysis.

| Example | Description |
|---------|-------------|
| `exp_b01_plot_sine.py` | Plot ideal sinewave |
| `exp_b02_spectrum.py` | FFT spectrum analysis (coherent vs non-coherent) |
| `exp_b03_sine_fit.py` | Fit sine to noisy data |
| `exp_b04_aliasing.py` | Nyquist zones demonstration |

### Analog Output Analysis (a01-a14)
Analysis on analog output - vector of recovered signal (e.g., reconstructed sinewave).

| Example | Description |
|---------|-------------|
| `exp_a01_spec_plot_nonidealities.py` | 4 non-idealities comparison (noise, jitter, HD, kickback) |
| `exp_a02_spec_plot_jitter.py` | Jitter across Nyquist zones (1st-5th zones) |
| `exp_a03_err_pdf.py` | Error PDF comparison (4 non-idealities) |
| `exp_a04_err_hist_sine_phase.py` | Error histogram by phase (8 bins) |
| `exp_a05_jitter_calculation.py` | Jitter: time vs frequency domain |
| `exp_a06_err_hist_sine_code.py` | Error histogram by code |
| `exp_a07_fit_static_nol.py` | Extract k2, k3 nonlinearity coefficients |
| `exp_a08_inl_dnl_sweep.py` | INL/DNL vs record length (N = 2^10 to 2^16) |
| `exp_a09_spec_plot_phase.py` | Spectrum with phase (4 frequencies) |
| `exp_a10_err_auto_correlation.py` | Autocorrelation (12 non-ideality patterns) |
| `exp_a11_err_envelope_spectrum.py` | Error envelope spectrum |
| `exp_a12_err_spectrum.py` | Error spectrum |
| `exp_a13_tom_decomp.py` | TOM decomposition |
| `exp_a14_spec_plot_2tone.py` | Two-tone intermodulation |

### Digital Output Analysis (d01-d05)
Analysis on digital codes from ADC architectures (pipeline, SAR, etc.).

| Example | Description |
|---------|-------------|
| `exp_d01_bit_activity.py` | Pipeline stage bit activity |
| `exp_d02_fg_cal_sine.py` | Foreground gain calibration |
| `exp_d03_redundancy_comparison.py` | Pipeline architecture comparison |
| `exp_d04_weight_scaling.py` | Digital weight scaling |
| `exp_d05_enob_bit_sweep.py` | ENOB vs bit sweep |

## Key Functions

### Spectrum Analysis
```python
from adctoolbox import spec_plot

# Analyze signal spectrum
enob, sndr, sfdr, snr, thd, pwr, nf, nsd = spec_plot(
    signal,
    fs=800e6,      # Sampling frequency
    harmonic=5,    # Number of harmonics to analyze
    label=1        # Show plot labels
)
```

### Error Analysis
```python
from adctoolbox import err_pdf, err_auto_correlation

# Error probability density function
noise_lsb, mu, sigma, kl_div = err_pdf(signal, resolution=12, plot=True)

# Autocorrelation analysis
lags, autocorr = err_auto_correlation(error, max_lag=100)
```

### INL/DNL Extraction
```python
from adctoolbox import inl_dnl_from_sine

# Extract INL and DNL from sine wave test
inl, dnl, codes, hist = inl_dnl_from_sine(
    adc_codes,
    num_bits=10,
    clip_percent=0.01
)
```

### Digital Calibration
```python
from adctoolbox import fg_cal_sine

# Foreground gain calibration for pipeline/SAR ADCs
weights_cal, offset, analog_cal, freq, snr, thd = fg_cal_sine(
    digital_output,
    freq=0,        # Auto-detect frequency
    order=5        # Polynomial order
)
```

## MATLAB Version

MATLAB implementation available on GitHub with identical functionality.

```matlab
% Analog output analysis
aout_data = readmatrix('sinewave.csv');
[plot_files, status] = toolset_aout(aout_data, 'output/test1');

% Digital output analysis
bits = readmatrix('sar_bits.csv');
[plot_files, status] = toolset_dout(bits, 'output/test1');
```

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



