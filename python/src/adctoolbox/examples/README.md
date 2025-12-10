# ADCToolbox Examples

## Quick Start

**Step 1: Install ADCToolbox**
```bash
pip install adctoolbox
```

**Step 2: Copy examples to your workspace**
```bash
adctoolbox-get-examples
```

**Step 3: Run examples**
```bash
cd adctoolbox_examples

# Basic examples
python exp_b01_plot_sine.py
python exp_b02_spectrum.py
python exp_b03_sine_fit.py
python exp_b04_aliasing.py

# Analog analysis examples
python exp_a01_spec_plot_nonidealities.py
python exp_a02_spec_plot_jitter_fin.py
python exp_a03_err_pdf.py
python exp_a04_err_hist_sine_phase.py
python exp_a05_jitter_calculation.py
python exp_a06_err_hist_sine_code.py
python exp_a07_extract_static_nonlin.py
python exp_a08_inl_dnl_sweep.py
python exp_a09_spec_plot_phase.py
python exp_a10_err_auto_correlation.py
python exp_a11_err_envelope_spectrum.py
python exp_a12_err_spectrum.py
python exp_a13_tom_decomp.py
python exp_a14_spec_plot_2tone.py

# Digital analysis examples
python exp_d01_bit_activity.py
python exp_d02_fg_cal_sine.py
python exp_d03_redundancy_comparison.py
python exp_d04_weight_scaling.py
python exp_d05_enob_bit_sweep.py

# Toolset examples (complete workflows)
python exp_toolset_aout.py
python exp_toolset_dout.py
```

**Note**: All examples save figures to `./output/` directory **without displaying them** (non-blocking mode). This allows you to run multiple examples in sequence without manual intervention. If you want to see figures pop up, add `plt.show()` at the end of any example script.

---

## Categories

### **Basic (b01-b04)** - Basic Functions
Foundation tools for signal generation, visualization, and analysis.

| Example | Description |
|---------|-------------|
| `b01_plot_sine` | Plot ideal sinewave |
| `b02_spectrum` | FFT spectrum analysis |
| `b03_sine_fit` | Fit sine to noisy data |
| `b04_aliasing` | Nyquist zones demonstration |

### **Analog Output (a01-a14)** - Processing Recovered Signal
Analysis on analog output (vector of recovered signal, e.g., reconstructed sinewave).

| Example | Description |
|---------|-------------|
| `a01_spec_plot_nonidealities` | 4 non-idealities comparison (noise, jitter, HD, kickback) |
| `a02_spec_plot_jitter_fin` | Jitter across Nyquist zones |
| `a03_err_pdf` | Error PDF comparison (4 non-idealities) |
| `a04_err_hist_sine_phase` | Error histogram by phase (8 bins) |
| `a05_jitter_calculation` | Jitter: time vs frequency domain |
| `a06_err_hist_sine_code` | Error histogram by code |
| `a07_extract_static_nonlin` | Extract k2, k3 nonlinearity coefficients |
| `a08_inl_dnl_sweep` | INL/DNL vs record length (N = 2^10 to 2^16) |
| `a09_spec_plot_phase` | Spectrum with phase (4 frequencies) |
| `a10_err_auto_correlation` | Autocorrelation (12 non-ideality patterns) |
| `a11_err_envelope_spectrum` | Error envelope spectrum |
| `a12_err_spectrum` | Error spectrum |
| `a13_tom_decomp` | TOM decomposition |
| `a14_spec_plot_2tone` | Two-tone intermodulation |

### **Digital Output (d01-d05)** - Processing ADC Digital Codes
Analysis on digital output codes from ADC architectures (pipeline, SAR, etc.).

| Example | Description |
|---------|-------------|
| `d01_bit_activity` | Pipeline stage bit activity |
| `d02_fg_cal_sine` | Foreground gain calibration |
| `d03_redundancy_comparison` | Pipeline architecture comparison |
| `d04_weight_scaling` | Digital weight scaling |
| `d05_enob_bit_sweep` | ENOB vs bit sweep |

### **Toolset** - Complete Analysis Workflows
Comprehensive analysis suites that run multiple tools automatically.

| Example | Description |
|---------|-------------|
| `toolset_aout` | Run all 9 analog output analysis tools (spec_plot, err_pdf, INL/DNL, etc.) |
| `toolset_dout` | Run all 6 digital output analysis tools (bit_activity, weight_scaling, enob_sweep, etc.) |

---

## Standard Parameters

```python
N = 2**13              # 8192 samples
Fs = 800e6             # 800 MHz
Fin = 80e6             # 80 MHz (coherent)
A = 0.49               # Amplitude
DC = 0.5               # DC offset
base_noise = 50e-6     # 50 uV

# Distortion
hd2_dB = -80           # -80 dB HD2
hd3_dB = -66 or -73    # -66/-73 dB HD3
jitter_rms = 2e-12     # 2 ps
```

## Key Functions

- **Spectrum**: `spec_plot(signal, fs=Fs)`
- **Error PDF**: `err_pdf(signal, resolution=12, plot=True)`
- **INL/DNL**: `inl_dnl_from_sine(codes, num_bits=10, clip_percent=0.01)`
- **Autocorrelation**: `err_auto_correlation(error, max_lag=100)`
- **Sine Fit**: `sine_fit(signal, freq=None)`
