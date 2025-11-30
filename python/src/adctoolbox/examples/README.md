# ADCToolbox Examples

This directory contains example scripts demonstrating how to use the ADCToolbox Python package.

## Accessing Example Data

All examples now include packaged data files that work immediately after installation:

```python
from adctoolbox.examples.data import get_example_data_path
import numpy as np

# Load example data
data_path = get_example_data_path('sinewave_jitter_400fs.csv')
signal = np.loadtxt(data_path, delimiter=',')
```

**Available datasets:**
- `sinewave_jitter_400fs.csv` - Jitter impairment (145 KB)
- `sinewave_noise_270uV.csv` - Noise analysis (145 KB)
- `sinewave_gain_error_0P98.csv` - Gain/INL demo (113 KB)
- `sinewave_clipping_0P012.csv` - Non-ideal conditions (139 KB)
- `dout_SAR_12b_weight_1.csv` - SAR calibration (200 KB)

## Directory Structure

Examples are numbered within each directory to show recommended learning progression:

```
examples/
├── quickstart/          # Quick start guides (START HERE)
│   └── example_00_basic_workflow.py      # Basic ADC analysis workflow
│
├── common/              # Common utilities examples (learn these early)
│   ├── example_00_sine_fit.py
│   ├── example_01_find_bin.py
│   ├── example_02_find_fin.py
│   └── example_03_alias.py
│
├── aout/                # Analog output analysis examples (main tools)
│   ├── example_00_spec_plot.py
│   ├── example_01_spec_plot_phase.py
│   ├── example_02_err_pdf.py
│   ├── example_03_err_hist_sine.py
│   ├── example_04_err_auto_correlation.py
│   ├── example_05_err_envelope_spectrum.py
│   ├── example_06_tom_decomp.py
│   ├── example_07_inl_sine.py
│   └── example_08_spec_plot_2tone.py
│
├── dout/                # Digital output calibration examples
│   ├── example_00_fg_cal_sine.py
│   ├── example_01_fg_cal_sine_os.py
│   ├── example_02_fg_cal_sine_2freq.py
│   └── example_03_overflow_chk.py
│
├── data_generation/     # Data generation examples
│   └── example_generate_test_data.py
│
└── workflows/           # Complete workflow examples (advanced)
    ├── example_00_complete_adc_analysis.py
    └── example_01_calibration_workflow.py
```

## Quick Start

### Installation

```bash
pip install adctoolbox
```

### Basic Usage

```python
import numpy as np
from adctoolbox.aout import spec_plot
from adctoolbox.common import find_bin

# Generate test sinewave
N = 2**12
J = find_bin(1, 0.1, N)  # Find coherent bin
signal = 0.5 * np.sin(2 * np.pi * J / N * np.arange(N)) + 0.5

# Analyze spectrum
enob, sndr, sfdr, snr, thd = spec_plot(signal, label=True)
print(f"ENoB: {enob:.2f}, SNDR: {sndr:.2f} dB")
```

## Example Categories

### 1. Analog Output Analysis (AOUT)
Tools for analyzing the calibrated/analog ADC output:
- **spec_plot**: Frequency spectrum analysis (ENoB, SNDR, SFDR, THD)
- **spec_plot_phase**: Phase-domain spectrum with polar representation
- **tom_decomp**: Thompson decomposition (dependent/independent error)
- **err_hist_sine**: Error histogram by code or phase
- **err_pdf**: Error probability density function
- **err_auto_correlation**: Error autocorrelation analysis
- **err_envelope_spectrum**: Error envelope spectrum
- **inl_sine**: INL (Integral Nonlinearity) analysis
- **spec_plot_2tone**: Two-tone intermodulation analysis

### 2. Digital Output Calibration (DOUT)
Tools for calibrating and analyzing digital ADC codes:
- **fg_cal_sine**: Foreground calibration using sinewave
- **fg_cal_sine_os**: Oversampling calibration
- **fg_cal_sine_2freq**: Two-frequency calibration
- **overflow_chk**: Check for code overflow/wrapping

### 3. Common Utilities
Helper functions used across the toolbox:
- **sine_fit**: Fit sinewave to data (frequency, amplitude, phase, DC)
- **find_bin**: Find coherent frequency bin for FFT
- **find_fin**: Find input frequency from data
- **alias**: Calculate aliased frequency

### 4. Data Generation
Generate synthetic ADC data for testing:
- Generate sinewaves with various impairments (jitter, nonlinearity, etc.)

### 5. Complete Workflows
End-to-end examples combining multiple tools:
- Complete ADC characterization workflow
- Calibration and performance comparison

## Running Examples

### Four Ways to Run Examples:

After `pip install adctoolbox`, you can run examples using any of these methods:

#### 1. CLI Commands (Easiest - Recommended)
```bash
adctoolbox-quickstart                    # Start here
adctoolbox-example-sine-fit              # Learn sine fitting
adctoolbox-example-spec-plot             # Spectrum analysis
adctoolbox-example-calibration           # Digital calibration
adctoolbox-example-workflow              # Complete workflow
```

#### 2. Python Module (Explicit)
```bash
python -m adctoolbox.examples.quickstart.example_00_basic_workflow
python -m adctoolbox.examples.common.example_00_sine_fit
python -m adctoolbox.examples.aout.example_00_spec_plot
python -m adctoolbox.examples.dout.example_00_fg_cal_sine
python -m adctoolbox.examples.workflows.example_00_complete_adc_analysis
```

#### 3. From Python Shell
```python
from adctoolbox.examples.quickstart import example_00_basic_workflow
# Script runs on import
```

#### 4. Direct Execution (Development)
```bash
cd /path/to/adctoolbox/python/src/adctoolbox/examples
python common/example_00_sine_fit.py
```

### Recommended Learning Path:

1. **Start with quickstart** → `adctoolbox-quickstart`
2. **Learn common utilities** → `adctoolbox-example-sine-fit`
3. **Explore spectrum analysis** → `adctoolbox-example-spec-plot`
4. **Try calibration** → `adctoolbox-example-calibration`
5. **Advanced workflows** → `adctoolbox-example-workflow`

## Example Data

Most examples include synthetic data generation. For examples using real ADC data:
- Place your CSV files in `examples/data/`
- Or modify the file path in the example script

## Notes

- All frequencies are normalized (Fin/Fs) unless otherwise specified
- Most tools use matplotlib for visualization
- Examples save figures to `examples/output/` directory (created automatically)

## Need Help?

- Check individual example files for detailed comments
- See the main documentation: https://github.com/Arcadia-1/ADCToolbox
- Report issues: https://github.com/Arcadia-1/ADCToolbox/issues
