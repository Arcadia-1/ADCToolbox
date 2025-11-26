# ADCToolbox

A comprehensive Python toolbox for **ADC (Analog-to-Digital Converter)** characterization and analysis.

Delivers clear **multi-angle diagnostic views** of ADC behavior, enabling deeper insight and faster issue location.

## Features

- **Spectrum Analysis**: ENOB, SNDR, SFDR, SNR, THD, Noise Floor
- **Error Analysis**: PDF, Autocorrelation, Envelope Spectrum, Histogram Analysis
- **Jitter Detection**: Amplitude vs Phase Noise Decomposition
- **Calibration**: Foreground Calibration for SAR/Pipeline ADCs
- **INL/DNL Extraction**: Histogram-based nonlinearity analysis
- **100% MATLAB Parity**: Validated against reference MATLAB implementation

---

## Installation

```bash
pip install adctoolbox
```

or (recommended)

```bash
uv pip install adctoolbox
```

**Requirements**: Python >= 3.8, numpy, scipy, matplotlib, pandas

---

## Quick Start

### Spectrum Analysis

```python
from adctoolbox.aout import spec_plot
import numpy as np

# Load ADC output data
data = np.loadtxt("adc_output.csv", delimiter=',')

# Analyze spectrum and get performance metrics
ENoB, SNDR, SFDR, SNR, THD, pwr, NF, _ = spec_plot(data)

print(f"ENOB: {ENoB:.2f} bits")
print(f"SNDR: {SNDR:.2f} dB")
print(f"SFDR: {SFDR:.2f} dB")
```

### Sine Wave Fitting

```python
from adctoolbox.common import sine_fit

# Fit sine wave and extract parameters
data_fit, freq, mag, dc, phi = sine_fit(data)
err = data - data_fit

print(f"Frequency: {freq:.6f}")
print(f"Magnitude: {mag:.4f}")
print(f"DC offset: {dc:.4f}")
print(f"Phase: {phi:.2f} deg")
```

### Jitter Analysis

```python
from adctoolbox.aout import err_hist_sine

# Extract jitter from error histogram
emean, erms, phase, anoi, pnoi, err, xx = err_hist_sine(
    data, bin=99, fin=freq, disp=0
)

# Calculate jitter RMS (assuming input frequency Fin in Hz)
jitter_rms = pnoi / (2 * np.pi * Fin)
print(f"Jitter RMS: {jitter_rms*1e15:.2f} fs")
```

### Error PDF Analysis

```python
from adctoolbox.aout import err_pdf

# Analyze error distribution
noise_lsb, mu, sigma, KL_div, x, fx, gauss = err_pdf(err)

print(f"Noise RMS: {noise_lsb:.4f} LSB")
print(f"Mean: {mu:.4f} LSB")
print(f"Std Dev: {sigma:.4f} LSB")
print(f"KL Divergence: {KL_div:.4f}")
```

### INL/DNL Extraction

```python
from adctoolbox.aout import inl_sine

# Extract INL and DNL from sine histogram
INL, DNL, code = inl_sine(data)

print(f"Max INL: {np.max(np.abs(INL)):.4f} LSB")
print(f"Max DNL: {np.max(np.abs(DNL)):.4f} LSB")
```

---

## Available Tools

### Spectrum Analysis (`adctoolbox.aout`)
- **`spec_plot`** - FFT spectrum with ENOB, SNDR, SFDR, SNR, THD, noise floor
- **`spec_plot_phase`** - Phase spectrum with polar plot
- **`tom_decomp`** - Thompson decomposition (deterministic vs random error)

### Error Analysis (`adctoolbox.aout`)
- **`err_pdf`** - Error PDF with KDE and Gaussian fitting
- **`err_hist_sine`** - Histogram-based error analysis with jitter detection
- **`err_auto_correlation`** - Error autocorrelation function
- **`err_envelope_spectrum`** - Envelope spectrum via Hilbert transform
- **`inl_sine`** - INL/DNL extraction from sine histogram

### Common Utilities (`adctoolbox.common`)
- **`sine_fit`** - Multi-parameter sine fitting with auto frequency search
- **`find_bin`** - FFT bin finder with sub-bin resolution
- **`find_fin`** - Input frequency finder
- **`alias`** - Frequency aliasing calculator
- **`cap2weight`** - Capacitor array to weight conversion

### Calibration (`adctoolbox.dout`)
- **`fg_cal_sine`** - Foreground calibration using sinewave
- **`overflow_chk`** - Overflow detection and validation

### Oversampling (`adctoolbox.oversampling`)
- **`ntf_analyzer`** - Noise Transfer Function analysis

---

## Package Structure

```
adctoolbox/
├── aout/           # Analog output analysis
├── common/         # Common utilities
├── dout/           # Digital output calibration
├── oversampling/   # Oversampling analysis
└── utils/          # Utility functions
```

---

## Documentation

- **GitHub Repository**: [https://github.com/Arcadia-1/ADCToolbox](https://github.com/Arcadia-1/ADCToolbox)
- **Full Documentation**: See GitHub repository for complete guides
- **API Reference**: Comprehensive docstrings in all modules
- **Test Suite**: 15 unit tests with 100% MATLAB-Python parity validation

---

## Validation

**Tested against MATLAB reference implementation**:
- 15 Python unit tests
- 96 comparison test cases
- 100% numerical parity achieved
- Results: 27 PERFECT + 51 EXCELLENT + 13 GOOD + 5 ACCEPTABLE

---

## Example Use Cases

- **ADC Characterization**: Measure ENOB, SNDR, SFDR for performance verification
- **Jitter Analysis**: Separate amplitude noise from phase noise/jitter
- **Nonlinearity Testing**: Extract INL/DNL using sine histogram method
- **Calibration**: Foreground calibration for SAR and pipeline ADCs
- **Error Diagnosis**: Multi-view error analysis (time, frequency, phase domains)
- **Research**: ADC algorithm development with validated reference implementation

---

## License

MIT License - See [LICENSE](https://github.com/Arcadia-1/ADCToolbox/blob/main/LICENSE) for details.

## Citation

If you use this toolbox in your research, please cite:

```bibtex
@software{adctoolbox2025,
  author = {Zhang, Zhishuai and Jie, Lu},
  title = {ADCToolbox},
  year = {2025},
  url = {https://github.com/Arcadia-1/ADCToolbox}
}
```

---

## Authors

Zhishuai Zhang, Lu Jie

---

## Contributing

Contributions welcome! Please visit the [GitHub repository](https://github.com/Arcadia-1/ADCToolbox) for contribution guidelines.

**Development Setup**:
```bash
git clone https://github.com/Arcadia-1/ADCToolbox.git
cd ADCToolbox/python
pip install -e .[dev]
python tests/run_all_tests.py
```

---

## Links

- **GitHub**: [https://github.com/Arcadia-1/ADCToolbox](https://github.com/Arcadia-1/ADCToolbox)
- **Issues**: [https://github.com/Arcadia-1/ADCToolbox/issues](https://github.com/Arcadia-1/ADCToolbox/issues)
- **MATLAB Version**: Also available in the same repository
