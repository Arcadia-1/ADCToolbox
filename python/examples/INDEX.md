# ADCToolbox Examples Index

Quick reference for all example scripts in this directory.

## Quickstart
📍 **Start here if you're new to ADCToolbox**

| File | Description | Tools Used |
|------|-------------|------------|
| `quickstart/basic_workflow.py` | Complete beginner's guide to ADC analysis | fg_cal_sine, spec_plot, tom_decomp, err_hist_sine |

---

## Analog Output Analysis (AOUT)
🔍 **Tools for analyzing calibrated ADC output signals**

| File | Tool | Description |
|------|------|-------------|
| `aout/example_spec_plot.py` | **spec_plot** | Frequency spectrum analysis (ENoB, SNDR, SFDR, THD) |
| `aout/example_spec_plot_phase.py` | **spec_plot_phase** | Phase-domain spectrum with polar representation |
| `aout/example_tom_decomp.py` | **tom_decomp** | Thompson decomposition (dependent/independent error) |
| `aout/example_err_hist_sine.py` | **err_hist_sine** | Error histogram by code or phase |
| `aout/example_err_pdf.py` | **err_pdf** | Error probability density function |
| `aout/example_err_auto_correlation.py` | **err_auto_correlation** | Error autocorrelation analysis |
| `aout/example_err_envelope_spectrum.py` | **err_envelope_spectrum** | Error envelope spectrum |
| `aout/example_inl_sine.py` | **inl_sine** | INL (Integral Nonlinearity) analysis |
| `aout/example_spec_plot_2tone.py` | **spec_plot_2tone** | Two-tone intermodulation analysis |

**Status:** ⚠️ Need to create remaining AOUT examples (only spec_plot.py created so far)

---

## Digital Output Calibration (DOUT)
⚙️ **Tools for calibrating ADC digital codes**

| File | Tool | Description |
|------|------|-------------|
| `dout/example_fg_cal_sine.py` | **fg_cal_sine** | Foreground calibration using sinewave |
| `dout/example_fg_cal_sine_os.py` | **fg_cal_sine_os** | Oversampling calibration |
| `dout/example_fg_cal_sine_2freq.py` | **fg_cal_sine_2freq** | Two-frequency calibration |
| `dout/example_overflow_chk.py` | **overflow_chk** | Check for code overflow/wrapping |

**Status:** ⚠️ Need to create remaining DOUT examples (only fg_cal_sine.py created so far)

---

## Common Utilities
🛠️ **Helper functions used across the toolbox**

| File | Tool | Description |
|------|------|-------------|
| `common/example_sine_fit.py` | **sine_fit** | Fit sinewave to extract frequency, amplitude, phase, DC |
| `common/example_find_bin.py` | **find_bin** | Find coherent frequency bin for FFT |
| `common/example_find_fin.py` | **find_fin** | Find input frequency from data |
| `common/example_alias.py` | **alias** | Calculate aliased frequency |

**Status:** ⚠️ Need to create remaining common examples (only sine_fit.py created so far)

---

## Complete Workflows
🔄 **End-to-end examples combining multiple tools**

| File | Description | Complexity |
|------|-------------|------------|
| `workflows/complete_adc_analysis.py` | Full ADC characterization (calibration + 9 analysis tools) | Advanced |
| `workflows/calibration_workflow.py` | Focus on calibration methods comparison | Intermediate |

**Status:** ⚠️ Need to create calibration_workflow.py

---

## Data Generation
📊 **Generate synthetic ADC test data**

| File | Description |
|------|-------------|
| `data_generation/example_generate_test_data.py` | Generate sinewaves with various impairments |

**Status:** ⚠️ Need to create this example

---

## Running Examples

### Run individual example:
```bash
cd examples/quickstart
python basic_workflow.py
```

### Run specific category:
```bash
# Analog analysis examples
cd examples/aout
python example_spec_plot.py

# Calibration examples
cd examples/dout
python example_fg_cal_sine.py

# Utility examples
cd examples/common
python example_sine_fit.py

# Complete workflow
cd examples/workflows
python complete_adc_analysis.py
```

---

## Example Output

All examples save figures to `examples/output/` directory (created automatically).

Example output includes:
- PNG figures with analysis results
- Performance metric summaries
- Comparison plots (before/after calibration)
- Detailed analysis reports

---

## Creating More Examples

To add more examples, follow this template structure:

```python
"""
Example: tool_name - Brief Description

Detailed explanation of what this example demonstrates.
"""

import numpy as np
import matplotlib.pyplot as plt
from adctoolbox.module import tool_name

# Create output directory
import os
output_dir = "../output"
os.makedirs(output_dir, exist_ok=True)

# Example code here...

# Save results
plt.savefig(os.path.join(output_dir, 'example_output.png'), dpi=150)
```

---

## Need Help?

- **Documentation:** https://github.com/Arcadia-1/ADCToolbox
- **Issues:** https://github.com/Arcadia-1/ADCToolbox/issues
- **Main README:** `examples/README.md`

---

## Example Status Summary

✅ **Completed (5 files):**
- quickstart/basic_workflow.py
- aout/example_spec_plot.py
- dout/example_fg_cal_sine.py
- common/example_sine_fit.py
- workflows/complete_adc_analysis.py

⚠️ **To be created (remaining tools):**
- 7 more AOUT examples
- 3 more DOUT examples
- 3 more common examples
- 1 data generation example
- 1 additional workflow example

**Total:** 5/20 examples created
