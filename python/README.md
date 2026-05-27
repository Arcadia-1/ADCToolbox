# ADCToolbox

A Python toolbox for ADC characterization, calibration, modeling, and
visualization.

ADCToolbox provides spectrum analysis, sine fitting, analog error diagnostics,
digital bit-weight calibration, SAR behavioral models, time-interleaved ADC
helpers, and ready-to-run examples.

## Installation

```bash
pip install adctoolbox
```

Check the installed version:

```bash
python -c "import adctoolbox; print(adctoolbox.__version__)"
```

## Quick Start

```python
import numpy as np
from adctoolbox import analyze_spectrum, find_coherent_frequency

fs = 800e6
n_fft = 2**13
fin, _ = find_coherent_frequency(fs=fs, fin_target=80e6, n_fft=n_fft)

t = np.arange(n_fft) / fs
signal = 0.49 * np.sin(2 * np.pi * fin * t)

result = analyze_spectrum(signal, fs=fs, max_scale_range=(-0.5, 0.5))
print(f"ENOB: {result['enob']:.2f} bits")
print(f"SNDR: {result['sndr_dbc']:.2f} dBc")
```

## Examples

Copy the full example tree into your workspace:

```bash
adctoolbox-get-examples
cd adctoolbox_examples
python 01_basic/exp_b01_environment_check.py
python 02_spectrum/exp_s01_analyze_spectrum_simplest.py
```

The package currently ships 59 runnable examples across spectrum analysis,
signal generation, analog debug, digital calibration, toolset dashboards,
unit conversions, time-interleaved ADC analysis, and subsample output behavior.

## Common APIs

```python
from adctoolbox import (
    analyze_spectrum,
    fit_sine_4param,
    analyze_error_pdf,
    analyze_error_by_phase,
    analyze_inl_from_sine,
    calibrate_weight_sine,
    sar_convert,
    sar_reconstruct,
    sar_apply_cap_mismatch,
)
```

Most modern APIs return dictionaries with stable snake_case keys. For example,
`analyze_spectrum(...)` returns metrics such as `enob`, `sndr_dbc`,
`snr_dbc`, `sfdr_dbc`, `thd_dbc`, `sig_pwr_dbfs`, and `nsd_dbfs_hz`.

## Codex Skills

ADCToolbox also ships bundled Codex skills. Install them into an explicit
destination:

```bash
adctoolbox-install-skill --dest ~/.codex/skills
adctoolbox-install-skill --status --dest ~/.codex/skills
```

For local skill development:

```bash
adctoolbox-install-skill --dev --editable --force --dest ~/.codex/skills
```

## Documentation

- GitHub: https://github.com/Arcadia-1/ADCToolbox
- Docs: https://arcadia-1.github.io/ADCToolbox/
- PyPI: https://pypi.org/project/adctoolbox/

## License

MIT License.

## Citation

```bibtex
@software{adctoolbox2025,
  author = {Zhang, Zhishuai and Lu, Jie},
  title = {ADCToolbox: Comprehensive ADC Characterization and Analysis Toolkit},
  year = {2025},
  url = {https://github.com/Arcadia-1/ADCToolbox}
}
```
