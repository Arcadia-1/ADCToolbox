# ADCToolbox

Comprehensive toolbox for ADC characterization, calibration, and performance analysis.

## Features

- **Analog Output Analysis** (`toolset_aout`): 9 diagnostic tools for time-domain, frequency-domain, and statistical error analysis
- **Digital Output Analysis** (`toolset_dout`): 6 tools for bit-weighted ADCs with automatic calibration
- **Panel Generation**: Automatic summary visualizations combining multiple diagnostic plots
- **Dual Implementation**: Full MATLAB and Python implementations with cross-validation
- **Production-Ready**: Streamlined architecture with data-driven execution and consistent formatting

## Quick Start

### MATLAB

```matlab
% Analog output analysis
aout_data = readmatrix('sinewave.csv');
plot_files = toolset_aout(aout_data, 'output/test1');
toolset_aout_panel('output/test1', 'Prefix', 'aout');

% Digital output analysis
bits = readmatrix('sar_bits.csv');
plot_files = toolset_dout(bits, 'output/test1');
toolset_dout_panel('output/test1', 'Prefix', 'dout');
```

### Python

```python
from adctoolbox import toolset_aout, toolset_dout
import numpy as np

# Analog output analysis
aout_data = np.loadtxt('sinewave.csv')
plot_files = toolset_aout(aout_data, 'output/test1')

# Digital output analysis
bits = np.loadtxt('sar_bits.csv')
plot_files = toolset_dout(bits, 'output/test1')
```

## Structure

- `matlab/` - MATLAB implementation
  - `src/` - Source code (toolsets, panel functions, individual tools)
  - `tests/` - Test suite
  - `toolbox/` - Toolbox packaging
- `python/` - Python implementation
  - `src/adctoolbox/` - Package source
    - `aout/` - Analog output tools
    - `dout/` - Digital output tools
    - `common/` - Shared utilities
  - `tests/` - Unit tests and comparison tests
- `doc/` - Documentation
  - `AlgorithmOverview.md` - Architecture and algorithm descriptions
  - `toolset_aout.md` - Analog toolset documentation
  - `toolset_dout.md` - Digital toolset documentation
  - [Individual tool docs]
- `reference_output/` - Reference test outputs

## Documentation

### Core Documentation

- **[Algorithm Overview](doc/AlgorithmOverview.md)** - Architecture, design principles, and working principles
- **[toolset_aout](doc/toolset_aout.md)** - Analog output analysis suite (9 tools + panel)
- **[toolset_dout](doc/toolset_dout.md)** - Digital output analysis suite (6 tools + panel)

### Individual Tool Documentation

**Analog Output Tools:**
- [tomDecomp](doc/tomDecomp.md) - Time-domain error decomposition
- [specPlot](doc/specPlot.md) - Frequency spectrum analysis
- [specPlotPhase](doc/specPlotPhase.md) - Phase-domain error analysis
- [errHistSine](doc/errHistSine.md) - Error histogram (code/phase)
- [errPDF](doc/errPDF.md) - Error probability density function
- [errAutoCorrelation](doc/errAutoCorrelation.md) - Autocorrelation analysis
- [errEnvelopeSpectrum](doc/errEnvelopeSpectrum.md) - Envelope spectrum analysis

**Digital Output Tools:**
- [bitActivity](doc/bitActivity.md) - Bit toggle rate analysis
- [weightScaling](doc/weightScaling.md) - Weight/radix visualization
- [overflowChk](doc/overflowChk.md) - Overflow detection
- [ENoB_bitSweep](doc/ENoB_bitSweep.md) - ENoB vs bits analysis
- [FGCalSine](doc/FGCalSine.md) - Foreground sine calibration

**Utilities:**
- [UtilityFunctions](doc/UtilityFunctions.md) - Common utility functions

## Installation

### MATLAB

1. Add to MATLAB path:
   ```matlab
   addpath('matlab/src')
   ```

2. Or install as toolbox (see `matlab/README.md`)

### Python

```bash
cd python
pip install -e .
```

**Running Tests:**

```bash
cd python/tests
pytest unit/                    # Unit tests
python compare/run_all_comparisons.py  # MATLAB-Python comparison
```

## Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| **2.0** | 2025-12-03 | Streamlined toolsets, separated panel functions, data-driven execution |
| **1.0** | 2025-01-28 | Initial release |

## License



