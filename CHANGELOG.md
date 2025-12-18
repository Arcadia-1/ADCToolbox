# Changelog

All notable changes to ADCToolbox will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Work in Progress
- Additional examples and tutorials
- Performance optimizations

## [0.4.0] - 2025-12-18

**Documentation Release** - Complete Sphinx documentation overhaul with algorithm guides.

### Added
- **Complete Documentation Overhaul**:
  - 15 detailed algorithm documentation pages with Python API
  - Updated installation guide emphasizing `adctoolbox-get-examples`
  - Enhanced quickstart guide with learning path
  - All API reference docs updated to Python snake_case naming

### Changed
- **Documentation Structure**:
  - Installation guide shortened, git clone moved to bottom
  - Quickstart restructured to start with basic examples (exp_b01, exp_b02, then exp_s01)
  - Used actual code from examples instead of synthetic snippets
  - Emphasized "Learning with Examples" throughout documentation

### Removed
- Deleted 13 obsolete MATLAB-named algorithm documentation files
- Removed obsolete `src/__init__.py` file

### Fixed
- Version number synchronization across all files
- Dynamic versioning in `pyproject.toml`
- Documentation links and references updated to v0.4.0

## [0.3.0] - 2025-12-18

**Major Refactoring Release** - Complete Python architecture modernization with 45 examples.

### Breaking Changes
- **API Naming Convention**: All functions converted from MATLAB camelCase to Python snake_case
  - `sineFit` → `fit_sine_4param`
  - `INLsine` → `analyze_inl_from_sine`
  - `specPlot` → `analyze_spectrum`
  - `specPlotPhase` → `analyze_spectrum_polar`
  - `spec_plot_2tone` → `analyze_two_tone_spectrum`
  - `errPDF` → `analyze_error_pdf`
  - `errHistSine` → `analyze_error_by_value` / `analyze_error_by_phase`
  - `errAutoCorrelation` → `analyze_error_autocorr`
  - `errEnvelopeSpectrum` → `analyze_error_envelope_spectrum`
  - `tomDecomp` → `analyze_decomposition_time` / `analyze_decomposition_polar`
  - `FGCalSine` → `calibrate_weight_sine`
  - And 20+ more function renamings

- **Module Structure**: Consolidated and reorganized for better maintainability
  - `fundamentals`: Sine fitting, frequency utils, unit conversions, FOM metrics, validation
  - `spectrum`: Single-tone, two-tone, polar analysis (extracted from aout)
  - `aout`: Analog error analysis (10+ functions)
  - `dout`: Digital calibration (3 functions)
  - `siggen`: Signal generator with non-idealities
  - `oversampling`: NTF analysis

- **Return Values**: All functions now return dictionaries instead of tuples
  - Old: `enob, sndr, sfdr, ... = analyze_spectrum(...)`
  - New: `result = analyze_spectrum(...); enob = result['enob']`

### Added
- **45 Ready-to-Run Examples** (up from 21) across 6 categories:
  - `01_basic/` - Fundamentals (2 examples)
  - `02_spectrum/` - FFT-Based Analysis (14 examples)
  - `03_generate_signals/` - Non-Ideality Modeling (6 examples)
  - `04_debug_analog/` - Error Characterization (13 examples)
  - `05_debug_digital/` - Calibration & Redundancy (5 examples)
  - `06_calculate_metric/` - Utility Functions (5 examples)

- **Enhanced Error Analysis Functions**:
  - `analyze_error_by_phase`: AM/PM decomposition
  - `analyze_error_spectrum`: Error frequency analysis
  - `analyze_decomposition_polar`: Polar harmonic visualization
  - `fit_static_nonlin`: Extract k2/k3 nonlinearity coefficients

- **Expanded Fundamentals Module**:
  - Comprehensive unit conversions (dB, power, voltage, frequency, NSD)
  - FOM calculations (Walden, Schreier)
  - Noise/jitter limit calculations
  - Data validation utilities (validate_aout_data, validate_dout_data)

- **Complete Documentation Overhaul**:
  - 15 detailed algorithm documentation pages (fit_sine_4param, analyze_spectrum, etc.)
  - Updated installation guide emphasizing examples
  - Enhanced quickstart guide with learning path
  - Updated API reference docs

- **CLI Improvements**:
  - `adctoolbox-get-examples`: One-command example deployment
  - Organized output directory structure

### Fixed
- **CRITICAL**: Full-scale range calculation - DC offset no longer affects signal power measurements
  - Changed `max_scale_range` from `np.max(np.abs(data))` to `np.max(data) - np.min(data)`
  - File: `_prepare_fft_input.py:51-52`
  - Impact: Signals with DC offset now show correct power (was off by up to 27 dB!)

- **CRITICAL**: Power correction factor - Signal power was 6 dB too low
  - Changed `power_correction` from `4.0` to `16.0`
  - File: `compute_spectrum.py:67`
  - Impact: All power-related metrics (signal power, NSD) now correct

- **CRITICAL**: In-band SFDR search - SFDR now respects OSR parameter for oversampled ADCs
  - Implemented search limitation to in-band range `[0, Fs/2/OSR]`
  - File: `compute_spectrum.py:311-327`
  - Impact: Proper Delta-Sigma ADC analysis with OSR > 1

- **CRITICAL**: Spectrum normalization - Fundamental peak now shows at 0 dBFS instead of -1.76 dB
  - Added spectrum normalization: `spec_normalized_db = spec_mag_db - spec_mag_db[bin_idx]`
  - File: `compute_spectrum.py:211-213, 391`
  - Impact: Clear, correct spectrum visualization aligned with MATLAB

### Changed
- Enhanced `compute_spectrum()` with `coherent_averaging` parameter
- Enhanced `plot_spectrum()` with `plot_harmonics_up_to` parameter (default: 3)
- Updated `plot_spectrum()`, `plot_spectrum_polar()`, and `plot_two_tone_spectrum()` parameter names for consistency
- All aout functions now use absolute imports (`from adctoolbox.common.*` instead of `from ..common.*`)
- All aout functions now include MATLAB counterpart documentation in module docstrings

---

## [0.2.1] - 2025-12-06

### Added
- 21 ready-to-run examples organized in 3 categories:
  - Basic (b01-b04): Foundation functions
  - Analog Analysis (a01-a14): Processing recovered signal
  - Digital Analysis (d01-d05): Processing digital codes
- `adctoolbox-get-examples` CLI command to copy examples to workspace
- CI workflow testing basic examples (b01-b04)
- Comprehensive examples README with 3-step Quick Start
- Examples organized in `python/src/adctoolbox/examples/`

### Fixed
- `spec_plot` return value mismatch in `exp_b02_spectrum.py` (was expecting 9 values, returns 8)
- `inl_dnl_from_sine` now clips data before histogram (matches MATLAB implementation)
- `inl_dnl_from_sine` gives correct tiny INL/DNL (±0.2 LSB) for ideal signals

### Changed
- CI now tests example execution instead of pytest
- `err_pdf` function now does sine fitting internally
- `extract_static_nonlin` returns only k2, k3 (k1 removed)
- Updated README.md with Python-first Quick Start
- Examples use standard parameters for consistency (N=2^13, Fs=800MHz, etc.)

## [0.2.0] - 2025-12-04

### Added
- Restructured test suite into `integration/`, `unit/`, and `compare/` directories
- Added hamming window support to `spec_plot.py`
- Enhanced comparison logging with relative error tracking
- Converted verify scripts to proper pytest format

### Fixed
- Window function mismatch in `enob_bit_sweep` (was using boxcar, should use hamming)
  - Error reduced from 5.73e-03 to 2.22e-07 (25,000× improvement!)
- Critical bug fixes in MATLAB code identified

### Changed
- Pythonic API for window types (string-based: 'hann', 'hamming', 'boxcar')
- Test organization now matches MATLAB structure (run_* vs verify_*)

## [0.1.0] - 2025-11-28

### Added
- Initial GitHub Actions CI setup with smoke tests
- Complete Python-MATLAB validation (100% pass rate)
- Toolset functions: `toolset_aout.py`, `toolset_dout.py`
- Validation functions: `validate_aout_data.py`, `validate_dout_data.py`
- Analysis functions: `bit_activity.py`, `weight_scaling.py`, `enob_bit_sweep.py`

### Fixed
- CSV format handler for MATLAB-Python compatibility
- Import system cleanup (removed complex path manipulation)

### Changed
- Moved tests from `python/src/adctoolbox/test/` to `python/tests/`
- Package size reduced from ~50 MB to ~2-5 MB
- Standardized MATLAB test suite with uniform patterns

## [0.0.1] - 2025-01-28

### Added
- Initial Python package structure
- Core analysis functions ported from MATLAB:
  - `spec_plot`, `spec_plot_phase`, `tom_decomp`
  - `err_hist_sine`, `err_pdf`, `err_auto_correlation`, `err_envelope_spectrum`
  - `sine_fit`, `find_bin`, `find_fin`, `alias`
  - `fg_cal_sine`, `inl_sine`, `overflow_chk`
- MATLAB implementation with 17 core functions
- Test dataset with 40+ CSV files
- Basic documentation

---

## Version Format

- **MAJOR.MINOR.PATCH** (Semantic Versioning)
- **MAJOR**: Breaking API changes
- **MINOR**: New features, backwards compatible
- **PATCH**: Bug fixes, backwards compatible

## Links

- PyPI: https://pypi.org/project/adctoolbox/
- GitHub: https://github.com/yourusername/ADCToolbox
- Documentation: https://github.com/yourusername/ADCToolbox/blob/main/README.md
