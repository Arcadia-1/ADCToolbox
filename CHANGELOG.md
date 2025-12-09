# Changelog

All notable changes to ADCToolbox will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **BREAKING**: All dout functions renamed to match filenames for consistency
  - `cal_weight_sine()` → `calibrate_weight_sine()`
  - `cal_weight_sine_os()` → `calibrate_weight_sine_osr()`
  - `cal_weight_sine_2freq()` → `calibrate_weight_two_tone()`
  - `bit_activity()` → `check_bit_activity()`
  - `overflow_chk()` → `check_overflow()`
  - `weight_scaling()` → `plot_weight_radix()`
  - `sweep_bit_enob()` → `analyze_enob_sweep()`
  - Updated all imports across 16 files (examples, tests, toolsets)
- **BREAKING**: `analyze_spectrum()` now returns a dictionary instead of tuple
  - Before: `enob, sndr, sfdr, snr, thd, pwr, nf, nsd = analyze_spectrum(...)`
  - After: `result = analyze_spectrum(...)`  → Access via `result['enob']`, `result['sndr_db']`, etc.
  - Dictionary keys: `enob`, `sndr_db`, `sfdr_db`, `snr_db`, `thd_db`, `sig_pwr_dbfs`, `noise_floor_db`, `nsd_dbfs_hz`
- **BREAKING**: `plot_envelope_spectrum()` now returns a dictionary (same structure as `analyze_spectrum`)
- All aout functions now use absolute imports (`from adctoolbox.common.*` instead of `from ..common.*`)
- All aout functions now include MATLAB counterpart documentation in module docstrings

### Fixed
- Fixed `ModuleNotFoundError` in `calibrate_weight_two_tone.py`
  - Changed `from ..common.alias import alias` to `from adctoolbox.common.calc_aliased_freq import calc_aliased_freq`
  - Updated all `alias()` calls to `calc_aliased_freq()`
- Fixed `IndexError` in `analyze_spectrum` when indexing harmonics
  - Added `int()` wrapper around `calc_aliased_freq()` calls to ensure integer indices
- Fixed `exp_a01` example to use `find_coherent_frequency` instead of deprecated `find_bin`

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
