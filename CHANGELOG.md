# Changelog

All notable changes to ADCToolbox will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive spectrum examples (18 total):
  - Basic workflows (s00-s03): simplest → interactive → savefig → manual
  - FFT concepts (s04-s05): FFT length, OSR comparison
  - Windowing (s06-s09): spectral leakage, window types, coherent signals
  - Averaging methods (s10-s12): power averaging, coherent averaging, coherent + OSR
  - Polar visualization (s21-s24): polar plots, coherent averaging, kickback
  - Two-tone IMD (s31-s32): two-tone analysis, IMD comparison
- Complete API reference documentation in `agent_playground/ADCToolbox_API_Reference.md`
  - 100+ public functions documented
  - All module descriptions
  - Usage examples
  - Version history
- Debug scripts and documentation for spectrum analysis fixes in `agent_playground/`

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
