# Changelog

All notable changes to ADCToolbox will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- `fit_sine_4param` and every analysis that calls it (`analyze_error_pdf`,
  `analyze_error_spectrum`, `analyze_error_autocorr`,
  `analyze_error_envelope_spectrum`, `analyze_error_by_phase`,
  `analyze_error_by_value`, the harmonic decompositions, `fit_static_nonlin`,
  `estimate_frequency`, `sweep_performance_vs_osr`) now refine the frequency
  until it converges: `max_iterations` defaults to 100 instead of 1, as in
  MATLAB `sinfit`. Typical fits stop after 2 to 5 iterations. The default call
  no longer raises "did not converge in 1 iterations".
- `estimate_frequency(..., frequency_estimate=...)` takes the estimate in the
  units of `fs` (Hz when `fs` is in Hz). It was read as cycles per sample
  whatever `fs` was, so a value in Hz returned a frequency of about 1e17 Hz.
- `sweep_performance_vs_osr` raises `ValueError` for several records instead
  of concatenating them (four 60 dB records read as -2 dB SNDR).

### Fixed
- `fit_sine_4param` returned the amplitude and phase of the linearized step at
  the previous frequency. With the old single-step default a clean
  non-coherent tone kept a residual of up to -29 dBc (N=256) or -75 dBc
  (N=1024), which the error analyses reported as ADC error: the error PDF of
  an ideal 14-bit tone with 12.6 cycles read 1.43 LSB rms instead of 0.29.
  A, B and C are now the least-squares solution at the returned frequency.
- `fit_sine_4param` no longer returns a fit that is worse than its starting
  point (chirps, noise) or runs to the frequency limit with an amplitude of
  several hundred thousand (odd-length records with the tone in the top bin,
  less than one cycle in the record). It falls back to the initial estimate,
  sets `converged=False` and warns. The FFT start now includes the top bin.
- `fit_static_nonlin` returned `k2/g` and `k3/g^2`, with
  `g = 1 + 0.75*k3*A^2` the gain of the cubic term on the fundamental
  (true 0.10/-0.30 at A=1 read 0.129/-0.499), and was biased on records with a
  non-integer number of cycles (k2 12 % high at 7.4 cycles).
- `calibrate_weight_sine_lite` returned near-zero weights for an input sine
  starting at zero phase, and 110 LSB of error at phase pi. It now solves both
  quadrature gauges like `calibrate_weight_sine`.
- `calibrate_weight_sine` with a per-record `freq` array that mixes known
  values and 0 placeholders returned all-zero weights with `snr_db` of 200.
  Placeholders are now estimated. A 0-d array `freq` no longer raises, and the
  residual SNR is NaN, not 200 dB, when there is no reconstructed signal.
- In-band weight calibration (`osr > 1`) with an automatic frequency started
  the tone search up to 0.93 bin away when the tone sat just below the cutoff.
- `analyze_overflow` missed all-ones codes at some bit positions with
  non-integer weights (0 % instead of 1.19 %), because of an exact float
  comparison.
- `scale_calibration_output(target_weights=...)` doubled the scale when a bit
  never toggled in the capture.
- Coherent averaging with any window other than rectangular lost the window's
  side bins across runs: signal power read 1.7 dB low with Hann (the default),
  3 dB with Blackman-Harris and 5 dB with Kaiser, and SNR dropped with it.
  MATLAB `plotspec` has the same alignment and is not changed here.
- Auto `side_bin` was cut to the distance from DC or from the band edge, so
  part of the tone's own lobe counted as distortion (Blackman-Harris tone at
  bin 2: SNDR 38.9 dB for a 60 dB capture; OSR 16 near the band edge: 16.9 dB).
- `compute_spectrum` on integer codes with no declared range overflowed in
  the input dtype (int16 codes: signal at +20.7 dBFS). `max_scale_range` now
  also accepts a NumPy array, and power averaging skips all-zero records as
  coherent averaging already did.
- `ADC_Signal_Generator.apply_jitter(input_signal=...)` under-applied jitter
  near and above Nyquist (SNR 10 dB too high at fin/fs = 0.45, 19 dB at 0.9).
- `analyze_error_by_value` without `value_range` dropped or collapsed signals
  outside [0, 1], [-0.5, 0.5] and [-1, 1] (1.2 +/- 0.7 V lost 40 % of its
  samples; a +/-5 mV signal filled 2 of 100 bins).
- `analyze_error_by_phase` and `analyze_error_by_value` report the fitted
  frequency in `norm_freq`, not the caller's estimate.
- `validate_dout_data` accepts a boolean bit matrix; `predict_spurs` raises
  the intended `ValueError` when `fin` is missing.

### Removed
- 16 test functions (23 collected cases) that duplicated another test or
  asserted nothing a stricter test does not already check, and three test
  scripts that pytest never collected
  (`tests/run_full_test_suite.py`, `tests/compare/run_analog_comparisons.py`,
  `tests/compare/run_digital_comparisons.py`).

## [0.9.3] - 2026-11-10

**In-Band Calibration and Algorithm Fixes** - adds filtered weight fitting
for oversampled ADCs and corrects linearity, spectrum and calibration results.

### Added
- Python `calibrate_weight_sine(..., osr=...)` and
  `calibrate_weight_sine_lite(..., osr=...)` fit a lowpass-filtered residual for
  oversampled/noise-shaping ADCs. The same FIR filters bit, reference, DC and
  frequency-derivative columns independently per capture. The default `osr=1`
  keeps the existing full-band path; output waveforms keep their original
  length and sample rate. Full calibration adds `inband_snr_db`, `inband_enob`
  and `calibration_filter` metadata (#128).
- A held-out noise-shaping SAR calibration example, with spectra, weight
  comparisons and JSON metrics, in `10_oversampling/exp_o04_inband_weight_calibration.py`.

### Fixed
- Sine-histogram DNL/INL code labels and width normalization. Missing codes
  are reported at the correct code with DNL exactly -1; clipping no longer
  causes artificial INL drift. Python and MATLAB are aligned (#118).
- Multi-record sine calibration reconstructs each record's reference and
  frequency derivative correctly and retains its own DC offset. Multi-record
  `offset` is now an array; single-record behavior is preserved (#119).
- Explicit `full_scale` takes precedence over the voltage/code span heuristic
  for floating-point sine-linearity input (#120).
- SFDR selects the strongest integrated spur lobe rather than the largest
  individual peak bin (#121).
- Median noise estimation uses the correct statistics for power versus
  coherent averaging and the effective number of accepted records. Plot
  labels also use the accepted-record count (#122).
- Static nonlinearity generation meets the requested harmonic levels and
  preserves the input DC operating point (#123).
- Odd-length FFT boundary handling no longer halves the highest-bin filtered
  tone or produces invalid harmonic indices (#117).
- Copied foreground-calibration examples run outside the installed package
  while retaining their source snapshots; wheel integration tests also work
  in uv/macOS environments (#117).
- Farrow fractional delay preserves short input lengths; error-spectrum
  calls with `create_plot=False` preserve existing figures; low-frequency
  spectrum labels use the correct Hz units (#117).
- MATLAB rectangular-window coherent tones no longer trigger spurious peak
  interpolation and overly wide automatic signal masks (#125).

## [0.9.2] - 2026-11-03

**Calibration and SFDR Release** - adds Dither and supervised foreground
calibration APIs, fixes optimistic SFDR at masked spectrum boundaries, and
warns when the auto signal window hides a spur.

### Added
- Dither calibration (#102): public `DitherGainTracker`, `apply_dither_gain` and
  `DitherHDCTracker` in `adctoolbox` and `adctoolbox.calibration`, the
  `11_calibration/` examples (PN gain tracking, joint identification,
  fixed-point three-PN HDC, architecture controls) and an optional `accel`
  extra (`pip install "adctoolbox[accel]"`) that uses Numba for the HDC loop.
- Supervised foreground calibration (#103): `build_calibration_features`,
  `fit_calibration_polynomial`, `fit_calibration_mlp` and `apply_calibration`,
  with the `12_foreground/` examples.
- `analyze_error_envelope_spectrum(..., input_kind="residual")` analyzes an
  already computed error signal without refitting a sine, matching MATLAB
  `errevspec` (#90).
- `convert_cap_to_weight(..., input_order=..., output_order=...)` makes the
  LSB/MSB ordering explicit; `"msb_to_lsb"` matches MATLAB `cdacwgt` (#95).
- `ntfperf(..., n_grid=...)`; the default grid now matches MATLAB `ntfperf.m` (#92).
- `compute_spectrum` and `analyze_spectrum` accept `side_bin="auto"` as an alias
  for `None`, as `quick_sndr` already did. Numeric `side_bin` values are
  truncated to a non-negative integer, as in `quick_sndr`; other strings raise
  `TypeError`.
- Auto `side_bin` (the `compute_spectrum` default and `quick_sndr(side_bin="auto")`)
  raises a `UserWarning` when the detected signal window contains a spur or
  second tone that is at least 10 dB above the fundamental's leakage. With
  non-coherent sampling and a Hann window the window can span tens of bins, and
  such a component is counted as signal power, so SFDR and SNDR read
  optimistic. Metrics are unchanged.

### Changed
- `analyze_enob_sweep` gains `calibration_mode`. The default,
  `"prefix_of_full_calibration"`, calibrates all bits once and sweeps prefixes
  of that solution; `"recalibrate_each_subset"` refits each prefix (#89, #96).
  MATLAB `bitsweep` now also calibrates once with all bits instead of refitting
  each prefix.
- The errsin compatibility wrappers in `adctoolbox.aout` use the new
  MATLAB-compatible `errsin_compat` residual binning (#94).
- Documentation moved to <https://adctoolbox.tokenzhang.com/doc/>.

### Fixed
- SFDR read up to 1.8 dB optimistic when the spur's lobe touched the masked
  signal or DC bins or the band edge; the hidden part of the lobe is now
  restored from the fundamental's lobe-to-peak ratio. MATLAB `plotspec`
  `sideBin='auto'` no longer collapses to 0 on a noise-free record (#100, #101).
- `predict_spurs` reported every offset tone except the one at fs/2 twice, as the
  DFT coefficients k and M-k, each at half the tone's amplitude, so offset spurs
  read 6.02 dB low. It now reports each offset tone once (k = 1 ... M/2) at its
  full amplitude, and the integration test checks every predicted spur against
  the measured spectrum rather than the three largest.
- The MATLAB test runner now runs from the right working directory.
- Packaged `11_calibration` guides gave commands that only worked in the
  contributor's checkout; they now run from a copied examples workspace.

## [0.9.1] - 2026-07-06

**Contract Clarification Patch** - resolves coherent-bin selection and weight
radix output-shape ambiguities between Python and MATLAB.

### Added
- `find_coherent_frequency(..., policy="matlab_findbin")` for explicit parity
  with MATLAB `findbin` bin selection.

### Changed
- MATLAB `plotwgt` and legacy `weightScaling` now return radix vectors aligned
  to bit indices: `radix(1) = NaN`, `radix(i) = |weight(i-1)/weight(i)|`.

## [0.9.0] - 2026-06-11

**Oversampling Parity Release** — adds MATLAB-compatible oversampling entry
points, noise-shaping examples, and a MATLAB `noiseshape` helper.

### Added
- Python MATLAB-compatible oversampling APIs:
  - `adctoolbox.ifilter` / `adctoolbox.oversampling.ifilter`
  - `adctoolbox.perfosr` / `adctoolbox.oversampling.perfosr`
  - `adctoolbox.ntfperf` / `adctoolbox.oversampling.ntfperf`
- New `10_oversampling/` Python example group:
  - noise-shaped quantization spectrum analysis
  - in-band extraction with `ifilter`
  - NTF theory plus performance-vs-OSR sweep
- MATLAB `matlab/src/noiseshape.m` for lightweight noise-shaped quantization
  signal generation, with `matlab/tests/common/test_noiseshape.m`.
- Unit tests for MATLAB-compatible oversampling APIs and oversampling examples.

### Changed
- `ADC_Signal_Generator.apply_noise_shaping()` now accepts custom NTF
  coefficients.
- `sweep_performance_vs_osr()` supports MATLAB-style `logscale` and `smooth`
  options.
- `extract_freq_components()` now validates real-valued inputs and normalized
  passbands in `[0, 0.5]`.
- CI now runs oversampling unit tests and the `10_oversampling/` example
  scripts.

## [0.8.3] - 2026-05-26

**Example Output Cleanup Release** — keeps SAR calibration examples focused on
figures and avoids writing CSV side artifacts.

### Changed
- SAR digital-debug examples now save PNG figures only; intermediate Monte
  Carlo and sweep statistics remain in memory for plotting instead of being
  written to CSV files.
- Example console messages no longer advertise CSV side outputs.

## [0.8.2] - 2026-05-25

**SAR Calibration Examples Release** — adds focused SAR mismatch and
foreground-calibration examples, and refines SAR mismatch modeling.

### Added
- New SAR digital-debug examples:
  - `exp_d15_sar_unit_cap_mismatch_uncal_spectra.py`
  - `exp_d16_sar_unit_cap_mismatch_mc.py`
  - `exp_d17_sar_msb_error_binary_vs_repeat_calibration.py`
  - `exp_d18_sar_redundant_mismatch_training_length_sweep.py`
- `sar_apply_cap_mismatch`, a Pelgrom/unit-cap-scaled SAR CDAC mismatch helper
  with optional explicit capacitor unit counts.
- Spectrum plot regression coverage for stable annotation placement after
  caller-side axis-limit changes.

### Changed
- SAR examples and bundled skill docs now use the explicit
  `sar_apply_cap_mismatch` name.
- Standard and Virtuoso spectrum plots keep metric annotations fixed in axes
  coordinates, making subplot and post-plot y-limit changes more robust.

### Fixed
- `sar_apply_mismatch` remains available with its legacy per-weight gaussian
  perturbation semantics for backward compatibility.

## [0.8.1] - 2026-05-24

**Spectrum Plot Calibration Patch** — aligns Python windowed spectrum bin
heights with MATLAB `plotspec.m`.

### Fixed
- Python spectrum plotting now uses MATLAB-style RMS window power scaling, so
  coherent Hann full-scale tones report the center bin below 0 dBFS while the
  full main-lobe sum remains 0 dBFS.
- `compute_spectrum` no longer applies an extra ENBW division to the integrated
  signal power after window RMS scaling.
- Virtuoso-style spectrum plots use the same raw dBFS noise-line convention as
  the standard plotter.

## [0.8.0] - 2026-05-18

**SAR + Spectrum Robustness Release** — behavior-model API cleanup, stronger FFT side-bin handling, and MATLAB data-generation parity updates.

### Added
- **ADC behavioral models submodule** (`adctoolbox.models`):
  - `sar_convert`, `sar_reconstruct`, `sar_ideal_weights`, `sar_apply_mismatch`:
    binary / sub-radix-2 SAR forward model with explicit CDAC weights,
    `quant_range=(v_min, v_max)`, sampling noise, comparator noise, and cap mismatch support.
  - Convention: input is interpreted relative to `quant_range`; SAR weights are
    normalized by `sum(bit_weights) + 1 LSB`. A non-redundant 4-bit ADC uses
    `[8, 4, 2, 1] / 16`; a redundant `[8, 4, 4, 2, 1]` array uses `/20`.
  - 16 pytest cases including ENoB=N validation at 4, 8, 12, 16, 20 bit
    (passes to within FFT-noise-floor tolerance, ±0.05 b for N ≥ 12).
- **Spectrum side-bin regression coverage**:
  - Added near-Nyquist SAR FFT-length example.
  - Added tests for finite axis handling and side-bin defaults.
- **MATLAB data generation scripts**:
  - Added data-generation scripts for sinewave non-idealities, SAR dout, pipeline dout, jitter sweeps, and batch generation.
- **Codex skill installer status / editable modes**:
  - `adctoolbox-install-skill --status --dest <skills-dir>` reports whether
    bundled skills are missing, copied, symlinked, and in sync.
  - `adctoolbox-install-skill --editable --dest <skills-dir>` installs
    symlinks for local skill development.

### Changed
- Spectrum helpers now use safer automatic side-bin defaults and more robust noise-floor display handling near edge cases.
- `matlab/src/plotspec.m` now handles Nyquist-bin cases more robustly.
- `adctoolbox-install-skill` now requires an explicit `--dest`; it no longer
  writes to `~/.codex/skills` or `$CODEX_HOME/skills` implicitly.

## [0.7.0] - 2026-04-29

**Time-Interleave + Skill Overhaul Release** — new TI-ADC submodule and a fully revamped `adctoolbox-user-guide` skill.

### Added
- **Time-Interleave (TI-ADC) submodule** (`adctoolbox.timeinterleave`):
  - `deinterleave` / `interleave`: stream <-> per-channel sub-streams
  - `extract_mismatch_sine`: extract gain/offset/timing mismatch from sine input
  - `predict_spurs`: predict TI mismatch spur locations and amplitudes
  - `fractional_delay_fft` / `fractional_delay_farrow`: fractional-delay primitives
  - `calibrate_foreground`: foreground timing-skew calibration built on the FD primitives
  - 3 examples (`08_timeinterleave/`) covering deinterleave/interleave, foreground calibration, and autocorrelation-based background skew calibration (aligned with MATLAB + reference paper)
- **`exp_s00_fft_fundamentals.py`** example: pedagogical N=15 vs N=16 FFT comparison

### Changed
- **`adctoolbox-user-guide` skill rewritten** with progressive disclosure:
  - SKILL.md slimmed; advanced debug content moved to `references/`
  - `api-quickref` partitioned into Basic / Advanced sections
  - `example-map` extended with an advanced examples section
  - Added executable smoke tests; aligned skill + README to the real public API
  - Eval prompt suite locked with baseline + post-revamp run

### Fixed
- Stale docstring examples corrected
- Skill / README API drift: docs now match the current public API surface

### Removed
- **Two-tone spectrum analysis** (carried over from Unreleased): removed `analyze_two_tone_spectrum`, `compute_two_tone_spectrum`, `plot_two_tone_spectrum` and the 3 two-tone example scripts (`exp_s21`–`exp_s23`); no MATLAB counterpart existed

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
