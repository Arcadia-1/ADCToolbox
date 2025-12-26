Changelog
=========

For the complete changelog with detailed version history, see the `CHANGELOG.md <https://github.com/Arcadia-1/ADCToolbox/blob/main/CHANGELOG.md>`_ file in the repository.

Version 0.4.0 (Latest)
----------------------

**Release Date**: 2025-12-18

**Documentation Release** - Complete Sphinx documentation overhaul with algorithm guides.

Added
~~~~~

* **Complete Documentation Overhaul**:

  - 15 detailed algorithm documentation pages with Python API
  - Updated installation guide emphasizing ``adctoolbox-get-examples``
  - Enhanced quickstart guide with learning path
  - All API reference docs updated to Python snake_case naming

Changed
~~~~~~~

* **Documentation Structure**:

  - Installation guide shortened, git clone moved to bottom
  - Quickstart restructured to start with basic examples (exp_b01, exp_b02, then exp_s01)
  - Used actual code from examples instead of synthetic snippets
  - Emphasized "Learning with Examples" throughout documentation

Removed
~~~~~~~

* Deleted 13 obsolete MATLAB-named algorithm documentation files
* Removed obsolete ``src/__init__.py`` file

Fixed
~~~~~

* Version number synchronization across all files
* Dynamic versioning in ``pyproject.toml``
* Documentation links and references updated to v0.4.0

Version 0.3.0
-------------

**Release Date**: 2025-12-18

**Major Refactoring Release** - Complete Python architecture modernization with 45 examples.

Breaking Changes
~~~~~~~~~~~~~~~~

* **API Naming**: All functions converted from MATLAB camelCase to Python snake_case

  - ``sineFit`` → ``fit_sine_4param``
  - ``INLsine`` → ``analyze_inl_from_sine``
  - ``specPlot`` → ``analyze_spectrum``
  - ``errPDF`` → ``analyze_error_pdf``
  - ``FGCalSine`` → ``calibrate_weight_sine``
  - And many more...

* **Module Structure**: Consolidated and reorganized for better maintainability

  - ``fundamentals``: Sine fitting, frequency utils, unit conversions, FOM metrics
  - ``spectrum``: Single-tone, two-tone, polar analysis
  - ``aout``: Analog error analysis (10 functions)
  - ``dout``: Digital calibration (3 functions)
  - ``siggen``: Signal generator with non-idealities
  - ``oversampling``: NTF analysis

* **Return Values**: All functions now return dictionaries instead of tuples for clarity

New Features
~~~~~~~~~~~~

* **45 Ready-to-Run Examples** (up from 21) across 6 categories:

  - ``01_basic/`` - Fundamentals (2 examples)
  - ``02_spectrum/`` - FFT-Based Analysis (14 examples)
  - ``03_generate_signals/`` - Non-Ideality Modeling (6 examples)
  - ``04_debug_analog/`` - Error Characterization (13 examples)
  - ``05_debug_digital/`` - Calibration & Redundancy (5 examples)
  - ``07_conversions/`` - Conversions (5 examples)

* **Enhanced Error Analysis**:

  - ``analyze_error_by_phase``: AM/PM decomposition
  - ``analyze_error_spectrum``: Error frequency analysis
  - ``analyze_decomposition_polar``: Polar harmonic visualization
  - ``fit_static_nonlin``: Extract k2/k3 coefficients

* **Expanded Fundamentals Module**:

  - Comprehensive unit conversions (dB, power, voltage, frequency, NSD)
  - FOM calculations (Walden, Schreier)
  - Noise/jitter limit calculations
  - Data validation utilities

* **CLI Improvements**:

  - ``adctoolbox-get-examples``: One-command example deployment
  - Organized output directory structure

Documentation
~~~~~~~~~~~~~

* **Complete Documentation Overhaul**:

  - All algorithm docs updated to Python API
  - 15 detailed algorithm documentation pages
  - Removed 13 obsolete MATLAB-named docs
  - Updated installation guide with emphasis on examples
  - Enhanced quickstart guide with learning path

* **New Algorithm Documentation**:

  - ``fit_sine_4param``: IEEE Std 1057/1241 sine fitting
  - ``analyze_inl_from_sine``: INL/DNL from histogram method
  - ``analyze_spectrum``: Comprehensive FFT analysis
  - ``analyze_error_by_phase``: AM/PM error decomposition
  - ``fit_static_nonlin``: Nonlinearity coefficient extraction
  - And 10 more detailed guides

Improvements
~~~~~~~~~~~~

* **Better API Consistency**: All analyze functions follow pattern: ``analyze_*(..., show_plot=True)``
* **Clearer Returns**: Dictionary returns with self-documenting keys
* **Enhanced Plotting**: Optional plotting with ``show_plot`` parameter, custom axes support
* **Validation**: Input validation with clear error messages
* **Type Hints**: Added to core functions for better IDE support

Bug Fixes
~~~~~~~~~

* Fixed frequency estimation edge cases in ``fit_sine_4param``
* Corrected INL/DNL clipping behavior in ``analyze_inl_from_sine``
* Improved numerical stability in weight calibration

Version 0.2.4
-------------

Legacy release with MATLAB-style naming conventions.

Features
~~~~~~~~

* 21 ready-to-run examples
* Analog output analysis (9 diagnostic tools)
* Digital output analysis (6 tools)
* Dual MATLAB and Python implementations
* Full documentation

Previous Versions
-----------------

For historical version information, please refer to the `CHANGELOG.md <https://github.com/Arcadia-1/ADCToolbox/blob/main/CHANGELOG.md>`_ file.
