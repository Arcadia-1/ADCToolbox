Examples Gallery
================

ADCToolbox includes 51 ready-to-run examples organized into 6 categories. This gallery demonstrates common use cases and analysis workflows.

Getting the Examples
--------------------

To copy all examples to your workspace:

.. code-block:: bash

    adctoolbox-get-examples

This creates an ``adctoolbox_examples/`` directory with all examples organized by category.

Example Categories
------------------

Spectrum Analysis (14 examples)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Examples demonstrating spectrum computation, FFT metrics, and visualization:

* **exp_s01-s03**: Basic spectrum analysis (simplest, interactive, save figure)
* **exp_s04**: Dynamic range sweep
* **exp_s05**: Harmonic spur annotation
* **exp_s06**: FFT length and OSR sweep
* **exp_s07**: Power vs coherent averaging
* **exp_s08**: Windowing functions comparison
* **exp_s10-s12**: Polar spectrum visualization (noise, nonlinearity, averaging)
* **exp_s21-s23**: Two-tone spectrum analysis with IMD products

Signal Generation (6 examples)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Examples for creating test signals with various non-idealities:

* **exp_g01**: Thermal noise effects on signal quality
* **exp_g03**: Quantization noise scaling (2-16 bits)
* **exp_g04**: Jitter-induced SNR degradation
* **exp_g05**: Static nonlinearity harmonic distortion
* **exp_g06**: Isolated nonlinearity effects (8 types)
* **exp_g07**: Interference effects on spectrum (8 types)

Analog Output Analysis (15 examples)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Comprehensive analog output diagnostic examples:

* **exp_a01**: 4-parameter sine fitting
* **exp_a02-a04**: Error analysis (by value, by phase, jitter calculation)
* **exp_a11-a12**: Harmonic decomposition (time domain and polar)
* **exp_a21-a25**: Statistical analysis (PDF, spectrum, autocorrelation, envelope, comparison)
* **exp_a31-a32**: Nonlinearity fitting and INL/DNL extraction
* **exp_a41-a42**: Phase plane analysis (standard and error phase plane)

Digital Output Analysis (7 examples)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Examples for bit-weighted ADC analysis:

* **exp_d01-d03**: Weight calibration (lite, full, redundancy comparison)
* **exp_d11**: Bit activity checking
* **exp_d12**: ENOB bit sweep
* **exp_d13**: Weight scaling and radix analysis
* **exp_d14**: Overflow detection

Comprehensive Dashboards (4 examples)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All-in-one diagnostic dashboard generators:

* **exp_t01**: Single analog output dashboard (12 tools)
* **exp_t02**: Batch analog output dashboards (15 non-idealities)
* **exp_t03**: Single digital output dashboard (6 tools)
* **exp_t04**: Batch digital output dashboards (7 configurations)

Metric Calculation (5 examples)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Examples for computing figures of merit and unit conversions:

* **exp_b01**: Aliasing and Nyquist zone calculations
* **exp_b02**: Comprehensive unit conversions (9 categories)
* **exp_b03**: ADC figure of merit calculations (Walden FOM, Schreier FOM)
* **exp_b05**: Signal/noise amplitude to SNR conversions
* **exp_b06**: Noise spectral density (NSD) conversions

Running Examples
----------------

Navigate to the examples directory and run any example:

.. code-block:: bash

    cd adctoolbox_examples
    python exp_s01_analyze_spectrum_simplest.py

All examples save their outputs (plots, data files) to an ``output/`` subdirectory within each category folder.

Expected Console Outputs
------------------------

For reference, the expected console output from each example category is documented:

* :doc:`example_print_02_spectrum` - Spectrum analysis examples (14 examples)
* :doc:`example_print_03_generate_signals` - Signal generation examples (6 examples)
* :doc:`example_print_04_debug_analog` - Analog output analysis examples (15 examples)
* :doc:`example_print_05_debug_digital` - Digital output analysis examples (7 examples)
* :doc:`example_print_06_use_toolsets` - Comprehensive dashboard examples (4 examples)
* :doc:`example_print_07_calculate_metric` - Metric calculation examples (5 examples)

These documentation files show the exact console output and validation results you should expect when running each example.

Example Template
----------------

Most examples follow this structure:

.. code-block:: python

    """
    Example: Description of what this example demonstrates
    """

    import numpy as np
    from adctoolbox import function_name

    # Parameters
    N = 2**13          # Number of samples
    Fs = 800e6         # Sampling frequency: 800 MHz

    # Generate or load test data
    data = ...

    # Perform analysis
    result = function_name(data, ...)

    # Validate results
    assert condition, "Validation message"

    # Save outputs
    # (outputs saved to output/ directory)

For more details on specific examples, see the source code in the ``adctoolbox/examples/`` directory.
