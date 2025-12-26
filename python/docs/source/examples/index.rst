Examples Gallery
================

ADCToolbox includes 21 ready-to-run examples organized into 6 categories. This gallery demonstrates common use cases and analysis workflows.

Getting the Examples
--------------------

To copy all examples to your workspace:

.. code-block:: bash

    adctoolbox-get-examples

This creates an ``adctoolbox_examples/`` directory with all 21 examples.

Example Categories
------------------

Basic Examples (4 examples)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **exp_b01_plot_sine.py**: Sine wave visualization
* **exp_b02_spectrum.py**: FFT spectrum analysis
* **exp_b03_sine_fit.py**: Fit sine wave to noisy data
* **exp_b04_aliasing.py**: Nyquist zones demonstration

Spectrum Analysis (varies)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Examples demonstrating spectrum computation, FFT metrics, and visualization:

* Compute single-tone and two-tone spectra
* Calculate SFDR, SNDR, ENOB
* Generate polar spectrum plots

Signal Generation (varies)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Examples for creating test signals:

* Coherent frequency calculation
* Test signal generation
* Multi-tone signal synthesis

Analog Output Analysis (14 examples)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Comprehensive analog output diagnostic examples:

* **exp_a01_fit_sine_4param.py**: Basic 4-parameter sine fitting
* **exp_a11-a14_analyze_error_by_*.py**: Error analysis by value/phase
* **exp_a21_analyze_error_pdf.py**: Error PDF analysis
* **exp_a22_analyze_error_spectrum.py**: Error spectrum analysis
* **exp_a23_analyze_error_autocorr.py**: Error autocorrelation
* **exp_a24_analyze_error_envelope_spectrum.py**: Envelope spectrum analysis
* **exp_a31-a34_analyze_harmonic_*.py**: Harmonic decomposition analysis
* **exp_a41_analyze_inl_from_sine.py**: INL/DNL analysis from sine wave

Digital Output Analysis (5 examples)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Examples for bit-weighted ADC analysis:

* **exp_d01_check_bit_activity.py**: Pipeline bit activity checking
* **exp_d02_calibrate_weight_sine.py**: Foreground calibration with sine wave
* **exp_d03_check_overflow.py**: Overflow detection
* **exp_d04_analyze_enob_sweep.py**: ENOB vs redundancy analysis
* **exp_d05_plot_weight_radix.py**: Weight/radix visualization

Metric Calculation (varies)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Examples for computing various figures of merit and performance metrics.

Running Examples
----------------

Navigate to the examples directory and run any example:

.. code-block:: bash

    cd adctoolbox_examples
    python exp_b01_plot_sine.py

All examples save their outputs (plots, data files) to an ``output/`` subdirectory within the examples folder.

Example Output
--------------

Each example includes:

* Clear documentation explaining the purpose
* Standard test parameters (N=2^13, Fs=800MHz, etc.)
* Assertion-based validation
* Output to ``output/`` subdirectory
* Descriptive plot filenames

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
