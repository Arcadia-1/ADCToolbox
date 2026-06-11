Examples
================

ADCToolbox includes 63 ready-to-run examples organized into 10 categories. This
page demonstrates common use cases and analysis workflows.

Getting the Examples
--------------------

To copy all examples to your workspace:

.. code-block:: bash

    adctoolbox-get-examples

This creates an ``adctoolbox_examples/`` directory with all examples organized by category.

Running Examples
----------------

Navigate to the examples directory and run any example. Examples are organized by category:

.. code-block:: bash
    
    cd adctoolbox_examples
    python 01_basic/exp_b01_environment_check.py
    python 02_spectrum/exp_s01_analyze_spectrum_simplest.py
    python 02_spectrum/exp_s02_analyze_spectrum_interactive.py

Examples that generate figures save PNG outputs to an ``output/`` subdirectory
within each category folder.

**Category Folders:**

* ``02_spectrum/`` - Spectrum analysis examples
* ``03_generate_signals/`` - Signal generation examples
* ``04_debug_analog/`` - Analog output analysis examples
* ``05_debug_digital/`` - Digital output analysis examples
* ``06_use_toolsets/`` - Comprehensive dashboard examples
* ``07_conversions/`` - Conversion and metric calculation examples
* ``08_time_interleave/`` - Time-interleaved ADC analysis examples
* ``09_downsample/`` - Subsample and aliasing examples
* ``10_oversampling/`` - Oversampling and noise-shaping examples

Expected Outputs
----------------

For reference, selected expected outputs are documented below. These pages show
representative console output, figures, and validation results for the main
example groups.

.. toctree::
   :maxdepth: 1
   :caption: Expected Outputs by Category

   expected_output_02_spectrum
   expected_output_03_generate_signals
   expected_output_04_debug_analog
   expected_output_05_debug_digital
   expected_output_06_use_toolsets
   expected_output_07_conversions
