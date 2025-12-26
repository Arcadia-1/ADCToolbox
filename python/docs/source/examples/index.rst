Examples Gallery
================

ADCToolbox includes 51 ready-to-run examples organized into 6 categories. This gallery demonstrates common use cases and analysis workflows.

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

All examples save their outputs (plots, data files) to an ``output/`` subdirectory within each category folder.

**Category Folders:**

* ``02_spectrum/`` - Spectrum analysis examples
* ``03_generate_signals/`` - Signal generation examples
* ``04_debug_analog/`` - Analog output analysis examples
* ``05_debug_digital/`` - Digital output analysis examples
* ``06_use_toolsets/`` - Comprehensive dashboard examples
* ``07_conversions/`` - Conversion and metric calculation examples

Expected Outputs
----------------

For reference, the expected outputs from each example category are documented below.
These documentation files show the console output, figures, and validation results you should expect when running each example.

.. toctree::
   :maxdepth: 1
   :caption: Expected Outputs by Category

   expected_output_02_spectrum
   expected_output_03_generate_signals
   expected_output_04_debug_analog
   expected_output_05_debug_digital
   expected_output_06_use_toolsets
   expected_output_07_conversions
