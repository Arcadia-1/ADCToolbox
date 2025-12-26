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

Expected Console Outputs
------------------------

For reference, the expected console output from each example category is documented below.
These documentation files show the exact console output and validation results you should expect when running each example.

.. toctree::
   :maxdepth: 1
   :caption: Example Outputs by Category

   example_print_02_spectrum
   example_print_03_generate_signals
   example_print_04_debug_analog
   example_print_05_debug_digital
   example_print_06_use_toolsets
   example_print_07_conversions
