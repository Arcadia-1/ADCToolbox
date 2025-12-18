Quick Start Guide
=================

This guide will help you get started with ADCToolbox quickly.

Basic Usage
-----------

Spectrum Analysis
~~~~~~~~~~~~~~~~~

Analyze an ADC output spectrum:

.. code-block:: python

    import numpy as np
    from adctoolbox import compute_spectrum

    # Load ADC data
    data = np.loadtxt('adc_output.csv')

    # Compute spectrum
    result = compute_spectrum(
        data,
        fs=800e6,           # Sampling frequency: 800 MHz
        window='hann',      # Window function
        nfft=8192          # FFT points
    )

    # Access metrics
    print(f"SFDR: {result['metrics']['sfdr_db']:.2f} dB")
    print(f"ENOB: {result['metrics']['enob']:.2f} bits")

Sine Fitting
~~~~~~~~~~~~

Fit a sine wave to ADC data:

.. code-block:: python

    from adctoolbox import fit_sine_4param

    # Fit sine wave
    result = fit_sine_4param(data, frequency_estimate=0.1)

    # Extract parameters
    print(f"Amplitude: {result['amplitude']:.4f}")
    print(f"Frequency: {result['frequency']:.6f}")
    print(f"Phase: {result['phase']:.4f} rad")
    print(f"DC Offset: {result['dc_offset']:.4f}")

INL/DNL Analysis
~~~~~~~~~~~~~~~~

Analyze INL and DNL from sine wave data:

.. code-block:: python

    from adctoolbox import analyze_inl_from_sine

    # Compute INL/DNL
    result = analyze_inl_from_sine(data, output_dir='output')

    # Results include:
    # - INL/DNL plots
    # - Statistical metrics
    # - Data arrays for further analysis

Using Toolsets
--------------

Run Complete Analog Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Execute all 9 analog diagnostic tools at once:

.. code-block:: python

    from adctoolbox.aout import toolset_aout

    # Run all analog analysis tools
    status = toolset_aout(
        aout_data,
        output_dir='output/test1',
        visible=False    # Set to True to display plots
    )

    # Creates 9 diagnostic plots + 1 panel overview

Run Complete Digital Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Execute digital output analysis tools:

.. code-block:: python

    from adctoolbox.dout import toolset_dout
    import numpy as np

    # Load digital output bits
    bits = np.loadtxt('sar_bits.csv', delimiter=',')

    # Run digital analysis tools
    status = toolset_dout(
        bits,
        output_dir='output/test1',
        visible=False
    )

    # Creates 3 diagnostic plots + 1 panel overview

Working with Examples
---------------------

Copy Examples to Your Workspace
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    cd /path/to/your/workspace
    adctoolbox-get-examples

Run Example Scripts
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    cd adctoolbox_examples

    # Basic examples
    python exp_b01_plot_sine.py
    python exp_b02_spectrum.py

    # Analog analysis examples
    python exp_a01_spec_plot_nonidealities.py
    python exp_a03_err_pdf.py

    # Digital analysis examples
    python exp_d01_bit_activity.py
    python exp_d02_fg_cal_sine.py

All outputs are saved to the ``output/`` directory.

Common Patterns
---------------

Coherent Frequency Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calculate a coherent test frequency for ADC testing:

.. code-block:: python

    from adctoolbox import find_coherent_frequency

    freq = find_coherent_frequency(
        fin_desired=100e6,  # Desired input frequency: 100 MHz
        fs=800e6,           # Sampling frequency: 800 MHz
        N=8192,             # Number of samples
        find_bin=True       # Return FFT bin index
    )

    print(f"Coherent frequency: {freq['frequency']:.2f} Hz")
    print(f"FFT bin: {freq['bin']}")

Error Analysis
~~~~~~~~~~~~~~

Analyze errors in ADC output by code value:

.. code-block:: python

    from adctoolbox import analyze_error_by_value

    result = analyze_error_by_value(
        data,
        output_dir='output',
        freq_cal=None    # Auto-detect frequency
    )

    # Creates error vs. code plot with statistics

Next Steps
----------

* Explore the :doc:`api/index` for detailed function documentation
* Read the :doc:`algorithms/index` for algorithm theory
* Try the :doc:`examples/index` for more use cases
* Check the :doc:`changelog` for version history
