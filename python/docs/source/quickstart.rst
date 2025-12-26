Quick Start Guide
=================

This guide will help you get started with ADCToolbox quickly.

Learning with Examples (Recommended)
-------------------------------------

**The best way to learn ADCToolbox is through the 45 ready-to-run examples.**

Get All Examples
~~~~~~~~~~~~~~~~

.. code-block:: bash

    cd /path/to/your/workspace
    adctoolbox-get-examples

This creates an ``adctoolbox_examples/`` directory with examples organized into 6 categories:

* **01_basic/** - Fundamentals (2 examples)
* **02_spectrum/** - FFT-Based Analysis (14 examples)
* **03_generate_signals/** - Non-Ideality Modeling (6 examples)
* **04_debug_analog/** - Error Characterization (13 examples)
* **05_debug_digital/** - Calibration & Redundancy (5 examples)
* **07_conversions/** - Conversions (5 examples)

Run Your First Examples
~~~~~~~~~~~~~~~~~~~~~~~~

Start with the basics, then move to spectrum analysis:

.. code-block:: bash

    cd adctoolbox_examples/01_basic

    # Verify environment
    python exp_b01_environment_check.py

    # Learn coherent sampling
    python exp_b02_coherent_vs_non_coherent.py

    # Move to spectrum analysis
    cd ../02_spectrum
    python exp_s01_analyze_spectrum_simplest.py

    # Try more spectrum examples
    python exp_s02_analyze_spectrum_interactive.py
    python exp_s21_analyze_two_tone_spectrum.py

All outputs are saved to the ``output/`` directory within each category.

Browse More Examples
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Spectrum analysis (14 examples)
    cd 02_spectrum
    python exp_s01_analyze_spectrum_simplest.py
    python exp_s04_sweep_dynamic_range.py
    python exp_s08_windowing_deep_dive.py

    # Signal generation (6 examples)
    cd ../03_generate_signals
    python exp_g01_generate_signal_demo.py

    # Analog debugging (13 examples)
    cd ../04_debug_analog
    python exp_a01_fit_sine_4param.py
    python exp_a21_analyze_error_pdf.py

    # Digital debugging (5 examples)
    cd ../05_debug_digital
    python exp_d01_bit_activity.py
    python exp_d02_cal_weight_sine.py

    # Metrics & utilities (5 examples)
    cd ../07_conversions
    python exp_b01_aliasing_nyquist_zones.py

Basic Usage
-----------

Spectrum Analysis
~~~~~~~~~~~~~~~~~

Analyze an ADC output spectrum:

.. code-block:: python

    import numpy as np
    from adctoolbox import analyze_spectrum, amplitudes_to_snr, snr_to_nsd

    # Generate test signal
    N_fft = 2**13
    Fs = 100e6
    Fin = 123/N_fft * Fs  # Coherent frequency
    t = np.arange(N_fft) / Fs
    A = 0.5
    noise_rms = 10e-6
    signal = 0.5 * np.sin(2*np.pi*Fin*t) + np.random.randn(N_fft) * noise_rms

    # Calculate theoretical metrics
    snr_ref = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=noise_rms)
    nsd_ref = snr_to_nsd(snr_ref, fs=Fs, osr=1)
    print(f"Theoretical SNR: {snr_ref:.2f} dB, NSD: {nsd_ref:.2f} dBFS/Hz")

    # Analyze spectrum
    result = analyze_spectrum(signal, fs=Fs)

    # Access metrics
    print(f"ENOB: {result['enob']:.2f} bits")
    print(f"SNDR: {result['sndr_db']:.2f} dB")
    print(f"SFDR: {result['sfdr_db']:.2f} dB")
    print(f"SNR: {result['snr_db']:.2f} dB")
    print(f"NSD: {result['nsd_dbfs_hz']:.2f} dBFS/Hz")

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

Analyze errors in ADC output using various methods:

.. code-block:: python

    from adctoolbox import (
        analyze_error_pdf,
        analyze_error_spectrum,
        analyze_error_autocorr,
        analyze_error_envelope_spectrum
    )

    # Error PDF (probability distribution)
    result_pdf = analyze_error_pdf(data, resolution=12, show_plot=True)
    print(f"Error std: {result_pdf['sigma']:.2f} LSB")

    # Error spectrum (frequency domain)
    result_spectrum = analyze_error_spectrum(data, fs=800e6, show_plot=True)

    # Error autocorrelation (temporal correlation)
    result_autocorr = analyze_error_autocorr(data, max_lag=100, show_plot=True)

    # Error envelope spectrum (AM patterns)
    result_envelope = analyze_error_envelope_spectrum(data, fs=800e6, show_plot=True)

Next Steps
----------

* Explore the :doc:`api/index` for detailed function documentation
* Read the :doc:`algorithms/index` for algorithm theory
* Try the :doc:`examples/index` for more use cases
* Check the :doc:`changelog` for version history
