Quick Start Guide
=================

This guide will help you get started with ADCToolbox quickly.

Spectrum Analysis
-----------------

Analyze an ADC output spectrum:

.. code-block:: python

    import numpy as np
    from adctoolbox import analyze_spectrum, find_coherent_frequency

    # Generate test signal
    N = 2**13
    Fs = 100e6
    Fin_target = 10e6
    Fin, _ = find_coherent_frequency(fs=Fs, fin_target=Fin_target, n_fft=N)

    t = np.arange(N) / Fs
    A = 0.5
    DC = 0.5
    noise_rms = 10e-6
    signal = A * np.sin(2*np.pi*Fin*t) + DC + np.random.randn(N) * noise_rms

    # Analyze spectrum
    result = analyze_spectrum(signal, fs=Fs)

    # Access metrics
    print(f"ENOB: {result['enob']:.2f} bits")
    print(f"SNDR: {result['sndr_db']:.2f} dB")
    print(f"SFDR: {result['sfdr_db']:.2f} dB")
    print(f"SNR: {result['snr_db']:.2f} dB")

Using Toolsets
--------------

Analog Output Dashboard
~~~~~~~~~~~~~~~~~~~~~~~

Generate a comprehensive 12-panel diagnostic dashboard for analog ADC output:

.. code-block:: python

    import numpy as np
    from adctoolbox.toolset import generate_aout_dashboard
    from adctoolbox import find_coherent_frequency

    # Generate test signal
    N = 2**16
    Fs = 800e6
    Fin_target = 10e6
    Fin, _ = find_coherent_frequency(fs=Fs, fin_target=Fin_target, n_fft=N)

    t = np.arange(N) / Fs
    A = 0.49
    DC = 0.5
    noise_rms = 50e-6
    resolution = 12
    signal = A * np.sin(2*np.pi*Fin*t) + DC + np.random.randn(N) * noise_rms

    # Generate dashboard with 12 analysis plots
    fig, axes = generate_aout_dashboard(
        signal=signal,
        fs=Fs,
        freq=Fin,
        resolution=resolution,
        output_path='aout_dashboard.png'
    )

Digital Output Dashboard
~~~~~~~~~~~~~~~~~~~~~~~~

Generate a comprehensive 6-panel diagnostic dashboard for digital ADC bits:

.. code-block:: python

    import numpy as np
    from adctoolbox.toolset import generate_dout_dashboard
    from adctoolbox import find_coherent_frequency

    # Generate test signal
    N = 2**13
    Fs = 1e9
    Fin_target = 300e6
    Fin, _ = find_coherent_frequency(fs=Fs, fin_target=Fin_target, n_fft=N)

    t = np.arange(N) / Fs
    A = 0.49
    DC = 0.5
    resolution = 12

    # Generate quantized signal and extract bits
    signal = A * np.sin(2*np.pi*Fin*t) + DC
    quantized_signal = np.clip(np.floor(signal * (2**resolution)), 0, 2**resolution - 1).astype(int)
    bits = (quantized_signal[:, None] >> np.arange(resolution - 1, -1, -1)) & 1

    # Generate dashboard with 6 analysis plots
    fig, axes = generate_dout_dashboard(
        bits=bits,
        freq=Fin/Fs,  # Normalized frequency
        weights=None,  # Use binary weights by default
        output_path='dout_dashboard.png'
    )

Next Steps
----------

* Explore the :doc:`api/index` for detailed function documentation
* Read the :doc:`algorithms/index` for algorithm theory
* Try the :doc:`examples/index` for more use cases
* Check the :doc:`changelog` for version history
