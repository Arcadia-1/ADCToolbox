.. rst-class:: adt-home-page

ADCToolbox
==========

.. raw:: html

   <section class="adt-hero">
     <div class="adt-hero-content">
       <p class="adt-kicker">ADC characterization for Python</p>
       <h1>ADCToolbox</h1>
       <p class="adt-lede">
         Spectrum analysis, SAR modeling, digital calibration, and debug
         dashboards for converter development.
       </p>
       <div class="adt-actions">
         <a class="adt-button adt-button-primary" href="quickstart.html">Quick start</a>
         <a class="adt-button adt-button-secondary" href="api/index.html">API reference</a>
       </div>
       <div class="adt-install" role="group" aria-label="Install command">
         <span>pip install adctoolbox</span>
       </div>
     </div>
   </section>

Complete Documentation
----------------------

.. toctree::
   :maxdepth: 2
   :caption: Browse the Sphinx docs

   installation
   quickstart
   api/index
   algorithms/index
   examples/index
   python_matlab_parity
   changelog

Common Starting Points
----------------------

.. grid:: 1 1 2 4
   :gutter: 2
   :class-container: adt-quick-grid

   .. grid-item-card:: Install
      :link: installation
      :link-type: doc

      Set up the package, upgrade an existing install, and verify your local
      Python environment.

   .. grid-item-card:: Quick Start
      :link: quickstart
      :link-type: doc

      Run the first spectrum analysis and generate analog and digital debug
      dashboards.

   .. grid-item-card:: Examples
      :link: examples/index
      :link-type: doc

      Use ready-to-run scripts for spectrum, signal generation, calibration,
      debug, conversions, and time interleaving.

   .. grid-item-card:: API Reference
      :link: api/index
      :link-type: doc

      Browse public functions by module with signatures and docstrings.

Analysis Areas
--------------

.. grid:: 1 1 2 3
   :gutter: 2
   :class-container: adt-feature-grid

   .. grid-item-card:: Spectrum
      :link: api/spectrum
      :link-type: doc

      FFT-based SNDR, SNR, SFDR, THD, ENOB, NSD, windowing, averaging, and
      polar spectrum analysis.

   .. grid-item-card:: SAR Modeling
      :link: api/index
      :link-type: doc

      Vectorized SAR conversion, digital reconstruction, ideal weights, and
      mismatch-aware behavioral modeling.

   .. grid-item-card:: Calibration
      :link: algorithms/calibrate_weight_sine
      :link-type: doc

      Foreground bit-weight calibration from sine-wave decisions, plus
      lightweight and example-driven workflows.

   .. grid-item-card:: Analog Debug
      :link: api/aout
      :link-type: doc

      Error decomposition, phase-plane views, residual statistics, static
      nonlinearity fitting, and INL from sine tests.

   .. grid-item-card:: Digital Debug
      :link: api/dout
      :link-type: doc

      Bit activity, overflow checks, ENOB sweeps, residual scatter plots, and
      weight radix analysis.

   .. grid-item-card:: Oversampling
      :link: api/oversampling
      :link-type: doc

      Delta-sigma and oversampling analysis utilities, including NTF
      visualization.

Outputs
-------

.. grid:: 1 1 2 3
   :gutter: 2
   :class-container: adt-output-grid

   .. grid-item::

      .. image:: quickstart_spectrum.png
         :alt: ADCToolbox spectrum analysis output
         :class: adt-output-image

      Spectrum metrics

   .. grid-item::

      .. image:: quickstart_aout_dashboard.png
         :alt: ADCToolbox analog output dashboard
         :class: adt-output-image

      Analog dashboard

   .. grid-item::

      .. image:: quickstart_dout_dashboard.png
         :alt: ADCToolbox digital output dashboard
         :class: adt-output-image

      Digital dashboard

Reference
---------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
