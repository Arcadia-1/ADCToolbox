Installation
============

Quick Start
-----------

Install ADCToolbox from PyPI:

.. code-block:: bash

    pip install adctoolbox

**Get All 45 Examples** (Recommended for learning)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    cd /path/to/your/workspace
    adctoolbox-get-examples

This creates an ``adctoolbox_examples/`` directory with all examples organized into 6 categories:

* ``01_basic/`` - Fundamentals (2 examples)
* ``02_spectrum/`` - FFT-Based Analysis (14 examples)
* ``03_generate_signals/`` - Non-Ideality Modeling (6 examples)
* ``04_debug_analog/`` - Error Characterization (13 examples)
* ``05_debug_digital/`` - Calibration & Redundancy (5 examples)
* ``07_conversions/`` - Conversions (5 examples)

Run your first examples:

.. code-block:: bash

    cd adctoolbox_examples/01_basic
    python exp_b01_environment_check.py
    python exp_b02_coherent_vs_non_coherent.py

    cd ../02_spectrum
    python exp_s01_analyze_spectrum_simplest.py

Requirements
------------

* Python >= 3.8
* NumPy >= 1.20.0
* Matplotlib >= 3.3.0
* SciPy >= 1.6.0

Upgrade & Verification
----------------------

To upgrade an existing installation:

.. code-block:: bash

    pip install --upgrade adctoolbox

To verify the installation and check the version:

.. code-block:: bash

    python -c "import adctoolbox; print(adctoolbox.__version__)"

Development Installation
------------------------

For development or to access the latest code:

.. code-block:: bash

    git clone https://github.com/Arcadia-1/ADCToolbox.git
    cd ADCToolbox/python
    pip install -e ".[docs]"
