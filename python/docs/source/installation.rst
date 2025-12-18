Installation
============

Requirements
------------

* Python 3.8 or higher
* NumPy >= 1.20.0
* Matplotlib >= 3.3.0
* SciPy >= 1.6.0

Install from PyPI
-----------------

To install the latest stable release from PyPI:

.. code-block:: bash

    pip install adctoolbox

Upgrade
-------

To upgrade an existing installation:

.. code-block:: bash

    pip install --upgrade adctoolbox

Development Installation
------------------------

To install from source for development:

.. code-block:: bash

    git clone https://github.com/your-org/ADCToolbox.git
    cd ADCToolbox/python
    pip install -e ".[docs]"

Verify Installation
-------------------

To verify the installation and check the version:

.. code-block:: bash

    python -c "import adctoolbox; print(adctoolbox.__version__)"

Get Examples
------------

ADCToolbox includes 21 ready-to-run examples. To copy them to your workspace:

.. code-block:: bash

    # Navigate to your workspace directory
    cd /path/to/your/workspace

    # Copy examples
    adctoolbox-get-examples

This creates an ``adctoolbox_examples/`` directory with all examples organized by category.
