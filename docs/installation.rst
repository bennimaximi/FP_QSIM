Installation
============

This page explains how to install `fp-qsim` for development and how to
verify that the installation works.

Prerequisites
-------------

Before installing `fp-qsim`, make sure you have:

- Python 3.11 or newer
- `pip` available
- Git, if you want to clone the repository
- A virtual environment tool such as `venv`

Create a virtual environment
----------------------------

It is recommended to install the project inside a virtual environment.

.. code-block:: bash

   python3 -m venv .venv
   source .venv/bin/activate

On Windows, use:

.. code-block:: bash

   .venv\Scripts\activate

Install the project
-------------------

Clone the repository and install the project in editable mode:

.. code-block:: bash

   git clone <repository-url>
   cd fp-qsim
   pip install -e .

This installs the package together with its main dependencies.

Install development dependencies
--------------------------------

If you also want the documentation and development tooling, install the
development extras as well.

.. code-block:: bash

   pip install -e .[dev]

This is useful if you want to:

- run the test suite
- build the Sphinx documentation
- use benchmarking tools during development

Main dependencies
-----------------

The project depends on several scientific and development packages, including:

- `numpy`
- `sympy`
- `matplotlib`
- `qiskit`
- `qiskit-aer`
- `pytest`
- `mypy`
- `ruff`

Verify the installation
-----------------------

You can verify that the installation succeeded by importing the main
dependencies in Python:

.. code-block:: bash

   python -c "import qiskit, qiskit_aer, numpy, sympy; print('Installation successful')"

You can also check that the project itself installs correctly with:

.. code-block:: bash

   pip show fp-qsim

Development tools
-----------------

This project includes configuration for static analysis and formatting tools.

To run the type checker:

.. code-block:: bash

   mypy .

To run the linter:

.. code-block:: bash

   ruff check .

To run the formatter:

.. code-block:: bash

   ruff format .

To run the tests:

.. code-block:: bash

   pytest

Build the documentation
-----------------------

If you installed the development dependencies, you can build the Sphinx
documentation with:

.. code-block:: bash

   sphinx-build -b html docs docs/_build/html

Troubleshooting
---------------

If installation fails, check the following:

- You are using Python 3.11 or newer.
- Your virtual environment is activated.
- You are running the commands from the project root directory.
- `pip` is up to date:

  .. code-block:: bash

     python -m pip install --upgrade pip

If `qiskit-aer` fails to install, first make sure your Python version is
supported and that your environment is clean.

Notes
-----

The package uses `hatchling` as its build backend, so installation is driven
by the metadata in `pyproject.toml`.
