Quick Start
===========

This page gets you from a fresh install to a first FP-QSIM statevector run in a
few minutes.

What you will do
----------------

1. Install the package in a virtual environment.
2. Build and transpile a small circuit.
3. Run one simulator and print the resulting statevector.

Step 1: Install FP-QSIM
-----------------------

From the project root:

.. code-block:: bash

   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e .

On Windows activation:

.. code-block:: bash

   .venv\Scripts\activate

Step 2: Run your first simulation
---------------------------------

Create a file named ``quickstart_demo.py`` with:

.. code-block:: python

   import numpy as np
   from qiskit import QuantumCircuit, transpile

   from fp_qsim import CustomSimulatorManualOptimized

   qc = QuantumCircuit(2)
   qc.h(0)
   qc.cx(0, 1)

   # Keep gate set explicit for manual simulator paths.
   tqc = transpile(qc, basis_gates=["u", "cx"], optimization_level=0)

   sim = CustomSimulatorManualOptimized(cx_backend="numba")
   state = sim.run(tqc)

   probs = np.abs(state) ** 2
   print("statevector:", state)
   print("probabilities:", probs)

Run it:

.. code-block:: bash

   python quickstart_demo.py

Expected result
---------------

- The script prints a complex statevector of length ``4``.
- The probabilities are close to ``[0.5, 0.0, 0.0, 0.5]`` for the Bell-state circuit.

Optional: Use CUDA if available
-------------------------------

If your system has CUDA and you want GPU execution, replace simulator creation with:

.. code-block:: python

   from fp_qsim import CustomSimulatorManualGPU
   sim = CustomSimulatorManualGPU()

If CUDA is not available, continue with ``CustomSimulatorManualOptimized``.

Next steps
----------

- Continue with :doc:`how_to` for task-focused workflows.
- Open :doc:`notebooks` for interactive examples.
- Use :doc:`api` for complete class and function references.
