API Reference
=============

This page documents the full ``fp_qsim`` package under ``src/fp_qsim``.

Package Exports
---------------

.. automodule:: fp_qsim
   :members: MockSimulator, CustomSimulatorManual, CustomSimulatorManualOptimized, mocked_statevector
   :undoc-members:
   :show-inheritance:

Simulator Module
----------------

.. automodule:: fp_qsim.simulator
   :members: MockSimulator, CustomSimulatorManual
   :undoc-members:
   :show-inheritance:

Optimized Simulator Module
--------------------------

.. automodule:: fp_qsim.simulator_optimized
   :members:
   :undoc-members:
   :show-inheritance:

Optimized Simulator Function Map
--------------------------------

The optimized simulator combines explicit index traversal with backend-specific
execution paths. Use the map below to quickly find the right entry point.

.. csv-table::
   :header: "Function / Method", "Category", "Purpose"
   :widths: 32, 18, 50

   "``_apply_u1_loop_body``", "Core kernel logic", "Shared upper/lower index traversal for single-qubit updates."
   "``_apply_u1_python``", "Python backend", "Applies a single-qubit gate via loop traversal without JIT."
   "``_apply_u1_numba``", "Numba backend", "Applies a single-qubit gate through the compiled kernel path."
   "``_apply_cx_python_inplace``", "Python backend", "In-place CX swap kernel with explicit control/target traversal."
   "``_apply_cx_numba_inplace``", "Numba backend", "JIT-compiled in-place CX kernel for faster execution."
   "``CustomSimulatorManualOptimized.apply_u_single_qubit``", "Public API", "Backend-dispatched single-qubit update on state tensors."
   "``CustomSimulatorManualOptimized.apply_cx``", "Public API", "Backend-dispatched CX application on state tensors."
   "``CustomSimulatorManualOptimized.run``", "Public API", "Main simulation loop for ``u``/``cx`` and unitary fallback gates."
   "``CustomSimulatorManualOptimized.run_batch``", "Public API", "Batch circuit execution wrapper."

Kernel-Level Reference
~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: fp_qsim.simulator_optimized

.. autofunction:: _apply_u1_loop_body

.. autofunction:: _apply_u1_python

.. autofunction:: _apply_u1_numba

.. autofunction:: _apply_cx_python_inplace

.. autofunction:: _apply_cx_numba_inplace

Runtime Entry Points
~~~~~~~~~~~~~~~~~~~~

See :meth:`fp_qsim.simulator_optimized.CustomSimulatorManualOptimized.apply_unitary`,
:meth:`fp_qsim.simulator_optimized.CustomSimulatorManualOptimized.apply_u_single_qubit`,
:meth:`fp_qsim.simulator_optimized.CustomSimulatorManualOptimized.apply_cx`,
:meth:`fp_qsim.simulator_optimized.CustomSimulatorManualOptimized.run`, and
:meth:`fp_qsim.simulator_optimized.CustomSimulatorManualOptimized.run_batch` for
the main runtime flow.

Statevector Utilities
---------------------

.. automodule:: fp_qsim.state_vector
   :members:
   :undoc-members:
   :show-inheritance:

Pauli Utilities
---------------

.. automodule:: fp_qsim.pauli
   :members:
   :undoc-members:
   :show-inheritance:
