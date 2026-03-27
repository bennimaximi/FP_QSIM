How-To Guides
=============

This section is for users who already know the basics of ``fp-qsim`` and need
practical directions to complete specific tasks.

How to choose the right simulator for your workload
---------------------------------------------------

Use this checklist when deciding which simulator to run for a circuit batch.

1. Use ``MockSimulator`` when you need a Qiskit Aer-backed reference result.
2. Use ``CustomSimulatorManual`` when your circuit is mostly ``u`` and ``cx``
   and you want a pure-project baseline.
3. Use ``CustomSimulatorManualOptimized(cx_backend="numba")`` for CX-heavy
   workloads where throughput matters.

Quick decision flow:

- If your goal is validation against a known backend: start with
  ``MockSimulator``.
- If your goal is runtime on medium and larger CX-dominant circuits: use
  ``CustomSimulatorManualOptimized``.

How to verify two simulators produce equivalent statevectors
------------------------------------------------------------

Use this procedure when changing kernels or adding optimizations.

1. Build or load a circuit and transpile to ``["u", "cx"]`` when you compare
   manual paths.
2. Run both simulators on the same circuit.
3. Align global phase before asserting numerical equality.
4. Assert the maximum absolute difference is below your tolerance.

Example script:

.. code-block:: python

   import numpy as np
   from qiskit import QuantumCircuit, transpile
   from fp_qsim import (
       MockSimulator,
       CustomSimulatorManualOptimized,
   )

   qc = QuantumCircuit(5)
   qc.h(0)
   qc.cx(0, 1)
   qc.cx(1, 2)
   qc.rx(0.2, 3)
   qc.save_statevector()

   tqc = transpile(qc, basis_gates=["u", "cx"], optimization_level=0)

   ref = MockSimulator().run(tqc)
   test = CustomSimulatorManualOptimized(cx_backend="numba").run(tqc)

   # Remove global phase offset before comparison.
   idx = np.argmax(np.abs(ref))
   phase = np.angle(ref[idx]) - np.angle(test[idx])
   aligned = test * np.exp(1j * phase)

   max_err = float(np.max(np.abs(ref - aligned)))
   assert max_err < 1e-9, f"statevector mismatch: {max_err}"

How to benchmark optimized CX execution
---------------------------------------

Use this when you need a reproducible performance snapshot for docs or PRs.

1. Run benchmark-marked tests and export JSON:

   .. code-block:: bash

      uv run pytest -q -k benchmark --benchmark-group-by=param:n_qubits --benchmark-sort=name --benchmark-json docs/_static/benchmark_results_q5_16.json

2. Regenerate benchmark images:

   .. code-block:: bash

      uv run python docs/plot_benchmark.py

3. Rebuild docs and inspect benchmark pages:

   .. code-block:: bash

      uv run sphinx-build -b html docs docs/_build/html

4. Verify that:
   - runtime medians update as expected
   - custom/aer ratio trend is still reasonable
   - optimized CX speedup trend does not regress for larger qubit counts

How to troubleshoot documentation build failures
------------------------------------------------

Use this workflow when ``make html`` or ``sphinx-build`` fails.

1. Build from the project root first:

   .. code-block:: bash

      uv run sphinx-build -b html docs docs/_build/html

2. If that succeeds but ``make.bat html`` fails on Windows, the shell likely
   cannot resolve ``sphinx-build`` from ``PATH``. Keep using the explicit
   command above.

3. If autodoc import errors appear, verify that package imports work:

   .. code-block:: bash

      uv run python -c "import fp_qsim; print('import ok')"

4. If notebook pages fail, check that referenced notebook paths in
   ``docs/notebooks.rst`` still exist and match filenames.

5. Clean and rebuild if stale files are suspected:

   .. code-block:: bash

      Remove-Item -Recurse -Force docs/_build
      uv run sphinx-build -b html docs docs/_build/html
