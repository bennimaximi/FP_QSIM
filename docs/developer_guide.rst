Developer Guide
===============

This guide is the maintainer-facing entry point for developing and extending
``fp-qsim``.

What This Project Is
--------------------

FP-QSIM is a statevector-focused quantum simulation project with two goals:

1. Provide clear simulator implementations for correctness and inspection.
2. Benchmark custom kernels against reference workflows (especially CX-heavy paths).

The codebase combines:

- baseline simulators in ``src/fp_qsim/simulator.py``
- optimized kernels and backend switching in ``src/fp_qsim/simulator_optimized.py``
- validation and benchmark tests under ``tests/``
- Sphinx docs and executable notebooks under ``docs/``

How the Codebase Is Organized
-----------------------------

Top-level layout:

- ``src/fp_qsim/``: package source.
- ``tests/``: correctness and benchmark tests.
- ``docs/``: Sphinx documentation and notebooks.
- ``playground/``: local experimentation notebooks/scripts.

Main source modules:

- ``src/fp_qsim/__init__.py``
  Re-exports public API symbols:
   ``MockSimulator``, ``CustomSimulatorManual``,
   ``CustomSimulatorManualOptimized``, and
  ``mocked_statevector``.

- ``src/fp_qsim/simulator.py``
  Contains:

  1. ``MockSimulator``: wrapper around Aer statevector simulation.
   2. ``CustomSimulatorManual``: manual special handling for ``u`` and ``cx``.

- ``src/fp_qsim/simulator_optimized.py``
  Contains ``CustomSimulatorManualOptimized`` plus low-level kernels:

  - ``_apply_cx_python_inplace`` / ``_apply_cx_numba_inplace``
  - ``_apply_u1_python`` / ``_apply_u1_numba``

  The backend switch is controlled via
  ``cx_backend: Literal["python", "numba"]``.

- ``src/fp_qsim/state_vector.py``
  Provides ``mocked_statevector`` as a reference statevector helper.

- ``src/fp_qsim/pauli.py``
  Utility Pauli matrix constructors.

Test modules:

- ``tests/test_simulator.py``: baseline correctness and simulator-runtime benchmark group.
- ``tests/test_simulator_optimized.py``: optimized backend correctness,
  backend equivalence, run-batch checks, and CX-heavy benchmark group.

How to Set It Up
----------------

Requirements:

- Python 3.11+
- ``uv`` (recommended) or ``pip``

Recommended setup (uv-first):

.. code-block:: bash

   uv sync --group dev
   source .venv/bin/activate

If you prefer pip:

.. code-block:: bash

   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e .

Install development extras with pip:

.. code-block:: bash

   pip install -e .[dev]

Quick environment sanity check:

.. code-block:: bash

   uv run python -c "import numpy, qiskit, qiskit_aer, numba; print('ok')"

How to Run and Test It
----------------------

Run all tests:

.. code-block:: bash

   uv run pytest tests/

Run only optimized simulator tests:

.. code-block:: bash

   uv run pytest tests/test_simulator_optimized.py

Run benchmark-marked tests:

.. code-block:: bash

   uv run pytest -q -k benchmark --benchmark-group-by=param:n_qubits --benchmark-columns=median --benchmark-sort=name

Export benchmark JSON used by docs plots:

.. code-block:: bash

   uv run pytest -q -k benchmark --benchmark-group-by=param:n_qubits --benchmark-sort=name --benchmark-json docs/_static/benchmark_results_q5_16.json

Regenerate benchmark figures:

.. code-block:: bash

   uv run python docs/plot_benchmark.py

How to Document and Contribute
------------------------------

Build docs:

.. code-block:: bash

   cd docs
   make html

Alternative build command from project root:

.. code-block:: bash

   uv run sphinx-build -b html docs docs/_build/html

Documentation inputs:

- API reference pages are driven by autodoc entries in ``docs/api.rst``.
- Notebook pages are listed in ``docs/notebooks.rst``.
- Sphinx configuration is in ``docs/conf.py``.

Contribution checklist:

1. Keep new code typed (``mypy`` is configured with ``disallow_untyped_defs = true``).
2. Add or update tests for behavior changes.
3. Run lint and type checks:

   .. code-block:: bash

      uv run ruff check .
      uv run ruff format .
      uv run mypy .

4. Run test suite before opening a PR:

   .. code-block:: bash

      uv run pytest tests/

5. Update docs (API/docs page/notebook) whenever user-facing behavior changes.

If pre-commit is used locally:

.. code-block:: bash

   uv run pre-commit install
   uv run pre-commit run --all-files

How to Extend It
----------------

Typical extension patterns:

1. Add a new simulator variant.
   - Implement the class in ``src/fp_qsim/simulator.py`` or
     ``src/fp_qsim/simulator_optimized.py``.
   - Keep ``run(circuit, shots=...) -> np.ndarray`` semantics consistent.
   - Export it via ``src/fp_qsim/__init__.py`` when it is part of public API.

2. Add or optimize gate kernels.
   - For optimized paths, add low-level helper(s) near existing kernel functions in
     ``src/fp_qsim/simulator_optimized.py``.
   - Wire backend selection through ``CustomSimulatorManualOptimized``.
   - Keep fallback behavior for unsupported gates via operator/einsum path.

3. Extend batch execution behavior.
   - ``run_batch`` currently executes serially and ignores ``max_workers``.
   - If parallelism is introduced, preserve deterministic outputs and update tests.

4. Add validation and performance evidence.
   - Add correctness tests in ``tests/test_simulator.py`` and/or
     ``tests/test_simulator_optimized.py``.
   - Add benchmark coverage for new hot paths where performance claims are made.
   - Update docs pages (benchmark/API/notebooks) so changes are discoverable.

Known Pitfalls
--------------

- ``shots`` is currently a placeholder for statevector simulators.
  ``run`` methods return deterministic statevectors, not sampled counts.

- Global phase differences can cause direct vector comparisons to fail.
  Use phase alignment (see test helpers) before asserting equality for complex circuits.

- Performance claims should be based on transpiled circuits in ``["u", "cx"]`` basis.
  Other gates may fall back to generic operator/einsum handling.

- Numba has warm-up/compile overhead.
  Small circuits can appear slower before larger-qubit speedups dominate.

- ``run_batch(..., max_workers=...)`` in ``CustomSimulatorManualOptimized`` is serial today.
  The parameter is currently a compatibility placeholder.

- Sphinx API import depends on path injection in ``docs/conf.py``.
  Breaking that ``sys.path`` setup can cause autodoc import failures.

- Randomized tests/benchmarks must use explicit seeds to avoid flaky behavior.
