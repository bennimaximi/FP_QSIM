Explanation: Why FP-QSIM Exists
================================

This page provides background and context for FP-QSIM. It is not a step-by-step
how-to and not a full API listing. Instead, it answers why this project is
structured the way it is and how the pieces fit together.

What Problem FP-QSIM Solves
---------------------------

Quantum simulation workflows often involve two conflicting needs:

- clear, inspectable logic for learning and debugging
- fast execution for larger statevector experiments

FP-QSIM keeps both needs visible. It offers an implementation that is easy to
reason about and an optimized path that emphasizes runtime performance. This
split makes trade-offs explicit rather than hiding them behind one opaque
implementation.

Why There Are Multiple Simulator Paths
--------------------------------------

You will see both a general tensor-contraction style and manual loop-based
kernels in the project.

The general approach is valuable because:

- it is expressive and close to mathematical notation
- it supports broad gate handling with less specialized code

The manual/optimized approach is valuable because:

- it makes index-level behavior explicit
- it enables targeted acceleration (for example, numba-backed kernels)
- it is easier to benchmark specific hot paths such as single-qubit updates and
  CX-heavy circuits

In short, one path prioritizes generality and clarity of expression, while the
other prioritizes control over performance-critical details.

Why Benchmarks Are First-Class Documentation
--------------------------------------------

Benchmarks are included as documentation artifacts, not only as engineering
checks. This is intentional.

Performance claims are most useful when users can inspect:

- what was measured
- for which qubit ranges
- under which simulation style

By surfacing benchmark plots and test-backed measurements, FP-QSIM makes
performance discussions concrete and reproducible.

Why Notebooks Are Included
--------------------------

Notebook examples are used to bridge ideas and implementation.

They help users:

- validate correctness against a trusted reference
- inspect statevector behavior interactively
- experiment with variants before changing library code

This supports study-oriented use: understanding behavior before optimizing or
extending it.

How To Use This Context
-----------------------

A practical reading order is:

1. Installation for environment setup.
2. Notebook examples for intuition and experimentation.
3. API reference for exact function and class contracts.
4. Benchmark results for performance interpretation.

If you are learning, start with notebooks and this explanation page. If you are
integrating into another project, start with installation and API reference,
then use benchmark data to guide backend choices.

Design Perspective
------------------

FP-QSIM follows a documentation approach where different pages serve different
questions:

- tutorials/notebooks: How do I do this?
- reference/API: What exactly is available?
- explanation (this page): Why is it designed this way?

Keeping these concerns separate helps avoid overloaded tutorials and keeps
reference concise while still giving deeper conceptual context when needed.
