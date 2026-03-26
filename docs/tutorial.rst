Tutorial: Build a Quantum Coin Game
===================================

This tutorial is a guided lesson. You will build a tiny, playable
"quantum coin" game and verify each step as you go.

By the end, you will be able to:

1. Build and transpile a simple 1-qubit circuit.
2. Run the circuit with ``CustomSimulatorManual``.
3. Turn statevector amplitudes into measurement probabilities.
4. Sample outcomes to create a game loop.

What you will build
-------------------

You will create a game with one rule:

- The game prepares a qubit, rotates it, and "flips" a weighted quantum coin.
- You guess ``0`` or ``1`` before each flip.
- You score one point for each correct guess.

Before you start
----------------

Use a Python environment where ``fp_qsim`` is installed.

If you need setup instructions, start with :doc:`installation`.

Step 1: Create and inspect the circuit
--------------------------------------

Create a new Python file (for example ``quantum_coin_game.py``) and paste:

.. code-block:: python

   import numpy as np
   from qiskit import QuantumCircuit, transpile

   from fp_qsim import CustomSimulatorManual

   qc = QuantumCircuit(1)
   qc.h(0)
   qc.rz(0.4, 0)

   # Keep the manual simulator on supported gates.
   tqc = transpile(qc, basis_gates=["u", "cx"], optimization_level=0)
   print(tqc)

Run it.

Checkpoint:

1. The printed circuit should contain only ``u`` (and no unsupported operations).
2. No exception should be raised.

Step 2: Simulate and compute probabilities
------------------------------------------

Extend the same file with:

.. code-block:: python

   sim = CustomSimulatorManual()
   state = sim.run(tqc)

   # For one qubit: |state[0]|^2 is P(0), |state[1]|^2 is P(1).
   probabilities = np.abs(state) ** 2
   probabilities = probabilities / probabilities.sum()

   print("statevector:", state)
   print("P(0), P(1):", probabilities)

Run again.

Checkpoint:

1. You should see a complex-valued statevector of length 2.
2. The two probabilities should be non-negative and sum to 1 (within floating-point tolerance).

Step 3: Add one playable round
------------------------------

Now add one round where the player guesses the outcome:

.. code-block:: python

   rng = np.random.default_rng(seed=7)

   guess_text = input("Guess the outcome (0 or 1): ").strip()
   if guess_text not in {"0", "1"}:
       raise ValueError("Please enter 0 or 1.")

   guess = int(guess_text)
   outcome = int(rng.choice([0, 1], p=probabilities))

   print(f"Measured outcome: {outcome}")
   if guess == outcome:
       print("Correct! +1 point")
   else:
       print("Not this time.")

Run again.

Checkpoint:

1. Entering ``0`` or ``1`` should complete without crashing.
2. You should always see one measured outcome and one result message.

Step 4: Turn it into a 5-round game
-----------------------------------

Replace the Step 3 block with this loop:

.. code-block:: python

   rng = np.random.default_rng(seed=7)
   score = 0
   rounds = 5

   for round_idx in range(1, rounds + 1):
       guess_text = input(f"Round {round_idx}/{rounds} - guess 0 or 1: ").strip()
       if guess_text not in {"0", "1"}:
           print("Invalid input. This round counts as a miss.")
           continue

       guess = int(guess_text)
       outcome = int(rng.choice([0, 1], p=probabilities))
       print(f"Outcome: {outcome}")

       if guess == outcome:
           score += 1
           print("Nice! +1")
       else:
           print("Miss")

   print(f"Final score: {score}/{rounds}")

Run it and play all rounds.

Checkpoint:

1. The game always ends with ``Final score: .../5``.
2. Your score changes based on your guesses.

Step 5: Make the coin more biased
---------------------------------

Change the circuit in Step 1 from:

.. code-block:: python

   qc.h(0)
   qc.rz(0.4, 0)

to:

.. code-block:: python

   qc.ry(1.1, 0)

Run the game again and compare behavior.

Checkpoint:

1. ``P(0), P(1)`` should change noticeably.
2. The outcomes should reflect the new bias over multiple rounds.

Troubleshooting
---------------

- If you see import errors for ``fp_qsim``, reinstall in editable mode from project root:

  .. code-block:: bash

     pip install -e .

- If the transpiled circuit includes unexpected gates, force basis gates exactly as shown:

  .. code-block:: python

     tqc = transpile(qc, basis_gates=["u", "cx"], optimization_level=0)

- If probabilities do not sum to 1 exactly, that is normal for floating-point arithmetic.

What you learned
----------------

You used ``fp_qsim`` to build a complete practical loop:

1. define a circuit,
2. transpile to simulator-friendly gates,
3. simulate to a statevector,
4. convert amplitudes to probabilities,
5. and use outcomes in an interactive program.

Next, continue with :doc:`how_to` for task-focused workflows and :doc:`notebooks` for interactive examples.
