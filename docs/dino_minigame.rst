Runner Mini Game
========================

This is a runner-style mini game.

.. raw:: html

    <section class="dino-game-shell" id="dino-game-shell">
      <p class="dino-game-kicker">Mini Game</p>
      <h2 class="dino-game-title">Dino Runner</h2>
      <p class="dino-game-help">Press <strong>Space</strong> / <strong>Arrow Up</strong> / tap the canvas to jump. Avoid obstacles.</p>

      <div class="dino-game-bar">
        <span id="dino-score">Score: 0</span>
        <span id="dino-best">Best: 0</span>
        <button id="dino-toggle" type="button">Start</button>
        <button id="dino-reset" type="button">Reset</button>
      </div>

      <canvas id="dino-canvas" width="760" height="220" aria-label="Offline runner mini game"></canvas>
      <p id="dino-status" class="dino-game-status">Ready. Press Start.</p>
    </section>

.. raw:: html

    <script src="_static/dino_runner.js"></script>
