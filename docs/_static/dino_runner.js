(function () {
  function ready(fn) {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", fn);
      return;
    }
    fn();
  }

  ready(function () {
    var canvas = document.getElementById("dino-canvas");
    var shell = document.getElementById("dino-game-shell");
    var scoreEl = document.getElementById("dino-score");
    var bestEl = document.getElementById("dino-best");
    var statusEl = document.getElementById("dino-status");
    var toggleBtn = document.getElementById("dino-toggle");
    var resetBtn = document.getElementById("dino-reset");

    if (!canvas || !shell || !scoreEl || !bestEl || !statusEl || !toggleBtn || !resetBtn) {
      return;
    }

    var ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }

    var W = canvas.width;
    var H = canvas.height;
    var groundY = H - 34;
    var gravity = 0.66;
    var jumpV = -11.5;
    var baseSpeed = 6.2;

    var running = false;
    var gameOver = false;
    var rafId = null;
    var lastTs = 0;
    var spawnCooldown = 0;

    var dino = {
      x: 74,
      y: groundY - 34,
      w: 30,
      h: 34,
      vy: 0,
      onGround: true,
      runCycle: 0
    };

    var obstacles = [];
    var score = 0;
    var best = Number(window.localStorage.getItem("fpqsim_dino_best") || "0");
    bestEl.textContent = "Best: " + best;

    function setStatus(msg) {
      statusEl.textContent = msg;
    }

    function resetGameState() {
      dino.y = groundY - dino.h;
      dino.vy = 0;
      dino.onGround = true;
      obstacles = [];
      score = 0;
      spawnCooldown = 30;
      gameOver = false;
      updateScore(0);
    }

    function updateScore(delta) {
      score += delta;
      scoreEl.textContent = "Score: " + Math.floor(score);
      if (score > best) {
        best = Math.floor(score);
        bestEl.textContent = "Best: " + best;
        window.localStorage.setItem("fpqsim_dino_best", String(best));
      }
    }

    function jump() {
      if (!running || gameOver) {
        return;
      }
      if (dino.onGround) {
        dino.vy = jumpV;
        dino.onGround = false;
      }
    }

    function spawnObstacle(speed) {
      var heights = [28, 34, 40, 46];
      var h = heights[Math.floor(Math.random() * heights.length)];
      var variant = Math.random() < 0.58 ? "single" : "double";
      var w = variant === "single" ? 16 + Math.random() * 8 : 24 + Math.random() * 10;
      obstacles.push({
        x: W + 20,
        y: groundY - h,
        w: w,
        h: h,
        variant: variant,
        speed: speed
      });
    }

    function intersects(a, b) {
      return a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
    }

    function drawBackground() {
      ctx.clearRect(0, 0, W, H);

      var grad = ctx.createLinearGradient(0, 0, 0, H);
      grad.addColorStop(0, "#f8fcff");
      grad.addColorStop(1, "#eef7ff");
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, W, H);

      ctx.strokeStyle = "#7ea1b8";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(0, groundY + 0.5);
      ctx.lineTo(W, groundY + 0.5);
      ctx.stroke();
    }

    function drawDino() {
      var bodyColor = "#213f55";
      var shadeColor = "#1a3447";

      var bobOffset = dino.onGround ? Math.sin(dino.runCycle * 0.18) * 0.6 : -1.2;
      var y = dino.y + bobOffset;

      // Tail
      ctx.fillStyle = bodyColor;
      ctx.fillRect(dino.x - 6, y + 18, 8, 6);

      // Body + neck + head
      ctx.fillRect(dino.x + 2, y + 10, 17, 18);
      ctx.fillRect(dino.x + 16, y + 6, 6, 10);
      ctx.fillRect(dino.x + 20, y + 4, 11, 12);

      // Mouth notch
      ctx.fillStyle = shadeColor;
      ctx.fillRect(dino.x + 26, y + 13, 5, 2);

      // Eye
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(dino.x + 24, y + 7, 3, 3);

      // Legs (animated on ground)
      ctx.fillStyle = bodyColor;
      if (!dino.onGround) {
        ctx.fillRect(dino.x + 8, y + 24, 5, 7);
        ctx.fillRect(dino.x + 14, y + 23, 5, 8);
        return;
      }

      var frame = Math.floor(dino.runCycle / 6) % 2;
      if (frame === 0) {
        ctx.fillRect(dino.x + 8, y + 24, 5, 8);
        ctx.fillRect(dino.x + 14, y + 25, 5, 7);
      } else {
        ctx.fillRect(dino.x + 8, y + 25, 5, 7);
        ctx.fillRect(dino.x + 14, y + 24, 5, 8);
      }
    }

    function drawCactus(ob) {
      var x = ob.x;
      var y = ob.y;
      var w = ob.w;
      var h = ob.h;

      var trunkW = Math.max(8, Math.floor(w * 0.36));
      var trunkX = x + Math.floor((w - trunkW) / 2);

      ctx.fillStyle = "#2d6e45";
      ctx.fillRect(trunkX, y, trunkW, h);

      // Arms for visual cactus silhouette.
      var armY = y + Math.floor(h * 0.38);
      if (ob.variant === "single" || ob.variant === "double") {
        ctx.fillRect(trunkX - 6, armY, 6, 4);
        ctx.fillRect(trunkX - 3, armY - 8, 3, 8);
      }
      if (ob.variant === "double") {
        ctx.fillRect(trunkX + trunkW, armY + 2, 6, 4);
        ctx.fillRect(trunkX + trunkW + 3, armY - 6, 3, 8);
      }

      // Spine highlights.
      ctx.fillStyle = "#7dc08f";
      for (var i = 3; i < h - 2; i += 8) {
        ctx.fillRect(trunkX + 1, y + i, 1, 2);
        ctx.fillRect(trunkX + trunkW - 2, y + i + 2, 1, 2);
      }
    }

    function drawObstacles() {
      for (var i = 0; i < obstacles.length; i += 1) {
        drawCactus(obstacles[i]);
      }
    }

    function tick(ts) {
      if (!running) {
        return;
      }

      if (!lastTs) {
        lastTs = ts;
      }

      var dt = (ts - lastTs) / 16.6667;
      lastTs = ts;

      var speed = baseSpeed + Math.min(score / 250, 4.2);

      dino.vy += gravity * dt;
      dino.y += dino.vy * dt;
      if (dino.y >= groundY - dino.h) {
        dino.y = groundY - dino.h;
        dino.vy = 0;
        dino.onGround = true;
      }

      if (dino.onGround) {
        dino.runCycle += dt;
      }

      spawnCooldown -= dt;
      if (spawnCooldown <= 0) {
        spawnObstacle(speed);
        spawnCooldown = 30 + Math.random() * 42;
      }

      for (var i = obstacles.length - 1; i >= 0; i -= 1) {
        obstacles[i].x -= obstacles[i].speed * dt;
        if (obstacles[i].x + obstacles[i].w < 0) {
          obstacles.splice(i, 1);
        }
      }

      for (var j = 0; j < obstacles.length; j += 1) {
        if (intersects(dino, obstacles[j])) {
          gameOver = true;
          running = false;
          toggleBtn.textContent = "Start";
          setStatus("Game over. Press Start to run again.");
          break;
        }
      }

      updateScore(0.45 * dt);

      drawBackground();
      drawDino();
      drawObstacles();

      if (!gameOver) {
        rafId = window.requestAnimationFrame(tick);
      }
    }

    function startGame() {
      if (running) {
        return;
      }
      if (gameOver) {
        resetGameState();
      }
      running = true;
      lastTs = 0;
      toggleBtn.textContent = "Pause";
      setStatus("Running. Jump with Space / Arrow Up / click.");
      rafId = window.requestAnimationFrame(tick);
    }

    function pauseGame() {
      running = false;
      toggleBtn.textContent = "Start";
      setStatus("Paused.");
      if (rafId !== null) {
        window.cancelAnimationFrame(rafId);
        rafId = null;
      }
    }

    function hardReset() {
      pauseGame();
      gameOver = false;
      resetGameState();
      drawBackground();
      drawDino();
      setStatus("Ready. Press Start.");
    }

    toggleBtn.addEventListener("click", function () {
      if (running) {
        pauseGame();
      } else {
        startGame();
      }
    });

    resetBtn.addEventListener("click", function () {
      hardReset();
    });

    canvas.addEventListener("pointerdown", function () {
      if (!running) {
        startGame();
      } else {
        jump();
      }
    });

    window.addEventListener("keydown", function (ev) {
      if (!shell.contains(document.activeElement) && document.activeElement !== document.body) {
        return;
      }
      if (ev.code === "Space" || ev.code === "ArrowUp") {
        ev.preventDefault();
        if (!running) {
          startGame();
        } else {
          jump();
        }
      }
    });

    resetGameState();
    drawBackground();
    drawDino();
  });
})();
