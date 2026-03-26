(function () {
  "use strict";

  function launchConfetti(x, y) {
    var canvas = document.createElement("canvas");
    canvas.style.position = "fixed";
    canvas.style.inset = "0";
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    canvas.style.pointerEvents = "none";
    canvas.style.zIndex = "9999";
    document.body.appendChild(canvas);

    var ctx = canvas.getContext("2d");
    if (!ctx) {
      document.body.removeChild(canvas);
      return;
    }

    var width = window.innerWidth;
    var height = window.innerHeight;
    canvas.width = width;
    canvas.height = height;

    var colors = ["#0e8a16", "#f9b234", "#0f5cc0", "#e63946", "#00a8a8"];
    var particles = [];
    var count = 100;

    for (var i = 0; i < count; i += 1) {
      particles.push({
        x: x,
        y: y,
        vx: (Math.random() - 0.5) * 10,
        vy: -Math.random() * 8 - 2,
        g: 0.15 + Math.random() * 0.08,
        size: 4 + Math.random() * 6,
        rot: Math.random() * Math.PI,
        vr: (Math.random() - 0.5) * 0.2,
        color: colors[Math.floor(Math.random() * colors.length)],
        life: 70 + Math.floor(Math.random() * 20)
      });
    }

    var frame = 0;

    function tick() {
      frame += 1;
      ctx.clearRect(0, 0, width, height);

      for (var i = particles.length - 1; i >= 0; i -= 1) {
        var p = particles[i];
        p.vy += p.g;
        p.x += p.vx;
        p.y += p.vy;
        p.rot += p.vr;
        p.life -= 1;

        if (p.life <= 0 || p.y > height + 20) {
          particles.splice(i, 1);
          continue;
        }

        ctx.save();
        ctx.translate(p.x, p.y);
        ctx.rotate(p.rot);
        ctx.fillStyle = p.color;
        ctx.fillRect(-p.size / 2, -p.size / 2, p.size, p.size * 0.6);
        ctx.restore();
      }

      if (particles.length > 0 && frame < 180) {
        window.requestAnimationFrame(tick);
      } else if (canvas.parentNode) {
        canvas.parentNode.removeChild(canvas);
      }
    }

    window.requestAnimationFrame(tick);
  }

  function bindConfetti() {
    var apiLinks = document.querySelectorAll('a[href$="api.html"], a[href="api.html"]');
    if (!apiLinks.length) {
      return;
    }

    apiLinks.forEach(function (link) {
      link.addEventListener("click", function (event) {
        var rect = link.getBoundingClientRect();
        var x = rect.left + rect.width / 2;
        var y = rect.top + rect.height / 2;
        launchConfetti(x, y);

        // Let the animation be visible briefly before navigation continues.
        event.preventDefault();
        window.setTimeout(function () {
          window.location.href = link.getAttribute("href") || "api.html";
        }, 320);
      });
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bindConfetti);
  } else {
    bindConfetti();
  }
})();
