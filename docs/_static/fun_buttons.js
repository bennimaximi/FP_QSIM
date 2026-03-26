(function () {
  function ready(fn) {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", fn);
      return;
    }
    fn();
  }

  ready(function () {
    var root = document.documentElement;
    var zone = document.getElementById("fun-zone");
    if (!zone) {
      return;
    }

    var output = document.getElementById("doc-fun-output");
    var cardGrid = document.querySelector(".doc-card-grid");
    var buttons = zone.querySelectorAll("[data-fun-action]");

    var facts = [
      "Superposition is not randomness; it is a weighted complex state.",
      "A global phase never changes measurement probabilities.",
      "Entanglement means one tensor product no longer factors cleanly.",
      "A statevector for n qubits has 2^n amplitudes.",
      "Transpilation changes gate basis, not the physics of the circuit."
    ];

    var themes = [
      { accent: "#007a87", accent2: "#ef6c2f", soft: "#eaf8fb" },
      { accent: "#1b7f4a", accent2: "#d26e00", soft: "#edf9f1" },
      { accent: "#8b2f2f", accent2: "#cc8a00", soft: "#fff3ed" },
      { accent: "#255e94", accent2: "#aa4f00", soft: "#ebf4ff" }
    ];

    var currentTheme = 0;

    function setOutput(text) {
      if (output) {
        output.textContent = text;
      }
    }

    function themeJump() {
      currentTheme = (currentTheme + 1) % themes.length;
      var t = themes[currentTheme];
      root.style.setProperty("--card-accent", t.accent);
      root.style.setProperty("--card-accent-2", t.accent2);
      root.style.setProperty("--card-accent-soft", t.soft);
      setOutput("Theme jumped. The docs just changed mood.");
    }

    function shuffleCards() {
      if (!cardGrid) {
        setOutput("No cards found to shuffle, but we tried.");
        return;
      }
      cardGrid.classList.remove("fun-shuffle");
      void cardGrid.offsetWidth;
      cardGrid.classList.add("fun-shuffle");
      setOutput("Cards did a tiny shuffle dance.");
    }

    function launchStars() {
      var colors = ["#007a87", "#ef6c2f", "#f2c14e", "#3da35d", "#4a6fa5"];
      for (var i = 0; i < 22; i += 1) {
        var star = document.createElement("span");
        star.className = "doc-fun-star";
        star.style.backgroundColor = colors[i % colors.length];
        star.style.setProperty("--dx", String(Math.round((Math.random() - 0.5) * 260)) + "px");
        star.style.setProperty("--dy", String(Math.round((Math.random() - 0.5) * 140)) + "px");
        zone.appendChild(star);
        (function (node) {
          window.setTimeout(function () {
            if (node.parentNode) {
              node.parentNode.removeChild(node);
            }
          }, 700);
        })(star);
      }
      setOutput("Star burst launched. Absolutely necessary for science.");
    }

    function showFact() {
      var fact = facts[Math.floor(Math.random() * facts.length)];
      setOutput("Quantum fact: " + fact);
    }

    buttons.forEach(function (btn) {
      btn.addEventListener("click", function () {
        var action = btn.getAttribute("data-fun-action");
        if (action === "theme-jump") {
          themeJump();
          return;
        }
        if (action === "shuffle-cards") {
          shuffleCards();
          return;
        }
        if (action === "launch-stars") {
          launchStars();
          return;
        }
        if (action === "show-fact") {
          showFact();
        }
      });
    });
  });
})();
