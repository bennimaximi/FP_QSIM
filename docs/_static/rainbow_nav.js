(function () {
  function ready(fn) {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", fn);
      return;
    }
    fn();
  }

  function shouldAnimateClick(event) {
    if (event.defaultPrevented) {
      return false;
    }
    if (event.button !== 0) {
      return false;
    }
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) {
      return false;
    }
    return true;
  }

  ready(function () {
    var root = document.documentElement;

    document.addEventListener("click", function (event) {
      var anchor = event.target.closest("a[href]");
      if (!anchor) {
        return;
      }

      var href = anchor.getAttribute("href");
      if (!href) {
        return;
      }

      if (href.indexOf("developer_guide") === -1) {
        return;
      }

      if (!shouldAnimateClick(event)) {
        return;
      }

      event.preventDefault();

      root.classList.remove("rainbow-nav-active");
      void root.offsetWidth;
      root.classList.add("rainbow-nav-active");

      window.setTimeout(function () {
        window.location.assign(anchor.href);
      }, 540);
    });
  });
})();
