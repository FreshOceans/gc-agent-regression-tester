/* Local dashboard capture helper (no external CDN dependency). */
(function () {
  function cloneWithInlineStyles(source) {
    var clone = source.cloneNode(true);
    var sourceNodes = [source].concat(Array.prototype.slice.call(source.querySelectorAll("*")));
    var cloneNodes = [clone].concat(Array.prototype.slice.call(clone.querySelectorAll("*")));

    for (var i = 0; i < sourceNodes.length && i < cloneNodes.length; i += 1) {
      var srcNode = sourceNodes[i];
      var dstNode = cloneNodes[i];
      if (!(dstNode instanceof Element) || !(srcNode instanceof Element)) {
        continue;
      }
      var computed = window.getComputedStyle(srcNode);
      for (var j = 0; j < computed.length; j += 1) {
        var prop = computed[j];
        dstNode.style.setProperty(
          prop,
          computed.getPropertyValue(prop),
          computed.getPropertyPriority(prop)
        );
      }
    }
    return clone;
  }

  function elementToSvgDataUrl(element) {
    var rect = element.getBoundingClientRect();
    var width = Math.ceil(rect.width);
    var height = Math.ceil(rect.height);
    if (width < 2 || height < 2) {
      throw new Error("Dashboard capture area is too small.");
    }

    var clone = cloneWithInlineStyles(element);
    clone.setAttribute("xmlns", "http://www.w3.org/1999/xhtml");
    clone.style.margin = "0";

    var wrapper = document.createElement("div");
    wrapper.setAttribute("xmlns", "http://www.w3.org/1999/xhtml");
    wrapper.style.width = width + "px";
    wrapper.style.height = height + "px";
    wrapper.style.boxSizing = "border-box";
    wrapper.appendChild(clone);

    var serializer = new XMLSerializer();
    var xhtml = serializer.serializeToString(wrapper);
    var svg =
      '<svg xmlns="http://www.w3.org/2000/svg" width="' +
      width +
      '" height="' +
      height +
      '" viewBox="0 0 ' +
      width +
      " " +
      height +
      '">' +
      '<foreignObject width="100%" height="100%">' +
      xhtml +
      "</foreignObject>" +
      "</svg>";
    return "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svg);
  }

  function downloadBlob(blob, filename) {
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  function buildFilename() {
    var suiteNode = document.getElementById("suite-name");
    var suite = suiteNode ? suiteNode.textContent : "dashboard";
    var safeSuite = String(suite || "dashboard")
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 60) || "dashboard";
    var now = new Date();
    var stamp =
      now.getFullYear().toString() +
      String(now.getMonth() + 1).padStart(2, "0") +
      String(now.getDate()).padStart(2, "0") +
      "-" +
      String(now.getHours()).padStart(2, "0") +
      String(now.getMinutes()).padStart(2, "0") +
      String(now.getSeconds()).padStart(2, "0");
    return safeSuite + "-dashboard-" + stamp + ".png";
  }

  async function captureDashboardPng(targetId) {
    var target = document.getElementById(targetId);
    if (!target) {
      throw new Error("Dashboard export target was not found.");
    }
    var svgDataUrl = elementToSvgDataUrl(target);
    var img = await new Promise(function (resolve, reject) {
      var image = new Image();
      image.onload = function () {
        resolve(image);
      };
      image.onerror = function () {
        reject(new Error("Could not render dashboard image."));
      };
      image.src = svgDataUrl;
    });

    var scale = Math.max(1, window.devicePixelRatio || 1);
    var canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(img.width * scale));
    canvas.height = Math.max(1, Math.round(img.height * scale));
    var ctx = canvas.getContext("2d");
    ctx.scale(scale, scale);
    ctx.drawImage(img, 0, 0);

    var blob = await new Promise(function (resolve) {
      canvas.toBlob(resolve, "image/png");
    });
    if (!blob) {
      throw new Error("Dashboard PNG export failed.");
    }
    downloadBlob(blob, buildFilename());
  }

  window.RTHDashboardCapture = {
    exportPng: captureDashboardPng,
  };
})();
