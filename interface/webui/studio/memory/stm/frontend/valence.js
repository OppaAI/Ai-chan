/* valence overlay — positive / neutral / negative (memory-graph parity) */
(function () {
  function valenceOf(slot) {
    let v = (slot.valence_tag || "").toLowerCase();
    if (v === "pos") v = "positive";
    if (v === "neg") v = "negative";
    if (!v || v === "neutral") {
      let s = Number(slot.emotion ?? slot.valence_score);
      if (Number.isNaN(s) || (s === 0 && slot.emotion == null && slot.valence_score == null)) {
        const fe = slot.factors && slot.factors.emotion;
        if (fe != null) s = Number(fe) * 2 - 1; // factor is 0..1
      }
      if (!Number.isNaN(s)) {
        if (s >= 0.25) v = "positive";
        else if (s <= -0.25) v = "negative";
        else v = "neutral";
      } else v = "neutral";
    }
    if (v === "positive" || v === "pos") return "positive";
    if (v === "negative" || v === "neg") return "negative";
    return "neutral";
  }
  function valenceHue(slot) {
    const v = valenceOf(slot);
    if (v === "positive") return "pos";
    if (v === "negative") return "neg";
    return "neutral";
  }
  function valenceColor(hue) {
    if (hue === "pos") return "#f0c14a";
    if (hue === "neg") return "#3de0ff";
    return "#8a9bb8";
  }
  function valenceGlow(hue, alpha) {
    if (hue === "pos") return `rgba(240,193,74,${alpha})`;
    if (hue === "neg") return `rgba(61,224,255,${alpha})`;
    return `rgba(168,136,232,${alpha})`;
  }

  const _create = typeof createMemEl === "function" ? createMemEl : null;
  const _update = typeof updateMemEl === "function" ? updateMemEl : null;
  const _show = typeof showFactors === "function" ? showFactors : null;
  if (!_create || !_update) return;

  createMemEl = function (slot, id) {
    const el = _create(slot, id);
    const score = scoreOf(slot);
    const vis = visualFromScore(score);
    const vHue = valenceHue(slot);
    el.classList.remove("valence-pos", "valence-neg", "valence-neutral");
    el.classList.add("valence-" + vHue);
    el.dataset.valence = valenceOf(slot);
    // Valence tints the halo; intensity still tracks score (important = bright)
    const px = vis.glowPx != null ? vis.glowPx : 8 + Math.pow(score, 0.85) * 36;
    el.style.boxShadow = `0 0 ${px}px ${valenceGlow(vHue, vis.glow)}`;
    if (!el.querySelector(".mem-valence")) {
      const badge = document.createElement("div");
      badge.className = "mem-valence";
      badge.textContent = valenceOf(slot)[0].toUpperCase();
      const inner = el.querySelector(".mem-inner");
      if (inner) inner.appendChild(badge);
    } else {
      el.querySelector(".mem-valence").textContent = valenceOf(slot)[0].toUpperCase();
    }
    return el;
  };

  updateMemEl = function (el, slot, rank) {
    _update(el, slot, rank);
    const score = scoreOf(slot);
    const vis = visualFromScore(score);
    const vHue = valenceHue(slot);
    el.classList.remove("valence-pos", "valence-neg", "valence-neutral");
    el.classList.add("valence-" + vHue);
    el.dataset.valence = valenceOf(slot);
    const px = vis.glowPx != null ? vis.glowPx : 8 + Math.pow(score, 0.85) * 36;
    el.style.boxShadow = `0 0 ${px}px ${valenceGlow(vHue, vis.glow)}`;
    let badge = el.querySelector(".mem-valence");
    if (!badge) {
      badge = document.createElement("div");
      badge.className = "mem-valence";
      const inner = el.querySelector(".mem-inner");
      if (inner) inner.appendChild(badge);
    }
    if (badge) badge.textContent = valenceOf(slot)[0].toUpperCase();
  };

  if (_show) {
    showFactors = function (slot) {
      _show(slot);
      const v = valenceOf(slot);
      const vColor = valenceColor(valenceHue(slot));
      const panel = document.getElementById("factorPanel");
      if (!panel) return;
      const row = document.createElement("div");
      row.style.marginTop = "4px";
      row.style.color = "var(--dim)";
      row.innerHTML =
        `valence <b style="color:${vColor}">${v}</b>` +
        (slot.emotion != null
          ? ` <span style="opacity:.7">(${Number(slot.emotion).toFixed(2)})</span>`
          : "");
      panel.appendChild(row);
    };
  }
})();
