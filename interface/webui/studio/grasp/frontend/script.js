const FACTOR_ORDER = [
  "emotion", "importance", "recency", "relevance",
  "novelty", "question", "entity", "recall_freq",
  "primacy",
];

async function api(path, opts) {
  const r = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

function el(id) { return document.getElementById(id); }

function fmtScore(x) {
  return (Number(x) || 0).toFixed(3);
}

function renderState(data) {
  const state = data.state || data;
  const evictions = data.evictions || state.evictions || [];

  el("modeBadge").textContent = state.mode || "demo";
  const m = state.miller || {};
  el("metaBar").textContent =
    `slots ${state.size ?? 0} · tokens ${state.total_tokens ?? 0}` +
    ` · miller ${m.min ?? "?"}/${m.center ?? "?"}/${m.max ?? "?"}` +
    ` · turn #${state.turn_counter ?? 0}` +
    ` · anchor ${state.anchor_size ?? 0} tokens`;

  const root = el("slots");
  const slots = state.slots || [];
  if (!slots.length) {
    root.className = "slots empty";
    root.textContent = "No slots yet — Seed demo or fill a turn.";
  } else {
    root.className = "slots";
    root.innerHTML = slots.map((s, i) => {
      const factors = s.factors || {};
      const weak = (s.score || 0) < 0.35;
      const bars = FACTOR_ORDER.map((k) => {
        const v = Number(factors[k] ?? 0);
        const pct = Math.max(0, Math.min(100, v * 100));
        return `<div class="factor">
          <div class="name"><span>${k}</span><span>${v.toFixed(2)}</span></div>
          <div class="track"><div class="fill" style="width:${pct}%"></div></div>
        </div>`;
      }).join("");
      return `<article class="slot${weak ? " weak" : ""}">
        <div class="rank">#${i + 1}</div>
        <div class="body">
          <div class="lines"><span class="u">U:</span> ${escapeHtml(s.user || "")}</div>
          <div class="lines"><span class="a">A:</span> ${escapeHtml(s.assistant || "")}</div>
          <div class="stats">
            <span>score <strong>${fmtScore(s.score)}</strong></span>
            <span>tokens ${s.tokens ?? 0}</span>
            <span>created @${s.created_turn ?? "?"}</span>
            <span>recall ×${s.recall_count ?? 0}</span>
          </div>
          <div class="factors">${bars}</div>
        </div>
      </article>`;
    }).join("");
  }

  const eroot = el("evictions");
  if (!evictions.length) {
    eroot.className = "evict-list empty";
    eroot.textContent = "None yet";
  } else {
    eroot.className = "evict-list";
    eroot.innerHTML = evictions.map((e) =>
      `<div class="evict">
        <div><span class="sc">evicted</span> score ${fmtScore(e.score)} · recall ×${e.recall_count ?? 0} · t#${e.created_turn ?? "?"}</div>
        <div>${escapeHtml((e.user || "").slice(0, 100))}</div>
      </div>`
    ).join("");
  }
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

async function refresh() {
  const data = await api("/api/state");
  renderState(data);
}

async function seed() {
  const data = await api("/api/demo/seed", { method: "POST", body: "{}" });
  renderState(data);
}

async function touch() {
  const data = await api("/api/touch", { method: "POST", body: "{}" });
  renderState(data);
}

async function reset() {
  const data = await api("/api/reset", { method: "POST", body: "{}" });
  renderState(data);
}

async function fill() {
  const user = el("inUser").value;
  const assistant = el("inAsst").value;
  const data = await api("/api/fill", {
    method: "POST",
    body: JSON.stringify({ user, assistant }),
  });
  if (data.ok) {
    el("inUser").value = "";
    el("inAsst").value = "";
    renderState(data);
  } else {
    alert(data.error || "fill failed");
  }
}

el("btnRefresh").onclick = () => refresh().catch(console.error);
el("btnSeed").onclick = () => seed().catch(console.error);
el("btnTouch").onclick = () => touch().catch(console.error);
el("btnReset").onclick = () => reset().catch(console.error);
el("btnFill").onclick = () => fill().catch(console.error);

refresh().catch((e) => {
  el("metaBar").textContent = "API offline — start studio backend";
  console.error(e);
});
