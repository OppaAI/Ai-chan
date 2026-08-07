/* Grasp Studio — circular WM animation (memory-graph theme).
 * Size + glow track score (like retain). Live state from /api/*.
 */
const FACTOR_ORDER = [
  "emotion", "importance", "recency", "relevance",
  "novelty", "question", "entity", "recall_freq", "primacy",
];
const CIRC = 2 * Math.PI * 44; // r≈44 in viewBox 0..100

let isAnimating = false;
let dockRects = [];
let knownIds = new Set();
let lastState = null;

function $(id) { return document.getElementById(id); }

async function api(path, opts) {
  const r = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

function esc(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function scoreOf(s) { return Number(s.score) || 0; }

/** Map score → visual scale (0.72 … 1.18) and glow opacity */
function visualFromScore(score) {
  const t = Math.max(0, Math.min(1, score));
  return {
    scale: 0.72 + t * 0.46,
    glow: 0.06 + t * 0.28,
    brightness: 0.85 + t * 0.25,
  };
}

function slotKey(s, i) {
  return `${s.user_ts ?? s.created_turn ?? i}`;
}

function measureDocks() {
  dockRects = [];
  const stage = $("stage");
  if (!stage) return;
  const sr = stage.getBoundingClientRect();
  const max = lastState?.miller?.max ?? 9;
  for (let i = 0; i < max; i++) {
    const d = $("dock-" + i);
    if (!d) continue;
    const r = d.getBoundingClientRect();
    dockRects.push({
      x: r.left - sr.left + (r.width - 96) / 2,
      y: r.top - sr.top + (r.height - 96) / 2,
    });
  }
}

function buildRack(miller) {
  const rack = $("rack");
  const max = miller?.max ?? 9;
  const center = miller?.center ?? 7;
  rack.innerHTML = "";
  for (let i = 0; i < max; i++) {
    const dock = document.createElement("div");
    dock.className = "slot-dock" + (i < center ? "" : " overflow");
    dock.id = "dock-" + i;
    dock.innerHTML = `<span class="dock-num">${i + 1}</span>`;
    rack.appendChild(dock);
  }
  const dots = $("millerDots");
  dots.innerHTML = "";
  for (let i = 0; i < center; i++) {
    const d = document.createElement("div");
    d.className = "miller-dot";
    d.id = "gm" + (i + 1);
    dots.appendChild(d);
  }
  requestAnimationFrame(measureDocks);
}

function updateChrome(state) {
  const m = state.miller || {};
  $("modeBadge").textContent = state.mode || "demo";
  $("metaBar").textContent =
    `slots ${state.size ?? 0} · tokens ${state.total_tokens ?? 0}` +
    ` · miller ${m.min ?? "?"}/${m.center ?? "?"}/${m.max ?? "?"}` +
    ` · turn #${state.turn_counter ?? 0}`;

  const budget = state.token_budget > 0 ? state.token_budget : Math.max(2000, (state.total_tokens || 0) * 1.5 || 2000);
  const pct = Math.min(100, ((state.total_tokens || 0) / budget) * 100);
  $("budgetFill").style.width = pct + "%";
  $("budgetText").textContent =
    state.token_budget > 0
      ? `${state.total_tokens ?? 0} / ${state.token_budget}`
      : `${state.total_tokens ?? 0} tok`;

  const center = m.center ?? 7;
  const n = state.size ?? 0;
  for (let i = 1; i <= center; i++) {
    const d = $("gm" + i);
    if (!d) continue;
    d.className = "miller-dot";
    if (i <= n) {
      if (n <= (m.min ?? 5)) d.classList.add("on");
      else if (n < center) d.classList.add("warn");
      else d.classList.add("danger");
    }
  }
  $("gmLabel").textContent = `${n}/${center}`;

  const max = m.max ?? 9;
  for (let i = 0; i < max; i++) {
    const d = $("dock-" + i);
    if (d) d.classList.toggle("active", i < n);
  }
}

function createMemEl(slot, id) {
  const score = scoreOf(slot);
  const vis = visualFromScore(score);
  const d = document.createElement("div");
  d.className = "mem" + (score < 0.35 ? " weak" : "");
  d.dataset.id = id;
  d.style.filter = `brightness(${vis.brightness})`;
  d.style.boxShadow = `0 0 ${18 + score * 22}px rgba(168,136,232,${vis.glow})`;
  const dash = Math.max(0, Math.min(1, score)) * CIRC;
  d.innerHTML = `
    <svg viewBox="0 0 100 100">
      <circle class="track" cx="50" cy="50" r="44"/>
      <circle class="ring" cx="50" cy="50" r="44"
        stroke-dasharray="${dash} ${CIRC - dash}"/>
    </svg>
    <div class="mem-inner">
      <div class="mem-rank">#${slot._rank ?? "?"}</div>
      <div class="mem-score">${score.toFixed(2)}</div>
      <div class="mem-text">${esc((slot.user || "").slice(0, 28))}</div>
      <div class="mem-tokens">${slot.tokens ?? 0}t · r×${slot.recall_count ?? 0}</div>
    </div>`;
  d.onclick = () => showFactors(slot);
  $("stage").appendChild(d);
  return d;
}

function updateMemEl(el, slot, rank) {
  const score = scoreOf(slot);
  const vis = visualFromScore(score);
  el.classList.toggle("weak", score < 0.35);
  el.style.filter = `brightness(${vis.brightness})`;
  el.style.boxShadow = `0 0 ${18 + score * 22}px rgba(168,136,232,${vis.glow})`;
  const dash = Math.max(0, Math.min(1, score)) * CIRC;
  const ring = el.querySelector("circle.ring");
  if (ring) ring.setAttribute("stroke-dasharray", `${dash} ${CIRC - dash}`);
  const se = el.querySelector(".mem-score");
  if (se) se.textContent = score.toFixed(2);
  const re = el.querySelector(".mem-rank");
  if (re) re.textContent = "#" + rank;
  const te = el.querySelector(".mem-tokens");
  if (te) te.textContent = `${slot.tokens ?? 0}t · r×${slot.recall_count ?? 0}`;
  const tx = el.querySelector(".mem-text");
  if (tx) tx.textContent = (slot.user || "").slice(0, 28);
  el.dataset.scale = String(vis.scale);
  el.onclick = () => showFactors(slot);
}

function showFactors(slot) {
  const f = slot.factors || {};
  const rows = FACTOR_ORDER.map((k) => {
    const v = Number(f[k] ?? 0);
    const pct = Math.max(0, Math.min(100, v * 100));
    return `<div class="factor-row">
      <span>${k}</span>
      <div class="bar"><i style="width:${pct}%"></i></div>
      <span>${v.toFixed(2)}</span>
    </div>`;
  }).join("");
  $("factorPanel").className = "factor-panel";
  $("factorPanel").innerHTML =
    `<div style="margin-bottom:8px;color:var(--purple)">${esc((slot.user || "").slice(0, 80))}</div>` +
    rows +
    `<div style="margin-top:8px;color:var(--dim)">score <b style="color:var(--text)">${scoreOf(slot).toFixed(3)}</b></div>`;
}

function layoutNodes(slots, animate) {
  measureDocks();
  slots.forEach((s, i) => {
    const id = slotKey(s, i);
    const el = document.querySelector(`.mem[data-id="${CSS.escape(id)}"]`);
    if (!el || !dockRects[i]) return;
    const scale = Number(el.dataset.scale || 1);
    const x = dockRects[i].x;
    const y = dockRects[i].y;
    if (!animate) el.style.transition = "none";
    if (!el.style.left) {
      el.style.left = x + "px";
      el.style.top = y + "px";
      el.style.transform = `scale(${scale})`;
    } else {
      const bx = parseFloat(el.style.left) || 0;
      const by = parseFloat(el.style.top) || 0;
      el.style.transform = `translate(${x - bx}px, ${y - by}px) scale(${scale})`;
    }
    if (!animate) {
      el.offsetWidth;
      el.style.transition = "";
    }
    updateMemEl(el, s, i + 1);
  });
}

async function playMergeIntro(user, assistant, turnLabel) {
  const stage = $("stage");
  const sr = stage.getBoundingClientRect();
  const mr = $("mergeZone").getBoundingClientRect();
  const mx = mr.left - sr.left, my = mr.top - sr.top, mw = mr.width, mh = mr.height;

  const pU = document.createElement("div");
  pU.className = "pill u";
  pU.textContent = "U: " + (user || "").slice(0, 36);
  pU.style.left = mx + 24 + "px";
  pU.style.top = my + mh / 2 - 14 + "px";
  stage.appendChild(pU);

  const pA = document.createElement("div");
  pA.className = "pill a";
  pA.textContent = "A: " + (assistant || "").slice(0, 36);
  pA.style.left = mx + mw - 240 + "px";
  pA.style.top = my + mh / 2 - 14 + "px";
  stage.appendChild(pA);

  await sleep(40);
  pU.classList.add("show");
  pA.classList.add("show");
  await sleep(320);

  const cx = mx + mw / 2 - 40;
  pU.style.transform = `translate(${cx - (mx + 24)}px, 0) scale(0.9)`;
  pA.style.transform = `translate(${cx - (mx + mw - 240)}px, 0) scale(0.9)`;
  await sleep(320);

  $("mergeArrow").classList.add("show");
  pU.style.opacity = "0";
  pA.style.opacity = "0";

  const merged = document.createElement("div");
  merged.className = "pill merged show";
  merged.textContent = turnLabel || "Interaction";
  merged.style.left = cx + 10 + "px";
  merged.style.top = my + mh / 2 - 14 + "px";
  stage.appendChild(merged);
  await sleep(380);

  const spawn = { x: cx + 10, y: my + mh / 2 - 14 };
  pU.remove();
  pA.remove();
  merged.style.opacity = "0";
  setTimeout(() => merged.remove(), 300);
  $("mergeArrow").classList.remove("show");
  return spawn;
}

function renderEvictions(list) {
  const root = $("evictList");
  if (!list || !list.length) {
    root.innerHTML = `<div class="factor-panel muted">None yet</div>`;
    return;
  }
  root.innerHTML = list.slice(0, 12).map((e) =>
    `<div class="evict-item">
      <div><span class="sc">evicted</span> score ${(Number(e.score)||0).toFixed(3)} · t#${e.created_turn ?? "?"}</div>
      <div>${esc((e.user || "").slice(0, 64))}</div>
    </div>`
  ).join("");
  requestAnimationFrame(() => {
    root.querySelectorAll(".evict-item").forEach((el) => el.classList.add("show"));
  });
}

async function applyState(data, { animateNew = false, filled = null } = {}) {
  const state = data.state || data;
  lastState = state;
  const slots = state.slots || [];
  const miller = state.miller || { min: 5, center: 7, max: 9 };

  if (!$("dock-0") || $("rack").children.length !== (miller.max ?? 9)) {
    buildRack(miller);
    await sleep(30);
  }
  updateChrome(state);
  renderEvictions(data.evictions || state.evictions || []);

  const nextIds = new Set(slots.map((s, i) => slotKey(s, i)));

  document.querySelectorAll(".mem").forEach((el) => {
    if (!nextIds.has(el.dataset.id)) {
      el.style.transition = "opacity .35s, transform .45s";
      el.style.opacity = "0";
      el.style.transform += " scale(0.4)";
      setTimeout(() => el.remove(), 400);
    }
  });

  let spawn = null;
  if (animateNew && filled) {
    spawn = await playMergeIntro(
      filled.user,
      filled.assistant,
      `Turn #${state.turn_counter ?? ""}`
    );
  }

  slots.forEach((s, i) => {
    const id = slotKey(s, i);
    s._rank = i + 1;
    let el = document.querySelector(`.mem[data-id="${CSS.escape(id)}"]`);
    if (!el) {
      el = createMemEl(s, id);
      if (spawn) {
        el.style.left = spawn.x + "px";
        el.style.top = spawn.y + "px";
        el.style.opacity = "1";
        requestAnimationFrame(() => el.classList.add("entered"));
      } else {
        el.classList.add("entered");
      }
      knownIds.add(id);
    }
    updateMemEl(el, s, i + 1);
  });

  await sleep(40);
  layoutNodes(slots, true);
  knownIds = nextIds;
}

async function refresh() {
  if (isAnimating) return;
  isAnimating = true;
  try {
    const data = await api("/api/state");
    await applyState(data, { animateNew: false });
  } finally {
    isAnimating = false;
  }
}

async function seed() {
  if (isAnimating) return;
  isAnimating = true;
  try {
    document.querySelectorAll(".mem").forEach((m) => m.remove());
    knownIds.clear();
    const data = await api("/api/demo/seed", { method: "POST", body: "{}" });
    const state = data.state || data;
    const slots = state.slots || [];
    lastState = state;
    buildRack(state.miller || { max: 9, center: 7, min: 5 });
    updateChrome(state);
    renderEvictions(data.evictions || []);
    await sleep(40);
    for (let i = 0; i < slots.length; i++) {
      const s = slots[i];
      const id = slotKey(s, i);
      s._rank = i + 1;
      const el = createMemEl(s, id);
      measureDocks();
      if (dockRects[i]) {
        el.style.left = dockRects[i].x + "px";
        el.style.top = dockRects[i].y + "px";
      }
      requestAnimationFrame(() => el.classList.add("entered"));
      knownIds.add(id);
      await sleep(90);
    }
    layoutNodes(slots, true);
  } finally {
    isAnimating = false;
  }
}

async function touch() {
  if (isAnimating) return;
  isAnimating = true;
  try {
    const data = await api("/api/touch", { method: "POST", body: "{}" });
    const stage = $("stage");
    const stageRect = stage.getBoundingClientRect();
    document.querySelectorAll(".mem").forEach((el) => {
      el.classList.add("touched");
      const fl = document.createElement("div");
      fl.className = "float";
      fl.textContent = "+recall";
      const elRect = el.getBoundingClientRect();
      fl.style.left = (elRect.left - stageRect.left) + "px";
      fl.style.top = (elRect.top - stageRect.top) + "px";
      stage.appendChild(fl);
      setTimeout(() => fl.remove(), 900);
      setTimeout(() => el.classList.remove("touched"), 500);
    });
    await applyState(data, { animateNew: false });
  } finally {
    isAnimating = false;
  }
}

async function reset() {
  if (isAnimating) return;
  isAnimating = true;
  try {
    const data = await api("/api/reset", { method: "POST", body: "{}" });
    document.querySelectorAll(".mem").forEach((m) => m.remove());
    knownIds.clear();
    await applyState(data, { animateNew: false });
  } finally {
    isAnimating = false;
  }
}

async function fill() {
  if (isAnimating) return;
  const user = $("inUser").value;
  const assistant = $("inAsst").value;
  if (!user && !assistant) return;
  isAnimating = true;
  try {
    const data = await api("/api/fill", {
      method: "POST",
      body: JSON.stringify({ user, assistant }),
    });
    if (data.ok) {
      $("inUser").value = "";
      $("inAsst").value = "";
      await applyState(data, {
        animateNew: true,
        filled: { user, assistant },
      });
    } else {
      alert(data.error || "fill failed");
    }
  } finally {
    isAnimating = false;
  }
}

$("btnRefresh").onclick = () => refresh().catch(console.error);
$("btnSeed").onclick = () => seed().catch(console.error);
$("btnTouch").onclick = () => touch().catch(console.error);
$("btnReset").onclick = () => reset().catch(console.error);
$("btnFill").onclick = () => fill().catch(console.error);
$("backBtn").onclick = (e) => {
  e.preventDefault();
  if (history.length > 1) history.back();
  else location.href = "/";
};

window.addEventListener("resize", () => {
  if (lastState) layoutNodes(lastState.slots || [], false);
});

refresh().catch((e) => {
  $("metaBar").textContent = "API offline — start grasp studio backend";
  console.error(e);
});
