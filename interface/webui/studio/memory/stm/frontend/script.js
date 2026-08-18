/* STM Studio — circular WM animation (memory-graph theme).
 * Size + glow track score (like retain). Live state from /api/*.
 * Demo is opt-in (default stopped); Live is the default source.
 */
const API_ROOT = GraphBoot.apiBase();
const FACTOR_ORDER = [
  "emotion", "importance", "recency", "relevance",
  "novelty", "question", "entity", "recall_freq", "primacy",
];
const CIRC = 2 * Math.PI * 44; // r≈44 in viewBox 0..100

let isAnimating = false;
let dockRects = [];
let lastState = null;
let liveOn = true;           // Live is the default
let livePollTimer = null;
let lastTurnSeen = null;

let demoRunning = false;     // Demo default stopped
let demoAbort = false;
let demoPromise = null;

function $(id) { return document.getElementById(id); }

async function api(path, opts) {
  const r = await fetch(API_ROOT + path, {
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

/** Map score → visual scale / glow / brightness.
 *  Wider range so important memories glow brightly and low-score
 *  candidates about to be evicted look clearly dimmer. */
function visualFromScore(score) {
  const t = Math.max(0, Math.min(1, score));
  // Power curve: low scores stay small/dim, high scores expand
  const p = Math.pow(t, 0.85);
  return {
    scale: 0.62 + p * 0.62,          // ~0.62 … 1.24
    glow: 0.04 + p * 0.55,           // much wider glow range
    glowPx: 8 + p * 36,              // radius of halo
    brightness: 0.72 + p * 0.42,     // 0.72 … 1.14
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
  const mode = state.mode || (liveOn ? "live" : "demo");
  $("modeBadge").textContent = mode;
  $("metaBar").textContent =
    `slots ${state.size ?? 0} · tokens ${state.total_tokens ?? 0}` +
    ` · miller ${m.min ?? "?"}/${m.center ?? "?"}/${m.max ?? "?"}` +
    ` · turn #${state.turn_counter ?? 0}`;

  const budget = state.token_budget > 0
    ? state.token_budget
    : Math.max(2000, (state.total_tokens || 0) * 1.5 || 2000);
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

function applyScoreVisual(el, score) {
  const vis = visualFromScore(score);
  el.classList.toggle("weak", score < 0.35);
  el.style.filter = `brightness(${vis.brightness})`;
  // Glow intensity + radius both scale with score (important = bright, decaying = dim)
  el.style.boxShadow = `0 0 ${vis.glowPx}px rgba(168,136,232,${vis.glow})`;
  el.dataset.scale = String(vis.scale);
  const dash = Math.max(0, Math.min(1, score)) * CIRC;
  const ring = el.querySelector("circle.ring");
  if (ring) ring.setAttribute("stroke-dasharray", `${dash} ${CIRC - dash}`);
  return vis;
}

function createMemEl(slot, id) {
  const score = scoreOf(slot);
  const d = document.createElement("div");
  d.className = "mem" + (score < 0.35 ? " weak" : "");
  d.dataset.id = id;
  applyScoreVisual(d, score);
  d.innerHTML = `
    <svg viewBox="0 0 100 100">
      <circle class="track" cx="50" cy="50" r="44"/>
      <circle class="ring" cx="50" cy="50" r="44"
        stroke-dasharray="${Math.max(0, Math.min(1, score)) * CIRC} ${CIRC}"/>
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
  applyScoreVisual(el, score);
  const se = el.querySelector(".mem-score");
  if (se) se.textContent = score.toFixed(2);
  const re = el.querySelector(".mem-rank");
  if (re) re.textContent = "#" + rank;
  const te = el.querySelector(".mem-tokens");
  if (te) te.textContent = `${slot.tokens ?? 0}t · r×${slot.recall_count ?? 0}`;
  const tx = el.querySelector(".mem-text");
  if (tx) tx.textContent = (slot.user || "").slice(0, 28);
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

/** Slow, readable join: user + assistant drift together, fuse, then drop into STM. */
async function playMergeIntro(user, assistant, turnLabel) {
  const stage = $("stage");
  const sr = stage.getBoundingClientRect();
  const mr = $("mergeZone").getBoundingClientRect();
  const mx = mr.left - sr.left, my = mr.top - sr.top, mw = mr.width, mh = mr.height;

  const pU = document.createElement("div");
  pU.className = "pill u";
  pU.innerHTML = `<span class="pill-tag">USER</span><span class="pill-body">${esc((user || "").slice(0, 40))}</span>`;
  pU.style.left = mx + 18 + "px";
  pU.style.top = my + mh / 2 - 22 + "px";
  stage.appendChild(pU);

  const pA = document.createElement("div");
  pA.className = "pill a";
  pA.innerHTML = `<span class="pill-tag">ASST</span><span class="pill-body">${esc((assistant || "").slice(0, 40))}</span>`;
  pA.style.left = mx + mw - 250 + "px";
  pA.style.top = my + mh / 2 - 22 + "px";
  stage.appendChild(pA);

  await sleep(80);
  pU.classList.add("show");
  await sleep(520);          // linger so the pair is readable
  pA.classList.add("show");
  await sleep(480);

  // Drift toward center — slower so the join is obvious
  const cx = mx + mw / 2 - 70;
  pU.style.transition = "transform 0.85s cubic-bezier(0.22, 1, 0.36, 1), opacity 0.4s ease";
  pA.style.transition = "transform 0.85s cubic-bezier(0.22, 1, 0.36, 1), opacity 0.4s ease";
  pU.style.transform = `translate(${cx - (mx + 18)}px, 0) scale(0.92)`;
  pA.style.transform = `translate(${cx - (mx + mw - 250)}px, 0) scale(0.92)`;
  await sleep(900);

  $("mergeArrow").classList.add("show");
  await sleep(220);

  // Soft dissolve into interaction chip
  pU.style.opacity = "0";
  pA.style.opacity = "0";

  const merged = document.createElement("div");
  merged.className = "pill merged show";
  merged.innerHTML = `<span class="pill-tag">INTERACTION</span><span class="pill-body">${esc(turnLabel || "Turn")}</span>`;
  merged.style.left = cx + 10 + "px";
  merged.style.top = my + mh / 2 - 22 + "px";
  stage.appendChild(merged);
  await sleep(700);          // hold the fused interaction

  // Sink toward the rack
  merged.style.transition = "transform 0.7s ease, opacity 0.5s ease";
  merged.style.transform = "translateY(36px) scale(0.85)";
  await sleep(380);
  merged.style.opacity = "0";

  const spawn = { x: cx + 10, y: my + mh / 2 - 14 };
  pU.remove();
  pA.remove();
  setTimeout(() => merged.remove(), 320);
  $("mergeArrow").classList.remove("show");
  await sleep(120);
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
      el.style.transition = "opacity .35s, transform .45s, filter .3s, box-shadow .3s";
      el.style.opacity = "0";
      el.style.transform += " scale(0.4)";
      el.style.filter = "brightness(0.5)";
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
    }
    updateMemEl(el, s, i + 1);
  });

  await sleep(40);
  layoutNodes(slots, true);
}

/* ---------- Live control (default ON) ---------- */

function setLiveUI(on) {
  liveOn = on;
  const btn = $("btnLive");
  const ind = $("liveIndicator");
  const label = $("liveLabel");
  if (!btn) return;
  btn.classList.toggle("is-live", on);
  btn.classList.toggle("is-stopped", !on);
  btn.setAttribute("aria-pressed", String(on));
  if (on) {
    ind.className = "live-indicator recording";
    label.textContent = "Live";
  } else {
    ind.className = "live-indicator stopped";
    label.textContent = "Live";
  }
  if ($("modeBadge") && !demoRunning) {
    $("modeBadge").textContent = on ? "live" : "idle";
  }
}

function toggleLive() {
  if (liveOn) {
    // Stop live → red dot becomes stop square, text dims
    setLiveUI(false);
  } else {
    // Resume live (and stop demo if running)
    if (demoRunning) stopDemo();
    setLiveUI(true);
    pollLive();
  }
}

/* ---------- Demo (default stopped) ---------- */

function setDemoUI(running) {
  demoRunning = running;
  const play = $("btnDemoPlay");
  const pause = $("btnDemoPause");
  const status = $("demoStatus");
  if (play) play.disabled = running;
  if (pause) pause.disabled = !running;
  if (status) status.textContent = running ? "Demo running…" : "Demo stopped";
  if ($("modeBadge")) $("modeBadge").textContent = running ? "demo" : (liveOn ? "live" : "idle");
}

function stopDemo() {
  demoAbort = true;
  demoRunning = false;
  setDemoUI(false);
}

async function runFullDemo() {
  if (demoRunning || isAnimating) return;
  // Demo takes over: pause live so the scripted sequence is visible
  setLiveUI(false);
  demoAbort = false;
  setDemoUI(true);

  const script = [
    { kind: "seed" },
    { kind: "wait", ms: 900 },
    { kind: "fill", user: "Remember my favorite color is violet.", assistant: "Got it — violet is locked in." },
    { kind: "wait", ms: 1400 },
    { kind: "fill", user: "I prefer short answers.", assistant: "Understood. I'll keep replies tight." },
    { kind: "wait", ms: 1400 },
    { kind: "fill", user: "Don't bring up work after 9pm.", assistant: "Noted. Quiet hours after 9." },
    { kind: "wait", ms: 1200 },
    { kind: "touch" },
    { kind: "wait", ms: 1000 },
    { kind: "fill", user: "What's my favorite color?", assistant: "Violet — you told me earlier." },
    { kind: "wait", ms: 1400 },
    { kind: "fill", user: "Random fact: I like rain sounds.", assistant: "Rain sounds noted." },
    { kind: "wait", ms: 1200 },
    { kind: "fill", user: "Another filler to pressure the Miller window.", assistant: "Slot pressure rising." },
    { kind: "wait", ms: 1200 },
    { kind: "fill", user: "Yet more context so eviction can fire.", assistant: "Working memory is near capacity." },
    { kind: "wait", ms: 1200 },
    { kind: "fill", user: "One more — expect a weak item to recede.", assistant: "Eviction likely." },
    { kind: "wait", ms: 1600 },
    { kind: "touch" },
    { kind: "wait", ms: 800 },
  ];

  try {
    for (const step of script) {
      if (demoAbort) break;
      if (step.kind === "wait") {
        await sleep(step.ms);
      } else if (step.kind === "seed") {
        await seed({ fromDemo: true });
      } else if (step.kind === "fill") {
        // Drive the same animation path as a real fill
        isAnimating = true;
        try {
          const data = await api("/fill", {
            method: "POST",
            body: JSON.stringify({ user: step.user, assistant: step.assistant }),
          });
          if (data.ok) {
            await applyState(data, {
              animateNew: true,
              filled: { user: step.user, assistant: step.assistant },
            });
          }
        } finally {
          isAnimating = false;
        }
      } else if (step.kind === "touch") {
        await touch({ fromDemo: true });
      }
    }
  } catch (e) {
    console.error("Demo error", e);
  } finally {
    setDemoUI(false);
    demoAbort = false;
  }
}

/* ---------- API actions ---------- */

async function refreshCognition() {
  try {
    const data = await api("/cognition");
    const e = data.evaluation || {};
    const panel = $("cognitionPanel");
    if (panel) panel.textContent = [
      "state: " + (e.state_status || "unknown"),
      "population: " + Number(e.state_population || 0).toFixed(2),
      "goals: " + (e.active_goals || 0) + " · loops: " + (e.open_loops || 0),
      "lessons: " + (e.durable_lessons || 0),
      "tool success: " + (e.tool_success_rate == null ? "—" : Math.round(e.tool_success_rate * 100) + "%"),
    ].join("\n");
  } catch (err) {
    const panel = $("cognitionPanel");
    if (panel) panel.textContent = "Cognitive metrics unavailable";
  }
}

async function refresh() {
  if (isAnimating) return;
  isAnimating = true;
  try {
    const data = await api("/state");
    await refreshCognition();
    await applyState(data, { animateNew: false });
    lastTurnSeen = Number((data.state || data).turn_counter ?? 0);
  } finally {
    isAnimating = false;
  }
}

async function seed(opts = {}) {
  if (isAnimating && !opts.fromDemo) return;
  isAnimating = true;
  try {
    document.querySelectorAll(".mem").forEach((m) => m.remove());
    const data = await api("/demo/seed", { method: "POST", body: "{}" });
    const state = data.state || data;
    const slots = state.slots || [];
    lastState = state;
    buildRack(state.miller || { max: 9, center: 7, min: 5 });
    updateChrome(state);
    renderEvictions(data.evictions || []);
    await sleep(40);
    for (let i = 0; i < slots.length; i++) {
      if (demoAbort) break;
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
      await sleep(110);
    }
    layoutNodes(slots, true);
  } finally {
    isAnimating = false;
  }
}

async function touch(opts = {}) {
  if (isAnimating && !opts.fromDemo) return;
  isAnimating = true;
  try {
    const data = await api("/touch", { method: "POST", body: "{}" });
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
  stopDemo();
  isAnimating = true;
  try {
    const data = await api("/reset", { method: "POST", body: "{}" });
    document.querySelectorAll(".mem").forEach((m) => m.remove());
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
    const data = await api("/fill", {
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

/* ---------- Wire UI ---------- */

$("btnRefresh").onclick = () => refresh().catch(console.error);
$("btnLive").onclick = () => toggleLive();
$("btnSeed").onclick = () => seed().catch(console.error);
$("btnTouch").onclick = () => touch().catch(console.error);
$("btnReset").onclick = () => reset().catch(console.error);
$("btnFill").onclick = () => fill().catch(console.error);

$("btnDemoPlay").onclick = () => {
  if (demoRunning) return;
  runFullDemo().catch(console.error);
};
$("btnDemoPause").onclick = () => stopDemo();

$("backBtn").onclick = (e) => {
  e.preventDefault();
  if (history.length > 1) history.back();
  else location.href = "/";
};

let resizeRaf = 0;
window.addEventListener("resize", () => {
  if (!lastState) return;
  if (resizeRaf) cancelAnimationFrame(resizeRaf);
  resizeRaf = requestAnimationFrame(() => {
    resizeRaf = 0;
    layoutNodes(lastState.slots || [], false);
  });
});

async function pollLive() {
  if (!liveOn || isAnimating || demoRunning) return;
  try {
    const data = await api("/state?mode=auto");
    const state = data.state || data;
    const turn = Number(state.turn_counter ?? 0);
    const live = state.mode === "live";
    if (!live) return;
    if (live && lastTurnSeen !== null && turn > lastTurnSeen) {
      const slot = (state.slots || []).reduce(
        (a, s) =>
          !a || Number(s.created_turn ?? -1) > Number(a.created_turn ?? -1) ? s : a,
        null
      );
      lastTurnSeen = turn;
      await applyState(data, {
        animateNew: true,
        filled: slot ? { user: slot.user || "", assistant: slot.assistant || "" } : null,
      });
    } else {
      await applyState(data, { animateNew: false });
      if (lastTurnSeen === null || (live && turn > lastTurnSeen)) lastTurnSeen = turn;
    }
  } catch (e) {
    console.debug("STM live poll failed", e);
  }
}

// Default: Live ON, Demo stopped
setLiveUI(true);
setDemoUI(false);

refresh().catch((e) => {
  $("metaBar").textContent = "API offline — start STM studio backend";
  console.error(e);
});

livePollTimer = setInterval(pollLive, 600);
