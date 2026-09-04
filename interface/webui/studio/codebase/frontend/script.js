const svg = d3.select("#canvas");
const W = 560, H = 720;
let nodes = [], edges = [];
let graphRequestGeneration = 0;
let scene;
let simulation = null;
let activeSystems = new Set();
let selectedId = null;
const statusEl = document.getElementById("status");
const statsEl = document.getElementById("stats");
const detailsEl = document.getElementById("details");
const searchInput = document.getElementById("search-q");
const searchBtn = document.getElementById("search-btn");
const searchHits = document.getElementById("search-hits");
const searchStats = document.getElementById("search-stats");
const systemFilters = document.getElementById("system-filters");
const showLabels = document.getElementById("show-labels");
const limitInput = document.getElementById("limit");

function setStatus(text) { statusEl.textContent = text; }
function esc(text) { const node = document.createElement("span"); node.textContent = text || ""; return node.innerHTML; }

const SYSTEMS = {
  brain:     { label: "Brain",     color: "#c651a8", glow: "#ff6eb3" },
  senses:    { label: "Senses",    color: "#51d4c8", glow: "#64ffd2" },
  core:      { label: "Core",      color: "#a888e8", glow: "#d8bcff" },
  tools:     { label: "Arms",      color: "#7298e8", glow: "#8ab4ff" },
  interface: { label: "Interface", color: "#8ab4ff", glow: "#b0c8ff" },
  training:  { label: "Training",  color: "#e88c6a", glow: "#ffae8e" },
  tests:     { label: "Tests",     color: "#9aa0b4", glow: "#bcc0d0" },
  support:   { label: "Support",   color: "#887b9a", glow: "#a89bb8" },
};

const MODULES = [
  { id: "stm",      name: "STM",      system: "brain",     desc: "Short-term memory buffer. Holds recent conversation context and working memory.", files: ["memory/stm.py", "memory/buffer.py"] },
  { id: "ltm",      name: "LTM",      system: "brain",     desc: "Long-term memory store. Vector-based embedding storage for persistent facts.", files: ["memory/ltm.py", "memory/vectors.py"] },
  { id: "itm",      name: "ITM",      system: "brain",     desc: "Intermediate-term memory. Compresses STM into durable LTM entries.", files: ["memory/itm.py", "memory/summarize.py"] },
  { id: "kb",       name: "KB",       system: "brain",     desc: "Knowledge base. Structured fact graph and entity resolver.", files: ["memory/kb.py", "memory/graph.py"] },
  { id: "reason",   name: "Reasoner", system: "brain",     desc: "Cognitive reasoning engine. Chain-of-thought planner.", files: ["cognition/reason.py", "cognition/chain.py"] },
  { id: "gate",     name: "Gate",     system: "brain",     desc: "Attention gate. Filters stimuli reaching working memory.", files: ["cognition/gate.py", "cognition/attention.py"] },
  { id: "vision",   name: "Vision",   system: "senses",    desc: "Visual perception. Image parsing, OCR, scene description.", files: ["senses/vision.py", "senses/ocr.py"] },
  { id: "hearing",  name: "Hearing",  system: "senses",    desc: "Audio input. Speech-to-text and sound event detection.", files: ["senses/audio.py", "senses/stt.py"] },
  { id: "web",      name: "Web",      system: "senses",    desc: "Web sense. Live page fetching and search integration.", files: ["senses/web.py", "senses/search.py"] },
  { id: "touch",    name: "Touch",    system: "senses",    desc: "Input touch handler. File uploads and gesture parsing.", files: ["senses/touch.py", "senses/gesture.py"] },
  { id: "voice",    name: "Voice",    system: "senses",    desc: "Speech synthesis and voice output formatting.", files: ["senses/tts.py", "senses/voice.py"] },
  { id: "state",    name: "State",    system: "core",      desc: "Central state manager. Session lifecycle and context.", files: ["core/state.py", "core/session.py"] },
  { id: "config",   name: "Config",   system: "core",      desc: "Configuration registry. Environment and secrets store.", files: ["core/config.py", "core/env.py"] },
  { id: "identity", name: "Identity", system: "core",      desc: "Personality module. Tone rules and persona switching.", files: ["core/identity.py", "core/persona.py"] },
  { id: "bus",      name: "Bus",      system: "core",      desc: "Internal message bus. Pub-sub backbone.", files: ["core/bus.py", "core/events.py"] },
  { id: "coder",    name: "Coder",    system: "tools",     desc: "Code generation and execution. Python sandbox.", files: ["tools/coder.py", "tools/sandbox.py"] },
  { id: "shell",    name: "Shell",    system: "tools",     desc: "System shell interface. Safe command execution.", files: ["tools/shell.py", "tools/guard.py"] },
  { id: "browser",  name: "Browser",  system: "tools",     desc: "Headless browser automation and screenshots.", files: ["tools/browser.py", "tools/puppet.py"] },
  { id: "files",    name: "Files",    system: "tools",     desc: "File system manager. Read, write, compress.", files: ["tools/files.py", "tools/fs.py"] },
  { id: "calc",     name: "Calc",     system: "tools",     desc: "Math and data toolkit. Calculator and plotter.", files: ["tools/calc.py", "tools/plot.py"] },
  { id: "sched",    name: "Scheduler", system: "core",     desc: "Task scheduler. Cron-like job runner.", files: ["orchestra/scheduler.py"] },
  { id: "pipeline", name: "Pipeline",  system: "core",     desc: "Workflow pipeline. DAG-based step runner.", files: ["orchestra/pipeline.py"] },
  { id: "agent",    name: "Agent",     system: "core",     desc: "Main agent orchestration loop.", files: ["orchestra/agent.py"] },
  { id: "time",     name: "Clock",     system: "core",     desc: "Time keeper. NTP sync and timers.", files: ["orchestra/clock.py"] },
  { id: "health",   name: "Health",    system: "core",     desc: "Health monitor. Self-diagnostics.", files: ["orchestra/health.py"] },
  { id: "ui",       name: "WebUI",     system: "interface", desc: "Web front-end, static assets, API.", files: ["interface/webui/webui.py"] },
  { id: "train",    name: "Trainer",   system: "training",  desc: "Model fine-tuning pipelines.", files: ["training/"] },
  { id: "tests",    name: "Tests",     system: "tests",     desc: "Unit, integration, and stress test suite.", files: ["tests/"] },
];

const LINKS = [
  { s: "stm", t: "reason" }, { s: "stm", t: "gate" }, { s: "ltm", t: "itm" },
  { s: "itm", t: "stm" }, { s: "kb", t: "reason" }, { s: "gate", t: "state" },
  { s: "reason", t: "state" }, { s: "vision", t: "gate" }, { s: "hearing", t: "gate" },
  { s: "web", t: "gate" }, { s: "touch", t: "state" }, { s: "voice", t: "state" },
  { s: "state", t: "bus" }, { s: "config", t: "state" }, { s: "identity", t: "state" },
  { s: "bus", t: "agent" }, { s: "agent", t: "coder" }, { s: "agent", t: "shell" },
  { s: "agent", t: "browser" }, { s: "agent", t: "files" }, { s: "agent", t: "calc" },
  { s: "sched", t: "agent" }, { s: "pipeline", t: "agent" }, { s: "health", t: "state" },
  { s: "coder", t: "calc" }, { s: "browser", t: "web" }, { s: "ui", t: "state" },
  { s: "ui", t: "vision" }, { s: "ui", t: "hearing" }, { s: "ui", t: "voice" },
  { s: "ui", t: "kb" }, { s: "ui", t: "ltm" }, { s: "ui", t: "stm" },
  { s: "train", t: "state" }, { s: "tests", t: "agent" }, { s: "tests", t: "ui" },
];

function drawBackdrop() {
  const defs = svg.append("defs");

  const glow = defs.append("filter").attr("id", "glow-soft");
  glow.append("feGaussianBlur").attr("stdDeviation", "3").attr("result", "b");
  glow.append("feMerge").selectAll("feMergeNode").data(["b","SourceGraphic"]).enter().append("feMergeNode").attr("in", d=>d);

  const halo = defs.append("filter").attr("id", "glow-strong");
  halo.append("feGaussianBlur").attr("stdDeviation", "6").attr("result", "b");
  halo.append("feMerge").selectAll("feMergeNode").data(["b","SourceGraphic"]).enter().append("feMergeNode").attr("in", d=>d);

  // Soft web background — radial gradient + faint grid
  const bg = defs.append("radialGradient").attr("id", "bg-grad").attr("cx", "50%").attr("cy", "50%").attr("r", "70%");
  bg.append("stop").attr("offset", "0%").attr("stop-color", "rgba(168,136,232,0.08)");
  bg.append("stop").attr("offset", "100%").attr("stop-color", "rgba(15,10,24,0)");

  svg.append("rect").attr("x", 0).attr("y", 0).attr("width", W).attr("height", H).attr("fill", "url(#bg-grad)").attr("class", "bg");

  scene = svg.append("g").attr("id", "viewport");
  scene.append("g").attr("id", "edges");
  scene.append("g").attr("id", "nodes");
}

function graphNodeToDisplay(node) {
  const system = SYSTEMS[node.system] ? node.system : "support";
  const moduleName = node.module || node.path || node.id;
  return {
    id: node.id,
    module: moduleName,
    name: moduleName.split("/").pop(),
    system,
    color: node.color || SYSTEMS[system].color,
    // seed from backend (stable across reloads); force layout will overwrite these
    x: W * (0.1 + 0.8 * (node.x_seed ?? Math.random())),
    y: H * (0.1 + 0.8 * (node.y_seed ?? Math.random())),
    r: Math.max(8, Math.min(20, 7 + Math.log2((node.file_count || 1) + 1) * 3)),
    desc: `${node.title || "Indexed module"} · ${node.loc || 0} lines · ${node.function_count || 0} functions`,
    files: [moduleName],
    complexity: node.complexity || 1,
    dependency_count: node.dependency_count || 0,
  };
}

function startSimulation() {
  if (simulation) simulation.stop();
  simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(edges).id(d => d.id).distance(d => 60 + (1 - (d.weight || 0.2)) * 60).strength(d => 0.15 + (d.weight || 0.2) * 0.4))
    .force("charge", d3.forceManyBody().strength(-260).distanceMax(420))
    .force("center", d3.forceCenter(W / 2, H / 2))
    .force("collide", d3.forceCollide().radius(d => d.r + 4))
    .force("x", d3.forceX(W / 2).strength(0.04))
    .force("y", d3.forceY(H / 2).strength(0.04))
    .alpha(1)
    .alphaDecay(0.025)
    .on("tick", tick);
}

function tick() {
  // Clamp nodes inside the canvas so nothing escapes the viewport
  nodes.forEach(n => {
    n.x = Math.max(n.r + 4, Math.min(W - n.r - 4, n.x));
    n.y = Math.max(n.r + 4, Math.min(H - n.r - 4, n.y));
  });

  scene.select("#edges").selectAll("line.edge").each(function(d) {
    const s = nodes.find(n => n.id === d.source.id || n.id === d.source);
    const t = nodes.find(n => n.id === d.target.id || n.id === d.target);
    if (!s || !t) return;
    d3.select(this).attr("x1", s.x).attr("y1", s.y).attr("x2", t.x).attr("y2", t.y);
  });
  scene.select("#edges").selectAll("line.edge-spark").each(function(d) {
    const s = nodes.find(n => n.id === d.source.id || n.id === d.source);
    const t = nodes.find(n => n.id === d.target.id || n.id === d.target);
    if (!s || !t) return;
    d3.select(this).attr("x1", s.x).attr("y1", s.y).attr("x2", t.x).attr("y2", t.y);
  });
  scene.select("#nodes").selectAll(".node").attr("transform", d => `translate(${d.x},${d.y})`);
}

// ── Edges with curved sparks ──
function drawEdges() {
  const edgeGroup = scene.select("#edges");
  edgeGroup.selectAll("*").remove();

  edges.forEach(l => {
    const color = SYSTEMS[nodes.find(n => n.id === (l.source.id || l.source))?.system]?.color || "#4a3a6a";

    edgeGroup.append("line")
      .attr("class", "edge")
      .datum(l)
      .attr("data-source", l.source.id || l.source)
      .attr("data-target", l.target.id || l.target)
      .attr("stroke", color)
      .attr("stroke-width", l.kind === "dependency" ? 1.1 : 0.6)
      .attr("stroke-opacity", l.kind === "dependency" ? 0.35 : 0.18)
      .attr("stroke-dasharray", l.kind === "dependency" ? null : "2,4");

    edgeGroup.append("line")
      .attr("class", "edge-spark")
      .datum(l)
      .attr("stroke", color)
      .attr("stroke-width", 1.4)
      .attr("stroke-dasharray", "6 60")
      .attr("stroke-opacity", 0.9)
      .style("animation", `data-spark ${2 + Math.random() * 3}s linear infinite`)
      .style("animation-delay", `${Math.random() * 4}s`);
  });
}

// ── Nodes with neural glow ──
function drawNodes() {
  const nodeGroup = scene.select("#nodes");
  nodeGroup.selectAll("*").remove();

  const nodeEnter = nodeGroup.selectAll(".node")
    .data(nodes, d => d.id)
    .enter()
    .append("g")
    .attr("class", "node")
    .attr("data-id", d => d.id)
    .attr("data-system", d => d.system)
    .style("cursor", "pointer")
    .on("click", (event, d) => selectNode(d))
    .on("mouseenter", (event, d) => {
      d3.select(event.currentTarget).select("circle.main").attr("stroke-width", 2.5).attr("r", d.r + 2.5);
      d3.select(event.currentTarget).select(".node-glow").attr("opacity", 0.35);
      highlightEdges(d.id, true);
    })
    .on("mouseleave", (event, d) => {
      const isSel = selectedId === d.id;
      d3.select(event.currentTarget).select("circle.main").attr("stroke-width", isSel ? 2.5 : 1.6).attr("r", d.r);
      d3.select(event.currentTarget).select(".node-glow").attr("opacity", 0.15);
      highlightEdges(d.id, false);
    });

  nodeEnter.append("circle")
    .attr("class", "node-glow")
    .attr("r", d => d.r + 8)
    .attr("fill", d => SYSTEMS[d.system]?.color || d.color)
    .attr("opacity", 0.15)
    .attr("filter", "url(#glow-strong)")
    .style("animation", (d, i) => `node-breathe ${2.5 + (i % 7) * 0.4}s ease-in-out infinite`)
    .style("animation-delay", `${(i => i * 0.13)(nodes.indexOf(d))}s`);

  nodeEnter.append("circle")
    .attr("class", "main")
    .attr("r", d => d.r)
    .attr("fill", "#0f0a18")
    .attr("stroke", d => SYSTEMS[d.system]?.color || d.color)
    .attr("stroke-width", 1.6);

  nodeEnter.append("circle")
    .attr("class", "core")
    .attr("r", d => Math.max(1.5, d.r * 0.28))
    .attr("fill", d => SYSTEMS[d.system]?.color || d.color)
    .attr("filter", "url(#glow-soft)");

  nodeEnter.append("text")
    .attr("class", "node-label")
    .attr("y", d => d.r + 11)
    .attr("text-anchor", "middle")
    .attr("fill", "#e8d8ff")
    .attr("font-size", 8.5)
    .attr("font-family", "system-ui, ui-sans-serif, sans-serif")
    .attr("font-weight", "600")
    .attr("letter-spacing", 0.4)
    .style("pointer-events", "none")
    .style("text-shadow", "0 0 4px rgba(15,10,24,0.9)")
    .style("opacity", showLabels.checked ? 1 : 0)
    .text(d => d.name.toUpperCase());

  // Bind to current simulation positions
  nodeEnter.attr("transform", d => `translate(${d.x},${d.y})`);
}

function highlightEdges(nodeId, on) {
  scene.selectAll(".edge").filter(function() {
    const s = this.getAttribute("data-source");
    const t = this.getAttribute("data-target");
    return s === nodeId || t === nodeId;
  })
  .attr("stroke-opacity", function() { return on ? 0.9 : (this.getAttribute("stroke-dasharray") ? 0.18 : 0.35); })
  .attr("stroke-width", on ? 2.2 : (this.getAttribute("stroke-dasharray") ? 0.6 : 1.1));
}

function selectNode(d) {
  selectedId = d.id;
  scene.selectAll(".node .main").attr("stroke-width", 1.6).attr("r", n => n.r);
  const sel = scene.selectAll(".node").filter(n => n.id === d.id);
  sel.select(".main").attr("stroke-width", 2.5).attr("r", d.r + 2.5);
  sel.select(".node-glow").classed("firing", true);
  setTimeout(() => sel.select(".node-glow").classed("firing", false), 700);

  const sys = SYSTEMS[d.system] || { label: d.system, color: d.color };
  detailsEl.innerHTML = `
    <p class="details-kicker" style="color:${sys.color}">${esc(sys.label)} · ${esc(d.system)}</p>
    <h2>${esc(d.name)}</h2>
    <p>${esc(d.desc)}</p>
    <p style="color:var(--dim);font-size:10px;letter-spacing:0.1em;text-transform:uppercase;margin-top:12px;">Source files</p>
    <div style="margin-top:4px;">${d.files.map(f => `<span class="badge">${esc(f)}</span>`).join("")}</div>
    <p style="color:var(--dim);font-size:10px;letter-spacing:0.1em;text-transform:uppercase;margin-top:12px;">Network</p>
    <p style="margin-top:2px;font-size:11px;">${d.dependency_count || 0} outgoing · ${d.files.length} files</p>
  `;
}

function clearSelection() {
  selectedId = null;
  scene.selectAll(".node .main").attr("stroke-width", 1.6).attr("r", n => n.r);
  detailsEl.innerHTML = `<p class="details-kicker">Module briefing</p><h2>Select a module</h2><p>Each node is a grouped part of the codebase. Drag, scroll, or click to inspect.</p>`;
}

function buildFilters() {
  systemFilters.innerHTML = "";
  Object.entries(SYSTEMS).forEach(([key, sys]) => {
    const btn = document.createElement("button");
    btn.className = "btn filter-btn active";
    btn.dataset.system = key;
    btn.textContent = sys.label;
    btn.style.borderColor = sys.color;
    btn.style.color = "#fff";
    btn.onclick = () => toggleSystem(key, btn);
    systemFilters.appendChild(btn);
    activeSystems.add(key);
  });
}

function toggleSystem(sys, btn) {
  if (activeSystems.has(sys)) {
    activeSystems.delete(sys);
    btn.classList.remove("active");
    btn.style.opacity = 0.4;
  } else {
    activeSystems.add(sys);
    btn.classList.add("active");
    btn.style.opacity = 1;
  }
  updateVisibility();
}

function updateVisibility() {
  scene.selectAll(".node").each(function(d) {
    const visible = activeSystems.has(d.system);
    d3.select(this).style("opacity", visible ? 1 : 0.08).style("pointer-events", visible ? "all" : "none");
  });
  scene.selectAll(".edge, .edge-spark").each(function(d) {
    const s = nodes.find(n => n.id === (d.source.id || d.source));
    const t = nodes.find(n => n.id === (d.target.id || d.target));
    const visible = s && t && activeSystems.has(s.system) && activeSystems.has(t.system);
    d3.select(this).style("opacity", visible ? 1 : 0.02);
  });
  const visibleCount = nodes.filter(n => activeSystems.has(n.system)).length;
  statsEl.textContent = `${visibleCount} modules visible · ${edges.length} connections`;
}

async function doSearch() {
  const q = searchInput.value.trim().toLowerCase();
  if (!q) { searchHits.innerHTML = ""; searchStats.textContent = "Enter a question to search."; return; }
  let hits = [];
  try {
    const response = await fetch(`/studio/codebase/api/search?q=${encodeURIComponent(q)}&limit=8`);
    if (!response.ok) throw new Error(`Search failed (${response.status})`);
    const payload = await response.json();
    hits = (payload.hits || []).map(hit => ({
      id: hit.path,
      name: hit.path,
      system: hit.path.startsWith("cognition/") ? "brain" : (hit.path.startsWith("sensory/") ? "senses" : (hit.path.startsWith("system/") ? "core" : (hit.path.startsWith("agentic/") ? "tools" : "support"))),
      desc: hit.text || "Indexed source excerpt",
      files: [hit.path],
    }));
  } catch (_error) {
    hits = nodes.filter(n => n.name.toLowerCase().includes(q) || n.desc.toLowerCase().includes(q) || n.id.toLowerCase().includes(q) || n.files.some(f => f.toLowerCase().includes(q)));
  }
  searchStats.textContent = `${hits.length} hit${hits.length !== 1 ? "s" : ""}`;
  searchHits.innerHTML = hits.map(h => `<div class="search-hit" data-id="${esc(h.id)}"><strong style="color:${SYSTEMS[h.system]?.color || '#888'}">${esc(h.name)}</strong><span> · ${esc(h.desc.slice(0, 70))}…</span></div>`).join("");
  searchHits.querySelectorAll(".search-hit").forEach(el => {
    el.addEventListener("click", () => {
      const node = nodes.find(n => n.id === el.dataset.id || n.files.includes(el.dataset.id));
      if (node) selectNode(node);
    });
  });
}

function exportMarkdown() {
  let md = "# Aiko Codebase Atlas\n\n";
  Object.entries(SYSTEMS).forEach(([key, sys]) => {
    const mods = nodes.filter(n => n.system === key);
    if (!mods.length) return;
    md += `## ${sys.label} (${key})\n\n`;
    mods.forEach(m => { md += `### ${m.name}\n${m.desc}\n\n**Files:** ${m.files.join(", ")}\n\n`; });
  });
  const blob = new Blob([md], { type: "text/markdown" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "aiko-codebase-atlas.md";
  a.click();
  setStatus("Exported atlas.md");
  setTimeout(() => setStatus("Neural atlas active"), 2000);
}

const zoom = d3.zoom().scaleExtent([0.4, 5]).on("zoom", event => {
  scene.attr("transform", event.transform);
});
document.getElementById("zoom-in").onclick = () => svg.transition().call(zoom.scaleBy, 1.3);
document.getElementById("zoom-out").onclick = () => svg.transition().call(zoom.scaleBy, 0.75);
document.getElementById("zoom-reset").onclick = () => svg.transition().call(zoom.transform, d3.zoomIdentity);
document.getElementById("print-atlas").onclick = () => window.print();

document.getElementById("export-md").onclick = exportMarkdown;
document.getElementById("refresh").onclick = () => init();
document.getElementById("ingest").onclick = async () => {
  setStatus("Re-indexing…");
  try {
    const response = await fetch("/studio/codebase/api/ingest?force=true");
    if (!response.ok) throw new Error(`Indexing failed (${response.status})`);
    await response.json();
    await init();
  } catch (_error) {
    setStatus("Indexing unavailable");
  }
};
searchBtn.onclick = doSearch;
searchInput.addEventListener("keydown", e => { if (e.key === "Enter") doSearch(); });
showLabels.addEventListener("change", () => {
  scene.selectAll(".node-label").style("opacity", showLabels.checked ? 1 : 0);
});
limitInput.addEventListener("change", init);

svg.on("click", (e) => { if (e.target.tagName === "svg" || e.target.classList.contains("bg")) clearSelection(); });

async function init() {
  const requestGeneration = ++graphRequestGeneration;
  svg.selectAll("*").remove();
  drawBackdrop();

  const limit = parseInt(limitInput.value) || 400;
  let needIngest = false;
  try {
    const response = await fetch(`/studio/codebase/api/graph?limit=${encodeURIComponent(limit)}`);
    if (requestGeneration !== graphRequestGeneration) return;
    if (!response.ok) throw new Error(`Graph failed (${response.status})`);
    const graph = await response.json();
    if (requestGeneration !== graphRequestGeneration) return;
    if (!graph.meta?.exists || !graph.nodes?.length) {
      needIngest = !graph.meta?.exists;
      throw new Error(needIngest ? "Codebase index missing" : "No codebase index");
    }
    nodes = graph.nodes.map(graphNodeToDisplay);
    const ids = new Set(nodes.map(node => node.id));
    edges = (graph.edges || []).filter(edge => {
      const s = edge.source.id || edge.source;
      const t = edge.target.id || edge.target;
      return ids.has(s) && ids.has(t);
    }).map(e => ({ source: e.source.id || e.source, target: e.target.id || e.target, kind: e.kind, weight: e.weight || 0.2 }));
    setStatus(`Atlas ready · ${nodes.length} indexed modules`);
  } catch (_error) {
    if (requestGeneration !== graphRequestGeneration) return;
    if (needIngest) {
      setStatus("Indexing codebase…");
      try {
        const ing = await fetch("/studio/codebase/api/ingest?force=false");
        if (ing.ok) await ing.json();
        if (requestGeneration !== graphRequestGeneration) return;
        setStatus("Re-checking index…");
        const retry = await fetch(`/studio/codebase/api/graph?limit=${encodeURIComponent(limit)}`);
        if (retry.ok) {
          const graph = await retry.json();
          if (graph.meta?.exists && graph.nodes?.length) {
            nodes = graph.nodes.map(graphNodeToDisplay);
            const ids = new Set(nodes.map(n => n.id));
            edges = (graph.edges || []).filter(e => {
              const s = e.source.id || e.source, t = e.target.id || e.target;
              return ids.has(s) && ids.has(t);
            }).map(e => ({ source: e.source.id || e.source, target: e.target.id || e.target, kind: e.kind, weight: e.weight || 0.2 }));
            setStatus(`Atlas ready · ${nodes.length} indexed modules`);
            drawEdges(); drawNodes(); buildFilters(); updateVisibility(); svg.call(zoom); startSimulation();
            return;
          }
        }
      } catch (_) { /* fall through to demo */ }
    }
    if (requestGeneration !== graphRequestGeneration) return;
    nodes = MODULES.slice(0, limit).map((m, i) => ({
      ...m,
      x: W * (0.1 + 0.8 * ((m.id.charCodeAt(0) % 17) / 17)),
      y: H * (0.1 + 0.8 * ((m.id.charCodeAt(m.id.length - 1) % 19) / 19)),
      r: 11,
      color: SYSTEMS[m.system].color,
      dependency_count: 0,
    }));
    edges = LINKS.map(l => ({ source: l.s, target: l.t, kind: "demo", weight: 0.5 }))
      .filter(l => nodes.some(n => n.id === l.source) && nodes.some(n => n.id === l.target));
    setStatus("Demo atlas · index unavailable");
  }

  if (requestGeneration !== graphRequestGeneration) return;
  drawEdges();
  drawNodes();
  buildFilters();
  updateVisibility();
  svg.call(zoom);
  startSimulation();
}

init();
