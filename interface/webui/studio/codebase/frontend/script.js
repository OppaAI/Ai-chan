const svg = d3.select("#canvas");
const W = 560, H = 720;
let nodes = [], edges = [];
let activeSystems = new Set();
let selectedId = null;
let moduleRequestController = null;
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

// ── Color system ──
const SYSTEMS = {
  brain:  { label: "Brain",  color: "#c651a8", glow: "#ff6eb3" },
  senses: { label: "Senses", color: "#51d4c8", glow: "#64ffd2" },
  core:   { label: "Core",   color: "#a888e8", glow: "#d8bcff" },
  tools:  { label: "Arms",   color: "#7298e8", glow: "#8ab4ff" },
  move:   { label: "Legs",   color: "#51bfa5", glow: "#64ffd2" }
};

// ── Module data (Aiko codebase atlas) ──
const MODULES = [
  // Brain · cognition and memory
  { id: "stm",      name: "STM",           system: "brain", x: 260, y: 88,  r: 14, desc: "Short-term memory buffer. Holds recent conversation context and working memory for the current session.", files: ["memory/stm.py", "memory/buffer.py"] },
  { id: "ltm",      name: "LTM",           system: "brain", x: 300, y: 88,  r: 14, desc: "Long-term memory store. Vector-based embedding storage for persistent facts and user preferences.", files: ["memory/ltm.py", "memory/vectors.py", "memory/embed.py"] },
  { id: "itm",      name: "ITM",           system: "brain", x: 280, y: 112, r: 12, desc: "Intermediate-term memory. Summarization layer that compresses STM into durable LTM entries.", files: ["memory/itm.py", "memory/summarize.py"] },
  { id: "kb",       name: "KB",            system: "brain", x: 280, y: 62,  r: 11, desc: "Knowledge base. Structured fact graph and entity resolver for grounded reasoning.", files: ["memory/kb.py", "memory/graph.py"] },
  { id: "reason",   name: "Reasoner",      system: "brain", x: 248, y: 130, r: 10, desc: "Cognitive reasoning engine. Chain-of-thought planner and logic evaluator.", files: ["cognition/reason.py", "cognition/chain.py"] },
  { id: "gate",     name: "Gate",          system: "brain", x: 312, y: 130, r: 10, desc: "Attention gate. Filters incoming stimuli and decides what reaches working memory.", files: ["cognition/gate.py", "cognition/attention.py"] },

  // Senses · web, input, voice
  { id: "vision",   name: "Vision",        system: "senses", x: 232, y: 142, r: 11, desc: "Visual perception. Image parsing, OCR, and scene description.", files: ["senses/vision.py", "senses/ocr.py"] },
  { id: "hearing",  name: "Hearing",       system: "senses", x: 328, y: 142, r: 11, desc: "Audio input pipeline. Speech-to-text and sound event detection.", files: ["senses/audio.py", "senses/stt.py"] },
  { id: "web",      name: "Web",           system: "senses", x: 210, y: 108, r: 12, desc: "Web sense. Live page fetching, search integration, and content extraction.", files: ["senses/web.py", "senses/search.py", "senses/fetch.py"] },
  { id: "touch",    name: "Touch",         system: "senses", x: 350, y: 108, r: 10, desc: "Input touch handler. File uploads, drag-drop, and gesture parsing.", files: ["senses/touch.py", "senses/gesture.py"] },
  { id: "voice",    name: "Voice Out",     system: "senses", x: 280, y: 168, r: 10, desc: "Speech synthesis and voice output formatting.", files: ["senses/tts.py", "senses/voice.py"] },

  // Core · state and configuration
  { id: "state",    name: "State",         system: "core", x: 280, y: 236, r: 16, desc: "Central state manager. Session lifecycle, context switching, and atomic updates.", files: ["core/state.py", "core/session.py"] },
  { id: "config",   name: "Config",        system: "core", x: 248, y: 260, r: 12, desc: "Configuration registry. Environment, secrets, and dynamic parameter store.", files: ["core/config.py", "core/env.py"] },
  { id: "identity", name: "Identity",      system: "core", x: 312, y: 260, r: 12, desc: "Personality and identity module. Tone rules, persona switching, and self-model.", files: ["core/identity.py", "core/persona.py"] },
  { id: "bus",      name: "Event Bus",     system: "core", x: 280, y: 280, r: 11, desc: "Internal message bus. Pub-sub backbone for inter-module communication.", files: ["core/bus.py", "core/events.py"] },

  // Arms · tools and workflows
  { id: "coder",    name: "Coder",         system: "tools", x: 184, y: 220, r: 14, desc: "Code generation and execution. Python sandbox, linting, and diff tooling.", files: ["tools/coder.py", "tools/sandbox.py", "tools/exec.py"] },
  { id: "shell",    name: "Shell",         system: "tools", x: 164, y: 260, r: 12, desc: "System shell interface. Safe command execution with allow-list guards.", files: ["tools/shell.py", "tools/guard.py"] },
  { id: "browser",  name: "Browser",       system: "tools", x: 148, y: 300, r: 12, desc: "Headless browser automation. Page interaction, screenshots, and form filling.", files: ["tools/browser.py", "tools/puppet.py"] },
  { id: "files",    name: "Files",         system: "tools", x: 168, y: 340, r: 11, desc: "File system manager. Read, write, compress, and path resolution.", files: ["tools/files.py", "tools/fs.py"] },
  { id: "calc",     name: "Calc",          system: "tools", x: 196, y: 376, r: 10, desc: "Math and data toolkit. Calculator, plotter, and statistics helper.", files: ["tools/calc.py", "tools/plot.py"] },

  // Legs · orchestration and time
  { id: "sched",    name: "Scheduler",     system: "move", x: 376, y: 220, r: 13, desc: "Task scheduler. Cron-like job runner and delayed execution queue.", files: ["orchestra/scheduler.py", "orchestra/cron.py"] },
  { id: "pipeline", name: "Pipeline",      system: "move", x: 396, y: 260, r: 13, desc: "Workflow pipeline. DAG-based step runner with retry and rollback.", files: ["orchestra/pipeline.py", "orchestra/dag.py"] },
  { id: "agent",    name: "Agent Loop",    system: "move", x: 412, y: 300, r: 12, desc: "Main agent orchestration loop. Decision cycle and action dispatch.", files: ["orchestra/agent.py", "orchestra/loop.py"] },
  { id: "time",     name: "Clock",         system: "move", x: 392, y: 340, r: 10, desc: "Time keeper. NTP sync, timers, and temporal reasoning.", files: ["orchestra/clock.py", "orchestra/time.py"] },
  { id: "health",   name: "Health",        system: "move", x: 364, y: 376, r: 10, desc: "Health monitor. Self-diagnostics, heartbeat, and graceful degradation.", files: ["orchestra/health.py", "orchestra/beat.py"] }
];

const LINKS = [
  { source: "stm", target: "reason" }, { source: "stm", target: "gate" },
  { source: "ltm", target: "itm" }, { source: "itm", target: "stm" },
  { source: "kb", target: "reason" }, { source: "gate", target: "state" },
  { source: "reason", target: "state" }, { source: "vision", target: "gate" },
  { source: "hearing", target: "gate" }, { source: "web", target: "gate" },
  { source: "touch", target: "state" }, { source: "voice", target: "state" },
  { source: "state", target: "bus" }, { source: "config", target: "state" },
  { source: "identity", target: "state" }, { source: "bus", target: "agent" },
  { source: "agent", target: "coder" }, { source: "agent", target: "shell" },
  { source: "agent", target: "browser" }, { source: "agent", target: "files" },
  { source: "agent", target: "calc" }, { source: "sched", target: "agent" },
  { source: "pipeline", target: "agent" }, { source: "health", target: "state" },
  { source: "coder", target: "calc" }, { source: "browser", target: "web" }
];

// ── Draw robot figure ──
function drawFigure() {
  const defs = svg.append("defs");

  // Body gradient — cute pink to deep purple
  const bodyGrad = defs.append("linearGradient").attr("id", "body-grad")
    .attr("x1", "0%").attr("y1", "0%").attr("x2", "100%").attr("y2", "100%");
  bodyGrad.append("stop").attr("offset", "0%").attr("stop-color", "#ff6eb3");
  bodyGrad.append("stop").attr("offset", "50%").attr("stop-color", "#7a3a8a");
  bodyGrad.append("stop").attr("offset", "100%").attr("stop-color", "#1e0f33");

  // Glow filters
  const glowPink = defs.append("filter").attr("id", "glow-pink");
  glowPink.append("feGaussianBlur").attr("stdDeviation", "3").attr("result", "b");
  glowPink.append("feMerge").selectAll("feMergeNode").data(["b","SourceGraphic"]).enter().append("feMergeNode").attr("in", d=>d);

  const glowCyan = defs.append("filter").attr("id", "glow-cyan");
  glowCyan.append("feGaussianBlur").attr("stdDeviation", "2.5").attr("result", "b");
  glowCyan.append("feMerge").selectAll("feMergeNode").data(["b","SourceGraphic"]).enter().append("feMergeNode").attr("in", d=>d);

  const glowCore = defs.append("filter").attr("id", "glow-core");
  glowCore.append("feGaussianBlur").attr("stdDeviation", "5").attr("result", "b");
  glowCore.append("feMerge").selectAll("feMergeNode").data(["b","SourceGraphic"]).enter().append("feMergeNode").attr("in", d=>d);

  // Hex pattern for core
  const pattern = defs.append("pattern").attr("id", "hex").attr("width", 12).attr("height", 12).attr("patternUnits", "userSpaceOnUse");
  pattern.append("path").attr("d", "M6 0l5.2 3v6L6 12 .8 9V3z").attr("fill", "none").attr("stroke", "rgba(168,136,232,0.15)").attr("stroke-width", 0.5);

  const g = svg.append("g").attr("id", "figure");

  // ── Cute high-tech robot body ──
  // Soft rounded head
  g.append("ellipse").attr("cx", 280).attr("cy", 95).attr("rx", 62).attr("ry", 56)
    .attr("fill", "url(#body-grad)").attr("stroke", "#ff8ec8").attr("stroke-width", 2);

  // Head antenna nubs (ears)
  g.append("circle").attr("cx", 212).attr("cy", 95).attr("r", 10)
    .attr("fill", "#171124").attr("stroke", "#ff8ec8").attr("stroke-width", 2);
  g.append("circle").attr("cx", 348).attr("cy", 95).attr("r", 10)
    .attr("fill", "#171124").attr("stroke", "#ff8ec8").attr("stroke-width", 2);
  // Antenna tips
  g.append("circle").attr("cx", 212).attr("cy", 95).attr("r", 3).attr("fill", "#51d4c8").attr("filter", "url(#glow-cyan)");
  g.append("circle").attr("cx", 348).attr("cy", 95).attr("r", 3).attr("fill", "#51d4c8").attr("filter", "url(#glow-cyan)");

  // Cute big eyes with highlights
  const eyeGroup = g.append("g");
  // Left eye
  eyeGroup.append("ellipse").attr("cx", 258).attr("cy", 100).attr("rx", 14).attr("ry", 16)
    .attr("fill", "rgba(100,255,210,0.15)").attr("stroke", "#64ffd2").attr("stroke-width", 2);
  eyeGroup.append("ellipse").attr("cx", 258).attr("cy", 100).attr("rx", 8).attr("ry", 10)
    .attr("fill", "#0a1f1c").attr("stroke", "none");
  eyeGroup.append("circle").attr("cx", 262).attr("cy", 96).attr("r", 3).attr("fill", "#fff").attr("opacity", 0.9);
  // Right eye
  eyeGroup.append("ellipse").attr("cx", 302).attr("cy", 100).attr("rx", 14).attr("ry", 16)
    .attr("fill", "rgba(100,255,210,0.15)").attr("stroke", "#64ffd2").attr("stroke-width", 2);
  eyeGroup.append("ellipse").attr("cx", 302).attr("cy", 100).attr("rx", 8).attr("ry", 10)
    .attr("fill", "#0a1f1c").attr("stroke", "none");
  eyeGroup.append("circle").attr("cx", 306).attr("cy", 96).attr("r", 3).attr("fill", "#fff").attr("opacity", 0.9);

  // Blush spots (cute)
  g.append("ellipse").attr("cx", 242).attr("cy", 118).attr("rx", 6).attr("ry", 3)
    .attr("fill", "rgba(255,110,179,0.35)").attr("filter", "url(#glow-pink)");
  g.append("ellipse").attr("cx", 318).attr("cy", 118).attr("rx", 6).attr("ry", 3)
    .attr("fill", "rgba(255,110,179,0.35)").attr("filter", "url(#glow-pink)");

  // Tiny cute smile
  g.append("path").attr("d", "M272 128 Q280 134 288 128").attr("fill", "none")
    .attr("stroke", "#ff8ec8").attr("stroke-width", 2).attr("stroke-linecap", "round");

  // Neck
  g.append("rect").attr("x", 270).attr("y", 148).attr("width", 20).attr("height", 12).attr("rx", 4)
    .attr("fill", "#171124").attr("stroke", "#ff8ec8").attr("stroke-width", 1.5);

  // Torso — rounded cute body
  g.append("path").attr("d",
    "M230 164 Q280 156 330 164 L340 180 Q350 220 345 280 L335 340 Q330 360 310 365 L280 368 L250 365 Q230 360 225 340 L215 280 Q210 220 220 180 Z")
    .attr("fill", "url(#body-grad)").attr("stroke", "#ff8ec8").attr("stroke-width", 2);

  // Chest hex core (high-tech holographic)
  g.append("polygon").attr("points", "280,220 296,229 296,247 280,256 264,247 264,229")
    .attr("fill", "url(#hex)").attr("stroke", "#a888e8").attr("stroke-width", 1.5)
    .attr("filter", "url(#glow-core)");
  g.append("circle").attr("cx", 280).attr("cy", 238).attr("r", 5)
    .attr("fill", "#a888e8").attr("filter", "url(#glow-core)");

  // Circuit traces on torso
  const traces = g.append("g").attr("stroke", "rgba(168,136,232,0.3)").attr("stroke-width", 1).attr("fill", "none");
  traces.append("path").attr("d", "M250 180 L250 210 L264 220");
  traces.append("path").attr("d", "M310 180 L310 210 L296 220");
  traces.append("path").attr("d", "M280 256 L280 280");
  traces.append("path").attr("d", "M240 260 L260 260 L270 280");
  traces.append("path").attr("d", "M320 260 L300 260 L290 280");
  // Circuit dots
  [[250,180],[310,180],[250,210],[310,210],[264,220],[296,220],[280,280],[240,260],[320,260],[270,280],[290,280]].forEach(([cx,cy])=>{
    g.append("circle").attr("cx",cx).attr("cy",cy).attr("r",2).attr("fill","#a888e8").attr("opacity",0.7);
  });

  // Shoulder pads (rounded)
  g.append("ellipse").attr("cx", 210).attr("cy", 185).attr("rx", 18).attr("ry", 12)
    .attr("fill", "#171124").attr("stroke", "#ff8ec8").attr("stroke-width", 1.5);
  g.append("ellipse").attr("cx", 350).attr("cy", 185).attr("rx", 18).attr("ry", 12)
    .attr("fill", "#171124").attr("stroke", "#ff8ec8").attr("stroke-width", 1.5);

  // Left arm (tools side)
  g.append("path").attr("d", "M195 190 Q170 210 160 240 Q150 280 155 320 Q158 350 170 380")
    .attr("fill", "none").attr("stroke", "#ff8ec8").attr("stroke-width", 5).attr("stroke-linecap", "round");
  g.append("path").attr("d", "M195 190 Q170 210 160 240 Q150 280 155 320 Q158 350 170 380")
    .attr("fill", "none").attr("stroke", "#171124").attr("stroke-width", 2).attr("stroke-linecap", "round");
  // Arm joint lights
  [[165,240],[158,300],[164,350]].forEach(([cx,cy],i)=>{
    g.append("circle").attr("cx",cx).attr("cy",cy).attr("r",4)
      .attr("fill", i%2?"#51d4c8":"#ff6eb3").attr("filter","url(#glow-cyan)");
  });

  // Right arm (orchestra side)
  g.append("path").attr("d", "M365 190 Q390 210 400 240 Q410 280 405 320 Q402 350 390 380")
    .attr("fill", "none").attr("stroke", "#ff8ec8").attr("stroke-width", 5).attr("stroke-linecap", "round");
  g.append("path").attr("d", "M365 190 Q390 210 400 240 Q410 280 405 320 Q402 350 390 380")
    .attr("fill", "none").attr("stroke", "#171124").attr("stroke-width", 2).attr("stroke-linecap", "round");
  [[395,240],[402,300],[396,350]].forEach(([cx,cy],i)=>{
    g.append("circle").attr("cx",cx).attr("cy",cy).attr("r",4)
      .attr("fill", i%2?"#51d4c8":"#ff6eb3").attr("filter","url(#glow-cyan)");
  });

  // Cute rounded legs
  g.append("path").attr("d", "M250 365 Q240 420 242 480 Q244 520 248 560 L252 620 Q254 640 260 640 L275 640 Q280 640 280 620 L280 500")
    .attr("fill", "url(#body-grad)").attr("stroke", "#ff8ec8").attr("stroke-width", 2);
  g.append("path").attr("d", "M310 365 Q320 420 318 480 Q316 520 312 560 L308 620 Q306 640 300 640 L285 640 Q280 640 280 620 L280 500")
    .attr("fill", "url(#body-grad)").attr("stroke", "#ff8ec8").attr("stroke-width", 2);

  // Leg joint lights
  [[248,420],[246,500],[250,580],[312,420],[314,500],[310,580]].forEach(([cx,cy],i)=>{
    g.append("circle").attr("cx",cx).attr("cy",cy).attr("r",3)
      .attr("fill", i%2?"#51d4c8":"#ff6eb3").attr("opacity",0.8);
  });

  // Foot bases
  g.append("ellipse").attr("cx", 268).attr("cy", 645).attr("rx", 20).attr("ry", 8)
    .attr("fill", "#171124").attr("stroke", "#ff8ec8").attr("stroke-width", 1.5);
  g.append("ellipse").attr("cx", 292).attr("cy", 645).attr("rx", 20).attr("ry", 8)
    .attr("fill", "#171124").attr("stroke", "#ff8ec8").attr("stroke-width", 1.5);

  // Brain glow inside head (subtle)
  g.append("ellipse").attr("cx", 280).attr("cy", 85).attr("rx", 36).attr("ry", 28)
    .attr("fill", "rgba(198,81,168,0.08)").attr("filter", "url(#glow-pink)");

  // High-tech panel lines on head
  g.append("path").attr("d", "M240 75 Q280 65 320 75").attr("fill", "none")
    .attr("stroke", "rgba(255,142,200,0.4)").attr("stroke-width", 1);
  g.append("path").attr("d", "M235 95 Q280 105 325 95").attr("fill", "none")
    .attr("stroke", "rgba(255,142,200,0.4)").attr("stroke-width", 1);

  // Label
  g.append("text").attr("x", 280).attr("y", 708).attr("text-anchor", "middle")
    .attr("fill", "#d8bcff").attr("font-size", 9).attr("letter-spacing", 3)
    .attr("font-family", "system-ui, ui-sans-serif, sans-serif").attr("font-weight", "600")
    .text("AIKO · CODEBASE ATLAS");
}

// ── Draw edges ──
function drawEdges() {
  const edgeGroup = svg.append("g").attr("id", "edges");
  edges.forEach(l => {
    const s = nodes.find(n => n.id === l.source);
    const t = nodes.find(n => n.id === l.target);
    if (!s || !t) return;
    edgeGroup.append("line")
      .attr("class", "edge")
      .attr("data-source", l.source)
      .attr("data-target", l.target)
      .attr("x1", s.x).attr("y1", s.y)
      .attr("x2", t.x).attr("y2", t.y)
      .attr("stroke", SYSTEMS[s.system]?.color || "#4a3a6a")
      .attr("stroke-width", 1.2)
      .attr("stroke-opacity", 0.35)
      .attr("stroke-dasharray", "3,3");
  });
}

// ── Draw nodes ──
function drawNodes() {
  const nodeGroup = svg.append("g").attr("id", "nodes");

  const nodeEnter = nodeGroup.selectAll(".node")
    .data(nodes)
    .enter()
    .append("g")
    .attr("class", "node")
    .attr("data-id", d => d.id)
    .attr("data-system", d => d.system)
    .attr("transform", d => `translate(${d.x},${d.y})`)
    .style("cursor", "pointer")
    .on("click", (event, d) => selectNode(d))
    .on("mouseenter", (event, d) => {
      d3.select(event.currentTarget).select("circle").attr("stroke-width", 3).attr("r", d.r + 2);
      highlightEdges(d.id, true);
    })
    .on("mouseleave", (event, d) => {
      const isSel = selectedId === d.id;
      d3.select(event.currentTarget).select("circle").attr("stroke-width", isSel ? 3 : 1.5).attr("r", d.r);
      highlightEdges(d.id, false);
    });

  // Outer glow ring
  nodeEnter.append("circle")
    .attr("r", d => d.r + 4)
    .attr("fill", d => SYSTEMS[d.system].color)
    .attr("opacity", 0.12)
    .attr("filter", "url(#glow-pink)");

  // Main circle
  nodeEnter.append("circle")
    .attr("r", d => d.r)
    .attr("fill", "#0f0a18")
    .attr("stroke", d => SYSTEMS[d.system].color)
    .attr("stroke-width", 1.5);

  // Inner dot
  nodeEnter.append("circle")
    .attr("r", d => Math.max(2, d.r * 0.25))
    .attr("fill", d => SYSTEMS[d.system].color);

  // Labels
  nodeEnter.append("text")
    .attr("class", "node-label")
    .attr("y", d => d.r + 11)
    .attr("text-anchor", "middle")
    .attr("fill", "#d8bcff")
    .attr("font-size", 8)
    .attr("font-family", "system-ui, ui-sans-serif, sans-serif")
    .attr("font-weight", "600")
    .attr("letter-spacing", 0.5)
    .style("pointer-events", "none")
    .style("opacity", showLabels.checked ? 1 : 0)
    .text(d => d.name.toUpperCase());
}

function highlightEdges(nodeId, on) {
  d3.selectAll(".edge").filter(function() {
    const s = this.getAttribute("data-source");
    const t = this.getAttribute("data-target");
    return s === nodeId || t === nodeId;
  })
  .attr("stroke-opacity", on ? 0.85 : 0.35)
  .attr("stroke-width", on ? 2 : 1.2);
}

// ── Selection ──
function selectNode(d) {
  selectedId = d.id;
  // Reset all
  d3.selectAll(".node circle:nth-child(2)").attr("stroke-width", 1.5).attr("r", n => n.r);
  // Highlight selected
  const sel = d3.selectAll(".node").filter(n => n.id === d.id);
  sel.select("circle:nth-child(2)").attr("stroke-width", 3).attr("r", d.r + 2);

  // Details panel
  const sys = SYSTEMS[d.system];
  detailsEl.innerHTML = `
    <p class="details-kicker" style="color:${sys.color}">${esc(sys.label)} · ${esc(d.system)}</p>
    <h2>${esc(d.name)}</h2>
    <p>${esc(d.desc)}</p>
    <p style="color:var(--dim);font-size:10px;letter-spacing:0.1em;text-transform:uppercase;margin-top:12px;">Source files</p>
    <div style="margin-top:4px;">
      ${d.files.map(f => `<span class="badge">${esc(f)}</span>`).join("")}
    </div>
    <p style="margin-top:14px;">
      <span class="source-link" style="cursor:pointer" onclick="alert('Open ${esc(d.id)} module')">Open module →</span>
    </p>
  `;
}

function clearSelection() {
  selectedId = null;
  d3.selectAll(".node circle:nth-child(2)").attr("stroke-width", 1.5).attr("r", d => d.r);
  detailsEl.innerHTML = `
    <p class="details-kicker">Module briefing</p>
    <h2>Select a module</h2>
    <p>Each marker represents a grouped part of the codebase. Select one to see its responsibilities, functions, and files in plain language.</p>
  `;
}

// ── Filters ──
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
  d3.selectAll(".node").each(function(d) {
    const visible = activeSystems.has(d.system);
    d3.select(this).style("opacity", visible ? 1 : 0.08).style("pointer-events", visible ? "all" : "none");
  });
  d3.selectAll(".edge").each(function(d) {
    const s = nodes.find(n => n.id === d.source);
    const t = nodes.find(n => n.id === d.target);
    const visible = s && t && activeSystems.has(s.system) && activeSystems.has(t.system);
    d3.select(this).style("opacity", visible ? 1 : 0.03);
  });
  const visibleCount = nodes.filter(n => activeSystems.has(n.system)).length;
  statsEl.textContent = `${visibleCount} modules visible · ${edges.length} connections`;
}

// ── Search ──
function doSearch() {
  const q = searchInput.value.trim().toLowerCase();
  if (!q) {
    searchHits.innerHTML = "";
    searchStats.textContent = "Enter a question to search.";
    return;
  }
  const hits = nodes.filter(n =>
    n.name.toLowerCase().includes(q) ||
    n.desc.toLowerCase().includes(q) ||
    n.id.toLowerCase().includes(q) ||
    n.files.some(f => f.toLowerCase().includes(q))
  );
  searchStats.textContent = `${hits.length} hit${hits.length !== 1 ? "s" : ""}`;
  searchHits.innerHTML = hits.map(h => `
    <div class="search-hit" data-id="${esc(h.id)}">
      <strong style="color:${SYSTEMS[h.system].color}">${esc(h.name)}</strong>
      <span> · ${esc(h.desc.slice(0, 70))}…</span>
    </div>
  `).join("");
  searchHits.querySelectorAll(".search-hit").forEach(el => {
    el.addEventListener("click", () => {
      const id = el.dataset.id;
      const node = nodes.find(n => n.id === id);
      if (node) {
        selectNode(node);
        // Pan to node
        const scale = 1.8;
        const t = d3.zoomIdentity.translate(W/2 - node.x*scale, H/2 - node.y*scale).scale(scale);
        svg.transition().duration(600).call(zoom.transform, t);
      }
    });
  });
}

// ── Export ──
function exportMarkdown() {
  let md = "# Aiko Codebase Atlas\n\n";
  Object.entries(SYSTEMS).forEach(([key, sys]) => {
    const mods = nodes.filter(n => n.system === key);
    if (!mods.length) return;
    md += `## ${sys.label} (${key})\n\n`;
    mods.forEach(m => {
      md += `### ${m.name}\n${m.desc}\n\n**Files:** ${m.files.join(", ")}\n\n`;
    });
  });
  const blob = new Blob([md], { type: "text/markdown" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "aiko-codebase-atlas.md";
  a.click();
  setStatus("Exported atlas.md");
  setTimeout(() => setStatus(""), 2000);
}

// ── Init ──
function init() {
  drawFigure();

  // Build node/edge arrays from data
  const limit = parseInt(limitInput.value) || 400;
  nodes = MODULES.slice(0, limit);
  edges = LINKS.map(l => ({ source: l.source, target: l.target })).filter(l =>
    nodes.some(n => n.id === l.source) && nodes.some(n => n.id === l.target)
  );

  drawEdges();
  drawNodes();
  buildFilters();
  updateVisibility();

  // Zoom
  svg.call(zoom);
}

// ── Zoom controls ──
const zoom = d3.zoom().scaleExtent([0.5, 4]).on("zoom", event => {
  svg.selectAll("#figure, #edges, #nodes").attr("transform", event.transform);
});
document.getElementById("zoom-in").onclick = () => svg.transition().call(zoom.scaleBy, 1.3);
document.getElementById("zoom-out").onclick = () => svg.transition().call(zoom.scaleBy, 0.75);
document.getElementById("zoom-reset").onclick = () => svg.transition().call(zoom.transform, d3.zoomIdentity);
document.getElementById("print-atlas").onclick = () => window.print();

// Events
document.getElementById("export-md").onclick = exportMarkdown;
document.getElementById("refresh").onclick = () => { setStatus("Refreshed"); init(); };
document.getElementById("ingest").onclick = () => { setStatus("Re-indexing…"); setTimeout(() => { setStatus("Indexed 24 modules"); }, 1200); };
searchBtn.onclick = doSearch;
searchInput.addEventListener("keydown", e => { if (e.key === "Enter") doSearch(); });
showLabels.addEventListener("change", () => {
  d3.selectAll(".node-label").style("opacity", showLabels.checked ? 1 : 0);
});
limitInput.addEventListener("change", init);

// Canvas click to clear
svg.on("click", (e) => {
  if (e.target.tagName === "svg") clearSelection();
});

init();
setStatus("Atlas loaded");
