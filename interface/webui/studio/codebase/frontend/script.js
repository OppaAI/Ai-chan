const svg = d3.select("#canvas");
const W = 560, H = 720;
let nodes = [], edges = [];
let activeSystems = new Set();
let selectedId = null;
let moduleRequestController = null;
const statusEl = document.getElementById("status");
const statsEl = document.getElementById("stats");
const detailsEl = document.getElementById("details");

function setStatus(text) { statusEl.textContent = text; }
function esc(text) { const node = document.createElement("span"); node.textContent = text || ""; return node.innerHTML; }

function drawFigure() {
  const defs = svg.append("defs");
  const bodyGradient = defs.append("linearGradient").attr("id", "body-gradient").attr("x1", "0%").attr("x2", "100%");
  bodyGradient.append("stop").attr("offset", "0%").attr("stop-color", "#2b1a37");
  bodyGradient.append("stop").attr("offset", "52%").attr("stop-color", "#171124");
  bodyGradient.append("stop").attr("offset", "100%").attr("stop-color", "#21192e");
  const glow = defs.append("filter").attr("id", "glow");
  glow.append("feGaussianBlur").attr("stdDeviation", 2.2).attr("result", "blur");
  glow.append("feMerge").selectAll("feMergeNode").data(["blur", "SourceGraphic"]).enter().append("feMergeNode").attr("in", d => d);
  const g = svg.append("g").attr("id", "figure");
  const outline = { fill: "url(#body-gradient)", stroke: "#8a6ca9", "stroke-width": 1.3 };
  // The left half is a soft anatomy silhouette; the right half is a panelled machine diagram.
  g.append("path").attr("d", "M280 18C235 18 209 48 212 94c2 31 17 52 38 65l-2 25-42 18c-25 11-39 34-48 64l-43 126 31 12 46-107 17 28-24 170-22 176h69l30-159 18 159h69l-22-176-24-170 17-28 46 107 31-12-43-126c-9-30-23-53-48-64l-42-18-2-25c21-13 36-34 38-65 3-46-23-76-68-76Z").attr("fill", outline.fill).attr("stroke", outline.stroke).attr("stroke-width", outline["stroke-width"]);
  g.append("path").attr("d", "M280 18V671 M212 94h136 M250 159h60 M206 202c18 17 44 25 74 25s56-8 74-25 M209 325c24 18 47 26 71 26s47-8 71-26 M230 495h100").attr("fill", "none").attr("stroke", "#5f497a").attr("stroke-width", 1);
  // Face, lenses, and neck plating.
  [[252,82,23,13],[308,82,23,13]].forEach(([cx,cy,rx,ry]) => g.append("ellipse").attr("cx",cx).attr("cy",cy).attr("rx",rx).attr("ry",ry).attr("fill","rgba(81,212,200,.08)").attr("stroke","#51d4c8").attr("stroke-width",1.2));
  g.append("path").attr("d","M268 124Q280 132 292 124 M250 160h60l-5 25h-50Z").attr("fill","none").attr("stroke","#a888e8").attr("stroke-width",1.1);
  // Transparent organ layer keeps the anatomy readable behind modules.
  g.append("path").attr("d","M244 244c-18 12-17 48 6 59 13 7 23-2 30-14 7 12 17 21 30 14 23-11 24-47 6-59-14-9-27 1-36 13-9-12-22-22-36-13Z").attr("fill","rgba(198,81,168,.13)").attr("stroke","#c651a8").attr("stroke-width",1.1).attr("filter","url(#glow)");
  g.append("path").attr("d","M264 337c-24 18-20 48-8 67l24 61 24-61c12-19 16-49-8-67Z").attr("fill","rgba(168,136,232,.1)").attr("stroke","#a888e8").attr("stroke-width",1);
  // Mechanical right-side panels, conduits, and joints reference the supplied android cutaway without copying it.
  g.append("path").attr("d","M280 31h42l14 28v66l-20 28h-36 M280 184h58l19 27-12 95-27 19h-38 M280 350h57l18 28-18 99-25 22h-32").attr("fill","none").attr("stroke","#d8bcff").attr("stroke-width",1.15);
  [62,74,107,194,211,226,247,265,282,369,390,414,443].forEach((y, index) => g.append("line").attr("x1",290).attr("x2",index < 3 ? 325 : 339).attr("y1",y).attr("y2",y).attr("stroke",index % 2 ? "#51d4c8" : "#a888e8").attr("stroke-width",1));
  [[338,240,13],[338,404,13],[408,338,12],[152,338,12],[218,514,10],[342,514,10]].forEach(([cx,cy,r]) => g.append("circle").attr("cx",cx).attr("cy",cy).attr("r",r).attr("fill","#171124").attr("stroke","#a888e8").attr("stroke-width",1));
  g.append("text").attr("x",280).attr("y",708).attr("text-anchor","middle").attr("fill","#6e5a8d").attr("font-size",8).attr("letter-spacing",3).text("CODEBASE ANATOMY MAP");
}

drawFigure();

const zoom = d3.zoom().scaleExtent([.65, 3]).on("zoom", event => svg.selectAll("#figure,#edges,#nodes").attr("transform", event.transform));
svg.call(zoom);
document.getElementById("zoom-in").onclick = () => svg.transition().call(zoom.scaleBy, 1.25);
document.getElementById("zoom-out").onclick = () => svg.transition().call(zoom.scaleBy, .8);
document.getElementById("zoom-reset").onclick = () => svg.transition().call(zoom.transform, d3.zoomIdentity);
document.getElementById("print-atlas").onclick = () => window.print();

function buildFilters() {
  const systems = [...new Set(nodes.map(node => node.body_part))].sort();
  activeSystems = new Set(systems);
  const host = document.getElementById("system-filters");
  host.innerHTML = systems.map(system => `<button class="btn filter-btn active" data-system="${esc(system)}">${esc(system)}</button>`).join("");
  host.querySelectorAll("button").forEach(button => button.onclick = () => {
    const system = button.dataset.system;
    activeSystems.has(system) ? activeSystems.delete(system) : activeSystems.add(system);
    button.classList.toggle("active", activeSystems.has(system)); render();
  });
}

async function loadGraph() {
  const limit = document.getElementById("limit").value || 400;
  setStatus("Loading atlas…");
  try {
    const response = await fetch(`/studio/codebase/api/graph?limit=${encodeURIComponent(limit)}`);
    const graph = await response.json(); nodes = graph.nodes || []; edges = graph.edges || [];
    statsEl.textContent = `${nodes.length} modules from ${limit} sampled files`;
    buildFilters(); setStatus(graph.meta?.exists ? "Index ready" : "Index missing — re-index to begin"); render();
  } catch (error) { setStatus(`Could not load atlas: ${error.message}`); }
}

function render() {
  svg.selectAll("#edges,#nodes,#callouts").remove();
  const visibleNodes = nodes.filter(node => activeSystems.has(node.body_part));
  const byId = new Map(visibleNodes.map(node => [node.id, node]));
  const edgeGroup = svg.append("g").attr("id","edges");
  edges.forEach(edge => { const source=byId.get(edge.source), target=byId.get(edge.target); if (!source || !target) return; const dependency=edge.kind === "dependency"; edgeGroup.append("line").attr("class","edge").attr("data-source",edge.source).attr("data-target",edge.target).attr("x1",source.x*W).attr("y1",source.y*H).attr("x2",target.x*W).attr("y2",target.y*H).attr("stroke",dependency ? "#a888e8" : "#4a3a6a").attr("stroke-width",dependency ? 1.2 : .7).attr("opacity",dependency ? .62 : .3); });
  const group = svg.append("g").attr("id","nodes");
  const selection = group.selectAll("g.node").data(visibleNodes).enter().append("g").attr("class","node").attr("data-id",d=>d.id).attr("transform", d => `translate(${d.x*W},${d.y*H})`).style("cursor","pointer").on("mouseenter", (_, d) => highlightRelations(d.id)).on("mouseleave", () => highlightRelations(selectedId)).on("click", (event, d) => { selectedId = d.id; highlightRelations(selectedId); showModule(event, d); });
  selection.append("circle").attr("r",8).attr("fill",d=>d.color).attr("opacity",.12).attr("stroke",d => d.coverage === null ? "#4a3a6a" : d.coverage >= 80 ? "#51bfa5" : d.coverage >= 50 ? "#e8c84a" : "#e8516a").attr("stroke-width",1.1);
  selection.append("circle").attr("r",d => 3.2 + Math.min(2, d.dependency_count / 4)).attr("fill",d=>d.color).attr("stroke","#d8bcff").attr("stroke-width",d => d.change_count > 8 ? 1.4 : .7);
  if (document.getElementById("show-labels").checked) selection.append("text").attr("x",7).attr("y",-5).attr("fill","#d8bcff").attr("font-size",7.2).attr("letter-spacing",.4).text(d => d.module.split("/").slice(-1)[0].slice(0,22));
}

function highlightRelations(moduleId) {
  const hasFocus = Boolean(moduleId);
  svg.selectAll(".node").attr("opacity", d => !hasFocus || d.id === moduleId ? 1 : .25);
  svg.selectAll(".edge").attr("opacity", function () {
    if (!hasFocus) return d3.select(this).attr("stroke-width") === "1.2" ? .62 : .3;
    return this.dataset.source === moduleId || this.dataset.target === moduleId ? 1 : .08;
  }).attr("stroke-width", function () {
    return hasFocus && (this.dataset.source === moduleId || this.dataset.target === moduleId) ? 2.4 : (d3.select(this).attr("stroke") === "#a888e8" ? 1.2 : .7);
  });
}

async function showModule(event, node) {
  moduleRequestController?.abort();
  const controller = new AbortController();
  moduleRequestController = controller;
  svg.selectAll("g.node circle:last-child").attr("r",3.2); d3.select(event.currentTarget).select("circle:last-child").attr("r",5);
  detailsEl.innerHTML = `<p class="details-kicker">Reading module</p><h2>${esc(node.module)}</h2><p>Building a natural-language brief from the indexed source…</p>`;
  try {
    const response = await fetch(`/studio/codebase/api/module?module=${encodeURIComponent(node.module)}`, { signal: controller.signal }); const detail = await response.json();
    if (moduleRequestController !== controller) return;
    if (detail.error) throw new Error(detail.error);
    const functions = detail.functions.length ? `<ul>${detail.functions.map(name => `<li><code>${esc(name)}()</code></li>`).join("")}</ul>` : "<p>No callable symbols were extracted from the indexed excerpts.</p>";
    const files = detail.files.slice(0, 7).map(file => `<li><code>${esc(file)}</code></li>`).join("");
    const coverage = detail.coverage === null ? "unavailable" : `${detail.coverage}%`;
    const pills = Object.entries(detail.metrics).map(([label, value]) => `<span class="badge">${esc(label.replace("_", " "))}: ${esc(String(value))}</span>`).join("") + `<span class="badge">changes: ${detail.change_count}</span><span class="badge">coverage: ${coverage}</span>`;
    const links = detail.source_links.slice(0, 7).map(link => `<li><code>${esc(link.path)}</code> ${link.github ? `<a class="source-link" href="${esc(link.github)}" target="_blank" rel="noopener">GitHub</a>` : ""} <a class="source-link" href="${esc(link.vscode)}">IDE</a></li>`).join("");
    const relationList = values => values.length ? `<ul>${values.map(value => `<li><code>${esc(value)}</code></li>`).join("")}</ul>` : "<p>None found in indexed imports.</p>";
    detailsEl.innerHTML = `<p class="details-kicker">${esc(detail.body_part)} system · ${detail.files.length} files</p><h2>${esc(detail.module)}</h2><p>${esc(detail.summary)}</p><div>${pills}</div>${detail.docstrings.length ? `<p class="details-kicker">Source intent</p><p>${esc(detail.docstrings.join(" · "))}</p>` : ""}<p class="details-kicker">Calls</p>${relationList(detail.dependencies)}<p class="details-kicker">Called by</p>${relationList(detail.dependents)}<p class="details-kicker">Functions and classes</p>${functions}<p class="details-kicker">Open source</p><ul>${links}</ul>${detail.excerpt ? `<p class="details-kicker">Indexed context</p><p>${esc(detail.excerpt)}…</p>` : ""}`;
  } catch (error) {
    if (moduleRequestController !== controller || error.name === "AbortError") return;
    detailsEl.innerHTML = `<p class="details-kicker">Module briefing</p><h2>${esc(node.module)}</h2><p>${esc(error.message)}</p>`;
  }
}

document.getElementById("refresh").onclick = loadGraph;
document.getElementById("show-labels").onchange = render;
document.getElementById("ingest").onclick = async () => { setStatus("Indexing codebase…"); try { const response=await fetch("/studio/codebase/api/ingest?force=false"); const result=await response.json(); setStatus(result.ok ? `Indexed ${result.docs_added} updated files` : result.error || "Indexing failed"); loadGraph(); } catch (error) { setStatus(error.message); } };
document.getElementById("search-btn").onclick = async () => { const query=document.getElementById("search-q").value.trim(); if (!query) return; setStatus("Searching…"); const response=await fetch(`/studio/codebase/api/search?q=${encodeURIComponent(query)}&limit=8`); const result=await response.json(); document.getElementById("search-stats").textContent=`${result.meta.count || 0} matching excerpts`; document.getElementById("search-hits").innerHTML=result.hits.map(hit => `<div class="search-hit" title="${esc(hit.path)}"><b>${esc(hit.path)}</b><br><span>${esc((hit.text || "").slice(0,100))}</span></div>`).join("") || "<div class=\"stat\">No matching code was found.</div>"; setStatus("Search complete"); };
document.getElementById("export-md").onclick = () => { const limit = document.getElementById("limit").value || 400; window.open(`/studio/codebase/api/export/markdown?limit=${encodeURIComponent(limit)}`, "_blank", "noopener"); };
loadGraph();
