const svg = d3.select("#canvas");
const W = 560, H = 720;
let nodes = [], edges = [];
let activeSystems = new Set();
const statusEl = document.getElementById("status");
const statsEl = document.getElementById("stats");
const detailsEl = document.getElementById("details");

function setStatus(text) { statusEl.textContent = text; }
function esc(text) { const node = document.createElement("span"); node.textContent = text || ""; return node.innerHTML; }

function drawFigure() {
  const defs = svg.append("defs");
  defs.append("filter").attr("id", "glow").append("feGaussianBlur").attr("stdDeviation", 2).attr("result", "blur");
  const g = svg.append("g").attr("id", "figure");
  const stroke = { fill: "rgba(22,126,151,.08)", stroke: "#1cb7ca", "stroke-width": 1.35 };
  // Technical mannequin outline; the interior is intentionally reserved for modules.
  g.append("ellipse").attr("cx",280).attr("cy",94).attr("rx",68).attr("ry",77).attr("fill",stroke.fill).attr("stroke",stroke.stroke).attr("stroke-width",stroke["stroke-width"]);
  g.append("path").attr("d","M212 89H348 M280 17V170 M218 53H342 M218 125H342 M258 170L252 190H308L302 170 M178 205L102 358L124 370L219 271 M382 205L458 358L436 370L341 271 M202 307L184 513L166 680H234L260 499 M358 307L376 513L394 680H326L300 499").attr("fill","none").attr("stroke","#1cb7ca").attr("stroke-width",1.35);
  g.append("path").attr("d","M202 307Q280 345 358 307 M184 513H376 M166 680H234 M326 680H394 M196 250H364").attr("fill","none").attr("stroke","#155e76").attr("stroke-width",1);
  [[251,80,23,14],[309,80,23,14]].forEach(([cx,cy,rx,ry]) => g.append("ellipse").attr("cx",cx).attr("cy",cy).attr("rx",rx).attr("ry",ry).attr("fill","rgba(53,231,242,.09)").attr("stroke","#35e7f2").attr("stroke-width",1));
  g.append("path").attr("d","M267 120Q280 128 293 120 M280 170V503").attr("fill","none").attr("stroke","#35e7f2").attr("stroke-width",1).attr("stroke-dasharray","3 4").attr("opacity",.7);
  [230,250,270,290,310,330].forEach(y => g.append("line").attr("x1",264).attr("x2",296).attr("y1",y).attr("y2",y).attr("stroke","#1a91a8").attr("stroke-width",1));
  g.append("text").attr("x",280).attr("y",710).attr("text-anchor","middle").attr("fill","#3a8ba0").attr("font-size",8).attr("letter-spacing",3).text("CODEBASE BODY MAP");
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
  edges.forEach(edge => { const source=byId.get(edge.source), target=byId.get(edge.target); if (!source || !target) return; const dependency=edge.kind === "dependency"; edgeGroup.append("line").attr("x1",source.x*W).attr("y1",source.y*H).attr("x2",target.x*W).attr("y2",target.y*H).attr("stroke",dependency ? "#55e8f3" : "#238ba0").attr("stroke-width",dependency ? 1.2 : .7).attr("opacity",dependency ? .68 : .3); });
  const group = svg.append("g").attr("id","nodes");
  const selection = group.selectAll("g.node").data(visibleNodes).enter().append("g").attr("class","node").attr("transform", d => `translate(${d.x*W},${d.y*H})`).style("cursor","pointer").on("click", (event, d) => showModule(event, d));
  selection.append("circle").attr("r",8).attr("fill",d=>d.color).attr("opacity",.12).attr("stroke",d => d.coverage === null ? "#456978" : d.coverage >= 80 ? "#63f5b0" : d.coverage >= 50 ? "#f5d76e" : "#ff7f82").attr("stroke-width",1.1);
  selection.append("circle").attr("r",d => 3.2 + Math.min(2, d.dependency_count / 4)).attr("fill",d=>d.color).attr("stroke","#d9ffff").attr("stroke-width",d => d.change_count > 8 ? 1.4 : .7);
  if (document.getElementById("show-labels").checked) selection.append("text").attr("x",7).attr("y",-5).attr("fill","#b9f4f8").attr("font-size",7.2).attr("letter-spacing",.4).text(d => d.module.split("/").slice(-1)[0].slice(0,22));
}

async function showModule(event, node) {
  svg.selectAll("g.node circle:last-child").attr("r",3.2); d3.select(event.currentTarget).select("circle:last-child").attr("r",5);
  detailsEl.innerHTML = `<p class="details-kicker">Reading module</p><h2>${esc(node.module)}</h2><p>Building a natural-language brief from the indexed source…</p>`;
  try {
    const response = await fetch(`/studio/codebase/api/module?module=${encodeURIComponent(node.module)}`); const detail = await response.json();
    if (detail.error) throw new Error(detail.error);
    const functions = detail.functions.length ? `<ul>${detail.functions.map(name => `<li><code>${esc(name)}()</code></li>`).join("")}</ul>` : "<p>No callable symbols were extracted from the indexed excerpts.</p>";
    const files = detail.files.slice(0, 7).map(file => `<li><code>${esc(file)}</code></li>`).join("");
    const coverage = detail.coverage === null ? "unavailable" : `${detail.coverage}%`;
    const pills = Object.entries(detail.metrics).map(([label, value]) => `<span class="badge">${esc(label.replace("_", " "))}: ${esc(String(value))}</span>`).join("") + `<span class="badge">changes: ${detail.change_count}</span><span class="badge">coverage: ${coverage}</span>`;
    const links = detail.source_links.slice(0, 7).map(link => `<li><code>${esc(link.path)}</code> ${link.github ? `<a class="source-link" href="${esc(link.github)}" target="_blank" rel="noopener">GitHub</a>` : ""} <a class="source-link" href="${esc(link.vscode)}">IDE</a></li>`).join("");
    const relationList = values => values.length ? `<ul>${values.map(value => `<li><code>${esc(value)}</code></li>`).join("")}</ul>` : "<p>None found in indexed imports.</p>";
    detailsEl.innerHTML = `<p class="details-kicker">${esc(detail.body_part)} system · ${detail.files.length} files</p><h2>${esc(detail.module)}</h2><p>${esc(detail.summary)}</p><div>${pills}</div>${detail.docstrings.length ? `<p class="details-kicker">Source intent</p><p>${esc(detail.docstrings.join(" · "))}</p>` : ""}<p class="details-kicker">Calls</p>${relationList(detail.dependencies)}<p class="details-kicker">Called by</p>${relationList(detail.dependents)}<p class="details-kicker">Functions and classes</p>${functions}<p class="details-kicker">Open source</p><ul>${links}</ul>${detail.excerpt ? `<p class="details-kicker">Indexed context</p><p>${esc(detail.excerpt)}…</p>` : ""}`;
  } catch (error) { detailsEl.innerHTML = `<p class="details-kicker">Module briefing</p><h2>${esc(node.module)}</h2><p>${esc(error.message)}</p>`; }
}

document.getElementById("refresh").onclick = loadGraph;
document.getElementById("show-labels").onchange = render;
document.getElementById("ingest").onclick = async () => { setStatus("Indexing codebase…"); try { const response=await fetch("/studio/codebase/api/ingest?force=false"); const result=await response.json(); setStatus(result.ok ? `Indexed ${result.docs_added} updated files` : result.error || "Indexing failed"); loadGraph(); } catch (error) { setStatus(error.message); } };
document.getElementById("search-btn").onclick = async () => { const query=document.getElementById("search-q").value.trim(); if (!query) return; setStatus("Searching…"); const response=await fetch(`/studio/codebase/api/search?q=${encodeURIComponent(query)}&limit=8`); const result=await response.json(); document.getElementById("search-stats").textContent=`${result.meta.count || 0} matching excerpts`; document.getElementById("search-hits").innerHTML=result.hits.map(hit => `<div class="search-hit" title="${esc(hit.path)}"><b>${esc(hit.path)}</b><br><span>${esc((hit.text || "").slice(0,100))}</span></div>`).join("") || "<div class=\"stat\">No matching code was found.</div>"; setStatus("Search complete"); };
document.getElementById("export-md").onclick = () => { const limit = document.getElementById("limit").value || 400; window.open(`/studio/codebase/api/export/markdown?limit=${encodeURIComponent(limit)}`, "_blank", "noopener"); };
loadGraph();
