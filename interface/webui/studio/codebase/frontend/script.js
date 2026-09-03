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
  // High-tech neon pink + purple body gradient (cuter, glowing).
  const bodyGradient = defs.append("linearGradient").attr("id", "body-gradient").attr("x1", "0%").attr("x2", "100%");
  bodyGradient.append("stop").attr("offset", "0%").attr("stop-color", "#ff6eb3");
  bodyGradient.append("stop").attr("offset", "40%").attr("stop-color", "#6b2a6b");
  bodyGradient.append("stop").attr("offset", "100%").attr("stop-color", "#2b0f3a");
  // Cute glow filter.
  const glow = defs.append("filter").attr("id", "glow");
  glow.append("feGaussianBlur").attr("stdDeviation", 2.5).attr("result", "blur");
  glow.append("feMerge").selectAll("feMergeNode").data(["blur", "SourceGraphic"]).enter().append("feMergeNode").attr("in", d => d);

  const g = svg.append("g").attr("id", "figure");
  const outline = { fill: "url(#body-gradient)", stroke: "#ff8ec8", "stroke-width": 2.2 };

  // Cuter, rounder robot body — soft rounded head, cute rounded shoulders, rounded legs.
  g.append("path").attr("d", "M280 22C248 22 224 52 224 92c0 24 10 44 26 58l-2 24-30 14c-18 8-30 26-38 50l-32 100 22 9 34-84 14 22-16 136-14 140h52l22-116 12 116h52l-14-140-16-136 14-22 34 84 22-9-32-100c-8-24-20-42-38-50l-30-14-2-24c16-14 26-34 26-58 0-40-24-70-56-70Z").attr("fill", outline.fill).attr("stroke", outline.stroke).attr("stroke-width", outline["stroke-width"]);

  // Body outline segments (high-tech neon lines).
  g.append("path").attr("d", "M280 22V640 M224 92h112 M260 148h40 M220 190c14 12 30 18 46 18s32-6 46-18 M220 300c16 13 32 18 48 18s32-5 48-18 M245 460h70").attr("fill", "none").attr("stroke", "#ff6eb3").attr("stroke-width", 1.4);

  // Cute bright glowing eyes (high-tech neon teal with pink glow rim).
  [[255,84,20,15],[305,84,20,15]].forEach(([cx,cy,rx,ry]) => g.append("ellipse").attr("cx",cx).attr("cy",cy).attr("rx",rx).attr("ry",ry).attr("fill","rgba(100,255,210,.32)").attr("stroke","#ff8ec8").attr("stroke-width",2.5));
  // Cute smile mouth.
  g.append("path").attr("d","M268 126Q280 142 292 126 M250 158h60l-4 22h-56Z").attr("fill","none").attr("stroke","#ff8ec8").attr("stroke-width",2);
  // Cute rounded ear nubs — listen markers.
  [[212,104,9],[328,104,9]].forEach(([cx,cy,r]) => g.append("circle").attr("cx",cx).attr("cy",cy).attr("r",r).attr("fill","rgba(255,150,200,.38)").attr("stroke","#ff8ec8").attr("stroke-width",2));
  // Glowing brain / cognition (cute pink glow) — listen/ears nearby, brain in head.
  g.append("path").attr("d","M248 235c-14 9-12 38 8 50 11 5 21-2 26-10 6 9 15 16 26 10 20-12 22-41 8-50-10-7-22 1-30 10-8-9-18-17-30-10Z").attr("fill","rgba(255,100,180,.22)").attr("stroke","#ff8ec8").attr("stroke-width",1.5).attr("filter","url(#glow)");
  // Voice / mouth layer (cute neon glow) — speak = mouth.
  g.append("path").attr("d","M262 326c-16 12-12 38-6 56l18 48 18-48c6-18 10-44-6-56Z").attr("fill","rgba(255,160,210,.18)").attr("stroke","#ff8ec8").attr("stroke-width",1.2).attr("filter","url(#glow)");
  // Bright neon right-side conduits (high-tech machine joints).
  g.append("path").attr("d","M280 34h38l11 22v50l-14 22h-28 M280 178h50l14 20-8 74-20 14h-32 M280 334h46l14 22-14 82-18 16h-28").attr("fill","none").attr("stroke","#d0a0ff").attr("stroke-width",1.2);
  // Cute joint dots (neon pink + teal alternating).
  [64,76,112,198,216,232,252,270,288,374,396,422,448].forEach((y, index) => g.append("line").attr("x1",290).attr("x2",index < 3 ? 326 : 344).attr("y1",y).attr("y2",y).attr("stroke",index % 2 ? "#51d4c8" : "#ff8ec8").attr("stroke-width",1.2));
  [[338,236,12],[338,396,12],[408,334,11],[152,334,12],[220,508,9],[342,508,9]].forEach(([cx,cy,r]) => g.append("circle").attr("cx",cx).attr("cy",cy).attr("r",r).attr("fill","#171124").attr("stroke","#ff8ec8").attr("stroke-width",1.2));
  // Label at bottom.
  g.append("text").attr("x",280).attr("y",708).attr("text-anchor","middle").attr("fill","#d8bcff").attr("font-size",9).attr("letter-spacing",3).attr("font-family","system-ui, ui-sans-serif, sans-serif").attr("font-weight","600").text("AIKO · CODEBASE ATLAS");
}

drawFigure();

const zoom = d3.zoom().scaleExtent([.65, 3]).on("zoom", event => svg.selectAll("#figure,#edges,#nodes").attr("transform", event.transform));
svg.call(zoom);
document.getElementById("zoom-in").onclick = () => svg.transition().call(zoom.scaleBy, 1.25);
document.getElementById("zoom-out").onclick = () => svg.transition().call(zoom.scaleBy, .8);
document.getElementById("zoom-reset").onclick = () => svg.transition().call(zoom.transform, d3.zoomIdentity);
document.getElementById("print-atlas").onclick = () => window.print();