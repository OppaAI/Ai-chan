const svg = d3.select("#canvas");
const W=400,H=600;

function drawFigure(){
  // sharp human silhouette (head, torso, limbs)
  const g = svg.append("g").attr("id","figure");
  // head
  g.append("ellipse").attr("cx",200).attr("cy",70).attr("rx",36).attr("ry",42).attr("class","figure-fill figure-outline");
  // eyes
  g.append("circle").attr("cx",188).attr("cy",68).attr("r",4).attr("fill","#4ecdc4").attr("opacity",0.9);
  g.append("circle").attr("cx",212).attr("cy",68).attr("r",4).attr("fill","#4ecdc4").attr("opacity",0.9);
  // ears
  g.append("ellipse").attr("cx",165).attr("cy",70).attr("rx",6).attr("ry",10).attr("fill","#ffe66d").attr("opacity",0.7);
  g.append("ellipse").attr("cx",235).attr("cy",70).attr("rx",6).attr("ry",10).attr("fill","#ffe66d").attr("opacity",0.7);
  // mouth
  g.append("ellipse").attr("cx",200).attr("cy",86).attr("rx",10).attr("ry",4).attr("fill","#ff9f43").attr("opacity",0.8);
  // torso
  g.append("path").attr("d","M165 115 L235 115 L225 260 L175 260 Z").attr("class","figure-fill figure-outline");
  // arms
  g.append("path").attr("d","M165 125 L110 200 L105 210 L155 240 L175 220").attr("class","figure-fill figure-outline");
  g.append("path").attr("d","M235 125 L290 200 L295 210 L245 240 L225 220").attr("class","figure-fill figure-outline");
  // legs
  g.append("path").attr("d","M175 260 L150 420 L140 580 L170 580 L190 420").attr("class","figure-fill figure-outline");
  g.append("path").attr("d","M225 260 L250 420 L260 580 L230 580 L210 420").attr("class","figure-fill figure-outline");
  // spine line
  g.append("line").attr("x1",200).attr("y1",115).attr("x2",200).attr("y2",260).attr("stroke","#636e72").attr("stroke-dasharray","4 4").attr("opacity",0.5);
}

drawFigure();

let nodes=[], edges=[];
const statusEl=document.getElementById("status");
const statsEl=document.getElementById("stats");
const detailsEl=document.getElementById("details");

function setStatus(t){ statusEl.textContent=t; }

async function loadGraph(){
  const lim=document.getElementById("limit").value||400;
  setStatus("loading…");
  try{
    const r=await fetch(`/studio/codebase/api/graph?limit=${lim}`);
    const j=await r.json();
    nodes=j.nodes||[]; edges=j.edges||[];
    statsEl.textContent=`${nodes.length} files • ${edges.length} body edges • ${j.meta?.path||""}`;
    setStatus(j.meta?.exists ? "codebase.db ready" : "codebase.db missing — run ingest");
    render();
  }catch(e){ setStatus("error: "+e); }
}

function render(){
  svg.selectAll(".node").remove();
  svg.selectAll(".edge").remove();
  const showLabels=document.getElementById("show-labels").checked;
  // edges (body co-located, faint)
  const eg = svg.append("g").attr("id","edges");
  edges.forEach(e=>{
    const s=nodes.find(n=>n.id===e.source), t=nodes.find(n=>n.id===e.target);
    if(!s||!t) return;
    eg.append("line").attr("class","edge").attr("x1",s.x*W).attr("y1",s.y*H).attr("x2",t.x*W).attr("y2",t.y*H).attr("stroke","#3a2a5a").attr("stroke-width",0.7).attr("opacity",0.35);
  });
  // nodes
  const ng = svg.append("g").attr("id","nodes");
  const sel = ng.selectAll("g.node").data(nodes).enter().append("g").attr("class","node").attr("transform",d=>`translate(${d.x*W},${d.y*H})`).on("click", (ev,d)=>{
    detailsEl.style.display="block";
    detailsEl.innerHTML=`<b>${d.path}</b><br/><span style="font-size:10px;color:${d.color}">${d.body_part}</span><br/><span style="font-size:11px;word-break:break-all">${d.title}</span>`;
  });
  sel.append("circle").attr("r",4.5).attr("fill",d=>d.color).attr("stroke","#0d0a14").attr("stroke-width",1.2).attr("opacity",0.9);
  if(showLabels){
    sel.append("text").attr("x",7).attr("y",3).attr("font-size","7px").attr("fill","#cbb8ff").attr("opacity",0.85).text(d=>d.path.split("/").pop().slice(0,18));
  }
}

document.getElementById("refresh").onclick=loadGraph;
document.getElementById("show-labels").onchange=render;
document.getElementById("ingest").onclick=async()=>{
  setStatus("ingesting… (SHA1 incremental)");
  const r=await fetch("/studio/codebase/api/ingest?force=false");
  const j=await r.json();
  alert(JSON.stringify(j,null,2));
  loadGraph();
};
document.getElementById("search-btn").onclick=async()=>{
  const q=document.getElementById("search-q").value.trim();
  if(!q) return;
  setStatus("searching…");
  const r=await fetch(`/studio/codebase/api/search?q=${encodeURIComponent(q)}&limit=8`);
  const j=await r.json();
  document.getElementById("search-stats").textContent=j.meta.count+" hits";
  const hitsEl=document.getElementById("search-hits");
  hitsEl.innerHTML=j.hits.map(h=>`<div><b>${h.path}</b> <span style="color:var(--dim)">[${(h.score||0).toFixed(3)}]</span><br/>${(h.text||"").slice(0,120).replace(/</g,"&lt;")}</div>`).join("");
  // highlight
  const hitIds=new Set(j.hits.map(h=>h.id));
  svg.selectAll(".node circle").attr("stroke",d=> hitIds.has(d.id) ? "#fff" : "#0d0a14").attr("stroke-width",d=> hitIds.has(d.id) ? 2.2 : 1.2);
  setStatus("search done");
};
loadGraph();
