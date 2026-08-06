/* Knowledge Graph Studio — knowledge + entity nodes only */
const API_BASE = window.KNOWLEDGE_API_BASE || '';
let graph = { nodes: [], edges: [], meta: {} };
let simulation = null;
let zoomBeh = null;

function importanceOf(d) {
  const sc = d.scores || {};
  if (sc.importance != null) return Math.max(0, Math.min(1, Number(sc.importance)));
  if (sc.retain != null) return Math.max(0, Math.min(1, Number(sc.retain)));
  if (d.size != null) {
    const s = Math.max(0.20, Number(d.size));
    const t = Math.max(0, Math.min(1, (s - 0.20) / 1.10));
    return Math.pow(t, 1 / 1.25);
  }
  return d.type === 'entity' ? 0.4 : 0.3;
}

function nodeRadius(d) {
  const r = importanceOf(d);
  if (d.type === 'entity') return 5 + 16 * Math.pow(r, 1.25);
  return 5 + 24 * Math.pow(r, 1.35);
}

function nodeOpacity(d) {
  return 0.22 + importanceOf(d) * 0.78;
}

function glowStrength(d) {
  return 0.4 + importanceOf(d) * 2.8;
}

function nodeColor(d) {
  if (d.type === 'entity') return '#a78bfa';
  return '#4ade80';
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

async function loadGraph() {
  const limit = document.getElementById('limit').value || 200;
  const res = await fetch(`${API_BASE}/api/graph?limit=${encodeURIComponent(limit)}`);
  if (!res.ok) throw new Error(`graph request failed: ${res.status}`);
   graph = await res.json();
  if (graph && graph.meta && graph.meta.error) {
    throw new Error(`graph export failed: ${graph.meta.error}`);
  render();
}

function filteredNodesEdges() {
  const minI = parseFloat(document.getElementById('min-imp').value || '0') || 0;
  const entQ = (document.getElementById('filter-entity').value || '').trim().toLowerCase();

  let nodes = (graph.nodes || []).map(n => ({ ...n })).filter(d => {
    if (d.type === 'entity') {
      if (entQ && !(String(d.label || d.text || '').toLowerCase().includes(entQ))) return false;
      return true;
    }
    if (importanceOf(d) < minI) return false;
    if (entQ) {
      const ents = (d.entities || []).map(e => String(e).toLowerCase());
      if (!ents.some(e => e.includes(entQ))) return false;
    }
    return true;
  });

  const keep = new Set(nodes.filter(n => n.type === 'knowledge').map(n => n.id));
  for (const e of (graph.edges || [])) {
    if (e.type === 'about' && keep.has(e.source)) keep.add(e.target);
  }
  nodes = (graph.nodes || []).map(n => ({ ...n })).filter(n => {
    if (n.type === 'knowledge') return keep.has(n.id) && importanceOf(n) >= minI;
    if (n.type === 'entity') {
      if (!keep.has(n.id)) return false;
      if (entQ && !(String(n.label || '').toLowerCase().includes(entQ))) return false;
      return true;
    }
    return false;
  });

  const idSet = new Set(nodes.map(n => n.id));
  const links = (graph.edges || [])
    .map(e => ({ ...e }))
    .filter(e => idSet.has(e.source) && idSet.has(e.target));
  return { nodes, links };
}

function showDetails(d) {
  const el = document.getElementById('details');
  el.style.display = 'block';
  const sc = d.scores || {};
  const keys = ['importance', 'access', 'recency', 'connectivity'];
  const bars = keys.map(k => {
    const v = Math.max(0, Math.min(1, Number(sc[k] || 0)));
    return `<div class="detail-label">${k}</div>
      <div class="score-bar"><i style="width:${(v * 100).toFixed(0)}%"></i></div>
      <div class="detail-value">${v.toFixed(3)}</div>`;
  }).join('');
  const rows = [
    ['type', d.type],
    ['id', d.id],
    ['title', d.doc_title || d.label || ''],
    ['source', d.doc_source || '—'],
    ['status', d.status || '—'],
    ['access', d.access_count ?? '—'],
    ['entities', (d.entities || []).join(', ') || '—'],
    ['text', d.text || ''],
  ];
  el.innerHTML = '<h3 style="margin:0 0 8px;font-size:13px">Node</h3>' +
    rows.map(([k, v]) =>
      `<div class="detail-label">${k}</div><div class="detail-value">${escapeHtml(String(v))}</div>`
    ).join('') +
    '<h3 style="margin:12px 0 6px;font-size:13px">Scores</h3>' + bars;
}

function render() {
  if (simulation) {
    simulation.stop();
    simulation = null;
  }
  const svg = d3.select('#svg');
  svg.selectAll('*').remove();
  const canvas = document.getElementById('canvas');
  const w = canvas.clientWidth || 800;
  const h = canvas.clientHeight || 600;
  svg.attr('viewBox', [0, 0, w, h]);

  const { nodes, links } = filteredNodesEdges();
  nodes.forEach(d => {
    d._col = nodeColor(d);
    d._glassId = 'g' + String(d.id).replace(/[^a-zA-Z0-9]/g, '_');
  });

  const gRoot = svg.append('g');
  zoomBeh = d3.zoom().scaleExtent([0.15, 4]).on('zoom', (ev) => gRoot.attr('transform', ev.transform));
  svg.call(zoomBeh);

  const defs = svg.append('defs');
  nodes.forEach(d => {
    const grad = defs.append('radialGradient').attr('id', d._glassId)
      .attr('cx', '35%').attr('cy', '30%').attr('r', '65%');
    grad.append('stop').attr('offset', '0%').attr('stop-color', '#fff').attr('stop-opacity', 0.35);
    grad.append('stop').attr('offset', '45%').attr('stop-color', d._col).attr('stop-opacity', 0.9);
    grad.append('stop').attr('offset', '100%').attr('stop-color', d._col).attr('stop-opacity', 0.5);
  });
  const glow = defs.append('filter').attr('id', 'glow')
    .attr('x', '-50%').attr('y', '-50%').attr('width', '200%').attr('height', '200%');
  glow.append('feGaussianBlur').attr('in', 'SourceGraphic').attr('stdDeviation', '1.5').attr('result', 'blur');
  glow.append('feMerge').selectAll('feMergeNode').data(['blur', 'SourceGraphic']).join('feMergeNode').attr('in', d => d);

  const link = gRoot.append('g').selectAll('line').data(links).join('line')
    .attr('stroke', d => d.type === 'same_doc' ? '#5b4a6e' : '#6ee7a8')
    .attr('stroke-width', d => d.type === 'about' ? 1.2 : 0.7)
    .attr('stroke-opacity', 0.35);

  const node = gRoot.append('g').selectAll('g').data(nodes).join('g')
    .style('cursor', 'pointer')
    .call(d3.drag()
      .on('start', (ev, d) => {
        if (!ev.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      })
      .on('drag', (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
      .on('end', (ev, d) => {
        if (!ev.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      })
    )
    .on('click', (ev, d) => { ev.stopPropagation(); showDetails(d); });

  node.append('circle')
    .attr('r', d => nodeRadius(d) + glowStrength(d) * 0.4)
    .attr('fill', d => d._col)
    .attr('opacity', d => 0.12 + importanceOf(d) * 0.2)
    .attr('filter', 'url(#glow)');

  node.append('circle')
    .attr('r', nodeRadius)
    .attr('fill', d => `url(#${d._glassId})`)
    .attr('stroke', d => d._col)
    .attr('stroke-width', 0.9)
    .attr('stroke-opacity', d => 0.35 + importanceOf(d) * 0.4)
    .attr('opacity', nodeOpacity);

  simulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links).id(d => d.id).distance(70).strength(0.35))
    .force('charge', d3.forceManyBody().strength(-180))
    .force('center', d3.forceCenter(w / 2, h / 2))
    .force('collision', d3.forceCollide().radius(d => nodeRadius(d) + 4))
    .on('tick', () => {
      link
        .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
      node.attr('transform', d => `translate(${d.x},${d.y})`);
    });
}

function svgZoom(k) {
  if (!zoomBeh) return;
    const svg = d3.select('`#svg`');
    svg.transition().call(zoomBeh.scaleBy, k);
}

function init() {
  document.getElementById('reload').onclick = loadGraph;
  document.getElementById('refilter').onclick = () => render();
  document.getElementById('z-in').onclick = () => svgZoom(1.25);
  document.getElementById('z-out').onclick = () => svgZoom(0.8);
  document.getElementById('z-fit').onclick = () => {
    if (!zoomBeh) return;
    d3.select('#svg').transition().call(zoomBeh.transform, d3.zoomIdentity);
  };

  loadGraph().catch(err => {
    const el = document.getElementById('details');
    el.style.display = 'block';
    el.textContent = 'Load failed: ' + err;
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
