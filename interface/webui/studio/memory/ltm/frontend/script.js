const API_BASE = GraphBoot.apiBase();
const ARC_KEYS = ['salience', 'spacing', 'connectivity', 'valence', 'access'];
/* quiet rim — low opacity, not competing with glass body */
const ARC_COLORS = ['#ffd27a66', '#6bcf7f66', '#b794f666', '#3de0ff55', '#e8eef855'];

let graph = { nodes: [], edges: [], meta: {} };
let simulation = null;
let zoomBehavior = null;
let userEntityId = null;
let currentUsername = 'OppaAI';

async function getCurrentUser() {
  try {
    const resp = await fetch('/api/auth/me', { credentials: 'include' });
    if (resp.ok) {
      const data = await resp.json();
      currentUsername = data.username || currentUsername;
    }
  } catch (e) {}
  return currentUsername;
}

function qs() {
  const p = new URLSearchParams();
  const uid = document.getElementById('user-id').value.trim();
  if (uid) p.set('user_id', uid);
  p.set('limit', document.getElementById('limit').value || '200');
  p.set('include_history', document.getElementById('include-history').checked);
  p.set('include_entities', document.getElementById('include-entities').checked);
  p.set('include_knowledge', document.getElementById('include-knowledge').checked);
  p.set('include_experience', document.getElementById('include-experience').checked);
  p.set('include_episodes', document.getElementById('include-episodes').checked);
  const df = document.getElementById('date-from').value;
  const dt = document.getElementById('date-to').value;
  if (df) p.set('date_from', df);
  if (dt) p.set('date_to', dt);
  return p.toString();
}

async function loadGraph() {
  document.getElementById('status').textContent = 'Loading…';
  try {
    const resp = await fetch(`${API_BASE}/graph?` + qs());
    graph = await resp.json();
    stretchRetain(graph.nodes);
    const m = graph.meta || {};
    document.getElementById('stats').innerHTML =
      `neurons: ${m.memory_count ?? 0}<br/>entities: ${m.entity_count ?? 0}<br/>knowledge: ${(graph.nodes||[]).filter(n=>n.type==='knowledge').length}<br/>experience: ${(graph.nodes||[]).filter(n=>n.type==='experience').length}<br/>episodes: ${m.episode_count ?? 0}<br/>synapses: ${m.edge_count ?? 0}`;
    document.getElementById('status').textContent =
      `${(graph.nodes||[]).length} nodes · ${(graph.edges||[]).length} edges`;
    await getCurrentUser();
    userEntityId = `ent:${currentUsername.toLowerCase()}`;
    render();
  } catch (e) {
    document.getElementById('status').textContent = 'Load failed';
    console.error(e);
  }
}

/** Retain in [0,1] — drives BOTH size and brightness */
function retainOf(d) {
  if (d._dispRetain != null) return d._dispRetain;
  const sc = d.scores || {};
  if (sc.retain != null) return Math.max(0, Math.min(1, Number(sc.retain)));
  if (d.size != null) {
    // backend size ≈ 0.18 + 1.27 * retain^1.18 → approximate invert
    const s = Math.max(0.18, Number(d.size));
    const t = Math.max(0, Math.min(1, (s - 0.18) / 1.27));
    return Math.pow(t, 1 / 1.18);
  }
  return d.type === 'entity' ? 0.4 : 0.35;
}

/**
 * Contrast-stretch memory retain values across the visible node set, so
 * size/brightness differences are actually visible when scores cluster
 * (raw retain is often ~flat, e.g. 0.48 for half the nodes).
 * Stronger curve + slight valence lift so cyan/gold nodes don't stay tiny/dim.
 */
function stretchRetain(nodes) {
  for (const n of nodes) if (n._dispRetain != null) delete n._dispRetain;
  const mem = (nodes || []).filter(n => n.type === 'memory');
  const r = mem.map(retainOf).filter(v => isFinite(v));
  if (r.length < 3) return;
  let lo = Math.min.apply(null, r);
  let hi = Math.max.apply(null, r);
  // Force a usable dynamic range even when scores are tightly clustered
  if (hi - lo < 0.18) { lo = Math.max(0, lo - 0.22); hi = Math.min(1, hi + 0.22); }
  const span = (hi - lo) || 1;
  for (const n of mem) {
    const v = retainOf(n);
    // Power curve expands mid/high retain; floor keeps low nodes visible
    let stretched = 0.08 + 0.92 * Math.pow((v - lo) / span, 0.72);
    // Mild valence lift so pos/neg still read larger/brighter than pure neutrals
    const hue = valenceHue(n);
    if (hue === 'pos' || hue === 'neg') stretched = Math.min(1, stretched + 0.07);
    if (n.pinned) stretched = Math.max(stretched, 0.78);
    n._dispRetain = Math.max(0.08, Math.min(1, stretched));
  }
}

function valenceHue(d) {
  if (d.type === 'entity') return 'entity';
  if (d.type === 'knowledge') return 'knowledge';
  if (d.type === 'experience') return 'experience';
  if (d.type === 'episode') return 'episode';
  if (d.imprint) return 'imprint';
  let v = (d.valence_tag || 'neutral').toLowerCase();
  const vs = d.valence_score;
  if ((v === 'neutral' || !v) && vs != null && vs !== '') {
    const s = Number(vs);
    if (s <= -1) v = 'neg';
    else if (s >= 1) v = 'pos';
  }
  if (v === 'neg') return 'neg';
  if (v === 'pos') return 'pos';
  return 'neutral';
}

function hueColor(hue) {
  if (hue === 'entity') return '#b794f6';
  if (hue === 'knowledge') return '#4ade80';
  if (hue === 'experience') return '#fb923c';
  if (hue === 'episode') return '#51d4c8';
  if (hue === 'imprint') return '#c651a8';
  if (hue === 'neg') return '#3de0ff';
  if (hue === 'pos') return '#f0c14a';
  return '#8a9bb8';
}

/** Glass fill opacity — steeper retain curve + valence boost for demo-like contrast */
function nodeOpacity(d) {
  let r = retainOf(d);
  // Quadratic-ish so mid-retain is already bright; floor keeps dim nodes readable
  let o = 0.22 + Math.pow(r, 0.85) * 0.78;
  const hue = valenceHue(d);
  if (hue === 'pos' || hue === 'neg') o = Math.min(1, o + 0.08);
  if (d.pinned) o = Math.max(o, 0.88);
  if (d.type === 'memory' && (d.status === 'superseded' || d.is_tip === false)) {
    o = Math.min(o, 0.32);
  }
  return o;
}

function glowStrength(d) {
  let r = retainOf(d);
  if (d.pinned) r = Math.max(r, 0.92);
  if (d.status === 'superseded') return 0.45;
  const hue = valenceHue(d);
  // Emotional nodes get extra glow so cyan/gold stand out like the demo
  const emo = (hue === 'pos' || hue === 'neg') ? 0.55 : 0;
  return 0.55 + r * 3.4 + emo;
}

function nodeRadius(d) {
  let r = retainOf(d);
  // Pinned still large but not so dominant that everything else looks uniform
  if (d.pinned) r = Math.max(r, 0.72);
  const hue = valenceHue(d);
  // Slight size lift for emotional nodes (demo has varied cyan/gold sizes)
  if ((hue === 'pos' || hue === 'neg') && d.type === 'memory') {
    r = Math.min(1, r + 0.06);
  }
  if (d.type === 'entity') return 4.5 + 14 * Math.pow(r, 1.15);
  // Wider radius range: small neutrals vs large pinned/emotional
  return 4 + 28 * Math.pow(r, 1.22);
}

function edgeOpacity(e, nodeById) {
  const sid = typeof e.source === 'object' ? e.source.id : e.source;
  const tid = typeof e.target === 'object' ? e.target.id : e.target;
  const s = nodeById.get(sid);
  const t = nodeById.get(tid);
  const rs = s ? retainOf(s) : 0.25;
  const rt = t ? retainOf(t) : 0.25;
  // Geometric-ish mean + power expands contrast when retains cluster
  const mid = Math.pow(Math.max(0.05, rs * rt), 0.45);
  const w = Math.max(0, Math.min(1, Number(e.weight) || 0.4));
  const wBoost = 0.35 + 0.65 * w; // weak weight stays dim
  if (e.type === 'supersedes') return Math.min(0.95, (0.25 + mid * 0.7) * wBoost);
  if (e.type === 'mentions' || e.type === 'grounded_in' || e.type === 'practiced_in' || e.type === 'distilled_into') {
    return Math.min(0.9, (0.06 + mid * 0.75) * wBoost);
  }
  return Math.min(0.7, (0.04 + mid * 0.55) * wBoost);
}

function lineageText(d, graph) {
  if (!d || d.type !== 'memory') return '';
  const edges = (graph.edges || []).filter(e => e.type === 'supersedes');
  const nodes = Object.fromEntries((graph.nodes || []).map(n => [n.id, n]));
  const idOf = (x) => (x && typeof x === 'object' ? x.id : x);
  const textOf = (n) => String((n && (n.text || n.label)) || '').slice(0, 100);
  // export: source=newer → target=older
  const olderId = idOf(edges.find(e => idOf(e.source) === d.id)?.target);
  const newerId = idOf(edges.find(e => idOf(e.target) === d.id)?.source);
  const bits = [];
  const selfText = textOf(d);
  if (d.status === 'superseded' || d.is_tip === false) {
    bits.push('(superseded)');
    bits.push('Was: ' + selfText);
    if (newerId && nodes[newerId]) bits.push('Now: ' + textOf(nodes[newerId]));
  } else {
    if (olderId && nodes[olderId]) bits.push('Was: ' + textOf(nodes[olderId]));
    bits.push('Now: ' + selfText);
  }
  return bits.join('\n');
}

function showDetails(d) {
  const el = document.getElementById('details');
  el.style.display = 'block';
  const sc = d.scores || {};
  const scoreRows = ARC_KEYS.map(k => {
    const v = Math.max(0, Math.min(1, Number(sc[k] || 0)));
    return `<div class="detail-label">${k}</div>
      <div class="score-bar"><i style="width:${(v*100).toFixed(0)}%"></i></div>
      <div class="detail-value">${v.toFixed(3)}</div>`;
  }).join('');
  const rows = [
    ['type', d.type],
    ['id', d.id],
    ['kind', d.kind],
    ['status', d.status],
    ['pinned', d.pinned ? 'yes' : 'no'],
    ['valence', d.valence_tag || '—'],
    ['retain', retainOf(d).toFixed(3)],
    ['created', d.created_at || '—'],
    ['entities', (d.entities || []).join(', ') || '—'],
    ['supersedes', d.supersedes_id || '—'],
    ['text', d.text || d.label || ''],
  ];
  if (d.type === 'episode') {
    rows.splice(6, 0,
      ['when', d.created_at || d.date || '—'],
      ['date', d.date || '—'],
      ['salience', d.salience_score != null ? d.salience_score : '—'],
      ['arousal', d.arousal_score != null ? d.arousal_score : '—'],
      ['recalls', d.access_count ?? 0],
      ['distilled', d.distilled_at || '—'],
      ['source', d.source || '—'],
      ['session', d.session_id || '—'],
    );
  }
  let html = '<h3>Neuron</h3>' + rows.map(([k,v]) =>
    `<div class="detail-label">${k}</div><div class="detail-value">${escapeHtml(String(v))}</div>`
  ).join('') + '<h3 style="margin-top:14px">Factor scores</h3>' + scoreRows;

  if (d.type === 'episode') {
    const extra = [
      ['recall_count', d.recall_count != null ? d.recall_count : '—'],
      ['distilled', d.distilled ? 'yes' : 'no'],
      ['distilled_at', d.distilled_at || '—'],
      ['distilled_into', (d.distilled_into || []).join(', ') || '—'],
    ];
    html += '<h3 style="margin-top:14px">Episodic</h3>' + extra.map(([k,v]) =>
      `<div class="detail-label">${k}</div><div class="detail-value">${escapeHtml(String(v))}</div>`
    ).join('');
  }

  // Phase 16 — lineage (uses global `graph` from loadGraph)
  const lin = lineageText(d, graph);
  if (lin) {
    html += '<h3 style="margin-top:14px">Lineage</h3>'
      + `<div class="detail-value" style="white-space:pre-wrap">${escapeHtml(lin)}</div>`;
  }
  el.innerHTML = html;
}

function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

async function runSearch() {
  const q = document.getElementById('search-q').value.trim();
  const box = document.getElementById('search-stats');
  if (!q) { box.textContent = 'type a query'; return; }
  box.textContent = 'Searching…';
  try {
    const p = new URLSearchParams();
    p.set('q', q);
    const uid = document.getElementById('user-id').value.trim();
    if (uid) p.set('user_id', uid);
    p.set('limit', '10');
    const resp = await fetch(`${API_BASE}/search?` + p.toString());
    const data = await resp.json();
    const hits = data.hits || [];
    if (!hits.length) { box.textContent = 'No hits'; return; }
    box.innerHTML = hits.map(h =>
      `<div style="margin-bottom:10px;padding:8px;border:1px solid var(--dim);border-radius:4px;background:#120a1c">
         <div style="color:var(--pink);text-transform:uppercase;font-size:9px;letter-spacing:.1em">
           ${escapeHtml(h.store)} · ${escapeHtml(h.kind)} · ${Number(h.score||0).toFixed(3)}
         </div>
         <div style="margin-top:4px;color:var(--text)">${escapeHtml(h.text)}</div>
       </div>`
    ).join('');
  } catch (e) {
    box.textContent = 'Search failed';
    console.error(e);
  }
}

function render() {
  const svg = d3.select('#canvas');
  svg.selectAll('*').remove();
  const area = document.getElementById('canvas-area');
  const w = area.clientWidth || 1000;
  const h = area.clientHeight || 700;
  svg.attr('viewBox', `0 0 ${w} ${h}`);

  // subtle neural field (sparse dots, not galaxy stars)
  const field = svg.append('g').attr('class', 'field');
  for (let i = 0; i < 80; i++) {
    field.append('circle')
      .attr('cx', Math.random() * w)
      .attr('cy', Math.random() * h)
      .attr('r', Math.random() * 0.9)
      .attr('fill', '#3de0ff')
      .attr('opacity', 0.04 + Math.random() * 0.08);
  }

  const st = document.getElementById('filter-status')?.value || 'all';
  const val = document.getElementById('filter-valence')?.value || 'all';
  const minR = parseFloat(document.getElementById('filter-min-retain')?.value || '0') || 0;
  const entQ = (document.getElementById('filter-entity')?.value || '').trim().toLowerCase();
  const showMem = document.getElementById('layer-memory')?.checked !== false;
  const showEnt = document.getElementById('layer-entity')?.checked !== false;
  const showKb = document.getElementById('layer-knowledge')?.checked !== false;
  const showExp = document.getElementById('layer-experience')?.checked !== false;
  const showEp = document.getElementById('layer-episodes')?.checked !== false;

  let nodes = (graph.nodes || []).map(n => ({ ...n })).filter(d => {
    if (d.type === 'entity') return false;
    // Exclude layer types that are toggled off so their about-hubs are not pulled in
    if (d.type === 'memory' && !showMem) return false;
    if (d.type === 'knowledge' && !showKb) return false;
    if (d.type === 'experience' && !showExp) return false;
    if (d.type === 'episode' && !showEp) return false;
    if (st !== 'all' && (d.status || 'active') !== st) return false;
    if (val !== 'all' && (d.valence_tag || 'neutral') !== val) return false;
    if (retainOf(d) < minR) return false;
    if (entQ) {
      const ents = (d.entities || []).map(e => String(e).toLowerCase());
      if (!ents.some(e => e.includes(entQ))) return false;
    }
    return true;
  });
  const keep = new Set(nodes.map(n => n.id));
  const nodeTypeOf = new Map((graph.nodes || []).map(n => [n.id, n.type]));
  // Add knowledge / experience / episode nodes when their layers are on
  // (redundant with filter above when show*=true; keeps explicit layer intent clear)
  for (const n of (graph.nodes || [])) {
    if (n.type === 'knowledge' && showKb) keep.add(n.id);
    if (n.type === 'experience' && showExp) keep.add(n.id);
    if (n.type === 'episode' && showEp) keep.add(n.id);
  }
  // Pull in entity hubs connected to any kept node via any edge type
  // (mentions, about, related_to, grounded_in, practiced_in, co_mentions).
  // Bounded to entity nodes so filtered-out memories can't leak back in.
  let grew = true;
  while (grew) {
    grew = false;
    for (const e of (graph.edges || [])) {
      const s = keep.has(e.source), t = keep.has(e.target);
      if (s && !t) { if (nodeTypeOf.get(e.target) === 'entity') { keep.add(e.target); grew = true; } }
      else if (t && !s) { if (nodeTypeOf.get(e.source) === 'entity') { keep.add(e.source); grew = true; } }
    }
  }
  nodes = (graph.nodes || []).map(n => ({ ...n })).filter(n => {
    if (!keep.has(n.id)) return false;
    if (n.type === 'memory' && !showMem) return false;
    if (n.type === 'entity') {
      if (!showEnt) return false;
      if (entQ && !(String(n.label || n.text || '').toLowerCase().includes(entQ))) return false;
      return true;
    }
    if (n.type === 'knowledge') return showKb;
    if (n.type === 'experience') return showExp;
    if (n.type === 'episode') return showEp;
    return true;
  });
  const keep2 = new Set(nodes.map(n => n.id));
  const links = (graph.edges || []).map(e => ({ ...e }))
    .filter(e => keep2.has(e.source) && keep2.has(e.target));
  const nodeById = new Map(nodes.map(n => [n.id, n]));

  if (!nodes.length) {
    svg.append('text').attr('x', w/2).attr('y', h/2).attr('text-anchor','middle')
      .attr('fill','var(--dim)').text('No memories yet');
    return;
  }

  const g = svg.append('g');
  zoomBehavior = GraphBoot.makeZoom({ scaleExtent: [0.2, 4], target: g });
  svg.call(zoomBehavior);

  // Subtle neural grid background
  const grid = g.append('g').attr('class', 'neural-grid');
  const gridStep = 60;
  for (let x = 0; x <= w; x += gridStep) {
    grid.append('line')
      .attr('x1', x).attr('y1', 0).attr('x2', x).attr('y2', h)
      .attr('stroke', '#3de0ff12').attr('stroke-width', 0.5);
  }
  for (let y = 0; y <= h; y += gridStep) {
    grid.append('line')
      .attr('x1', 0).attr('y1', y).attr('x2', w).attr('y2', y)
      .attr('stroke', '#3de0ff12').attr('stroke-width', 0.5);
  }

  const defs = svg.append('defs');

  // per-node glass gradients + glow filters
  nodes.forEach((d, i) => {
    const hue = valenceHue(d);
    const col = hueColor(hue);
    const op = nodeOpacity(d);
    const gid = `glass-${i}`;
    const grad = defs.append('radialGradient')
      .attr('id', gid)
      .attr('cx', '35%').attr('cy', '30%').attr('r', '70%');
    grad.append('stop').attr('offset', '0%').attr('stop-color', '#ffffff').attr('stop-opacity', 0.35 * op);
    grad.append('stop').attr('offset', '45%').attr('stop-color', col).attr('stop-opacity', 0.55 * op);
    grad.append('stop').attr('offset', '100%').attr('stop-color', col).attr('stop-opacity', 0.15 * op);
    d._glassId = gid;
    d._hueCol = col;
    d._op = op;

    const fid = `glow-${i}`;
    GraphBoot.addGlowFilter(defs, fid, glowStrength(d));
    d._glowId = fid;
  });

  defs.append('marker').attr('id','arrow-sup').attr('viewBox','0 0 10 10')
    .attr('refX', 22).attr('refY', 5).attr('markerWidth', 6).attr('markerHeight', 6).attr('orient','auto')
    .append('path').attr('d','M 0 1 L 10 5 L 0 9 Z').attr('fill', 'var(--orange)').attr('opacity', 0.8);

  const link = g.append('g').selectAll('line').data(links).join('line')
    .attr('stroke', d => d.type === 'supersedes' ? 'var(--orange)' : (d.type === 'distilled_into' ? '#51d4c8' : '#3de0ff'))
    .attr('stroke-width', d => {
      if (d.type === 'supersedes') return 1.8;
      if (d.type === 'distilled_into') return 1.5;
      const w = Math.max(0, Math.min(1, Number(d.weight) || 0.4));
      return 0.8 + w * 2.5;
    })
    .attr('stroke-opacity', d => edgeOpacity(d, nodeById))
    .attr('stroke-linecap', 'round')
    .attr('marker-end', d => d.type === 'supersedes' ? 'url(#arrow-sup)' : null);

  const node = g.append('g').selectAll('g').data(nodes).join('g')
    .style('cursor', 'pointer')
    .attr('opacity', d => d.status === 'superseded' ? 0.55 : 1)
    .call(GraphBoot.makeDrag(() => simulation))
    .on('click', (event, d) => { event.stopPropagation(); showDetails(d); });

  // outer glow disc (soft synapse halo) with subtle pulse
  node.append('circle')
    .attr('r', d => nodeRadius(d) + 3)
    .attr('fill', d => d._hueCol)
    .attr('opacity', d => 0.08 + retainOf(d) * 0.18)
    .attr('filter', d => `url(#${d._glowId})`)
    .attr('class', 'pulse-glow');

  // glass body
  node.append('circle')
    .attr('r', nodeRadius)
    .attr('fill', d => `url(#${d._glassId})`)
    .attr('stroke', d => d.pinned ? '#ffffff' : d._hueCol)
    .attr('stroke-width', d => d.pinned ? 1.8 : 0.9)
    .attr('stroke-opacity', d => d.pinned ? 0.7 : 0.35 + retainOf(d) * 0.4)
    .attr('stroke-dasharray', d =>
              (d.type === 'memory' && (d.status === 'superseded' || d.is_tip === false)) ? '3,2' : null
            );

  // quiet rim arcs (factor scores) — outside body, low opacity
  node.each(function(d) {
    const sc = d.scores || {};
    const r = nodeRadius(d) + 4;
    const gArc = d3.select(this);
    let a0 = -Math.PI / 2;
    const slice = (Math.PI * 2) / ARC_KEYS.length;
    ARC_KEYS.forEach((k, i) => {
      const v = Math.max(0, Math.min(1, Number(sc[k] || 0)));
      const a1 = a0 + v * slice * 0.92;
      if (v > 0.04) {
        const arc = d3.arc()
          .innerRadius(r).outerRadius(r + 1.6)
          .startAngle(a0 + Math.PI/2).endAngle(a1 + Math.PI/2);
        gArc.append('path')
          .attr('d', arc())
          .attr('fill', ARC_COLORS[i])
          .attr('opacity', 0.55);
      }
      a0 += slice;
    });
  });

  node.append('text').attr('class','node-label')
    .attr('dy', d => nodeRadius(d) + 11)
    .attr('text-anchor','middle')
    .text(d => {
      const t = d.label || d.id;
      return t.length > 20 ? t.slice(0, 18) + '…' : t;
    });

  // Neural net initial spread: layered rings by degree, minimal center pull
  const centerX = w * 0.5, centerY = h * 0.5;
  const degById = new Map();
  links.forEach(e => {
    const s = typeof e.source === 'object' ? e.source.id : e.source;
    const t = typeof e.target === 'object' ? e.target.id : e.target;
    degById.set(s, (degById.get(s) || 0) + 1);
    degById.set(t, (degById.get(t) || 0) + 1);
  });
  nodes.forEach((n, i) => {
    if (n.id === userEntityId) {
      n.x = centerX; n.y = centerY; n.fx = centerX; n.fy = centerY;
      return;
    }
    if (n.x != null && n.y != null) return;
    const deg = degById.get(n.id) || 0;
    // High-degree nodes near center (hubs), isolates in outer rings
    const ringIdx = deg >= 5 ? 0 : deg >= 2 ? 1 : 2;
    const baseR = [0.15, 0.45, 0.75][ringIdx];
    const jitter = (Math.random() - 0.5) * 0.12;
    const ang = (i * 2.399963) + jitter;
    const scale = w * 0.42;
    n.x = centerX + Math.cos(ang) * baseR * scale;
    n.y = centerY + Math.sin(ang) * baseR * scale;
  });

  simulation = GraphBoot.makeSimulation(nodes, links, {
    w,
    h,
    // Stronger repulsion + longer links = neural net spread
    charge: d => ((degById.get(d.id) || 0) === 0 ? -500 : -350),
    centerStrength: 0.01,
    nodeRadius,
    collisionPadding: 16,
    linkDistance: d => {
      if (d.type === 'mentions') return 140;
      if (d.type === 'supersedes') return 180;
      if (d.type === 'grounded_in' || d.type === 'practiced_in') return 150;
      if (d.type === 'distilled_into') return 160;
      return 120;
    },
    linkStrength: d => {
      if (d.type === 'mentions') return 0.35;
      if (d.type === 'supersedes') return 0.20;
      if (d.type === 'distilled_into') return 0.20;
      return 0.25;
    },
  })
    .force('x', d3.forceX(centerX).strength(d => {
      const deg = degById.get(d.id) || 0;
      if (d.id === userEntityId) return 0.15;
      return deg === 0 ? 0.001 : 0.008;
    }))
    .force('y', d3.forceY(centerY).strength(d => {
      const deg = degById.get(d.id) || 0;
      if (d.id === userEntityId) return 0.15;
      return deg === 0 ? 0.001 : 0.008;
    }))
    .alphaDecay(0.012)
    .on('tick', () => {
      link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
          .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
      node.attr('transform', d => `translate(${d.x},${d.y})`);
    });

  // Give the simulation a short burst so sparse graphs settle open instead of blobbing
  simulation.alpha(1).restart();

  svg.on('click', () => { document.getElementById('details').style.display = 'none'; });
}

document.getElementById('refresh').onclick = loadGraph;
document.getElementById('apply').onclick = loadGraph;
document.getElementById('apply-filters').onclick = () => render();
document.getElementById('search-btn').onclick = runSearch;
document.getElementById('search-q').addEventListener('keydown', (e) => { if (e.key === 'Enter') runSearch(); });
document.getElementById('export').onclick = () => {
  const blob = new Blob([JSON.stringify(graph, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'neural-memory.json';
  a.click();
};
document.getElementById('zoom-in').onclick = () => d3.select('#canvas').transition().call(zoomBehavior.scaleBy, 1.3);
document.getElementById('zoom-out').onclick = () => d3.select('#canvas').transition().call(zoomBehavior.scaleBy, 0.7);
document.getElementById('zoom-fit').onclick = () => {
  d3.select('#canvas').transition().duration(400).call(zoomBehavior.transform, d3.zoomIdentity);
};

loadGraph();
