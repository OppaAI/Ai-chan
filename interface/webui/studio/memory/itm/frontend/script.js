const API = location.pathname.replace(/\/+$/, '') + '/api';

let currentView = 'episodes';

function qs() {
  const p = new URLSearchParams();
  const uid = document.getElementById('user-id').value.trim();
  if (uid) p.set('user_id', uid);
  const stage = document.getElementById('stage').value;
  if (stage !== 'all') p.set('stage', stage);
  const df = document.getElementById('date-from').value;
  const dt = document.getElementById('date-to').value;
  if (df) p.set('date_from', df);
  if (dt) p.set('date_to', dt);
  const q = document.getElementById('q').value.trim();
  if (q) p.set('q', q);
  p.set('limit', '200');
  return p.toString();
}

function esc(s) {
  const d = document.createElement('div');
  d.textContent = s == null ? '' : String(s);
  return d.innerHTML;
}

function fmtDate(ts) {
  if (!ts) return '—';
  const d = new Date(ts);
  if (isNaN(d.getTime())) return ts.slice(0, 19).replace('T', ' ');
  return d.toLocaleString();
}

function valBadge(tag) {
  const t = (tag || 'neutral').toLowerCase();
  const cls = t === 'pos' ? 'val-pos' : t === 'neg' ? 'val-neg' : 'val-neu';
  return `<span class="badge ${cls}">${esc(t)}</span>`;
}

async function loadPipeline() {
  try {
    const p = new URLSearchParams();
    const uid = document.getElementById('user-id').value.trim();
    if (uid) p.set('user_id', uid);
    const r = await fetch(`${API}/pipeline?` + p.toString());
    const d = await r.json();
    if (!d.ok) { return; }
    document.getElementById('pipe-staging').textContent = d.staging ?? 0;
    document.getElementById('pipe-storage').textContent = d.storage ?? 0;
    document.getElementById('pipe-distilled').textContent = d.distilled ?? 0;
    document.getElementById('pipe-total').textContent =
      `recalled ${d.recalled_total ?? 0}× across ${d.storage ?? 0} episodes`;
  } catch (e) {
    document.getElementById('pipe-total').textContent = 'pipeline unavailable';
  }
}

function renderTimeline(episodes) {
  const tl = document.getElementById('timeline');
  document.getElementById('empty').style.display = episodes.length ? 'none' : 'block';
  tl.querySelectorAll('.ep-card').forEach(n => n.remove());
  for (const ep of episodes) {
    const card = document.createElement('div');
    card.className = 'ep-card ' + ep.stage;
    card.dataset.id = ep.id;
    const ents = (ep.entities || []).map(e => `<span class="ent">${esc(e)}</span>`).join('');
    const meta = [];
    meta.push(fmtDate(ep.timestamp));
    if (ep.stage === 'distilled') {
      meta.push('<span class="tag distilled">distilled</span>');
    } else if (ep.stage === 'staging') {
      meta.push('<span class="tag staging">staging</span>');
    }
    if (ep.recall_count) meta.push(`recalled ${ep.recall_count}×`);
    if (ep.last_recalled_at) meta.push('last ' + fmtDate(ep.last_recalled_at));
    card.innerHTML = `
      <div class="ep-head">
        <span class="ep-date">${esc(meta.join(' · '))}</span>
        ${valBadge(ep.valence_tag)}
      </div>
      <div class="ep-trace">${esc(ep.trace)}</div>
      <div class="ep-foot">
        <span class="ents">${ents || '<span class="muted">no entities</span>'}</span>
        <span class="muted">id ${ep.id}${ep.session_id ? ' · ' + esc(ep.session_id) : ''}</span>
      </div>
    `;
    card.addEventListener('click', () => openDetail(ep.id));
    tl.appendChild(card);
  }
}

async function loadEpisodes() {
  document.getElementById('status').textContent = 'Loading…';
  currentView = 'episodes';
  try {
    const r = await fetch(`${API}/episodes?` + qs());
    const d = await r.json();
    if (!d.ok) { document.getElementById('status').textContent = d.error || 'load failed'; return; }
    renderTimeline(d.episodes || []);
    document.getElementById('status').textContent = `${d.count} episodes`;
  } catch (e) {
    document.getElementById('status').textContent = 'Load failed';
  }
}

async function loadStaging() {
  document.getElementById('status').textContent = 'Loading staging…';
  currentView = 'staging';
  try {
    const p = new URLSearchParams();
    const uid = document.getElementById('user-id').value.trim();
    if (uid) p.set('user_id', uid);
    p.set('limit', '200');
    const r = await fetch(`${API}/staging?` + p.toString());
    const d = await r.json();
    if (!d.ok) { document.getElementById('status').textContent = d.error || 'load failed'; return; }
    renderTimeline(d.episodes || []);
    document.getElementById('status').textContent = `${d.count} staged`;
  } catch (e) {
    document.getElementById('status').textContent = 'Load failed';
  }
}

function closeDetail() {
  const dv = document.getElementById('detail-overlay');
  if (dv) dv.remove();
}

async function openDetail(id) {
  try {
    const p = new URLSearchParams();
    const uid = document.getElementById('user-id').value.trim();
    if (uid) p.set('user_id', uid);
    const r = await fetch(`${API}/episode/${id}?` + p.toString());
    const d = await r.json();
    if (!d.ok || !d.episode) return;
    const ep = d.episode;
    const facts = (ep.distilled_facts || []).map(f =>
      `<div class="fact"><span class="fact-text">${esc(f.text)}</span><span class="muted">${esc(f.created_at || '')}</span></div>`
    ).join('') || '<div class="muted">not distilled</div>';
    const ov = document.createElement('div');
    ov.id = 'detail-overlay';
    ov.className = 'overlay';
    ov.innerHTML = `
      <div class="overlay-box">
        <div class="overlay-head">
          <h2>Episode #${ep.id}</h2>
          <button class="btn" id="detail-close">✕</button>
        </div>
        <div class="ep-trace">${esc(ep.trace)}</div>
        <div class="kv">
          ${[['timestamp', fmtDate(ep.timestamp)],
             ['date', ep.date || '—'],
             ['valence', ep.valence_tag || '—'],
             ['arousal', ep.arousal_score != null ? Number(ep.arousal_score).toFixed(3) : '—'],
             ['salience', ep.salience_score != null ? Number(ep.salience_score).toFixed(3) : '—'],
             ['recall_count', ep.recall_count ?? 0],
             ['last_recalled_at', fmtDate(ep.last_recalled_at)],
             ['source', ep.source || '—'],
             ['session_id', ep.session_id || '—'],
             ['distilled_at', fmtDate(ep.distilled_at)],
            ].map(([k, v]) => `<div class="kv-row"><span class="kv-k">${k}</span><span class="kv-v">${esc(v)}</span></div>`).join('')}
        </div>
        <h3>Entities</h3>
        <div class="ents">${(ep.entities || []).map(e => `<span class="ent">${esc(e)}</span>`).join('') || '<span class="muted">none</span>'}</div>
        <h3>Distilled into (EM → SM)</h3>
        ${facts}
      </div>
    `;
    ov.addEventListener('click', (e) => { if (e.target === ov) closeDetail(); });
    document.body.appendChild(ov);
    document.getElementById('detail-close').onclick = closeDetail;
  } catch (e) {
    console.error(e);
  }
}

document.getElementById('refresh').onclick = () => { loadPipeline(); currentView === 'staging' ? loadStaging() : loadEpisodes(); };
document.getElementById('apply').onclick = () => { loadPipeline(); loadEpisodes(); };
document.getElementById('show-staging').onclick = loadStaging;

loadPipeline();
loadEpisodes();