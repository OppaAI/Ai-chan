(function () {
  'use strict';

  const API_BASE = (typeof GraphBoot !== 'undefined' && GraphBoot.apiBase)
    ? GraphBoot.apiBase()
    : (location.pathname.replace(/\/+$/, '') + '/api');

  let workflows = [];
  let selectedId = null;
  let currentGraph = null;

  const els = {
    list: document.getElementById('workflow-list'),
    graphs: document.getElementById('graph-list'),
    editor: document.getElementById('spec-editor'),
    message: document.getElementById('message'),
    source: document.getElementById('spec-source'),
    editorTitle: document.getElementById('editor-title'),
    graphMeta: document.getElementById('graph-meta'),
    canvas: document.getElementById('canvas'),
    detail: document.getElementById('node-detail'),
    status: document.getElementById('status'),
  };

  function setMessage(text, kind) {
    els.message.textContent = text || '';
    els.message.className = 'message' + (kind ? ' ' + kind : '');
  }

  async function api(path, opts) {
    const resp = await fetch(API_BASE + path, {
      headers: { 'Content-Type': 'application/json', ...(opts && opts.headers) },
      ...opts,
    });
    const text = await resp.text();
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch (_) { data = { raw: text }; }
    if (!resp.ok) {
      const detail = (data && (data.detail || data.error)) || resp.statusText;
      throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
    }
    return data;
  }

  function renderWorkflowList() {
    els.list.innerHTML = '';
    workflows.forEach(function (w) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'list-item' + (w.id === selectedId ? ' active' : '');
      if (w.id === selectedId) {
        btn.setAttribute('aria-current', 'true');
      }
      btn.innerHTML =
        '<div class="title">' + escapeHtml(w.name || w.id) + '</div>' +
        '<div class="meta">' + escapeHtml(w.id) + ' · ' +
        escapeHtml(w.spec_source || 'no config') + '</div>';
      btn.addEventListener('click', function () { selectWorkflow(w.id); });
      els.list.appendChild(btn);
    });
  }

  function renderGraphIds(ids) {
    els.graphs.innerHTML = '';
    (ids || []).forEach(function (id) {
      const div = document.createElement('div');
      div.className = 'list-item';
      div.innerHTML = '<div class="title">' + escapeHtml(id) + '</div>';
      els.graphs.appendChild(div);
    });
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  async function loadWorkflows() {
    const data = await api('/workflows');
    workflows = data.workflows || [];
    renderWorkflowList();
    renderGraphIds(data.registered_graphs || []);
    els.status.textContent = 'Layer 4 · ' + workflows.length + ' workflows';
  }

  async function selectWorkflow(id) {
    const requestedId = id;
    selectedId = id;
    renderWorkflowList();
    setMessage('Loading Spec…');
    try {
      const data = await api('/workflows/' + encodeURIComponent(requestedId) + '/spec');
      // Only apply response if this workflow is still selected
      if (selectedId !== requestedId) return;
      els.editor.value = JSON.stringify(data.spec, null, 2);
      els.source.textContent = data.source || '—';
      els.editorTitle.textContent = 'Spec · ' + requestedId;
      setMessage('Loaded from ' + (data.source || 'unknown'), 'ok');
      await preview();
    } catch (err) {
      // Only show error if this workflow is still selected
      if (selectedId !== requestedId) return;
      setMessage(String(err.message || err), 'err');
    }
  }

  function parseEditorSpec() {
    const text = els.editor.value.trim();
    if (!text) throw new Error('Spec editor is empty');
    const parsed = JSON.parse(text);
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      throw new Error('Spec must be a JSON object');
    }
    return parsed;
  }

  async function validate() {
    try {
      const spec = parseEditorSpec();
      const data = await api('/validate', {
        method: 'POST',
        body: JSON.stringify({ spec: spec }),
      });
      if (!data.ok) {
        setMessage('Invalid: ' + (data.error || 'unknown'), 'err');
        return;
      }
      els.editor.value = JSON.stringify(data.spec, null, 2);
      setMessage('Spec valid (normalized)', 'ok');
    } catch (err) {
      setMessage(String(err.message || err), 'err');
    }
  }

  async function preview() {
    try {
      const spec = parseEditorSpec();
      const data = await api('/preview', {
        method: 'POST',
        body: JSON.stringify({ spec: spec }),
      });
      currentGraph = data.graph;
      els.graphMeta.textContent =
        (currentGraph.id || 'graph') + ' · ' + (currentGraph.nodes || []).length + ' nodes';
      renderGraph(currentGraph);
      setMessage('Preview OK · source=' + (currentGraph.source || '—'), 'ok');
    } catch (err) {
      setMessage(String(err.message || err), 'err');
    }
  }

  async function save() {
    if (!selectedId) {
      setMessage('Select a workflow first', 'err');
      return;
    }
    const saveId = selectedId;
    try {
      const spec = parseEditorSpec();
      const data = await api('/workflows/' + encodeURIComponent(saveId) + '/spec', {
        method: 'PUT',
        body: JSON.stringify({ spec: spec }),
      });
      // Only apply response if the same workflow is still selected
      if (selectedId !== saveId) return;
      els.editor.value = JSON.stringify(data.spec, null, 2);
      els.source.textContent = 'spec.json';
      setMessage('Saved ' + (data.path || 'spec.json'), 'ok');
      await loadWorkflows();
      renderWorkflowList();
    } catch (err) {
      // Only show error if the same workflow is still selected
      if (selectedId !== saveId) return;
      setMessage(String(err.message || err), 'err');
    }
  }

  function renderGraph(graph) {
    const svg = d3.select('#canvas');
    svg.selectAll('*').remove();
    els.detail.classList.add('hidden');

    const nodes = (graph && graph.nodes) || [];
    if (!nodes.length) return;

    const rect = els.canvas.getBoundingClientRect();
    const w = Math.max(320, rect.width || 800);
    const h = Math.max(200, rect.height || 320);
    svg.attr('viewBox', '0 0 ' + w + ' ' + h);

    const NODE_W = 160;
    const NODE_H = 48;
    const gapX = 36;
    const totalW = nodes.length * NODE_W + (nodes.length - 1) * gapX;
    const startX = Math.max(24, (w - totalW) / 2);
    const y = h / 2 - NODE_H / 2;

    const positions = {};
    nodes.forEach(function (n, i) {
      positions[n.id] = { x: startX + i * (NODE_W + gapX), y: y };
    });

    const gRoot = svg.append('g');

    // edges
    const edges = graph.edges || [];
    edges.forEach(function (e) {
      const a = positions[e.source];
      const b = positions[e.target];
      if (!a || !b) return;
      gRoot.append('path')
        .attr('class', 'edge-line')
        .attr('d',
          'M' + (a.x + NODE_W) + ',' + (a.y + NODE_H / 2) +
          ' C' + (a.x + NODE_W + 30) + ',' + (a.y + NODE_H / 2) +
          ' ' + (b.x - 30) + ',' + (b.y + NODE_H / 2) +
          ' ' + b.x + ',' + (b.y + NODE_H / 2)
        );
    });

    // nodes
    nodes.forEach(function (n) {
      const p = positions[n.id];
      const g = gRoot.append('g')
        .attr('transform', 'translate(' + p.x + ',' + p.y + ')')
        .style('cursor', 'pointer')
        .on('click', function () {
          gRoot.selectAll('.node-box').classed('selected', false);
          g.select('.node-box').classed('selected', true);
          els.detail.classList.remove('hidden');
          els.detail.textContent = JSON.stringify(
            { id: n.id, tool: n.tool, depends_on: n.depends_on, args: n.args },
            null,
            2
          );
        });
      g.append('rect')
        .attr('class', 'node-box')
        .attr('width', NODE_W)
        .attr('height', NODE_H);
      g.append('text')
        .attr('class', 'node-label')
        .attr('x', 12)
        .attr('y', 20)
        .text(n.id);
      g.append('text')
        .attr('class', 'node-sub')
        .attr('x', 12)
        .attr('y', 36)
        .text(n.tool || '');
    });
  }

  document.getElementById('btn-reload').addEventListener('click', function () {
    loadWorkflows().then(function () {
      if (selectedId) return selectWorkflow(selectedId);
      setMessage('Workflows reloaded', 'ok');
    }).catch(function (err) { setMessage(String(err.message || err), 'err'); });
  });
  document.getElementById('btn-validate').addEventListener('click', validate);
  document.getElementById('btn-preview').addEventListener('click', preview);
  document.getElementById('btn-save').addEventListener('click', save);

  window.addEventListener('resize', function () {
    if (currentGraph) renderGraph(currentGraph);
  });

  loadWorkflows()
    .then(function () {
      if (workflows.length) return selectWorkflow(workflows[0].id);
      setMessage('No Spec-backed workflows found');
    })
    .catch(function (err) {
      setMessage('Failed to load workflows: ' + (err.message || err), 'err');
    });
})();
