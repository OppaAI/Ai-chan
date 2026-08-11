const SIDE = 60;
        const NODE_W = 180;
        const NODE_H = 52;
        const PORT_R = 4;
        const PORT_HIT_R = 10;
        const LAYER_GAP = 200;
        const NODE_GAP = 80;

        let currentPlaybooks = [];
        let selectedPlaybook = null;
        let selectedNodeId = null;
        let selectedEdgeKey = null;
        let transform = d3.zoomIdentity;
        let currentZoom = null;

        // Visual editor state
        let dragEdgeState = null; // { sourceId, x1, y1 }

        // Detect base path for API calls via the shared studio bootstrap
        const API_BASE = GraphBoot.apiBase();

        function uid() {
            return 'node_' + Math.random().toString(36).slice(2, 9);
        }

        function escapeHTML(str) {
            const d = document.createElement('div');
            d.textContent = str || '';
            return d.innerHTML;
        }

        async function fetchPlaybooks() {
            try {
                const resp = await fetch(`${API_BASE}/playbooks`);
                currentPlaybooks = await resp.json();

                // Reconcile selectedPlaybook with refreshed data
                if (selectedPlaybook) {
                    const refreshed = currentPlaybooks.find(pb => pb.id === selectedPlaybook.id);
                    if (refreshed) {
                        selectedPlaybook = refreshed;
                        ensureLayout(selectedPlaybook);
                        renderGraph();
                    } else {
                        // Playbook no longer exists, clear selection
                        selectedPlaybook = null;
                        selectedNodeId = null;
                        selectedEdgeKey = null;
                        const svg = d3.select('#canvas');
                        svg.selectAll('*').remove();
                        hideDetails();
                    }
                }

                renderPlaybooksList();
            } catch (err) {
                console.error('Failed to fetch playbooks:', err);
                document.getElementById('playbooks-list').innerHTML =
                    '<div style="color:var(--pink);font-size:11px">Failed to load playbooks</div>';
            }
        }

        function renderPlaybooksList() {
            const container = document.getElementById('playbooks-list');
            container.innerHTML = '';
            if (!currentPlaybooks.length) {
                container.innerHTML = '<div style="color:var(--dim);font-size:11px">No playbooks available</div>';
                return;
            }
            currentPlaybooks.forEach(pb => {
                const nodeCount = Array.isArray(pb.nodes) ? pb.nodes.length : 0;
                const button = document.createElement('button');
                const _specBacked = (pb.id === 'gen_job_post' || pb.id === 'aurora_forecast');
                button.className = 'playbook-item' + (selectedPlaybook && selectedPlaybook.id === pb.id ? ' active' : '') + (_specBacked ? ' spec-backed' : '');
                button.dataset.id = pb.id || '';
                button.innerHTML = `
                    <div class="pb-name">${escapeHTML(pb.name || pb.id)}</div>
                    <div class="pb-id">${escapeHTML(pb.id)}</div>
                    <div class="pb-meta"><span>${nodeCount} nodes</span></div>
                `;
                button.onclick = () => selectPlaybook(pb);
                container.appendChild(button);
            });
        }

        function selectPlaybook(playbook) {
            selectedPlaybook = playbook;
            selectedNodeId = null;
            selectedEdgeKey = null;
            ensureLayout(playbook);
            renderPlaybooksList();
            renderGraph();
            hideDetails();
            updateToolbar();
            if (typeof loadSpecForPlaybook === 'function') loadSpecForPlaybook(playbook);
        }

        function ensureLayout(playbook) {
            if (!playbook || !playbook.nodes) return;
            const needsLayout = playbook.nodes.some(n => !n._layout);
            if (needsLayout) {
                const positions = computeLayout(playbook.nodes, playbook.edges);
                playbook.nodes.forEach(n => {
                    if (!n._layout) n._layout = positions[n.id] || { x: 300, y: 300 };
                });
            }
        }

        function edgeTypeColor(edge) {
            const t = edge.type || 'depends_on';
            if (t === 'fallback_to') return 'fallback';
            if (t === 'loop_to') return 'loop';
            return 'depends';
        }

        function computeLayout(nodes, edges) {
            if (!nodes.length) return {};
            const adj = {};
            const revAdj = {};
            nodes.forEach(n => { adj[n.id] = []; revAdj[n.id] = []; });
            edges.forEach(e => {
                const sid = typeof e.source === 'string' ? e.source : e.source?.id;
                const tid = typeof e.target === 'string' ? e.target : e.target?.id;
                if (sid && tid && adj[sid] && revAdj[tid]) {
                    adj[sid].push(tid);
                    revAdj[tid].push(sid);
                }
            });

            const levels = {};
            const visited = new Set();
            function assignLevel(id, level) {
                if (visited.has(id)) return;
                visited.add(id);
                levels[id] = level;
                (adj[id] || []).forEach(child => assignLevel(child, level + 1));
            }
            const roots = nodes.filter(n => !(revAdj[n.id] || []).length);
            if (!roots.length) {
                roots.push(nodes[0]);
                assignLevel(nodes[0].id, 0);
            }
            roots.forEach(r => assignLevel(r.id, 0));

            const unvisited = nodes.filter(n => levels[n.id] === undefined);
            let maxLevel = Math.max(0, ...Object.values(levels));
            unvisited.forEach(n => { levels[n.id] = ++maxLevel; });

            const levelNodes = {};
            Object.entries(levels).forEach(([id, lv]) => {
                if (!levelNodes[lv]) levelNodes[lv] = [];
                levelNodes[lv].push(id);
            });
            const maxNodesInLevel = Math.max(1, ...Object.values(levelNodes).map(g => g.length));
            const maxWidth = Math.min(maxNodesInLevel * (NODE_W + NODE_GAP), 1400);

            const positions = {};
            const startY = 60;
            const startX = 80;
            Object.entries(levelNodes).forEach(([lv, ids]) => {
                const level = parseInt(lv);
                const rowH = NODE_H + 60;
                const totalW = (ids.length - 1) * (NODE_W + NODE_GAP);
                const offsetX = (maxWidth - totalW) / 2;
                ids.forEach((id, i) => {
                    positions[id] = {
                        x: startX + offsetX + i * (NODE_W + NODE_GAP),
                        y: startY + level * rowH
                    };
                });
            });

            return positions;
        }

        function getPortPos(node, side) {
            const pos = node._layout || { x: 0, y: 0 };
            if (side === 'left') return { x: pos.x, y: pos.y + NODE_H / 2 };
            if (side === 'right') return { x: pos.x + NODE_W, y: pos.y + NODE_H / 2 };
            return pos;
        }

        function makeEdgeKey(e) {
            const s = typeof e.source === 'string' ? e.source : e.source?.id;
            const t = typeof e.target === 'string' ? e.target : e.target?.id;
            return `${s}|${t}|${e.type || 'depends_on'}`;
        }

        function renderGraph() {
            const svg = d3.select('#canvas');
            svg.selectAll('*').remove();
            const container = document.getElementById('canvas-area');
            const width = container.clientWidth || 1200;
            const height = container.clientHeight || 800;
            svg.attr('viewBox', `0 0 ${width} ${height}`);

            if (!selectedPlaybook || !selectedPlaybook.nodes?.length) {
                svg.append('text')
                    .attr('x', width / 2).attr('y', height / 2)
                    .attr('text-anchor', 'middle')
                    .attr('fill', 'var(--dim)')
                    .attr('font-size', '14px')
                    .text('No graph data available');
                document.getElementById('graph-info').textContent = 'No data';
                return;
            }

            const nodes = selectedPlaybook.nodes;
            const edges = selectedPlaybook.edges || [];

            // Hide the canvas temporarily to prevent visual jump/flicker
            svg.style('opacity', '0');

            const g = svg.append('g');

            // Zoom
            currentZoom = GraphBoot.makeZoom({
                scaleExtent: [0.3, 3],
                target: g,
                onZoom: (event) => { transform = event.transform; },
            });
            svg.call(currentZoom);

            // Compute center transform synchronously
            if (nodes.length > 0) {
                const minX = Math.min(...nodes.map(n => n._layout.x));
                const minY = Math.min(...nodes.map(n => n._layout.y));
                const maxX = Math.max(...nodes.map(n => n._layout.x + NODE_W));
                const maxY = Math.max(...nodes.map(n => n._layout.y + NODE_H));

                const graphW = maxX - minX;
                const graphH = maxY - minY;
                const scale = Math.min(width / (graphW + 120), height / (graphH + 120), 1.2);
                const tx = (width - graphW * scale) / 2 - minX * scale;
                const ty = (height - graphH * scale) / 2 - minY * scale;

                // Apply zoom transform instantly (no transition)
                svg.call(currentZoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
            }

            // Grid
            const defs = svg.append('defs');
            const pattern = defs.append('pattern')
                .attr('id', 'grid')
                .attr('width', 40).attr('height', 40)
                .attr('patternUnits', 'userSpaceOnUse');
            pattern.append('path')
                .attr('d', 'M 40 0 L 0 0 0 40')
                .attr('fill', 'none')
                .attr('stroke', 'var(--dimmer)')
                .attr('stroke-width', 0.5)
                .attr('opacity', 0.5);
            g.append('rect')
                .attr('x', -2000).attr('y', -2000)
                .attr('width', 6000).attr('height', 6000)
                .attr('fill', 'url(#grid)');

            // Arrow markers
            ['depends', 'loop', 'fallback'].forEach(type => {
                const color = type === 'loop' ? 'var(--pink)' : type === 'fallback' ? '#e8843a' : 'var(--dim)';
                defs.append('marker')
                    .attr('id', `arrow-${type}`)
                    .attr('viewBox', '0 0 10 10')
                    .attr('refX', 8).attr('refY', 5)
                    .attr('markerWidth', 6).attr('markerHeight', 6)
                    .attr('orient', 'auto')
                    .append('path')
                    .attr('d', 'M 0 1 L 10 5 L 0 9 Z')
                    .attr('fill', color);
            });

            const tooltip = document.getElementById('tooltip');
            const edgeGroup = g.append('g').attr('class', 'edges');
            const nodeGroup = g.append('g').attr('class', 'nodes');
            const dragGroup = g.append('g').attr('class', 'drag-layer');

            // ── Draw edges ─────────────────────────────────────────────────
            edges.forEach(edge => {
                const srcId = typeof edge.source === 'string' ? edge.source : edge.source?.id;
                const tgtId = typeof edge.target === 'string' ? edge.target : edge.target?.id;
                const src = nodes.find(n => n.id === srcId);
                const tgt = nodes.find(n => n.id === tgtId);
                if (!src || !tgt) return;

                const sPos = src._layout;
                const tPos = tgt._layout;
                const x1 = sPos.x + NODE_W;
                const y1 = sPos.y + NODE_H / 2;
                const x2 = tPos.x;
                const y2 = tPos.y + NODE_H / 2;

                const edgeType = edge.type || 'depends_on';
                let edgeClass = 'edge-depends';
                let marker = 'url(#arrow-depends)';
                if (edgeType === 'fallback_to') {
                    edgeClass = 'edge-fallback';
                    marker = 'url(#arrow-fallback)';
                } else if (edgeType === 'loop_to') {
                    edgeClass = 'edge-loop';
                    marker = 'url(#arrow-loop)';
                }

                const eKey = makeEdgeKey(edge);
                if (selectedEdgeKey === eKey) {
                    edgeClass += ' edge-selected';
                }

                const dx = x2 - x1;
                const dy = y2 - y1;
                const dist = Math.sqrt(dx * dx + dy * dy);
                const curvature = Math.min(dist * 0.15, 40);

                const pathD = `M ${x1} ${y1} C ${x1 + curvature} ${y1}, ${x2 - curvature} ${y2}, ${x2} ${y2}`;

                edgeGroup.append('path')
                    .attr('d', pathD)
                    .attr('class', edgeClass)
                    .attr('marker-end', marker)
                    .attr('data-key', eKey)
                    .style('cursor', 'pointer')
                    .on('click', (event) => {
                        event.stopPropagation();
                        selectEdge(edge);
                    })
                    .on('mouseenter', (event) => {
                        tooltip.textContent = `${srcId} → ${tgtId}${edgeType === 'loop_to' ? ' [loop]' : edgeType === 'fallback_to' ? ' [fallback]' : ''}`;
                        tooltip.style.left = (event.pageX + 12) + 'px';
                        tooltip.style.top = (event.pageY - 20) + 'px';
                        tooltip.classList.add('visible');
                        d3.select(event.target).classed('edge-highlight', true);
                    })
                    .on('mouseleave', () => {
                        tooltip.classList.remove('visible');
                        d3.select(event.target).classed('edge-highlight', false);
                    });
            });

            // ── Draw nodes ─────────────────────────────────────────────────
            const nodeDrag = d3.drag()
                .on('start', function(event, d) {
                    d3.select(this).select('.node-card').classed('dragging', true);
                })
                .on('drag', function(event, d) {
                    d._layout.x = event.x;
                    d._layout.y = event.y;
                    d3.select(this).attr('transform', `translate(${d._layout.x}, ${d._layout.y})`);
                    // Update connected edge paths in-place for smooth dragging
                    const edgesArr = selectedPlaybook.edges || [];
                    edgesArr.forEach(e => {
                        const s = typeof e.source === 'string' ? e.source : e.source?.id;
                        const t = typeof e.target === 'string' ? e.target : e.target?.id;
                        if (s !== d.id && t !== d.id) return;
                        const srcNode = nodes.find(n => n.id === s);
                        const tgtNode = nodes.find(n => n.id === t);
                        if (!srcNode || !tgtNode) return;
                        const sx = srcNode._layout.x + NODE_W;
                        const sy = srcNode._layout.y + NODE_H / 2;
                        const tx = tgtNode._layout.x;
                        const ty = tgtNode._layout.y + NODE_H / 2;
                        const dx = tx - sx, dy = ty - sy;
                        const dist = Math.sqrt(dx * dx + dy * dy);
                        const curvature = Math.min(dist * 0.15, 40);
                        const newD = `M ${sx} ${sy} C ${sx + curvature} ${sy}, ${tx - curvature} ${ty}, ${tx} ${ty}`;
                        const eKey = makeEdgeKey(e);
                        edgeGroup.select(`path[data-key="${eKey}"]`).attr('d', newD);
                    });
                })
                .on('end', function() {
                    d3.select(this).select('.node-card').classed('dragging', false);
                    renderGraph(); // clean re-render
                });

            nodes.forEach(node => {
                const pos = node._layout;
                const ng = nodeGroup.append('g')
                    .datum(node)
                    .attr('transform', `translate(${pos.x}, ${pos.y})`)
                    .call(nodeDrag);

                // Shadow
                ng.append('rect')
                    .attr('x', 1).attr('y', 1)
                    .attr('width', NODE_W).attr('height', NODE_H)
                    .attr('rx', 8).attr('ry', 8)
                    .attr('fill', 'rgba(0,0,0,0.25)');

                // Main card
                ng.append('rect')
                    .attr('class', 'node-card' + (selectedNodeId === node.id ? ' selected' : ''))
                    .attr('width', NODE_W).attr('height', NODE_H)
                    .attr('rx', 8).attr('ry', 8)
                    .on('click', (event) => {
                        event.stopPropagation();
                        selectNode(node);
                    })
                    .on('mouseenter', (event) => {
                        const toolName = node.tool ? node.tool.split('.').pop() : node.id;
                        tooltip.textContent = `${node.id}: ${toolName}`;
                        tooltip.style.left = '0px';
                        tooltip.style.top = '-32px';
                        tooltip.classList.add('visible');
                    })
                    .on('mouseleave', () => {
                        tooltip.classList.remove('visible');
                    });

                // Inner highlight (top edge sheen)
                ng.append('rect')
                    .attr('x', 1).attr('y', 1)
                    .attr('width', NODE_W - 2).attr('height', 1)
                    .attr('rx', 1)
                    .attr('fill', 'rgba(168,136,232,0.08)')
                    .style('pointer-events', 'none');

                const isEntry = !edges.some(e => {
                    const tid = typeof e.target === 'string' ? e.target : e.target?.id;
                    return tid === node.id;
                });
                const hasOutgoing = edges.some(e => {
                    const sid = typeof e.source === 'string' ? e.source : e.source?.id;
                    return sid === node.id;
                });
                const isTerminal = !hasOutgoing;

                // Entry badge (top-left)
                if (isEntry) {
                    ng.append('rect')
                        .attr('class', 'node-badge entry')
                        .attr('x', 4).attr('y', 3)
                        .attr('width', 44).attr('height', 14)
                        .attr('rx', 2);
                    ng.append('text')
                        .attr('x', 26).attr('y', 13)
                        .attr('text-anchor', 'middle')
                        .attr('font-size', '7px')
                        .attr('font-family', 'sans-serif')
                        .attr('fill', 'var(--cyan)')
                        .attr('font-weight', '600')
                        .attr('letter-spacing', '0.05em')
                        .text('ENTRY')
                        .style('pointer-events', 'none');
                }

                // Terminal badge (top-right)
                if (isTerminal) {
                    ng.append('rect')
                        .attr('class', 'node-badge terminal')
                        .attr('x', NODE_W - 48).attr('y', 3)
                        .attr('width', 44).attr('height', 14)
                        .attr('rx', 2);
                    ng.append('text')
                        .attr('x', NODE_W - 26).attr('y', 13)
                        .attr('text-anchor', 'middle')
                        .attr('font-size', '7px')
                        .attr('font-family', 'sans-serif')
                        .attr('fill', 'var(--pink)')
                        .attr('font-weight', '600')
                        .attr('letter-spacing', '0.05em')
                        .text('TERMINAL')
                        .style('pointer-events', 'none');
                }

                // Status dot
                ng.append('circle')
                    .attr('class', 'node-status-dot')
                    .attr('cx', isEntry ? NODE_W - 12 : (isTerminal ? NODE_W - 56 : NODE_W - 12))
                    .attr('cy', 12)
                    .attr('r', 3.5)
                    .attr('fill', 'var(--cyan)');

                // Tool label (line 1)
                const toolName = node.tool ? node.tool.split('.').pop() : node.id;
                ng.append('text')
                    .attr('class', 'node-tool-label')
                    .attr('x', NODE_W / 2).attr('y', NODE_H / 2 - 6)
                    .text(toolName.length > 22 ? toolName.substring(0, 20) + '…' : toolName);

                // ID label (line 2)
                ng.append('text')
                    .attr('class', 'node-id-label')
                    .attr('x', NODE_W / 2).attr('y', NODE_H / 2 + 14)
                    .text(node.id);

                // Ports — input (left)
                ng.append('circle')
                    .attr('class', 'port-circle port-input')
                    .attr('cx', 0).attr('cy', NODE_H / 2)
                    .attr('r', PORT_R);
                ng.append('circle')
                    .attr('class', 'port-hit')
                    .attr('cx', 0).attr('cy', NODE_H / 2)
                    .attr('r', PORT_HIT_R)
                    .attr('fill', 'transparent')
                    .style('cursor', 'crosshair')
                    .on('mouseup', (event) => {
                        event.stopPropagation();
                        finishDragEdge(node.id, 'input');
                    });

                // Ports — output (right)
                ng.append('circle')
                    .attr('class', 'port-circle port-output')
                    .attr('cx', NODE_W).attr('cy', NODE_H / 2)
                    .attr('r', PORT_R);
                ng.append('circle')
                    .attr('class', 'port-hit')
                    .attr('cx', NODE_W).attr('cy', NODE_H / 2)
                    .attr('r', PORT_HIT_R)
                    .attr('fill', 'transparent')
                    .style('cursor', 'crosshair')
                    .on('mousedown', (event) => {
                        event.stopPropagation();
                        startDragEdge(node.id, event);
                    });
            });

            // Temp drag line
            if (dragEdgeState) {
                dragGroup.append('path')
                    .attr('class', 'edge-drag-line')
                    .attr('d', dragEdgeState.d);
            }

            document.getElementById('graph-info').textContent =
                `${nodes.length} nodes, ${edges.length} edges`;
            const statusEl = document.getElementById('header-status');
            if (statusEl) {
                statusEl.textContent = `${nodes.length}n · ${edges.length}e`;
            }

            // Restore canvas opacity now that it is centered correctly
            svg.style('opacity', '1');
        }

        // ── Selection & Details ─────────────────────────────────────────────
        function selectNode(node) {
            selectedNodeId = node.id;
            selectedEdgeKey = null;
            renderGraph();
            showNodeDetails(node);
            updateToolbar();
        }

        function selectEdge(edge) {
            selectedEdgeKey = makeEdgeKey(edge);
            selectedNodeId = null;
            renderGraph();
            showEdgeDetails(edge);
            updateToolbar();
        }

        function clearSelection() {
            selectedNodeId = null;
            selectedEdgeKey = null;
            hideDetails();
            updateToolbar();
            renderGraph();
        }

        function updateToolbar() {
            const tb = document.getElementById('floating-toolbar');
            if (!tb) return;
            tb.style.display = (selectedNodeId || selectedEdgeKey) ? 'flex' : 'none';
        }

        function showNodeDetails(node) {
            const panel = document.getElementById('details-panel');
            const content = document.getElementById('details-content');
            panel.style.display = 'block';

            const deps = selectedPlaybook?.edges
                ?.filter(e => {
                    const tid = typeof e.target === 'string' ? e.target : e.target?.id;
                    return tid === node.id;
                }) || [];
            const dependents = selectedPlaybook?.edges
                ?.filter(e => {
                    const sid = typeof e.source === 'string' ? e.source : e.source?.id;
                    return sid === node.id;
                }) || [];

            content.innerHTML = `
                <div class="details-panel-header">
                    <h3>Node Details</h3>
                    <button class="details-close" onclick="clearSelection()">×</button>
                </div>
                <div class="detail-row"><div class="detail-label">Node ID</div><input class="detail-input" id="edit-node-id" value="${escapeHTML(node.id)}"></div>
                <div class="detail-row"><div class="detail-label">Tool</div><input class="detail-input" id="edit-node-tool" value="${escapeHTML(node.tool || '')}"></div>
                <div class="detail-row"><div class="detail-label">Args (JSON)</div><textarea class="detail-textarea" id="edit-node-args">${escapeHTML(JSON.stringify(node.args || {}, null, 2))}</textarea></div>
                <div class="detail-row"><div class="detail-label">Loop To</div><input class="detail-input" id="edit-node-loop" value="${escapeHTML(node.loop_to || '')}" placeholder="target node id or leave blank"></div>
                <div class="detail-row"><div class="detail-label">Fallback To</div><input class="detail-input" id="edit-node-fallback" value="${escapeHTML(node.fallback_to || '')}" placeholder="target node id or leave blank"></div>
                <div class="detail-row"><div class="detail-label">Max Visits</div><input class="detail-input" id="edit-node-maxvisits" type="number" value="${node.max_visits || ''}" placeholder="∞"></div>
                <div style="display:flex; gap:6px; margin-top:12px;">
                    <button class="btn btn-primary" onclick="saveNodeChanges('${escapeHTML(node.id)}')">Save</button>
                    <button class="btn btn-danger" onclick="deleteSelected()">Delete Node</button>
                </div>
                <div style="margin-top:10px; padding-top:10px; border-top:1px solid var(--border);">
                    <div class="detail-row"><div class="detail-label">Inputs (${deps.length})</div><div class="detail-value">${deps.map(d => {
                        const sid = typeof d.source === 'string' ? d.source : d.source?.id;
                        return `<code>${escapeHTML(sid)}</code>`;
                    }).join(', ') || '—'}</div></div>
                    <div class="detail-row"><div class="detail-label">Outputs (${dependents.length})</div><div class="detail-value">${dependents.map(d => {
                        const tid = typeof d.target === 'string' ? d.target : d.target?.id;
                        return `<code>${escapeHTML(tid)}</code>`;
                    }).join(', ') || '—'}</div></div>
                </div>
            `;
        }

        function showEdgeDetails(edge) {
            const panel = document.getElementById('details-panel');
            const content = document.getElementById('details-content');
            panel.style.display = 'block';

            const type = edge.type || 'depends_on';
            const typeColor = type === 'fallback_to' ? '#e8843a' : type === 'loop_to' ? 'var(--pink)' : 'var(--dim)';

            content.innerHTML = `
                <div class="details-panel-header">
                    <h3>Edge Details</h3>
                    <button class="details-close" onclick="clearSelection()">×</button>
                </div>
                <div class="detail-row"><div class="detail-label">Type</div>
                    <select class="detail-select" id="edit-edge-type">
                        <option value="depends_on" ${type === 'depends_on' ? 'selected' : ''}>depends_on</option>
                        <option value="loop_to" ${type === 'loop_to' ? 'selected' : ''}>loop_to</option>
                        <option value="fallback_to" ${type === 'fallback_to' ? 'selected' : ''}>fallback_to</option>
                    </select>
                </div>
                <div class="detail-row"><div class="detail-label">From</div><div class="detail-value"><code>${escapeHTML(edge.source)}</code></div></div>
                <div class="detail-row"><div class="detail-label">To</div><div class="detail-value"><code>${escapeHTML(edge.target)}</code></div></div>
                ${edge.tool_call ? `<div class="detail-row"><div class="detail-label">Tool Call</div><div class="detail-value"><pre class="text-xs" style="max-height:80px;overflow:auto;background:rgba(255,255,255,0.03);padding:4px;border-radius:3px">${escapeHTML(JSON.stringify(edge.tool_call, null, 2))}</pre></div></div>` : ''}
                ${edge.skill ? `<div class="detail-row"><div class="detail-label">Skill</div><div class="detail-value"><code>${escapeHTML(edge.skill)}</code></div></div>` : ''}
                <div style="display:flex; gap:6px; margin-top:12px;">
                    <button class="btn btn-primary" onclick="saveEdgeChanges('${escapeHTML(edge.source)}','${escapeHTML(edge.target)}')">Save</button>
                    <button class="btn btn-danger" onclick="deleteSelected()">Delete Edge</button>
                </div>
            `;
        }

        function hideDetails() {
            document.getElementById('details-panel').style.display = 'none';
        }

        // ── Edit Actions ──────────────────────────────────────────────────────
        function saveNodeChanges(oldId) {
            const node = selectedPlaybook.nodes.find(n => n.id === oldId);
            if (!node) return;
            const newId = document.getElementById('edit-node-id').value.trim();
            const newTool = document.getElementById('edit-node-tool').value.trim();
            const newArgs = document.getElementById('edit-node-args').value;
            const newLoop = document.getElementById('edit-node-loop').value.trim();
            const newFallback = document.getElementById('edit-node-fallback').value.trim();
            const newMax = document.getElementById('edit-node-maxvisits').value;

            if (newId && newId !== oldId) {
                if (selectedPlaybook.nodes.some(n => n.id === newId)) {
                    alert('Node ID already exists'); return;
                }
                selectedPlaybook.edges.forEach(e => {
                    if (e.source === oldId) e.source = newId;
                    if (e.target === oldId) e.target = newId;
                });
                node.id = newId;
            }
            if (newTool) node.tool = newTool;
            try { node.args = JSON.parse(newArgs); } catch { alert('Invalid JSON in Args'); return; }
            if (newLoop) node.loop_to = newLoop; else delete node.loop_to;
            if (newFallback) node.fallback_to = newFallback; else delete node.fallback_to;
            if (newMax) node.max_visits = parseInt(newMax); else delete node.max_visits;

            selectedNodeId = node.id;
            renderGraph();
            showNodeDetails(node);
        }

        function saveEdgeChanges(src, tgt) {
            const edge = selectedPlaybook.edges.find(e => e.source === src && e.target === tgt);
            if (!edge) return;
            const newType = document.getElementById('edit-edge-type').value;
            edge.type = newType;
            selectedEdgeKey = makeEdgeKey(edge);
            renderGraph();
            showEdgeDetails(edge);
        }

        function deleteSelected() {
            if (selectedNodeId) {
                selectedPlaybook.nodes = selectedPlaybook.nodes.filter(n => n.id !== selectedNodeId);
                selectedPlaybook.edges = selectedPlaybook.edges.filter(e => {
                    const s = typeof e.source === 'string' ? e.source : e.source?.id;
                    const t = typeof e.target === 'string' ? e.target : e.target?.id;
                    return s !== selectedNodeId && t !== selectedNodeId;
                });
                selectedNodeId = null;
            } else if (selectedEdgeKey) {
                const parts = selectedEdgeKey.split('|');
                const s = parts[0], t = parts[1], ty = parts[2];
                selectedPlaybook.edges = selectedPlaybook.edges.filter(e => {
                    const es = typeof e.source === 'string' ? e.source : e.source?.id;
                    const et = typeof e.target === 'string' ? e.target : e.target?.id;
                    return !(es === s && et === t && (e.type || 'depends_on') === ty);
                });
                selectedEdgeKey = null;
            }
            hideDetails();
            updateToolbar();
            renderGraph();
        }

        // ── Port-to-Port Edge Creation ───────────────────────────────────────
        function startDragEdge(sourceId, event) {
            event.preventDefault();
            const srcNode = selectedPlaybook.nodes.find(n => n.id === sourceId);
            if (!srcNode) return;
            const x1 = srcNode._layout.x + NODE_W;
            const y1 = srcNode._layout.y + NODE_H / 2;
            dragEdgeState = { sourceId, x1, y1 };

            const svg = document.getElementById('canvas');
            function onMove(ev) {
                if (!dragEdgeState) return;
                // Get SVG coordinates accounting for current zoom transform
                const pt = d3.pointer(ev, svg.querySelector('g'));
                dragEdgeState.d = `M ${dragEdgeState.x1} ${dragEdgeState.y1} L ${pt[0]} ${pt[1]}`;
                const line = d3.select('#drag-line');
                if (line.empty()) {
                    renderGraph();
                } else {
                    line.attr('d', dragEdgeState.d);
                }
            }
            function onUp() {
                document.removeEventListener('mousemove', onMove);
                document.removeEventListener('mouseup', onUp);
                if (dragEdgeState) {
                    dragEdgeState = null;
                    renderGraph();
                }
            }
            document.addEventListener('mousemove', onMove);
            document.addEventListener('mouseup', onUp);
            renderGraph();
        }

        function finishDragEdge(targetId, portSide) {
            if (!dragEdgeState || portSide !== 'input') {
                dragEdgeState = null;
                renderGraph();
                return;
            }
            const sourceId = dragEdgeState.sourceId;
            dragEdgeState = null;
            if (sourceId === targetId) {
                renderGraph();
                return;
            }
            const exists = selectedPlaybook.edges.some(e => {
                const s = typeof e.source === 'string' ? e.source : e.source?.id;
                const t = typeof e.target === 'string' ? e.target : e.target?.id;
                return s === sourceId && t === targetId;
            });
            if (exists) {
                renderGraph();
                return;
            }
            selectedPlaybook.edges.push({ source: sourceId, target: targetId, type: 'depends_on' });
            renderGraph();
        }

        // ── Add Node from Palette ────────────────────────────────────────────
        document.querySelectorAll('.palette-item').forEach(item => {
            item.addEventListener('click', () => {
                if (!selectedPlaybook) { alert('Select a playbook first'); return; }
                const tool = item.dataset.tool;
                const id = uid();
                const center = getViewportCenter();
                const newNode = {
                    id, tool: tool === 'custom' ? 'tools.custom' : tool,
                    args: {},
                    _layout: { x: center.x - NODE_W / 2, y: center.y - NODE_H / 2 }
                };
                selectedPlaybook.nodes.push(newNode);
                selectNode(newNode);
            });
        });

        function getViewportCenter() {
            const svg = document.getElementById('canvas');
            const pt = svg.createSVGPoint();
            const rect = svg.getBoundingClientRect();
            pt.x = rect.width / 2;
            pt.y = rect.height / 2;
            const ctm = svg.getScreenCTM();
            if (!ctm) return { x: 400, y: 300 };
            const svgP = pt.matrixTransform(ctm.inverse());
            const t = transform;
            return { x: (svgP.x - t.x) / t.k, y: (svgP.y - t.y) / t.k };
        }

        // ── Auto Layout ──────────────────────────────────────────────────────
        document.getElementById('btn-auto-layout').addEventListener('click', () => {
            if (!selectedPlaybook) return;
            selectedPlaybook.nodes.forEach(n => delete n._layout);
            ensureLayout(selectedPlaybook);
            renderGraph();
        });

        // ── Toolbar Actions ────────────────────────────────────────────────────
        document.getElementById('btn-delete').addEventListener('click', deleteSelected);
        document.getElementById('btn-disconnect').addEventListener('click', () => {
            if (!selectedNodeId) return;
            selectedPlaybook.edges = selectedPlaybook.edges.filter(e => {
                const s = typeof e.source === 'string' ? e.source : e.source?.id;
                const t = typeof e.target === 'string' ? e.target : e.target?.id;
                return s !== selectedNodeId && t !== selectedNodeId;
            });
            renderGraph();
        });

        // ── Keyboard Shortcuts ───────────────────────────────────────────────
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Delete' || e.key === 'Backspace') {
                if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement.tagName)) return;
                deleteSelected();
            }
        });

        document.getElementById('canvas').addEventListener('click', (e) => {
            if (e.target.tagName === 'svg') clearSelection();
        });

        document.getElementById('refresh-btn').addEventListener('click', fetchPlaybooks);
        document.getElementById('export-btn').addEventListener('click', () => {
            if (!selectedPlaybook) return;
            const pb = currentPlaybooks.find(p => p.id === selectedPlaybook.id);
            if (!pb) return;
            // Strip internal _layout before export
            const exportPb = JSON.parse(JSON.stringify(pb));
            exportPb.nodes.forEach(n => delete n._layout);
            const data = JSON.stringify(exportPb, null, 2);
            const blob = new Blob([data], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url; a.download = `${selectedPlaybook.id}.json`;
            document.body.appendChild(a); a.click(); document.body.removeChild(a);
            URL.revokeObjectURL(url);
        });

        document.getElementById('zoom-in').addEventListener('click', () => {
            if (currentZoom) d3.select('#canvas').transition().duration(300).call(currentZoom.scaleBy, 1.3);
        });
        document.getElementById('zoom-out').addEventListener('click', () => {
            if (currentZoom) d3.select('#canvas').transition().duration(300).call(currentZoom.scaleBy, 0.7);
        });
        document.getElementById('zoom-fit').addEventListener('click', () => {
            if (!selectedPlaybook?.nodes?.length || !currentZoom) return;
            const positions = selectedPlaybook.nodes.map(n => (n._layout || { x: 0, y: 0 }));
            const minX = Math.min(...positions.map(p => p.x));
            const minY = Math.min(...positions.map(p => p.y));
            const maxX = Math.max(...positions.map(p => p.x + NODE_W));
            const maxY = Math.max(...positions.map(p => p.y + NODE_H));
            const container = document.getElementById('canvas-area');
            const w = container.clientWidth || 1200;
            const h = container.clientHeight || 800;
            const scale = Math.min(w / (maxX - minX + 160), h / (maxY - minY + 160), 2);
            const tx = (w - (maxX + minX) * scale) / 2;
            const ty = (h - (maxY + minY) * scale) / 2;
            d3.select('#canvas').transition().duration(500).call(
                currentZoom.transform,
                d3.zoomIdentity.translate(tx, ty).scale(scale)
            );
        });

        fetchPlaybooks();
        setInterval(fetchPlaybooks, 30000);


        // ── Layer 5 Spec panel ───────────────────────────────────────────────
        const SPEC_PLAYBOOK_TO_WORKFLOW = {
            gen_job_post: 'job_hunt',
            aurora_forecast: 'aurora_forecast',
        };

        function workflowKeyForPlaybook(pb) {
            return pb && pb.id ? (SPEC_PLAYBOOK_TO_WORKFLOW[pb.id] || null) : null;
        }

        function setSpecMessage(text, kind) {
            const el = document.getElementById('spec-message');
            if (!el) return;
            el.textContent = text || '';
            el.className = 'spec-message' + (kind ? ' ' + kind : '');
        }

        function openSpecDrawer() {
            const d = document.getElementById('spec-drawer');
            if (d) d.hidden = false;
        }

        function closeSpecDrawer() {
            const d = document.getElementById('spec-drawer');
            if (d) d.hidden = true;
        }

        async function loadSpecForPlaybook(pb) {
            const wk = workflowKeyForPlaybook(pb);
            const title = document.getElementById('spec-drawer-title');
            const source = document.getElementById('spec-source');
            const editor = document.getElementById('spec-editor');
            if (!wk) {
                closeSpecDrawer();
                if (title) title.textContent = 'Spec';
                if (source) source.textContent = '—';
                if (editor) editor.value = '';
                setSpecMessage('This playbook is not Spec-backed (view-only graph)');
                return;
            }
            openSpecDrawer();
            if (title) title.textContent = 'Spec · ' + wk;
            setSpecMessage('Loading Spec…');
            try {
                const resp = await fetch(API_BASE + '/workflows/' + encodeURIComponent(wk) + '/spec');
                const data = await resp.json();
                if (!resp.ok) throw new Error((data && data.detail) || resp.statusText);
                if (editor) editor.value = JSON.stringify(data.spec, null, 2);
                if (source) source.textContent = data.source || '—';
                setSpecMessage('Loaded from ' + (data.source || 'unknown'), 'ok');
            } catch (err) {
                setSpecMessage(String(err.message || err), 'err');
            }
        }

        async function validateSpec() {
            const editor = document.getElementById('spec-editor');
            try {
                const spec = JSON.parse((editor && editor.value) || '');
                const resp = await fetch(API_BASE + '/validate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ spec: spec }),
                });
                const data = await resp.json();
                if (!data.ok) {
                    setSpecMessage('Invalid: ' + (data.error || 'unknown'), 'err');
                    return;
                }
                if (editor) editor.value = JSON.stringify(data.spec, null, 2);
                setSpecMessage('Spec valid (normalized)', 'ok');
            } catch (err) {
                setSpecMessage(String(err.message || err), 'err');
            }
        }

        async function previewFromSpec() {
            const editor = document.getElementById('spec-editor');
            try {
                const spec = JSON.parse((editor && editor.value) || '');
                const resp = await fetch(API_BASE + '/preview', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ spec: spec }),
                });
                const data = await resp.json();
                if (!resp.ok) throw new Error((data && (data.detail || data.error)) || resp.statusText);
                selectedPlaybook = data.graph;
                ensureLayout(selectedPlaybook);
                renderGraph();
                setSpecMessage('Preview OK · ' + ((data.graph.nodes || []).length) + ' nodes', 'ok');
            } catch (err) {
                setSpecMessage(String(err.message || err), 'err');
            }
        }

        async function saveSpec() {
            const wk = workflowKeyForPlaybook(selectedPlaybook);
            if (!wk) {
                setSpecMessage('Not a Spec-backed playbook', 'err');
                return;
            }
            const editor = document.getElementById('spec-editor');
            try {
                const spec = JSON.parse((editor && editor.value) || '');
                const resp = await fetch(API_BASE + '/workflows/' + encodeURIComponent(wk) + '/spec', {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ spec: spec }),
                });
                const data = await resp.json();
                if (!resp.ok) throw new Error((data && data.detail) || resp.statusText);
                if (editor) editor.value = JSON.stringify(data.spec, null, 2);
                const source = document.getElementById('spec-source');
                if (source) source.textContent = 'spec.json';
                setSpecMessage('Saved ' + (data.path || 'spec.json'), 'ok');
            } catch (err) {
                setSpecMessage(String(err.message || err), 'err');
            }
        }

        document.getElementById('btn-validate')?.addEventListener('click', validateSpec);
        document.getElementById('btn-preview-spec')?.addEventListener('click', previewFromSpec);
        document.getElementById('btn-save-spec')?.addEventListener('click', saveSpec);
        document.getElementById('btn-close-spec')?.addEventListener('click', closeSpecDrawer);
        document.getElementById('spec-toggle-btn')?.addEventListener('click', () => {
            const d = document.getElementById('spec-drawer');
            if (!d) return;
            d.hidden = !d.hidden;
            if (!d.hidden && selectedPlaybook) loadSpecForPlaybook(selectedPlaybook);
        });
