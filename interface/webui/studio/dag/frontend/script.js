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
        let selectedEdgeId = null;
        let transform = d3.zoomIdentity;
        let currentZoom = null;
        let draggingEdge = null;

        // Detect base path for API calls via the shared studio bootstrap
        const API_BASE = GraphBoot.apiBase();

        async function fetchPlaybooks() {
            try {
                const resp = await fetch(`${API_BASE}/playbooks`);
                currentPlaybooks = await resp.json();

                // Reconcile selectedPlaybook with refreshed data
                if (selectedPlaybook) {
                    const refreshed = currentPlaybooks.find(pb => pb.id === selectedPlaybook.id);
                    if (refreshed) {
                        selectedPlaybook = refreshed;
                        renderGraph(selectedPlaybook);
                    } else {
                        // Playbook no longer exists, clear selection
                        selectedPlaybook = null;
                        selectedNodeId = null;
                        selectedEdgeId = null;
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
                button.className = 'playbook-item' + (selectedPlaybook && selectedPlaybook.id === pb.id ? ' active' : '');
                button.innerHTML = `
                    <div class="pb-name">${escapeHTML(pb.name || pb.id)}</div>
                    <div class="pb-id">${escapeHTML(pb.id)}</div>
                    <div class="pb-meta"><span>${nodeCount} nodes</span></div>
                `;
                button.onclick = () => selectPlaybook(pb);
                container.appendChild(button);
            });
        }

        function escapeHTML(str) {
            const d = document.createElement('div');
            d.textContent = str || '';
            return d.innerHTML;
        }

        function selectPlaybook(playbook) {
            selectedPlaybook = playbook;
            selectedNodeId = null;
            selectedEdgeId = null;
            renderPlaybooksList();
            renderGraph(playbook);
            hideDetails();
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

        function renderGraph(playbook) {
            const svg = d3.select('#canvas');
            svg.selectAll('*').remove();
            const container = document.getElementById('canvas-area');
            const width = container.clientWidth || 1200;
            const height = container.clientHeight || 800;
            svg.attr('viewBox', `0 0 ${width} ${height}`);

            if (!playbook.nodes?.length) {
                svg.append('text')
                    .attr('x', width / 2).attr('y', height / 2)
                    .attr('text-anchor', 'middle')
                    .attr('fill', 'var(--dim)')
                    .attr('font-size', '14px')
                    .text('No graph data available');
                document.getElementById('graph-info').textContent = 'No data';
                return;
            }

            const nodes = playbook.nodes;
            const edges = playbook.edges || [];

            const positions = computeLayout(nodes, edges);
            nodes.forEach(n => { n._layout = positions[n.id] || { x: 200, y: 200 }; });

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

            // Draw edges
            const edgeGroup = g.append('g').attr('class', 'edges');
            const arrowMarkers = {};
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

            // Edge hover tooltip container
            const tooltip = document.getElementById('tooltip');

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

                const midX = (x1 + x2) / 2;
                const midY = (y1 + y2) / 2;
                const dx = x2 - x1;
                const dy = y2 - y1;
                const dist = Math.sqrt(dx * dx + dy * dy);
                const curvature = Math.min(dist * 0.15, 40);

                const pathD = `M ${x1} ${y1} C ${x1 + curvature} ${y1}, ${x2 - curvature} ${y2}, ${x2} ${y2}`;

                const path = edgeGroup.append('path')
                    .attr('d', pathD)
                    .attr('class', edgeClass)
                    .attr('marker-end', marker)
                    .attr('data-source', srcId)
                    .attr('data-target', tgtId)
                    .attr('data-edge-type', edgeType)
                    .style('cursor', 'pointer')
                    .on('click', (event) => {
                        event.stopPropagation();
                        selectedEdgeId = `${srcId}->${tgtId}`;
                        showEdgeDetails(edge, src, tgt);
                    })
                     .on('mouseenter', (event) => {
                         tooltip.textContent = `${srcId} → ${tgtId}${edgeType === 'loop_to' ? ' [loop]' : edgeType === 'fallback_to' ? ' [fallback]' : ''}`;
                         tooltip.style.left = (event.offsetX + 12) + 'px';
                         tooltip.style.top = (event.offsetY - 20) + 'px';
                         tooltip.classList.add('visible');
                         d3.select(event.target).classed('edge-highlight', true);
                     })
                     .on('mouseleave', () => {
                         tooltip.classList.remove('visible');
                         d3.select(event.target).classed('edge-highlight', false);
                     });
            });

            // Draw nodes
            const nodeGroup = g.append('g').attr('class', 'nodes');

            nodes.forEach(node => {
                const pos = node._layout;
                const ng = nodeGroup.append('g')
                    .attr('transform', `translate(${pos.x}, ${pos.y})`)
                    .style('cursor', 'pointer');

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
                        selectedNodeId = node.id;
                        selectedEdgeId = null;
                        renderGraph(playbook);
                        showNodeDetails(node);
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

                const isEntry = !(node.depends_on && node.depends_on.length > 0);
                const hasOutgoing = selectedPlaybook?.edges?.some(
                    e => (typeof e.source === 'string' ? e.source : e.source?.id) === node.id
                );
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
                    .attr('cx', 0).attr('cy', NODE_H / 2)
                    .attr('r', PORT_HIT_R)
                    .attr('fill', 'transparent')
                    .attr('data-target-id', node.id)
                    .style('cursor', 'crosshair');

                // Ports — output (right)
                ng.append('circle')
                    .attr('class', 'port-circle port-output')
                    .attr('cx', NODE_W).attr('cy', NODE_H / 2)
                    .attr('r', PORT_R);
                ng.append('circle')
                    .attr('cx', NODE_W).attr('cy', NODE_H / 2)
                    .attr('r', PORT_HIT_R)
                    .attr('fill', 'transparent')
                    .attr('data-source-id', node.id)
                    .style('cursor', 'crosshair')
                    .call(d3.drag()
                        .on("start", function(event) {
                            const srcId = d3.select(this).attr("data-source-id");
                            const srcNode = nodes.find(n => n.id === srcId);
                            const sPos = srcNode._layout;
                            draggingEdge = {
                                srcId: srcId,
                                startX: sPos.x + NODE_W,
                                startY: sPos.y + NODE_H / 2
                            };
                        })
                        .on("drag", function(event) {
                            if (!draggingEdge) return;
                            let tempPath = svg.select("#temp-drag-edge");
                            if (tempPath.empty()) {
                                tempPath = edgeGroup.append("path")
                                    .attr("id", "temp-drag-edge")
                                    .attr("class", "edge-depends")
                                    .style("pointer-events", "none");
                            }
                            const pt = d3.pointer(event, g.node());
                            const x1 = draggingEdge.startX;
                            const y1 = draggingEdge.startY;
                            const x2 = pt[0];
                            const y2 = pt[1];
                            const dx = x2 - x1;
                            const dy = y2 - y1;
                            const dist = Math.sqrt(dx * dx + dy * dy);
                            const curvature = Math.min(dist * 0.15, 40);
                            tempPath.attr("d", `M ${x1} ${y1} C ${x1 + curvature} ${y1}, ${x2 - curvature} ${y2}, ${x2} ${y2}`);
                        })
                        .on("end", function(event) {
                            svg.select("#temp-drag-edge").remove();
                            if (!draggingEdge) return;
                            const mousePt = d3.pointer(event, g.node());
                            let bestTarget = null;
                            let bestDist = 20;
                            svg.selectAll('.port-input + circle').each(function() {
                                const tgtId = d3.select(this).attr('data-target-id');
                                const tgtNode = nodes.find(n => n.id === tgtId);
                                if (tgtNode) {
                                    const tPos = tgtNode._layout;
                                    const tx = tPos.x;
                                    const ty = tPos.y + NODE_H / 2;
                                    const d = Math.hypot(mousePt[0] - tx, mousePt[1] - ty);
                                    if (d < bestDist && tgtId !== draggingEdge.srcId) {
                                        bestDist = d;
                                        bestTarget = tgtId;
                                    }
                                }
                            });
                            if (bestTarget) {
                                if (!selectedPlaybook.edges) selectedPlaybook.edges = [];
                                const exists = selectedPlaybook.edges.find(e => 
                                    (typeof e.source === 'string' ? e.source : e.source?.id) === draggingEdge.srcId && 
                                    (typeof e.target === 'string' ? e.target : e.target?.id) === bestTarget
                                );
                                if (!exists) {
                                    selectedPlaybook.edges.push({
                                        source: draggingEdge.srcId,
                                        target: bestTarget,
                                        type: "depends_on"
                                    });
                                    renderGraph(selectedPlaybook);
                                }
                            }
                            draggingEdge = null;
                        })
                    );
            });

            document.getElementById('graph-info').textContent =
                `${nodes.length} nodes, ${edges.length} edges`;
            const statusEl = document.getElementById('header-status');
            if (statusEl) {
                statusEl.textContent = `${nodes.length}n · ${edges.length}e`;
            }

            // Restore canvas opacity now that it is centered correctly
            svg.style('opacity', '1');
        }

        function showNodeDetails(node) {
            const panel = document.getElementById('details-panel');
            const content = document.getElementById('details-content');
            panel.style.display = 'block';
            const deps = selectedPlaybook?.edges?.filter(e => {
                const tid = typeof e.target === 'string' ? e.target : e.target?.id;
                return tid === node.id;
            }) || [];
            const dependents = selectedPlaybook?.edges?.filter(e => {
                const sid = typeof e.source === 'string' ? e.source : e.source?.id;
                return sid === node.id;
            }) || [];
            content.innerHTML = `
                <div class="details-panel-header"><h3>Node Spec</h3><button class="details-close" onclick="document.getElementById('details-panel').style.display='none'">×</button></div>
                <div class="detail-row"><div class="detail-label">ID</div><input class="detail-input" id="edit-node-id" value="${escapeHTML(node.id)}"></div>
                <div class="detail-row"><div class="detail-label">Tool</div><input class="detail-input" id="edit-node-tool" value="${escapeHTML(node.tool || '')}"></div>
                <div class="detail-row"><div class="detail-label">Args (JSON)</div><textarea class="detail-textarea" id="edit-node-args">${escapeHTML(JSON.stringify(node.args || {}, null, 2))}</textarea></div>
                <div class="detail-row"><div class="detail-label">Run If (JSON)</div><textarea class="detail-textarea" id="edit-node-runif" placeholder="optional condition">${escapeHTML(node.run_if ? JSON.stringify(node.run_if, null, 2) : '')}</textarea></div>
                <div class="detail-row"><div class="detail-label">When (JSON)</div><textarea class="detail-textarea" id="edit-node-when" placeholder="optional condition">${escapeHTML(node.when ? JSON.stringify(node.when, null, 2) : '')}</textarea></div>
                <div class="detail-row"><div class="detail-label">Loop To</div><input class="detail-input" id="edit-node-loop" value="${escapeHTML(node.loop_to || '')}"></div>
                <div class="detail-row"><div class="detail-label">Loop Condition</div><textarea class="detail-textarea" id="edit-node-loopcondition" placeholder="optional condition">${escapeHTML(node.loop_condition ? JSON.stringify(node.loop_condition, null, 2) : '')}</textarea></div>
                <div class="detail-row"><div class="detail-label">Fallback To</div><input class="detail-input" id="edit-node-fallback" value="${escapeHTML(node.fallback_to || '')}"></div>
                <div class="detail-row"><div class="detail-label">Max Visits</div><input class="detail-input" id="edit-node-maxvisits" type="number" min="1" value="${node.max_visits ?? ''}"></div>
                <div class="detail-row"><div class="detail-label">Timeout (s)</div><input class="detail-input" id="edit-node-timeout" type="number" min="0" step="0.1" value="${node.timeout_seconds ?? ''}"></div>
                <div class="detail-row"><div class="detail-label">Max Retries</div><input class="detail-input" id="edit-node-retries" type="number" min="0" value="${node.max_retries ?? 0}"></div>
                <div class="detail-row"><div class="detail-label">Backoff (s)</div><input class="detail-input" id="edit-node-backoff" type="number" min="0" step="0.1" value="${node.retry_backoff_seconds ?? 1}"></div>
                <div class="detail-row"><div class="detail-label">Interrupt</div><input id="edit-node-interrupt" type="checkbox" ${node.interrupt ? 'checked' : ''}></div>
                <div class="detail-row"><div class="detail-label">Needs Approval</div><input id="edit-node-approval" type="checkbox" ${node.needs_approval ? 'checked' : ''}></div>
                <div style="display:flex;gap:6px;margin-top:12px"><button class="btn btn-primary" onclick="saveNodeChanges('${escapeHTML(node.id)}')">Apply</button></div>
                <div class="detail-row"><div class="detail-label">Inputs (${deps.length})</div><div class="detail-value">${deps.map(d => `<code>${escapeHTML(typeof d.source === 'string' ? d.source : d.source?.id)}</code>`).join(', ') || '—'}</div></div>
                <div class="detail-row"><div class="detail-label">Outputs (${dependents.length})</div><div class="detail-value">${dependents.map(d => `<code>${escapeHTML(typeof d.target === 'string' ? d.target : d.target?.id)}</code>`).join(', ') || '—'}</div></div>
            `;
        }

        function parseOptionalJSON(id, label) {
            const value = document.getElementById(id).value.trim();
            if (!value) return null;
            try { return JSON.parse(value); } catch { throw new Error(`Invalid JSON in ${label}`); }
        }

        function saveNodeChanges(oldId) {
            const node = selectedPlaybook.nodes.find(n => n.id === oldId);
            if (!node) return;
            const newId = document.getElementById('edit-node-id').value.trim();
            if (!newId || (!node.id && !newId)) return alert('Node ID is required');
            if (newId !== oldId && selectedPlaybook.nodes.some(n => n.id === newId)) return alert('Node ID already exists');
            try {
                node.tool = document.getElementById('edit-node-tool').value.trim();
                if (!node.tool) throw new Error('Tool is required');
                node.args = JSON.parse(document.getElementById('edit-node-args').value || '{}');
                const runIf = parseOptionalJSON('edit-node-runif', 'Run If');
                const when = parseOptionalJSON('edit-node-when', 'When');
                const loopCondition = parseOptionalJSON('edit-node-loopcondition', 'Loop Condition');
                if (runIf) node.run_if = runIf; else delete node.run_if;
                if (when) node.when = when; else delete node.when;
                if (loopCondition) node.loop_condition = loopCondition; else delete node.loop_condition;
            } catch (err) { return alert(err.message || 'Invalid node spec'); }
            if (newId !== oldId) {
                selectedPlaybook.edges.forEach(e => {
                    if (e.source === oldId) e.source = newId;
                    if (e.target === oldId) e.target = newId;
                });
                node.id = newId;
            }
            const textFields = [['loop_to', 'edit-node-loop'], ['fallback_to', 'edit-node-fallback']];
            textFields.forEach(([field, id]) => { const value = document.getElementById(id).value.trim(); if (value) node[field] = value; else delete node[field]; });
            const numericFields = [['max_visits', 'edit-node-maxvisits', parseInt], ['timeout_seconds', 'edit-node-timeout', parseFloat], ['max_retries', 'edit-node-retries', parseInt], ['retry_backoff_seconds', 'edit-node-backoff', parseFloat]];
            numericFields.forEach(([field, id, parser]) => { const value = document.getElementById(id).value.trim(); if (value) node[field] = parser(value); else delete node[field]; });
            node.interrupt = document.getElementById('edit-node-interrupt').checked;
            node.needs_approval = document.getElementById('edit-node-approval').checked;
            selectedNodeId = node.id;
            renderGraph(selectedPlaybook);
            showNodeDetails(node);
        }

        function showEdgeDetails(edge, src, tgt) {
            const panel = document.getElementById('details-panel');
            const content = document.getElementById('details-content');
            panel.style.display = 'block';

            const type = edge.type || 'depends_on';
            const typeColor = type === 'fallback_to' ? '#e8843a' : type === 'loop_to' ? 'var(--pink)' : 'var(--dim)';
            const srcId = typeof edge.source === 'string' ? edge.source : edge.source?.id;
            const tgtId = typeof edge.target === 'string' ? edge.target : edge.target?.id;

            content.innerHTML = `
                <div class="details-panel-header">
                    <h3>Edge Details</h3>
                    <button class="details-close" onclick="document.getElementById('details-panel').style.display='none'">×</button>
                </div>
                <div class="detail-row"><div class="detail-label">Type</div>
                    <select class="detail-input" id="edit-edge-type">
                        <option value="depends_on"${type === 'depends_on' ? ' selected' : ''}>depends_on (normal)</option>
                        <option value="loop_to"${type === 'loop_to' ? ' selected' : ''}>loop_to (bounded loop)</option>
                        <option value="fallback_to"${type === 'fallback_to' ? ' selected' : ''}>fallback_to (on failure)</option>
                    </select></div>
                <div class="detail-row"><div class="detail-label">From</div><div class="detail-value"><code>${escapeHTML(src?.id || '?')}</code></div></div>
                <div class="detail-row"><div class="detail-label">To</div><div class="detail-value"><code>${escapeHTML(tgt?.id || '?')}</code></div></div>
                <div style="display:flex;gap:6px;margin-top:10px">
                    <button class="btn btn-primary" id="edge-apply-btn">Apply</button>
                    <button class="btn" id="edge-delete-btn" style="color:var(--pink)">Delete edge</button>
                </div>
                ${edge.tool_call ? `<div class="detail-row" style="margin-top:8px"><div class="detail-label">Tool Call</div><div class="detail-value"><pre class="text-xs" style="max-height:80px;overflow:auto;background:rgba(255,255,255,0.03);padding:4px;border-radius:3px">${escapeHTML(JSON.stringify(edge.tool_call, null, 2))}</pre></div></div>` : ''}
            `;
            document.getElementById('edge-apply-btn').onclick = () => {
                const nt = document.getElementById('edit-edge-type').value;
                edge.type = nt;
                syncEdgesToNodes();
                renderGraph(selectedPlaybook);
                showEdgeDetails(edge, src, tgt);
            };
            document.getElementById('edge-delete-btn').onclick = () => {
                selectedPlaybook.edges = (selectedPlaybook.edges || []).filter(e => {
                    const s = typeof e.source === 'string' ? e.source : e.source?.id;
                    const t = typeof e.target === 'string' ? e.target : e.target?.id;
                    return !(s === srcId && t === tgtId && (e.type || 'depends_on') === type);
                });
                selectedEdgeId = null;
                hideDetails();
                renderGraph(selectedPlaybook);
            };
        }

        function hideDetails() {
            document.getElementById('details-panel').style.display = 'none';
            selectedNodeId = null;
            selectedEdgeId = null;
        }

        document.getElementById('canvas').addEventListener('click', () => {
            hideDetails();
            if (selectedPlaybook) renderGraph(selectedPlaybook);
        });

        document.getElementById('refresh-btn').addEventListener('click', fetchPlaybooks);
        document.getElementById('save-btn').addEventListener('click', savePlaybook);

        function syncEdgesToNodes() {
            selectedPlaybook.nodes.forEach(node => { node.depends_on = []; delete node.loop_to; delete node.fallback_to; });
            (selectedPlaybook.edges || []).forEach(edge => {
                const source = typeof edge.source === 'string' ? edge.source : edge.source?.id;
                const target = typeof edge.target === 'string' ? edge.target : edge.target?.id;
                const targetNode = selectedPlaybook.nodes.find(node => node.id === target);
                const sourceNode = selectedPlaybook.nodes.find(node => node.id === source);
                if (!targetNode || !sourceNode) return;
                if ((edge.type || 'depends_on') === 'depends_on') targetNode.depends_on.push(source);
                if (edge.type === 'loop_to') sourceNode.loop_to = target;
                if (edge.type === 'fallback_to') sourceNode.fallback_to = target;
            });
        }

        async function savePlaybook() {
            if (!selectedPlaybook) return;
            syncEdgesToNodes();
            try {
                const resp = await fetch(`${API_BASE}/playbooks/${encodeURIComponent(selectedPlaybook.id)}`, { method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(selectedPlaybook) });
                const data = await resp.json();
                if (!resp.ok) throw new Error(data.detail || resp.statusText);
                selectedPlaybook = data.playbook;
                document.getElementById('header-status').textContent = 'Saved';
                renderGraph(selectedPlaybook);
                const node = selectedPlaybook.nodes.find(n => n.id === selectedNodeId);
                if (node) showNodeDetails(node);
            } catch (err) { alert(`Could not save playbook: ${err.message || err}`); }
        }
        document.getElementById('export-btn').addEventListener('click', () => {
            if (!selectedPlaybook) return;
            const pb = currentPlaybooks.find(p => p.id === selectedPlaybook.id);
            if (!pb) return;
            const data = JSON.stringify(pb, null, 2);
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

        // ── n8n-style: tool palette ──────────────────────────────────
        let toolGroups = {};
        let selectedTool = null;

        async function fetchTools() {
            try {
                const resp = await fetch(`${API_BASE}/tools`);
                const data = await resp.json();
                toolGroups = data.groups || {};
                renderPalette('');
            } catch (err) {
                console.error('Failed to fetch tools:', err);
            }
        }

        function renderPalette(filter) {
            const container = document.getElementById('tool-palette');
            if (!container) return;
            container.innerHTML = '';
            const q = (filter || '').toLowerCase();
            Object.entries(toolGroups).forEach(([domain, tools]) => {
                const shown = tools.filter(t => !q || t.name.toLowerCase().includes(q) || (t.description || '').toLowerCase().includes(q));
                if (!shown.length) return;
                const h = document.createElement('div');
                h.style.cssText = 'font-size:10px;color:var(--pink);letter-spacing:0.12em;text-transform:uppercase;margin:8px 0 4px';
                h.textContent = domain;
                container.appendChild(h);
                shown.slice(0, 30).forEach(t => {
                    const b = document.createElement('button');
                    b.className = 'playbook-item' + (selectedTool === t.name ? ' active' : '');
                    b.innerHTML = `<div class="pb-name" style="font-size:12px">${escapeHTML(t.name)}${t.needs_approval ? ' 🔒' : ''}</div><div class="pb-id">${escapeHTML((t.description || '').slice(0, 90))}</div>`;
                    b.title = t.description || t.name;
                    b.onclick = () => { selectedTool = t.name; renderPalette(document.getElementById('palette-search')?.value || ''); };
                    b.ondblclick = () => { selectedTool = t.name; addNodeFromPalette(); };
                    container.appendChild(b);
                });
            });
        }

        const paletteSearch = document.getElementById('palette-search');
        if (paletteSearch) paletteSearch.addEventListener('input', (e) => renderPalette(e.target.value));

        function defaultArgsFor(toolName) {
            // Sensible n8n-style defaults so a fresh node is runnable.
            if (toolName === 'make_plan') return { goal: '$prompt', max_steps: 5 };
            if (toolName === 'synthesize_report') return { evidence: '', prompt: '$prompt', style: 'plain' };
            if (toolName === 'write_report') return { title: '$title', content: '', report_dir: 'reports' };
            if (toolName === 'save_note') return { title: '$title', content: '$prompt', folder: 'notes' };
            if (toolName === 'combine_evidence') return { parts: [] };
            if (toolName === 'log_triage') return { log_file: 'logs/aiko.log', lines: 150 };
            if (toolName === 'sys_health') return {};
            if (toolName === 'text_summarize') return { text: '$prompt', max_sentences: 5 };
            if (toolName === 'text_translate') return { text: '$prompt', target: 'Japanese' };
            if (toolName === 'code_plan') return { goal: '$prompt' };
            if (toolName === 'needle_team_run') return { task: '$prompt' };
            return {};
        }

        function addNodeFromPalette() {
            if (!selectedPlaybook) return alert('Select a workflow first');
            if (!selectedPlaybook.nodes) selectedPlaybook.nodes = [];
            const tool = selectedTool || 'synthesize_report';
            const base = tool.split('.').pop().replace(/[^A-Za-z0-9]+/g, '_').toLowerCase().slice(0, 24) || 'node';
            let nid = base, k = 1;
            const ids = new Set(selectedPlaybook.nodes.map(n => n.id));
            while (ids.has(nid)) nid = `${base}_${++k}`;
            const newNode = { id: nid, tool, args: defaultArgsFor(tool) };
            selectedPlaybook.nodes.push(newNode);
            selectedNodeId = nid; selectedEdgeId = null;
            renderGraph(selectedPlaybook);
            showNodeDetails(newNode);
        }

        document.getElementById('add-node-btn').addEventListener('click', addNodeFromPalette);

        document.getElementById('delete-selected-btn').addEventListener('click', () => {
            if (!selectedPlaybook) return;
            if (selectedNodeId) {
                if (confirm(`Delete node ${selectedNodeId}?`)) {
                    selectedPlaybook.nodes = selectedPlaybook.nodes.filter(n => n.id !== selectedNodeId);
                    if (selectedPlaybook.edges) {
                        selectedPlaybook.edges = selectedPlaybook.edges.filter(e => {
                            const sid = typeof e.source === 'string' ? e.source : e.source?.id;
                            const tid = typeof e.target === 'string' ? e.target : e.target?.id;
                            return sid !== selectedNodeId && tid !== selectedNodeId;
                        });
                    }
                    selectedNodeId = null;
                    hideDetails();
                    renderGraph(selectedPlaybook);
                }
            } else if (selectedEdgeId) {
                if (confirm(`Delete edge ${selectedEdgeId}?`)) {
                    const [sId, tId] = selectedEdgeId.split('->');
                    if (selectedPlaybook.edges) {
                        selectedPlaybook.edges = selectedPlaybook.edges.filter(e => {
                            const sid = typeof e.source === 'string' ? e.source : e.source?.id;
                            const tid = typeof e.target === 'string' ? e.target : e.target?.id;
                            return !(sid === sId && tid === tId);
                        });
                    }
                    selectedEdgeId = null;
                    hideDetails();
                    renderGraph(selectedPlaybook);
                }
            } else {
                alert("Select a node or edge to delete");
            }
        });

        fetchPlaybooks();
        fetchTools();
        setInterval(fetchPlaybooks, 30000);

        // ── n8n-style: graph CRUD + validate + run + import ──────────
        function showRunPanel(html) {
            const p = document.getElementById('run-panel');
            document.getElementById('run-content').innerHTML = html;
            p.style.display = 'block';
        }

        document.getElementById('new-graph-btn')?.addEventListener('click', async () => {
            const id = prompt('New workflow id (letters, numbers, _ -):', 'my_workflow');
            if (!id) return;
            const name = prompt('Display name:', id) || id;
            try {
                const resp = await fetch(`${API_BASE}/playbooks`, { method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ id: id.trim(), name, goal: name, triggers: [], nodes: [{ id: 'start', tool: 'make_plan', args: { goal: '$prompt', max_steps: 5 } }] }) });
                const data = await resp.json();
                if (!resp.ok) throw new Error(data.detail || resp.statusText);
                await fetchPlaybooks();
                const found = currentPlaybooks.find(p => p.id === data.playbook.id);
                if (found) selectPlaybook(found);
            } catch (err) { alert(`Could not create graph: ${err.message || err}`); }
        });

        document.getElementById('duplicate-graph-btn')?.addEventListener('click', async () => {
            if (!selectedPlaybook) return alert('Select a workflow first');
            const nid = prompt('New id for the copy:', `${selectedPlaybook.id}_copy`);
            if (!nid) return;
            try {
                const resp = await fetch(`${API_BASE}/playbooks/${encodeURIComponent(selectedPlaybook.id)}/duplicate`, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ new_id: nid.trim() }) });
                const data = await resp.json();
                if (!resp.ok) throw new Error(data.detail || resp.statusText);
                await fetchPlaybooks();
                const found = currentPlaybooks.find(p => p.id === data.playbook.id);
                if (found) selectPlaybook(found);
            } catch (err) { alert(`Could not duplicate: ${err.message || err}`); }
        });

        document.getElementById('delete-graph-btn')?.addEventListener('click', async () => {
            if (!selectedPlaybook) return;
            if (!confirm(`Delete workflow ${selectedPlaybook.id}? Built-ins are protected.`)) return;
            try {
                const resp = await fetch(`${API_BASE}/playbooks/${encodeURIComponent(selectedPlaybook.id)}`, { method: 'DELETE' });
                const data = await resp.json();
                if (!resp.ok) throw new Error(data.detail || resp.statusText);
                selectedPlaybook = null; selectedNodeId = null; selectedEdgeId = null;
                hideDetails();
                await fetchPlaybooks();
                d3.select('#canvas').selectAll('*').remove();
            } catch (err) { alert(`Could not delete: ${err.message || err}`); }
        });

        document.getElementById('validate-btn')?.addEventListener('click', async () => {
            if (!selectedPlaybook) return;
            try {
                const resp = await fetch(`${API_BASE}/playbooks/${encodeURIComponent(selectedPlaybook.id)}/validate`, { method: 'POST' });
                const data = await resp.json();
                const cls = data.ok ? 'color:var(--cyan)' : 'color:var(--pink)';
                showRunPanel(`<div style="${cls};font-size:12px">${data.ok ? '✓ Valid' : '✗ ' + escapeHTML((data.errors || []).join('; '))}</div>
                    ${(data.warnings || []).map(w => `<div style="color:var(--dim);font-size:11px">⚠ ${escapeHTML(w)}</div>`).join('')}
                    <div style="color:var(--dim);font-size:11px;margin-top:6px">${data.nodes} nodes · entries: ${escapeHTML((data.entry_points || []).join(', ') || '—')}</div>`);
                document.getElementById('header-status').textContent = data.ok ? 'Valid' : 'Invalid';
            } catch (err) { alert(`Validate failed: ${err.message || err}`); }
        });

        document.getElementById('run-btn')?.addEventListener('click', async () => {
            if (!selectedPlaybook) return;
            const prompt = prompt('Dry-run prompt ($prompt):', 'Studio dry-run: describe what to check');
            if (prompt === null) return;
            showRunPanel('<div style="color:var(--dim);font-size:12px">Running… (bounded 60s, 2 workers)</div>');
            try {
                const resp = await fetch(`${API_BASE}/playbooks/${encodeURIComponent(selectedPlaybook.id)}/run`, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ prompt, timeout_s: 60 }) });
                const data = await resp.json();
                if (!resp.ok) throw new Error(data.detail || resp.statusText);
                showRunPanel(`<div style="color:var(--cyan);font-size:12px">✓ Done · score ${data.goal_score ?? '—'}</div>
                    <div style="font-size:11px;margin:6px 0;white-space:pre-wrap;max-height:120px;overflow:auto">${escapeHTML((data.final_answer || '').slice(0, 1200))}</div>
                    ${(data.nodes || []).map(n => `<div style="font-size:11px;color:${n.ok ? 'var(--text)' : 'var(--pink)'}">${n.ok ? '✓' : '✗'} <code>${escapeHTML(n.id)}</code> ${escapeHTML(n.tool)} — ${escapeHTML((n.content || n.error || '').slice(0, 160))}</div>`).join('')}`);
            } catch (err) { showRunPanel(`<div style="color:var(--pink);font-size:12px">Run failed: ${escapeHTML(err.message || err)}</div>`); }
        });

        document.getElementById('import-btn')?.addEventListener('click', () => document.getElementById('import-file')?.click());
        document.getElementById('import-file')?.addEventListener('change', async (e) => {
            const f = e.target.files?.[0];
            if (!f) return;
            try {
                const text = await f.text();
                const obj = JSON.parse(text);
                const payload = Array.isArray(obj) ? obj[0] : (obj.playbook || obj);
                if (!payload || !payload.id) throw new Error('JSON must contain a playbook object with an id');
                const resp = await fetch(`${API_BASE}/playbooks`, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
                const data = await resp.json();
                if (!resp.ok) throw new Error(data.detail || resp.statusText);
                await fetchPlaybooks();
            } catch (err) { alert(`Import failed: ${err.message || err}`); }
            e.target.value = '';
        });
