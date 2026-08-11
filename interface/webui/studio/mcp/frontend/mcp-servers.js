// Detect base path for API calls (studio is mounted at /studio/mcp/)
        const API_BASE = GraphBoot.apiBase();

        let currentServer = null;
        let allServers = [];
        let loadInFlight = false;

        function escapeHTML(str) {
            const d = document.createElement('div');
            d.textContent = str || '';
            return d.innerHTML;
        }

        function formatPort(port) {
            return port ? `:${port}` : '—';
        }

        function statusClassOf(status) {
            return status === 'running' || status === 'stopped' ? status : 'unknown';
        }

        async function loadServers() {
            if (loadInFlight) return;
            loadInFlight = true;
            try {
                const resp = await fetch(`${API_BASE}/servers`);
                if (!resp.ok) {
                    console.error(`Failed to load servers: HTTP ${resp.status} ${resp.statusText}`);
                    document.getElementById('header-status').textContent = 'Failed to load';
                    return;
                }
                const data = await resp.json();
                allServers = (data.servers || []).map(server => ({
                    ...server,
                    tools: server.tools || []
                }));

                // Render list first so active-state is applied on the new DOM nodes.
                renderServerList();

                // Reconcile currentServer with refreshed data after the list exists.
                if (currentServer) {
                    const refreshed = allServers.find(s => s.name === currentServer.name);
                    if (refreshed) {
                        currentServer = refreshed;
                        selectServer(currentServer);
                    } else {
                        // Server no longer exists, clear selection
                        currentServer = null;
                        document.getElementById('detail-empty').style.display = 'block';
                        document.getElementById('detail-content').style.display = 'none';
                    }
                }

                document.getElementById('header-status').textContent = `${allServers.length} servers`;
            } catch (err) {
                console.error('Failed to load servers:', err);
                document.getElementById('header-status').textContent = 'Failed to load';
            } finally {
                loadInFlight = false;
            }
        }

        function renderServerList() {
            const container = document.getElementById('servers-list');

            if (!allServers.length) {
                container.innerHTML = '<div style="color:var(--dim);font-size:11px;padding:20px;text-align:center">No MCP servers found</div>';
                return;
            }

            container.innerHTML = '';
            allServers.forEach(server => {
                const button = document.createElement('button');
                button.type = 'button';
                button.className = 'server-item' + (currentServer && currentServer.name === server.name ? ' active' : '');
                button.setAttribute('data-server-name', server.name);

                const statusClass = statusClassOf(server.status);

                button.innerHTML = `
                    <div class="server-name">
                        ${escapeHTML(server.name)}
                        <span class="status-badge ${statusClass}">${escapeHTML(server.status || 'unknown')}</span>
                    </div>
                    <div class="server-meta">
                        <span>Port${formatPort(server.port)}</span>
                        <span>${server.tools.length} tools</span>
                    </div>
                `;
                button.onclick = () => selectServer(server);
                container.appendChild(button);
            });
        }

        function selectServer(server) {
            currentServer = server;
            renderServerList();

            // Show detail panel
            document.getElementById('detail-empty').style.display = 'none';
            document.getElementById('detail-content').style.display = 'block';

            const statusClass = statusClassOf(server.status);

            // Populate detail grid
            const grid = document.getElementById('detail-grid');
            grid.innerHTML = `
                <div class="detail-row">
                    <div class="detail-label">Name</div>
                    <div class="detail-value">${escapeHTML(server.name)}</div>
                </div>
                ${server.description ? `<div class="detail-row">
                    <div class="detail-label">Description</div>
                    <div class="detail-value">${escapeHTML(server.description)}</div>
                </div>` : ''}
                <div class="detail-row">
                    <div class="detail-label">Path</div>
                    <div class="detail-value"><code>${escapeHTML(server.path)}</code></div>
                </div>
                <div class="detail-row">
                    <div class="detail-label">Port</div>
                    <div class="detail-value">${formatPort(server.port)}</div>
                </div>
                <div class="detail-row">
                    <div class="detail-label">Status</div>
                    <div class="detail-value">
                        <span class="status-badge ${statusClass}">${escapeHTML(server.status || 'unknown')}</span>
                    </div>
                </div>
                <div class="detail-row">
                    <div class="detail-label">Tools</div>
                    <div class="detail-value">${server.tools.length}</div>
                </div>
            `;

            // Populate tools list
            const toolsList = document.getElementById('tools-list');
            if (server.tools.length === 0) {
                toolsList.innerHTML = '<div style="color:var(--dim);font-size:11px;padding:20px;text-align:center">No tools available or server not running</div>';
            } else {
                toolsList.innerHTML = server.tools.map(tool => `
                    <div class="tool-card">
                        <div class="tool-name">${escapeHTML(tool.name || 'Unknown')}</div>
                        <div class="tool-desc">${escapeHTML(tool.description || 'No description')}</div>
                        ${tool.source_file ? `<div class="tool-params" style="color:var(--dim);font-size:10px">source: ${escapeHTML(tool.source_file)}</div>` : ''}
                    </div>
                `).join('');
            }
        }

        document.getElementById('refresh-btn').addEventListener('click', loadServers);

        // Initial load
        loadServers();
        setInterval(loadServers, 30000);
