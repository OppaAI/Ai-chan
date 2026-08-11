// Detect base path for API calls (studio is mounted at /studio/mcp/)
        const API_BASE = GraphBoot.apiBase();

        let currentServer = null;
        let allServers = [];

        function escapeHTML(str) {
            const d = document.createElement('div');
            d.textContent = str || '';
            return d.innerHTML;
        }

        function formatPort(port) {
            return port ? `:${port}` : '—';
        }

        async function loadServers() {
            try {
                const resp = await fetch(`${API_BASE}/servers`);
                const data = await resp.json();
                allServers = data.servers || [];

                // Reconcile currentServer with refreshed data
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

                renderServerList();
                document.getElementById('header-status').textContent = `${allServers.length} servers`;
            } catch (err) {
                console.error('Failed to load servers:', err);
                document.getElementById('header-status').textContent = 'Failed to load';
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

                let statusClass = 'unknown';
                if (server.status === 'running') statusClass = 'running';
                else if (server.status === 'stopped') statusClass = 'stopped';

                button.innerHTML = `
                    <div class="server-name">
                        ${escapeHTML(server.name)}
                        <span class="status-badge ${statusClass}">${server.status || 'unknown'}</span>
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
            document.querySelectorAll('.server-item').forEach(el => {
                el.classList.toggle('active', el.textContent.includes(server.name));
            });

            // Show detail panel
            document.getElementById('detail-empty').style.display = 'none';
            document.getElementById('detail-content').style.display = 'block';

            // Derive statusClass locally (was missing, causing a ReferenceError)
            let statusClass = 'unknown';
            if (server.status === 'running') statusClass = 'running';
            else if (server.status === 'stopped') statusClass = 'stopped';

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
                        <span class="status-badge ${statusClass}">${server.status || 'unknown'}</span>
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
