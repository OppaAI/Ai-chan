// Set up filter click events once DOM loads
document.querySelectorAll('.filter-tab').forEach(tab => {
    tab.addEventListener('click', () => {
        const filter = tab.dataset.filter;
        renderDraftList(filter);
    });
});

// Detect base path for API calls (studio is mounted at /studio/approval/)
const API_BASE = window.location.pathname.replace(/\/+$/, '') + '/api';

let currentDraft = null;
let allDrafts = [];
let originalDraftText = '';
let sourceView = 'rendered';

function escapeHTML(str) {
    const d = document.createElement('div');
    d.textContent = str || '';
    return d.innerHTML;
}

function formatDate(dateStr) {
    if (!dateStr) return '—';
    try {
        const d = new Date(dateStr);
        return d.toLocaleString();
    } catch {
        return dateStr;
    }
}

async function loadDrafts() {
    try {
        const resp = await fetch(`${API_BASE}/drafts?status=all`);
        const data = await resp.json();
        allDrafts = data.drafts || [];
        renderDraftList('all');
        document.getElementById('header-status').textContent = `${allDrafts.length} drafts`;
        document.getElementById('draft-count').textContent = allDrafts.length;
    } catch (err) {
        console.error('Failed to load drafts:', err);
        document.getElementById('header-status').textContent = 'Failed to load';
    }
}

function filterDrafts(filter) {
    switch (filter) {
        case 'pending':
            return allDrafts.filter(d => !d.human_approved && !d.posted);
        case 'approved':
            return allDrafts.filter(d => d.human_approved && !d.posted);
        case 'posted':
            return allDrafts.filter(d => d.posted);
        default:
            return allDrafts;
    }
}

function renderDraftList(filter) {
    const container = document.getElementById('drafts-list');
    const drafts = filterDrafts(filter);

    document.querySelectorAll('.filter-tab').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.filter === filter);
    });

    if (!drafts.length) {
        container.innerHTML = '<div style="color:var(--dim);font-size:11px;padding:20px;text-align:center">No drafts found</div>';
        return;
    }

    container.innerHTML = '';
    drafts.forEach(draft => {
        const div = document.createElement('div');
        div.className = 'draft-item' + (currentDraft && currentDraft.draft_dir === draft.draft_dir ? ' active' : '');

        const statusBadges = [];
        if (draft.posted) {
            statusBadges.push('<span class="status-badge posted">Posted</span>');
        } else if (draft.human_approved) {
            statusBadges.push('<span class="status-badge approved">Approved</span>');
        } else {
            statusBadges.push('<span class="status-badge pending">Pending</span>');
        }
        if (draft.llm_enriched) {
            statusBadges.push('<span class="status-badge enriched">LLM Enriched</span>');
        }

        div.innerHTML = `
            <div class="draft-name">${escapeHTML(draft.relative_path)}</div>
            <div class="draft-meta">
                <span>${formatDate(draft.created_at)}</span>
                <span>${escapeHTML(draft.category || 'general')}</span>
                <span>${escapeHTML(draft.provider)}</span>
            </div>
            <div class="draft-status">${statusBadges.join('')}</div>
        `;
        div.onclick = () => selectDraft(draft);
        container.appendChild(div);
    });
}

function selectDraft(draft) {
    if (currentDraft && currentDraft.draft_dir !== draft.draft_dir && isContentDirty()) {
        const proceed = confirm('You have unsaved changes to this draft. Discard them and switch?');
        if (!proceed) return;
    }
    currentDraft = draft;
    document.querySelectorAll('.draft-item').forEach(el => {
        el.classList.toggle('active', el.textContent.includes(draft.relative_path));
    });

    // Show info panel
    document.getElementById('info-empty').style.display = 'none';
    document.getElementById('info-content').style.display = 'block';

    // Populate info grid
    const grid = document.getElementById('info-grid');
    grid.innerHTML = `
        <div class="info-row">
            <div class="info-label">Path</div>
            <div class="info-value"><code>${escapeHTML(draft.relative_path)}</code></div>
        </div>
        <div class="info-row">
            <div class="info-label">Date</div>
            <div class="info-value">${escapeHTML(draft.date || '—')}</div>
        </div>
        <div class="info-row">
            <div class="info-label">Category</div>
            <div class="info-value">${escapeHTML(draft.category || 'general')}</div>
        </div>
        <div class="info-row">
            <div class="info-label">Provider</div>
            <div class="info-value">${escapeHTML(draft.provider)}</div>
        </div>
        <div class="info-row">
            <div class="info-label">Created</div>
            <div class="info-value">${formatDate(draft.created_at)}</div>
        </div>
        <div class="info-row">
            <div class="info-label">LLM Enriched</div>
            <div class="info-value" style="color: ${draft.llm_enriched ? 'var(--green)' : 'var(--dim)'}">${draft.llm_enriched ? 'Yes' : 'No'}</div>
        </div>
        <div class="info-row">
            <div class="info-label">Status</div>
            <div class="info-value">
                ${draft.posted ? '<span class="status-badge posted">Posted</span>' :
                  draft.human_approved ? '<span class="status-badge approved">Approved</span>' :
                  '<span class="status-badge pending">Pending Review</span>'}
            </div>
        </div>
    `;

    // Action buttons in the exact order: Approve, Publish, Reject
    const actions = document.getElementById('action-buttons');
    if (draft.posted) {
        actions.innerHTML = '<span style="color: var(--dim); font-size: 11px;">This draft has already been posted</span>';
    } else if (draft.human_approved) {
        actions.innerHTML = `
            <button class="btn btn-warning" onclick="toggleApprove('${escapeHTML(draft.draft_dir)}', false)" style="background: var(--orange); border-color: var(--orange); color: var(--bg);">Unapprove</button>
            <button class="btn btn-primary" onclick="postDraft('${escapeHTML(draft.draft_dir)}')">Publish</button>
            <button class="btn btn-danger" onclick="rejectDraft('${escapeHTML(draft.draft_dir)}')">Reject</button>
        `;
    } else {
        actions.innerHTML = `
            <button class="btn btn-success" onclick="toggleApprove('${escapeHTML(draft.draft_dir)}', true)">Approve</button>
            <button class="btn btn-primary" style="opacity: 0.5; cursor: not-allowed;" title="Approve before publishing" disabled>Publish</button>
            <button class="btn btn-danger" onclick="rejectDraft('${escapeHTML(draft.draft_dir)}')">Reject</button>
        `;
    }

    // Load full draft content
    loadDraftContent(draft.draft_dir);
}

function autoGrowTextarea(el) {
    el.style.height = 'auto';
    el.style.height = el.scrollHeight + 'px';
}

function isContentDirty() {
    const contentEl = document.getElementById('content-text');
    if (contentEl.style.display === 'none') return false;
    return contentEl.value !== originalDraftText;
}

function updateEditStatus() {
    const contentEl = document.getElementById('content-text');
    const status = document.getElementById('edit-status');
    const dirty = isContentDirty();
    contentEl.classList.toggle('dirty', dirty);
    status.className = 'edit-status' + (dirty ? ' dirty' : '');
    status.textContent = dirty ? 'Unsaved changes' : '';
}

async function loadDraftContent(draftDir) {
    try {
        const resp = await fetch(`${API_BASE}/drafts/${encodeURIComponent(draftDir)}`);
        if (!resp.ok) throw new Error('Failed to load draft detail');
        const data = await resp.json();

        const contentEl = document.getElementById('content-text');
        const emptyEl = document.getElementById('content-empty');
        const editActions = document.getElementById('content-edit-actions');

        if (data.draft_text) {
            contentEl.value = data.draft_text;
            originalDraftText = data.draft_text;
            contentEl.style.display = 'block';
            editActions.style.display = 'flex';
            emptyEl.style.display = 'none';
            autoGrowTextarea(contentEl);
            updateEditStatus();
        } else {
            contentEl.value = '';
            originalDraftText = '';
            contentEl.style.display = 'none';
            editActions.style.display = 'none';
            emptyEl.style.display = 'flex';
            emptyEl.querySelector('p').textContent = 'No draft content available';
        }

        // Source webpage panel
        const posting = data.meta?.posting || data.posting;
        const urlContent = document.getElementById('url-content');
        const urlEmpty = document.getElementById('url-empty');
        const sourceTitle = document.getElementById('source-title');
        const urlIframe = document.getElementById('url-iframe');
        const urlText = document.getElementById('url-text');

        if (posting && posting.url) {
            urlEmpty.style.display = 'none';
            urlContent.style.display = 'flex';

            sourceTitle.textContent = posting.url;
            sourceTitle.style.textTransform = 'none';
            sourceTitle.style.letterSpacing = 'normal';
            sourceTitle.style.color = 'var(--cyan)';
            sourceTitle.title = 'Open in a new tab';
            sourceTitle.onclick = () => window.open(posting.url, '_blank', 'noopener');
            sourceTitle.style.cursor = 'pointer';

            urlIframe.src = posting.url;
            urlText.textContent = 'Loading text view…';

            // Fetch the scraped text as a fallback view
            fetch(`${API_BASE}/fetch-url?url=${encodeURIComponent(posting.url)}`)
                .then(resp => resp.json())
                .then(result => {
                    if (result.error) {
                        urlText.innerHTML = `<p style="color: var(--red); font-size: 11px;">Failed to load: ${escapeHTML(result.error)}</p>`;
                    } else {
                        urlText.textContent = result.content || 'No content available';
                    }
                })
                .catch(err => {
                    urlText.innerHTML = `<p style="color: var(--red); font-size: 11px;">Error loading URL: ${escapeHTML(err.message)}</p>`;
                });

            applySourceView(sourceView);
        } else {
            urlContent.style.display = 'none';
            urlEmpty.style.display = 'flex';
            sourceTitle.textContent = 'Source Webpage';
            sourceTitle.style.textTransform = 'uppercase';
            sourceTitle.style.letterSpacing = '0.15em';
            sourceTitle.style.color = 'var(--orange)';
            sourceTitle.onclick = null;
            sourceTitle.style.cursor = 'default';
            urlIframe.src = 'about:blank';
        }
    } catch (err) {
        console.error('Failed to load draft content:', err);
        document.getElementById('content-text').style.display = 'none';
        document.getElementById('content-edit-actions').style.display = 'none';
        document.getElementById('content-empty').style.display = 'flex';
        document.getElementById('content-empty').querySelector('p').textContent = 'Failed to load content';
    }
}

function applySourceView(view) {
    sourceView = view;
    const rendered = document.getElementById('url-rendered');
    const text = document.getElementById('url-text');
    const btnRendered = document.getElementById('view-rendered-btn');
    const btnText = document.getElementById('view-text-btn');
    rendered.style.display = view === 'rendered' ? 'flex' : 'none';
    text.style.display = view === 'text' ? 'block' : 'none';
    btnRendered.classList.toggle('active', view === 'rendered');
    btnText.classList.toggle('active', view === 'text');
}

document.getElementById('view-rendered-btn').addEventListener('click', () => applySourceView('rendered'));
document.getElementById('view-text-btn').addEventListener('click', () => applySourceView('text'));

document.getElementById('content-text').addEventListener('input', (e) => {
    autoGrowTextarea(e.target);
    updateEditStatus();
});

document.getElementById('cancel-content-btn').addEventListener('click', () => {
    const contentEl = document.getElementById('content-text');
    contentEl.value = originalDraftText;
    autoGrowTextarea(contentEl);
    updateEditStatus();
});

document.getElementById('save-content-btn').addEventListener('click', async () => {
    if (!currentDraft) return;
    const contentEl = document.getElementById('content-text');
    const status = document.getElementById('edit-status');
    const saveBtn = document.getElementById('save-content-btn');
    const newText = contentEl.value;

    saveBtn.disabled = true;
    status.className = 'edit-status';
    status.textContent = 'Saving…';

    try {
        const resp = await fetch(`${API_BASE}/drafts/${encodeURIComponent(currentDraft.draft_dir)}/update-content`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ draft_text: newText }),
        });
        const data = await resp.json();
        if (!resp.ok || !data.success) {
            throw new Error(data.message || 'Save failed');
        }
        originalDraftText = newText;
        status.className = 'edit-status saved';
        status.textContent = 'Saved';
        contentEl.classList.remove('dirty');
        setTimeout(() => {
            if (status.textContent === 'Saved') status.textContent = '';
        }, 2500);
    } catch (err) {
        console.error('Failed to save draft content:', err);
        status.className = 'edit-status error';
        status.textContent = 'Failed to save';
    } finally {
        saveBtn.disabled = false;
    }
});

async function toggleApprove(draftDir, approve) {
    try {
        const resp = await fetch(`${API_BASE}/drafts/${encodeURIComponent(draftDir)}/toggle-approval`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ approve }),
        });
        const data = await resp.json();
        if (data.success) {
            loadDrafts();
            if (currentDraft && currentDraft.draft_dir === draftDir) {
                selectDraft({ ...currentDraft, human_approved: data.meta.human_approved, meta: data.meta });
            }
        } else {
            alert('Failed to update draft: ' + (data.message || 'Unknown error'));
        }
    } catch (err) {
        console.error('Toggle approval failed:', err);
        alert('Failed to update draft approval status');
    }
}

document.getElementById('refresh-btn').addEventListener('click', loadDrafts);

// Initial load
loadDrafts();
setInterval(loadDrafts, 30000);