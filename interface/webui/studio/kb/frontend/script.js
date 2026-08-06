"use strict";

// Detect base path for API calls (studio is mounted at /studio/kb/ or run standalone)
const API_BASE = window.location.pathname.replace(/\/+$/, '') + '/api';

const state = {
    summary: null,
    docs: [],
    view: "list", // "list" | "detail"
    currentDoc: null,
    filter: { q: "", kind: "" },
};

const $ = (id) => document.getElementById(id);

function esc(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
}

function fmtDate(iso) {
    if (!iso) return "—";
    try { return new Date(iso).toLocaleString(); } catch { return iso; }
}

function statusBadge(status) {
    const s = (status || "active");
    const label = s === "active" ? "active" : s === "superseded" ? "superseded" : s;
    return `<span class="badge ${esc(s)}">${esc(label)}</span>`;
}

async function api(path) {
    const res = await fetch(API_BASE + path);
    if (!res.ok) throw new Error((await res.text()) || res.status);
    return res.json();
}

async function loadSummary() {
    try {
        state.summary = await api("/summary");
        $("stat-docs").textContent = state.summary.docs ?? 0;
        $("stat-chunks").textContent = state.summary.chunks ?? 0;
        $("stat-active").textContent = state.summary.active ?? 0;
        $("stat-superseded").textContent = state.summary.superseded ?? 0;
        $("stat-archived").textContent = state.summary.archived ?? 0;
        renderKindStats();
    } catch (e) {
        console.error("summary failed", e);
    }
}

function renderKindStats() {
    const kinds = state.summary?.by_kind || {};
    const select = $("kind-filter");
    select.innerHTML = '<option value="">All kinds</option>';
    Object.keys(kinds).forEach((k) => {
        const opt = document.createElement("option");
        opt.value = k;
        opt.textContent = k;
        select.appendChild(opt);
    });
    if (state.filter.kind) select.value = state.filter.kind;

    const box = $("kind-stats");
    box.innerHTML = Object.entries(kinds)
        .map(([k, v]) => `<div class="kind-stat-row"><span class="k">${esc(k)}</span><span class="v">${v}</span></div>`)
        .join("");
}

async function loadDocs() {
    try {
        const params = new URLSearchParams();
        if (state.filter.q) params.set("q", state.filter.q);
        if (state.filter.kind) params.set("kind", state.filter.kind);
        params.set("limit", "200");
        const data = await api("/docs?" + params.toString());
        state.docs = data.docs || [];
        renderDocs();
    } catch (e) {
        console.error("docs failed", e);
    }
}

function renderDocs() {
    const list = $("docs-list");
    if (state.docs.length === 0) {
        list.innerHTML = '<div style="color:var(--dim);padding:20px;text-align:center;">No documents found.</div>';
        return;
    }
    list.innerHTML = state.docs.map((d) => `
        <div class="doc-item" data-doc="${esc(d.id)}">
            <div class="doc-title">${esc(d.title || "Untitled")}</div>
            <div class="doc-meta">
                <span class="tag">${esc(d.kind || "ingested")}</span>
                <span class="tag">${d.chunk_count ?? 0} chunks</span>
                ${(d.superseded_chunks ?? 0) > 0 ? `<span class="tag">${d.superseded_chunks} superseded</span>` : ""}
                <span class="tag">${esc(fmtDate(d.created_at))}</span>
            </div>
            ${d.source ? `<div class="doc-meta"><span class="tag" style="word-break:break-all;">${esc(d.source)}</span></div>` : ""}
        </div>`).join("");

    list.querySelectorAll(".doc-item").forEach((el) => {
        el.addEventListener("click", () => openDoc(el.dataset.doc));
    });
}

async function openDoc(docId) {
    try {
        const data = await api(`/docs/${encodeURIComponent(docId)}`);
        state.currentDoc = data;
        state.view = "detail";
        $("docs-list").style.display = "none";
        $("detail-view").style.display = "block";
        renderDetail();
    } catch (e) {
        console.error("doc detail failed", e);
    }
}

function renderDetail() {
    const d = state.currentDoc.doc;
    const chunks = state.currentDoc.chunks || [];
    const content = $("detail-content");
    content.innerHTML = `
        <h2>${esc(d.title || "Untitled")}</h2>
        <div class="detail-meta">
            <div class="dv"><div class="lbl">Kind</div><div class="val">${esc(d.kind)}</div></div>
            <div class="dv"><div class="lbl">Source</div><div class="val">${esc(d.source || "—")}</div></div>
            <div class="dv"><div class="lbl">Created</div><div class="val">${esc(fmtDate(d.created_at))}</div></div>
            <div class="dv"><div class="lbl">Chunks</div><div class="val">${chunks.length}</div></div>
        </div>
        ${chunks.length === 0 ? '<div style="color:var(--dim);">No chunks.</div>' : chunks.map((c) => `
            <div class="chunk-item">
                <div class="chunk-head">
                    <span class="chunk-idx">#${c.chunk_index}</span>
                    ${statusBadge(c.status)}
                    <span style="color:var(--dim);font-size:11px;">access ${c.access_count ?? 0}</span>
                    ${c.supersedes_id ? `<span style="color:var(--dim);font-size:11px;">supersedes ${esc(c.supersedes_id.slice(0,8))}…</span>` : ""}
                </div>
                <div class="chunk-body">${esc(c.text_preview || "")}</div>
                <div class="chunk-entities">${renderEntities(c.entities)}</div>
            </div>`).join("")}
    `;
}

function renderEntities(entitiesJson) {
    let ents = [];
    try { ents = JSON.parse(entitiesJson || "[]"); } catch { ents = []; }
    if (!Array.isArray(ents) || ents.length === 0) return "";
    return `<div class="chunk-entities">${ents.map((e) => `<span class="entity-chip">${esc(typeof e === "string" ? e : e.name || JSON.stringify(e))}</span>`).join("")}</div>`;
}

function backToList() {
    state.view = "list";
    $("detail-view").style.display = "none";
    $("docs-list").style.display = "block";
}

$("refresh-btn").addEventListener("click", () => {
    loadSummary();
    loadDocs();
});

$("search-input").addEventListener("input", (e) => {
    state.filter.q = e.target.value;
    loadDocs();
});

$("kind-filter").addEventListener("change", (e) => {
    state.filter.kind = e.target.value;
    loadDocs();
});

$("back-btn").addEventListener("click", backToList);

loadSummary();
loadDocs();
