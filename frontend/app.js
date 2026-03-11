/**
 * Autonomous Document Workflow Engine — Frontend App
 * Polls the API for documents and events, renders the live dashboard.
 */

const API = '';
const POLL_MS = 3000;

let selectedDocId = null;
let knownEventIds = new Set();

// Global store: document_id → doc object
const docStore = new Map();

// ── DOM Refs ───────────────────────────────────────────────────────────────
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const uploadProgress = document.getElementById('uploadProgress');
const progressFill = document.getElementById('progressFill');
const progressLabel = document.getElementById('progressLabel');
const docList = document.getElementById('docList');
const docCount = document.getElementById('docCount');
const eventFeed = document.getElementById('eventFeed');
const emptyDetail = document.getElementById('emptyDetail');
const docDetail = document.getElementById('docDetail');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');

// ── Upload ─────────────────────────────────────────────────────────────────
uploadZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) uploadFile(fileInput.files[0]);
});
uploadZone.addEventListener('dragover', e => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
});
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    if (e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0]);
});

async function uploadFile(file) {
    uploadProgress.classList.remove('hidden');
    progressFill.style.width = '10%';
    progressLabel.textContent = `Uploading ${file.name}…`;

    const form = new FormData();
    form.append('file', file);

    try {
        progressFill.style.width = '60%';
        const resp = await fetch(`${API}/upload`, { method: 'POST', body: form });
        progressFill.style.width = '100%';
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            progressLabel.textContent = `❌ ${err.detail || 'Upload failed'}`;
            setTimeout(() => uploadProgress.classList.add('hidden'), 3000);
            return;
        }
        progressLabel.textContent = `✅ ${file.name} received — processing…`;
        setTimeout(() => {
            uploadProgress.classList.add('hidden');
            progressFill.style.width = '0%';
        }, 2000);
        fileInput.value = '';
    } catch (e) {
        progressLabel.textContent = `❌ Network error: ${e.message}`;
        setTimeout(() => uploadProgress.classList.add('hidden'), 3000);
    }
}

// ── Polling ────────────────────────────────────────────────────────────────
async function pollDocuments() {
    try {
        const resp = await fetch(`${API}/documents`);
        if (!resp.ok) return;
        const docs = await resp.json();

        // Update global store
        docs.forEach(d => docStore.set(d.document_id, d));

        renderDocList(docs);

        // Keep Evolution Lab doc dropdown in sync
        if (!document.getElementById('tabContentEvo').classList.contains('hidden')) {
            populateEvoDocSelect();
        }

        // Re-render selected doc if its data changed
        if (selectedDocId && docStore.has(selectedDocId)) {
            renderDocDetail(docStore.get(selectedDocId));
        }

        setOnline(true);
    } catch (e) {
        setOnline(false);
    }
}

async function pollEvents() {
    try {
        const resp = await fetch(`${API}/events?limit=200`);
        if (!resp.ok) return;
        const events = await resp.json();
        events.slice().reverse().forEach(evt => {
            if (!knownEventIds.has(evt.id)) {
                knownEventIds.add(evt.id);
                prependEvent(evt);
            }
        });
    } catch (e) { /* silent */ }
}

function setOnline(online) {
    statusDot.className = 'status-dot ' + (online ? 'online' : 'error');
    statusText.textContent = online ? 'Engine Online' : 'Disconnected';
}

// ── Document List ──────────────────────────────────────────────────────────
function renderDocList(docs) {
    docCount.textContent = docs.length;

    if (!docs.length) {
        docList.innerHTML = '<div class="empty-state">No documents yet.<br/>Upload one to get started.</div>';
        return;
    }

    // Clear and rebuild with proper event listeners (no inline onclick)
    docList.innerHTML = '';
    docs.forEach(d => {
        const item = document.createElement('div');
        item.className = `doc-item${d.document_id === selectedDocId ? ' active' : ''}`;
        item.id = `docitem-${d.document_id}`;

        const docType = d.result && d.result.doc_type ? d.result.doc_type : null;
        item.innerHTML = `
      <div class="doc-item-name" title="${esc(d.filename)}">${esc(d.filename)}</div>
      <div class="doc-item-meta">
        <span class="status-badge ${d.status}">${d.status}</span>
        ${docType ? `<span class="doc-type-badge">${esc(docType)}</span>` : ''}
      </div>`;

        // Safe click handler using document_id lookup in docStore
        item.addEventListener('click', () => selectDoc(d.document_id));
        docList.appendChild(item);
    });
}

function selectDoc(docId) {
    selectedDocId = docId;
    // Update active highlight
    document.querySelectorAll('.doc-item').forEach(el => el.classList.remove('active'));
    const el = document.getElementById(`docitem-${docId}`);
    if (el) el.classList.add('active');

    const doc = docStore.get(docId);
    if (doc) renderDocDetail(doc);
}

// ── Document Detail ────────────────────────────────────────────────────────
function renderDocDetail(doc) {
    emptyDetail.classList.add('hidden');
    docDetail.classList.remove('hidden');

    document.getElementById('detailFilename').textContent = doc.filename;

    const statusEl = document.getElementById('detailStatus');
    statusEl.textContent = doc.status;
    statusEl.className = `status-badge ${doc.status}`;

    const result = doc.result || {};
    const docType = result.doc_type || '—';
    document.getElementById('detailType').textContent = docType;

    // Classification
    const classif = result.classification_detail || {};
    document.getElementById('classificationContent').innerHTML = `
    <div class="classification-grid">
      <div class="classif-item"><div class="field-label">Type</div><div class="field-value">${esc(docType)}</div></div>
      <div class="classif-item"><div class="field-label">Confidence</div><div class="field-value">${esc(classif.confidence || '—')}</div></div>
      <div class="classif-item" style="flex:2"><div class="field-label">Reasoning</div><div class="field-value">${esc(classif.reasoning || '—')}</div></div>
    </div>`;

    // If still processing or error, show status
    if (doc.status === 'processing') {
        document.getElementById('classificationContent').innerHTML =
            '<div class="field-value" style="color:var(--warning)">⏳ Processing… check back in a moment.</div>';
    } else if (doc.status === 'error') {
        document.getElementById('classificationContent').innerHTML =
            '<div class="field-value" style="color:var(--error)">❌ Processing failed for this document.<br/>It may be a scanned/image PDF (no extractable text).</div>';
    }

    // Summary
    const summSection = document.getElementById('summarySection');
    const summary = result.summary;
    if (summary) {
        summSection.classList.remove('hidden');
        document.getElementById('summaryContent').textContent = summary;
    } else {
        summSection.classList.add('hidden');
    }

    // Extracted
    const extSection = document.getElementById('extractionSection');
    const extracted = result.extracted;
    if (extracted && Object.keys(extracted).filter(k => k !== 'error').length) {
        extSection.classList.remove('hidden');
        document.getElementById('extractionContent').innerHTML = `
      <div class="field-grid">${Object.entries(extracted).map(([k, v]) => {
            if (k === 'error') return '';
            const val = (v == null || v === '')
                ? '<span class="null">not found</span>'
                : Array.isArray(v)
                    ? esc(v.join(', '))
                    : esc(String(v));
            return `<div class="field-item">
            <div class="field-label">${esc(k.replace(/_/g, ' '))}</div>
            <div class="field-value">${val}</div>
          </div>`;
        }).join('')
            }</div>`;
    } else {
        extSection.classList.add('hidden');
    }
}

// ── Event Feed ─────────────────────────────────────────────────────────────
const EVENT_LABELS = {
    DOCUMENT_RECEIVED: '📥 Document Received',
    INGESTION_COMPLETE: '📄 Ingestion Complete',
    INGESTION_FAILED: '❌ Ingestion Failed',
    CLASSIFICATION_COMPLETE: '🏷️  Classified',
    ROUTING_COMPLETE: '🔀 Routing Complete',
    EXTRACTION_COMPLETE: '🔎 Extraction Complete',
    SUMMARY_COMPLETE: '📝 Summary Complete',
    WORKFLOW_COMPLETE: '✅ Workflow Complete',
    WORKFLOW_ERROR: '⚠️  Workflow Error',
};

function prependEvent(evt) {
    const empty = eventFeed.querySelector('.empty-state');
    if (empty) empty.remove();

    const el = document.createElement('div');
    el.className = `event-item ${evt.event_type}`;

    const p = evt.payload || {};
    let detail = '';
    if (evt.event_type === 'CLASSIFICATION_COMPLETE') detail = `→ ${p.label} (${p.confidence})`;
    else if (evt.event_type === 'INGESTION_COMPLETE') detail = `${p.char_count ? p.char_count.toLocaleString() : '?'} chars`;
    else if (evt.event_type === 'EXTRACTION_COMPLETE') detail = p.doc_type || '';
    else if (evt.event_type === 'WORKFLOW_COMPLETE') detail = '🎉 Ready';
    else if (p.error) detail = String(p.error).slice(0, 80);

    // Look up filename from docStore
    const doc = docStore.get(evt.document_id);
    const docLabel = doc ? doc.filename : evt.document_id.slice(0, 8) + '…';

    el.innerHTML = `
    <div class="event-type">${EVENT_LABELS[evt.event_type] || evt.event_type}</div>
    <div class="event-doc" title="${esc(docLabel)}">${esc(docLabel.length > 30 ? docLabel.slice(0, 28) + '…' : docLabel)}</div>
    ${detail ? `<div class="event-detail">${esc(detail)}</div>` : ''}
    <div class="event-time">${new Date(evt.timestamp * 1000).toLocaleTimeString()}</div>`;

    eventFeed.insertBefore(el, eventFeed.firstChild);
    while (eventFeed.children.length > 80) eventFeed.removeChild(eventFeed.lastChild);
}

// ── Helpers ────────────────────────────────────────────────────────────────
function esc(s) {
    return String(s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

// ── Init ───────────────────────────────────────────────────────────────────
pollDocuments();
pollEvents();
setInterval(pollDocuments, POLL_MS);
setInterval(pollEvents, POLL_MS);

// ══════════════════════════════════════════════════════════════════════════
// EVOLUTION LAB
// ══════════════════════════════════════════════════════════════════════════

let activeRunId = null;
let evoPoller = null;
let evoRunTotalGens = 5;
const fitnessHistory = []; // [{gen, best}]

// ── Tab Switching ──────────────────────────────────────────────────────────
function switchTab(tab) {
    document.getElementById('tabContentDocs').classList.toggle('hidden', tab !== 'docs');
    document.getElementById('tabContentEvo').classList.toggle('hidden', tab !== 'evo');
    document.getElementById('tabDocs').classList.toggle('active', tab === 'docs');
    document.getElementById('tabEvo').classList.toggle('active', tab === 'evo');
    if (tab === 'evo') {
        populateEvoDocSelect();
        loadPastRuns();
    }
}

// Populate the test document dropdown with already-processed docs
function populateEvoDocSelect() {
    const sel = document.getElementById('evoDocSelect');
    const current = sel.value;
    // Remove existing doc options (keep the placeholder)
    while (sel.options.length > 1) sel.remove(1);
    docStore.forEach((doc, id) => {
        if (doc.status === 'complete') {
            const opt = document.createElement('option');
            opt.value = id;
            opt.textContent = doc.filename.length > 40 ? doc.filename.slice(0, 38) + '…' : doc.filename;
            if (id === current) opt.selected = true;
            sel.appendChild(opt);
        }
    });
}

// ── Start Evolution Run ────────────────────────────────────────────────────
async function startEvolution() {
    const docId = document.getElementById('evoDocSelect').value;
    if (!docId) { alert('Please select a test document first.'); return; }

    const gens = parseInt(document.getElementById('evoGens').value, 10);
    const pop = parseInt(document.getElementById('evoPop').value, 10);
    const topK = parseInt(document.getElementById('evoTopK').value, 10);
    const mutRate = parseFloat(document.getElementById('evoMut').value);
    const workers = parseInt(document.getElementById('evoWorkers').value, 10);

    document.getElementById('evoStartBtn').disabled = true;
    evoRunTotalGens = gens;
    fitnessHistory.length = 0;

    try {
        const resp = await fetch(`${API}/evolve`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                test_document_id: docId,
                generations: gens,
                population_size: pop,
                top_k: topK,
                mutation_rate: mutRate,
                parallel_workers: workers,
            }),
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            alert(`❌ ${err.detail || 'Failed to start evolution.'}`);
            document.getElementById('evoStartBtn').disabled = false;
            return;
        }
        const data = await resp.json();
        activeRunId = data.run_id;
        startEvoPolling(activeRunId);
        showRunStatus(activeRunId, 'running', 0, gens, 0);
    } catch (e) {
        alert(`❌ Network error: ${e.message}`);
        document.getElementById('evoStartBtn').disabled = false;
    }
}

// ── Polling ────────────────────────────────────────────────────────────────
function startEvoPolling(runId) {
    if (evoPoller) clearInterval(evoPoller);
    evoPoller = setInterval(() => pollEvoStatus(runId), 4000);
    pollEvoStatus(runId); // immediate first poll
}

async function pollEvoStatus(runId) {
    try {
        const resp = await fetch(`${API}/evolve/${runId}/status`);
        if (!resp.ok) return;
        const run = await resp.json();
        updateRunStatus(run);
        if (run.status === 'complete' || run.status === 'error') {
            clearInterval(evoPoller);
            document.getElementById('evoStartBtn').disabled = false;
            loadPastRuns();
        }
    } catch (e) { /* silent */ }
}

function updateRunStatus(run) {
    const genHistory = run.generation_history || [];
    // Rebuild fitness history from generation_history
    fitnessHistory.length = 0;
    genHistory.forEach(g => {
        fitnessHistory.push({ gen: g.generation, best: g.best_fitness });
    });

    showRunStatus(
        run.run_id,
        run.status,
        run.current_generation,
        run.generations,
        run.best_fitness,
    );

    if (fitnessHistory.length > 0) drawFitnessChart();
    if (run.status === 'complete' && run.pareto_front && run.pareto_front.length) {
        renderParetoFront(run.pareto_front, run.run_id);
    }
}

function showRunStatus(runId, status, currentGen, totalGens, bestFit) {
    const statusEl = document.getElementById('evoRunStatus');
    statusEl.classList.add('visible');

    document.getElementById('evoRunId').textContent = `run: ${runId}`;
    const pill = document.getElementById('evoStatusPill');
    pill.textContent = status;
    pill.className = `evo-status-pill ${status}`;

    document.getElementById('evoCurrentGen').textContent = currentGen;
    document.getElementById('evoTotalGens').textContent = totalGens;
    document.getElementById('evoBestFit').textContent =
        bestFit > 0 ? bestFit.toFixed(4) : '—';

    const pct = totalGens > 0 ? Math.round((currentGen / totalGens) * 100) : 0;
    document.getElementById('evoGenFill').style.width = `${pct}%`;
}

// ── Canvas Fitness Chart ───────────────────────────────────────────────────
function drawFitnessChart() {
    const wrap = document.getElementById('evoChartWrap');
    wrap.style.display = '';
    const canvas = document.getElementById('evoChart');
    const ctx = canvas.getContext('2d');
    const W = canvas.offsetWidth; const H = 120;
    canvas.width = W; canvas.height = H;

    ctx.clearRect(0, 0, W, H);

    const data = fitnessHistory.map(d => d.best);
    if (data.length < 1) return;

    const minV = Math.min(...data) - 0.05;
    const maxV = Math.max(...data) + 0.05;
    const range = maxV - minV || 0.1;

    const pad = { t: 10, b: 24, l: 40, r: 16 };
    const chartW = W - pad.l - pad.r;
    const chartH = H - pad.t - pad.b;

    const xStep = data.length > 1 ? chartW / (data.length - 1) : chartW;

    // Grid lines (3 horizontal)
    ctx.strokeStyle = 'rgba(255,255,255,.07)';
    ctx.lineWidth = 1;
    [0.25, 0.5, 0.75].forEach(t => {
        const y = pad.t + chartH * (1 - t);
        ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    });

    // Y-axis labels
    ctx.fillStyle = '#64748b'; ctx.font = '9px Inter,sans-serif'; ctx.textAlign = 'right';
    [0, 0.5, 1].forEach(t => {
        const v = minV + range * t;
        const y = pad.t + chartH * (1 - t);
        ctx.fillText(v.toFixed(2), pad.l - 4, y + 3);
    });

    // X-axis labels
    ctx.textAlign = 'center'; ctx.fillStyle = '#64748b';
    data.forEach((_, i) => {
        const x = pad.l + i * xStep;
        ctx.fillText(i + 1, x, H - 6);
    });

    // Gradient fill under the line
    const grad = ctx.createLinearGradient(0, pad.t, 0, H - pad.b);
    grad.addColorStop(0, 'rgba(167,139,250,.35)');
    grad.addColorStop(1, 'rgba(167,139,250,0)');
    ctx.beginPath();
    data.forEach((v, i) => {
        const x = pad.l + i * xStep;
        const y = pad.t + chartH * (1 - (v - minV) / range);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.lineTo(pad.l + (data.length - 1) * xStep, H - pad.b);
    ctx.lineTo(pad.l, H - pad.b);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // Line
    ctx.beginPath();
    ctx.strokeStyle = '#a78bfa'; ctx.lineWidth = 2;
    data.forEach((v, i) => {
        const x = pad.l + i * xStep;
        const y = pad.t + chartH * (1 - (v - minV) / range);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Dots
    ctx.fillStyle = '#f472b6';
    data.forEach((v, i) => {
        const x = pad.l + i * xStep;
        const y = pad.t + chartH * (1 - (v - minV) / range);
        ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2); ctx.fill();
    });
}

// ── Pareto Front Table ─────────────────────────────────────────────────────
function renderParetoFront(paretoFront, runId) {
    const wrap = document.getElementById('evoParetoWrap');
    wrap.style.display = '';
    document.getElementById('evoParetoCount').textContent =
        `(${paretoFront.length} solution${paretoFront.length !== 1 ? 's' : ''})`;

    const tbody = document.getElementById('evoParetoBody');
    tbody.innerHTML = '';

    paretoFront
        .sort((a, b) => (b.score || 0) - (a.score || 0))
        .forEach((item, idx) => {
            const g = item.genome || {};
            const score = item.score || 0;
            const pct = Math.round(score * 100);

            const tr = document.createElement('tr');
            tr.innerHTML = `
              <td>${idx + 1}</td>
              <td style="font-family:'JetBrains Mono',monospace;color:var(--text3)">${esc(g.genome_id || '—')}</td>
              <td><span class="badge">${esc(g.llm_model || '—')}</span></td>
              <td>${g.chunk_size || '—'}</td>
              <td>${g.retriever_k || '—'}</td>
              <td>
                <div class="fitness-meter">
                  <div class="fitness-bar-bg"><div class="fitness-bar-fill" style="width:${pct}%"></div></div>
                  <div class="fitness-score">${score.toFixed(3)}</div>
                </div>
              </td>
              <td>
                <button class="apply-sop-btn" data-run="${esc(runId)}">Apply</button>
              </td>`;

            // Wire up Apply button
            tr.querySelector('.apply-sop-btn').addEventListener('click', async function () {
                const rId = this.getAttribute('data-run');
                this.disabled = true;
                this.textContent = '⏳';
                try {
                    const resp = await fetch(`${API}/evolve/${rId}/apply-best`, { method: 'POST' });
                    if (resp.ok) {
                        this.textContent = '✅ Applied!';
                        this.style.background = 'var(--success)';
                    } else {
                        const err = await resp.json().catch(() => ({}));
                        alert(`❌ ${err.detail || 'Apply failed.'}`);
                        this.disabled = false; this.textContent = 'Apply';
                    }
                } catch (e) {
                    alert(`❌ ${e.message}`); this.disabled = false; this.textContent = 'Apply';
                }
            });

            tbody.appendChild(tr);
        });
}

// ── Past Runs List ──────────────────────────────────────────────────────────
async function loadPastRuns() {
    try {
        const resp = await fetch(`${API}/evolve?limit=10`);
        if (!resp.ok) return;
        const runs = await resp.json();
        const el = document.getElementById('evoPastRuns');
        if (!runs.length) {
            el.innerHTML = '<div class="empty-state">No evolution runs yet.</div>';
            return;
        }
        el.innerHTML = '';
        runs.forEach(r => {
            const row = document.createElement('div');
            row.className = 'evo-run-row';
            row.innerHTML = `
              <span class="evo-run-row-id">${esc(r.run_id)}</span>
              <span class="evo-status-pill ${r.status}">${r.status}</span>
              <span class="evo-run-row-fit">${r.best_fitness > 0 ? r.best_fitness.toFixed(3) : '—'}</span>`;
            row.addEventListener('click', () => {
                activeRunId = r.run_id;
                evoRunTotalGens = r.generations;
                startEvoPolling(r.run_id);
            });
            el.appendChild(row);
        });
    } catch (e) { /* silent */ }
}
