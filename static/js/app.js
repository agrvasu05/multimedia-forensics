const API_BASE =
  (typeof location !== 'undefined' && location.protocol.startsWith('http'))
    ? location.origin
    : 'http://localhost:8000';
const ANALYZE_URL = API_BASE + '/analyze/text';

const els = {};

function $(id) { return document.getElementById(id); }

function setText(id, val) { const el = $(id); if (el) el.textContent = val; }

function init() {
  els.form = $('analyze-form');
  els.textInput = $('text-input');
  els.submitBtn = $('submit-btn');
  els.loading = $('loading');
  els.results = $('results');
  els.construction = $('construction');
  els.errorMsg = $('error-msg');

  els.form.addEventListener('submit', handleSubmit);
}

async function handleSubmit(e) {
  e.preventDefault();
  const text = els.textInput.value.trim();
  if (!text) return;

  setLoading(true);
  hideError();
  hideResults();

  try {
    const res = await fetch(ANALYZE_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, localize: true }),
    });
    if (!res.ok) {
      const body = await res.text().catch(() => '');
      throw new Error(`HTTP ${res.status}: ${body.slice(0, 200)}`);
    }
    const data = await res.json();
    renderResults(data);
  } catch (err) {
    console.error(err);
    showError(`Analysis failed: ${err.message || err}. Is the API running at ${API_BASE}?`);
  } finally {
    setLoading(false);
  }
}

function setLoading(on) {
  els.submitBtn.disabled = on;
  els.loading.style.display = on ? 'inline-flex' : 'none';
}

function showError(msg) {
  els.errorMsg.textContent = msg;
  els.errorMsg.style.display = 'block';
}
function hideError() { els.errorMsg.style.display = 'none'; }
function hideResults() { els.results.style.display = 'none'; }
function showResults() { els.results.style.display = 'block'; }

function renderResults(data) {
  showResults();

  const label = data.label;
  const pAi = data.p_ai;
  const conf = data.confidence;
  const isAI = label === 'ai_generated';

  setText('result-label', isAI ? 'AI-GENERATED' : 'HUMAN');
  const labelEl = $('result-label');
  if (labelEl) labelEl.className = 'badge ' + (isAI ? 'badge-ai' : 'badge-human');
  setText('result-confidence', (conf * 100).toFixed(1) + '%');
  setText('result-pai', (pAi * 100).toFixed(2) + '%');

  const scoreBar = $('score-fill');
  if (scoreBar) {
    scoreBar.style.width = (pAi * 100) + '%';
    scoreBar.className = 'score-bar-fill' + (isAI ? ' danger' : '');
  }

  setText('explanation', data.explanation || '');

  const branches = $('branch-list');
  branches.innerHTML = '';
  const b = data.branches || {};
  const branchData = [
    { name: 'Supervised', value: b.supervised, color: '#22C55E' },
    { name: 'Zero-shot', value: b.zero_shot, color: '#3B82F6' },
    { name: 'DetectGPT', value: b.detectgpt_curvature, color: '#A855F7' },
    { name: 'Perplexity', value: b.gltr?.perplexity, color: '#F59E0B', isPerplexity: true },
    { name: 'Burstiness', value: b.burstiness, color: '#EC4899' },
  ];
  branchData.forEach(bd => {
    const val = bd.value;
    if (val === null || val === undefined) return;
    const num = parseFloat(val);
    let display = num.toFixed(4);
    let barWidth = Math.min(Math.abs(num) * 100, 100);
    if (bd.isPerplexity) {
      display = num.toFixed(1);
      barWidth = Math.min(num / 100, 1);
    }
    if (bd.name === 'Burstiness') {
      barWidth = Math.min(Math.abs(num) * 50 + 25, 100);
    }
    const li = document.createElement('div');
    li.className = 'result-item';
    li.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
        <span style="font-size:13px;color:var(--text-secondary)">${bd.name}</span>
        <span class="font-mono" style="font-size:13px;color:${bd.color}">${display}</span>
      </div>
      <div class="score-bar"><div class="score-bar-fill" style="width:${barWidth}%;background:${bd.color}"></div></div>
    `;
    branches.appendChild(li);
  });

  const spans = data.span_scores;
  if (spans && spans.length > 0) {
    const spanGrid = $('span-grid');
    spanGrid.innerHTML = '';
    spanGrid.style.display = 'grid';
    spans.forEach(s => {
      const sPai = parseFloat(s.p_ai);
      const sIsAI = sPai >= 0.5;
      const div = document.createElement('div');
      div.style.cssText = 'background:var(--bg-primary);border-radius:8px;padding:12px;border-left:3px solid ' + (sIsAI ? '#EF4444' : '#22C55E');
      div.innerHTML = `
        <div style="font-size:12px;color:var(--text-muted);margin-bottom:4px">Sentences ${s.sentence_start}–${s.sentence_end}</div>
        <div style="font-size:12px;color:var(--text-secondary);margin-bottom:6px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${s.text_preview}</div>
        <div style="font-size:12px;font-family:'Fira Code',monospace;color:${sIsAI ? '#F87171' : '#4ADE80'}">p_ai: ${sPai.toFixed(3)}</div>
      `;
      spanGrid.appendChild(div);
    });
  } else {
    $('span-section').style.display = 'none';
  }

  $('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

document.addEventListener('DOMContentLoaded', init);