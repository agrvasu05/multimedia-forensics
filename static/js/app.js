const API_BASE =
  (typeof location !== 'undefined' && location.protocol.startsWith('http'))
    ? location.origin
    : 'http://localhost:8000';

const els = {};
let selectedFile = null;

function $(id) { return document.getElementById(id); }
function setText(id, val) { const el = $(id); if (el) el.textContent = val; }

let webcamStream = null;

function startWebcam() {
  const modal = $('webcam-modal');
  const video = $('webcam-video');
  modal.classList.remove('hidden');
  navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
    .then(stream => {
      webcamStream = stream;
      video.srcObject = stream;
    })
    .catch(err => {
      modal.classList.add('hidden');
      showImageError('Webcam access denied: ' + err.message);
    });
}

function stopWebcam() {
  const modal = $('webcam-modal');
  const video = $('webcam-video');
  if (webcamStream) {
    webcamStream.getTracks().forEach(t => t.stop());
    webcamStream = null;
  }
  video.srcObject = null;
  modal.classList.add('hidden');
}

async function captureWebcam() {
  const video = $('webcam-video');
  const canvas = $('webcam-canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  stopWebcam();

  const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.95));
  const file = new File([blob], 'webcam.jpg', { type: 'image/jpeg' });
  selectedFile = file;

  els.imagePreview.src = URL.createObjectURL(blob);
  els.imageFilename.textContent = 'webcam.jpg';
  els.imageSize.textContent = (blob.size / 1024).toFixed(1) + ' KB';
  els.imagePreviewContainer.classList.remove('hidden');
  els.imageSubmitBtn.disabled = false;

  const mode = document.querySelector('input[name="detect-mode"]:checked')?.value || 'ai';

  setImageLoading(true);
  hideImageError();
  hideResults();

  try {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('mode', mode);
    const res = await fetch(API_BASE + '/analyze/image', {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const body = await res.text().catch(() => '');
      throw new Error(`HTTP ${res.status}: ${body.slice(0, 200)}`);
    }
    const data = await res.json();
    data._mode = mode;
    renderImageResults(data);
  } catch (err) {
    showImageError('Analysis failed: ' + err.message);
  } finally {
    setImageLoading(false);
  }
}

function init() {
  els.textForm = $('analyze-form');
  els.textInput = $('text-input');
  els.submitBtn = $('submit-btn');
  els.loading = $('loading');
  els.errorMsg = $('error-msg');
  els.results = $('results');

  els.imageForm = $('image-form');
  els.imageInput = $('image-input');
  els.dropZone = $('drop-zone');
  els.imageSubmitBtn = $('image-submit-btn');
  els.imageLoading = $('image-loading');
  els.imageErrorMsg = $('image-error-msg');
  els.imagePreviewContainer = $('image-preview-container');
  els.imagePreview = $('image-preview');
  els.imageFilename = $('image-filename');
  els.imageSize = $('image-size');

  els.textForm.addEventListener('submit', handleTextSubmit);

  els.dropZone.addEventListener('click', () => els.imageInput.click());
  els.imageInput.addEventListener('change', handleFileSelect);
  els.dropZone.addEventListener('dragover', (e) => { e.preventDefault(); els.dropZone.classList.add('dragover'); });
  els.dropZone.addEventListener('dragleave', () => els.dropZone.classList.remove('dragover'));
  els.dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    els.dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
      els.imageInput.files = e.dataTransfer.files;
      handleFileSelect();
    }
  });

  els.imageForm.addEventListener('submit', handleImageSubmit);

  document.querySelectorAll('input[name="detect-mode"]').forEach(radio => {
    radio.addEventListener('change', (e) => {
      document.querySelectorAll('#image-form .tab-btn').forEach(btn => btn.classList.remove('active'));
      e.target.closest('label').classList.add('active');
    });
  });
}

function switchTab(tab) {
  const textTab = $('tab-text');
  const imageTab = $('tab-image');
  if (tab === 'text') {
    textTab.classList.add('active');
    imageTab.classList.remove('active');
    els.textForm.style.display = 'block';
    els.imageForm.style.display = 'none';
  } else {
    textTab.classList.remove('active');
    imageTab.classList.add('active');
    els.textForm.style.display = 'none';
    els.imageForm.style.display = 'block';
  }
  hideResults();
}

function handleFileSelect() {
  const file = els.imageInput.files[0];
  if (!file) return;
  selectedFile = file;
  els.imagePreview.src = URL.createObjectURL(file);
  els.imageFilename.textContent = file.name;
  els.imageSize.textContent = (file.size / 1024).toFixed(1) + ' KB';
  els.imagePreviewContainer.classList.remove('hidden');
  els.imageSubmitBtn.disabled = false;
}

function clearImage() {
  selectedFile = null;
  els.imageInput.value = '';
  els.imagePreviewContainer.classList.add('hidden');
  els.imageSubmitBtn.disabled = true;
}

async function handleTextSubmit(e) {
  e.preventDefault();
  const text = els.textInput.value.trim();
  if (!text) return;

  setLoading(true);
  hideError();
  hideResults();

  try {
    const res = await fetch(API_BASE + '/analyze/text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, localize: true }),
    });
    if (!res.ok) {
      const body = await res.text().catch(() => '');
      throw new Error(`HTTP ${res.status}: ${body.slice(0, 200)}`);
    }
    const data = await res.json();
    renderTextResults(data);
  } catch (err) {
    console.error(err);
    showError(`Analysis failed: ${err.message || err}. Is the API running at ${API_BASE}?`);
  } finally {
    setLoading(false);
  }
}

async function handleImageSubmit(e) {
  e.preventDefault();
  if (!selectedFile) return;

  const mode = document.querySelector('input[name="detect-mode"]:checked').value;

  setImageLoading(true);
  hideImageError();
  hideResults();

  try {
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('mode', mode);
    const res = await fetch(API_BASE + '/analyze/image', {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const body = await res.text().catch(() => '');
      throw new Error(`HTTP ${res.status}: ${body.slice(0, 200)}`);
    }
    const data = await res.json();
    data._mode = mode;
    renderImageResults(data);
  } catch (err) {
    console.error(err);
    showImageError(`Analysis failed: ${err.message || err}`);
  } finally {
    setImageLoading(false);
  }
}

function setLoading(on) {
  els.submitBtn.disabled = on;
  els.loading.style.display = on ? 'inline-flex' : 'none';
}

function setImageLoading(on) {
  els.imageSubmitBtn.disabled = on;
  els.imageLoading.style.display = on ? 'inline-flex' : 'none';
}

function showError(msg) {
  els.errorMsg.textContent = msg;
  els.errorMsg.style.display = 'block';
}
function hideError() { els.errorMsg.style.display = 'none'; }

function showImageError(msg) {
  els.imageErrorMsg.textContent = msg;
  els.imageErrorMsg.style.display = 'block';
}
function hideImageError() { els.imageErrorMsg.style.display = 'none'; }

function hideResults() { els.results.style.display = 'none'; }
function showResults() { els.results.style.display = 'block'; }

function renderTextResults(data) {
  showResults();
  const isAI = data.label === 'ai_generated';
  setText('result-label', isAI ? 'AI-GENERATED' : 'HUMAN');
  const labelEl = $('result-label');
  if (labelEl) labelEl.className = 'badge ' + (isAI ? 'badge-ai' : 'badge-human');
  setText('result-confidence', (data.confidence * 100).toFixed(1) + '%');
  setText('result-pai', (data.p_ai * 100).toFixed(2) + '%');
  const scoreBar = $('score-fill');
  if (scoreBar) {
    scoreBar.style.width = (data.p_ai * 100) + '%';
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
    if (bd.isPerplexity) { display = num.toFixed(1); barWidth = Math.min(num / 100, 1); }
    if (bd.name === 'Burstiness') { barWidth = Math.min(Math.abs(num) * 50 + 25, 100); }
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
  const spanSection = $('span-section');
  if (spans && spans.length > 0) {
    const spanGrid = $('span-grid');
    spanGrid.innerHTML = '';
    spanSection.style.display = 'block';
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
    if (spanSection) spanSection.style.display = 'none';
  }

  $('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}
function renderImageResults(data) {
  showResults();
  const mode = data._mode || 'ai';
  const branches = $('branch-list');
  branches.innerHTML = '';

  if (mode === 'tamper') {
    const tamper = data.tamper_detection || {};
    const isTampered = tamper.label === 'tampered';
    setText('result-label', isTampered ? 'TAMPERED' : 'REAL');
    const labelEl = $('result-label');
    if (labelEl) labelEl.className = 'badge ' + (isTampered ? 'badge-ai' : 'badge-human');
    setText('result-confidence', (tamper.confidence * 100).toFixed(1) + '%');
    setText('result-pai', (tamper.p_tampered * 100).toFixed(2) + '%');
    const scoreBar = $('score-fill');
    if (scoreBar) {
      scoreBar.style.width = (tamper.p_tampered * 100) + '%';
      scoreBar.className = 'score-bar-fill' + (isTampered ? ' danger' : '');
    }
    setText('explanation', data.explanation || '');

    const li = document.createElement('div');
    li.className = 'result-item';
    li.innerHTML = `
      <div style="display:flex;justify-content:space-between;margin-bottom:4px">
        <span style="font-size:13px;color:var(--text-secondary)">P(Tampered)</span>
        <span class="font-mono" style="font-size:13px;color:${isTampered ? '#EF4444' : '#22C55E'}">${(tamper.p_tampered * 100).toFixed(2)}%</span>
      </div>
      <div class="score-bar"><div class="score-bar-fill" style="width:${tamper.p_tampered * 100}%;background:${isTampered ? '#EF4444' : '#22C55E'}"></div></div>
      <div style="display:flex;justify-content:space-between;margin-top:6px;margin-bottom:4px">
        <span style="font-size:13px;color:var(--text-secondary)">P(Real)</span>
        <span class="font-mono" style="font-size:13px;color:#3B82F6">${(tamper.p_real * 100).toFixed(2)}%</span>
      </div>
      <div style="display:flex;justify-content:space-between;margin-top:6px">
        <span style="font-size:13px;color:var(--text-secondary)">Model</span>
        <span class="font-mono" style="font-size:12px;color:var(--text-muted)">DualBranch (CASIA v2)</span>
      </div>
    `;
    branches.appendChild(li);

  } else if (mode === 'screen') {
    const screen = data.screen_detection || {};
    const isScreen = screen.label === 'screen';
    setText('result-label', isScreen ? 'SCREEN' : 'REAL PHOTO');
    const labelEl = $('result-label');
    if (labelEl) labelEl.className = 'badge ' + (isScreen ? 'badge-ai' : 'badge-human');
    setText('result-confidence', (screen.confidence * 100).toFixed(1) + '%');
    setText('result-pai', (screen.p_screen * 100).toFixed(2) + '%');
    const scoreBar = $('score-fill');
    if (scoreBar) {
      scoreBar.style.width = (screen.p_screen * 100) + '%';
      scoreBar.className = 'score-bar-fill' + (isScreen ? ' danger' : '');
    }
    setText('explanation', data.explanation || '');

    const li = document.createElement('div');
    li.className = 'result-item';
    li.innerHTML = `
      <div style="display:flex;justify-content:space-between;margin-bottom:4px">
        <span style="font-size:13px;color:var(--text-secondary)">P(Screen)</span>
        <span class="font-mono" style="font-size:13px;color:${isScreen ? '#EF4444' : '#22C55E'}">${(screen.p_screen * 100).toFixed(2)}%</span>
      </div>
      <div class="score-bar"><div class="score-bar-fill" style="width:${screen.p_screen * 100}%;background:${isScreen ? '#EF4444' : '#22C55E'}"></div></div>
      <div style="display:flex;justify-content:space-between;margin-top:6px;margin-bottom:4px">
        <span style="font-size:13px;color:var(--text-secondary)">P(Real Photo)</span>
        <span class="font-mono" style="font-size:13px;color:#3B82F6">${(screen.p_real * 100).toFixed(2)}%</span>
      </div>
      <div style="display:flex;justify-content:space-between;margin-top:6px">
        <span style="font-size:13px;color:var(--text-secondary)">Model</span>
        <span class="font-mono" style="font-size:12px;color:var(--text-muted)">MobileNetV2 (Screen)</span>
      </div>
    `;
    branches.appendChild(li);

  } else {
    const ai = data.ai_detection || {};
    const isAI = ai.label === 'ai_generated';
    const isUncertain = ai.label === 'uncertain';
    setText('result-label', isUncertain ? 'INCONCLUSIVE' : (isAI ? 'AI-GENERATED' : 'REAL'));
    const labelEl = $('result-label');
    if (labelEl) labelEl.className = 'badge ' + (isUncertain ? 'badge-uncertain' : (isAI ? 'badge-ai' : 'badge-human'));
    setText('result-confidence', (ai.confidence * 100).toFixed(1) + '%');
    setText('result-pai', (ai.p_ai * 100).toFixed(2) + '%');
    const scoreBar = $('score-fill');
    if (scoreBar) {
      scoreBar.style.width = (ai.p_ai * 100) + '%';
      scoreBar.className = 'score-bar-fill' + (isAI ? ' danger' : '');
    }
    setText('explanation', data.explanation || '');

    const li = document.createElement('div');
    li.className = 'result-item';
    li.innerHTML = `
      <div style="display:flex;justify-content:space-between;margin-bottom:4px">
        <span style="font-size:13px;color:var(--text-secondary)">P(AI)</span>
        <span class="font-mono" style="font-size:13px;color:${isAI ? '#EF4444' : '#22C55E'}">${(ai.p_ai * 100).toFixed(2)}%</span>
      </div>
      <div class="score-bar"><div class="score-bar-fill" style="width:${ai.p_ai * 100}%;background:${isAI ? '#EF4444' : '#22C55E'}"></div></div>
      <div style="display:flex;justify-content:space-between;margin-top:6px;margin-bottom:4px">
        <span style="font-size:13px;color:var(--text-secondary)">P(Real)</span>
        <span class="font-mono" style="font-size:13px;color:#3B82F6">${(ai.p_real * 100).toFixed(2)}%</span>
      </div>
      <div style="display:flex;justify-content:space-between;margin-top:6px">
        <span style="font-size:13px;color:var(--text-secondary)">Model</span>
        <span class="font-mono" style="font-size:12px;color:var(--text-muted)">ViT-Base (CIFAKE)</span>
      </div>
    `;
    branches.appendChild(li);
  }

  const spanSection = $('span-section');
  if (spanSection) spanSection.style.display = 'none';

  $('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

document.addEventListener('DOMContentLoaded', init);
