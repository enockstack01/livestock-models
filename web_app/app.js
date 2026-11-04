import { SSEParser, cleanAnimal, displayName, escapeHtml, formatApiError, renderFindings, renderFlags, renderReport, scaleBox } from './lib.js';

const $ = (s) => document.querySelector(s);
const kv = {
  get: (st, k) => { try { return st.getItem(k); } catch { return null; } },
  set: (st, k, v) => { try { st.setItem(k, v); } catch { /* storage blocked: the app still works for this page view */ } },
};
const S = {
  vision: kv.get(localStorage, 'ldx_vision') || 'http://127.0.0.1:8001',
  llm: kv.get(localStorage, 'ldx_llm') || 'http://127.0.0.1:8002',
  token: kv.get(sessionStorage, 'ldx_token') || '',        // per tab, never persisted
};
const state = { blob: null, img: null, det: null, classes: [], reportRes: null, demo: false };
const COLORS = ['#e53935', '#1e88e5', '#fb8c00', '#8e24aa', '#00897b', '#6d4c41'];

// ---------------------------------------------------------------- networking
async function call(base, path, { method = 'GET', body, json } = {}) {
  const headers = {};
  if (S.token) headers.Authorization = `Bearer ${S.token.trim()}`;
  if (json) headers['Content-Type'] = 'application/json';
  let r;
  try {
    r = await fetch(base.replace(/\/$/, '') + path, { method, headers, body: json ? JSON.stringify(json) : body });
  } catch { throw new Error(formatApiError(0, null)); }
  if (!r.ok) throw new Error(formatApiError(r.status, await r.json().catch(() => null)));
  return r;
}

async function pollHealth() {
  const probe = async (base, el, label, okText) => {
    try {
      const r = await fetch(base.replace(/\/$/, '') + '/ready', { cache: 'no-store' });
      const j = await r.json();
      const llm = j.llm_reachable;
      if (label === 'Vision') { el.textContent = r.ok ? 'Vision: online' : 'Vision: no model'; el.className = 'pill ' + (r.ok ? 'on' : 'warn'); }
      else { el.textContent = llm ? 'Advice: AI online' : 'Advice: standard guidance only'; el.className = 'pill ' + (llm ? 'on' : 'warn'); }
    } catch { el.textContent = `${label}: unreachable`; el.className = 'pill off'; }
  };
  await Promise.all([probe(S.vision, $('#pillV'), 'Vision'), probe(S.llm, $('#pillL'), 'Advice')]);
}

async function loadModelInfo() {
  if (!S.token) return;
  try {
    const m = await (await call(S.vision, '/v1/model')).json();
    state.classes = m.classes || [];
    $('#species').innerHTML = '<option value="">Not sure</option>' + (m.species || []).map((s) => `<option value="${escapeHtml(s)}">${escapeHtml(s[0].toUpperCase() + s.slice(1))}</option>`).join('');
  } catch { /* shown when the user presses Check */ }
}

// ---------------------------------------------------------------- photo
async function toResizedBlob(file, maxSide = 1280) {
  const bmp = await createImageBitmap(file, { imageOrientation: 'from-image' });
  const k = Math.min(1, maxSide / Math.max(bmp.width, bmp.height));
  const c = document.createElement('canvas');
  c.width = Math.round(bmp.width * k); c.height = Math.round(bmp.height * k);
  c.getContext('2d').drawImage(bmp, 0, 0, c.width, c.height);
  return new Promise((res) => c.toBlob(res, 'image/jpeg', 0.85));       // longest side <= 1280 px, JPEG ~85 %: as the platform spec says
}

$('#photo').addEventListener('change', async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  try {
    state.blob = await toResizedBlob(f);
    $('#preview').src = URL.createObjectURL(state.blob); $('#preview').hidden = false;
    $('#dropText').textContent = f.name; $('#detectBtn').disabled = false; $('#msg').textContent = '';
  } catch { $('#msg').textContent = 'That file could not be read as an image.'; $('#msg').className = 'err'; }
});

// ---------------------------------------------------------------- detection (Livestock Vision API, YOLO11x)
$('#detectBtn').addEventListener('click', async () => {
  if (!S.token) { $('#msg').textContent = 'Open Settings and paste your access token first.'; $('#msg').className = 'err'; return; }
  const fd = new FormData();
  fd.append('image', state.blob, 'photo.jpg');
  fd.append('species', $('#species').value); fd.append('district', $('#district').value); fd.append('animal_id', $('#animalId').value);
  $('#detectBtn').disabled = true; $('#msg').className = 'muted'; $('#msg').textContent = 'Analysing the photo...';
  try {
    state.det = await (await call(S.vision, '/v1/detect', { method: 'POST', body: fd })).json();
    $('#msg').textContent = '';
    await showDetection();
    await generateReport(false);
  } catch (e) { $('#msg').textContent = e.message; $('#msg').className = 'err'; }
  $('#detectBtn').disabled = false;
});

async function showDetection() {
  const d = state.det;
  state.img = await new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = rej; im.src = $('#preview').src; });
  const c = $('#canvas'), W = Math.min(720, state.img.width);
  c.width = W; c.height = Math.round(W * d.image.height / d.image.width);
  const g = c.getContext('2d'); g.drawImage(state.img, 0, 0, c.width, c.height);
  g.lineWidth = Math.max(2, W / 250); g.font = `${Math.max(12, W / 45)}px system-ui`;
  d.detections.forEach((x, i) => {
    const b = scaleBox(x.box, d.image.width, d.image.height, c.width, c.height);
    g.strokeStyle = COLORS[i % COLORS.length]; g.strokeRect(b.x, b.y, b.w, b.h);
    const label = `${displayName(x.class, state.classes)} ${Math.round(x.confidence * 100)}%`;
    const tw = g.measureText(label).width + 8, th = Math.max(16, W / 38);
    g.fillStyle = COLORS[i % COLORS.length]; g.fillRect(b.x, Math.max(0, b.y - th), tw, th);
    g.fillStyle = '#fff'; g.fillText(label, b.x + 4, Math.max(th - 4, b.y - 4));
  });
  $('#flags').innerHTML = renderFlags(d.flags);
  $('#findingsList').innerHTML = renderFindings(d, state.classes);
  $('#modelLine').textContent = `Detected by ${d.model_version} in ${Math.round(d.latency_ms)} ms. Case ${d.diagnosis_id}.`;
  renderFeedback();
  for (const id of ['stepFindings', 'stepReport', 'stepChat']) $('#' + id).hidden = false;
}

// ---------------------------------------------------------------- report (LLM API, Qwen)
async function generateReport(refresh) {
  const body = { diagnosis_id: state.det.diagnosis_id, language: $('#language').value, stream: true, refresh,
    animal: cleanAnimal({ weight_kg: $('#weight').value, age_months: $('#age').value, lactating: $('#lactating').checked, pregnant: $('#pregnant').checked }) };
  $('#reportBox').innerHTML = ''; $('#reportStatus').className = 'muted'; $('#reportStatus').textContent = 'Writing the advice report...'; $('#regen').hidden = true;
  let r;
  try { r = await call(S.llm, '/v1/report', { method: 'POST', json: body }); }
  catch (e) { $('#reportStatus').textContent = e.message; $('#reportStatus').className = 'err'; $('#regen').hidden = false; return; }
  const parser = new SSEParser(), dec = new TextDecoder();
  let final = null;
  for await (const chunk of r.body) {
    for (const ev of parser.push(dec.decode(chunk, { stream: true }))) {
      if (ev.event === 'delta' && ev.data) $('#reportStatus').textContent = `Writing the advice report... (${ev.data.chars} characters)`;
      if (ev.event === 'final') final = ev.data;
    }
  }
  if (!final) { $('#reportStatus').textContent = 'The report did not finish. Try again.'; $('#reportStatus').className = 'err'; $('#regen').hidden = false; return; }
  state.reportRes = final;
  $('#reportStatus').textContent = final.cached ? 'Saved report from earlier.' : '';
  $('#reportBox').innerHTML = renderReport(final);
  $('#regen').hidden = false;
}
$('#regen').addEventListener('click', () => state.det && generateReport(true));
$('#language').addEventListener('change', () => state.det && generateReport(false));

// ---------------------------------------------------------------- chat
function addMsg(role, text) {
  const d = document.createElement('div'); d.className = `msg ${role}`; d.textContent = text;       // textContent: model text is never parsed as HTML
  $('#chatLog').appendChild(d); $('#chatLog').scrollTop = 1e9; return d;
}
const history = [];
async function ask() {
  const q = $('#chatInput').value.trim();
  if (!q || !state.det) return;
  $('#chatInput').value = ''; addMsg('user', q); history.push({ role: 'user', content: q });
  const bubble = addMsg('bot', '...'); let answer = '';
  try {
    const r = await call(S.llm, '/v1/chat', { method: 'POST', json: { messages: history.slice(-12), diagnosis_id: state.det.diagnosis_id, language: $('#language').value, stream: true } });
    const parser = new SSEParser(), dec = new TextDecoder();
    for await (const chunk of r.body) for (const ev of parser.push(dec.decode(chunk, { stream: true }))) {
      if (ev.event === 'delta' && ev.data) { answer += ev.data.text; bubble.textContent = answer; $('#chatLog').scrollTop = 1e9; }
      if (ev.event === 'error' && ev.data) bubble.textContent = answer + `\n[${ev.data.message}]`;
    }
    history.push({ role: 'assistant', content: answer });
  } catch (e) { bubble.textContent = e.message; history.pop(); }
}
$('#chatSend').addEventListener('click', ask);
$('#chatInput').addEventListener('keydown', (e) => { if (e.key === 'Enter') ask(); });

// ---------------------------------------------------------------- feedback (confirm / correct -> retraining data)
function renderFeedback() {
  $('#fbBox').innerHTML = '<div class="muted">Is this right? Your answer improves the model.</div><div class="actions"><button class="sec" id="fbYes">Yes, correct</button><button class="sec" id="fbNo">No, something is wrong</button></div><div id="fbEdit"></div><div id="fbMsg" class="muted"></div>';
  $('#fbYes').onclick = () => sendFeedback({ status: 'confirmed' });
  $('#fbNo').onclick = () => {
    const opts = (state.classes.filter((c) => c.status !== 'retired').length ? state.classes : []).map((c) => `<option value="${escapeHtml(c.name)}">${escapeHtml(displayName(c.name, state.classes))}</option>`).join('');
    $('#fbEdit').innerHTML = state.det.detections.map((d, i) => `<div class="row"><select data-i="${i}">${opts}<option value="__remove">(nothing here: remove)</option></select> instead of ${escapeHtml(displayName(d.class, state.classes))}</div>`).join('')
      + '<button class="primary" id="fbSend">Send correction</button>';
    document.querySelectorAll('#fbEdit select').forEach((s, i) => { s.value = state.det.detections[i].class; });
    $('#fbSend').onclick = () => {
      const findings = [...document.querySelectorAll('#fbEdit select')].map((s) => ({ class: s.value, box: state.det.detections[+s.dataset.i].box })).filter((f) => f.class !== '__remove');
      sendFeedback({ status: 'corrected', findings });
    };
  };
}
async function sendFeedback(b) {
  try {
    await call(S.vision, '/v1/feedback', { method: 'POST', json: { diagnosis_id: state.det.diagnosis_id, ...b } });
    $('#fbBox').innerHTML = '<span class="badge ok">Thank you. Your answer was saved.</span>';
  } catch (e) { $('#fbMsg').textContent = e.message; $('#fbMsg').className = 'err'; }
}

// ---------------------------------------------------------------- settings
$('#openSettings').addEventListener('click', () => { $('#visionUrl').value = S.vision; $('#llmUrl').value = S.llm; $('#token').value = S.token; $('#settings').showModal(); });
$('#saveSettings').addEventListener('click', () => {
  S.vision = $('#visionUrl').value.trim() || S.vision; S.llm = $('#llmUrl').value.trim() || S.llm; S.token = $('#token').value.trim();
  kv.set(localStorage, 'ldx_vision', S.vision); kv.set(localStorage, 'ldx_llm', S.llm); kv.set(sessionStorage, 'ldx_token', S.token);
  pollHealth(); loadModelInfo();
});

// ---------------------------------------------------------------- demo mode (#demo): a recorded real response, for screenshots and docs
async function demo() {
  const d = await (await fetch('demo/sample.json')).json();
  state.demo = true; state.classes = d.classes; state.det = d.detection;
  const blob = await (await fetch('demo/sample.jpg')).blob();
  $('#preview').src = URL.createObjectURL(blob); $('#preview').hidden = false; $('#dropText').textContent = 'sample.jpg';
  await showDetection();
  state.reportRes = d.report; $('#reportBox').innerHTML = renderReport(d.report); $('#regen').hidden = true;
  addMsg('user', 'Can I still milk her?'); addMsg('bot', d.chat);
}

pollHealth(); setInterval(pollHealth, 15000); loadModelInfo();
if (location.hash === '#demo') demo();
