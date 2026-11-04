// Pure helpers (no DOM, no network) so they can be unit-tested with Node:  node --test web_app/tests/lib.test.mjs
// LLM output is UNTRUSTED text: everything that reaches innerHTML goes through escapeHtml.

export function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"'`]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;', '`': '&#96;' }[c]));
}

/** Incremental Server-Sent-Events parser: chunks from fetch() can split an event anywhere. */
export class SSEParser {
  constructor() { this.buffer = ''; }
  push(chunk) {
    this.buffer += chunk.replace(/\r\n/g, '\n');
    const events = [];
    let i;
    while ((i = this.buffer.indexOf('\n\n')) >= 0) {
      const block = this.buffer.slice(0, i);
      this.buffer = this.buffer.slice(i + 2);
      const ev = parseBlock(block);
      if (ev) events.push(ev);
    }
    return events;
  }
}
function parseBlock(block) {
  let event = 'message';
  const data = [];
  for (const line of block.split('\n')) {
    if (line.startsWith('event:')) event = line.slice(6).trim();
    else if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
  }
  if (!data.length) return null;
  try { return { event, data: JSON.parse(data.join('\n')) }; } catch { return { event, data: null }; }
}

/** Detector boxes are in the processed image's pixels; map them onto a canvas of another size. */
export function scaleBox(box, imgW, imgH, drawW, drawH) {
  const [x1, y1, x2, y2] = box;
  const sx = drawW / imgW, sy = drawH / imgH;
  return { x: x1 * sx, y: y1 * sy, w: (x2 - x1) * sx, h: (y2 - y1) * sy };
}

export function formatApiError(status, body) {
  const e = body && body.error;
  if (e) return `${e.message}${e.request_id ? ` (request ${e.request_id})` : ''}`;
  if (status === 0) return 'Cannot reach the server. Check the address in Settings and your connection.';
  return `Request failed (HTTP ${status}).`;
}

export function cleanAnimal(f) {
  const out = {};
  const num = (v) => (v === '' || v == null || Number.isNaN(Number(v)) ? null : Number(v));
  if (num(f.weight_kg) != null && num(f.weight_kg) > 0) out.weight_kg = num(f.weight_kg);
  if (num(f.age_months) != null && num(f.age_months) >= 0) out.age_months = Math.round(num(f.age_months));
  if (f.lactating) out.lactating = true;
  if (f.pregnant) out.pregnant = true;
  return Object.keys(out).length ? out : null;
}

export function displayName(cls, classes) {
  const c = (classes || []).find((x) => x.name === cls);
  return (c && c.display && c.display.en) || cls.replace(/_/g, ' ');
}

export function pct(x) { return `${Math.round(x * 100)}%`; }

export function renderFindings(res, classes) {
  if (!res.detections.length) {
    return '<p class="muted">Nothing was detected. This does not prove the animal is healthy: retake a closer, well-lit photo of the affected area, or ask a vet.</p>';
  }
  const rows = res.detections.map((d, i) =>
    `<li><span class="dot c${i % 6}"></span><b>${escapeHtml(displayName(d.class, classes))}</b> <span class="muted">${pct(d.confidence)}</span></li>`).join('');
  return `<ul class="findings">${rows}</ul>`;
}

export function renderFlags(flags) {
  const out = [];
  if (flags.includes('possible_notifiable_disease')) {
    out.push('<div class="banner bad"><b>Possible notifiable disease.</b> Isolate the animal and contact the sector veterinary officer / RAB immediately.</div>');
  }
  if (flags.includes('possible_zoonotic_risk')) {
    out.push('<div class="banner warn"><b>Possible risk to people.</b> Wear gloves and wash hands after handling this animal.</div>');
  }
  return out.join('');
}

const li = (items) => (items && items.length ? `<ul>${items.map((x) => `<li>${escapeHtml(x)}</li>`).join('')}</ul>` : '<p class="muted">None.</p>');

/** res = the LLM API's /v1/report response. Returns HTML (every dynamic string escaped). */
export function renderReport(res) {
  const r = res.report;
  const degraded = res.degraded || res.source !== 'llm';
  const badge = degraded
    ? '<span class="badge warn">Standard guidance: the AI report is unavailable right now</span>'
    : `<span class="badge ok">AI report (${escapeHtml(res.model_version)})</span>`;
  const sev = ['low', 'moderate', 'high', 'urgent'].includes(r.severity) ? r.severity : 'moderate';
  const banners = [
    r.notifiable ? '<div class="banner bad"><b>Notifiable disease suspected.</b> Isolate the animal and contact the sector veterinary officer / RAB now.</div>' : '',
    r.zoonotic ? '<div class="banner warn"><b>Can spread to people.</b> Use gloves and wash hands.</div>' : '',
  ].join('');
  const t = r.treatment || {};
  const adj = (res.safety_adjustments || []).length
    ? `<details class="muted"><summary>Safety checks applied (${res.safety_adjustments.length})</summary>${li(res.safety_adjustments)}</details>` : '';
  return `
    <div class="report">
      <div class="rhead">${badge}<span class="sev ${sev}">Severity: ${escapeHtml(sev)}</span></div>
      ${banners}
      <p class="summary">${escapeHtml(r.summary)}</p>
      <h4>Possible conditions</h4>${li(r.possible_conditions)}
      <h4>What to do now</h4>${li(r.immediate_actions)}
      <h4>Care and treatment${t.requires_vet ? ' <span class="badge warn">veterinarian needed</span>' : ''}</h4>
      <h5>Supportive care</h5>${li(t.supportive_care)}
      <h5>Medicines (never without a vet)</h5>${li(t.medicines)}
      <h5>Milk and meat withdrawal</h5>${li(t.withdrawal_periods)}
      <h4>Prevention</h4>${li(r.prevention)}
      ${r.follow_up_days != null ? `<p><b>Check again in ${escapeHtml(r.follow_up_days)} day(s).</b></p>` : ''}
      ${adj}
      <p class="disclaimer">${escapeHtml(r.disclaimer || '')}</p>
    </div>`;
}
