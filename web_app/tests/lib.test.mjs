import assert from 'node:assert/strict';
import { test } from 'node:test';
import { SSEParser, cleanAnimal, displayName, escapeHtml, formatApiError, renderFindings, renderFlags, renderReport, scaleBox } from '../lib.js';

const REPORT = {
  diagnosis_id: 'lv_1', model_version: 'Qwen/Qwen3.6-27B-FP8', source: 'llm', degraded: false, safety_adjustments: ['added isolation advice'],
  report: { summary: 'Skin nodules.', possible_conditions: ['lumpy skin disease'], severity: 'high', notifiable: true, zoonotic: false,
    immediate_actions: ['Isolate the animal'], treatment: { supportive_care: ['Water'], medicines: ['X (use under veterinary supervision)'], requires_vet: true, withdrawal_periods: ['Ask the vet'] },
    prevention: ['Control flies'], follow_up_days: 2, disclaimer: 'Not a diagnosis.' },
};

test('escapeHtml neutralises markup, quotes and backticks', () => {
  assert.equal(escapeHtml('<img src=x onerror="alert(1)">'), '&lt;img src=x onerror=&quot;alert(1)&quot;&gt;');
  assert.equal(escapeHtml("a&b'c`d"), 'a&amp;b&#39;c&#96;d');
  assert.equal(escapeHtml(null), '');
});

test('LLM output can never inject markup into the report', () => {
  const evil = structuredClone(REPORT);
  evil.report.summary = '<script>steal()</script>';
  evil.report.immediate_actions = ['<img src=x onerror=alert(1)>'];
  evil.model_version = '"><svg onload=alert(1)>';
  evil.safety_adjustments = ['<b onmouseover=1>'];
  const html = renderReport(evil);
  assert.ok(!/<script|<img|<svg|<b onmouseover/i.test(html));
  assert.ok(html.includes('&lt;script&gt;steal()&lt;/script&gt;'));
});

test('report shows source, severity, notifiable banner and the vet requirement', () => {
  const html = renderReport(REPORT);
  assert.ok(html.includes('AI report (Qwen/Qwen3.6-27B-FP8)') && html.includes('Severity: high'));
  assert.ok(html.includes('Notifiable disease suspected') && html.includes('veterinarian needed') && html.includes('Check again in 2 day(s)'));
  assert.ok(html.includes('Safety checks applied (1)') && html.includes('Not a diagnosis.'));
});

test('a degraded report is clearly labelled as standard guidance, not AI output', () => {
  const html = renderReport({ ...REPORT, source: 'advice_pack', degraded: true, model_version: 'advice-pack-v2' });
  assert.ok(html.includes('Standard guidance') && !html.includes('AI report ('));
});

test('unknown severity from a misbehaving server falls back to a safe class', () => {
  const html = renderReport({ ...REPORT, report: { ...REPORT.report, severity: '" onclick="x' } });
  assert.ok(html.includes('class="sev moderate"'));
});

test('SSE parser handles events split across chunks and CRLF', () => {
  const p = new SSEParser();
  const stream = 'event: start\ndata: {"a":1}\n\nevent: delta\ndata: {"chars":5}\n\nevent: final\ndata: {"ok":true}\n\n';
  const out = [];
  for (let i = 0; i < stream.length; i += 7) out.push(...p.push(stream.slice(i, i + 7)));     // 7-character chunks split everything
  assert.deepEqual(out.map((e) => e.event), ['start', 'delta', 'final']);
  assert.deepEqual(out[2].data, { ok: true });
  assert.deepEqual(new SSEParser().push('event: x\r\ndata: {"z":2}\r\n\r\n'), [{ event: 'x', data: { z: 2 } }]);
  assert.deepEqual(new SSEParser().push('event: x\ndata: {broken\n\n'), [{ event: 'x', data: null }]);
  assert.deepEqual(new SSEParser().push('event: x\ndata: {"half"'), []);                      // incomplete event is held back
});

test('scaleBox maps detector pixels onto the displayed canvas', () => {
  assert.deepEqual(scaleBox([100, 50, 300, 150], 400, 200, 800, 400), { x: 200, y: 100, w: 400, h: 200 });
  assert.deepEqual(scaleBox([0, 0, 10, 10], 100, 100, 100, 100), { x: 0, y: 0, w: 10, h: 10 });
});

test('cleanAnimal drops empty and invalid values', () => {
  assert.equal(cleanAnimal({ weight_kg: '', age_months: '', lactating: false, pregnant: false }), null);
  assert.deepEqual(cleanAnimal({ weight_kg: '250', age_months: '18.4', lactating: true, pregnant: false }), { weight_kg: 250, age_months: 18, lactating: true });
  assert.equal(cleanAnimal({ weight_kg: '-3', age_months: 'abc' }), null);
});

test('errors are shown with their request id, and network failures get a useful message', () => {
  assert.equal(formatApiError(422, { error: { message: 'bad', request_id: 'r1' } }), 'bad (request r1)');
  assert.ok(formatApiError(0, null).includes('Cannot reach'));
  assert.equal(formatApiError(502, null), 'Request failed (HTTP 502).');
});

test('findings and flag banners render; empty result is honest about not proving health', () => {
  const classes = [{ name: 'skin_nodule', display: { en: 'Skin nodules' } }];
  assert.ok(renderFindings({ detections: [{ class: 'skin_nodule', confidence: 0.884 }] }, classes).includes('Skin nodules'));
  assert.ok(renderFindings({ detections: [] }, classes).includes('does not prove'));
  assert.equal(displayName('mange_crusting', []), 'mange crusting');
  assert.ok(renderFlags(['possible_notifiable_disease']).includes('RAB') && renderFlags(['possible_zoonotic_risk']).includes('gloves'));
  assert.equal(renderFlags([]), '');
});
