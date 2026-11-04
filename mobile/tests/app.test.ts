import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { flagsFor, forDetections, lookup, offlineGuidance } from '../src/core/guidance.ts';
import { HttpError, NetworkError, type Http } from '../src/core/http.ts';
import { MAX_ATTEMPTS, backoffMs, disagreements, due, markFailed, newItem, pendingCount, type QueueItem } from '../src/core/queue.ts';
import { fromDevice, isOnline, runScan, syncQueue, type Config, type Context, type ScanDeps } from '../src/core/scan.ts';
import type { AdvicePack, ApiDetection, DetectResponse, ReportResponse } from '../src/core/types.ts';

const PACK: AdvicePack = JSON.parse(readFileSync(new URL('../src/data/diseases.json', import.meta.url), 'utf8'));
const SHARED: AdvicePack = JSON.parse(readFileSync(new URL('../../shared/advice_pack.json', import.meta.url), 'utf8'));
const FIXTURE = JSON.parse(readFileSync(new URL('./fixtures/fallback_parity.json', import.meta.url), 'utf8'));

// ------------------------------------------------------------------ the offline advice pack
test('the bundled advice pack is an exact copy of the shared one the server uses', () => {
  assert.deepEqual(PACK, SHARED);
});

test('TypeScript offline guidance reproduces the Python server fallback, field for field', () => {
  assert.equal(FIXTURE.pack_version, PACK.data_version, 'regenerate: python tools/make_guidance_fixture.py');
  for (const c of FIXTURE.cases) {
    const g = offlineGuidance(PACK, c.detections, c.flags);
    const e = c.expected;
    assert.equal(g.severity, e.severity, c.name);
    assert.equal(g.notifiable, e.notifiable, c.name);
    assert.equal(g.zoonotic, e.zoonotic, c.name);
    assert.deepEqual(g.possible_conditions, e.possible_conditions, c.name);
    assert.deepEqual(g.immediate_actions, e.immediate_actions, c.name);
    assert.deepEqual(g.prevention, e.prevention, c.name);
    assert.deepEqual(g.treatment.supportive_care, e.supportive_care, c.name);
    assert.equal(g.follow_up_days, e.follow_up_days, c.name);
    assert.equal(g.treatment.requires_vet, e.requires_vet, c.name);
    assert.deepEqual(flagsFor(PACK, c.detections), c.flags.filter((f: string) => true), `${c.name}: flags`);
  }
});

test('guidance never recommends a medicine or dose, and always requires a vet', () => {
  for (const e of PACK.diseases) {
    const g = offlineGuidance(PACK, [{ class: e.label, confidence: 0.9, box: [0, 0, 1, 1] }]);
    assert.deepEqual(g.treatment.medicines, []);
    assert.ok(g.treatment.requires_vet);
    assert.ok(!/\d+\s?(mg|ml|iu)\b/i.test(JSON.stringify(g)), `${e.label}: no dose-like text`);
  }
});

test('every plan class and every current-model class resolves to advice', () => {
  const plan = JSON.parse(readFileSync(new URL('../../shared/labels.json', import.meta.url), 'utf8')).classes.map((c: any) => c.name);
  for (const n of [...plan, 'healthy', 'lumpy_skin_disease', 'foot_and_mouth_disease']) assert.ok(lookup(PACK, n), n);
  assert.equal(forDetections(PACK, [{ class: 'skin_nodule', confidence: 0.5 }, { class: 'lumpy_skin_disease', confidence: 0.9 }]).length, 1, 'aliases collapse');
});

// ------------------------------------------------------------------ test doubles
const CFG: Config = { visionUrl: 'http://v:8001/', llmUrl: 'http://l:8002', token: 'tok' };
const CTX: Context = { language: 'en', species: 'cattle', district: 'Huye', animalId: 'cow-7', animal: { weight_kg: 250 } };
const DET: DetectResponse = { diagnosis_id: 'lv_abc', model_version: 'yolo11x-weak-v1', species: 'cattle', detections: [{ class: 'lumpy_skin_disease', confidence: 0.9, box: [1, 1, 50, 50] }],
  summary: { lumpy_skin_disease: 1 }, flags: ['possible_notifiable_disease'], annotated_image_url: 'u', image: { width: 640, height: 480 }, latency_ms: 80 };
const REPORT: ReportResponse = { diagnosis_id: 'lv_abc', model_version: 'Qwen/Qwen3.6-27B-FP8', language: 'en', source: 'llm', degraded: false, safety_adjustments: [],
  report: { summary: 'AI summary', possible_conditions: ['LSD'], severity: 'high', notifiable: true, zoonotic: false, immediate_actions: ['Isolate'], treatment: { supportive_care: [], medicines: [], requires_vet: true, withdrawal_periods: [] }, prevention: [], follow_up_days: 2 } };
const ON_DEVICE = async () => ({ detections: [{ class: 'lumpy_skin_disease', confidence: 0.62, box: [2, 2, 40, 40] }] as ApiDetection[], modelVersion: 'yolo11n-weak-v1', imageSize: { width: 640, height: 480 } });

type Handler = (url: string, body?: any, fields?: Record<string, string>) => any;
class FakeHttp implements Http {
  calls: { method: string; url: string; token?: string; timeoutMs: number; fields?: Record<string, string>; body?: any }[] = [];
  routes: Record<string, Handler | Error | any>;
  constructor(routes: Record<string, Handler | Error | any>) { this.routes = routes; }
  private run(method: string, url: string, o: { token?: string; timeoutMs: number }, extra: any) {
    this.calls.push({ method, url, token: o.token, timeoutMs: o.timeoutMs, ...extra });
    const path = new URL(url).pathname;
    const r = this.routes[`${method} ${path}`];
    if (r === undefined) return Promise.reject(new HttpError(404, 'no route'));
    const v = typeof r === 'function' ? r(url, extra.body, extra.fields) : r;
    return v instanceof Error ? Promise.reject(v) : Promise.resolve(v);
  }
  get(url: string, o: any) { return this.run('GET', url, o, {}); }
  postJson(url: string, body: unknown, o: any) { return this.run('POST', url, o, { body }); }
  postForm(url: string, fields: Record<string, string>, _f: unknown, o: any) { return this.run('POST', url, o, { fields }); }
}
const mkDeps = (http: Http) => {
  const queued: QueueItem[] = [];
  let n = 0;
  const deps: ScanDeps = { http, onDevice: ON_DEVICE, pack: PACK, queue: { add: async (i) => { queued.push(i); } }, now: () => 1_000_000, id: () => `q${++n}` };
  return { deps, queued };
};
const PHOTO = { uri: 'file:///p.jpg', width: 800, height: 600 };
const ONLINE = { 'GET /ready': { ready: true }, 'POST /v1/detect': DET, 'POST /v1/report': REPORT };

// ------------------------------------------------------------------ online path
test('online: YOLO11x detects, the LLM writes the report, nothing is queued, the token is sent', async () => {
  const http = new FakeHttp(ONLINE);
  const { deps, queued } = mkDeps(http);
  const r = await runScan(PHOTO, CTX, CFG, deps);
  assert.equal(r.mode, 'online'); assert.equal(r.confirmed, true); assert.equal(r.modelVersion, 'yolo11x-weak-v1'); assert.equal(r.reportSource, 'llm');
  assert.equal(r.report.summary, 'AI summary'); assert.equal(r.diagnosisId, 'lv_abc'); assert.deepEqual(r.image, { width: 640, height: 480 }); assert.equal(queued.length, 0);
  assert.deepEqual(http.calls.map((c) => `${c.method} ${new URL(c.url).pathname}`), ['GET /ready', 'POST /v1/detect', 'POST /v1/report']);
  assert.ok(http.calls.slice(1).every((c) => c.token === 'tok'));
  assert.equal(http.calls[0].timeoutMs, 8000);                                  // the 8-second connectivity probe from the spec
  assert.deepEqual(http.calls[1].fields, { species: 'cattle', district: 'Huye', animal_id: 'cow-7' });
  assert.deepEqual(http.calls[2].body, { diagnosis_id: 'lv_abc', language: 'en', animal: { weight_kg: 250 }, stream: false });
  assert.ok(http.calls[1].url.startsWith('http://v:8001/v1/detect'));          // trailing slash in settings is tolerated
});

test('online but the LLM is down: confirmed detections with the standard guidance, clearly noted', async () => {
  for (const failure of [new HttpError(503, 'down'), new NetworkError(), new HttpError(500, 'boom')]) {
    const { deps } = mkDeps(new FakeHttp({ ...ONLINE, 'POST /v1/report': failure }));
    const r = await runScan(PHOTO, CTX, CFG, deps);
    assert.equal(r.mode, 'online'); assert.equal(r.confirmed, true); assert.equal(r.reportSource, 'on_device_pack');
    assert.equal(r.report.notifiable, true); assert.ok(r.report.immediate_actions.length > 0); assert.ok(r.notes.some((n) => n.includes('standard guidance')));
  }
});

test('the server itself degraded to its advice pack: the app says so', async () => {
  const { deps } = mkDeps(new FakeHttp({ ...ONLINE, 'POST /v1/report': { ...REPORT, source: 'advice_pack', degraded: true } }));
  const r = await runScan(PHOTO, CTX, CFG, deps);
  assert.equal(r.reportSource, 'advice_pack'); assert.ok(r.notes.some((n) => n.includes('AI report is unavailable')));
});

// ------------------------------------------------------------------ offline path
test('no connection: on-device YOLO11n, advice pack guidance with the notifiable alert, photo queued', async () => {
  const http = new FakeHttp({ 'GET /ready': new NetworkError() });
  const { deps, queued } = mkDeps(http);
  const r = await runScan(PHOTO, CTX, CFG, deps);
  assert.equal(r.mode, 'offline'); assert.equal(r.confirmed, false); assert.equal(r.modelVersion, 'yolo11n-weak-v1'); assert.equal(r.reportSource, 'on_device_pack');
  assert.deepEqual(r.flags, ['possible_notifiable_disease']); assert.equal(r.report.notifiable, true); assert.ok(r.notes.some((n) => n.includes('Preliminary')));
  assert.equal(http.calls.length, 1);                                           // only the probe: no further network attempts
  assert.deepEqual(r.image, { width: 640, height: 480 });                       // offline boxes are drawn in the on-device frame
  assert.equal(queued.length, 1);
  const q = queued[0];
  assert.equal(q.id, r.queuedId); assert.equal(q.status, 'pending'); assert.equal(q.offlineModelVersion, 'yolo11n-weak-v1'); assert.equal(q.animalId, 'cow-7');
  assert.deepEqual(q.offlineDetections[0].class, 'lumpy_skin_disease'); assert.deepEqual(q.imageSize, { width: 800, height: 600 }); assert.equal(q.animal?.weight_kg, 250);
});

test('no token: offline straight away, without touching the network', async () => {
  const http = new FakeHttp(ONLINE);
  const { deps } = mkDeps(http);
  const r = await runScan(PHOTO, CTX, { ...CFG, token: '' }, deps);
  assert.equal(r.mode, 'offline'); assert.equal(http.calls.length, 0); assert.ok(r.notes[0].includes('Not signed in'));
});

test('a photo is never lost: failures after a good probe still fall back to the phone model', async () => {
  for (const [failure, fragment] of [[new NetworkError('connection dropped'), 'did not answer'], [new HttpError(503, 'not ready'), 'did not answer'],
                                     [new HttpError(429, 'slow down'), 'did not answer'], [new HttpError(401, 'expired'), 'Sign in again']] as const) {
    const { deps, queued } = mkDeps(new FakeHttp({ ...ONLINE, 'POST /v1/detect': failure }));
    const r = await runScan(PHOTO, CTX, CFG, deps);
    assert.equal(r.mode, 'offline'); assert.equal(queued.length, 1); assert.ok(r.notes.some((n) => n.includes(fragment)), String(failure));
  }
});

test('an unreadable image is reported to the user instead of silently queued forever', async () => {
  const { deps, queued } = mkDeps(new FakeHttp({ ...ONLINE, 'POST /v1/detect': new HttpError(422, 'The file is not a readable image') }));
  await assert.rejects(runScan(PHOTO, CTX, CFG, deps), /not a readable image/);
  assert.equal(queued.length, 0);
});

test('a missing offline model is a clear error, not a crash or a silent empty result', async () => {
  const { deps } = mkDeps(new FakeHttp({ 'GET /ready': new NetworkError() }));
  deps.onDevice = async () => { throw new Error('No trained model is installed in this build yet'); };
  await assert.rejects(runScan(PHOTO, CTX, CFG, deps), /No trained model/);
});

test('isOnline: ready server only; a 503 "model not ready" counts as offline', async () => {
  assert.equal(await isOnline(CFG, new FakeHttp({ 'GET /ready': { ready: true } })), true);
  assert.equal(await isOnline(CFG, new FakeHttp({ 'GET /ready': new HttpError(503, 'model not ready') })), false);
  assert.equal(await isOnline({ ...CFG, visionUrl: '' }, new FakeHttp({})), false);
});

test('device detections map to the API shape', () => {
  assert.deepEqual(fromDevice([{ label: 'tick', score: 0.5, box: { x1: 1, y1: 2, x2: 3, y2: 4 } }]), [{ class: 'tick', confidence: 0.5, box: [1, 2, 3, 4] }]);
});

// ------------------------------------------------------------------ queue + sync
const item = (id: string, createdAt: number, over: Partial<QueueItem> = {}): QueueItem =>
  newItem({ id, uri: `file:///${id}.jpg`, createdAt, language: 'en', species: 'cattle', district: 'Huye', animalId: id, offlineDetections: [{ class: 'healthy', confidence: 0.6, box: [0, 0, 9, 9] }], offlineModelVersion: 'yolo11n-weak-v1', ...over });
class Store { items = new Map<string, QueueItem>(); update = async (i: QueueItem) => { this.items.set(i.id, i); }; }

test('sync: oldest first, offline result attached, report fetched, disagreement surfaced', async () => {
  const seen: any[] = [];
  const http = new FakeHttp({ 'GET /ready': { ready: true }, 'POST /v1/report': REPORT,
    'POST /v1/detect': (_u: string, _b: any, f: any) => { seen.push(f); return { ...DET, offline_comparison: f.animal_id === 'b' ? { agrees: false, only_online: ['lumpy_skin_disease'], only_offline: ['healthy'] } : { agrees: true, only_online: [], only_offline: [] } }; } });
  const store = new Store();
  const r = await syncQueue([item('b', 2000), item('a', 1000)], CFG, { http, queue: store, now: () => 5000 });
  assert.deepEqual(seen.map((f) => f.animal_id), ['a', 'b']);                   // oldest photo first
  assert.deepEqual(JSON.parse(seen[0].offline_result), { model_version: 'yolo11n-weak-v1', detections: [{ class: 'healthy', confidence: 0.6, box: [0, 0, 9, 9] }] });
  assert.ok(r.items.every((i) => i.status === 'done' && i.detection && i.report));
  assert.deepEqual(r.events.map((e) => (e as any).agrees), [true, false]);
  assert.deepEqual(disagreements(r.items).map((i) => i.id), ['b']);
  assert.equal(pendingCount(r.items), 0);
  assert.equal(store.items.get('a')?.status, 'done');                            // persisted after every step
});

test('sync: still offline -> nothing is touched; network drop mid-way stops the run and backs off', async () => {
  const off = await syncQueue([item('a', 1)], CFG, { http: new FakeHttp({ 'GET /ready': new NetworkError() }), queue: new Store(), now: () => 10 });
  assert.deepEqual(off.events, [{ type: 'offline' }]); assert.equal(off.items[0].status, 'pending');
  let calls = 0;
  const http = new FakeHttp({ 'GET /ready': { ready: true }, 'POST /v1/detect': () => { calls++; return new NetworkError(); } });
  const r = await syncQueue([item('a', 1), item('b', 2)], CFG, { http, queue: new Store(), now: () => 100_000 });
  assert.equal(calls, 1);                                                        // did not hammer the second photo
  const a = r.items.find((i) => i.id === 'a')!; const b = r.items.find((i) => i.id === 'b')!;
  assert.equal(a.status, 'pending'); assert.equal(a.attempts, 1); assert.equal(a.nextAttemptAt, 100_000 + 30_000); assert.equal(b.status, 'pending');
  assert.deepEqual(due(r.items, 100_001).map((i) => i.id), ['b']);                                           // 'b' was never attempted: still due; 'a' waits out its backoff
  assert.deepEqual(due(r.items, 130_001).map((i) => i.id), ['a', 'b']);                                       // after the backoff both are due, oldest first
});

test('sync: a permanent error (bad image) is given up on; one bad photo does not block the rest', async () => {
  const http = new FakeHttp({ 'GET /ready': { ready: true }, 'POST /v1/report': REPORT,
    'POST /v1/detect': (_u: string, _b: any, f: any) => (f.animal_id === 'bad' ? new HttpError(422, 'not a readable image') : DET) });
  const r = await syncQueue([item('bad', 1), item('good', 2)], CFG, { http, queue: new Store(), now: () => 10 });
  assert.equal(r.items.find((i) => i.id === 'bad')!.status, 'failed'); assert.equal(r.items.find((i) => i.id === 'good')!.status, 'done');
  assert.deepEqual(r.events.map((e) => e.type), ['failed', 'confirmed']);
});

test('sync: an LLM outage does not undo a confirmed detection', async () => {
  const http = new FakeHttp({ 'GET /ready': { ready: true }, 'POST /v1/detect': DET, 'POST /v1/report': new HttpError(503, 'llm down') });
  const r = await syncQueue([item('a', 1)], CFG, { http, queue: new Store(), now: () => 10 });
  assert.equal(r.items[0].status, 'done'); assert.equal(r.items[0].report, null); assert.equal(r.items[0].detection?.diagnosis_id, 'lv_abc');
});

test('queue rules: exponential backoff, server errors retry, client errors give up, 401 waits for sign-in, attempts are capped', () => {
  assert.deepEqual([1, 2, 3, 4, 8, 20].map(backoffMs), [30_000, 60_000, 120_000, 240_000, 3_840_000 > 3_600_000 ? 3_600_000 : 3_840_000, 3_600_000]);
  const i = item('x', 0);
  assert.equal(markFailed(i, { status: 503, message: 'x' }, 0).status, 'pending');
  assert.equal(markFailed(i, { status: 429, message: 'x' }, 0).status, 'pending');
  assert.equal(markFailed(i, { status: 422, message: 'x' }, 0).status, 'failed');
  assert.equal(markFailed(i, { status: 413, message: 'x' }, 0).status, 'failed');
  const t = markFailed({ ...i, attempts: 3 }, { status: 401, message: 'x' }, 1000);
  assert.equal(t.status, 'pending'); assert.equal(t.attempts, 3); assert.ok(t.lastError?.includes('Sign in'));          // a stale token is not the photo's fault
  assert.equal(markFailed({ ...i, attempts: MAX_ATTEMPTS - 1 }, { message: 'net' }, 0).status, 'failed');               // gives up eventually
  assert.equal(due([{ ...i, status: 'syncing', nextAttemptAt: 0 }], 11 * 60_000).length, 1);                           // killed mid-sync: retried
  assert.equal(due([{ ...i, status: 'syncing', nextAttemptAt: 0 }], 5 * 60_000).length, 0);
});
