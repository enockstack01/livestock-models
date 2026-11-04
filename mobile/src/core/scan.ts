// The online/offline brain of the app.
//   online  (Vision API reachable) -> the server model detects (confirmed) -> LLM API writes the report; if the LLM is down: advice-pack guidance
//   offline / anything fails       -> on-device YOLO11n detects ("preliminary"), guidance from the bundled advice pack, photo queued
//   reconnect                      -> syncQueue sends each queued photo to the server model with the offline result attached; disagreements are reported
import { HttpError, NetworkError, TIMEOUTS, type Http } from './http.ts';
import { flagsFor, offlineGuidance } from './guidance.ts';
import { disagreements, due, markDone, markFailed, markSyncing, newItem, type QueueItem } from './queue.ts';
import type { Animal, AdvicePack, ApiDetection, DetectResponse, Language, Report, ReportResponse } from './types.ts';

export type Photo = { uri: string; width: number; height: number };
export type Context = { language: Language; species: string; district: string; animalId: string; animal?: Animal };
export type Config = { visionUrl: string; llmUrl: string; token: string };
export type OnDevice = (p: Photo) => Promise<{ detections: ApiDetection[]; modelVersion: string; imageSize?: { width: number; height: number } }>;
export type ScanDeps = {
  http: Http; onDevice: OnDevice; pack: AdvicePack; queue: { add(i: QueueItem): Promise<void> }; now: () => number; id: () => string;
};
export type ReportSource = 'llm' | 'advice_pack' | 'on_device_pack';
export type ScanResult = {
  mode: 'online' | 'offline'; confirmed: boolean; modelVersion: string; detections: ApiDetection[]; flags: string[];
  image?: { width: number; height: number }; diagnosisId?: string; report: Report; reportSource: ReportSource; reportModel?: string;
  notes: string[]; queuedId?: string;
};

const base = (u: string) => u.replace(/\/+$/, '');
const file = (p: Photo) => ({ uri: p.uri, name: 'photo.jpg', type: 'image/jpeg' });

/** Is the Vision API reachable AND ready? 8 s limit; any failure means "go offline". */
export async function isOnline(cfg: Config, http: Http): Promise<boolean> {
  if (!cfg.token || !cfg.visionUrl) return false;
  try { await http.get(`${base(cfg.visionUrl)}/ready`, { timeoutMs: TIMEOUTS.probeMs }); return true; } catch { return false; }
}

export const fromDevice = (dets: { label: string; score: number; box: { x1: number; y1: number; x2: number; y2: number } }[]): ApiDetection[] =>
  dets.map((d) => ({ class: d.label, confidence: d.score, box: [d.box.x1, d.box.y1, d.box.x2, d.box.y2] }));

async function fetchReport(cfg: Config, http: Http, id: string, ctx: Context): Promise<ReportResponse> {
  return http.postJson(`${base(cfg.llmUrl)}/v1/report`, { diagnosis_id: id, language: ctx.language, animal: ctx.animal ?? null, stream: false },
    { token: cfg.token, timeoutMs: TIMEOUTS.reportMs });
}

async function scanOffline(photo: Photo, ctx: Context, deps: ScanDeps, notes: string[]): Promise<ScanResult> {
  const off = await deps.onDevice(photo);                                       // throws if no offline model is installed
  const flags = flagsFor(deps.pack, off.detections);
  const item = newItem({ id: deps.id(), uri: photo.uri, createdAt: deps.now(), language: ctx.language, species: ctx.species, district: ctx.district,
    animalId: ctx.animalId, animal: ctx.animal, offlineDetections: off.detections, offlineModelVersion: off.modelVersion,
    imageSize: { width: photo.width, height: photo.height } });
  await deps.queue.add(item);
  return { mode: 'offline', confirmed: false, modelVersion: off.modelVersion, detections: off.detections, flags, image: off.imageSize,   // boxes are in the on-device frame
    report: offlineGuidance(deps.pack, off.detections, flags), reportSource: 'on_device_pack', queuedId: item.id,
    notes: [...notes, 'Preliminary result from the on-device model. It will be confirmed automatically when you are back online.'] };
}

export async function runScan(photo: Photo, ctx: Context, cfg: Config, deps: ScanDeps): Promise<ScanResult> {
  if (!(await isOnline(cfg, deps.http))) {
    return scanOffline(photo, ctx, deps, !cfg.token ? ['Not signed in: using the on-device model.'] : ['No connection to the server: using the on-device model.']);
  }
  let det: DetectResponse;
  try {
    det = await deps.http.postForm(`${base(cfg.visionUrl)}/v1/detect`,
      { species: ctx.species, district: ctx.district, animal_id: ctx.animalId }, file(photo), { token: cfg.token, timeoutMs: TIMEOUTS.detectMs });
  } catch (e) {
    // a photo is never lost to a bad connection or an expired login: fall back to the phone's own model
    if (e instanceof NetworkError || (e instanceof HttpError && (e.status === 401 || e.status === 403 || e.status === 429 || e.status >= 500))) {
      return scanOffline(photo, ctx, deps, [e instanceof HttpError && (e.status === 401 || e.status === 403) ? 'Sign in again to use online detection.' : 'The server did not answer: using the on-device model.']);
    }
    throw e;                                                                    // e.g. 422 "not a readable image": retrying elsewhere cannot help
  }
  const notes: string[] = [];
  let report: Report, reportSource: ReportSource, reportModel: string | undefined;
  try {
    const r = await fetchReport(cfg, deps.http, det.diagnosis_id, ctx);
    report = r.report; reportSource = r.source; reportModel = r.model_version;
    if (r.degraded) notes.push('The AI report is unavailable right now: showing standard guidance.');
  } catch {
    report = offlineGuidance(deps.pack, det.detections, det.flags); reportSource = 'on_device_pack';
    notes.push('The advice service did not answer: showing standard guidance. You can ask for the full report later.');
  }
  return { mode: 'online', confirmed: true, modelVersion: det.model_version, detections: det.detections, flags: det.flags, image: det.image,
    diagnosisId: det.diagnosis_id, report, reportSource, reportModel, notes };
}

export type SyncEvent =
  | { type: 'confirmed'; id: string; agrees: boolean; onlyOnline: string[]; onlyOffline: string[] }
  | { type: 'failed'; id: string; message: string; permanent: boolean }
  | { type: 'offline' };
export type SyncDeps = { http: Http; queue: { update(i: QueueItem): Promise<void> }; now: () => number };

/** Send every due queued photo to the server model with the offline result attached, then fetch the report. Stops at the first network failure. */
export async function syncQueue(items: QueueItem[], cfg: Config, deps: SyncDeps): Promise<{ items: QueueItem[]; events: SyncEvent[] }> {
  const events: SyncEvent[] = [];
  const out = new Map(items.map((i) => [i.id, i]));
  if (!(await isOnline(cfg, deps.http))) return { items, events: [{ type: 'offline' }] };
  for (const item of due(items, deps.now())) {
    let cur = markSyncing(item, deps.now());
    out.set(cur.id, cur); await deps.queue.update(cur);
    try {
      const det: DetectResponse = await deps.http.postForm(`${base(cfg.visionUrl)}/v1/detect`,
        { species: cur.species, district: cur.district, animal_id: cur.animalId,
          offline_result: JSON.stringify({ model_version: cur.offlineModelVersion, detections: cur.offlineDetections }) },
        file({ uri: cur.uri, width: 0, height: 0 }), { token: cfg.token, timeoutMs: TIMEOUTS.detectMs });
      let report: ReportResponse | null = null;
      try { report = await fetchReport(cfg, deps.http, det.diagnosis_id, cur); } catch { /* the confirmed detection is what matters; the report can be fetched later */ }
      cur = markDone(cur, det, report, deps.now());
      const c = det.offline_comparison;
      events.push({ type: 'confirmed', id: cur.id, agrees: c ? c.agrees : true, onlyOnline: c?.only_online ?? [], onlyOffline: c?.only_offline ?? [] });
    } catch (e) {
      const err = { status: e instanceof HttpError ? e.status : undefined, message: e instanceof Error ? e.message : String(e) };
      cur = markFailed(cur, err, deps.now());
      events.push({ type: 'failed', id: cur.id, message: cur.lastError ?? err.message, permanent: cur.status === 'failed' });
      out.set(cur.id, cur); await deps.queue.update(cur);
      if (e instanceof NetworkError) break;                                     // still offline: do not drain the battery on the rest
      continue;
    }
    out.set(cur.id, cur); await deps.queue.update(cur);
  }
  return { items: [...out.values()], events };
}

export { disagreements };
