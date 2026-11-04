// Offline queue: photos taken without internet wait here and are confirmed by the server model when the phone reconnects.
// Pure state transitions (no storage, no network): persistence is a separate adapter, so every rule is unit-tested.
import type { Animal, ApiDetection, DetectResponse, Language, ReportResponse } from './types.ts';

export type QueueStatus = 'pending' | 'syncing' | 'done' | 'failed';
export type QueueItem = {
  id: string; uri: string; createdAt: number; language: Language; species: string; district: string; animalId: string; animal?: Animal;
  offlineDetections: ApiDetection[]; offlineModelVersion: string; imageSize?: { width: number; height: number };
  status: QueueStatus; attempts: number; nextAttemptAt: number; lastError?: string;
  // filled in when confirmed online:
  detection?: DetectResponse; report?: ReportResponse | null; syncedAt?: number;
};

export const MAX_ATTEMPTS = 8;
const BASE_DELAY_MS = 30_000;
const MAX_DELAY_MS = 60 * 60_000;

export const backoffMs = (attempts: number) => Math.min(MAX_DELAY_MS, BASE_DELAY_MS * 2 ** Math.max(0, attempts - 1));

export function newItem(p: Omit<QueueItem, 'status' | 'attempts' | 'nextAttemptAt'>): QueueItem {
  return { ...p, status: 'pending', attempts: 0, nextAttemptAt: p.createdAt };
}

export const due = (items: QueueItem[], now: number) =>
  items.filter((i) => (i.status === 'pending' || (i.status === 'syncing' && now - i.nextAttemptAt > 10 * 60_000)) && i.nextAttemptAt <= now)
    .sort((a, b) => a.createdAt - b.createdAt);                   // oldest first; a 'syncing' item stuck for 10 min (app was killed) is retried

export const markSyncing = (i: QueueItem, now: number): QueueItem => ({ ...i, status: 'syncing', nextAttemptAt: now });

export function markDone(i: QueueItem, detection: DetectResponse, report: ReportResponse | null, now: number): QueueItem {
  return { ...i, status: 'done', detection, report, syncedAt: now, lastError: undefined };
}

/** Retry later for network trouble and server overload; give up at once on errors a retry cannot fix (bad image, bad token...). */
export function markFailed(i: QueueItem, err: { status?: number; message: string }, now: number): QueueItem {
  const attempts = i.attempts + 1;
  const transient = err.status === undefined || err.status === 408 || err.status === 429 || err.status >= 500;
  const permanent = !transient || attempts >= MAX_ATTEMPTS;
  // 401 is special: the token expired, not the photo's fault: keep it and retry after the user signs in again
  if (err.status === 401) return { ...i, status: 'pending', attempts: i.attempts, nextAttemptAt: now + 5 * 60_000, lastError: 'Sign in again to confirm this photo.' };
  return { ...i, attempts, status: permanent ? 'failed' : 'pending', nextAttemptAt: now + backoffMs(attempts), lastError: err.message };
}

export const pendingCount = (items: QueueItem[]) => items.filter((i) => i.status === 'pending' || i.status === 'syncing').length;

/** Items whose confirmation by the server model differs from the on-device result: the user should be told. */
export const disagreements = (items: QueueItem[]) =>
  items.filter((i) => i.status === 'done' && i.detection?.offline_comparison && !i.detection.offline_comparison.agrees);
