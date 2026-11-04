// Persistence (expo-sqlite): the offline queue and the connection settings survive app restarts and phone reboots.
// NOTE: the access token is stored in the app's private SQLite file. For a public release move it to expo-secure-store (Keychain / Keystore).
import { getDb } from '../db/diseases';
import type { QueueItem } from '../core/queue';
import type { Language } from '../core/types';

export type Settings = { visionUrl: string; llmUrl: string; token: string; language: Language };
export const DEFAULT_SETTINGS: Settings = { visionUrl: 'https://livestock-vision.api.agricoders.com', llmUrl: 'https://llm.api.agricoders.com', token: '', language: 'en' };

async function db() {
  const d = await getDb();
  await d.execAsync(`
    CREATE TABLE IF NOT EXISTS queue (id TEXT PRIMARY KEY NOT NULL, created_at INTEGER NOT NULL, json TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY NOT NULL, value TEXT NOT NULL);
  `);
  return d;
}

export const queueStore = {
  async list(): Promise<QueueItem[]> {
    const d = await db();
    return (await d.getAllAsync<{ json: string }>('SELECT json FROM queue ORDER BY created_at ASC')).map((r) => JSON.parse(r.json));
  },
  async add(i: QueueItem): Promise<void> { await this.update(i); },
  async update(i: QueueItem): Promise<void> {
    const d = await db();
    await d.runAsync('INSERT OR REPLACE INTO queue (id, created_at, json) VALUES (?, ?, ?)', [i.id, i.createdAt, JSON.stringify(i)]);
  },
  async remove(id: string): Promise<void> { await (await db()).runAsync('DELETE FROM queue WHERE id = ?', [id]); },
};

export async function loadSettings(): Promise<Settings> {
  const d = await db();
  const rows = await d.getAllAsync<{ key: string; value: string }>('SELECT key, value FROM settings');
  const m = Object.fromEntries(rows.map((r) => [r.key, r.value]));
  return { ...DEFAULT_SETTINGS, ...m } as Settings;
}

export async function saveSettings(s: Settings): Promise<void> {
  const d = await db();
  await d.withTransactionAsync(async () => {
    for (const [k, v] of Object.entries(s)) await d.runAsync('INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)', [k, String(v)]);
  });
}
