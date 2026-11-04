// Offline knowledge base: a local SQLite database seeded from the bundled JSON on first launch (and again
// whenever data_version in the JSON increases, e.g. after a vet reviews/localises the content). No network.
import * as SQLite from 'expo-sqlite';
import seed from '../data/diseases.json';

export type Disease = {
  label: string;
  title: string;
  species: string;
  notifiable: boolean;
  urgency: 'none' | 'soon' | 'urgent' | 'emergency';
  signs: string[];
  doNow: string[];
  prevention: string[];
  seeVet: string;
};

let dbPromise: Promise<SQLite.SQLiteDatabase> | null = null;

async function init(): Promise<SQLite.SQLiteDatabase> {
  const db = await SQLite.openDatabaseAsync('livestockdx.db');
  await db.execAsync(`
    PRAGMA journal_mode = WAL;
    CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY NOT NULL, value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS diseases (
      label TEXT PRIMARY KEY NOT NULL, title TEXT NOT NULL, species TEXT NOT NULL, notifiable INTEGER NOT NULL,
      urgency TEXT NOT NULL, signs TEXT NOT NULL, do_now TEXT NOT NULL, prevention TEXT NOT NULL, see_vet TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS aliases (alias TEXT PRIMARY KEY NOT NULL, label TEXT NOT NULL);
  `);
  const row = await db.getFirstAsync<{ value: string }>('SELECT value FROM meta WHERE key = ?', 'data_version');
  if (!row || Number(row.value) < seed.data_version) {
    await db.withTransactionAsync(async () => {
      await db.runAsync('DELETE FROM diseases');
      await db.runAsync('DELETE FROM aliases');
      for (const d of seed.diseases) {
        await db.runAsync(
          'INSERT INTO diseases (label,title,species,notifiable,urgency,signs,do_now,prevention,see_vet) VALUES (?,?,?,?,?,?,?,?,?)',
          [d.label, d.title, d.species, d.notifiable ? 1 : 0, d.urgency, JSON.stringify(d.signs), JSON.stringify(d.do_now), JSON.stringify(d.prevention), d.see_vet],
        );
        for (const a of [d.label, ...d.aliases]) await db.runAsync('INSERT OR REPLACE INTO aliases (alias,label) VALUES (?,?)', [a, d.label]);
      }
      await db.runAsync('INSERT OR REPLACE INTO meta (key,value) VALUES (?,?)', ['data_version', String(seed.data_version)]);
    });
  }
  return db;
}

export const getDb = () => (dbPromise ??= init());

type Row = { label: string; title: string; species: string; notifiable: number; urgency: Disease['urgency']; signs: string; do_now: string; prevention: string; see_vet: string };
const toDisease = (r: Row): Disease => ({
  label: r.label, title: r.title, species: r.species, notifiable: r.notifiable === 1, urgency: r.urgency,
  signs: JSON.parse(r.signs), doNow: JSON.parse(r.do_now), prevention: JSON.parse(r.prevention), seeVet: r.see_vet,
});

/** Look up what a model class label means (accepts the model's own labels and our aliases). Offline, instant. */
export async function findDisease(modelLabel: string): Promise<Disease | null> {
  const db = await getDb();
  const row = await db.getFirstAsync<Row>(
    'SELECT d.* FROM diseases d JOIN aliases a ON a.label = d.label WHERE a.alias = ?', modelLabel.toLowerCase());
  return row ? toDisease(row) : null;
}

export async function listDiseases(): Promise<Disease[]> {
  const db = await getDb();
  return (await db.getAllAsync<Row>("SELECT * FROM diseases WHERE label <> 'healthy' ORDER BY notifiable DESC, title")).map(toDisease);
}

export const DISCLAIMER = seed.disclaimer;
export const REVIEWED_BY_VET = seed.reviewed_by_veterinarian;
