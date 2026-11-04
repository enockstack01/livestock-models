import { StatusBar } from 'expo-status-bar';
import { useCallback, useEffect, useRef, useState } from 'react';
import { AppState, Pressable, SafeAreaView, ScrollView, StyleSheet, Text, View } from 'react-native';
import { http } from './src/api/http';
import { pendingCount, type QueueItem } from './src/core/queue';
import { syncQueue } from './src/core/scan';
import { DISCLAIMER, REVIEWED_BY_VET, listDiseases, type Disease } from './src/db/diseases';
import { QueueScreen } from './src/screens/QueueScreen';
import { ScanScreen } from './src/screens/ScanScreen';
import { SettingsScreen } from './src/screens/SettingsScreen';
import { DEFAULT_SETTINGS, loadSettings, queueStore, type Settings } from './src/store/storage';
import { C, s } from './src/ui/theme';

type Tab = 'scan' | 'queue' | 'library' | 'settings';
const TABS: [Tab, string][] = [['scan', 'Scan'], ['queue', 'Queue'], ['library', 'Library'], ['settings', 'Settings']];

function Library() {
  const [items, setItems] = useState<Disease[]>([]);
  useEffect(() => { listDiseases().then(setItems); }, []);
  return (
    <ScrollView contentContainerStyle={s.screen}>
      {items.map((d) => (
        <View key={d.label} style={s.card}>
          <Text style={s.h2}>{d.title}</Text>
          <Text style={s.muted}>{d.species}</Text>
          {d.notifiable && <Text style={{ color: C.bad, fontWeight: '700', marginTop: 6 }}>Reportable disease: tell the authorities</Text>}
          <Text style={s.h3}>What it looks like</Text>{d.signs.map((t, i) => <Text key={i} style={s.li}>• {t}</Text>)}
          <Text style={s.h3}>What to do now</Text>{d.doNow.map((t, i) => <Text key={i} style={s.li}>• {t}</Text>)}
          <Text style={s.muted}>{d.seeVet}</Text>
        </View>
      ))}
    </ScrollView>
  );
}

export default function App() {
  const [tab, setTab] = useState<Tab>('scan');
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [ready, setReady] = useState(false);
  const [items, setItems] = useState<QueueItem[]>([]);
  const [syncing, setSyncing] = useState(false);
  const busy = useRef(false);
  const cfg = useRef(settings);
  cfg.current = settings;

  const refresh = useCallback(async () => setItems(await queueStore.list()), []);

  const sync = useCallback(async () => {
    if (busy.current) return;
    busy.current = true; setSyncing(true);
    try {
      const { token, visionUrl, llmUrl } = cfg.current;
      const r = await syncQueue(await queueStore.list(), { token, visionUrl, llmUrl }, { http, queue: queueStore, now: Date.now });
      setItems(r.items.sort((a, b) => a.createdAt - b.createdAt));
    } catch { /* offline or storage error: the queue is untouched and retried later */ }
    busy.current = false; setSyncing(false);
  }, []);

  useEffect(() => { loadSettings().then((x) => { setSettings(x); setReady(true); }).catch(() => setReady(true)); refresh(); }, [refresh]);

  // sync on start (once settings are loaded), whenever the app returns to the foreground, and every 30 s while photos are waiting
  useEffect(() => { if (ready) sync(); }, [ready, sync]);
  useEffect(() => {
    const sub = AppState.addEventListener('change', (st) => { if (st === 'active') sync(); });
    return () => sub.remove();
  }, [sync]);
  const waiting = pendingCount(items);
  useEffect(() => {
    if (!waiting) return;
    const t = setInterval(sync, 30000);
    return () => clearInterval(t);
  }, [waiting, sync]);

  const retry = async (i: QueueItem) => { await queueStore.update({ ...i, status: 'pending', attempts: 0, nextAttemptAt: 0, lastError: undefined }); await refresh(); sync(); };
  const remove = async (i: QueueItem) => { await queueStore.remove(i.id); await refresh(); };

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: C.bg }}>
      <StatusBar style="dark" />
      <View style={styles.top}><Text style={styles.title}>LivestockDx</Text>{waiting > 0 && <Text style={styles.pill}>{waiting} waiting</Text>}</View>
      <View style={styles.tabs}>
        {TABS.map(([k, name]) => (
          <Pressable key={k} style={[styles.tab, tab === k && styles.tabOn]} onPress={() => setTab(k)}><Text style={[styles.tabT, tab === k && { color: '#fff' }]}>{name}</Text></Pressable>
        ))}
      </View>
      <View style={{ flex: 1 }}>
        {tab === 'scan' && <ScanScreen settings={settings} pending={waiting} onQueueChanged={refresh} />}
        {tab === 'queue' && <QueueScreen items={items} syncing={syncing} onSync={sync} onRetry={retry} onDelete={remove} />}
        {tab === 'library' && <Library />}
        {tab === 'settings' && <SettingsScreen settings={settings} onSaved={(x) => { setSettings(x); sync(); }} />}
      </View>
      <Text style={styles.foot}>{REVIEWED_BY_VET ? '' : 'Content not yet vet-reviewed. '}{DISCLAIMER}</Text>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  top: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', padding: 16, paddingBottom: 8 },
  title: { fontSize: 20, fontWeight: '800', color: C.ink },
  pill: { backgroundColor: C.warnBg, color: C.warn, fontWeight: '700', paddingHorizontal: 12, paddingVertical: 4, borderRadius: 99, overflow: 'hidden' },
  tabs: { flexDirection: 'row', gap: 8, paddingHorizontal: 16 },
  tab: { paddingVertical: 8, paddingHorizontal: 12, borderRadius: 10, backgroundColor: '#fff', borderWidth: 1, borderColor: '#d3dcd2' },
  tabOn: { backgroundColor: C.brand, borderColor: C.brand },
  tabT: { fontWeight: '600', color: C.muted },
  foot: { fontSize: 11, color: C.muted, padding: 12, textAlign: 'center' },
});
