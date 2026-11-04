import { Pressable, ScrollView, Text, View } from 'react-native';
import { type QueueItem } from '../core/queue';
import { ReportCard } from '../ui/ReportCard';
import { C, s } from '../ui/theme';

const STATUS: Record<string, { text: string; bg: string; fg: string }> = {
  pending: { text: 'Waiting for connection', bg: C.warnBg, fg: C.warn }, syncing: { text: 'Confirming...', bg: C.warnBg, fg: C.warn },
  done: { text: 'Confirmed by the server model', bg: C.okBg, fg: C.ok }, failed: { text: 'Could not be confirmed', bg: C.badBg, fg: C.bad },
};

export function QueueScreen({ items, syncing, onSync, onRetry, onDelete }:
  { items: QueueItem[]; syncing: boolean; onSync: () => void; onRetry: (i: QueueItem) => void; onDelete: (i: QueueItem) => void }) {
  return (
    <ScrollView contentContainerStyle={s.screen}>
      <View style={s.row}>
        <Pressable style={[s.btn, syncing && { opacity: 0.5 }]} disabled={syncing} onPress={onSync}><Text style={s.btnT}>{syncing ? 'Syncing...' : 'Sync now'}</Text></Pressable>
      </View>
      {items.length === 0 && <Text style={s.muted}>No photos are waiting. Photos taken without internet appear here and are confirmed automatically when you are back online.</Text>}
      {[...items].reverse().map((i) => {
        const st = STATUS[i.status];
        const cmp = i.detection?.offline_comparison;
        return (
          <View key={i.id} style={s.card}>
            <Text style={[s.chip, { backgroundColor: st.bg, color: st.fg }]}>{st.text}</Text>
            <Text style={s.muted}>{new Date(i.createdAt).toLocaleString()}{i.animalId ? ` | ${i.animalId}` : ''}{i.district ? ` | ${i.district}` : ''}</Text>
            <Text style={s.li}>On the phone ({i.offlineModelVersion}): {i.offlineDetections.length ? i.offlineDetections.map((d) => `${d.class.replace(/_/g, ' ')} ${Math.round(d.confidence * 100)}%`).join(', ') : 'nothing detected'}</Text>
            {i.status === 'done' && i.detection && (
              <View>
                <Text style={s.li}>Server model ({i.detection.model_version}): {i.detection.detections.length ? i.detection.detections.map((d) => `${d.class.replace(/_/g, ' ')} ${Math.round(d.confidence * 100)}%`).join(', ') : 'nothing detected'}</Text>
                {cmp && !cmp.agrees && (
                  <View style={[s.banner, { backgroundColor: C.warnBg }]}>
                    <Text style={{ color: C.warn, fontWeight: '700' }}>The server model found something different. Use its result.</Text>
                    {cmp.only_online.length > 0 && <Text style={{ color: C.warn }}>Only the server found: {cmp.only_online.join(', ').replace(/_/g, ' ')}</Text>}
                    {cmp.only_offline.length > 0 && <Text style={{ color: C.warn }}>Only the phone found: {cmp.only_offline.join(', ').replace(/_/g, ' ')}</Text>}
                  </View>
                )}
                {i.detection.flags.includes('possible_notifiable_disease') && <View style={[s.banner, { backgroundColor: C.badBg }]}><Text style={{ color: C.bad, fontWeight: '700' }}>Possible notifiable disease: contact the sector veterinary officer / RAB.</Text></View>}
                {i.report ? <ReportCard report={i.report.report} source={i.report.source === 'llm' ? 'llm' : 'pack'} notes={i.report.degraded ? ['The AI report was unavailable: standard guidance.'] : []} />
                  : <Text style={s.muted}>The full report could not be fetched yet. It will be available after the next sync.</Text>}
              </View>
            )}
            {i.lastError && i.status !== 'done' && <Text style={[s.muted, { color: C.bad }]}>{i.lastError}</Text>}
            <View style={s.row}>
              {i.status === 'failed' && <Pressable style={[s.btn, s.btn2]} onPress={() => onRetry(i)}><Text style={s.btnT2}>Try again</Text></Pressable>}
              <Pressable style={[s.btn, s.btn2]} onPress={() => onDelete(i)}><Text style={s.btnT2}>Remove</Text></Pressable>
            </View>
          </View>
        );
      })}
    </ScrollView>
  );
}
