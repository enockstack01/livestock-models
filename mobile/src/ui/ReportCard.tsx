import { Text, View } from 'react-native';
import type { Report } from '../core/types';
import { C, s } from './theme';

const SEV: Record<string, { bg: string; fg: string }> = {
  low: { bg: C.okBg, fg: C.ok }, moderate: { bg: C.warnBg, fg: C.warn }, high: { bg: C.badBg, fg: C.bad }, urgent: { bg: C.badBg, fg: C.bad },
};
const List = ({ title, items }: { title: string; items: string[] }) => (items.length ? (
  <View><Text style={s.h3}>{title}</Text>{items.map((t, i) => <Text key={i} style={s.li}>• {t}</Text>)}</View>
) : null);

/** Renders a report from the LLM API, the server's advice pack, or the on-device advice pack: same shape, labelled by `source`. */
export function ReportCard({ report, source, notes }: { report: Report; source: string; notes: string[] }) {
  const sev = SEV[report.severity] ?? SEV.moderate;
  const label = source === 'llm' ? 'AI report' : 'Standard guidance';
  return (
    <View style={s.card}>
      <View style={{ flexDirection: 'row', gap: 8, flexWrap: 'wrap' }}>
        <Text style={[s.chip, { backgroundColor: source === 'llm' ? C.okBg : C.warnBg, color: source === 'llm' ? C.ok : C.warn }]}>{label}</Text>
        <Text style={[s.chip, { backgroundColor: sev.bg, color: sev.fg }]}>Severity: {report.severity}</Text>
        {report.treatment.requires_vet && <Text style={[s.chip, { backgroundColor: C.warnBg, color: C.warn }]}>Veterinarian needed</Text>}
      </View>
      {report.notifiable && <View style={[s.banner, { backgroundColor: C.badBg }]}><Text style={{ color: C.bad, fontWeight: '700' }}>Possible notifiable disease. Isolate the animal and contact the sector veterinary officer / RAB immediately.</Text></View>}
      {report.zoonotic && <View style={[s.banner, { backgroundColor: C.warnBg }]}><Text style={{ color: C.warn, fontWeight: '700' }}>Can spread to people: wear gloves and wash hands.</Text></View>}
      <Text style={[s.li, { marginTop: 10, fontSize: 15 }]}>{report.summary}</Text>
      <List title="Possible conditions" items={report.possible_conditions} />
      <List title="What to do now" items={report.immediate_actions} />
      <List title="Supportive care" items={report.treatment.supportive_care} />
      <List title="Medicines (never without a vet)" items={report.treatment.medicines} />
      <List title="Milk and meat withdrawal" items={report.treatment.withdrawal_periods} />
      <List title="Prevention" items={report.prevention} />
      {report.follow_up_days != null && <Text style={[s.li, { fontWeight: '700', marginTop: 8 }]}>Check again in {report.follow_up_days} day(s).</Text>}
      {notes.map((n, i) => <Text key={i} style={s.muted}>{n}</Text>)}
      <Text style={s.muted}>{report.disclaimer ?? 'Decision support only, not a veterinary diagnosis.'}</Text>
    </View>
  );
}
